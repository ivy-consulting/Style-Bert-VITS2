import warnings
from pathlib import Path
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import torch
import gradio as gr

import numpy as np
import pyannote.audio
from gradio.processing_utils import convert_to_16_bit_wav
from numpy.typing import NDArray
from pydantic import BaseModel

from style_bert_vits2.constants import (
    DEFAULT_ASSIST_TEXT_WEIGHT,
    DEFAULT_LENGTH,
    DEFAULT_LINE_SPLIT,
    DEFAULT_NOISE,
    DEFAULT_NOISEW,
    DEFAULT_SDP_RATIO,
    DEFAULT_SPLIT_INTERVAL,
    DEFAULT_STYLE,
    DEFAULT_STYLE_WEIGHT,
    Languages,
)
from style_bert_vits2.logging import logger
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.models.infer import get_net_g, infer
from style_bert_vits2.models.models import SynthesizerTrn
from style_bert_vits2.models.models_jp_extra import (
    SynthesizerTrn as SynthesizerTrnJPExtra,
)
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.voice import adjust_voice


class TTSModel:
    """
    Style-Bert-Vits2 の音声合成モデルを操作するクラス。
    モデル/ハイパーパラメータ/スタイルベクトルのパスとデバイスを指定して初期化し、model.infer() メソッドを呼び出すと音声合成を行える。
    """

    def __init__(
        self, model_path: Path, config_path: Path, style_vec_path: Path, device: str
    ) -> None:
        """
        Style-Bert-Vits2 の音声合成モデルを初期化する。
        この時点ではモデルはロードされていない (明示的にロードしたい場合は model.load() を呼び出す)。

        Args:
            model_path (Path): モデル (.safetensors) のパス
            config_path (Path): ハイパーパラメータ (config.json) のパス
            style_vec_path (Path): スタイルベクトル (style_vectors.npy) のパス
            device (str): 音声合成時に利用するデバイス (cpu, cuda, mps など)
        """

        self.model_path: Path = model_path
        self.config_path: Path = config_path
        self.style_vec_path: Path = style_vec_path
        self.device: str = device
        self.hyper_parameters: HyperParameters = HyperParameters.load_from_json(
            self.config_path
        )
        self.spk2id: dict[str, int] = self.hyper_parameters.data.spk2id
        self.id2spk: dict[int, str] = {v: k for k, v in self.spk2id.items()}

        num_styles: int = self.hyper_parameters.data.num_styles
        if hasattr(self.hyper_parameters.data, "style2id"):
            self.style2id: dict[str, int] = self.hyper_parameters.data.style2id
        else:
            self.style2id: dict[str, int] = {str(i): i for i in range(num_styles)}
        if len(self.style2id) != num_styles:
            raise ValueError(
                f"Number of styles ({num_styles}) does not match the number of style2id ({len(self.style2id)})"
            )

        self.__style_vector_inference: Optional[pyannote.audio.Inference] = None
        self.__style_vectors: NDArray[Any] = np.load(self.style_vec_path)
        if self.__style_vectors.shape[0] != num_styles:
            raise ValueError(
                f"The number of styles ({num_styles}) does not match the number of style vectors ({self.__style_vectors.shape[0]})"
            )

        self.__net_g: Union[SynthesizerTrn, SynthesizerTrnJPExtra, None] = None

    def load(self) -> None:
        """
        音声合成モデルをデバイスにロードする。
        """
        self.__net_g = get_net_g(
            model_path=str(self.model_path),
            version=self.hyper_parameters.version,
            device=self.device,
            hps=self.hyper_parameters,
        )

    def __get_style_vector(self, style_id: int, weight: float = 1.0) -> NDArray[Any]:
        """
        スタイルベクトルを取得する。

        Args:
            style_id (int): スタイル ID (0 から始まるインデックス)
            weight (float, optional): スタイルベクトルの重み. Defaults to 1.0.

        Returns:
            NDArray[Any]: スタイルベクトル
        """
        mean = self.__style_vectors[0]
        style_vec = self.__style_vectors[style_id]
        style_vec = mean + (style_vec - mean) * weight
        return style_vec

    def __get_style_vector_from_audio(
        self, audio_path: str, weight: float = 1.0
    ) -> NDArray[Any]:
        """
        音声からスタイルベクトルを推論する。

        Args:
            audio_path (str): 音声ファイルのパス
            weight (float, optional): スタイルベクトルの重み. Defaults to 1.0.
        Returns:
            NDArray[Any]: スタイルベクトル
        """

        # スタイルベクトルを取得するための推論モデルを初期化
        if self.__style_vector_inference is None:
            self.__style_vector_inference = pyannote.audio.Inference(
                model=pyannote.audio.Model.from_pretrained(
                    "pyannote/wespeaker-voxceleb-resnet34-LM"
                ),
                window="whole",
            )
            self.__style_vector_inference.to(torch.device(self.device))

        # 音声からスタイルベクトルを推論
        xvec = self.__style_vector_inference(audio_path)
        mean = self.__style_vectors[0]
        xvec = mean + (xvec - mean) * weight
        return xvec

    def infer(
        self,
        text: str,
        language: Languages = Languages.JP,
        speaker_id: int = 0,
        reference_audio_path: Optional[str] = None,
        sdp_ratio: float = DEFAULT_SDP_RATIO,
        noise: float = DEFAULT_NOISE,
        noise_w: float = DEFAULT_NOISEW,
        length: float = DEFAULT_LENGTH,
        line_split: bool = DEFAULT_LINE_SPLIT,
        split_interval: float = DEFAULT_SPLIT_INTERVAL,
        assist_text: Optional[str] = None,
        assist_text_weight: float = DEFAULT_ASSIST_TEXT_WEIGHT,
        use_assist_text: bool = False,
        style: str = DEFAULT_STYLE,
        style_weight: float = DEFAULT_STYLE_WEIGHT,
        given_tone: Optional[list[int]] = None,
        pitch_scale: float = 1.0,
        intonation_scale: float = 1.0,
    ) -> tuple[int, NDArray[Any]]:
        """
        テキストから音声を合成する。

        Args:
            text (str): 読み上げるテキスト
            language (Languages, optional): 言語. Defaults to Languages.JP.
            speaker_id (int, optional): 話者 ID. Defaults to 0.
            reference_audio_path (Optional[str], optional): 音声スタイルの参照元の音声ファイルのパス. Defaults to None.
            sdp_ratio (float, optional): DP と SDP の混合比。0 で DP のみ、1で SDP のみを使用 (値を大きくするとテンポに緩急がつく). Defaults to DEFAULT_SDP_RATIO.
            noise (float, optional): DP に与えられるノイズ. Defaults to DEFAULT_NOISE.
            noise_w (float, optional): SDP に与えられるノイズ. Defaults to DEFAULT_NOISEW.
            length (float, optional): 生成音声の長さ（話速）のパラメータ。大きいほど生成音声が長くゆっくり、小さいほど短く早くなる。 Defaults to DEFAULT_LENGTH.
            line_split (bool, optional): テキストを改行ごとに分割して生成するかどうか. Defaults to DEFAULT_LINE_SPLIT.
            split_interval (float, optional): 改行ごとに分割する場合の無音 (秒). Defaults to DEFAULT_SPLIT_INTERVAL.
            assist_text (Optional[str], optional): 感情表現の参照元の補助テキスト. Defaults to None.
            assist_text_weight (float, optional): 感情表現の補助テキストを適用する強さ. Defaults to DEFAULT_ASSIST_TEXT_WEIGHT.
            use_assist_text (bool, optional): 音声合成時に感情表現の補助テキストを使用するかどうか. Defaults to False.
            style (str, optional): 音声スタイル (Neutral, Happy など). Defaults to DEFAULT_STYLE.
            style_weight (float, optional): 音声スタイルを適用する強さ. Defaults to DEFAULT_STYLE_WEIGHT.
            given_tone (Optional[list[int]], optional): アクセントのトーンのリスト. Defaults to None.
            pitch_scale (float, optional): ピッチの高さ (1.0 から変更すると若干音質が低下する). Defaults to 1.0.
            intonation_scale (float, optional): 抑揚の平均からの変化幅 (1.0 から変更すると若干音質が低下する). Defaults to 1.0.

        Returns:
            tuple[int, NDArray[Any]]: サンプリングレートと音声データ (16bit PCM)
        """

        logger.info(f"Start generating audio data from text:\n{text}")
        if language != "JP" and self.hyper_parameters.version.endswith("JP-Extra"):
            raise ValueError(
                "The model is trained with JP-Extra, but the language is not JP"
            )
        if reference_audio_path == "":
            reference_audio_path = None
        if assist_text == "" or not use_assist_text:
            assist_text = None

        if self.__net_g is None:
            self.load()
        assert self.__net_g is not None
        if reference_audio_path is None:
            style_id = self.style2id[style]
            style_vector = self.__get_style_vector(style_id, style_weight)
        else:
            style_vector = self.__get_style_vector_from_audio(
                reference_audio_path, style_weight
            )
        if not line_split:
            with torch.no_grad():
                audio = infer(
                    text=text,
                    sdp_ratio=sdp_ratio,
                    noise_scale=noise,
                    noise_scale_w=noise_w,
                    length_scale=length,
                    sid=speaker_id,
                    language=language,
                    hps=self.hyper_parameters,
                    net_g=self.__net_g,
                    device=self.device,
                    assist_text=assist_text,
                    assist_text_weight=assist_text_weight,
                    style_vec=style_vector,
                    given_tone=given_tone,
                )
        else:
            texts = text.split("\n")
            texts = [t for t in texts if t != ""]
            audios = []
            with torch.no_grad():
                for i, t in enumerate(texts):
                    audios.append(
                        infer(
                            text=t,
                            sdp_ratio=sdp_ratio,
                            noise_scale=noise,
                            noise_scale_w=noise_w,
                            length_scale=length,
                            sid=speaker_id,
                            language=language,
                            hps=self.hyper_parameters,
                            net_g=self.__net_g,
                            device=self.device,
                            assist_text=assist_text,
                            assist_text_weight=assist_text_weight,
                            style_vec=style_vector,
                        )
                    )
                    if i != len(texts) - 1:
                        audios.append(np.zeros(int(44100 * split_interval)))
                audio = np.concatenate(audios)
        logger.info("Audio data generated successfully")
        if not (pitch_scale == 1.0 and intonation_scale == 1.0):
            _, audio = adjust_voice(
                fs=self.hyper_parameters.data.sampling_rate,
                wave=audio,
                pitch_scale=pitch_scale,
                intonation_scale=intonation_scale,
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audio = convert_to_16_bit_wav(audio)
        return (self.hyper_parameters.data.sampling_rate, audio)


class TTSModelInfo(BaseModel):
    name: str
    files: list[str]
    styles: list[str]
    speakers: list[str]


class TTSModelHolder:
    """
    Class to manage Style-Bert-Vits2 voice synthesis models.
    """

    def __init__(self, config_file: Path, device: str) -> None:
        """
        Initialize the TTSModelHolder class.

        Args:
            config_file (Path): Path to the JSON configuration file
            device (str): Device to be used for synthesis (cpu, cuda, etc.)
        """
        self.device: str = device
        self.config_file: Path = config_file
        self.root_dir: Path = config_file.parent
        self.model_files_dict: Dict[str, Dict[str, str]] = {}
        self.current_model: Optional[TTSModel] = None
        self.model_names: List[str] = []
        self.models_info: List[TTSModelInfo] = []
        self.refresh()

    def refresh(self) -> None:
        """
        Update the list of voice synthesis models.
        """
        with open(self.config_file, 'r') as f:
            config = json.load(f)

        self.model_files_dict = config.get('models', {})
        self.model_names = list(self.model_files_dict.keys())
        self.current_model = None
        self.models_info = []

        for model_name, paths in self.model_files_dict.items():
            model_path = paths["model_path"]
            config_path = paths["config_path"]
            style_vec_path = paths.get("style_vec_path", None)

            hyper_parameters = HyperParameters.load_from_json(config_path)
            style2id: Dict[str, int] = hyper_parameters.data.style2id
            styles = list(style2id.keys())
            spk2id: Dict[str, int] = hyper_parameters.data.spk2id
            speakers = list(spk2id.keys())

            self.models_info.append(
                TTSModelInfo(
                    name=model_name,
                    files=[model_path],
                    styles=styles,
                    speakers=speakers,
                )
            )

    def get_model(self, model_name: str) -> TTSModel:
        """
        Get the instance of the specified voice synthesis model.

        Args:
            model_name (str): Name of the voice synthesis model

        Returns:
            TTSModel: Instance of the voice synthesis model
        """
        if model_name not in self.model_files_dict:
            raise ValueError(f"Model `{model_name}` is not found")

        paths = self.model_files_dict[model_name]
        model_path = Path(paths["model_path"])
        config_path = Path(paths["config_path"])
        style_vec_path = Path(paths.get("style_vec_path", ""))

        if self.current_model is None or self.current_model.model_path != model_path:
            self.current_model = TTSModel(
                model_path=model_path,
                config_path=config_path,
                style_vec_path=style_vec_path,
                device=self.device,
            )

        return self.current_model

    def get_model_for_gradio(self, model_name: str) -> tuple[gr.Dropdown, gr.Button, gr.Dropdown]:
        if model_name not in self.model_files_dict:
            raise ValueError(f"Model `{model_name}` is not found")

        paths = self.model_files_dict[model_name]
        model_path = Path(paths["model_path"])
        if self.current_model is not None and self.current_model.model_path == model_path:
            # Already loaded
            speakers = list(self.current_model.spk2id.keys())
            styles = list(self.current_model.style2id.keys())
            return (
                gr.Dropdown(choices=styles, value=styles[0]),  # type: ignore
                gr.Button(interactive=True, value="音声合成"),
                gr.Dropdown(choices=speakers, value=speakers[0]),  # type: ignore
            )

        config_path = Path(paths["config_path"])
        style_vec_path = Path(paths.get("style_vec_path", ""))

        self.current_model = TTSModel(
            model_path=model_path,
            config_path=config_path,
            style_vec_path=style_vec_path,
            device=self.device,
        )
        speakers = list(self.current_model.spk2id.keys())
        styles = list(self.current_model.style2id.keys())
        return (
            gr.Dropdown(choices=styles, value=styles[0]),  # type: ignore
            gr.Button(interactive=True, value="音声合成"),
            gr.Dropdown(choices=speakers, value=speakers[0]),  # type: ignore
        )

    def update_model_files_for_gradio(self, model_name: str) -> gr.Dropdown:
        if model_name not in self.model_files_dict:
            raise ValueError(f"Model `{model_name}` is not found")
        model_files = [self.model_files_dict[model_name]["model_path"]]
        return gr.Dropdown(choices=model_files, value=model_files[0])  # type: ignore

    def update_model_names_for_gradio(self) -> tuple[gr.Dropdown, gr.Dropdown, gr.Button]:
        self.refresh()
        initial_model_name = self.model_names[0]
        initial_model_files = [self.model_files_dict[initial_model_name]["model_path"]]
        return (
            gr.Dropdown(choices=self.model_names, value=initial_model_name),  # type: ignore
            gr.Dropdown(choices=initial_model_files, value=initial_model_files[0]),  # type: ignore
            gr.Button(interactive=False),  # For tts_button
        )