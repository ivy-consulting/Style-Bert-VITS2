# """
# API server for TTS
# TODO: server_editor.pyと統合する?
# """

import nltk
nltk.download('averaged_perceptron_tagger_eng')
# import argparse
# import os
# import sys
# from io import BytesIO
# from pathlib import Path
# from typing import Any, Optional
# from urllib.parse import unquote
from pydantic import BaseModel
# import time

# import GPUtil
# import psutil
# import torch
# import uvicorn
# from fastapi import FastAPI, HTTPException, Query, Request, status
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import FileResponse, Response
from scipy.io import wavfile

from requests_toolbelt.multipart.encoder import MultipartEncoder
import json
import soundfile as sf

# from config import config
# from style_bert_vits2.constants import (
#     DEFAULT_ASSIST_TEXT_WEIGHT,
#     DEFAULT_LENGTH,
#     DEFAULT_LINE_SPLIT,
#     DEFAULT_NOISE,
#     DEFAULT_NOISEW,
#     DEFAULT_SDP_RATIO,
#     DEFAULT_SPLIT_INTERVAL,
#     DEFAULT_STYLE,
#     DEFAULT_STYLE_WEIGHT,
#     Languages,
# )
# from style_bert_vits2.logging import logger
# from style_bert_vits2.nlp import bert_models
# from style_bert_vits2.nlp.japanese import pyopenjtalk_worker as pyopenjtalk
# from style_bert_vits2.nlp.japanese.user_dict import update_dict
# from style_bert_vits2.tts_model import TTSModel, TTSModelHolder


# ln = config.server_config.language


# # pyopenjtalk_worker を起動
# ## pyopenjtalk_worker は TCP ソケットサーバーのため、ここで起動する
# pyopenjtalk.initialize_worker()

# # dict_data/ 以下の辞書データを pyopenjtalk に適用
# update_dict()

# # 事前に BERT モデル/トークナイザーをロードしておく
# ## ここでロードしなくても必要になった際に自動ロードされるが、時間がかかるため事前にロードしておいた方が体験が良い
# bert_models.load_model(Languages.JP)
# bert_models.load_tokenizer(Languages.JP)
# bert_models.load_model(Languages.EN)
# bert_models.load_tokenizer(Languages.EN)
# bert_models.load_model(Languages.ZH)
# bert_models.load_tokenizer(Languages.ZH)


# def raise_validation_error(msg: str, param: str):
#     logger.warning(f"Validation error: {msg}")
#     raise HTTPException(
#         status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
#         detail=[dict(type="invalid_params", msg=msg, loc=["query", param])],
#     )


# class AudioResponse(Response):
#     media_type = "multipart/form-data"


# loaded_models: list[TTSModel] = []


# def load_models(model_holder: TTSModelHolder):
#     global loaded_models
#     loaded_models = []
#     for model_name, model_paths in model_holder.model_files_dict.items():
#         model = TTSModel(
#             model_path=model_paths[0],
#             config_path=model_holder.root_dir / model_name / "config.json",
#             style_vec_path=model_holder.root_dir / model_name / "style_vectors.npy",
#             device=model_holder.device,
#         )
#         # 起動時に全てのモデルを読み込むのは時間がかかりメモリを食うのでやめる
#         # model.load()
#         loaded_models.append(model)

# def createBinary(audio_path):
#   data, samplerate = sf.read(audio_path, dtype='int16')
#   binary_data = data.tobytes()
#   return binary_data, samplerate


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
#     parser.add_argument(
#         "--dir", "-d", type=str, help="Model directory", default=config.assets_root
#     )
#     args = parser.parse_args()

    
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     logger.info("Its using the GPU")  if torch.cuda.is_available() else logger.warning("Its not using GPU, it will make it slow.")

#     model_dir = Path(args.dir)
#     model_holder = TTSModelHolder(model_dir, device)
#     if len(model_holder.model_names) == 0:
#         logger.error(f"Models not found in {model_dir}.")
#         sys.exit(1)

#     logger.info("Loading models...")
#     load_models(model_holder)

#     limit = config.server_config.limit
#     app = FastAPI()
#     allow_origins = config.server_config.origins
#     if allow_origins:
#         logger.warning(
#             f"CORS allow_origins={config.server_config.origins}. If you don't want, modify config.yml"
#         )
#         app.add_middleware(
#             CORSMiddleware,
#             allow_origins=config.server_config.origins,
#             allow_credentials=True,
#             allow_methods=["*"],
#             allow_headers=["*"],
#         )
#     # app.logger = logger
#     # ↑効いていなさそう。loggerをどうやって上書きするかはよく分からなかった。

#     @app.api_route("/voice", methods=["GET", "POST"], response_class=AudioResponse)
#     async def voice(
#         request: Request,
#         text: str = Query(..., min_length=1, max_length=limit, description="セリフ"),
#         encoding: str = Query(None, description="textをURLデコードする(ex, `utf-8`)"),
#         model_id: int = Query(
#             0, description="モデルID。`GET /models/info`のkeyの値を指定ください"
#         ),
#         speaker_name: str = Query(
#             None,
#             description="話者名(speaker_idより優先)。esd.listの2列目の文字列を指定",
#         ),
#         speaker_id: int = Query(
#             0, description="話者ID。model_assets>[model]>config.json内のspk2idを確認"
#         ),
#         sdp_ratio: float = Query(
#             DEFAULT_SDP_RATIO,
#             description="SDP(Stochastic Duration Predictor)/DP混合比。比率が高くなるほどトーンのばらつきが大きくなる",
#         ),
#         noise: float = Query(
#             DEFAULT_NOISE,
#             description="サンプルノイズの割合。大きくするほどランダム性が高まる",
#         ),
#         noisew: float = Query(
#             DEFAULT_NOISEW,
#             description="SDPノイズ。大きくするほど発音の間隔にばらつきが出やすくなる",
#         ),
#         length: float = Query(
#             DEFAULT_LENGTH,
#             description="話速。基準は1で大きくするほど音声は長くなり読み上げが遅まる",
#         ),
#         language: Languages = Query(ln, description="textの言語"),
#         auto_split: bool = Query(DEFAULT_LINE_SPLIT, description="改行で分けて生成"),
#         split_interval: float = Query(
#             DEFAULT_SPLIT_INTERVAL, description="分けた場合に挟む無音の長さ（秒）"
#         ),
#         assist_text: Optional[str] = Query(
#             None,
#             description="このテキストの読み上げと似た声音・感情になりやすくなる。ただし抑揚やテンポ等が犠牲になる傾向がある",
#         ),
#         assist_text_weight: float = Query(
#             DEFAULT_ASSIST_TEXT_WEIGHT, description="assist_textの強さ"
#         ),
#         style: Optional[str] = Query(DEFAULT_STYLE, description="スタイル"),
#         style_weight: float = Query(DEFAULT_STYLE_WEIGHT, description="スタイルの強さ"),
#         reference_audio_path: Optional[str] = Query(
#             None, description="スタイルを音声ファイルで行う"
#         ),
#     ):
#         audio_start_time = time.time()
#         """Infer text to speech(テキストから感情付き音声を生成する)"""
#         logger.info(
#             f"{request.client.host}:{request.client.port}/voice  { unquote(str(request.query_params) )}"
#         )
#         if request.method == "GET":
#             logger.warning(
#                 "The GET method is not recommended for this endpoint due to various restrictions. Please use the POST method."
#             )
#         if model_id >= len(
#             model_holder.model_names
#         ):  # /models/refresh があるためQuery(le)で表現不可
#             raise_validation_error(f"model_id={model_id} not found", "model_id")

#         model = loaded_models[model_id]

#         if speaker_name is None:
#             if speaker_id not in model.id2spk.keys():
#                 raise_validation_error(
#                     f"speaker_id={speaker_id} not found", "speaker_id"
#                 )
#         else:
#             if speaker_name not in model.spk2id.keys():
#                 raise_validation_error(
#                     f"speaker_name={speaker_name} not found", "speaker_name"
#                 )
#             speaker_id = model.spk2id[speaker_name]
#         if style not in model.style2id.keys():
#             raise_validation_error(f"style={style} not found", "style")
#         assert style is not None
#         if encoding is not None:
#             text = unquote(text, encoding=encoding)
#         sr, audio = model.infer(
#             text=text,
#             language=language,
#             speaker_id=speaker_id,
#             reference_audio_path=reference_audio_path,
#             sdp_ratio=sdp_ratio,
#             noise=noise,
#             noise_w=noisew,
#             length=length,
#             line_split=auto_split,
#             split_interval=split_interval,
#             assist_text=assist_text,
#             assist_text_weight=assist_text_weight,
#             use_assist_text=bool(assist_text),
#             style=style,
#             style_weight=style_weight,
#         )

        

#         wav_bytes_io = BytesIO()
#         wavfile.write(wav_bytes_io, sr, audio.astype('int16'))  # Ensure audio data type is int16
#         wav_bytes_io.seek(0)

#         try:
#             audio_data = wav_bytes_io
#             # Now you can do something with the audio data, such as saving it to a 
#             logger.info("audio data vlue:", audio_data.getvalue())
#             with open("output.wav", "wb") as f:
#                 f.write(audio_data.getvalue())
                
#             audio_end = time.time()

#             logger.success("Audio data generated and sent successfully")

#             audio_path = "./output.wav"
#             binary, samplerate = createBinary('./output.wav')

#             start_time = time.time()
#             samplerate_str = str(samplerate)

#             # create vowel and vowel_length
#             end_time = time.time()

#             logger.info(f"The time it take to generate a speech synthesis: {text} is {audio_end - audio_start_time} seconds")


#             multipart_data = MultipartEncoder(
#                 fields={
#                 # 音声データを含める
#                 'audio': ('output.wav', binary, 'audio/wav'),
#                 # サンプルレートもフィールドに追加
#                 'samplerate': ('samplerate', samplerate_str, 'text/plain')
#                 }
#             )


#             logger.info("Generating audio -----------------------------------------")

#             response = AudioResponse(multipart_data.to_string())
#             response.headers['Content-Type'] = multipart_data.content_type
#             return response

#         except Exception as e:
#             logger.error(f"Error generating audio: {e}")
        


        

#     @app.get("/models/info")
#     def get_loaded_models_info():
#         """ロードされたモデル情報の取得"""

#         result: dict[str, dict[str, Any]] = dict()
#         for model_id, model in enumerate(loaded_models):
#             result[str(model_id)] = {
#                 "config_path": model.config_path,
#                 "model_path": model.model_path,
#                 "device": model.device,
#                 "spk2id": model.spk2id,
#                 "id2spk": model.id2spk,
#                 "style2id": model.style2id,
#             }
#         return result

#     logger.info(f"server listen: http://127.0.0.1:5001")
#     logger.info(f"API docs: http://127.0.0.1:5000/docs")
#     uvicorn.run(
#         app, port=5000, host="0.0.0.0", log_level="warning"
#     )


"""
API server for TTS
TODO: server_editor.pyと統合する?
"""

import argparse
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote
import time

import GPUtil
import psutil
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from scipy.io import wavfile

from config import config
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
    DEFAULT_PITCH_SCALE,
    DEFAULT_INTONATION_SCALE,
    Languages,
)
from style_bert_vits2.logging import logger
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker as pyopenjtalk
from style_bert_vits2.nlp.japanese.user_dict import update_dict
from style_bert_vits2.tts_model import TTSModel, TTSModelHolder


ln = config.server_config.language


# pyopenjtalk_worker を起動
## pyopenjtalk_worker は TCP ソケットサーバーのため、ここで起動する
pyopenjtalk.initialize_worker()

# dict_data/ 以下の辞書データを pyopenjtalk に適用
update_dict()

# 事前に BERT モデル/トークナイザーをロードしておく
## ここでロードしなくても必要になった際に自動ロードされるが、時間がかかるため事前にロードしておいた方が体験が良い
bert_models.load_model(Languages.JP)
bert_models.load_tokenizer(Languages.JP)
bert_models.load_model(Languages.EN)
bert_models.load_tokenizer(Languages.EN)
bert_models.load_model(Languages.ZH)
bert_models.load_tokenizer(Languages.ZH)



def raise_validation_error(msg: str, param: str):
    logger.warning(f"Validation error: {msg}")
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=[dict(type="invalid_params", msg=msg, loc=["query", param])],
    )


class AudioResponse(Response):
    media_type = "audio/wav"

def createBinary(audio_path):
  data, samplerate = sf.read(audio_path, dtype='int16')
  binary_data = data.tobytes()
  return binary_data, samplerate

# Function to extract words from the cmudict file

def extract_words_from_cmudict(file_path):
    words = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Skip comment lines starting with ##
                if line.startswith('##'):
                    continue
                # Split the line by whitespace and take the first part (the word)
                parts = line.split()
                if parts:  # Ensure the line isn't empty
                    word = parts[0]
                    # Add the word to the set (case-insensitive)
                    words.add(word.lower())
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"Error reading file: {e}")
    return words

# Define the clean function
import string

def clean(sentence, dictionary_words):
    # Split the sentence into words
    words = sentence.split()
    # Remove punctuation from each word and keep only words that are in the dictionary (case-insensitive)
    filtered_words = [word for word in words if word.strip(string.punctuation).lower() in dictionary_words]
    # Join the filtered words back into a sentence
    return ' '.join(filtered_words)

file_path = "style_bert_vits2/nlp/english/cmudict.rep"

# Extract words from the file
dictionary_words = extract_words_from_cmudict(file_path)

loaded_models: list[TTSModel] = []

class ByteAudioResponse(Response):
    media_type = "multipart/form-data"


def load_models(model_holder: TTSModelHolder):
    for model_name, paths in model_holder.model_files_dict.items():
        model = TTSModel(
            model_path=Path(paths["model_path"]),
            config_path=Path(paths["config_path"]),
            style_vec_path=Path(paths.get("style_vec_path", "")),
            device=model_holder.device,
        )
        # Avoid loading all models at startup to save time and memory
        # model.load()
        loaded_models.append(model)



from fastapi import FastAPI, Request, HTTPException
from functools import wraps
import hashlib

app = FastAPI()

# Define your API keys and secret
API_KEYS = {
    'nodejs-service': 'nodejs-api-key',
    'python-service1': 'python1-api-key',
}


import hashlib

API_KEYS = {
    'ruby-service': 'sps-dasjidnkwoi0eqjdndaisjirwanjidansidnqwihrdasdas',
    'python-service1': 'python1-api-key',
}

SERVICE_SECRET = 'dhjasd44e3eqwe32eqw532eqweandi3j4'

def verify_signature(service_name: str, signature: str) -> bool:
    """
    Verify the incoming request signature.
    """
    api_key = API_KEYS.get(service_name)
    if not api_key:
        return False

    # Recreate the expected signature
    data = f"{service_name}{api_key}{SERVICE_SECRET}"
    expected_signature = hashlib.sha256(data.encode()).hexdigest()

    return signature == expected_signature


def authenticate_service(func):
    """
    Decorator to authenticate service-to-service requests.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs) -> Response:
        request: Request = kwargs.get("request")
        if not request:
            raise HTTPException(status_code=403, detail="Missing request")
            
        x_service_name: Optional[str] = request.headers.get("X-Service-Name") 
        x_signature: Optional[str] = request.headers.get("X-Signature")

        if not x_service_name or not x_signature:
            raise HTTPException(status_code=403, detail="Missing authentication headers")

        if not verify_signature(x_service_name, x_signature):
            raise HTTPException(status_code=403, detail="Invalid authentication signature")

        return await func(*args, **kwargs)

    return wrapper


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument(
        "--dir", "-d", type=str, help="Model directory", default=config.assets_root
    )
    args = parser.parse_args()

    logger.info(f"Checking args.cpu: {args.cpu}")

    if args.cpu:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not torch.cuda.is_available():
        logger.warning("The synthesis is not using GPU! use GPU to get better performance!")
    else:
        logger.success("The synthesis is using GPU! you will get a better performance!")

    # model_dir = Path(args.dir)
    model_config = Path("model_config.json")
    model_holder = TTSModelHolder(model_config, device)
    if len(model_holder.model_names) == 0:
        logger.error(f"Models not found in {model_config}.")
        sys.exit(1)

    logger.info("Loading models...")
    load_models(model_holder)

    limit = config.server_config.limit
    app = FastAPI()
    allow_origins = config.server_config.origins
    if allow_origins:
        logger.warning(
            f"CORS allow_origins={config.server_config.origins}. If you don't want, modify config.yml"
        )
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.server_config.origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    # app.logger = logger
    # ↑効いていなさそう。loggerをどうやって上書きするかはよく分からなかった。

    
    @app.api_route("/voice", methods=["POST"], response_class=AudioResponse)
    @authenticate_service
    async def voice(
        request: Request,
        text: str = Query(..., min_length=1, description="セリフ"),
        encoding: str = Query(None, description="textをURLデコードする(ex, `utf-8`)"),
        model_id: int = Query(
            0, description="モデルID。`GET /models/info`のkeyの値を指定ください"
        ),
        speaker_name: str = Query(
            None,
            description="話者名(speaker_idより優先)。esd.listの2列目の文字列を指定",
        ),
        speaker_id: int = Query(
            0, description="話者ID。model_assets>[model]>config.json内のspk2idを確認"
        ),
        sdp_ratio: float = Query(
            DEFAULT_SDP_RATIO,
            description="SDP(Stochastic Duration Predictor)/DP混合比。比率が高くなるほどトーンのばらつきが大きくなる",
        ),
        noise: float = Query(
            DEFAULT_NOISE,
            description="サンプルノイズの割合。大きくするほどランダム性が高まる",
        ),
        noisew: float = Query(
            DEFAULT_NOISEW,
            description="SDPノイズ。大きくするほど発音の間隔にばらつきが出やすくなる",
        ),
        length: float = Query(
            DEFAULT_LENGTH,
            description="話速。基準は1で大きくするほど音声は長くなり読み上げが遅まる",
        ),
        language: Languages = Query(ln, description="textの言語"),
        auto_split: bool = Query(DEFAULT_LINE_SPLIT, description="改行で分けて生成"),
        split_interval: float = Query(
            DEFAULT_SPLIT_INTERVAL, description="分けた場合に挟む無音の長さ（秒）"
        ),
        assist_text: Optional[str] = Query(
            None,
            description="このテキストの読み上げと似た声音・感情になりやすくなる。ただし抑揚やテンポ等が犠牲になる傾向がある",
        ),
        assist_text_weight: float = Query(
            DEFAULT_ASSIST_TEXT_WEIGHT, description="assist_textの強さ"
        ),
        style: Optional[str] = Query(DEFAULT_STYLE, description="スタイル"),
        style_weight: float = Query(DEFAULT_STYLE_WEIGHT, description="スタイルの強さ"),
        reference_audio_path: Optional[str] = Query(
            None, description="スタイルを音声ファイルで行う"
        ),
        pitch_scale: float = Query(DEFAULT_PITCH_SCALE, description="Enter a pitch scale."),
        intonation_scale: float = Query(DEFAULT_INTONATION_SCALE, description="Enter the intonation scale")
    ):
        start_time = time.time()

        """Infer text to speech(テキストから感情付き音声を生成する)"""
        logger.info(
            f"{request.client.host}:{request.client.port}/voice  { unquote(str(request.query_params) )}"
        )
        if request.method == "GET":
            logger.warning(
                "The GET method is not recommended for this endpoint due to various restrictions. Please use the POST method."
            )
        if model_id >= len(
            model_holder.model_names
        ):  # /models/refresh があるためQuery(le)で表現不可
            raise_validation_error(f"model_id={model_id} not found", "model_id")

        
        if language == "EN":
            text = clean(text, dictionary_words)

        #text = preproccesed text

        model = loaded_models[model_id]

        if speaker_name is None:
            if speaker_id not in model.id2spk.keys():
                raise_validation_error(
                    f"speaker_id={speaker_id} not found", "speaker_id"
                )
        else:
            if speaker_name not in model.spk2id.keys():
                raise_validation_error(
                    f"speaker_name={speaker_name} not found", "speaker_name"
                )
            speaker_id = model.spk2id[speaker_name]
        if style not in model.style2id.keys():
            raise_validation_error(f"style={style} not found", "style")
        assert style is not None
        if encoding is not None:
            text = unquote(text, encoding=encoding)

        try:
            sr, audio = model.infer(
                text=text,
                language=language,
                speaker_id=speaker_id,
                reference_audio_path=reference_audio_path,
                sdp_ratio=sdp_ratio,
                noise=noise,
                noise_w=noisew,
                length=length,
                line_split=auto_split,
                split_interval=split_interval,
                assist_text=assist_text,
                assist_text_weight=assist_text_weight,
                use_assist_text=bool(assist_text),
                style=style,
                style_weight=style_weight,
                pitch_scale=pitch_scale,
                intonation_scale=intonation_scale
            )
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA Out of Memory: {e}")
            torch.cuda.empty_cache()  # Clear cache on error
            raise HTTPException(status_code=500, detail="GPU memory exhausted, please try again.")
        
        # Clear GPU memory after successful inference
        torch.cuda.empty_cache()
        logger.success("Audio data generated and sent successfully")
        with BytesIO() as wavContent:
            wavfile.write(wavContent, sr, audio)
            end_time = time.time()
            logger.info(f"It took {end_time - start_time} seconds to generate the audio!")
            return Response(content=wavContent.getvalue(), media_type="audio/wav")
        

    @app.api_route("/audioByteArray", methods=["GET", "POST"], response_class=ByteAudioResponse)
    @authenticate_service
    async def voice(
        request: Request,
        text: str = Query(..., min_length=1, description="セリフ"),
        encoding: str = Query(None, description="textをURLデコードする(ex, `utf-8`)"),
        model_id: int = Query(
            0, description="モデルID。`GET /models/info`のkeyの値を指定ください"
        ),
        speaker_name: str = Query(
            None,
            description="話者名(speaker_idより優先)。esd.listの2列目の文字列を指定",
        ),
        speaker_id: int = Query(
            0, description="話者ID。model_assets>[model]>config.json内のspk2idを確認"
        ),
        sdp_ratio: float = Query(
            DEFAULT_SDP_RATIO,
            description="SDP(Stochastic Duration Predictor)/DP混合比。比率が高くなるほどトーンのばらつきが大きくなる",
        ),
        noise: float = Query(
            DEFAULT_NOISE,
            description="サンプルノイズの割合。大きくするほどランダム性が高まる",
        ),
        noisew: float = Query(
            DEFAULT_NOISEW,
            description="SDPノイズ。大きくするほど発音の間隔にばらつきが出やすくなる",
        ),
        length: float = Query(
            DEFAULT_LENGTH,
            description="話速。基準は1で大きくするほど音声は長くなり読み上げが遅まる",
        ),
        language: Languages = Query(ln, description="textの言語"),
        auto_split: bool = Query(DEFAULT_LINE_SPLIT, description="改行で分けて生成"),
        split_interval: float = Query(
            DEFAULT_SPLIT_INTERVAL, description="分けた場合に挟む無音の長さ（秒）"
        ),
        assist_text: Optional[str] = Query(
            None,
            description="このテキストの読み上げと似た声音・感情になりやすくなる。ただし抑揚やテンポ等が犠牲になる傾向がある",
        ),
        assist_text_weight: float = Query(
            DEFAULT_ASSIST_TEXT_WEIGHT, description="assist_textの強さ"
        ),
        style: Optional[str] = Query(DEFAULT_STYLE, description="スタイル"),
        style_weight: float = Query(DEFAULT_STYLE_WEIGHT, description="スタイルの強さ"),
        reference_audio_path: Optional[str] = Query(
            None, description="スタイルを音声ファイルで行う"
        ),
    ):
        audio_start_time = time.time()
        """Infer text to speech(テキストから感情付き音声を生成する)"""
        logger.info(
            f"{request.client.host}:{request.client.port}/voice  { unquote(str(request.query_params) )}"
        )
        if request.method == "GET":
            logger.warning(
                "The GET method is not recommended for this endpoint due to various restrictions. Please use the POST method."
            )
        if model_id >= len(
            model_holder.model_names
        ):  # /models/refresh があるためQuery(le)で表現不可
            raise_validation_error(f"model_id={model_id} not found", "model_id")

        model = loaded_models[model_id]

        if speaker_name is None:
            if speaker_id not in model.id2spk.keys():
                raise_validation_error(
                    f"speaker_id={speaker_id} not found", "speaker_id"
                )
        else:
            if speaker_name not in model.spk2id.keys():
                raise_validation_error(
                    f"speaker_name={speaker_name} not found", "speaker_name"
                )
            speaker_id = model.spk2id[speaker_name]
        if style not in model.style2id.keys():
            raise_validation_error(f"style={style} not found", "style")
        assert style is not None
        if encoding is not None:
            text = unquote(text, encoding=encoding)
        sr, audio = model.infer(
            text=text,
            language=language,
            speaker_id=speaker_id,
            reference_audio_path=reference_audio_path,
            sdp_ratio=sdp_ratio,
            noise=noise,
            noise_w=noisew,
            length=length,
            line_split=auto_split,
            split_interval=split_interval,
            assist_text=assist_text,
            assist_text_weight=assist_text_weight,
            use_assist_text=bool(assist_text),
            style=style,
            style_weight=style_weight,
        )

        

        wav_bytes_io = BytesIO()
        wavfile.write(wav_bytes_io, sr, audio.astype('int16'))  # Ensure audio data type is int16
        wav_bytes_io.seek(0)

        try:
            audio_data = wav_bytes_io
            # Now you can do something with the audio data, such as saving it to a 
            logger.info("audio data vlue:", audio_data.getvalue())
            with open("output.wav", "wb") as f:
                f.write(audio_data.getvalue())
                
            audio_end = time.time()

            logger.success("Audio data generated and sent successfully")

            audio_path = "./output.wav"
            binary, samplerate = createBinary('./output.wav')

            start_time = time.time()
            samplerate_str = str(samplerate)

            # create vowel and vowel_length
            end_time = time.time()

            logger.info(f"The time it take to generate a speech synthesis: {text} is {audio_end - audio_start_time} seconds")


            multipart_data = MultipartEncoder(
                fields={
                # 音声データを含める
                'audio': ('output.wav', binary, 'audio/wav'),
                # サンプルレートもフィールドに追加
                'samplerate': ('samplerate', samplerate_str, 'text/plain')
                }
            )


            logger.info("Generating audio -----------------------------------------")

            response = ByteAudioResponse(multipart_data.to_string())
            response.headers['Content-Type'] = multipart_data.content_type
            return response

        except Exception as e:
            logger.error(f"Error generating audio: {e}")

    @app.get("/models/info")
    def get_loaded_models_info():
        """Retrieve information about the loaded models"""

        result: dict[str, dict[str, Any]] = dict()
        for model_id, model in enumerate(loaded_models):
            result[str(model_id)] = {
                "config_path": str(model.config_path),
                "model_path": str(model.model_path),
                "device": model.device,
                "spk2id": model.spk2id,
                "id2spk": model.id2spk,
                "style2id": model.style2id,
            }
        return result

    logger.info(f"server listen: http://127.0.0.1:{config.server_config.port}")
    logger.info(f"API docs: http://127.0.0.1:5000/docs")
    uvicorn.run(
        app, port=5000, host="0.0.0.0", log_level="warning"
    )