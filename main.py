import argparse
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote
from scipy.io import wavfile

import torch

from config import config
from style_bert_vits2.constants import Languages
from style_bert_vits2.logging import logger
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker as pyopenjtalk
from style_bert_vits2.nlp.japanese.user_dict import update_dict
from style_bert_vits2.tts_model import TTSModel, TTSModelHolder

ln = config.server_config.language

pyopenjtalk.initialize_worker()
update_dict()
bert_models.load_model(Languages.JP)
bert_models.load_tokenizer(Languages.JP)
bert_models.load_model(Languages.EN)
bert_models.load_tokenizer(Languages.EN)
bert_models.load_model(Languages.ZH)
bert_models.load_tokenizer(Languages.ZH)

loaded_models = []

def load_models(model_holder: TTSModelHolder):
    global loaded_models
    loaded_models = []
    for model_name, model_paths in model_holder.model_files_dict.items():
        model = TTSModel(
            model_path=model_paths[0],
            config_path=model_holder.root_dir / model_name / "config.json",
            style_vec_path=model_holder.root_dir / model_name / "style_vectors.npy",
            device=model_holder.device,
        )
        loaded_models.append(model)

def voice(text: str, model_id: int, **kwargs) -> Optional[BytesIO]:
    if model_id < 0 or model_id >= len(model_holder.model_names):
        raise ValueError(f"Invalid model ID: {model_id}")
    model = loaded_models[model_id]
    sr, audio = model.infer(text=text, **kwargs)

    wav_bytes_io = BytesIO()
    wavfile.write(wav_bytes_io, sr, audio.astype('int16'))  # Ensure audio data type is int16
    wav_bytes_io.seek(0)  # Reset the pointer to the beginning of the BytesIO object

    return wav_bytes_io

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument(
        "--dir", "-d", type=str, help="Model directory", default=config.assets_root
    )
    args = parser.parse_args()

    if args.cpu:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_dir = Path(args.dir)
    model_holder = TTSModelHolder(model_dir, device)
    if len(model_holder.model_names) == 0:
        logger.error(f"Models not found in {model_dir}.")
        sys.exit(1)

    logger.info("Loading models...")
    load_models(model_holder)

    text = "今日はいい天気ですね。外に出て、散歩したい気分です。"
    model_id = 3  # Set the desired model ID

    try:
        audio_data = voice(text, model_id)
        # Now you can do something with the audio data, such as saving it to a 
        logger.info("audio data vlue:", audio_data.getvalue())
        with open("output.wav", "wb") as f:
            f.write(audio_data.getvalue())
        logger.info("Audio generated successfully.")
    except Exception as e:
        logger.error(f"Error generating audio: {e}")