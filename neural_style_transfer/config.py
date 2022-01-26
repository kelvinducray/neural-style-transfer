from functools import lru_cache

import torch
from pydantic import BaseSettings


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class Settings(BaseSettings):
    DEVICE: str = get_device()
    OUTPUT_IMG_SIZE: int = 512 if DEVICE == "cuda" else 128

    STYLE_IMG_DIR: str = "./images/style_images"
    CONTENT_IMG_DIR: str = "./images/content_images"


@lru_cache
def get_settings():
    return Settings()
