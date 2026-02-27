from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "BetaGen Boulder Analyzer"
    api_v1_prefix: str = "/api/v1"
    frontend_origin: str = "http://127.0.0.1:5173"
    data_root: str = "data"
    uploads_dirname: str = "uploads"
    outputs_dirname: str = "outputs"
    pose_default_model: str = "openpose"
    pose_every_n_frames: int = 1
    pose_output_resolution: str = "original"
    pose_save_intermediate_frames: bool = False
    pose_max_video_seconds: int = 180

    @property
    def uploads_dir(self) -> Path:
        return Path(self.data_root) / self.uploads_dirname

    @property
    def outputs_dir(self) -> Path:
        return Path(self.data_root) / self.outputs_dirname


@lru_cache
def get_settings() -> Settings:
    return Settings()
