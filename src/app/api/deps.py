from functools import lru_cache

from app.core.config import get_settings
from app.services.job_manager import JobManager


@lru_cache
def get_job_manager() -> JobManager:
    settings = get_settings()
    return JobManager(
        uploads_dir=settings.uploads_dir,
        outputs_dir=settings.outputs_dir,
        default_every_n_frames=settings.pose_every_n_frames,
        default_output_resolution=settings.pose_output_resolution,
        default_save_intermediate_frames=settings.pose_save_intermediate_frames,
        max_video_seconds=settings.pose_max_video_seconds,
    )
