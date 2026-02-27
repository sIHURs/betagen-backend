from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class PoseKeypoint(BaseModel):
    name: str
    x: float
    y: float
    confidence: float = Field(ge=0.0, le=1.0)


class PoseResult(BaseModel):
    keypoints: list[PoseKeypoint] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    bbox: list[float] | None = None


class FramePoseRecord(BaseModel):
    frame_index: int = Field(ge=0)
    keypoints: list[PoseKeypoint] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    bbox: list[float] | None = None


class ProcessingConfig(BaseModel):
    model: str = "openpose"
    every_n_frames: int = Field(default=1, ge=1)
    output_resolution: str = Field(default="original")
    save_intermediate_frames: bool = False


class UploadVideoResponse(BaseModel):
    video_id: str
    filename: str


class StartProcessRequest(BaseModel):
    every_n_frames: int | None = Field(default=None, ge=1)
    output_resolution: str | None = Field(default=None)
    save_intermediate_frames: bool | None = None


class StartProcessResponse(BaseModel):
    job_id: str
    video_id: str
    status: JobStatus


class JobInfoResponse(BaseModel):
    job_id: str
    video_id: str
    model: str
    status: JobStatus
    error: str | None = None
    outputs: dict[str, str] = Field(default_factory=dict)


class ResultsResponse(BaseModel):
    video_id: str
    files: dict[str, str]


class LocalProcessRequest(BaseModel):
    local_path: str
    model: str = "openpose"
    every_n_frames: int | None = Field(default=None, ge=1)
    output_resolution: str | None = Field(default=None)
    save_intermediate_frames: bool | None = None


class KeypointsPayload(BaseModel):
    video_id: str
    model: str
    frames: list[FramePoseRecord]
    meta: dict[str, Any] = Field(default_factory=dict)
