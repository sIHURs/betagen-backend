from __future__ import annotations

import shutil
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from app.models.registry import PoseModelRegistry
from app.pipelines.video_processor import VideoProcessor
from app.schemas.pose import JobStatus, JobInfoResponse, ProcessingConfig


@dataclass
class JobRecord:
    job_id: str
    video_id: str
    model: str
    status: JobStatus = JobStatus.pending
    error: str | None = None
    outputs: dict[str, str] = field(default_factory=dict)


class JobManager:
    def __init__(
        self,
        uploads_dir: Path,
        outputs_dir: Path,
        default_every_n_frames: int,
        default_output_resolution: str,
        default_save_intermediate_frames: bool,
        max_video_seconds: int,
    ) -> None:
        self.uploads_dir = uploads_dir
        self.outputs_dir = outputs_dir
        self.default_every_n_frames = default_every_n_frames
        self.default_output_resolution = default_output_resolution
        self.default_save_intermediate_frames = default_save_intermediate_frames
        self.max_video_seconds = max_video_seconds

        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        self._registry = PoseModelRegistry()
        self._processor = VideoProcessor(registry=self._registry, max_video_seconds=max_video_seconds)
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def save_upload(self, filename: str, data: bytes) -> tuple[str, Path]:
        ext = Path(filename).suffix.lower()
        video_id = uuid.uuid4().hex[:12]
        dst = self.uploads_dir / f"{video_id}{ext}"
        dst.write_bytes(data)
        return video_id, dst

    def register_local_video(self, local_path: Path) -> tuple[str, Path]:
        ext = local_path.suffix.lower()
        video_id = uuid.uuid4().hex[:12]
        dst = self.uploads_dir / f"{video_id}{ext}"
        shutil.copy2(local_path, dst)
        return video_id, dst

    def resolve_uploaded_video(self, video_id: str) -> Path:
        matches = sorted(self.uploads_dir.glob(f"{video_id}.*"))
        if not matches:
            raise FileNotFoundError(f"Video '{video_id}' not found")
        return matches[0]

    def start_job(
        self,
        video_id: str,
        input_path: Path,
        model: str,
        every_n_frames: int | None = None,
        output_resolution: str | None = None,
        save_intermediate_frames: bool | None = None,
    ) -> JobRecord:
        if model.lower() not in self._registry.supported_models():
            supported = ", ".join(self._registry.supported_models())
            raise ValueError(f"Unsupported model '{model}'. Supported: {supported}")

        job_id = uuid.uuid4().hex
        record = JobRecord(job_id=job_id, video_id=video_id, model=model)

        with self._lock:
            self._jobs[job_id] = record

        config = ProcessingConfig(
            model=model,
            every_n_frames=every_n_frames or self.default_every_n_frames,
            output_resolution=output_resolution or self.default_output_resolution,
            save_intermediate_frames=(
                self.default_save_intermediate_frames
                if save_intermediate_frames is None
                else save_intermediate_frames
            ),
        )

        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, video_id, input_path, config),
            daemon=True,
        )
        thread.start()
        return record

    def get_job(self, job_id: str) -> JobRecord:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return self._jobs[job_id]

    def list_result_files(self, video_id: str) -> dict[str, str]:
        out = self.outputs_dir / video_id
        if not out.exists():
            raise FileNotFoundError(video_id)

        files: dict[str, str] = {}
        keypoints = out / "keypoints.json"
        overlay = out / "overlay.mp4"
        if keypoints.exists():
            files["keypoints"] = str(keypoints)
        if overlay.exists():
            files["overlay"] = str(overlay)

        frames_dir = out / "frames"
        if frames_dir.exists():
            files["frames_dir"] = str(frames_dir)
        return files

    def resolve_output_file(self, video_id: str, file_type: str) -> Path:
        out = self.outputs_dir / video_id
        candidates = {
            "overlay": out / "overlay.mp4",
            "keypoints": out / "keypoints.json",
        }
        if file_type not in candidates:
            raise ValueError(file_type)
        path = candidates[file_type]
        if not path.exists():
            raise FileNotFoundError(str(path))
        return path

    def job_to_response(self, job: JobRecord) -> JobInfoResponse:
        return JobInfoResponse(
            job_id=job.job_id,
            video_id=job.video_id,
            model=job.model,
            status=job.status,
            error=job.error,
            outputs=job.outputs,
        )

    def _run_job(self, job_id: str, video_id: str, input_path: Path, config: ProcessingConfig) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.status = JobStatus.running

        try:
            outputs = self._processor.process_video(
                video_id=video_id,
                input_path=input_path,
                output_dir=self.outputs_dir / video_id,
                config=config,
            )
            with self._lock:
                record = self._jobs[job_id]
                record.status = JobStatus.completed
                record.outputs = outputs
        except Exception as exc:  # pragma: no cover - background error path
            with self._lock:
                record = self._jobs[job_id]
                record.status = JobStatus.failed
                record.error = str(exc)
