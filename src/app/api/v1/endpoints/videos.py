from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from app.api.deps import get_job_manager
from app.core.config import get_settings
from app.schemas.pose import (
    LocalProcessRequest,
    ResultsResponse,
    StartProcessRequest,
    StartProcessResponse,
    UploadVideoResponse,
)
from app.services.job_manager import JobManager

router = APIRouter(prefix="/videos", tags=["videos"])
ALLOWED_EXTENSIONS = {".mp4", ".mov"}
settings = get_settings()


@router.post("/upload", response_model=UploadVideoResponse)
async def upload_video(
    file: UploadFile = File(...),
    manager: JobManager = Depends(get_job_manager),
) -> UploadVideoResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only .mp4 and .mov are supported")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    video_id, _ = manager.save_upload(filename=file.filename or "upload.mp4", data=content)
    return UploadVideoResponse(video_id=video_id, filename=file.filename or "upload.mp4")


@router.post("/{video_id}/process", response_model=StartProcessResponse)
def start_processing(
    video_id: str,
    payload: StartProcessRequest | None = None,
    model: str = Query(default=settings.pose_default_model),
    manager: JobManager = Depends(get_job_manager),
) -> StartProcessResponse:
    try:
        input_path = manager.resolve_uploaded_video(video_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Video not found") from None

    try:
        record = manager.start_job(
            video_id=video_id,
            input_path=input_path,
            model=model,
            every_n_frames=payload.every_n_frames if payload else None,
            output_resolution=payload.output_resolution if payload else None,
            save_intermediate_frames=payload.save_intermediate_frames if payload else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return StartProcessResponse(job_id=record.job_id, video_id=video_id, status=record.status)


@router.post("/process-local", response_model=StartProcessResponse)
def process_local_video(
    payload: LocalProcessRequest,
    manager: JobManager = Depends(get_job_manager),
) -> StartProcessResponse:
    local_path = Path(payload.local_path).expanduser().resolve()
    if not local_path.exists() or not local_path.is_file():
        raise HTTPException(status_code=400, detail="local_path does not exist")
    if local_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only .mp4 and .mov are supported")

    video_id, input_path = manager.register_local_video(local_path)
    try:
        record = manager.start_job(
            video_id=video_id,
            input_path=input_path,
            model=payload.model,
            every_n_frames=payload.every_n_frames,
            output_resolution=payload.output_resolution,
            save_intermediate_frames=payload.save_intermediate_frames,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return StartProcessResponse(job_id=record.job_id, video_id=video_id, status=record.status)


@router.get("/{video_id}/results", response_model=ResultsResponse)
def get_video_results(
    video_id: str,
    manager: JobManager = Depends(get_job_manager),
) -> ResultsResponse:
    try:
        files = manager.list_result_files(video_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No output found for this video") from None
    return ResultsResponse(video_id=video_id, files=files)


@router.get("/{video_id}/download")
def download_video_result(
    video_id: str,
    type: str = Query(..., pattern="^(overlay|keypoints)$"),
    manager: JobManager = Depends(get_job_manager),
) -> FileResponse:
    try:
        path = manager.resolve_output_file(video_id, type)
    except ValueError:
        raise HTTPException(status_code=400, detail="type must be overlay or keypoints") from None
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Requested result file does not exist") from None

    media_type = "application/json" if type == "keypoints" else "video/mp4"
    return FileResponse(path=path, media_type=media_type, filename=path.name)
