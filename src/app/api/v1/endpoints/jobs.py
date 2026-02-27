from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_job_manager
from app.schemas.pose import JobInfoResponse
from app.services.job_manager import JobManager

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobInfoResponse)
def get_job_status(
    job_id: str,
    manager: JobManager = Depends(get_job_manager),
) -> JobInfoResponse:
    try:
        job = manager.get_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found") from None
    return manager.job_to_response(job)
