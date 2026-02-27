from fastapi import APIRouter

from app.api.v1.endpoints import analysis, climbs, jobs, videos

router = APIRouter()
router.include_router(climbs.router)
router.include_router(analysis.router)
router.include_router(videos.router)
router.include_router(jobs.router)
