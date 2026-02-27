from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as root_api_router
from app.api.v1.endpoints.health import router as health_router
from app.api.v1.endpoints.jobs import router as jobs_router
from app.api.v1.endpoints.videos import router as videos_router
from app.core.config import get_settings

settings = get_settings()
app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
# Keep root-level routes to match demo contract paths (/videos/*, /jobs/*).
app.include_router(videos_router)
app.include_router(jobs_router)
app.include_router(root_api_router)
