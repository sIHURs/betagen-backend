from fastapi import APIRouter

from app.schemas.analysis import AnalyzeRequest, AnalyzeResponse
from app.services.analyzer import MockAnalysisService

router = APIRouter(prefix="/analysis", tags=["analysis"])
service = MockAnalysisService()


@router.post("/mock", response_model=AnalyzeResponse)
def analyze_mock(payload: AnalyzeRequest) -> AnalyzeResponse:
    return service.analyze(payload)
