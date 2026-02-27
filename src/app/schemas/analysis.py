from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    climb_name: str = Field(min_length=1, max_length=120)
    wall_angle: int = Field(ge=0, le=90)
    attempts: int = Field(ge=1, le=100)


class AnalyzeResponse(BaseModel):
    climb_name: str
    grade_estimate: str
    confidence: float
    notes: list[str]
