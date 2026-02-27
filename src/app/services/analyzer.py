from app.schemas.analysis import AnalyzeRequest, AnalyzeResponse


class MockAnalysisService:
    def analyze(self, payload: AnalyzeRequest) -> AnalyzeResponse:
        if payload.wall_angle >= 40 and payload.attempts <= 3:
            grade = "V4"
            confidence = 0.82
            notes = [
                "Steep wall with low attempts suggests stronger power output.",
                "Add motion tracking to improve confidence."
            ]
        else:
            grade = "V2"
            confidence = 0.68
            notes = [
                "Baseline estimate from route metadata only.",
                "Upload video and hold map in next iteration."
            ]

        return AnalyzeResponse(
            climb_name=payload.climb_name,
            grade_estimate=grade,
            confidence=confidence,
            notes=notes,
        )
