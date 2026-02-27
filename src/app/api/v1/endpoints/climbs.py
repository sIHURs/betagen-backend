from fastapi import APIRouter

router = APIRouter(prefix="/climbs", tags=["climbs"])


@router.get("/sample")
def sample_climbs() -> list[dict[str, str]]:
    return [
        {"id": "c1", "name": "Blue Arete", "setter": "Team A"},
        {"id": "c2", "name": "Red Slab", "setter": "Team B"},
    ]
