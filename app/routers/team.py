from fastapi import APIRouter, HTTPException, Request

from config import TEAM_PROFILES_PATH
from models.schemas import Badge, SimilarTeam, StatLeaders, StyleVector, TeamProfileResponse

router = APIRouter(prefix="/team", tags=["team"])


@router.get("/{team_id}/{season}", response_model=TeamProfileResponse)
def get_team_profile(team_id: int, season: str, request: Request):
    key = f"{team_id}:{season}"
    profile = getattr(request.app.state, "team_profiles", {}).get(key)
    if profile is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Team/season not found. If the dataset exists, run the scraper to "
                f"build {TEAM_PROFILES_PATH.name} (includes stat leaders)."
            ),
        )

    return TeamProfileResponse(
        team_id=int(profile["team_id"]),
        team_name=str(profile["team_name"]),
        season=str(profile["season"]),
        record=str(profile["record"]),
        win_pct=float(profile["win_pct"]),
        style_vector=StyleVector(**profile["style_vector"]),
        badges=[Badge(**b) for b in profile["badges"]],
        leaders=StatLeaders(**profile["leaders"]),
        similar_teams=[SimilarTeam(**s) for s in profile["similar_teams"]],
    )
