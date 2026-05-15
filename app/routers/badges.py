from fastapi import APIRouter, HTTPException, Request

from analytics.playstyle import ALL_BADGES
from config import BADGE_LEADERS_PATH
from models.schemas import (
    Badge,
    BadgeExemplarTeam,
    BadgeLeaderEntry,
    SeasonBadgeLeadersResponse,
)

router = APIRouter(prefix="/badges", tags=["badges"])


@router.get("/{season}/leaders", response_model=SeasonBadgeLeadersResponse)
def get_season_badge_leaders(season: str, request: Request):
    index = getattr(request.app.state, "badge_leaders", {})
    season_data = index.get(season)
    if season_data is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Season not found. If data exists, run the scraper to build "
                f"{BADGE_LEADERS_PATH.name}."
            ),
        )

    badges = []
    for badge in ALL_BADGES:
        if badge.id not in season_data:
            continue
        entry = season_data[badge.id]
        top_raw = entry.get("top")
        runner_raw = entry.get("runner_up")
        badges.append(
            BadgeLeaderEntry(
                badge=Badge(**entry["badge"]),
                top=BadgeExemplarTeam(**top_raw) if top_raw else None,
                runner_up=BadgeExemplarTeam(**runner_raw) if runner_raw else None,
            )
        )

    return SeasonBadgeLeadersResponse(season=season, badges=badges)
