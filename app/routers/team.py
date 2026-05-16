from fastapi import APIRouter, HTTPException, Request

from analytics.eras import is_valid_era, normalize_era
from analytics.similarity import find_similar_teams
from config import STARTING_LINEUPS_PATH, TEAM_PROFILES_PATH
from models.schemas import (
    Badge,
    EraSimilarTeamsResponse,
    LineupPlayer,
    SimilarTeam,
    StartingLineupResponse,
    StatLeaders,
    StyleVector,
    TeamProfileResponse,
)

router = APIRouter(prefix="/team", tags=["team"])


def _similar_with_abbr(
    matches: list[dict],
    metadata: dict[str, dict],
) -> list[SimilarTeam]:
    out: list[SimilarTeam] = []
    for item in matches:
        tid = str(item["team_id"])
        abbr = str(metadata.get(tid, {}).get("abbreviation", ""))
        out.append(SimilarTeam(**{**item, "abbreviation": abbr}))
    return out


@router.get("/{team_id}/{season}/similar-by-era/{era}", response_model=EraSimilarTeamsResponse)
def get_similar_teams_by_era(
    team_id: int,
    season: str,
    era: str,
    request: Request,
):
    if not is_valid_era(era):
        raise HTTPException(
            status_code=400,
            detail="era must be one of: 1990s, 2000s, 2010s, 2020s",
        )
    era_norm = normalize_era(era)
    norm_df = getattr(request.app.state, "norm_df", None)
    if norm_df is None:
        raise HTTPException(
            status_code=503,
            detail="Team history not loaded — teams_historical.parquet is required.",
        )

    metadata = getattr(request.app.state, "team_metadata", {})
    matches = find_similar_teams(
        team_id,
        season,
        norm_df,
        k=6,
        era=era_norm,
    )
    return EraSimilarTeamsResponse(
        team_id=team_id,
        season=season,
        era=era_norm,
        similar_teams=_similar_with_abbr(matches, metadata),
    )


@router.get("/{team_id}/{season}/lineup", response_model=StartingLineupResponse)
def get_starting_lineup(team_id: int, season: str, request: Request):
    key = f"{team_id}:{season}"
    lineup = getattr(request.app.state, "starting_lineups", {}).get(key)
    if lineup is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Lineup not found. Run the scraper to build {STARTING_LINEUPS_PATH.name}."
            ),
        )
    starters = []
    for p in lineup["starters"]:
        roles = p.get("roles")
        if not roles and p.get("role"):
            roles = [p["role"]]
        _legacy = {
            "Primary Playmaker": "Playmaker",
            "Primary Scorer": "Designated Scorer",
        }
        roles = [_legacy.get(r, r) for r in roles]
        starters.append(
            LineupPlayer(
                player_id=int(p["player_id"]),
                name=str(p["name"]),
                roles=list(roles or ["Rotation Piece"]),
                gp=int(p["gp"]),
                mpg=float(p["mpg"]),
            )
        )
    return StartingLineupResponse(
        team_id=int(lineup["team_id"]),
        season=str(lineup["season"]),
        source=str(lineup["source"]),
        lineup_mpg=lineup.get("lineup_mpg"),
        starters=starters,
    )


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
