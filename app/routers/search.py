import json

from fastapi import APIRouter, Query, Request

from analytics.team_search import rank_team_search
from config import SEASON_INDEX_PATH, TEAM_METADATA_PATH
from models.schemas import SearchSuggestion

router = APIRouter(prefix="/search", tags=["search"])


@router.get("/teams", response_model=list[SearchSuggestion])
def search_teams(
    request: Request,
    q: str = Query(..., min_length=1),
):
    metadata = getattr(request.app.state, "team_metadata", None)
    choices = getattr(request.app.state, "team_search_choices", None)

    if not metadata:
        with TEAM_METADATA_PATH.open() as f:
            metadata = json.load(f)

    ranked_ids = rank_team_search(q, metadata, limit=6, score_cutoff=35)

    return [
        SearchSuggestion(
            team_id=int(tid),
            team_name=metadata[tid]["name"],
            abbreviation=metadata[tid]["abbreviation"],
        )
        for tid in ranked_ids
    ]


@router.get("/seasons/{team_id}")
def available_seasons(team_id: int, request: Request):
    index = getattr(request.app.state, "season_index", None)
    if index is None:
        with SEASON_INDEX_PATH.open() as f:
            index = json.load(f)
    return {"seasons": index.get(str(team_id), [])}
