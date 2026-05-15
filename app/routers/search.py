import json

from fastapi import APIRouter, Query, Request
from rapidfuzz import fuzz, process

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

    if not metadata or not choices:
        meta_path = TEAM_METADATA_PATH
        with meta_path.open() as f:
            metadata = json.load(f)
        choices = {tid: meta["name"] for tid, meta in metadata.items()}

    matches = process.extract(
        q,
        choices,
        scorer=fuzz.WRatio,
        limit=6,
        score_cutoff=40,
    )

    return [
        SearchSuggestion(
            team_id=int(tid),
            team_name=metadata[tid]["name"],
            abbreviation=metadata[tid]["abbreviation"],
        )
        for _, _, tid in matches
    ]


@router.get("/seasons/{team_id}")
def available_seasons(team_id: int, request: Request):
    index = getattr(request.app.state, "season_index", None)
    if index is None:
        with SEASON_INDEX_PATH.open() as f:
            index = json.load(f)
    return {"seasons": index.get(str(team_id), [])}
