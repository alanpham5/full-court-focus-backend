from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from config import PROSPECTS_JSON_PATH

router = APIRouter(prefix="/draft", tags=["draft"])



class ProspectListItem(BaseModel):
    """Lightweight row returned by GET /draft/prospects."""

    prospect_id: str
    player_name: str
    team: str
    height: str = ""
    weight: str = ""
    role: str = ""
    raw_stats: dict



def _prospects(request: Request) -> list[dict]:
    """Return the prospects list, preferring the in-memory cache."""
    cached = getattr(request.app.state, "prospects", None)
    if cached is not None:
        return cached
    if not PROSPECTS_JSON_PATH.exists():
        return []
    with PROSPECTS_JSON_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _prospects_by_id(request: Request) -> dict[str, dict]:
    """Return the prospects keyed by prospect_id, preferring the in-memory cache."""
    cached = getattr(request.app.state, "prospects_by_id", None)
    if cached is not None:
        return cached
    return {p["prospect_id"]: p for p in _prospects(request)}



@router.get("/prospects", response_model=list[ProspectListItem])
def list_prospects(request: Request):
    """Return every prospect's id, name, team, and raw counting-stat totals."""
    all_prospects = _prospects(request)
    return [
        ProspectListItem(
            prospect_id=p["prospect_id"],
            player_name=p["player_name"],
            team=p.get("team", ""),
            height=p.get("height", ""),
            weight=p.get("weight", ""),
            role=p.get("role", ""),
            raw_stats=p.get("raw_stats", {}),
        )
        for p in all_prospects
    ]


@router.get("/{prospect_id}")
def get_prospect(prospect_id: str, request: Request):
    """Return the full dataset for a single prospect."""
    index = _prospects_by_id(request)
    prospect = index.get(prospect_id)
    if prospect is None:
        raise HTTPException(
            status_code=404,
            detail=f"Prospect '{prospect_id}' not found. "
            f"Run build_prospects_dataset.py and ensure {PROSPECTS_JSON_PATH.name} exists.",
        )
    return prospect
