from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from config import PROSPECTS_JSON_PATH
from analytics.player_profiles.archetypes import calculate_npfv, calculate_pfv

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


def _collect_all_pfvs(all_prospects: list[dict]) -> list[float]:
    """Gather every prospect's PFV for population-level NPFV ranking."""
    pfvs: list[float] = []
    for p in all_prospects:
        raw = p.get("raw_stats", {})
        pfv = raw.get("pfv") or p.get("pfv")
        if pfv is not None:
            pfvs.append(float(pfv))
        else:
            pfvs.append(calculate_pfv(p))
    return pfvs


def _ensure_pfv_npfv_in_raw_stats(prospect: dict, all_pfvs: list[float]) -> dict:
    """Ensure pfv and npfv live inside raw_stats, computing if missing."""
    raw = dict(prospect.get("raw_stats", {}))

    # Resolve PFV
    pfv_val = raw.get("pfv") or prospect.get("pfv")
    if pfv_val is None:
        pfv_val = calculate_pfv(prospect)
    pfv_val = float(pfv_val)

    # Resolve NPFV
    npfv_val = raw.get("npfv") or prospect.get("npfv")
    if npfv_val is None:
        npfv_val = calculate_npfv(pfv_val, all_pfvs)
    npfv_val = float(npfv_val)

    raw["pfv"] = pfv_val
    raw["npfv"] = npfv_val
    return raw


@router.get("/prospects", response_model=list[ProspectListItem])
def list_prospects(request: Request):
    """Return every prospect's id, name, team, and raw counting-stat totals."""
    all_prospects = _prospects(request)
    all_pfvs = _collect_all_pfvs(all_prospects)

    results = []
    for p, pfv_val in zip(all_prospects, all_pfvs):
        raw = dict(p.get("raw_stats", {}))
        raw["pfv"] = pfv_val
        npfv_val = raw.get("npfv") or p.get("npfv")
        if npfv_val is None:
            npfv_val = calculate_npfv(pfv_val, all_pfvs)
        raw["npfv"] = float(npfv_val)

        results.append(
            ProspectListItem(
                prospect_id=p["prospect_id"],
                player_name=p["player_name"],
                team=p.get("team", ""),
                height=p.get("height", ""),
                weight=p.get("weight", ""),
                role=p.get("role", ""),
                raw_stats=raw,
            )
        )
    return results


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
    prospect = dict(prospect)

    # Collect all PFVs in the class for NPFV ranking
    all_prospects = _prospects(request)
    all_pfvs = _collect_all_pfvs(all_prospects)

    raw = _ensure_pfv_npfv_in_raw_stats(prospect, all_pfvs)
    prospect["raw_stats"] = raw

    # Remove top-level pfv/npfv if they exist (they live in raw_stats now)
    prospect.pop("pfv", None)
    prospect.pop("npfv", None)

    return prospect
