from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from config import PROSPECTS_JSON_PATH
from analytics.player_profiles.archetypes import (
    calculate_adjusted_pfv,
    calculate_apfv,
    calculate_pfv,
    remove_mpg_adjustment_from_metrics,
)

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


def _collect_all_adjusted_pfvs(all_prospects: list[dict]) -> list[float]:
    """Gather every prospect's MPG-adjusted PFV for population-level APFV ranking."""
    adjusted_pfvs: list[float] = []
    for p in all_prospects:
        raw = p.get("raw_stats", {})
        if raw.get("pfv") is None and p.get("pfv") is None:
            raw["pfv"] = calculate_pfv(p)
        adjusted_pfvs.append(calculate_adjusted_pfv(p, is_prospect=True))
    return adjusted_pfvs


def _ensure_pfv_apfv_in_raw_stats(prospect: dict, all_adjusted_pfvs: list[float]) -> dict:
    """Ensure pfv and apfv live inside raw_stats, computing if missing."""
    raw = dict(prospect.get("raw_stats", {}))

    # Resolve PFV
    pfv_val = raw.get("pfv") or prospect.get("pfv")
    if pfv_val is None:
        pfv_val = calculate_pfv(prospect)
    pfv_val = float(pfv_val)

    # Resolve APFV
    adjusted_pfv = calculate_adjusted_pfv(prospect, is_prospect=True)
    apfv_val = raw.get("apfv") or prospect.get("apfv")
    if apfv_val is None:
        apfv_val = calculate_apfv(adjusted_pfv, all_adjusted_pfvs)
    apfv_val = float(apfv_val)

    raw["pfv"] = pfv_val
    raw["apfv"] = apfv_val
    raw.pop("npfv", None)
    return raw


@router.get("/prospects", response_model=list[ProspectListItem])
def list_prospects(request: Request):
    """Return every prospect's id, name, team, and raw counting-stat totals."""
    all_prospects = _prospects(request)
    all_adjusted_pfvs = _collect_all_adjusted_pfvs(all_prospects)

    results = []
    for p, adjusted_pfv in zip(all_prospects, all_adjusted_pfvs):
        raw = dict(p.get("raw_stats", {}))
        pfv_val = raw.get("pfv") or p.get("pfv")
        if pfv_val is None:
            pfv_val = calculate_pfv(p)
        raw["pfv"] = pfv_val
        apfv_val = raw.get("apfv") or p.get("apfv")
        if apfv_val is None:
            apfv_val = calculate_apfv(adjusted_pfv, all_adjusted_pfvs)
        raw["apfv"] = float(apfv_val)
        raw.pop("npfv", None)

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

    # Collect all adjusted PFVs in the class for APFV ranking
    all_prospects = _prospects(request)
    all_adjusted_pfvs = _collect_all_adjusted_pfvs(all_prospects)

    raw = _ensure_pfv_apfv_in_raw_stats(prospect, all_adjusted_pfvs)
    prospect["raw_stats"] = raw

    # Remove top-level pfv/apfv if they exist (they live in raw_stats now)
    prospect.pop("pfv", None)
    prospect.pop("apfv", None)
    prospect.pop("npfv", None)

    return prospect
