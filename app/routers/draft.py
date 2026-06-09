from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from config import DATA_STATIC_DIR, PROSPECTS_JSON_PATH
from analytics.prospect_apfv import (
    PROSPECT_PERCENTILE_COLS,
    percentile_of_score,
)
from analytics.player_profiles.archetypes import (
    calculate_adjusted_pfv,
    calculate_apfv,
    calculate_pfv,
    height_bucket,
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
    pick: int | None = None
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


def _load_historical_prospect_file(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _historical_prospects_for_year(request: Request, year: int) -> list[dict] | None:
    cached = getattr(request.app.state, "historical_prospects_by_year", None)
    if cached is not None and year in cached:
        return cached[year]

    path = DATA_STATIC_DIR / "draft" / f"prospects_{year}.json"
    if not path.exists():
        return None
    return _load_historical_prospect_file(path)


def _all_prospect_population(request: Request) -> list[dict]:
    """Return current and historical prospects as one APFV comparison universe."""
    cached = getattr(request.app.state, "prospect_population", None)
    if cached is not None:
        return cached

    prospects = list(_prospects(request))
    draft_dir = DATA_STATIC_DIR / "draft"
    if draft_dir.exists():
        for path in draft_dir.glob("prospects_*.json"):
            try:
                prospects.extend(_load_historical_prospect_file(path))
            except Exception:
                continue
    return prospects


def _collect_all_adjusted_pfvs(prospects: list[dict]) -> list[float]:
    """Gather every prospect's MPG-adjusted PFV for population-level APFV ranking."""
    adjusted_pfvs = []
    for p in prospects:
        pfv_metrics = {
            col: dict(p[col])
            for col in PROSPECT_PERCENTILE_COLS
            if col in p and isinstance(p[col], dict)
        }
        adjusted_pfvs.append(calculate_adjusted_pfv(pfv_metrics, is_prospect=True))
    return adjusted_pfvs


def _ensure_global_percentiles(prospect: dict, request: Request) -> dict:
    """Ensure individual metric percentiles are calculated globally."""
    prospect_copy = dict(prospect)
    percentile_arrays = getattr(request.app.state, "global_prospect_percentile_arrays", {})
    for col in PROSPECT_PERCENTILE_COLS:
        if col in prospect_copy:
            metric = prospect_copy[col]
            if isinstance(metric, dict):
                metric_copy = dict(metric)
                val = metric_copy.get("value")
                if val is not None:
                    metric_copy["percentile"] = percentile_of_score(percentile_arrays.get(col, []), float(val))
                prospect_copy[col] = metric_copy
    return prospect_copy


def _ensure_pfv_apfv_in_raw_stats(prospect: dict, request: Request) -> dict:
    """Ensure globally-ranked PFV/APFV live inside raw_stats."""
    raw = dict(prospect.get("raw_stats", {}))

    percentile_arrays = getattr(request.app.state, "global_prospect_percentile_arrays", {})
    pfv_metrics = {}
    for col in PROSPECT_PERCENTILE_COLS:
        if col in prospect:
            metric = prospect[col]
            if isinstance(metric, dict):
                val = metric.get("value")
                if val is not None:
                    pct = percentile_of_score(percentile_arrays.get(col, []), float(val))
                    pfv_metrics[col] = {"value": float(val), "percentile": pct}

    pfv_val = calculate_pfv(pfv_metrics)
    raw["pfv"] = float(pfv_val)

    adjusted_pfv = calculate_adjusted_pfv(pfv_metrics, is_prospect=True)
    bucket = height_bucket(prospect.get("height", ""))

    global_adjusted_pfvs = getattr(request.app.state, "global_prospect_adjusted_pfvs", [])
    global_height_buckets = getattr(request.app.state, "global_prospect_height_buckets", [])

    bucket_adjusted_pfvs = [
        adj_pfv
        for adj_pfv, b in zip(global_adjusted_pfvs, global_height_buckets)
        if b == bucket
    ]

    if adjusted_pfv not in bucket_adjusted_pfvs:
        bucket_adjusted_pfvs.append(adjusted_pfv)

    apfv_val = calculate_apfv(adjusted_pfv, bucket_adjusted_pfvs)
    raw["apfv"] = float(apfv_val)
    raw.pop("npfv", None)

    return raw


@router.get("/prospects", response_model=list[ProspectListItem])
def list_prospects(request: Request, year: int | None = None):
    """Return every prospect's id, name, team, and raw counting-stat totals."""
    if year is not None:
        all_prospects = _historical_prospects_for_year(request, year)
        if all_prospects is None:
            raise HTTPException(
                status_code=404,
                detail=f"Prospects dataset for draft year {year} not found.",
            )
    else:
        all_prospects = _prospects(request)

    results = []
    for p in all_prospects:
        results.append(
            ProspectListItem(
                prospect_id=p["prospect_id"],
                player_name=p["player_name"],
                team=p.get("team", ""),
                height=p.get("height", ""),
                weight=p.get("weight", ""),
                role=p.get("role", ""),
                pick=p.get("pick"),
                raw_stats=_ensure_pfv_apfv_in_raw_stats(p, request),
            )
        )
    return results


@router.get("/{prospect_id}")
def get_prospect(prospect_id: str, request: Request):
    """Return the full dataset for a single prospect."""
    index = _prospects_by_id(request)
    prospect = index.get(prospect_id)

    if prospect is None:
        hist_map = getattr(request.app.state, "historical_prospects_map", {})
        path = hist_map.get(prospect_id)
        if path is not None:
            try:
                all_prospects = _load_historical_prospect_file(path)
                prospect = next((p for p in all_prospects if p["prospect_id"] == prospect_id), None)
            except Exception:
                pass

    if prospect is None:
        raise HTTPException(
            status_code=404,
            detail=f"Prospect '{prospect_id}' not found in current or historical draft classes.",
        )
    prospect = dict(prospect)

    # Recompute individual metric percentiles globally
    prospect = _ensure_global_percentiles(prospect, request)

    raw = _ensure_pfv_apfv_in_raw_stats(prospect, request)
    prospect["raw_stats"] = raw

    # Remove top-level pfv/apfv if they exist (they live in raw_stats now)
    prospect.pop("pfv", None)
    prospect.pop("apfv", None)
    prospect.pop("npfv", None)

    return prospect
