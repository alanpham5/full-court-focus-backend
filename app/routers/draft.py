from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from config import DATA_STATIC_DIR, PROSPECTS_JSON_PATH
from analytics.draft_year import (
    available_historical_draft_years,
    current_draft_year_from_date,
    current_draft_year_from_history,
    normalize_prospects_payload,
)
from analytics.prospect_apfv import (
    PROSPECT_PERCENTILE_COLS,
    percentile_of_score,
)
from analytics.player_profiles.archetypes import (
    calculate_apfv,
    calculate_impact_pfv,
    calculate_prospect_impact_adjusted_pfv,
    height_bucket,
)
from analytics.skill_ratings import SKILL_AXES, compute_skill_ratings
from models.schemas import PlayerMetricValue

router = APIRouter(prefix="/draft", tags=["draft"])

PROSPECT_BOX_KEYS = (
    "ppg",
    "rpg",
    "apg",
    "spg",
    "bpg",
    "mpg",
    "gp",
    "fg_pct",
    "fg3_pct",
    "ft_pct",
)


class ProspectListItem(BaseModel):

    prospect_id: str
    player_name: str
    team: str
    height: str = ""
    weight: str = ""
    role: str = ""
    pick: int | None = None
    apfv: float | None = None
    box_score: dict[str, PlayerMetricValue]
    skill_ratings: dict[str, float]



def _prospects(request: Request) -> list[dict]:
    cached = getattr(request.app.state, "prospects", None)
    if cached is not None:
        return cached
    if not PROSPECTS_JSON_PATH.exists():
        return []
    with PROSPECTS_JSON_PATH.open(encoding="utf-8") as fh:
        records, _ = normalize_prospects_payload(json.load(fh))
    return records


def _available_draft_years(request: Request) -> list[int]:
    """Historical draft classes available to select, ascending.

    Derived from the prospect files the historical scraper has written to disk,
    so the selectable set always matches the data that actually exists.
    """
    cached = getattr(request.app.state, "available_draft_years", None)
    if cached is not None:
        return list(cached)
    return available_historical_draft_years(DATA_STATIC_DIR / "draft")


def _draft_class_year(request: Request) -> int:
    cached = getattr(request.app.state, "draft_class_year", None)
    if cached is not None:
        return int(cached)
    if PROSPECTS_JSON_PATH.exists():
        try:
            with PROSPECTS_JSON_PATH.open(encoding="utf-8") as fh:
                _, meta = normalize_prospects_payload(json.load(fh))
            year = meta.get("draft_class_year")
            if year is not None:
                return int(year)
        except Exception:
            pass
    # No stored class year: infer from the historical classes on disk (newest
    # completed draft + 1), and only fall back to the date estimate as a last
    # resort when there is no data at all.
    return current_draft_year_from_history(_available_draft_years(request)) or (
        current_draft_year_from_date()
    )


def _prospects_by_id(request: Request) -> dict[str, dict]:
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
    adjusted_pfvs = []
    for p in prospects:
        pfv_metrics = {
            col: dict(p[col])
            for col in PROSPECT_PERCENTILE_COLS
            if col in p and isinstance(p[col], dict)
        }
        gp_val = float(p.get("gp", p.get("raw_stats", {}).get("gp", 0)) or 0)
        pfv_metrics["gp"] = {"value": gp_val, "percentile": 0.0}
        pfv_metrics["team"] = str(p.get("team", "") or "")
        adjusted_pfvs.append(calculate_prospect_impact_adjusted_pfv(pfv_metrics))
    return adjusted_pfvs


def _axis_percentiles(prospect: dict, request: Request) -> dict:
    arrays = getattr(request.app.state, "global_prospect_percentile_arrays", {})
    out: dict[str, float] = {}
    for axis in SKILL_AXES:
        metric = prospect.get(axis)
        if not isinstance(metric, dict):
            continue
        val = metric.get("value")
        if val is None:
            continue
        arr = arrays.get(axis)
        if arr is not None and len(arr) > 0:
            out[axis] = percentile_of_score(arr, float(val))
        elif metric.get("percentile") is not None:
            out[axis] = float(metric["percentile"])
    return out


def _prospect_skill_ratings(prospect: dict, request: Request, apfv) -> dict:
    return compute_skill_ratings(_axis_percentiles(prospect, request), apfv)


def _ensure_global_percentiles(prospect: dict, request: Request) -> dict:
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

    pfv_val = calculate_impact_pfv(pfv_metrics)
    raw["pfv"] = float(pfv_val)

    gp_val = float(prospect.get("gp", prospect.get("raw_stats", {}).get("gp", 0)) or 0)
    pfv_metrics["gp"] = {"value": gp_val, "percentile": 0.0}
    pfv_metrics["team"] = str(prospect.get("team", "") or "")
    adjusted_pfv = calculate_prospect_impact_adjusted_pfv(pfv_metrics)
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
    if year is not None:
        all_prospects = _historical_prospects_for_year(request, year)
        if all_prospects is None:
            raise HTTPException(
                status_code=404,
                detail=f"Prospects dataset for draft year {year} not found.",
            )
    else:
        all_prospects = _prospects(request)

    raws = [_ensure_pfv_apfv_in_raw_stats(p, request) for p in all_prospects]
    population = {
        key: [r[key] for r in raws if r.get(key) is not None]
        for key in PROSPECT_BOX_KEYS
    }

    results = []
    for p, raw in zip(all_prospects, raws):
        box_score = {}
        for key in PROSPECT_BOX_KEYS:
            val = raw.get(key)
            if val is None:
                box_score[key] = {"value": 0.0, "percentile": 0.0}
            else:
                box_score[key] = {
                    "value": float(val),
                    "percentile": percentile_of_score(population[key], float(val)),
                }
        apfv = raw.get("apfv")
        results.append(
            ProspectListItem(
                prospect_id=p["prospect_id"],
                player_name=p["player_name"],
                team=p.get("team", ""),
                height=p.get("height", ""),
                weight=p.get("weight", ""),
                role=p.get("role", ""),
                pick=p.get("pick"),
                apfv=apfv,
                box_score=box_score,
                skill_ratings=_prospect_skill_ratings(p, request, apfv),
            )
        )
    return results


@router.get("/meta")
def draft_meta(request: Request):
    """Metadata about the draft data: current class year and available history.

    ``available_draft_years`` is sourced from the prospect files on disk, so the
    frontend can build its past-draft picker from real data instead of assuming
    a contiguous range.
    """
    return {
        "draft_class_year": _draft_class_year(request),
        "available_draft_years": _available_draft_years(request),
    }


@router.get("/{prospect_id}")
def get_prospect(prospect_id: str, request: Request):
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

    prospect = _ensure_global_percentiles(prospect, request)

    raw = _ensure_pfv_apfv_in_raw_stats(prospect, request)
    prospect["raw_stats"] = raw
    prospect["skill_ratings"] = _prospect_skill_ratings(prospect, request, raw.get("apfv"))

    prospect.pop("pfv", None)
    prospect.pop("apfv", None)
    prospect.pop("npfv", None)

    return prospect
