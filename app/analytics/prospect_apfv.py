from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from analytics.player_profiles.archetypes import (
    calculate_apfv_batch_by_height,
    calculate_impact_pfv,
    calculate_prospect_impact_adjusted_pfv,
    height_bucket,
)
from analytics.draft_year import PROSPECTS_KEY, normalize_prospects_payload

logger = logging.getLogger(__name__)

PROSPECT_PERCENTILE_COLS = [
    "mpg",
    "pts_per36",
    "reb_per36",
    "ast_per36",
    "blk_per36",
    "stl_per36",
    "ts_pct",
    "efg_pct",
    "fg3a_rate",
    "fta_rate",
]


def numeric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def prospect_metric_value(prospect: dict, col: str) -> float | None:
    metric = prospect.get(col)
    if isinstance(metric, dict):
        value = numeric(metric.get("value"))
        if value is not None:
            return value
    value = numeric(metric)
    if value is not None:
        return value

    raw = prospect.get("raw_stats", {})
    if col == "mpg":
        return numeric(raw.get("mpg"))
    return None


def percentile_of_score(values: list[float], score: float) -> float:
    if not values:
        return 0.0
    rank = sum(1 for value in values if value <= score)
    return round(rank / len(values) * 100.0, 1)


def prospect_percentile_arrays(population: list[dict]) -> dict[str, list[float]]:
    arrays: dict[str, list[float]] = {}
    for col in PROSPECT_PERCENTILE_COLS:
        values = [
            value
            for prospect in population
            if (value := prospect_metric_value(prospect, col)) is not None
        ]
        arrays[col] = values
    return arrays


def global_prospect_metrics(
    prospect: dict,
    percentile_arrays: dict[str, list[float]],
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for col in PROSPECT_PERCENTILE_COLS:
        value = prospect_metric_value(prospect, col)
        if value is None:
            value = 0.0
            percentile = 0.0
        else:
            percentile = percentile_of_score(percentile_arrays.get(col, []), value)
        metrics[col] = {"value": value, "percentile": percentile}
    return metrics


def calculate_global_prospect_pfv_apfv(population: list[dict]) -> dict[str, dict[str, float]]:
    percentile_arrays = prospect_percentile_arrays(population)
    pfvs: list[float] = []
    adjusted_pfvs: list[float] = []
    height_buckets: list[str] = []

    for prospect in population:
        metrics = global_prospect_metrics(prospect, percentile_arrays)
        gp_val = prospect_metric_value(prospect, "gp")
        if gp_val is None:
            gp_val = float(prospect.get("raw_stats", {}).get("gp", 0) or 0)
        metrics["gp"] = {"value": float(gp_val or 0), "percentile": 0.0}
        metrics["team"] = str(prospect.get("team", "") or "")
        pfvs.append(calculate_impact_pfv(metrics))
        adjusted_pfvs.append(calculate_prospect_impact_adjusted_pfv(metrics))
        height_buckets.append(height_bucket(prospect.get("height", "")))

    apfvs = calculate_apfv_batch_by_height(
        adjusted_pfvs, height_buckets,
        curve_exponent=2.2, raw_anchor=0.50,
    )
    return {
        str(prospect.get("prospect_id")): {"pfv": pfv, "apfv": apfv}
        for prospect, pfv, apfv in zip(population, pfvs, apfvs)
        if prospect.get("prospect_id")
    }


def apply_global_prospect_pfv_apfv(records: list[dict], lookup: dict[str, dict[str, float]]) -> int:
    updated = 0
    for record in records:
        values = lookup.get(str(record.get("prospect_id", "")))
        if not values:
            continue
        raw = dict(record.get("raw_stats", {}))
        raw["pfv"] = float(values["pfv"])
        raw["apfv"] = float(values["apfv"])
        raw.pop("npfv", None)
        record["raw_stats"] = raw
        record.pop("pfv", None)
        record.pop("apfv", None)
        record.pop("npfv", None)
        updated += 1
    return updated


def load_prospect_json_files(
    prospects_json_path: Path,
    draft_dir: Path,
) -> list[tuple[Path, list[dict], dict]]:
    """Load every prospect JSON file as ``(path, records, meta)`` triples.

    Handles both the wrapped current-class file (``{"draft_class_year": ...,
    "prospects": [...]}``) and legacy bare-list historical files. ``meta`` holds
    the wrapper's non-record keys so they can be preserved when rewriting.
    """
    files: list[tuple[Path, list[dict], dict]] = []
    if prospects_json_path.exists():
        records, meta = normalize_prospects_payload(
            json.loads(prospects_json_path.read_text(encoding="utf-8"))
        )
        files.append((prospects_json_path, records, meta))

    if draft_dir.exists():
        for path in sorted(draft_dir.glob("prospects_*.json")):
            records, meta = normalize_prospects_payload(
                json.loads(path.read_text(encoding="utf-8"))
            )
            files.append((path, records, meta))
    return files


def normalize_global_prospect_apfv_files(prospects_json_path: Path, draft_dir: Path) -> dict[str, int]:
    files = load_prospect_json_files(prospects_json_path, draft_dir)
    population = [record for _, records, _ in files for record in records]
    lookup = calculate_global_prospect_pfv_apfv(population)

    updated_json = 0
    for path, records, meta in files:
        updated_json += apply_global_prospect_pfv_apfv(records, lookup)
        if meta:
            payload = {**meta, PROSPECTS_KEY: records}
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        else:
            path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    updated_parquet = 0
    for json_path, _records, _meta in files:
        parquet_path = json_path.with_suffix(".parquet")
        if not parquet_path.exists():
            continue
        df = pd.read_parquet(parquet_path)
        if "prospect_id" not in df.columns:
            continue
        df["pfv"] = df["prospect_id"].map(lambda pid: lookup.get(str(pid), {}).get("pfv"))
        df["apfv"] = df["prospect_id"].map(lambda pid: lookup.get(str(pid), {}).get("apfv"))
        df.to_parquet(parquet_path, index=False, engine="fastparquet")
        updated_parquet += len(df)

    logger.info(
        "Normalized global prospect PFV/APFV for %d JSON records and %d parquet records.",
        updated_json,
        updated_parquet,
    )
    return {"json_records": updated_json, "parquet_records": updated_parquet}
