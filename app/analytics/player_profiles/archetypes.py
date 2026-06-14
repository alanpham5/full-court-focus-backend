from __future__ import annotations

from typing import Any

import pandas as pd

from analytics.player_profiles.features import PLAYSTYLE_METRIC_KEYS, height_to_inches

MPG_ADJUSTED_PERCENTILE_KEYS = {
    "pts_per36",
    "ast_per36",
    "reb_per36",
    "stl_per36",
    "blk_per36",
    "ts_pct",
    "efg_pct",
    "fg3a_rate",
    "fta_rate",
}


def _v(row: pd.Series, key: str) -> float:
    val = pd.to_numeric(row.get(key, 0.0), errors="coerce")
    return float(val) if pd.notna(val) else 0.0


def get_player_archetypes(row: pd.Series) -> list[str]:
    pts = _v(row, "pts_per36_z")
    ast = _v(row, "ast_per36_z")
    fg3 = _v(row, "fg3a_rate_z")
    fta = _v(row, "fta_rate_z")
    efg = _v(row, "efg_pct_z")
    tov = _v(row, "tov_per36_z")
    blk = _v(row, "blk_per36_z")
    stl = _v(row, "stl_per36_z")
    reb = _v(row, "reb_per36_z")

    matches = []

    if pts > 0.9 and ast > 0.7:
        matches.append("high-volume creator")
    if fg3 > 0.8 and efg > 0.2:
        matches.append("perimeter scorer")
    if fta > 0.8 and pts > 0.4:
        matches.append("free-throw pressure scorer")
    if ast > 0.8 and tov < 0.5:
        matches.append("table setter")
    if efg > 0.8 and pts < 0.6:
        matches.append("efficient finisher")
    if blk > 0.9 and reb > 0.4:
        matches.append("rim protection")
    if stl > 0.8 and fg3 > 0.2:
        matches.append("3-and-D profile")
    if reb > 0.9:
        matches.append("rebounding defender")
    if stl > 0.75 or blk > 0.75:
        matches.append("event creator")

    if matches:
        return matches[:3]

    fallbacks = []
    if pts > 0.0 or ast > 0.0:
        fallbacks.append("balanced scorer")
    if reb > 0.0 or blk > 0.0 or stl > 0.0:
        fallbacks.append("box-score defender")

    if not fallbacks:
        fallbacks = ["balanced scorer", "box-score defender"]
    return fallbacks[:3]


def assign_player_role(row: pd.Series) -> str:
    pts = _v(row, "pts_per36_z")
    ast = _v(row, "ast_per36_z")
    reb = _v(row, "reb_per36_z")
    stl = _v(row, "stl_per36_z")
    blk = _v(row, "blk_per36_z")
    ts = _v(row, "ts_pct_z")
    efg = _v(row, "efg_pct_z")
    ast_pct = _v(row, "ast_pct_z")
    fg3 = _v(row, "fg3a_rate_z")
    fta = _v(row, "fta_rate_z")

    is_big = (reb > 0.8 or blk > 0.8) and fg3 < -0.6

    if ast > 1.0 and ast_pct > 0.8:
        return "Playmaker"

    if pts > 1.0 and ast_pct < 0.6:
        return "Designated Scorer"

    if ast > 0.3 and (pts > 0.2 or ast_pct > 0.3):
        return "Secondary Creator"

    if fg3 > 0.8 and efg > -0.5:
        return "Perimeter Specialist"

    if fta > 0.5 and pts > 0.0:
        return "Rim Attacker"

    if (stl > 0.5 or blk > 0.5) and pts < -0.2:
        return "Defensive Specialist"

    if is_big and fg3 < -0.2:
        return "Interior Presence"

    if blk > 1.0 and reb > 0.8 and fg3 < 0.0:
        return "Interior Presence"

    if is_big:
        return "Interior Presence"

    if fg3 > 0.3:
        return "Perimeter Specialist"
    if ast > 0.0:
        return "Secondary Creator"
    if pts > 0.0:
        return "Rim Attacker"
    return "Defensive Specialist"


def add_archetypes(career_df: pd.DataFrame) -> pd.DataFrame:
    out = career_df.copy()
    out["archetypes"] = out.apply(get_player_archetypes, axis=1)
    out["role"] = out.apply(assign_player_role, axis=1)
    return out


def style_summary(row: pd.Series, *, adjust_for_mpg: bool = True) -> dict[str, dict[str, float]]:
    mpg_pct = _v(row, "mpg_career_pctile")
    return {
        k: {
            "value": round(_v(row, k), 4),
            "percentile": round(
                adjust_percentile_for_mpg(k, _v(row, f"{k}_career_pctile"), mpg_pct)
                if adjust_for_mpg
                else _v(row, f"{k}_career_pctile"),
                1,
            ),
        }
        for k in PLAYSTYLE_METRIC_KEYS
    }


def adjust_percentile_for_mpg(key: str, percentile: float, mpg_percentile: float) -> float:
    if key not in MPG_ADJUSTED_PERCENTILE_KEYS:
        return float(percentile)
    mpg_factor = max(0.0, min(float(mpg_percentile), 100.0)) / 100.0
    return round(float(percentile) * (mpg_factor ** 1.5), 4)


def remove_mpg_adjustment_from_metrics(metrics: dict) -> dict:
    out = {
        key: dict(value) if isinstance(value, dict) else value
        for key, value in metrics.items()
    }
    mpg_pct = _metric_percentile(out, "mpg")
    mpg_factor = max(0.0, min(mpg_pct, 100.0)) / 100.0
    if mpg_factor == 0.0:
        return out
    divisor = mpg_factor ** 1.5
    for key in MPG_ADJUSTED_PERCENTILE_KEYS:
        item = out.get(key)
        if not isinstance(item, dict):
            continue
        pct = item.get("percentile")
        if pct is not None:
            item["percentile"] = round(min(float(pct) / divisor, 100.0), 1)
    return out


def calculate_pfv(metrics: dict) -> float:
    import math

    keys = [
        "pts_per36",
        "reb_per36",
        "ast_per36",
        "blk_per36",
        "stl_per36",
        "ts_pct",
    ]
    r = [_metric_percentile(metrics, k) / 100.0 for k in keys]

    n = 6
    sum_product = 0.0
    for i in range(n):
        r_current = r[i]
        r_next = r[(i + 1) % n]
        sum_product += r_current * r_next

    sin_term = math.sin(2.0 * math.pi / n)
    A = 0.5 * sin_term * sum_product
    A_max = n * 0.5 * sin_term

    pfv = A / A_max
    return round(pfv, 4)


def calculate_adjusted_pfv(metrics: dict, *, is_prospect: bool = False) -> float:
    pfv = calculate_pfv(metrics)

    if is_prospect:
        mpg_val = float(_metric_value(metrics, "mpg"))
        ts_val = float(_metric_value(metrics, "ts_pct"))
        gp_val = float(_metric_value(metrics, "gp"))

        mpg_factor = max(0.0, min(mpg_val / 24.0, 1.0)) ** 1.4

        if gp_val > 0:
            gp_factor = max(0.0, min(gp_val / 25.0, 1.0)) ** 0.5
        else:
            gp_factor = 1.0

        if ts_val > 0:
            eff_factor = max(0.0, min(ts_val / 0.60, 1.0)) ** 1.0
        else:
            eff_factor = 1.0

        return round(pfv * mpg_factor * gp_factor * eff_factor, 4)

    mpg_pct = _metric_percentile(metrics, "mpg")
    mpg_factor = mpg_pct / 100.0
    return round(pfv * (mpg_factor ** 1.5), 4)


def _metric_value(metrics: dict, key: str) -> float:
    item = metrics.get(key)
    if isinstance(item, dict):
        v = item.get("value", 0.0)
    elif isinstance(item, (int, float)):
        v = item
    else:
        v = 0.0
    try:
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0




def calculate_apfv(
    adjusted_pfv: float,
    all_adjusted_pfvs: list[float],
    *,
    curve_exponent: float = 1.5,
    raw_anchor: float | None = None,
) -> float:
    n = len(all_adjusted_pfvs)
    if n == 0:
        return 0.0
    rank = sum(1 for v in all_adjusted_pfvs if v <= adjusted_pfv)
    pctile = rank / n
    rank_score = pctile ** float(curve_exponent)
    if raw_anchor is None or raw_anchor <= 0.0:
        return round(min(rank_score * 0.99, 0.99), 4)
    raw_score = max(0.0, min(adjusted_pfv / float(raw_anchor), 1.0))
    blended = (rank_score * raw_score) ** 0.5
    return round(min(blended * 0.99, 0.99), 4)


def calculate_apfv_batch(adjusted_pfvs: list[float], **kwargs) -> list[float]:
    return [calculate_apfv(v, adjusted_pfvs, **kwargs) for v in adjusted_pfvs]


def height_bucket(height_val: Any) -> str:
    inches = height_to_inches(height_val)
    if inches <= 0:
        return "wing"
    if inches < 76:
        return "guard"
    elif inches <= 80:
        return "wing"
    else:
        return "big"


def calculate_apfv_batch_by_height(
    adjusted_pfvs: list[float],
    height_buckets: list[str],
    *,
    curve_exponent: float = 1.5,
    raw_anchor: float | None = None,
) -> list[float]:
    n = len(adjusted_pfvs)
    results = [0.0] * n

    groups: dict[str, list[int]] = {}
    for i, bucket in enumerate(height_buckets):
        groups.setdefault(bucket, []).append(i)

    for _bucket, indices in groups.items():
        group_vals = [adjusted_pfvs[i] for i in indices]
        for i in indices:
            results[i] = calculate_apfv(
                adjusted_pfvs[i], group_vals,
                curve_exponent=curve_exponent, raw_anchor=raw_anchor,
            )

    return results


def profile_payload(row: pd.Series, similar: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    metrics = style_summary(row)
    pfv = calculate_pfv(style_summary(row, adjust_for_mpg=False))
    return {
        "player_id": int(row["player_id"]),
        "player_name": str(row["player_name"]),
        "height": _clean(row.get("height")),
        "weight": _clean(row.get("weight")),
        "draft_year": _clean(row.get("draft_year")),
        "draft_position": _clean(row.get("draft_position")),
        "role": _clean(row.get("role")),
        "career_teams": row.get("career_teams", []),
        "career_span": str(row.get("career_span", "")),
        "career_games": int(row.get("career_games", 0) or 0),
        "archetypes": list(row.get("archetypes", [])),
        "playstyle_metrics": metrics,
        "pfv": pfv,
        "similar_players": similar or [],
    }


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if pd.isna(value):
        return None
    return value


def _metric_percentile(metrics: dict, key: str) -> float:
    metric = metrics.get(key, {})
    pct = 0.0
    if isinstance(metric, dict):
        pct = metric.get("percentile", 0.0)
    elif isinstance(metric, (int, float)):
        pct = metric

    if pct is None:
        return 0.0
    return float(pct)
