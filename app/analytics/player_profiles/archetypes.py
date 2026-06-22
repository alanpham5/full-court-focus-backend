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





PLAYER_IMPACT_WEIGHTS = {
    "scoring": 0.32,
    "playmaking": 0.18,
    "efficiency": 0.08,
    "rebounding": 0.09,
    "steals": 0.12,
    "blocks": 0.07,
    "spacing": 0.08,
    "ball_security": 0.06,
}
PLAYER_IMPACT_EXPONENT = 0.72
PLAYER_VERSATILITY_EXPONENT = 0.28
PLAYER_CREDIBILITY_MINUTES = 2500.0
PLAYER_LONGEVITY_MPG_EXP = 0.50
PLAYER_LONGEVITY_VOLUME_EXP = 0.25
PLAYER_LONGEVITY_SATURATION_MINUTES = 4000.0
PLAYER_LONGEVITY_VOLUME_CURVE = 0.5
PLAYER_APFV_CURVE = 1.2

_IMPACT_PCT_KEYS = {
    "pts": "pts_per36",
    "ts": "ts_pct",
    "efg": "efg_pct",
    "ast": "ast_per36",
    "ast_pct": "ast_pct",
    "reb": "reb_per36",
    "stl": "stl_per36",
    "blk": "blk_per36",
    "fg3a": "fg3a_rate",
    "tov": "tov_per36",
}


def player_credibility(career_minutes: float) -> float:
    """Sample-size credibility in [0, 1]; small-minute per-36 lines regress to median."""
    cm = max(float(career_minutes or 0.0), 0.0)
    return min(1.0, (cm / PLAYER_CREDIBILITY_MINUTES) ** 0.5)


def longevity_factor(career_minutes: float) -> float:
    """Saturating career-volume credit in [0, 1] for the APFV workload term.

    A raw career-minutes *rank* scaled linearly with accumulation, so a top-PFV
    rookie or young star sat near the bottom and APFV read as career-length more
    than impact. This curve instead saturates after roughly a full starter season
    (~PLAYER_LONGEVITY_SATURATION_MINUTES) and is applied with a light exponent,
    so it acts only as a small-sample gate: a single impactful rookie season can
    still earn a high APFV, while fractional-season samples are damped.
    """
    cm = max(float(career_minutes or 0.0), 0.0)
    return min(
        1.0,
        (cm / PLAYER_LONGEVITY_SATURATION_MINUTES) ** PLAYER_LONGEVITY_VOLUME_CURVE,
    )


def _skill_components(p: dict[str, float]) -> tuple[float, float, float, float, float]:
    """Derive scoring, playmaking, spacing, ball-security, and defense from percentiles."""
    scoring = p["pts"] * (0.55 + 0.45 * p["ts"])
    playmaking = 0.6 * p["ast"] + 0.4 * p["ast_pct"]
    spacing = p["fg3a"] * (0.4 + 0.6 * p["efg"])
    ball_security = 1.0 - p["tov"]
    defense = 1.0 - (1.0 - p["stl"]) * (1.0 - p["blk"])
    return scoring, playmaking, spacing, ball_security, defense


def player_impact_score(p: dict[str, float]) -> float:
    """Box-weighted, efficiency-aware impact magnitude in [0, 1]."""
    scoring, playmaking, spacing, ball_security, _defense = _skill_components(p)
    w = PLAYER_IMPACT_WEIGHTS
    return (
        w["scoring"] * scoring
        + w["playmaking"] * playmaking
        + w["efficiency"] * p["ts"]
        + w["rebounding"] * p["reb"]
        + w["steals"] * p["stl"]
        + w["blocks"] * p["blk"]
        + w["spacing"] * spacing
        + w["ball_security"] * ball_security
    )


def player_versatility_score(p: dict[str, float]) -> float:
    """Breadth across macro-skills (geometric mean, no-weakness) in [0, 1].

    Macro-skills are creation, efficiency, defense, rebounding, and spacing. The
    defense axis is an OR-gate so a guard can be a versatile defender through
    steals without being penalized for not blocking shots.
    """
    import math

    scoring, playmaking, spacing, _bs, defense = _skill_components(p)
    creation = 0.5 * max(scoring, playmaking) + 0.5 * (0.6 * scoring + 0.4 * playmaking)
    macros = [max(m, 1e-3) for m in (creation, p["ts"], defense, p["reb"], spacing)]
    return float(math.exp(sum(math.log(m) for m in macros) / len(macros)))


def calculate_impact_pfv(metrics: dict, *, credibility: float = 1.0) -> float:
    """Impact + versatility PFV from the global-percentile playstyle vector.

    `metrics` must carry global (cross-position) percentiles. Missing axes are
    treated as neutral (50th). Percentiles are shrunk toward the median by the
    sample-size `credibility` before scoring.
    """
    p: dict[str, float] = {}
    for short, key in _IMPACT_PCT_KEYS.items():
        raw = _metric_percentile_default(metrics, key, 50.0) / 100.0
        p[short] = 0.5 + credibility * (raw - 0.5)
    impact = player_impact_score(p)
    versatility = player_versatility_score(p)
    pfv = (impact ** PLAYER_IMPACT_EXPONENT) * (versatility ** PLAYER_VERSATILITY_EXPONENT)
    return round(pfv, 4)


def calculate_prospect_impact_adjusted_pfv(metrics: dict) -> float:
    """Prospect adjusted PFV on the impact+versatility base (§1.1).

    Uses the impact PFV (`calculate_impact_pfv`) as the rate-quality base and
    applies the prospect translatability dampers — minutes, games played, and
    efficiency on *raw* values — identical in form to the legacy
    `calculate_adjusted_pfv(is_prospect=True)`. Longevity does not apply to
    pre-draft players, so the NBA career-volume term is not used here.

    Legacy `calculate_adjusted_pfv` is left intact for the similarity quality
    axis (`prospect_quality_scores` / `nba_quality_scores`).
    """
    pfv = calculate_impact_pfv(metrics)

    mpg_val = float(_metric_value(metrics, "mpg"))
    ts_val = float(_metric_value(metrics, "ts_pct"))
    gp_val = float(_metric_value(metrics, "gp"))

    mpg_factor = max(0.0, min(mpg_val / 24.0, 1.0)) ** 1.4
    gp_factor = max(0.0, min(gp_val / 25.0, 1.0)) ** 0.5 if gp_val > 0 else 1.0
    eff_factor = max(0.0, min(ts_val / 0.60, 1.0)) if ts_val > 0 else 1.0

    return round(pfv * mpg_factor * gp_factor * eff_factor, 4)


def calculate_impact_apfv_batch(
    pfvs: list[float],
    mpg_percentiles: list[float],
    career_minutes: list[float],
    *,
    curve_exponent: float = PLAYER_APFV_CURVE,
) -> list[float]:
    """Workload/longevity-adjusted, globally ranked APFV in [0, 0.99].

    Rate quality (PFV) is scaled by a workload factor that blends per-game role
    (mpg percentile) with a *saturating* career-volume term (`longevity_factor`),
    so sustained production is rewarded up to a credibility point but does not
    keep compounding — an impactful young star is no longer buried beneath a
    long-career compiler. The adjusted values are ranked across the entire pool
    (not by height bucket), preserving absolute, cross-position impact ordering.
    """
    n = len(pfvs)
    if n == 0:
        return []

    adjusted = []
    for i in range(n):
        mpg_frac = max(0.0, min(float(mpg_percentiles[i] or 0.0), 100.0)) / 100.0
        longevity = longevity_factor(career_minutes[i])
        workload = (mpg_frac ** PLAYER_LONGEVITY_MPG_EXP) * (
            longevity ** PLAYER_LONGEVITY_VOLUME_EXP
        )
        adjusted.append(pfvs[i] * workload)

    adj_order = sorted(range(n), key=lambda i: adjusted[i])
    apfv = [0.0] * n
    for rank, i in enumerate(adj_order):
        pctile = (rank + 1) / n
        apfv[i] = round(min((pctile ** float(curve_exponent)) * 0.99, 0.99), 4)
    return apfv


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
    elif inches < 84:
        return "big"
    else:
        return "center"


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
    pfv = calculate_impact_pfv(
        global_scoring_metrics(row),
        credibility=player_credibility(row.get("career_minutes", 0.0)),
    )
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


def _metric_percentile_default(metrics: dict, key: str, default: float) -> float:
    """Like `_metric_percentile` but returns `default` for absent/empty axes."""
    metric = metrics.get(key)
    if isinstance(metric, dict):
        pct = metric.get("percentile")
    elif isinstance(metric, (int, float)):
        pct = metric
    else:
        pct = None
    return float(pct) if pct is not None else float(default)


def global_scoring_metrics(row: pd.Series) -> dict[str, dict[str, float]]:
    """Build a metrics dict of global (cross-position) percentiles for impact scoring."""
    return {
        key: {
            "value": round(_v(row, key), 4),
            "percentile": round(_v(row, f"{key}_global_pctile"), 1),
        }
        for key in PLAYSTYLE_METRIC_KEYS
    }
