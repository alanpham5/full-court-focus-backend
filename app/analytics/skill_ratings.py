from __future__ import annotations

SKILL_AXES = (
    "pts_per36",
    "reb_per36",
    "ast_per36",
    "blk_per36",
    "stl_per36",
    "ts_pct",
)

SHAPE_CONTRAST = 0.75


def _clamp(value: float) -> float:
    return max(0.0, min(100.0, value))


def compute_skill_ratings(
    percentiles: dict[str, float],
    apfv: float | None,
    contrast: float = SHAPE_CONTRAST,
) -> dict[str, float]:
    vals = [_clamp(float(percentiles.get(axis, 0.0) or 0.0)) for axis in SKILL_AXES]
    mean = sum(vals) / len(vals) if vals else 0.0
    level = apfv * 100.0 if isinstance(apfv, (int, float)) else mean
    return {
        axis: round(_clamp(level + contrast * (val - mean)), 1)
        for axis, val in zip(SKILL_AXES, vals)
    }
