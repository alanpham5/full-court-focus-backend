"""Recompute the displayed fingerprint percentiles (`playstyle_metrics`) in
player_profiles.json from the full pre-filter player population.

Reconstructs career aggregates from `player_season_features.parquet` (the same
normalized data the pipeline aggregates), recomputes the displayed percentiles
via `add_career_percentiles` + `style_summary`, and rewrites only
`playstyle_metrics`. Display percentiles are global (cross-position) and
credibility-shrunk — the same basis as PFV/APFV (§1.1) — so a skill a player is
not known for reads low in absolute terms rather than inflating against same-size
peers. Reproduces the pipeline's computation exactly. pfv/apfv are left untouched
(regenerate_player_pfv.py owns those). Run from the `app/` directory.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analytics.player_profiles.archetypes import style_summary  # noqa: E402
from analytics.player_profiles.features import (  # noqa: E402
    add_career_percentiles,
    aggregate_career_features,
)
from config import PLAYER_PROFILES_PATH, PLAYER_SEASON_FEATURES_PATH  # noqa: E402


def _read_parquet(path: Path) -> pd.DataFrame:
    for engine in ("fastparquet", "pyarrow"):
        try:
            return pd.read_parquet(path, engine=engine)
        except Exception:  # noqa: BLE001
            continue
    return pd.read_parquet(path)


def main() -> None:
    seasons = _read_parquet(Path(PLAYER_SEASON_FEATURES_PATH))
    career = add_career_percentiles(
        aggregate_career_features(seasons).reset_index(drop=True)
    )
    by_id = {int(r["player_id"]): r for _, r in career.iterrows()}

    profiles = json.loads(Path(PLAYER_PROFILES_PATH).read_text())

    updated = 0
    for payload in profiles.values():
        row = by_id.get(int(payload["player_id"]))
        if row is None:
            continue
        payload["playstyle_metrics"] = style_summary(row)
        updated += 1

    Path(PLAYER_PROFILES_PATH).write_text(json.dumps(profiles, indent=2))
    print(f"Updated playstyle_metrics for {updated} profiles -> {PLAYER_PROFILES_PATH}")


if __name__ == "__main__":
    main()
