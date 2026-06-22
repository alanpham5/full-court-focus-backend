"""Recompute NBA player PFV/APFV in player_profiles.json from career features.

Rewrites only the `pfv` and `apfv` fields using the impact+versatility model in
`archetypes.py`; `similar_players` and all other fields are preserved. Pulls
global percentiles, career minutes, and mpg percentile from the career features
parquet. Run from the `app/` directory.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analytics.player_profiles.archetypes import (
    calculate_impact_apfv_batch,
    calculate_impact_pfv,
    global_scoring_metrics,
    player_credibility,
)
from analytics.player_profiles.season_profiles import career_apfv_from_seasons
from config import (
    PLAYER_CAREER_FEATURES_PATH,
    PLAYER_PROFILES_PATH,
    PLAYER_SEASON_FEATURES_PATH,
)


def _read_parquet(path: Path) -> pd.DataFrame:
    for engine in ("fastparquet", "pyarrow"):
        try:
            return pd.read_parquet(path, engine=engine)
        except Exception:
            continue
    return pd.read_parquet(path)


def main() -> None:
    career = _read_parquet(Path(PLAYER_CAREER_FEATURES_PATH)).reset_index(drop=True)
    by_id = {int(r["player_id"]): r for _, r in career.iterrows()}

    profiles = json.loads(Path(PLAYER_PROFILES_PATH).read_text())

    season_df = _read_parquet(Path(PLAYER_SEASON_FEATURES_PATH))
    career_apfvs = career_apfv_from_seasons(season_df)

    pids: list[str] = []
    pfvs: list[float] = []
    mpg_pctiles: list[float] = []
    minutes: list[float] = []
    for pid, payload in profiles.items():
        row = by_id.get(int(payload["player_id"]))
        if row is None:
            continue
        pfv = calculate_impact_pfv(
            global_scoring_metrics(row),
            credibility=player_credibility(row.get("career_minutes", 0.0)),
        )
        payload["pfv"] = pfv
        pids.append(pid)
        pfvs.append(pfv)
        mpg_pctiles.append(float(row.get("mpg_global_pctile", 0.0) or 0.0))
        minutes.append(float(row.get("career_minutes", 0.0) or 0.0))

    fallback = dict(zip(pids, calculate_impact_apfv_batch(pfvs, mpg_pctiles, minutes)))
    n_season = 0
    for pid in pids:
        if pid in career_apfvs:
            profiles[pid]["apfv"] = career_apfvs[pid]
            n_season += 1
        else:
            profiles[pid]["apfv"] = fallback[pid]

    Path(PLAYER_PROFILES_PATH).write_text(json.dumps(profiles, indent=2))
    print(
        f"Updated {len(pids)} profiles -> {PLAYER_PROFILES_PATH} "
        f"({n_season} from season APFVs, {len(pids) - n_season} fallback)"
    )


if __name__ == "__main__":
    main()
