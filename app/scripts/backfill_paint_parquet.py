from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import LeagueDashTeamStats

_APP_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_ROOT = Path(__file__).resolve().parent
for _p in (_APP_ROOT, _SCRIPTS_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from parquet_io import read_teams_parquet, write_teams_parquet

STATIC_DIR = _APP_ROOT / "data" / "static"
PARQUET_DEFAULT = STATIC_DIR / "teams_historical.parquet"


def _fetch_pts_paint_season(season: str) -> pd.DataFrame | None:
    try:
        misc = LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense="Misc",
            per_mode_detailed="Per100Possessions",
        ).get_data_frames()[0]
    except Exception as e:
        print(f"  [WARN] Misc fetch failed for {season}: {e}")
        return None
    if "TEAM_ID" not in misc.columns or "PTS_PAINT" not in misc.columns:
        print(f"  [WARN] Misc response missing PTS_PAINT for {season}")
        return None
    out = misc[["TEAM_ID", "PTS_PAINT"]].copy()
    out["SEASON"] = season
    out["TEAM_ID"] = pd.to_numeric(out["TEAM_ID"], errors="coerce").astype("Int64")
    out["PTS_PAINT"] = pd.to_numeric(out["PTS_PAINT"], errors="coerce")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill PTS_PAINT (and PAINT_FGA alias) onto teams_historical.parquet from "
            "LeagueDashTeamStats Misc · Per100Possessions."
        ),
    )
    parser.add_argument("--parquet", type=Path, default=PARQUET_DEFAULT)
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.5,
        help="Seconds between NBA Stats calls",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch API data but do not write parquet",
    )
    parser.add_argument(
        "--season",
        action="append",
        default=None,
        help="Only backfill given season label (repeatable); default=all seasons present in parquet",
    )
    args = parser.parse_args()

    if not args.parquet.exists():
        raise SystemExit(f"Parquet not found: {args.parquet}")

    df = read_teams_parquet(args.parquet)
    seasons: list[str] = sorted(df["SEASON"].astype(str).unique())
    if args.season:
        wanted = set(args.season)
        seasons = [s for s in seasons if s in wanted]
        missing = wanted - set(seasons)
        if missing:
            print(f"  [WARN] Requested seasons not in parquet: {sorted(missing)}")

    if not seasons:
        raise SystemExit("No seasons to process.")

    print(f"Backfilling PTS_PAINT for {len(seasons)} season(s)…")
    frames: list[pd.DataFrame] = []
    for i, season in enumerate(seasons, start=1):
        patch = _fetch_pts_paint_season(season)
        if patch is not None:
            frames.append(patch)
            print(f"  [{i}/{len(seasons)}] {season}: {len(patch)} team row(s)")
        if i < len(seasons) and args.sleep > 0:
            time.sleep(args.sleep)

    if not frames:
        raise SystemExit("No patch data fetched — nothing to merge.")

    patch_df = pd.concat(frames, ignore_index=True)
    base = df.drop(columns=["PTS_PAINT", "PAINT_FGA"], errors="ignore").copy()
    base["TEAM_ID"] = pd.to_numeric(base["TEAM_ID"], errors="coerce").astype("Int64")
    base["SEASON"] = base["SEASON"].astype(str)

    merged = base.merge(patch_df, on=["TEAM_ID", "SEASON"], how="left")
    merged["PAINT_FGA"] = merged["PTS_PAINT"]

    na_p = merged["PTS_PAINT"].isna().sum()
    if na_p:
        print(
            f"  [WARN] {na_p} team-season row(s) still missing PTS_PAINT after merge "
            "(API gaps or TEAM_ID mismatch)"
        )

    if args.dry_run:
        print("Dry run — not writing parquet.")
        return

    write_teams_parquet(merged, args.parquet)
    print(f"Wrote parquet with PTS_PAINT / PAINT_FGA → {args.parquet}")


if __name__ == "__main__":
    main()
