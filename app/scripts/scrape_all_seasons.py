from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

_APP_ROOT = Path(__file__).resolve().parents[1]
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

import pandas as pd
from nba_api.stats.endpoints import LeagueDashTeamStats
from nba_api.stats.static import teams as nba_teams

from analytics.normalizer import normalize_by_season
from analytics.similarity import build_similar_teams_index
from analytics.team_profiles_build import build_team_profiles_json
from analytics.team_static_cache import merge_similar_teams_with_abbreviations
from parquet_io import read_teams_parquet, write_teams_parquet

OUTPUT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "static"))
PARQUET_PATH = os.path.join(OUTPUT_DIR, "teams_historical.parquet")
METADATA_PATH = os.path.join(OUTPUT_DIR, "team_metadata.json")
SEASON_INDEX_PATH = os.path.join(OUTPUT_DIR, "season_index.json")
SIMILAR_TEAMS_PATH = os.path.join(OUTPUT_DIR, "similar_teams.json")

FIRST_SEASON_YEAR = 1996
RATE_LIMIT_SLEEP = 1.5

ADVANCED_COLS = [
    "TEAM_ID",
    "TEAM_NAME",
    "SEASON",
    "W",
    "L",
    "W_PCT",
    "PACE",
    "OFF_RATING",
    "DEF_RATING",
    "NET_RATING",
    "TS_PCT",
    "EFG_PCT",
    "AST_PCT",
    "AST_TO",
    "TM_TOV_PCT",
    "OREB_PCT",
    "DREB_PCT",
    "REB_PCT",
    "FTA_RATE",
]

SHOOTING_COLS = [
    "TEAM_ID",
    "SEASON",
    "FG3A",
    "FG3_PCT",
    "PAINT_FGA",
    "PAINT_FGA_PCT",
    "MID_RANGE_FGA",
]


def season_str(year: int) -> str:
    return f"{year}-{str(year + 1)[2:]}"


def current_season_year() -> int:
    today = datetime.today()
    return today.year if today.month >= 10 else today.year - 1


def all_season_years(start: int = FIRST_SEASON_YEAR) -> list[int]:
    return list(range(start, current_season_year() + 1))


def _fetch_advanced_raw(season: str) -> pd.DataFrame | None:
    try:
        df = LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
        ).get_data_frames()[0]
        df["SEASON"] = season
        return df
    except Exception as e:
        print(f"  [WARN] Advanced fetch failed for {season}: {e}")
        return None


def _fetch_shooting_raw(season: str) -> pd.DataFrame | None:
    try:
        df = LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense="Base",
            per_mode_detailed="Per100Possessions",
        ).get_data_frames()[0]
        df["SEASON"] = season
        return df
    except Exception as e:
        print(f"  [WARN] Shooting fetch failed for {season}: {e}")
        return None


def merge_season(season: str) -> pd.DataFrame | None:
    adv = _fetch_advanced_raw(season)
    if adv is None:
        return None
    time.sleep(RATE_LIMIT_SLEEP)
    sht = _fetch_shooting_raw(season)
    if sht is None:
        return None
    time.sleep(RATE_LIMIT_SLEEP)

    adv_keep = [c for c in ADVANCED_COLS if c in adv.columns]
    sht_keep = [c for c in SHOOTING_COLS if c in sht.columns]

    merged = adv[adv_keep].merge(sht[sht_keep], on=["TEAM_ID", "SEASON"], how="left")
    return merged


def build_team_metadata() -> dict:
    all_teams = nba_teams.get_teams()
    metadata = {
        str(t["id"]): {
            "id": t["id"],
            "name": t["full_name"],
            "abbreviation": t["abbreviation"],
            "nickname": t["nickname"],
        }
        for t in all_teams
    }
    return metadata


def full_scrape():
    print(f"Starting full scrape: {FIRST_SEASON_YEAR} → present")
    rows = []

    for year in all_season_years():
        season = season_str(year)
        print(f"  Fetching {season}...")
        df = merge_season(season)
        if df is not None:
            rows.append(df)
        else:
            print(f"  Skipping {season} (fetch failed)")

    if not rows:
        raise RuntimeError("No data fetched — check nba_api connectivity")

    combined = pd.concat(rows, ignore_index=True)
    save(combined)


def incremental_scrape():
    season = season_str(current_season_year())
    print(f"Incremental scrape: {season}")

    if not os.path.exists(PARQUET_PATH):
        print("No existing dataset found — running full scrape instead")
        full_scrape()
        return

    existing = read_teams_parquet(PARQUET_PATH)

    existing = existing[existing["SEASON"] != season]

    new_data = merge_season(season)
    if new_data is None:
        print("Incremental fetch failed — keeping existing data")
        return

    combined = pd.concat([existing, new_data], ignore_index=True)
    save(combined)


def save(df: pd.DataFrame):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    write_teams_parquet(df, PARQUET_PATH)
    print(f"  ✓ Saved {len(df)} team-seasons → {PARQUET_PATH}")

    index = df.groupby("TEAM_ID")["SEASON"].apply(list).to_dict()
    index = {str(k): v for k, v in index.items()}
    with open(SEASON_INDEX_PATH, "w") as f:
        json.dump(index, f)
    print(f"  ✓ Season index → {SEASON_INDEX_PATH}")

    metadata = build_team_metadata()
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Team metadata → {METADATA_PATH}")

    norm = normalize_by_season(df)
    print("  Building similar-teams index (k=6)...")
    sim_index = build_similar_teams_index(norm, k=6)
    with open(SIMILAR_TEAMS_PATH, "w") as f:
        json.dump(sim_index, f)
    print(f"  ✓ Similar teams index ({len(sim_index)} keys) → {SIMILAR_TEAMS_PATH}")

    TEAM_PROFILES_OUTPUT = os.path.join(OUTPUT_DIR, "team_profiles.json")
    similar_display = merge_similar_teams_with_abbreviations(sim_index, metadata)
    print(
        f"  Building team_profiles.json (sequential leaders, {RATE_LIMIT_SLEEP}s between calls; "
        "failed rows retried once at the end)..."
    )
    profiles = build_team_profiles_json(
        df,
        norm,
        similar_display,
        rate_limit_sleep=RATE_LIMIT_SLEEP,
    )
    with open(TEAM_PROFILES_OUTPUT, "w") as f:
        json.dump(profiles, f)
    print(f"  ✓ Team profiles ({len(profiles)} keys) → {TEAM_PROFILES_OUTPUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--current",
        action="store_true",
        help="Only refresh the current season (for scheduled runs)",
    )
    args = parser.parse_args()

    if args.current:
        incremental_scrape()
    else:
        full_scrape()
