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

from analytics.badge_leaders import build_badge_leaders_index
from analytics.lineups_build import build_starting_lineups_index
from analytics.normalizer import normalize_by_season
from analytics.similarity import build_similar_teams_index
from analytics.team_profiles_build import build_team_profiles_json
from analytics.team_static_cache import (
    build_team_profile_static_cache,
    merge_similar_teams_with_abbreviations,
)
from parquet_io import read_teams_parquet, write_teams_parquet
from analytics.player_profiles.pipeline import (
    PlayerProfilePipeline,
    PlayerProfilePipelineConfig,
    copy_processed_outputs_to_static,
)
from analytics.player_profiles.storage import ProfileStorage, StorageConfig
from analytics.player_profiles.nba_client import NbaStatsClient

OUTPUT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "static"))
PARQUET_PATH = os.path.join(OUTPUT_DIR, "teams_historical.parquet")
METADATA_PATH = os.path.join(OUTPUT_DIR, "team_metadata.json")
SEASON_INDEX_PATH = os.path.join(OUTPUT_DIR, "season_index.json")
SIMILAR_TEAMS_PATH = os.path.join(OUTPUT_DIR, "similar_teams.json")
BADGE_LEADERS_PATH = os.path.join(OUTPUT_DIR, "badge_leaders.json")
STARTING_LINEUPS_PATH = os.path.join(OUTPUT_DIR, "starting_lineups.json")
TEAM_PROFILES_PATH = os.path.join(OUTPUT_DIR, "team_profiles.json")

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
]

SHOOTING_COLS = [
    "TEAM_ID",
    "SEASON",
    "FG3A",
    "FG3_PCT",
    "PAINT_FGA",
    "PAINT_FGA_PCT",
    "MID_RANGE_FGA",
    "FTA",
    "FGA",
]

MISC_COLS = ["TEAM_ID", "SEASON", "PTS_PAINT"]


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


def _fetch_misc_raw(season: str) -> pd.DataFrame | None:
    try:
        df = LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense="Misc",
            per_mode_detailed="Per100Possessions",
        ).get_data_frames()[0]
        df["SEASON"] = season
        return df
    except Exception as e:
        print(f"  [WARN] Misc fetch failed for {season}: {e}")
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

    misc = _fetch_misc_raw(season)
    if misc is not None:
        time.sleep(RATE_LIMIT_SLEEP)
        misc_keep = [c for c in MISC_COLS if c in misc.columns]
        if "PTS_PAINT" not in misc_keep:
            print(
                f"  [WARN] Misc response missing PTS_PAINT for {season} "
                "(paint percentile will default until fixed)"
            )
        else:
            merged = merged.merge(misc[misc_keep], on=["TEAM_ID", "SEASON"], how="left")
            ppaint = pd.to_numeric(merged["PTS_PAINT"], errors="coerce")
            if "PAINT_FGA" not in merged.columns:
                merged["PAINT_FGA"] = ppaint
            else:
                paint_fga = pd.to_numeric(merged["PAINT_FGA"], errors="coerce")
                merged["PAINT_FGA"] = paint_fga.mask(paint_fga.isna(), ppaint)
    else:
        print(
            f"  [WARN] Misc fetch failed for {season} "
            "(no PTS_PAINT / PAINT_FGA — paint percentile will default)"
        )

    if "FTA" in merged.columns and "FGA" in merged.columns:
        fga = pd.to_numeric(merged["FGA"], errors="coerce")
        fta = pd.to_numeric(merged["FTA"], errors="coerce")
        merged["FTA_RATE"] = fta / fga.replace(0, pd.NA)
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


def incremental_scrape() -> bool:
    """Refresh the current season. Returns True if new data was saved."""
    season = season_str(current_season_year())
    print(f"Incremental scrape: {season}")

    if not os.path.exists(PARQUET_PATH):
        print("No existing dataset found — running full scrape instead")
        full_scrape()
        return True

    existing = read_teams_parquet(PARQUET_PATH)

    existing = existing[existing["SEASON"] != season]

    new_data = merge_season(season)
    if new_data is None:
        print("Incremental fetch failed — keeping existing data")
        return False

    combined = pd.concat([existing, new_data], ignore_index=True)
    lineup_keys = [
        f"{int(tid)}:{season}"
        for tid in new_data["TEAM_ID"].unique()
    ]
    save(combined, lineup_keys=lineup_keys)
    return True


def _merge_lineups(keys: list[str]) -> dict:
    existing: dict = {}
    if os.path.exists(STARTING_LINEUPS_PATH):
        with open(STARTING_LINEUPS_PATH) as f:
            existing = json.load(f)
    print(
        f"  Building starting_lineups.json for {len(keys)} team-season(s) "
        f"({RATE_LIMIT_SLEEP}s between calls)..."
    )
    fresh = build_starting_lineups_index(keys, rate_limit_sleep=RATE_LIMIT_SLEEP)
    existing.update(fresh)
    return existing


def _merge_team_profiles(
    df: pd.DataFrame,
    norm: pd.DataFrame,
    similar_display: dict[str, list[dict]],
    *,
    profile_keys: list[str] | None,
) -> dict:
    if profile_keys is None or not os.path.exists(TEAM_PROFILES_PATH):
        print(
            f"  Building team_profiles.json (sequential leaders, {RATE_LIMIT_SLEEP}s between calls; "
            "failed rows retried once at the end)..."
        )
        return build_team_profiles_json(
            df,
            norm,
            similar_display,
            rate_limit_sleep=RATE_LIMIT_SLEEP,
        )

    wanted = set(profile_keys)
    existing: dict = {}
    with open(TEAM_PROFILES_PATH) as f:
        existing = json.load(f)
    static_profiles = build_team_profile_static_cache(df, norm, similar_display)

    profile_df = df[
        df.apply(lambda r: f"{int(r.TEAM_ID)}:{r.SEASON}" in wanted, axis=1)
    ].copy()
    profile_norm = norm[
        norm.apply(lambda r: f"{int(r.TEAM_ID)}:{r.SEASON}" in wanted, axis=1)
    ].copy()

    print(
        f"  Updating team_profiles.json for {len(wanted)} changed team-season(s) "
        f"({RATE_LIMIT_SLEEP}s between calls)..."
    )
    fresh = build_team_profiles_json(
        profile_df,
        profile_norm,
        similar_display,
        rate_limit_sleep=RATE_LIMIT_SLEEP,
    )
    merged: dict = {}
    for sk, static_payload in static_profiles.items():
        existing_payload = existing.get(sk, {})
        leaders = existing_payload.get("leaders")
        if leaders:
            merged[sk] = {**static_payload, "leaders": leaders}
    merged.update(fresh)
    return merged


def _load_env_defaults() -> None:
    env_path = _APP_ROOT / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def upload_local_to_gcs(local_dir: Path, gcs_uri: str) -> None:
    if not gcs_uri.startswith("gs://"):
        return
    try:
        from google.cloud import storage as gcs_storage
    except ModuleNotFoundError:
        print("  [WARN] google-cloud-storage not installed; skipping GCS upload.")
        return

    rest = gcs_uri[5:].strip("/")
    bucket_name, _, prefix = rest.partition("/")

    project = (
        os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCLOUD_PROJECT")
    )
    try:
        client = gcs_storage.Client(project=project)
        bucket = client.bucket(bucket_name)
    except Exception as e:
        print(f"  [WARN] Failed to initialize GCS client: {e}")
        return

    print(f"  Uploading player profiles data to gs://{bucket_name}/{prefix} ...")
    count = 0
    for path in local_dir.rglob("*"):
        if path.is_file():
            rel_path = path.relative_to(local_dir)
            blob_name = f"{prefix}/{rel_path}".strip("/")
            blob = bucket.blob(blob_name)

            if path.suffix == ".json":
                content_type = "application/json"
            elif path.suffix == ".parquet":
                content_type = "application/vnd.apache.parquet"
            else:
                content_type = "application/octet-stream"

            try:
                blob.upload_from_filename(str(path), content_type=content_type)
                count += 1
            except Exception as e:
                print(f"    Failed to upload {rel_path} to GCS: {e}")
    print(f"  ✓ Uploaded {count} files to GCS.")


def save(df: pd.DataFrame, *, lineup_keys: list[str] | None = None):
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

    print("  Building badge leaders (top 2 per badge per season)...")
    badge_leaders = build_badge_leaders_index(df, norm, metadata)
    with open(BADGE_LEADERS_PATH, "w") as f:
        json.dump(badge_leaders, f)
    print(
        f"  ✓ Badge leaders ({len(badge_leaders)} seasons) → {BADGE_LEADERS_PATH}"
    )

    print("  Building similar-teams index (k=6)...")
    sim_index = build_similar_teams_index(norm, k=6)
    with open(SIMILAR_TEAMS_PATH, "w") as f:
        json.dump(sim_index, f)
    print(f"  ✓ Similar teams index ({len(sim_index)} keys) → {SIMILAR_TEAMS_PATH}")

    similar_display = merge_similar_teams_with_abbreviations(sim_index, metadata)
    profiles = _merge_team_profiles(
        df,
        norm,
        similar_display,
        profile_keys=lineup_keys,
    )
    with open(TEAM_PROFILES_PATH, "w") as f:
        json.dump(profiles, f)
    print(f"  ✓ Team profiles ({len(profiles)} keys) → {TEAM_PROFILES_PATH}")

    if lineup_keys is None:
        deduped = df.drop_duplicates(["TEAM_ID", "SEASON"], keep="first")
        lineup_keys = [f"{int(r.TEAM_ID)}:{r.SEASON}" for _, r in deduped.iterrows()]
    lineups = _merge_lineups(lineup_keys)
    with open(STARTING_LINEUPS_PATH, "w") as f:
        json.dump(lineups, f)
    print(f"  ✓ Starting lineups ({len(lineups)} keys) → {STARTING_LINEUPS_PATH}")

    print("  Building player profiles and metadata...")
    seasons = sorted(df["SEASON"].unique().tolist())
    storage_uri = str(_APP_ROOT / "data" / "player_profiles")
    player_config = PlayerProfilePipelineConfig(
        seasons=seasons,
        current_season_start_year=int(seasons[-1][:4]),
        refresh_raw=False,
    )
    player_pipeline = PlayerProfilePipeline(
        storage=ProfileStorage(
            StorageConfig.from_uri(storage_uri)
        ),
        client=NbaStatsClient(timeout=60, retries=4, base_sleep=2.0),
        config=player_config,
    )
    player_pipeline.run()
    copy_processed_outputs_to_static(storage_uri, _APP_ROOT / "data" / "static")
    print("  ✓ Player profiles and metadata → app/data/static/")

    _load_env_defaults()
    gcs_uri = os.getenv("PLAYER_PROFILES_STORAGE_URI")
    if gcs_uri and gcs_uri.startswith("gs://"):
        upload_local_to_gcs(Path(storage_uri), gcs_uri)


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
