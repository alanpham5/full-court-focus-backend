#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

_APP_ROOT = Path(__file__).resolve().parents[1]
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

def _load_env_defaults() -> None:
    env_path = _APP_ROOT / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))

_load_env_defaults()

from config import (
    TEAMS_PARQUET_PATH,
    TEAM_METADATA_PATH,
    SEASON_INDEX_PATH,
    BADGE_LEADERS_PATH,
    SIMILAR_TEAMS_PATH,
    STARTING_LINEUPS_PATH,
    TEAM_PROFILES_PATH,
    PLAYER_CAREER_FEATURES_PATH,
    PROSPECTS_JSON_PATH,
)
from pipelines.team_stats_pipeline import TeamStatsPipeline
from pipelines.badge_leaders_pipeline import BadgeLeadersPipeline
from pipelines.similar_teams_pipeline import SimilarTeamsPipeline
from pipelines.lineups_pipeline import LineupsPipeline
from pipelines.team_profiles_pipeline import TeamProfilesPipeline
from pipelines.player_profiles_pipeline import PlayerProfilesPipeline
from pipelines.prospects_pipeline import ProspectsPipeline
from analytics.prospect_apfv import normalize_global_prospect_apfv_files

logger = logging.getLogger("pipeline_orchestrator")


def upload_local_to_gcs(local_dir: Path, gcs_uri: str) -> None:
    if not gcs_uri.startswith("gs://"):
        return
    if str(local_dir).startswith("gs://"):
        logger.info("Storage URI is a GCS path. Outputs were written directly to GCS; skipping local GCS sync.")
        return
    try:
        from google.cloud import storage as gcs_storage
    except ModuleNotFoundError:
        logger.warning("google-cloud-storage not installed; skipping GCS upload.")
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
        logger.warning("Failed to initialize GCS client: %s", e)
        return

    logger.info("Uploading player profiles data to gs://%s/%s ...", bucket_name, prefix)
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
                logger.error("Failed to upload %s to GCS: %s", rel_path, e)
    logger.info("✓ Uploaded %s files to GCS.", count)


def run_pipeline_orchestrator(
    stages_to_run: list[str],
    current_season_only: bool = False,
    storage_uri: str | None = None,
    gcs_project: str | None = None,
    refresh_raw: bool = False,
    timeout: int = 60,
    retries: int = 4,
    sleep_time: float = 2.0,
    rate_limit: float = 1.5,
) -> int:
    allowed_stages = {"team", "badge", "similar", "lineup", "profile", "player", "prospect"}
    stages_to_run = [s.strip().lower() for s in stages_to_run if s.strip()]
    for s in stages_to_run:
        if s not in allowed_stages:
            logger.error("Invalid stage: %s. Choose from: %s", s, allowed_stages)
            return 1

    logger.info("Executing pipeline stages: %s", " -> ".join(stages_to_run))
    df = None
    sim_index = None

    if storage_uri is None:
        storage_uri = str(_APP_ROOT / "data" / "player_profiles")

    try:
        if "team" in stages_to_run:
            pipeline = TeamStatsPipeline(
                parquet_path=TEAMS_PARQUET_PATH,
                metadata_path=TEAM_METADATA_PATH,
                season_index_path=SEASON_INDEX_PATH,
                rate_limit_sleep=rate_limit,
            )
            df = pipeline.run(current_season_only=current_season_only)
        elif os.path.exists(TEAMS_PARQUET_PATH):
            from parquet_io import read_teams_parquet
            df = read_teams_parquet(TEAMS_PARQUET_PATH)

        if "badge" in stages_to_run:
            pipeline = BadgeLeadersPipeline(
                parquet_path=TEAMS_PARQUET_PATH,
                metadata_path=TEAM_METADATA_PATH,
                output_path=BADGE_LEADERS_PATH,
            )
            pipeline.run(df)

        if "similar" in stages_to_run:
            pipeline = SimilarTeamsPipeline(
                parquet_path=TEAMS_PARQUET_PATH,
                output_path=SIMILAR_TEAMS_PATH,
                k=6,
            )
            sim_index = pipeline.run(df)
        elif os.path.exists(SIMILAR_TEAMS_PATH):
            with open(SIMILAR_TEAMS_PATH) as f:
                sim_index = json.load(f)

        lineup_keys = None
        if current_season_only and df is not None:
            from pipelines.team_stats_pipeline import season_str, current_season_year
            curr_season = season_str(current_season_year())
            curr_data = df[df["SEASON"] == curr_season]
            lineup_keys = [f"{int(tid)}:{curr_season}" for tid in curr_data["TEAM_ID"].unique()]

        if "lineup" in stages_to_run:
            pipeline = LineupsPipeline(
                parquet_path=TEAMS_PARQUET_PATH,
                output_path=STARTING_LINEUPS_PATH,
                rate_limit_sleep=rate_limit,
            )
            if lineup_keys is None:
                if df is not None:
                    deduped = df.drop_duplicates(["TEAM_ID", "SEASON"], keep="first")
                    lineup_keys = [f"{int(r.TEAM_ID)}:{r.SEASON}" for _, r in deduped.iterrows()]
                else:
                    lineup_keys = pipeline.get_known_keys()

            pipeline.run(lineup_keys, replace=False, timeout=15, retries=2)

        if "profile" in stages_to_run:
            pipeline = TeamProfilesPipeline(
                parquet_path=TEAMS_PARQUET_PATH,
                metadata_path=TEAM_METADATA_PATH,
                similar_teams_path=SIMILAR_TEAMS_PATH,
                output_path=TEAM_PROFILES_PATH,
                rate_limit_sleep=rate_limit,
            )
            pipeline.run(df, profile_keys=lineup_keys, sim_index=sim_index)

        if "player" in stages_to_run:
            pipeline = PlayerProfilesPipeline(
                storage_uri=storage_uri,
                gcs_project=gcs_project,
                timeout=timeout,
                retries=retries,
                sleep=sleep_time,
            )
            if df is not None:
                seasons = sorted(df["SEASON"].unique().tolist())
            else:
                from analytics.player_profiles.seasons import seasons_since_1996
                seasons = seasons_since_1996(2023)

            current_season_start_year = int(seasons[-1][:4])
            pipeline.run(
                seasons=seasons,
                current_season_start_year=current_season_start_year,
                refresh_raw=refresh_raw,
                copy_to_static_dir=_APP_ROOT / "data" / "static",
            )

            gcs_uri = os.getenv("PLAYER_PROFILES_STORAGE_URI")
            if gcs_uri and gcs_uri.startswith("gs://"):
                upload_local_to_gcs(Path(storage_uri), gcs_uri)

        if "prospect" in stages_to_run:
            from pipelines.prospects_pipeline import DEFAULT_SOURCE_URL
            pipeline = ProspectsPipeline(
                source_url=DEFAULT_SOURCE_URL,
                career_features_path=PLAYER_CAREER_FEATURES_PATH,
                json_output_path=PROSPECTS_JSON_PATH,
                parquet_output_path=PROSPECTS_JSON_PATH.with_suffix(".parquet"),
                similar_count=4,
            )
            pipeline.run()
            normalize_global_prospect_apfv_files(
                PROSPECTS_JSON_PATH,
                PROSPECTS_JSON_PATH.parent / "draft",
            )

        logger.info("Pipeline executed successfully!")
        return 0

    except Exception as exc:
        logger.exception("Pipeline execution failed: %s", exc)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Basketball Analytics Pipeline Orchestrator")
    parser.add_argument(
        "--current",
        action="store_true",
        help="Only refresh/process the current season (incremental scrape & update)",
    )
    parser.add_argument(
        "--stages",
        default="all",
        help="Comma-separated list of stages to run: team,badge,similar,lineup,profile,player,prospect (default: all)",
    )
    parser.add_argument(
        "--storage-uri",
        default=os.getenv("PLAYER_PROFILES_STORAGE_URI", str(_APP_ROOT / "data" / "player_profiles")),
        help="Storage root path for player profiles pipeline",
    )
    parser.add_argument(
        "--gcs-project",
        default=os.getenv("GOOGLE_CLOUD_PROJECT"),
        help="GCS project for player profiles pipeline",
    )
    parser.add_argument("--refresh-raw", action="store_true", help="Refresh raw files in player profiles stage")
    parser.add_argument("--timeout", type=int, default=60, help="Client timeout for player stats NBA API requests")
    parser.add_argument("--retries", type=int, default=4, help="Client retries for player stats NBA API requests")
    parser.add_argument("--sleep", type=float, default=2.0, help="Sleep duration between player stats requests")
    parser.add_argument("--rate-limit", type=float, default=1.5, help="Sleep duration between team stats/lineup requests")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.stages == "all":
        stages_to_run = ["team", "badge", "similar", "lineup", "profile", "player", "prospect"]
    else:
        stages_to_run = [s.strip().lower() for s in args.stages.split(",") if s.strip()]

    return run_pipeline_orchestrator(
        stages_to_run=stages_to_run,
        current_season_only=args.current,
        storage_uri=args.storage_uri,
        gcs_project=args.gcs_project,
        refresh_raw=args.refresh_raw,
        timeout=args.timeout,
        retries=args.retries,
        sleep_time=args.sleep,
        rate_limit=args.rate_limit,
    )


if __name__ == "__main__":
    sys.exit(main())
