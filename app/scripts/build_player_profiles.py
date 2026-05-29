from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_APP_ROOT = Path(__file__).resolve().parents[1]
for p in (_APP_ROOT, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DEFAULT_STORAGE_URI = str(_APP_ROOT / "data" / "player_profiles")


def _load_env_defaults() -> None:
    env_path = _APP_ROOT / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


from analytics.player_profiles.nba_client import NbaStatsClient
from analytics.player_profiles.pipeline import (
    PlayerProfilePipeline,
    PlayerProfilePipelineConfig,
    copy_processed_outputs_to_static,
)
from analytics.player_profiles.seasons import seasons_since_1996
from analytics.player_profiles.storage import ProfileStorage, StorageConfig


def main() -> None:
    _load_env_defaults()
    parser = argparse.ArgumentParser(
        description="Build NBA career-level player profiles from nba_api/stats.nba.com only.",
    )
    parser.add_argument(
        "--storage-uri",
        default=os.getenv("PLAYER_PROFILES_STORAGE_URI", DEFAULT_STORAGE_URI),
        help=(
            "Local path or gs://bucket/prefix. "
            "Defaults to PLAYER_PROFILES_STORAGE_URI or app/data/player_profiles."
        ),
    )
    parser.add_argument(
        "--gcs-project",
        default=os.getenv("GOOGLE_CLOUD_PROJECT"),
        help="Google Cloud project for gs:// storage. Defaults to GOOGLE_CLOUD_PROJECT.",
    )
    parser.add_argument("--start-season", type=int, default=1996)
    parser.add_argument("--end-season", type=int, default=None)
    parser.add_argument("--refresh-raw", action="store_true")
    parser.add_argument("--copy-to-static", action="store_true")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--sleep", type=float, default=1.2)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    seasons = [s for s in seasons_since_1996(args.end_season) if int(s[:4]) >= args.start_season]
    config = PlayerProfilePipelineConfig(
        seasons=seasons,
        current_season_start_year=args.end_season or int(seasons[-1][:4]),
        refresh_raw=args.refresh_raw,
    )
    pipeline = PlayerProfilePipeline(
        storage=ProfileStorage(
            StorageConfig.from_uri(args.storage_uri, gcs_project=args.gcs_project)
        ),
        client=NbaStatsClient(timeout=args.timeout, retries=args.retries, base_sleep=args.sleep),
        config=config,
    )
    summary = pipeline.run()
    print(summary)

    if args.copy_to_static:
        if args.storage_uri.startswith("gs://"):
            raise SystemExit("--copy-to-static is only supported for local storage builds")
        copy_processed_outputs_to_static(args.storage_uri, _APP_ROOT / "data" / "static")
        print(f"Copied API artifacts to {_APP_ROOT / 'data' / 'static'}")


if __name__ == "__main__":
    main()
