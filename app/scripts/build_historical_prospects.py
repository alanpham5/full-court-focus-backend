#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
import sys
import re
from pathlib import Path
from bs4 import BeautifulSoup

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
            import os
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))

_load_env_defaults()

from config import DATA_STATIC_DIR, PLAYER_CAREER_FEATURES_PATH
from analytics.prospect_apfv import normalize_global_prospect_apfv_files
from pipelines.historical_prospects_pipeline import HistoricalProspectsPipeline, fetch_html_with_retry

logger = logging.getLogger("historical_prospects_runner")

def get_available_years() -> list[int]:
    url = "https://basketball.realgm.com/nba/draft/past-drafts"
    try:
        logger.info("Fetching draft index page to discover draft years: %s", url)
        html = fetch_html_with_retry(url)
        soup = BeautifulSoup(html, "html.parser")
        select = soup.find("select")
        years = []
        if select:
            for opt in select.find_all("option"):
                val = opt.get("value", "")
                match = re.search(r'/past_drafts/(\d{4})', val)
                if match:
                    years.append(int(match.group(1)))
        if years:
            years = [y for y in years if y >= 2007]
            return sorted(list(set(years)))
    except Exception as e:
        logger.warning("Failed to scrape draft years dynamically: %s. Using fallback list.", e)
    
    return list(range(2007, 2026))

def main() -> int:
    parser = argparse.ArgumentParser(description="NBA Draft Historical Prospects Ingestion Pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all",
        action="store_true",
        help="Process all historical draft years since 2007",
    )
    group.add_argument(
        "--year",
        type=int,
        help="Process a specific historical draft year (e.g. 2023)",
    )
    group.add_argument(
        "--normalize-apfv-only",
        action="store_true",
        help="Only rewrite stored prospect PFV/APFV using the global current-plus-historical population",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-fetch of crawled drafts and player profile pages",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute similarity and overwrite outputs using cached raw data without re-fetching from web",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.5,
        help="Sleep duration (seconds) between requests to prevent rate limiting (default: 1.5)",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    static_draft_dir = DATA_STATIC_DIR / "draft"
    static_draft_dir.mkdir(parents=True, exist_ok=True)

    if args.normalize_apfv_only:
        years = []
    elif args.all:
        years = get_available_years()
        logger.info("Discovering draft years since 2007: %s", years)
    else:
        years = [args.year]
        if args.year < 2007:
            logger.error("Draft year must be >= 2007.")
            return 1

    pipeline = None
    success_count = 0
    failure_count = 0

    for year in years:
        json_path = static_draft_dir / f"prospects_{year}.json"
        parquet_path = static_draft_dir / f"prospects_{year}.parquet"
        
        if json_path.exists() and parquet_path.exists() and not args.force and not args.recompute:
            logger.info("Draft year %d has already been fully processed. Skipping. (Use --force or --recompute to override)", year)
            success_count += 1
            continue
            
        if pipeline is None:
            pipeline = HistoricalProspectsPipeline(
                career_features_path=PLAYER_CAREER_FEATURES_PATH,
                json_output_path=json_path,
                parquet_output_path=parquet_path,
                similar_count=4
            )
        else:
            pipeline.json_output_path = json_path
            pipeline.parquet_output_path = parquet_path

        try:
            force_scrape = args.force and not args.recompute
            pipeline.process_draft_class(year, force=force_scrape, sleep_time=args.sleep)
            success_count += 1
        except Exception as e:
            logger.exception("Failed to process draft class %d: %s", year, e)
            failure_count += 1

    if failure_count == 0:
        stats = normalize_global_prospect_apfv_files(DATA_STATIC_DIR / "prospects.json", static_draft_dir)
        logger.info("Global prospect APFV normalization complete: %s", stats)

        import os
        gcs_uri = os.getenv("PLAYER_PROFILES_STORAGE_URI")
        if gcs_uri and gcs_uri.startswith("gs://"):
            uploader = pipeline or HistoricalProspectsPipeline(career_features_path=PLAYER_CAREER_FEATURES_PATH)
            for path in [DATA_STATIC_DIR / "prospects.json", *(static_draft_dir.glob("prospects_*.json"))]:
                if path.exists():
                    uploader.upload_file_to_gcs(path, gcs_uri)
                    parquet_path = path.with_suffix(".parquet")
                    if parquet_path.exists():
                        uploader.upload_file_to_gcs(parquet_path, gcs_uri)

    logger.info("Pipeline execution finished: %d years completed, %d failed.", success_count, failure_count)
    return 0 if failure_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
