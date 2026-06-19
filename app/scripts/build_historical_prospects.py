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

from env_defaults import load_env_defaults

load_env_defaults()

from config import DATA_STATIC_DIR, PLAYER_CAREER_FEATURES_PATH
from analytics.prospect_apfv import normalize_global_prospect_apfv_files
from pipelines.historical_prospects_pipeline import (
    HistoricalProspectsPipeline,
    fetch_html_with_retry,
    process_draft_classes,
    upload_prospect_outputs_to_gcs,
)

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

    result = process_draft_classes(
        years, force=args.force, recompute=args.recompute, sleep=args.sleep,
    )
    pipeline = result["pipeline"]
    success_count = result["success"]
    failure_count = result["failure"]

    if failure_count == 0:
        stats = normalize_global_prospect_apfv_files(DATA_STATIC_DIR / "prospects.json", static_draft_dir)
        logger.info("Global prospect APFV normalization complete: %s", stats)
        upload_prospect_outputs_to_gcs(include_current=True, include_historical=True)

    logger.info("Pipeline execution finished: %d years completed, %d failed.", success_count, failure_count)
    return 0 if failure_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
