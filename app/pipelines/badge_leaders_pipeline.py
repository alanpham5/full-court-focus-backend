from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from analytics.badge_leaders import build_badge_leaders_index
from analytics.normalizer import normalize_by_season
from config import BADGE_LEADERS_PATH, TEAMS_PARQUET_PATH, TEAM_METADATA_PATH
from parquet_io import read_teams_parquet

logger = logging.getLogger(__name__)


class BadgeLeadersPipeline:
    def __init__(
        self,
        parquet_path: Path | str = TEAMS_PARQUET_PATH,
        metadata_path: Path | str = TEAM_METADATA_PATH,
        output_path: Path | str = BADGE_LEADERS_PATH,
    ):
        self.parquet_path = Path(parquet_path)
        self.metadata_path = Path(metadata_path)
        self.output_path = Path(output_path)

    def run(self, df: pd.DataFrame | None = None) -> dict:
        logger.info("Building badge leaders...")
        if df is None:
            if not self.parquet_path.exists():
                raise FileNotFoundError(f"Parquet file not found: {self.parquet_path}")
            df = read_teams_parquet(self.parquet_path)

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        with open(self.metadata_path) as f:
            metadata = json.load(f)

        norm = normalize_by_season(df)
        badge_leaders = build_badge_leaders_index(df, norm, metadata)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(badge_leaders, f)
        logger.info("Saved badge leaders to %s", self.output_path)

        n_seasons = len(badge_leaders)
        print(f"✓ Badge leaders completed: {n_seasons}/{n_seasons} seasons processed.")

        return badge_leaders

