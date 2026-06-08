from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from analytics.normalizer import normalize_by_season
from analytics.similarity import build_similar_teams_index
from config import SIMILAR_TEAMS_PATH, TEAMS_PARQUET_PATH
from parquet_io import read_teams_parquet

logger = logging.getLogger(__name__)


class SimilarTeamsPipeline:
    def __init__(
        self,
        parquet_path: Path | str = TEAMS_PARQUET_PATH,
        output_path: Path | str = SIMILAR_TEAMS_PATH,
        k: int = 6,
    ):
        self.parquet_path = Path(parquet_path)
        self.output_path = Path(output_path)
        self.k = k

    def run(self, df: pd.DataFrame | None = None) -> dict:
        logger.info("Building similar-teams index (k=%s)...", self.k)
        if df is None:
            if not self.parquet_path.exists():
                raise FileNotFoundError(f"Parquet file not found: {self.parquet_path}")
            df = read_teams_parquet(self.parquet_path)

        norm = normalize_by_season(df)
        sim_index = build_similar_teams_index(norm, k=self.k)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(sim_index, f)
        logger.info("Saved similar teams index to %s", self.output_path)

        n_teams = len(sim_index)
        print(f"✓ Similar teams completed: {n_teams}/{n_teams} teams processed.")

        return sim_index

