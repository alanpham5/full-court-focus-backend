from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from analytics.lineups_build import build_starting_lineups_index
from config import STARTING_LINEUPS_PATH, TEAMS_PARQUET_PATH
from parquet_io import read_teams_parquet

logger = logging.getLogger(__name__)


class LineupsPipeline:
    def __init__(
        self,
        parquet_path: Path | str = TEAMS_PARQUET_PATH,
        output_path: Path | str = STARTING_LINEUPS_PATH,
        rate_limit_sleep: float = 1.5,
    ):
        self.parquet_path = Path(parquet_path)
        self.output_path = Path(output_path)
        self.rate_limit_sleep = rate_limit_sleep

    def _season_start(self, season: str) -> int:
        try:
            return int(str(season).split("-", 1)[0])
        except (TypeError, ValueError):
            return -1

    def get_known_keys(self) -> list[str]:
        if not self.parquet_path.exists():
            return []
        df = read_teams_parquet(self.parquet_path)
        required = {"TEAM_ID", "SEASON"}
        missing = required - set(df.columns)
        if missing:
            raise RuntimeError(
                f"{self.parquet_path} is missing required column(s): "
                f"{', '.join(sorted(missing))}"
            )

        keys = {
            f"{int(row.TEAM_ID)}:{row.SEASON}"
            for _, row in df.drop_duplicates(["TEAM_ID", "SEASON"]).iterrows()
        }
        return sorted(
            keys,
            key=lambda k: (
                self._season_start(k.split(":", 1)[1]),
                int(k.split(":", 1)[0]),
            ),
        )

    def get_current_season_keys(self) -> list[str]:
        keys = self.get_known_keys()
        if not keys:
            return []
        latest = max((key.split(":", 1)[1] for key in keys), key=self._season_start)
        return [key for key in keys if key.endswith(f":{latest}")]

    def load_existing(self) -> dict[str, Any]:
        if not self.output_path.exists():
            return {}
        with self.output_path.open() as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise RuntimeError(f"{self.output_path} must contain a JSON object")
        return data

    def run(
        self,
        keys: list[str],
        *,
        replace: bool = False,
        progress_every: int = 25,
        dry_run: bool = False,
        timeout: int = 15,
        retries: int = 2,
    ) -> dict[str, Any]:
        existing = {} if replace else self.load_existing()

        seen = set()
        deduped_keys = []
        for key in keys:
            if key in seen:
                continue
            seen.add(key)
            deduped_keys.append(key)

        logger.info("Selected %s team-season(s) for lineup scrape.", len(deduped_keys))
        if dry_run:
            logger.info("Dry run: Lineups for %s would be scraped.", len(deduped_keys))
            return existing

        if not deduped_keys:
            n_merged = len(existing)
            print(f"✓ Lineups completed: {n_merged}/{n_merged} teams lineups processed.")
            return existing

        fresh = build_starting_lineups_index(
            deduped_keys,
            rate_limit_sleep=self.rate_limit_sleep,
            progress_every=progress_every,
            timeout=timeout,
            retries=retries,
        )

        merged = dict(existing)
        merged.update(fresh)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w") as f:
            json.dump(merged, f)
        logger.info("Saved %s starting lineups to %s", len(merged), self.output_path)

        success_count = len(merged)
        print(f"✓ Lineups completed: {success_count}/{success_count} teams lineups processed.")

        return merged

