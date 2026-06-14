from __future__ import annotations

import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd
from nba_api.stats.endpoints import LeagueDashTeamStats
from nba_api.stats.static import teams as nba_teams

from config import TEAMS_PARQUET_PATH, TEAM_METADATA_PATH, SEASON_INDEX_PATH
from parquet_io import write_teams_parquet, read_teams_parquet

logger = logging.getLogger(__name__)

FIRST_SEASON_YEAR = 1996

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


class TeamStatsPipeline:
    def __init__(
        self,
        parquet_path: Path | str = TEAMS_PARQUET_PATH,
        metadata_path: Path | str = TEAM_METADATA_PATH,
        season_index_path: Path | str = SEASON_INDEX_PATH,
        rate_limit_sleep: float = 1.5,
    ):
        self.parquet_path = Path(parquet_path)
        self.metadata_path = Path(metadata_path)
        self.season_index_path = Path(season_index_path)
        self.rate_limit_sleep = rate_limit_sleep

    def _fetch_with_retry(
        self,
        name: str,
        factory: Callable[[], Any],
        season: str,
        retries: int = 4,
    ) -> pd.DataFrame | None:
        last_err = None
        for attempt in range(1, retries + 1):
            try:
                endpoint = factory()
                df = endpoint.get_data_frames()[0]
                df["SEASON"] = season
                return df
            except Exception as exc:
                last_err = exc
                sleep = 2.0 * (2 ** (attempt - 1)) + random.random() * 0.5
                logger.warning(
                    "%s fetch failed for %s on attempt %s/%s: %s. Retrying in %.2fs...",
                    name,
                    season,
                    attempt,
                    retries,
                    exc,
                    sleep,
                )
                time.sleep(sleep)
        logger.error("%s fetch completely failed for %s after %s attempts: %s", name, season, retries, last_err)
        return None

    def _fetch_advanced_raw(self, season: str) -> pd.DataFrame | None:
        def fetch():
            return LeagueDashTeamStats(
                season=season,
                measure_type_detailed_defense="Advanced",
                per_mode_detailed="PerGame",
                timeout=30,
            )
        return self._fetch_with_retry("Advanced", fetch, season)

    def _fetch_shooting_raw(self, season: str) -> pd.DataFrame | None:
        def fetch():
            return LeagueDashTeamStats(
                season=season,
                measure_type_detailed_defense="Base",
                per_mode_detailed="Per100Possessions",
                timeout=30,
            )
        return self._fetch_with_retry("Base/Shooting", fetch, season)

    def _fetch_misc_raw(self, season: str) -> pd.DataFrame | None:
        def fetch():
            return LeagueDashTeamStats(
                season=season,
                measure_type_detailed_defense="Misc",
                per_mode_detailed="Per100Possessions",
                timeout=30,
            )
        return self._fetch_with_retry("Misc", fetch, season)

    def merge_season(self, season: str) -> pd.DataFrame | None:
        adv = self._fetch_advanced_raw(season)
        if adv is None:
            return None
        time.sleep(self.rate_limit_sleep)
        sht = self._fetch_shooting_raw(season)
        if sht is None:
            return None
        time.sleep(self.rate_limit_sleep)

        adv_keep = [c for c in ADVANCED_COLS if c in adv.columns]
        sht_keep = [c for c in SHOOTING_COLS if c in sht.columns]

        merged = adv[adv_keep].merge(sht[sht_keep], on=["TEAM_ID", "SEASON"], how="left")

        misc = self._fetch_misc_raw(season)
        if misc is not None:
            time.sleep(self.rate_limit_sleep)
            misc_keep = [c for c in MISC_COLS if c in misc.columns]
            if "PTS_PAINT" not in misc_keep:
                logger.warning(
                    "Misc response missing PTS_PAINT for %s (paint percentile will default until fixed)",
                    season,
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
            logger.warning(
                "Misc fetch failed for %s (no PTS_PAINT / PAINT_FGA — paint percentile will default)",
                season,
            )

        if "FTA" in merged.columns and "FGA" in merged.columns:
            fga = pd.to_numeric(merged["FGA"], errors="coerce")
            fta = pd.to_numeric(merged["FTA"], errors="coerce")
            merged["FTA_RATE"] = fta / fga.replace(0, pd.NA)
        return merged

    def build_team_metadata(self) -> dict:
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

    def save_metadata_and_indices(self, df: pd.DataFrame) -> None:
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.season_index_path.parent.mkdir(parents=True, exist_ok=True)

        index = df.groupby("TEAM_ID")["SEASON"].apply(list).to_dict()
        index = {str(k): v for k, v in index.items()}
        with open(self.season_index_path, "w") as f:
            json.dump(index, f)
        logger.info("Saved season index to %s", self.season_index_path)

        metadata = self.build_team_metadata()
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Saved team metadata to %s", self.metadata_path)

    def run(self, *, current_season_only: bool = False) -> pd.DataFrame:
        if current_season_only and self.parquet_path.exists():
            season = season_str(current_season_year())
            logger.info("Incremental team stats scrape for season: %s", season)
            existing = read_teams_parquet(self.parquet_path)
            existing = existing[existing["SEASON"] != season]

            new_data = self.merge_season(season)
            if new_data is None:
                logger.warning("Incremental fetch failed; keeping existing data")
                print("✓ Team stats completed: 0/1 seasons scraped.")
                return read_teams_parquet(self.parquet_path)

            combined = pd.concat([existing, new_data], ignore_index=True)
            print("✓ Team stats completed: 1/1 seasons scraped.")
        else:
            logger.info("Full team stats scrape starting...")
            rows = []
            years = all_season_years()
            from tqdm import tqdm
            for year in tqdm(years, desc="Scraping team stats", unit="season"):
                season = season_str(year)
                df = self.merge_season(season)
                if df is not None:
                    rows.append(df)
                else:
                    logger.warning("Skipping %s (fetch failed)", season)

            if not rows:
                raise RuntimeError("No data fetched — check nba_api connectivity")
            combined = pd.concat(rows, ignore_index=True)
            print(f"✓ Team stats completed: {len(rows)}/{len(years)} seasons scraped.")

        write_teams_parquet(combined, self.parquet_path)
        logger.info("Saved %s team-seasons to %s", len(combined), self.parquet_path)

        self.save_metadata_and_indices(combined)

        return combined

