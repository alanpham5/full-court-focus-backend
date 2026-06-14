from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from analytics.normalizer import normalize_by_season
from analytics.team_profiles_build import build_team_profiles_json
from analytics.team_static_cache import (
    build_team_profile_static_cache,
    merge_similar_teams_with_abbreviations,
)
from config import TEAM_PROFILES_PATH, TEAMS_PARQUET_PATH, TEAM_METADATA_PATH, SIMILAR_TEAMS_PATH
from parquet_io import read_teams_parquet

logger = logging.getLogger(__name__)


class TeamProfilesPipeline:
    def __init__(
        self,
        parquet_path: Path | str = TEAMS_PARQUET_PATH,
        metadata_path: Path | str = TEAM_METADATA_PATH,
        similar_teams_path: Path | str = SIMILAR_TEAMS_PATH,
        output_path: Path | str = TEAM_PROFILES_PATH,
        rate_limit_sleep: float = 1.5,
    ):
        self.parquet_path = Path(parquet_path)
        self.metadata_path = Path(metadata_path)
        self.similar_teams_path = Path(similar_teams_path)
        self.output_path = Path(output_path)
        self.rate_limit_sleep = rate_limit_sleep

    def load_team_metadata(self) -> dict:
        if self.metadata_path.exists():
            with self.metadata_path.open() as f:
                return json.load(f)
        from nba_api.stats.static import teams as nba_teams

        all_teams = nba_teams.get_teams()
        return {
            str(t["id"]): {
                "id": t["id"],
                "name": t["full_name"],
                "abbreviation": t["abbreviation"],
                "nickname": t["nickname"],
            }
            for t in all_teams
        }

    def run(
        self,
        df: pd.DataFrame | None = None,
        *,
        profile_keys: list[str] | None = None,
        sim_index: dict | None = None,
        progress_every: int = 100,
    ) -> dict[str, Any]:
        logger.info("Starting team profiles pipeline...")
        if df is None:
            if not self.parquet_path.exists():
                raise FileNotFoundError(f"Parquet file not found: {self.parquet_path}")
            df = read_teams_parquet(self.parquet_path)

        norm = normalize_by_season(df)
        meta = self.load_team_metadata()

        if sim_index is None:
            if not self.similar_teams_path.exists():
                raise FileNotFoundError(f"Similar teams index not found: {self.similar_teams_path}")
            with self.similar_teams_path.open() as f:
                sim_index = json.load(f)

        similar_display = merge_similar_teams_with_abbreviations(sim_index, meta)

        if profile_keys is None or not self.output_path.exists():
            logger.info("Full build of team profiles (this requests leaders sequentially via nba_api)...")
            profiles = build_team_profiles_json(
                df,
                norm,
                similar_display,
                rate_limit_sleep=self.rate_limit_sleep,
                progress_every=progress_every,
            )
        else:
            wanted = set(profile_keys)
            logger.info("Incremental update of team profiles for %s keys...", len(wanted))
            with self.output_path.open() as f:
                existing = json.load(f)

            static_profiles = build_team_profile_static_cache(df, norm, similar_display)

            profile_df = df[
                df.apply(lambda r: f"{int(r.TEAM_ID)}:{r.SEASON}" in wanted, axis=1)
            ].copy()
            profile_norm = norm[
                norm.apply(lambda r: f"{int(r.TEAM_ID)}:{r.SEASON}" in wanted, axis=1)
            ].copy()

            fresh = build_team_profiles_json(
                profile_df,
                profile_norm,
                similar_display,
                rate_limit_sleep=self.rate_limit_sleep,
                progress_every=progress_every,
            )

            profiles = {}
            for sk, static_payload in static_profiles.items():
                existing_payload = existing.get(sk, {})
                leaders = existing_payload.get("leaders")
                if leaders:
                    profiles[sk] = {**static_payload, "leaders": leaders}
            profiles.update(fresh)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(profiles, f)
        logger.info("Saved team profiles to %s", self.output_path)

        n_profiles = len(profiles)
        print(f"✓ Team profiles completed: {n_profiles}/{n_profiles} team profiles processed.")

        return profiles
