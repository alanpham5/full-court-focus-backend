from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from analytics.player_profiles.pipeline import (
    PlayerProfilePipeline,
    PlayerProfilePipelineConfig,
    copy_processed_outputs_to_static,
)
from analytics.player_profiles.storage import ProfileStorage, StorageConfig
from analytics.player_profiles.nba_client import NbaStatsClient

logger = logging.getLogger(__name__)


class PlayerProfilesPipeline:
    def __init__(
        self,
        storage_uri: str,
        gcs_project: str | None = None,
        timeout: int = 60,
        retries: int = 4,
        sleep: float = 2.0,
    ):
        self.storage_uri = storage_uri
        self.gcs_project = gcs_project
        self.timeout = timeout
        self.retries = retries
        self.sleep = sleep

    def run(
        self,
        seasons: list[str],
        current_season_start_year: int,
        *,
        refresh_raw: bool = False,
        copy_to_static_dir: Path | str | None = None,
    ) -> dict[str, Any]:
        logger.info("Starting player profiles pipeline wrapper...")
        config = PlayerProfilePipelineConfig(
            seasons=seasons,
            current_season_start_year=current_season_start_year,
            refresh_raw=refresh_raw,
        )
        pipeline = PlayerProfilePipeline(
            storage=ProfileStorage(
                StorageConfig.from_uri(self.storage_uri, gcs_project=self.gcs_project)
            ),
            client=NbaStatsClient(timeout=self.timeout, retries=self.retries, base_sleep=self.sleep),
            config=config,
        )
        summary = pipeline.run()

        if copy_to_static_dir is not None:
            logger.info("Copying player profile outputs to static: %s", copy_to_static_dir)
            copy_processed_outputs_to_static(
                self.storage_uri,
                Path(copy_to_static_dir),
                gcs_project=self.gcs_project,
            )

        n_players = summary.get("players", 0)
        print(f"✓ Player profiles completed: {n_players}/{n_players} players processed.")

        return summary

