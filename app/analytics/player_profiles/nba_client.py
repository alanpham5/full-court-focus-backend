from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd
from nba_api.stats.endpoints import leagueleaders, playerindex


logger = logging.getLogger(__name__)

NBA_HEADERS = {
    "Host": "stats.nba.com",
    "Connection": "keep-alive",
    "Accept": "application/json, text/plain, */*",
    "x-nba-stats-token": "true",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
    ),
    "x-nba-stats-origin": "stats",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
}


@dataclass(frozen=True)
class EndpointResult:
    name: str
    params: dict[str, Any]
    raw: dict[str, Any]
    frames: dict[str, pd.DataFrame]


class NbaStatsClient:
    def __init__(
        self,
        *,
        timeout: int = 60,
        retries: int = 4,
        base_sleep: float = 1.2,
        jitter: float = 0.35,
    ):
        self.timeout = timeout
        self.retries = retries
        self.base_sleep = base_sleep
        self.jitter = jitter

    def _call(
        self,
        name: str,
        factory: Callable[[], Any],
        params: dict[str, Any],
        *,
        retries: int | None = None,
    ) -> EndpointResult:
        last_err: Exception | None = None
        max_retries = retries if retries is not None else self.retries
        for attempt in range(1, max_retries + 1):
            try:
                endpoint = factory()
                raw = endpoint.get_dict()
                frames = endpoint.get_data_frames()
                frame_map = {}
                for i, frame in enumerate(frames):
                    if not isinstance(frame, pd.DataFrame):
                        continue
                    frame = frame.copy()
                    if "per_mode" in params:
                        frame["NBA_API_PER_MODE"] = params["per_mode"]
                    if "measure_type" in params:
                        frame["NBA_API_MEASURE_TYPE"] = params["measure_type"]
                    frame_map[f"{name}_{i}"] = frame
                time.sleep(self.base_sleep + random.random() * self.jitter)
                return EndpointResult(name=name, params=params, raw=raw, frames=frame_map)
            except Exception as exc:
                last_err = exc
                sleep = self.base_sleep * (2 ** (attempt - 1)) + random.random() * self.jitter
                logger.warning("%s failed on attempt %s/%s: %s", name, attempt, max_retries, exc)
                time.sleep(sleep)
        raise RuntimeError(f"{name} failed after {max_retries} attempts") from last_err

    def league_leaders_totals(self, season: str, stat_category: str = "PTS") -> EndpointResult:
        params = {"season": season, "stat_category": stat_category, "per_mode": "Totals"}
        return self._call(
            "league_leaders_totals",
            lambda: leagueleaders.LeagueLeaders(
                season=season,
                league_id="00",
                season_type_all_star="Regular Season",
                per_mode48="Totals",
                scope="S",
                stat_category_abbreviation=stat_category,
                timeout=self.timeout,
                headers=NBA_HEADERS,
            ),
            params,
        )

    def player_index(self, season: str) -> EndpointResult:
        params = {"season": season}
        return self._call(
            "player_index",
            lambda: playerindex.PlayerIndex(
                season=season,
                league_id="00",
                timeout=self.timeout,
            ),
            params,
        )

