from __future__ import annotations

import time
from typing import Any

from analytics.lineups import build_starting_lineup


def build_starting_lineups_index(
    keys: list[str],
    *,
    rate_limit_sleep: float = 1.5,
    progress_every: int = 50,
    timeout: int = 15,
    retries: int = 2,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    from tqdm import tqdm
    for sk in tqdm(keys, desc="Scraping starting lineups", unit="team"):
        tid_s, season = sk.split(":", 1)
        payload = build_starting_lineup(int(tid_s), season, timeout=timeout, retries=retries)
        if payload is not None:
            out[sk] = payload
        time.sleep(rate_limit_sleep)
    return out

