from __future__ import annotations

import time
from typing import Any

from analytics.lineups import build_starting_lineup


def build_starting_lineups_index(
    keys: list[str],
    *,
    rate_limit_sleep: float = 1.5,
    progress_every: int = 50,
) -> dict[str, dict[str, Any]]:
    """Build lineup payloads for ``team_id:season`` keys."""
    out: dict[str, dict[str, Any]] = {}
    n = len(keys)
    for i, sk in enumerate(keys, start=1):
        tid_s, season = sk.split(":", 1)
        payload = build_starting_lineup(int(tid_s), season)
        if payload is not None:
            out[sk] = payload
        else:
            print(f"  [WARN] lineup unavailable for {sk}")
        if progress_every and i % progress_every == 0:
            print(f"    starting_lineups {i}/{n}...")
        time.sleep(rate_limit_sleep)
    return out
