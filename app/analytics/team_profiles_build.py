from __future__ import annotations

import time
from typing import Any

import pandas as pd

from analytics.leaders import get_stat_leaders
from analytics.team_static_cache import build_team_profile_static_cache

ProfileItem = tuple[str, dict[str, Any]]


def _leaders_payload(sk: str, block: dict[str, Any], leaders: dict[str, Any]) -> dict[str, Any]:
    tid_s, season = sk.split(":", 1)
    tid = int(tid_s)
    sim = block["similar_teams"]
    if sim is None:
        sim = []
    return {
        "team_id": tid,
        "team_name": block["team_name"],
        "season": season,
        "record": block["record"],
        "win_pct": block["win_pct"],
        "style_vector": block["style_vector"],
        "badges": block["badges"],
        "leaders": leaders,
        "similar_teams": sim,
    }


def _try_leaders_payload(sk: str, block: dict[str, Any]) -> dict[str, Any] | None:
    tid_s, season = sk.split(":", 1)
    leaders = get_stat_leaders(int(tid_s), season)
    if not leaders:
        return None
    return _leaders_payload(sk, block, leaders)


def _pass_sequential(
    items: list[ProfileItem],
    *,
    rate_limit_sleep: float,
    progress_every: int,
    pass_label: str,
) -> tuple[dict[str, dict[str, Any]], list[ProfileItem]]:
    ok: dict[str, dict[str, Any]] = {}
    failed: list[ProfileItem] = []
    for i, (sk, block) in enumerate(items, start=1):
        payload = _try_leaders_payload(sk, block)
        if payload is not None:
            ok[sk] = payload
        else:
            print(
                f"  [WARN] leaders fetch empty for {sk} "
                f"({pass_label} {i}/{len(items)})"
            )
            failed.append((sk, block))
        if progress_every and i % progress_every == 0:
            print(f"    team_profiles {pass_label} {i}/{len(items)}...")
        time.sleep(rate_limit_sleep)
    return ok, failed


def build_team_profiles_json(
    df: pd.DataFrame,
    norm_df: pd.DataFrame,
    similar_with_abbr: dict[str, list[dict]],
    *,
    rate_limit_sleep: float = 1.5,
    progress_every: int = 100,
) -> dict[str, dict[str, Any]]:
    base = build_team_profile_static_cache(df, norm_df, similar_with_abbr)
    items: list[ProfileItem] = list(base.items())
    n = len(items)
    out: dict[str, dict[str, Any]] = {}

    if n == 0:
        return out

    ok1, failed1 = _pass_sequential(
        items,
        rate_limit_sleep=rate_limit_sleep,
        progress_every=progress_every,
        pass_label="pass1",
    )
    out.update(ok1)

    if not failed1:
        return out

    print(
        f"  … leaders retry queue: {len(failed1)} team-season(s) will be retried "
        f"(sequential, {rate_limit_sleep}s between calls)"
    )
    ok2, failed2 = _pass_sequential(
        failed1,
        rate_limit_sleep=rate_limit_sleep,
        progress_every=progress_every,
        pass_label="retry",
    )
    out.update(ok2)

    for sk, _block in failed2:
        print(
            f"  [WARN] leaders still unavailable for {sk} after retry — "
            "omitting from team_profiles"
        )

    return out
