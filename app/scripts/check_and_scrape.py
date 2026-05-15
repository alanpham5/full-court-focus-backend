"""Run incremental scrape only when data is stale vs the NBA season schedule."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_APP_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _APP_ROOT / "scripts"
for p in (_APP_ROOT, _SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from analytics.scrape_schedule import is_stale, latest_scheduled_run
from scrape_all_seasons import incremental_scrape

STATE_PATH = _APP_ROOT / "data" / "static" / "scrape_state.json"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def load_or_init_state() -> datetime:
    if STATE_PATH.exists():
        with STATE_PATH.open() as f:
            raw = json.load(f)
        return datetime.fromisoformat(raw["last_updated"]).astimezone(timezone.utc)

    now = _now_utc()
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_state(now)
    print(f"Initialized {STATE_PATH.name} with last_updated={now.isoformat()}")
    return now


def save_state(when: datetime) -> None:
    when = when.astimezone(timezone.utc)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATE_PATH.open("w") as f:
        json.dump({"last_updated": when.isoformat()}, f, indent=2)
        f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape current season if schedule-stale")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run scrape even if last_updated is current",
    )
    args = parser.parse_args()

    last_updated = load_or_init_state()
    now = _now_utc()
    due = latest_scheduled_run(now)

    if not args.force and not is_stale(last_updated, now):
        print(
            f"Data is up to date. last_updated={last_updated.isoformat()} "
            f"latest_due_run={due.isoformat() if due else 'n/a'}"
        )
        return 0

    if args.force:
        print("Forced scrape (--force).")
    else:
        print(
            f"Data is stale. last_updated={last_updated.isoformat()} "
            f"latest_due_run={due.isoformat() if due else 'n/a'}"
        )

    if not incremental_scrape():
        print("Scrape did not complete successfully; last_updated unchanged.")
        return 1

    save_state(_now_utc())
    print(f"Updated {STATE_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
