from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

_APP_ROOT = Path(__file__).resolve().parents[1]
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

from analytics.lineups_build import build_starting_lineups_index
from config import STARTING_LINEUPS_PATH, TEAMS_PARQUET_PATH
from parquet_io import read_teams_parquet

_KEY_RE = re.compile(r"^(?P<team_id>\d+):(?P<season>\d{4}-\d{2})$")


def _normalize_key(value: str) -> str:
    match = _KEY_RE.match(value.strip())
    if not match:
        raise argparse.ArgumentTypeError(
            f"{value!r} must look like TEAM_ID:SEASON, e.g. 1610612743:2023-24"
        )
    return f"{int(match.group('team_id'))}:{match.group('season')}"


def _season_start(season: str) -> int:
    try:
        return int(str(season).split("-", 1)[0])
    except (TypeError, ValueError):
        return -1


def _known_keys() -> list[str]:
    df = read_teams_parquet(TEAMS_PARQUET_PATH)
    required = {"TEAM_ID", "SEASON"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"{TEAMS_PARQUET_PATH} is missing required column(s): "
            f"{', '.join(sorted(missing))}"
        )

    keys = {
        f"{int(row.TEAM_ID)}:{row.SEASON}"
        for _, row in df.drop_duplicates(["TEAM_ID", "SEASON"]).iterrows()
    }
    return sorted(
        keys,
        key=lambda k: (
            _season_start(k.split(":", 1)[1]),
            int(k.split(":", 1)[0]),
        ),
    )


def _current_season_keys() -> list[str]:
    keys = _known_keys()
    if not keys:
        return []
    latest = max((key.split(":", 1)[1] for key in keys), key=_season_start)
    return [key for key in keys if key.endswith(f":{latest}")]


def _load_existing(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return data


def _dedupe(keys: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _selected_keys(args: argparse.Namespace, existing: dict[str, Any]) -> list[str]:
    keys = list(args.keys)

    if args.team_id is not None or args.season is not None:
        if args.team_id is None or args.season is None:
            raise SystemExit("--team-id and --season must be used together")
        keys.append(_normalize_key(f"{args.team_id}:{args.season}"))

    if args.all:
        keys.extend(_known_keys())

    if args.current:
        keys.extend(_current_season_keys())

    if args.missing:
        keys.extend([key for key in _known_keys() if key not in existing])

    if not keys:
        raise SystemExit(
            "Choose keys, --team-id/--season, --current, --missing, or --all."
        )

    return _dedupe(keys)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape starting lineups only and write starting_lineups.json. "
            "Other static data files are left untouched."
        )
    )
    parser.add_argument(
        "keys",
        nargs="*",
        type=_normalize_key,
        help="Team-season keys like 1610612743:2023-24",
    )
    parser.add_argument("--team-id", type=int, help="Single NBA team id to scrape")
    parser.add_argument("--season", help="Season for --team-id, e.g. 2023-24")
    parser.add_argument(
        "--current",
        action="store_true",
        help="Scrape every team in the latest season found in teams_historical.parquet",
    )
    parser.add_argument(
        "--missing",
        action="store_true",
        help="Scrape known team-seasons absent from the existing output file",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Scrape every team-season in teams_historical.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=STARTING_LINEUPS_PATH,
        help=f"Output JSON path (default: {STARTING_LINEUPS_PATH})",
    )
    parser.add_argument(
        "--rate-limit-sleep",
        type=float,
        default=1.5,
        help="Seconds to sleep between NBA API calls",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Write only freshly scraped lineups instead of merging into existing JSON",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected keys without calling the NBA API or writing JSON",
    )
    args = parser.parse_args()

    output = args.output
    existing = {} if args.replace else _load_existing(output)
    keys = _selected_keys(args, existing)

    print(f"Selected {len(keys)} team-season(s).")
    if args.dry_run:
        for key in keys:
            print(key)
        return 0

    fresh = build_starting_lineups_index(
        keys,
        rate_limit_sleep=args.rate_limit_sleep,
        progress_every=25,
    )
    merged = dict(existing)
    merged.update(fresh)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(merged, f)
    print(f"Saved {len(fresh)} refreshed lineup(s), {len(merged)} total -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
