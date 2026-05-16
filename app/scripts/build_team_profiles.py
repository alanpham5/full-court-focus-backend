from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_APP_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_ROOT = Path(__file__).resolve().parent
for p in (_APP_ROOT, _SCRIPTS_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from analytics.normalizer import normalize_by_season  # noqa: E402
from analytics.similarity import build_similar_teams_index  # noqa: E402
from analytics.team_profiles_build import build_team_profiles_json  # noqa: E402
from analytics.team_static_cache import merge_similar_teams_with_abbreviations  # noqa: E402
from parquet_io import read_teams_parquet  # noqa: E402

STATIC_DIR = _APP_ROOT / "data" / "static"
PARQUET_DEFAULT = STATIC_DIR / "teams_historical.parquet"
OUT_DEFAULT = STATIC_DIR / "team_profiles.json"
METADATA_DEFAULT = STATIC_DIR / "team_metadata.json"


def load_team_metadata(metadata_path: Path) -> dict:
    if metadata_path.exists():
        with metadata_path.open() as f:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild team_profiles.json from teams_historical.parquet (refetches stat leaders via nba_api).",
    )
    parser.add_argument("--parquet", type=Path, default=PARQUET_DEFAULT)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--metadata", type=Path, default=METADATA_DEFAULT)
    parser.add_argument(
        "--reuse-similar",
        type=Path,
        default=None,
        help="Path to similar_teams.json; if omitted, rebuilt from parquet via normalize_by_season",
    )
    parser.add_argument("--rate-limit", type=float, default=1.5, dest="rate_limit_sleep")
    parser.add_argument("--progress-every", type=int, default=100, dest="progress_every")
    args = parser.parse_args()

    if not args.parquet.exists():
        raise SystemExit(f"Parquet not found: {args.parquet}")

    df = read_teams_parquet(args.parquet)
    norm = normalize_by_season(df)
    meta = load_team_metadata(args.metadata)

    if args.reuse_similar is not None and args.reuse_similar.exists():
        with args.reuse_similar.open() as f:
            sim_index = json.load(f)
        print(f"Loaded similar-teams index from {args.reuse_similar}")
    else:
        print("Building similar-teams index (k=6) from parquet…")
        sim_index = build_similar_teams_index(norm, k=6)

    similar_display = merge_similar_teams_with_abbreviations(sim_index, meta)
    profiles = build_team_profiles_json(
        df,
        norm,
        similar_display,
        rate_limit_sleep=args.rate_limit_sleep,
        progress_every=args.progress_every,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(profiles, f)
    print(f"Wrote {len(profiles)} team-season profiles → {args.out}")


if __name__ == "__main__":
    main()
