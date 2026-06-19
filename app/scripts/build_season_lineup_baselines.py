import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env_defaults import load_env_defaults

load_env_defaults()

from analytics.season_baselines import build_all_season_baselines
from parquet_io import read_teams_parquet

STATIC = os.path.join(os.path.dirname(__file__), "..", "data", "static")
LOCAL_PATH = os.path.join(STATIC, "season_lineup_baselines.json")
STORAGE_KEY = "static/season_lineup_baselines.json"


def _load_inputs():
    sl = json.load(open(os.path.join(STATIC, "starting_lineups.json")))
    tm = json.load(open(os.path.join(STATIC, "team_metadata.json")))
    psf = read_teams_parquet(os.path.join(STATIC, "player_season_features.parquet"))
    teams_df = read_teams_parquet(os.path.join(STATIC, "teams_historical.parquet"))
    return sl, tm, psf, teams_df


def _all_seasons(starting_lineups):
    return sorted({k.split(":")[1] for k in starting_lineups.keys() if ":" in k})


def _write_local(payload):
    os.makedirs(STATIC, exist_ok=True)
    with open(LOCAL_PATH, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"  ✓ wrote {len(payload)} seasons -> {LOCAL_PATH}")


def _write_gcs(payload):
    uri = os.getenv("PLAYER_PROFILES_STORAGE_URI")
    if not uri:
        print("  [skip] PLAYER_PROFILES_STORAGE_URI not set — GCS upload skipped")
        return
    try:
        from analytics.player_profiles.storage import ProfileStorage, StorageConfig
        storage = ProfileStorage(StorageConfig.from_uri(uri))
        storage.write_json(STORAGE_KEY, payload)
        print(f"  ✓ wrote {len(payload)} seasons -> {uri}/{STORAGE_KEY}")
    except Exception as exc:
        print(f"  [WARN] GCS upload failed: {exc}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="rebuild every season")
    g.add_argument("--latest", action="store_true", help="update only the most recent season")
    g.add_argument("--season", help="update a single season, e.g. 2025-26")
    args = parser.parse_args()

    sl, tm, psf, teams_df = _load_inputs()
    all_seasons = _all_seasons(sl)

    if args.all:
        seasons = all_seasons
        existing = {}
    else:
        target = all_seasons[-1] if args.latest else args.season
        if target not in all_seasons:
            print(f"  [ERROR] season {target!r} not found in starting lineups", file=sys.stderr)
            sys.exit(1)
        seasons = [target]
        existing = json.load(open(LOCAL_PATH)) if os.path.exists(LOCAL_PATH) else {}

    t0 = time.time()
    print(f"Building season lineup baselines for {len(seasons)} season(s)...")
    built = build_all_season_baselines(psf, teams_df, sl, tm, seasons=seasons)

    payload = {**existing, **built}
    payload = {k: payload[k] for k in sorted(payload.keys())}

    _write_local(payload)
    _write_gcs(payload)
    print(f"Done in {time.time() - t0:.1f}s "
          f"(updated: {', '.join(seasons) if len(seasons) <= 5 else f'{len(seasons)} seasons'}).")


if __name__ == "__main__":
    main()
