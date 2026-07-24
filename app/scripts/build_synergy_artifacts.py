import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env_defaults import load_env_defaults

load_env_defaults()

from analytics.season_baselines import build_all_season_baselines
from analytics.lineup_synergy import calculate_lineup_synergy
from analytics.lineup_model import DEFAULT_MODEL_PATH, save_lineup_model, train_lineup_model
from scripts.train_lineup_synergy_model import build_training_rows, load_player_seasons
from parquet_io import read_teams_parquet

STATIC = os.path.join(os.path.dirname(__file__), "..", "data", "static")
BASELINES_PATH = os.path.join(STATIC, "season_lineup_baselines.json")
SCORES_PATH = os.path.join(STATIC, "lineup_synergy_scores.json")
MODEL_PATH = str(DEFAULT_MODEL_PATH)
BASELINES_STORAGE_KEY = "static/season_lineup_baselines.json"
SCORES_STORAGE_KEY = "static/lineup_synergy_scores.json"
MODEL_STORAGE_KEY = "static/lineup_synergy_model.json"


def _load_inputs():
    tp = json.load(open(os.path.join(STATIC, "team_profiles.json")))
    sl = json.load(open(os.path.join(STATIC, "starting_lineups.json")))
    tm = json.load(open(os.path.join(STATIC, "team_metadata.json")))
    pp = json.load(open(os.path.join(STATIC, "player_profiles.json")))
    psf = load_player_seasons()
    teams_df = read_teams_parquet(os.path.join(STATIC, "teams_historical.parquet"))
    return tp, sl, tm, pp, psf, teams_df


def _all_seasons(starting_lineups):
    return sorted({k.split(":")[1] for k in starting_lineups.keys() if ":" in k})


def _upload_json_gcs(key, payload):
    uri = os.getenv("PLAYER_PROFILES_STORAGE_URI")
    if not uri:
        print("  [skip] PLAYER_PROFILES_STORAGE_URI not set — GCS upload skipped")
        return
    try:
        from analytics.player_profiles.storage import ProfileStorage, StorageConfig
        storage = ProfileStorage(StorageConfig.from_uri(uri))
        storage.write_json(key, payload)
        print(f"  ✓ uploaded -> {uri}/{key}")
    except Exception as exc:
        print(f"  [WARN] GCS upload failed: {exc}", file=sys.stderr)


def build_baselines(sl, tm, psf, teams_df, seasons):
    existing = {}
    is_full = set(seasons) == set(_all_seasons(sl))
    if not is_full and os.path.exists(BASELINES_PATH):
        existing = json.load(open(BASELINES_PATH))
    built = build_all_season_baselines(psf, teams_df, sl, tm, seasons=seasons)
    payload = {**existing, **built}
    payload = {k: payload[k] for k in sorted(payload.keys())}
    os.makedirs(STATIC, exist_ok=True)
    with open(BASELINES_PATH, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"  ✓ wrote {len(payload)} seasons -> {BASELINES_PATH}")
    _upload_json_gcs(BASELINES_STORAGE_KEY, payload)
    return payload


def build_model(tp, sl, tm, psf, teams_df, baselines):
    features, targets, seasons = build_training_rows(
        sl, tp, tm, psf, teams_df, baselines
    )
    artifact = train_lineup_model(features, targets, seasons)
    save_lineup_model(artifact, MODEL_PATH)
    validation = artifact["validation"]
    print(
        f"  ✓ trained {len(targets)} lineups "
        f"(RMSE {validation['rmse']:.4f}, tail AUC {validation['tail_auc']:.4f}) "
        f"-> {MODEL_PATH}"
    )
    _upload_json_gcs(MODEL_STORAGE_KEY, artifact)
    return artifact


def build_scores(tp, sl, tm, pp, psf, teams_df, baselines, model):
    scores = {}
    t0 = time.time()
    total = len(sl)
    for i, (key, ld) in enumerate(sl.items()):
        ids = [s["player_id"] for s in ld["starters"]]
        try:
            res = calculate_lineup_synergy(
                ids, ld["season"], psf, teams_df, sl, tp, tm, pp,
                compute_similar=False, season_baselines=baselines,
                synergy_model=model,
            )
            scores[key] = round(float(res["synergy_score"]), 1)
        except Exception as exc:
            print(f"  [skip] {key}: {exc}", file=sys.stderr)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{total} ({time.time()-t0:.0f}s)", file=sys.stderr)
    with open(SCORES_PATH, "w") as f:
        json.dump(scores, f)
    print(f"  ✓ wrote {len(scores)} synergy scores -> {SCORES_PATH}")
    _upload_json_gcs(SCORES_STORAGE_KEY, scores)
    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild lineup baselines, train the quality-fit model, and regenerate scores."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="rebuild every season's baseline (default)")
    group.add_argument("--latest", action="store_true", help="rebuild only the most recent season's baseline")
    group.add_argument("--season", help="rebuild a single season's baseline, e.g. 2025-26")
    args = parser.parse_args()

    tp, sl, tm, pp, psf, teams_df = _load_inputs()
    all_seasons = _all_seasons(sl)

    if args.season:
        if args.season not in all_seasons:
            print(f"  [ERROR] season {args.season!r} not found in starting lineups", file=sys.stderr)
            sys.exit(1)
        seasons = [args.season]
    elif args.latest:
        seasons = [all_seasons[-1]]
    else:
        seasons = all_seasons

    t0 = time.time()
    print(f"[1/3] Season lineup baselines ({len(seasons)} season(s): "
          f"{', '.join(seasons) if len(seasons) <= 5 else f'{len(seasons)} seasons'})...")
    baselines = build_baselines(sl, tm, psf, teams_df, seasons)
    print(f"[2/3] Outcome-trained quality and fit model ({len(sl)} starting lineups)...")
    model = build_model(tp, sl, tm, psf, teams_df, baselines)
    print(f"[3/3] Lineup synergy scores ({len(sl)} starting lineups)...")
    build_scores(tp, sl, tm, pp, psf, teams_df, baselines, model)
    print(f"Done in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
