import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env_defaults import load_env_defaults

load_env_defaults()

from analytics.lineup_model import (
    DEFAULT_MODEL_PATH,
    build_lineup_model_features,
    save_lineup_model,
    train_lineup_model,
)
from analytics.lineup_synergy import assign_player_roles_absolute
from analytics.season_baselines import (
    build_all_season_baselines,
    compute_lineup_metrics,
    percentile_in,
)
from parquet_io import read_teams_parquet

STATIC = Path(__file__).resolve().parents[1] / "data" / "static"


def load_player_seasons():
    storage_uri = os.getenv("PLAYER_PROFILES_STORAGE_URI")
    if storage_uri:
        try:
            from analytics.player_profiles.storage import ProfileStorage, StorageConfig

            storage = ProfileStorage(StorageConfig.from_uri(storage_uri))
            if storage.exists("features/player_season_features.parquet"):
                return storage.read_parquet("features/player_season_features.parquet")
        except Exception as exc:
            print(f"[WARN] Could not load training features from storage: {exc}", file=sys.stderr)
    return read_teams_parquet(STATIC / "player_season_features.parquet")


def load_season_baselines():
    storage_uri = os.getenv("PLAYER_PROFILES_STORAGE_URI")
    if storage_uri:
        try:
            from analytics.player_profiles.storage import ProfileStorage, StorageConfig

            storage = ProfileStorage(StorageConfig.from_uri(storage_uri))
            if storage.exists("static/season_lineup_baselines.json"):
                return storage.read_json("static/season_lineup_baselines.json")
        except Exception as exc:
            print(f"[WARN] Could not load training baselines from storage: {exc}", file=sys.stderr)
    baselines_path = STATIC / "season_lineup_baselines.json"
    return json.loads(baselines_path.read_text()) if baselines_path.exists() else {}


def load_inputs():
    starting_lineups = json.loads((STATIC / "starting_lineups.json").read_text())
    team_profiles = json.loads((STATIC / "team_profiles.json").read_text())
    team_metadata = json.loads((STATIC / "team_metadata.json").read_text())
    player_seasons = load_player_seasons()
    teams = read_teams_parquet(STATIC / "teams_historical.parquet")
    baselines = load_season_baselines()
    return starting_lineups, team_profiles, team_metadata, player_seasons, teams, baselines


def style_percentiles(metrics, baseline):
    distributions = baseline["metrics"]
    ast_volume = percentile_in(distributions["ast"], metrics["ast"])
    ast_rate = percentile_in(distributions["ast_pct"], metrics["ast_pct"])
    return {
        "pace": percentile_in(distributions["pace"], metrics["pace"]),
        "spacing": percentile_in(distributions["space"], metrics["space"]),
        "paint": percentile_in(distributions["paint_fga"], metrics["paint_fga"]),
        "defense": percentile_in(distributions["def_score"], metrics["def_score"]),
        "playmaking": 0.55 * ast_volume + 0.45 * ast_rate,
        "rebounding": percentile_in(distributions["reb"], metrics["reb"]),
    }


def build_training_rows(
    starting_lineups,
    team_profiles,
    team_metadata,
    player_seasons,
    teams,
    baselines,
):
    team_abbr_to_id = {value["abbreviation"]: int(key) for key, value in team_metadata.items()}
    if not baselines:
        baselines = build_all_season_baselines(
            player_seasons,
            teams,
            starting_lineups,
            team_metadata,
        )
    features = []
    targets = []
    seasons = []
    for key, lineup in starting_lineups.items():
        season = lineup["season"]
        baseline = baselines.get(season)
        profile = team_profiles.get(key, {})
        win_pct = profile.get("win_pct")
        if not baseline or not isinstance(win_pct, (int, float)):
            continue
        starter_ids = [int(starter["player_id"]) for starter in lineup["starters"]]
        season_rows = player_seasons[player_seasons["SEASON"] == season]
        rows = season_rows[season_rows["PLAYER_ID"].isin(starter_ids)]
        if len(rows) != 5:
            continue
        rows = rows.set_index("PLAYER_ID").loc[starter_ids].reset_index()
        roles = assign_player_roles_absolute(rows)
        metrics = compute_lineup_metrics(rows, roles, teams, team_abbr_to_id, season)
        features.append(
            build_lineup_model_features(rows, roles, style_percentiles(metrics, baseline))
        )
        targets.append(float(win_pct))
        seasons.append(season)
    return features, targets, seasons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    started = time.time()
    inputs = load_inputs()
    features, targets, seasons = build_training_rows(*inputs)
    artifact = train_lineup_model(features, targets, seasons)
    save_lineup_model(artifact, args.output)
    training = artifact["training"]
    validation = artifact["validation"]
    print(
        f"trained {training['lineups']} lineups across {training['seasons']} seasons "
        f"({training['successful']} successful, {training['unsuccessful']} unsuccessful)"
    )
    print(
        f"validation rmse={validation['rmse']:.4f} mae={validation['mae']:.4f} "
        f"tail_auc={validation['tail_auc']:.4f}"
    )
    print(f"wrote {args.output} in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
