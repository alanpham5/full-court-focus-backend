import json
import os
from contextlib import asynccontextmanager

# Load .env before any first-party import that initializes an external SDK at
# import time. routers -> routers.auth -> firebase_config initializes Firebase
# on import and needs GOOGLE_CLOUD_PROJECT / FIREBASE_SERVICE_ACCOUNT_B64 set.
from env_defaults import load_env_defaults

load_env_defaults()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import (
    BADGE_LEADERS_PATH,
    SEASON_INDEX_PATH,
    STARTING_LINEUPS_PATH,
    LINEUP_SYNERGY_SCORES_PATH,
    SEASON_LINEUP_BASELINES_PATH,
    PLAYER_METADATA_PATH,
    PLAYER_PROFILES_PATH,
    PROSPECTS_JSON_PATH,
    TEAM_METADATA_PATH,
    TEAM_PROFILES_PATH,
    TEAMS_PARQUET_PATH,
    PLAYER_SEASON_FEATURES_PATH,
    PROSPECT_LINEUP_FEATURES_PATH,
)
from analytics.normalizer import normalize_by_season
from parquet_io import read_teams_parquet
from routers import badges, draft, players, search, team, lineups, game, auth


@asynccontextmanager
async def lifespan(app: FastAPI):
    if TEAM_PROFILES_PATH.exists():
        with TEAM_PROFILES_PATH.open() as f:
            app.state.team_profiles = json.load(f)
        print(f"  ✓ Team profiles ({len(app.state.team_profiles)} keys) from {TEAM_PROFILES_PATH.name}")
    else:
        app.state.team_profiles = {}
        print(f"  [WARN] {TEAM_PROFILES_PATH.name} missing — GET /team/{{id}}/{{season}} will 404")

    if TEAM_METADATA_PATH.exists():
        with TEAM_METADATA_PATH.open() as f:
            app.state.team_metadata = json.load(f)
        app.state.team_search_choices = {
            tid: meta["name"] for tid, meta in app.state.team_metadata.items()
        }
        print(f"  ✓ Team metadata ({len(app.state.team_metadata)} teams)")
    else:
        app.state.team_metadata = {}
        app.state.team_search_choices = {}
        print("  [WARN] team_metadata.json missing — search will return no teams")

    if SEASON_INDEX_PATH.exists():
        with SEASON_INDEX_PATH.open() as f:
            app.state.season_index = json.load(f)
        print(f"  ✓ Season index ({len(app.state.season_index)} teams)")
    else:
        app.state.season_index = {}
        print("  [WARN] season_index.json missing")

    if BADGE_LEADERS_PATH.exists():
        with BADGE_LEADERS_PATH.open() as f:
            app.state.badge_leaders = json.load(f)
        print(
            f"  ✓ Badge leaders ({len(app.state.badge_leaders)} seasons) "
            f"from {BADGE_LEADERS_PATH.name}"
        )
    else:
        app.state.badge_leaders = {}
        print(
            f"  [WARN] {BADGE_LEADERS_PATH.name} missing — "
            "GET /badges/{season}/leaders will 404"
        )

    if STARTING_LINEUPS_PATH.exists():
        with STARTING_LINEUPS_PATH.open() as f:
            app.state.starting_lineups = json.load(f)
        print(
            f"  ✓ Starting lineups ({len(app.state.starting_lineups)} keys) "
            f"from {STARTING_LINEUPS_PATH.name}"
        )
    else:
        app.state.starting_lineups = {}
        print(
            f"  [WARN] {STARTING_LINEUPS_PATH.name} missing — "
            "GET /team/{id}/{season}/lineup will 404"
        )

    if LINEUP_SYNERGY_SCORES_PATH.exists():
        with LINEUP_SYNERGY_SCORES_PATH.open() as f:
            app.state.lineup_synergy_scores = json.load(f)
        print(
            f"  ✓ Lineup synergy scores ({len(app.state.lineup_synergy_scores)} keys) "
            f"from {LINEUP_SYNERGY_SCORES_PATH.name}"
        )
    else:
        app.state.lineup_synergy_scores = {}
        print(
            f"  [WARN] {LINEUP_SYNERGY_SCORES_PATH.name} missing — "
            "Lineup IQ game will fall back to live percentile scoring"
        )

    app.state.season_lineup_baselines = {}
    _baselines_loaded = False
    _baselines_storage_uri = os.getenv("PLAYER_PROFILES_STORAGE_URI")
    if _baselines_storage_uri:
        try:
            from analytics.player_profiles.storage import ProfileStorage, StorageConfig
            _storage = ProfileStorage(StorageConfig.from_uri(_baselines_storage_uri))
            if _storage.exists("static/season_lineup_baselines.json"):
                app.state.season_lineup_baselines = _storage.read_json(
                    "static/season_lineup_baselines.json"
                )
                _baselines_loaded = True
                print(
                    f"  ✓ Season lineup baselines ({len(app.state.season_lineup_baselines)} seasons) "
                    f"from storage: {_baselines_storage_uri}"
                )
        except Exception as e:
            print(f"  [WARN] Failed to load season lineup baselines from storage: {e}")
    if not _baselines_loaded:
        if SEASON_LINEUP_BASELINES_PATH.exists():
            with SEASON_LINEUP_BASELINES_PATH.open() as f:
                app.state.season_lineup_baselines = json.load(f)
            print(
                f"  ✓ Season lineup baselines ({len(app.state.season_lineup_baselines)} seasons) "
                f"from {SEASON_LINEUP_BASELINES_PATH.name}"
            )
        else:
            print(
                f"  [WARN] {SEASON_LINEUP_BASELINES_PATH.name} missing — "
                "synergy will fall back to live per-season baseline computation"
            )

    if TEAMS_PARQUET_PATH.exists():
        app.state.teams_df = read_teams_parquet(TEAMS_PARQUET_PATH)
        app.state.norm_df = normalize_by_season(app.state.teams_df)
        print(f"  ✓ Teams parquet ({len(app.state.teams_df)} rows) for era similarity")
    else:
        app.state.teams_df = None
        app.state.norm_df = None
        print(f"  [WARN] {TEAMS_PARQUET_PATH.name} missing — era similarity unavailable")

    if PLAYER_PROFILES_PATH.exists():
        with PLAYER_PROFILES_PATH.open() as f:
            app.state.player_profiles = json.load(f)
        print(f"  ✓ Player profiles ({len(app.state.player_profiles)} players)")

        from analytics.player_profiles.archetypes import (
            calculate_adjusted_pfv,
            remove_mpg_adjustment_from_metrics,
            height_bucket,
        )
        player_adjusted_pfvs = []
        player_heights = []
        for p in app.state.player_profiles.values():
            p_metrics = p.get("playstyle_metrics", {})
            if p_metrics:
                player_adjusted_pfvs.append(
                    calculate_adjusted_pfv(remove_mpg_adjustment_from_metrics(p_metrics))
                )
                player_heights.append(p.get("height", ""))
        app.state.player_all_adjusted_pfvs = player_adjusted_pfvs
        app.state.player_all_height_buckets = [height_bucket(h) for h in player_heights]
    else:
        app.state.player_profiles = {}
        app.state.player_all_adjusted_pfvs = []
        app.state.player_all_height_buckets = []
        print(f"  [WARN] {PLAYER_PROFILES_PATH.name} missing — GET /players/{{id}} will 404")

    if PLAYER_METADATA_PATH.exists():
        with PLAYER_METADATA_PATH.open() as f:
            app.state.player_metadata = json.load(f)
        print(f"  ✓ Player metadata ({len(app.state.player_metadata)} players)")
    else:
        app.state.player_metadata = {}
        print(f"  [WARN] {PLAYER_METADATA_PATH.name} missing — player search will return no rows")

    if PROSPECTS_JSON_PATH.exists():
        from analytics.draft_year import (
            current_draft_year_from_date,
            normalize_prospects_payload,
        )

        with PROSPECTS_JSON_PATH.open(encoding="utf-8") as f:
            prospects_list, prospects_meta = normalize_prospects_payload(json.load(f))
        app.state.prospects = prospects_list
        app.state.prospects_by_id = {p["prospect_id"]: p for p in prospects_list}
        app.state.draft_class_year = (
            prospects_meta.get("draft_class_year") or current_draft_year_from_date()
        )
        print(
            f"  ✓ Prospects ({len(prospects_list)} prospects, "
            f"{app.state.draft_class_year} class) from {PROSPECTS_JSON_PATH.name}"
        )
    else:
        from analytics.draft_year import current_draft_year_from_date

        app.state.prospects = []
        app.state.prospects_by_id = {}
        app.state.draft_class_year = current_draft_year_from_date()
        print(f"  [WARN] {PROSPECTS_JSON_PATH.name} missing — GET /draft/* will return empty results")

    app.state.historical_prospects_map = {}
    app.state.historical_prospects = []
    app.state.historical_prospects_by_year = {}
    from config import DATA_STATIC_DIR
    static_draft_dir = DATA_STATIC_DIR / "draft"
    if static_draft_dir.exists():
        for path in static_draft_dir.glob("prospects_*.json"):
            try:
                with path.open(encoding="utf-8") as f:
                    data = json.load(f)
                    year = int(path.stem.split("_")[-1])
                    app.state.historical_prospects_by_year[year] = data
                    app.state.historical_prospects.extend(data)
                    for item in data:
                        app.state.historical_prospects_map[item["prospect_id"]] = path
            except Exception as e:
                print(f"  [WARN] Failed to load historical prospects from {path.name}: {e}")
        print(f"  ✓ Historical prospects index built ({len(app.state.historical_prospects_map)} players)")
    else:
        print("  [INFO] Historical drafts folder (static/draft/) does not exist yet")
    app.state.prospect_population = app.state.prospects + app.state.historical_prospects

    try:
        from analytics.prospect_apfv import prospect_percentile_arrays, global_prospect_metrics
        from analytics.player_profiles.archetypes import calculate_prospect_impact_adjusted_pfv, height_bucket

        population = app.state.prospect_population
        percentile_arrays = prospect_percentile_arrays(population)
        app.state.global_prospect_percentile_arrays = percentile_arrays

        global_prospect_adjusted_pfvs = []
        global_prospect_height_buckets = []
        for prospect in population:
            metrics = global_prospect_metrics(prospect, percentile_arrays)
            gp_val = float(prospect.get("gp", prospect.get("raw_stats", {}).get("gp", 0)) or 0)
            metrics["gp"] = {"value": gp_val, "percentile": 0.0}
            metrics["team"] = str(prospect.get("team", "") or "")
            global_prospect_adjusted_pfvs.append(calculate_prospect_impact_adjusted_pfv(metrics))
            global_prospect_height_buckets.append(height_bucket(prospect.get("height", "")))
            
        app.state.global_prospect_adjusted_pfvs = global_prospect_adjusted_pfvs
        app.state.global_prospect_height_buckets = global_prospect_height_buckets
        print(f"  ✓ Precomputed global prospect APFV pool and percentiles ({len(population)} prospects)")
    except Exception as e:
        app.state.global_prospect_percentile_arrays = {}
        app.state.global_prospect_adjusted_pfvs = []
        app.state.global_prospect_height_buckets = []
        print(f"  [WARN] Failed to precompute global prospect APFV pool: {e}")


    from analytics.player_profiles.storage import ProfileStorage, StorageConfig

    storage_uri = os.getenv("PLAYER_PROFILES_STORAGE_URI")
    loaded_from_storage = False

    if storage_uri:
        try:
            storage = ProfileStorage(StorageConfig.from_uri(storage_uri))
            if storage.exists("features/player_season_features.parquet"):
                app.state.player_season_features_df = storage.read_parquet("features/player_season_features.parquet")
                print(f"  ✓ Player season features ({len(app.state.player_season_features_df)} rows) loaded from storage: {storage_uri}")
                loaded_from_storage = True
            else:
                print(f"  [WARN] features/player_season_features.parquet not found in storage: {storage_uri}")
        except Exception as e:
            print(f"  [WARN] Failed to load player season features from storage ({storage_uri}): {e}")

    if not loaded_from_storage:
        if PLAYER_SEASON_FEATURES_PATH.exists():
            app.state.player_season_features_df = read_teams_parquet(PLAYER_SEASON_FEATURES_PATH)
            print(f"  ✓ Player season features ({len(app.state.player_season_features_df)} rows) from {PLAYER_SEASON_FEATURES_PATH.name}")
        else:
            app.state.player_season_features_df = None
            print(f"  [WARN] {PLAYER_SEASON_FEATURES_PATH.name} missing and storage not configured — lineup synergy will fail")

    from analytics.prospect_lineup import combine_lineup_features, prospect_lineup_index

    prospect_lineup_df = None
    if storage_uri:
        try:
            storage = ProfileStorage(StorageConfig.from_uri(storage_uri))
            if storage.exists("features/prospect_lineup_features.parquet"):
                prospect_lineup_df = storage.read_parquet("features/prospect_lineup_features.parquet")
        except Exception as e:
            print(f"  [WARN] Failed to load prospect lineup features from storage: {e}")
    if prospect_lineup_df is None and PROSPECT_LINEUP_FEATURES_PATH.exists():
        prospect_lineup_df = read_teams_parquet(PROSPECT_LINEUP_FEATURES_PATH)

    app.state.prospect_lineup_features_df = prospect_lineup_df
    app.state.lineup_features_df = combine_lineup_features(
        app.state.player_season_features_df, prospect_lineup_df
    )
    app.state.prospect_lineup_index = prospect_lineup_index(prospect_lineup_df)
    if prospect_lineup_df is not None:
        print(f"  ✓ Prospect lineup features ({len(prospect_lineup_df)} rows)")

    yield


app = FastAPI(
    title="Basketball Analytics API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(team.router)
app.include_router(search.router)
app.include_router(badges.router)
app.include_router(players.router)
app.include_router(draft.router)
app.include_router(lineups.router)
app.include_router(game.router)
app.include_router(auth.router)


@app.get("/health")
def health():
    return {"status": "ok"}
