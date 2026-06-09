import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import (
    BADGE_LEADERS_PATH,
    SEASON_INDEX_PATH,
    STARTING_LINEUPS_PATH,
    PLAYER_METADATA_PATH,
    PLAYER_PROFILES_PATH,
    PROSPECTS_JSON_PATH,
    TEAM_METADATA_PATH,
    TEAM_PROFILES_PATH,
    TEAMS_PARQUET_PATH,
    PLAYER_SEASON_FEATURES_PATH,
)
from analytics.normalizer import normalize_by_season
from parquet_io import read_teams_parquet
from routers import badges, draft, players, search, team, lineups


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

        # Collect MPG-adjusted PFVs and heights for population-level APFV ranking
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
        with PROSPECTS_JSON_PATH.open(encoding="utf-8") as f:
            prospects_list = json.load(f)
        app.state.prospects = prospects_list
        app.state.prospects_by_id = {p["prospect_id"]: p for p in prospects_list}
        print(f"  ✓ Prospects ({len(prospects_list)} prospects) from {PROSPECTS_JSON_PATH.name}")
    else:
        app.state.prospects = []
        app.state.prospects_by_id = {}
        print(f"  [WARN] {PROSPECTS_JSON_PATH.name} missing — GET /draft/* will return empty results")

    # Load historical prospects map
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

    # Compute global prospect APFV pool and percentile arrays during startup
    try:
        from analytics.prospect_apfv import prospect_percentile_arrays, global_prospect_metrics
        from analytics.player_profiles.archetypes import calculate_adjusted_pfv, height_bucket
        
        population = app.state.prospect_population
        percentile_arrays = prospect_percentile_arrays(population)
        app.state.global_prospect_percentile_arrays = percentile_arrays
        
        global_prospect_adjusted_pfvs = []
        global_prospect_height_buckets = []
        for prospect in population:
            metrics = global_prospect_metrics(prospect, percentile_arrays)
            global_prospect_adjusted_pfvs.append(calculate_adjusted_pfv(metrics, is_prospect=True))
            global_prospect_height_buckets.append(height_bucket(prospect.get("height", "")))
            
        app.state.global_prospect_adjusted_pfvs = global_prospect_adjusted_pfvs
        app.state.global_prospect_height_buckets = global_prospect_height_buckets
        print(f"  ✓ Precomputed global prospect APFV pool and percentiles ({len(population)} prospects)")
    except Exception as e:
        app.state.global_prospect_percentile_arrays = {}
        app.state.global_prospect_adjusted_pfvs = []
        app.state.global_prospect_height_buckets = []
        print(f"  [WARN] Failed to precompute global prospect APFV pool: {e}")


    import os
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


@app.get("/health")
def health():
    return {"status": "ok"}
