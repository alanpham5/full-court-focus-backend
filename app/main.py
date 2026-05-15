import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import (
    BADGE_LEADERS_PATH,
    SEASON_INDEX_PATH,
    TEAM_METADATA_PATH,
    TEAM_PROFILES_PATH,
)
from routers import badges, search, team


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


@app.get("/health")
def health():
    return {"status": "ok"}
