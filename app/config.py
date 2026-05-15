"""Environment-driven settings and shared constants."""

import os
from pathlib import Path

# App root: run uvicorn from this directory (e.g. `cd app && uvicorn main:app`).
BASE_DIR = Path(__file__).resolve().parent
DATA_STATIC_DIR = BASE_DIR / "data" / "static"
SIMILAR_TEAMS_PATH = DATA_STATIC_DIR / "similar_teams.json"
TEAM_METADATA_PATH = DATA_STATIC_DIR / "team_metadata.json"
SEASON_INDEX_PATH = DATA_STATIC_DIR / "season_index.json"
TEAM_PROFILES_PATH = DATA_STATIC_DIR / "team_profiles.json"

CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173",
).split(",")
