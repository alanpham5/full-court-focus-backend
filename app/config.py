from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_STATIC_DIR = BASE_DIR / "data" / "static"
SIMILAR_TEAMS_PATH = DATA_STATIC_DIR / "similar_teams.json"
TEAM_METADATA_PATH = DATA_STATIC_DIR / "team_metadata.json"
SEASON_INDEX_PATH = DATA_STATIC_DIR / "season_index.json"
TEAM_PROFILES_PATH = DATA_STATIC_DIR / "team_profiles.json"
BADGE_LEADERS_PATH = DATA_STATIC_DIR / "badge_leaders.json"
