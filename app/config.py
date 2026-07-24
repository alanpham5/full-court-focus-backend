from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_STATIC_DIR = BASE_DIR / "data" / "static"
SIMILAR_TEAMS_PATH = DATA_STATIC_DIR / "similar_teams.json"
TEAM_METADATA_PATH = DATA_STATIC_DIR / "team_metadata.json"
SEASON_INDEX_PATH = DATA_STATIC_DIR / "season_index.json"
TEAM_PROFILES_PATH = DATA_STATIC_DIR / "team_profiles.json"
BADGE_LEADERS_PATH = DATA_STATIC_DIR / "badge_leaders.json"
STARTING_LINEUPS_PATH = DATA_STATIC_DIR / "starting_lineups.json"
LINEUP_SYNERGY_SCORES_PATH = DATA_STATIC_DIR / "lineup_synergy_scores.json"
LINEUP_SYNERGY_MODEL_PATH = DATA_STATIC_DIR / "lineup_synergy_model.json"
SEASON_LINEUP_BASELINES_PATH = DATA_STATIC_DIR / "season_lineup_baselines.json"
TEAMS_PARQUET_PATH = DATA_STATIC_DIR / "teams_historical.parquet"
PLAYER_PROFILES_PATH = DATA_STATIC_DIR / "player_profiles.json"
PLAYER_METADATA_PATH = DATA_STATIC_DIR / "player_metadata.json"
PLAYER_CAREER_FEATURES_PATH = DATA_STATIC_DIR / "player_career_features.parquet"
PROSPECTS_JSON_PATH = DATA_STATIC_DIR / "prospects.json"
PLAYER_SEASON_FEATURES_PATH = DATA_STATIC_DIR / "player_season_features.parquet"
PROSPECT_LINEUP_FEATURES_PATH = DATA_STATIC_DIR / "prospect_lineup_features.parquet"

                                                                              
                                                                              
PROSPECT_TUNING_DIR = BASE_DIR / "data" / "tuning"
