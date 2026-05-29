import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

app_dir = Path(__file__).resolve().parents[1]
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from scripts.build_prospects_dataset import SIMILARITY_FEATURES, read_parquet_compat
from analytics.player_profiles.archetypes import assign_player_role

# Load prospects JSON and career features Parquet
with open(app_dir / "data/static/prospects.json", "r", encoding="utf-8") as f:
    prospects_data = json.load(f)

# Convert to DataFrame
prospects = pd.DataFrame(prospects_data)

career = read_parquet_compat(app_dir / "data/static/player_career_features.parquet")

# We want to fit the scaler on NBA features and scale prospects
nba_features = career[SIMILARITY_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)

# Unpack prospect values since they are dicts: {"value": X, "percentile": Y}
prospect_unpacked = {}
for col in SIMILARITY_FEATURES:
    if col in prospects.columns:
        prospect_unpacked[col] = prospects[col].apply(lambda x: x["value"] if isinstance(x, dict) else x)
    elif col == "mpg":
        prospect_unpacked[col] = prospects["mpg"].apply(lambda x: x["value"] if isinstance(x, dict) else x)

prospect_features = pd.DataFrame(prospect_unpacked)

scaler = StandardScaler()
scaler.fit(nba_features)
prospect_scaled = scaler.transform(prospect_features)

# Let's map similar NBA player to get position_group and ast_pct_z
career_by_id = career.set_index("player_id")

roles = []
for idx, row in prospects.iterrows():
    sim_players = row["similar_nba_players"]
    if not sim_players:
        roles.append("Secondary Creator")
        continue
    
    closest_id = sim_players[0]["player_id"]
    if closest_id in career_by_id.index:
        closest_nba = career_by_id.loc[closest_id]
        if isinstance(closest_nba, pd.DataFrame):
            closest_nba = closest_nba.iloc[0]
        pos_group = closest_nba.get("position_group", "W")
        ast_pct_z = closest_nba.get("ast_pct_z", 0.0)
    else:
        pos_group = "W"
        ast_pct_z = 0.0

    # Build a series for assign_player_role
    p_row = pd.Series({
        "pts_per36_z": prospect_scaled[idx, SIMILARITY_FEATURES.index("pts_per36")],
        "ast_per36_z": prospect_scaled[idx, SIMILARITY_FEATURES.index("ast_per36")],
        "reb_per36_z": prospect_scaled[idx, SIMILARITY_FEATURES.index("reb_per36")],
        "stl_per36_z": prospect_scaled[idx, SIMILARITY_FEATURES.index("stl_per36")],
        "blk_per36_z": prospect_scaled[idx, SIMILARITY_FEATURES.index("blk_per36")],
        "ts_pct_z": prospect_scaled[idx, SIMILARITY_FEATURES.index("ts_pct")],
        "efg_pct_z": prospect_scaled[idx, SIMILARITY_FEATURES.index("efg_pct")],
        "fg3a_rate_z": prospect_scaled[idx, SIMILARITY_FEATURES.index("fg3a_rate")],
        "fta_rate_z": prospect_scaled[idx, SIMILARITY_FEATURES.index("fta_rate")],
        "position_group": pos_group,
        "ast_pct_z": ast_pct_z
    })
    
    role = assign_player_role(p_row)
    roles.append(role)

prospects["role"] = roles
print("\nRole distribution using closest NBA player metadata:")
print(prospects["role"].value_counts())

print("\nSample player assignments:")
print(prospects[["player_name", "height", "role"]].head(15))
