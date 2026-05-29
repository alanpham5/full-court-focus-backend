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

prospects = pd.DataFrame(prospects_data)
career = read_parquet_compat(app_dir / "data/static/player_career_features.parquet")

# Unpack prospect values to float
prospect_unpacked = {}
for col in SIMILARITY_FEATURES:
    if col in prospects.columns:
        prospect_unpacked[col] = prospects[col].apply(lambda x: x["value"] if isinstance(x, dict) else x)
    elif col == "mpg":
        prospect_unpacked[col] = prospects["mpg"].apply(lambda x: x["value"] if isinstance(x, dict) else x)
prospect_features = pd.DataFrame(prospect_unpacked)

# Scale within the draft class!
scaler = StandardScaler()
prospect_scaled = scaler.fit_transform(prospect_features)

roles = []
for idx, row in prospects.iterrows():
    sim_players = row["similar_nba_players"]
    if not sim_players:
        pos_group = "W"
    else:
        pos_group = sim_players[0].get("position_group", "W")

    # Use ast_per36_z as a proxy for ast_pct_z since they are highly correlated
    ast_z = prospect_scaled[idx, SIMILARITY_FEATURES.index("ast_per36")]
    
    # Build a series for assign_player_role
    p_row = pd.Series({
        "pts_per36_z": prospect_scaled[idx, SIMILARITY_FEATURES.index("pts_per36")],
        "ast_per36_z": ast_z,
        "reb_per36_z": prospect_scaled[idx, SIMILARITY_FEATURES.index("reb_per36")],
        "stl_per36_z": prospect_scaled[idx, SIMILARITY_FEATURES.index("stl_per36")],
        "blk_per36_z": prospect_scaled[idx, SIMILARITY_FEATURES.index("blk_per36")],
        "ts_pct_z": prospect_scaled[idx, SIMILARITY_FEATURES.index("ts_pct")],
        "efg_pct_z": prospect_scaled[idx, SIMILARITY_FEATURES.index("efg_pct")],
        "fg3a_rate_z": prospect_scaled[idx, SIMILARITY_FEATURES.index("fg3a_rate")],
        "fta_rate_z": prospect_scaled[idx, SIMILARITY_FEATURES.index("fta_rate")],
        "position_group": pos_group,
        "ast_pct_z": ast_z # proxy
    })
    
    role = assign_player_role(p_row)
    roles.append(role)

prospects["role"] = roles
print("\nRole distribution using draft-class scaling + closest NBA player position group:")
print(prospects["role"].value_counts())

print("\nSample player assignments:")
print(prospects[["player_name", "height", "role"]].head(25))
