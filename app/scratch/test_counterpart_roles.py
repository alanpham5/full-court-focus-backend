import sys
import json
from pathlib import Path
import pandas as pd

app_dir = Path(__file__).resolve().parents[1]
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from scripts.build_prospects_dataset import read_parquet_compat

# Load prospects JSON and career features Parquet
with open(app_dir / "data/static/prospects.json", "r", encoding="utf-8") as f:
    prospects_data = json.load(f)

prospects = pd.DataFrame(prospects_data)
career = read_parquet_compat(app_dir / "data/static/player_career_features.parquet")
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
        roles.append(closest_nba.get("role", "Secondary Creator"))
    else:
        roles.append("Secondary Creator")

prospects["role"] = roles
print("\nRole distribution using closest NBA player's role directly:")
print(prospects["role"].value_counts())

print("\nSample player assignments:")
print(prospects[["player_name", "height", "role"]].head(25))
