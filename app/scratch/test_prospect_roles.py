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

# Load prospects and career features
prospects = read_parquet_compat(app_dir / "data/static/prospects.parquet")
career = read_parquet_compat(app_dir / "data/static/player_career_features.parquet")

nba_features = career[SIMILARITY_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)
prospect_features = prospects[SIMILARITY_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)

scaler = StandardScaler()
scaler.fit(nba_features)
prospect_scaled = scaler.transform(prospect_features)

# Add z-score columns
scaled_cols = [f"{col}_z" for col in SIMILARITY_FEATURES]
scaled_df = pd.DataFrame(prospect_scaled, columns=scaled_cols, index=prospects.index)
df = pd.concat([prospects, scaled_df], axis=1)

def assign_prospect_role(row: pd.Series) -> str:
    pts = float(row["pts_per36_z"])
    ast = float(row["ast_per36_z"])
    reb = float(row["reb_per36_z"])
    stl = float(row["stl_per36_z"])
    blk = float(row["blk_per36_z"])
    efg = float(row["efg_pct_z"])
    fg3 = float(row["fg3a_rate_z"])
    fta = float(row["fta_rate_z"])
    
    h_str = str(row.get("height", ""))
    height_inches = 0
    if h_str and "-" in h_str:
        try:
            parts = h_str.split("-")
            height_inches = int(parts[0]) * 12 + int(parts[1])
        except Exception:
            pass

    # 1. Playmaker: High assist relative to NBA careers
    if ast > 1.0:
        return "Playmaker"

    # 2. Interior Presence: Big, high rebounds/blocks, low 3-point attempts
    if (height_inches >= 81 or reb > 0.8) and fg3 < 0.0:
        return "Interior Presence"
    if blk > 1.0 and reb > 0.5 and fg3 < 0.0:
        return "Interior Presence"

    # 3. Designated Scorer: High scorer, average or below assists
    if pts > 0.8 and ast < 0.4:
        return "Designated Scorer"

    # 4. Secondary Creator: Above average passing and scoring
    if ast > 0.3 and (pts > 0.2 or ast > 0.1):
        return "Secondary Creator"

    # 5. Perimeter Specialist: Heavy 3-point volume, decent efficiency
    if fg3 > 0.8 and efg > -0.5:
        return "Perimeter Specialist"

    # 6. Rim Attacker: Good free throw drawing rate, decent scoring
    if fta > 0.5 and pts > 0.0:
        return "Rim Attacker"

    # 7. Defensive Specialist: High events (steals/blocks), lower scoring
    if (stl > 0.5 or blk > 0.5) and pts < -0.2:
        return "Defensive Specialist"

    # Fallbacks
    if height_inches >= 81:
        return "Interior Presence"
    if fg3 > 0.3:
        return "Perimeter Specialist"
    if ast > 0.0:
        return "Secondary Creator"
    if pts > 0.0:
        return "Rim Attacker"
    return "Defensive Specialist"

df["role"] = df.apply(assign_prospect_role, axis=1)
print(df["role"].value_counts())
print("\nSome sample players:")
for role in df["role"].unique():
    sample = df[df["role"] == role][["player_name", "height", "pts_per36", "ast_per36"]].head(3)
    print(f"\nRole: {role}")
    print(sample)
