import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

SIMILARITY_FEATURES = [
    "PACE",
    "OFF_RATING",
    "DEF_RATING",
    "AST_PCT",
    "AST_TO",
    "TM_TOV_PCT",
    "OREB_PCT",
    "DREB_PCT",
    "TS_PCT",
    "FG3A",
    "FG3_PCT",
    "PAINT_FGA",
    "FTA_RATE",
]


def normalize_by_season(df: pd.DataFrame) -> pd.DataFrame:
    result_parts: list[pd.DataFrame] = []
    df = df.copy()
    if "PAINT_FGA" not in df.columns and "PTS_PAINT" in df.columns:
        df["PAINT_FGA"] = pd.to_numeric(df["PTS_PAINT"], errors="coerce")

    for _, group in df.groupby("SEASON"):
        group = group.copy()
        available = [c for c in SIMILARITY_FEATURES if c in group.columns]
        z_names = [f"{c}_Z" for c in available]
        if not available:
            result_parts.append(group)
            continue

        scaler = StandardScaler()
        transformed = scaler.fit_transform(group[available])
        transformed = np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0)
        group[z_names] = transformed
        result_parts.append(group)

    return pd.concat(result_parts, ignore_index=True)
