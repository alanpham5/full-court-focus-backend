"""
Z-score normalize each stat within its season so teams from different
eras are comparable in the similarity engine.
"""

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
    """
    For each season, z-score all similarity features independently.
    Returns a new df with _Z suffix columns appended.
    """
    result_parts: list[pd.DataFrame] = []

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
