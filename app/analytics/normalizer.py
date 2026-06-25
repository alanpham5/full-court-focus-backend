import numpy as np
import pandas as pd

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

        # Per-column z-score, matching sklearn StandardScaler defaults
        # (population std, ddof=0; zero-variance columns left unscaled).
        values = group[available].to_numpy(dtype=float)
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
        std[std == 0] = 1.0
        transformed = (values - mean) / std
        transformed = np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0)
        group[z_names] = transformed
        result_parts.append(group)

    return pd.concat(result_parts, ignore_index=True)
