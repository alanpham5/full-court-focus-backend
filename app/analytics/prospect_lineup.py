from __future__ import annotations

import pandas as pd


def combine_lineup_features(
    player_df: pd.DataFrame | None,
    prospect_df: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if prospect_df is None or prospect_df.empty:
        return player_df
    if player_df is None or player_df.empty:
        return prospect_df
    combined = pd.concat([player_df, prospect_df], ignore_index=True)
    if "is_prospect" in combined.columns:
        combined["is_prospect"] = combined["is_prospect"].fillna(False).astype(bool)
    return combined


def prospect_lineup_index(prospect_df: pd.DataFrame | None) -> dict[int, dict]:
    index: dict[int, dict] = {}
    if prospect_df is None or prospect_df.empty:
        return index
    for _, row in prospect_df.iterrows():
        index[int(row["PLAYER_ID"])] = {
            "prospect_id": str(row.get("prospect_id", "")),
            "name": str(row.get("PLAYER_NAME", "")),
            "season": str(row.get("SEASON", "")),
            "counterpart_id": int(row.get("counterpart_id", 0) or 0),
            "counterpart_name": str(row.get("counterpart_name", "")),
            "draft_class_year": int(row.get("draft_class_year", 0) or 0),
        }
    return index


def prospects_for_season(prospect_df: pd.DataFrame | None, season: str) -> pd.DataFrame:
    if prospect_df is None or prospect_df.empty:
        return pd.DataFrame()
    return prospect_df[prospect_df["SEASON"] == season]
