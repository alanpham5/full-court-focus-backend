from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

MIN_ACTIVE_GAMES = 1
INACTIVE_MIN_CAREER_GAMES = 82
RECENT_SEASON_WINDOW = 3

POSITION_GROUPS = {
    "G": {"G", "PG", "SG", "PG-SG", "SG-PG"},
    "W": {"F", "SF", "SG-SF", "SF-SG", "SF-PF", "PF-SF", "G-F", "F-G"},
    "B": {"C", "PF", "PF-C", "C-PF", "F-C", "C-F"},
}

STYLE_FEATURES = [
    "pts_per36_z",
    "reb_per36_z",
    "ast_per36_z",
    "blk_per36_z",
    "stl_per36_z",
    "tov_per36_z",
    "fg3a_rate_z",
    "fta_rate_z",
    "ts_pct_z",
    "efg_pct_z",
    "ast_pct_z",
    "mpg_z",
]

PLAYSTYLE_METRIC_KEYS = [feature[:-2] for feature in STYLE_FEATURES if feature.endswith("_z")]

CANONICAL_TRADITIONAL_COLUMNS = [
    "PLAYER_ID",
    "PLAYER_NAME",
    "TEAM_ABBREVIATION",
    "GP",
    "MIN",
    "FGM",
    "FGA",
    "FG_PCT",
    "FG3M",
    "FG3A",
    "FG3_PCT",
    "FTM",
    "FTA",
    "FT_PCT",
    "OREB",
    "DREB",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PF",
    "PTS",
]

CANONICAL_SEASON_PLAYER_COLUMNS = [
    *CANONICAL_TRADITIONAL_COLUMNS,
    "SEASON",
    "SEASON_START",
    "POSITION_GROUP",
    "POSITION",
    "HEIGHT",
    "WEIGHT",
    "DRAFT_YEAR",
    "DRAFT_NUMBER",
    "pts_per36",
    "ast_per36",
    "reb_per36",
    "stl_per36",
    "blk_per36",
    "tov_per36",
    "fg3a_rate",
    "fta_rate",
    "ts_pct",
    "efg_pct",
    "ast_pct",
    "mpg",
]


@dataclass(frozen=True)
class PlayerFilterConfig:
    recent_season_window: int = RECENT_SEASON_WINDOW
    inactive_min_career_games: int = INACTIVE_MIN_CAREER_GAMES


def _num(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _col(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        return _num(df[name], default)
    return pd.Series(default, index=df.index, dtype="float64")


def _first_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def normalize_position(pos: Any) -> str:
    if pd.isna(pos) or not isinstance(pos, str) or not pos.strip():
        return ""
    pos = pos.upper().strip()
    mapping = {
        "GUARD": "G",
        "FORWARD": "F",
        "CENTER": "C",
        "GUARD-FORWARD": "G-F",
        "FORWARD-GUARD": "F-G",
        "FORWARD-CENTER": "F-C",
        "CENTER-FORWARD": "C-F",
    }
    return mapping.get(pos, pos)


def position_group(position: str | None) -> str:
    pos = normalize_position(position)
    for group, values in POSITION_GROUPS.items():
        if pos in values:
            return group
    for group, values in POSITION_GROUPS.items():
        if any(p in values for p in pos.split("-")):
            return group
    return "W"


def season_start_year(season: str) -> int:
    return int(str(season).split("-", 1)[0])


def _suffix_columns(df: pd.DataFrame, suffix: str, skip: set[str]) -> pd.DataFrame:
    rename = {c: f"{c}{suffix}" for c in df.columns if c not in skip}
    return df.rename(columns=rename)


def _endpoint_frame(frames: dict[str, pd.DataFrame], fallback_name: str = "") -> pd.DataFrame:
    if not frames:
        logger.warning("No frames found for %s", fallback_name)
        return pd.DataFrame()
    return next(iter(frames.values())).copy()


def build_season_player_table(
    season: str,
    traditional: pd.DataFrame,
    advanced: pd.DataFrame,
    misc: pd.DataFrame,
    scoring: pd.DataFrame,
    usage: pd.DataFrame,
    bio: pd.DataFrame,
    drives: pd.DataFrame,
    touches: pd.DataFrame,
    passing: pd.DataFrame,
    pullup: pd.DataFrame,
    catch_shoot: pd.DataFrame,
    hustle: pd.DataFrame,
    defense: pd.DataFrame,
    shot_chart: pd.DataFrame,
) -> pd.DataFrame:
    base = normalize_traditional_totals(traditional)
    if base.empty:
        return pd.DataFrame()

    base["SEASON"] = season
    base["SEASON_START"] = season_start_year(season)

    if not bio.empty and "PERSON_ID" in bio.columns:
        bio_clean = bio[['PERSON_ID', 'POSITION', 'HEIGHT', 'WEIGHT', 'DRAFT_YEAR', 'DRAFT_NUMBER']].copy()
        bio_clean = bio_clean.rename(columns={'PERSON_ID': 'PLAYER_ID'})
        bio_clean['PLAYER_ID'] = pd.to_numeric(bio_clean['PLAYER_ID'], errors='coerce').fillna(0).astype(int)
        bio_clean = bio_clean.drop_duplicates(subset=['PLAYER_ID'])
        base = base.merge(bio_clean, on='PLAYER_ID', how='left')

    for col in ["POSITION", "HEIGHT", "WEIGHT", "DRAFT_YEAR", "DRAFT_NUMBER"]:
        if col not in base.columns:
            base[col] = None

    base["POSITION_GROUP"] = base["POSITION"].apply(position_group)
    _add_rate_features(base)
    return base[CANONICAL_SEASON_PLAYER_COLUMNS].copy()


def normalize_traditional_totals(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=CANONICAL_TRADITIONAL_COLUMNS)

    out = frame.copy()
    rename = {}
    if "PLAYER" in out.columns and "PLAYER_NAME" not in out.columns:
        rename["PLAYER"] = "PLAYER_NAME"
    if "TEAM" in out.columns and "TEAM_ABBREVIATION" not in out.columns:
        rename["TEAM"] = "TEAM_ABBREVIATION"
    out = out.rename(columns=rename)

    for col in CANONICAL_TRADITIONAL_COLUMNS:
        if col not in out.columns:
            out[col] = "" if col in {"PLAYER_NAME", "TEAM_ABBREVIATION"} else 0

    out = out[CANONICAL_TRADITIONAL_COLUMNS].copy()
    for col in CANONICAL_TRADITIONAL_COLUMNS:
        if col in {"PLAYER_NAME", "TEAM_ABBREVIATION"}:
            out[col] = out[col].fillna("").astype(str)
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    out["PLAYER_ID"] = out["PLAYER_ID"].astype(int)
    return out


def _add_rate_features(df: pd.DataFrame) -> None:
    fga = _col(df, "FGA")
    fta = _col(df, "FTA")
    fgm = _col(df, "FGM")
    fg3m = _col(df, "FG3M")
    pts = _per36(df, "PTS")
    ast = _per36(df, "AST")
    reb = _per36(df, "REB")
    stl = _per36(df, "STL")
    blk = _per36(df, "BLK")
    tov = _per36(df, "TOV")
    gp = _col(df, "GP").replace(0, np.nan)
    minutes = _col(df, "MIN").replace(0, np.nan)

    df["pts_per36"] = pts
    df["ast_per36"] = ast
    df["reb_per36"] = reb
    df["stl_per36"] = stl
    df["blk_per36"] = blk
    df["tov_per36"] = tov
    df["fg3a_rate"] = _col(df, "FG3A") / fga.replace(0, np.nan)
    df["fta_rate"] = fta / fga.replace(0, np.nan)
    df["ts_pct"] = _col(df, "PTS") / (2.0 * (fga + 0.44 * fta).replace(0, np.nan))
    df["efg_pct"] = (fgm + 0.5 * fg3m) / fga.replace(0, np.nan)
    df["ast_pct"] = _col(df, "AST") / fgm.replace(0, np.nan)
    df["mpg"] = minutes / gp

    rate_cols = [
        "fg3a_rate",
        "fta_rate",
        "ts_pct",
        "efg_pct",
        "ast_pct",
        "mpg",
    ]
    df[rate_cols] = df[rate_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _per36(df: pd.DataFrame, col: str) -> pd.Series:
    values = _col(df, col)
    if "NBA_API_PER_MODE" in df.columns:
        per_mode = df["NBA_API_PER_MODE"].astype(str)
        if per_mode.str.lower().eq("per36").all():
            return values
    minutes = _col(df, "MIN", np.nan).replace(0, np.nan)
    converted = values / minutes * 36.0
    return converted.replace([np.inf, -np.inf], np.nan).fillna(values)


def _add_shot_location_features(df: pd.DataFrame, shot_chart: pd.DataFrame) -> None:
    for col in ("rim_freq", "midrange_freq", "corner3_freq"):
        df[col] = 0.0
    if shot_chart.empty or "PLAYER_ID" not in shot_chart.columns:
        return

    sc = shot_chart.copy()
    sc["SHOT_ZONE_BASIC"] = sc.get("SHOT_ZONE_BASIC", "").astype(str)
    sc["SHOT_ZONE_AREA"] = sc.get("SHOT_ZONE_AREA", "").astype(str)
    sc["SHOT_ZONE_RANGE"] = sc.get("SHOT_ZONE_RANGE", "").astype(str)
    grouped = sc.groupby("PLAYER_ID")
    attempts = grouped.size().rename("shot_attempts")
    rim = grouped["SHOT_ZONE_BASIC"].apply(lambda s: s.str.contains("Restricted Area", case=False, na=False).sum())
    mid = grouped["SHOT_ZONE_BASIC"].apply(
        lambda s: s.str.contains("Mid-Range|In The Paint", case=False, na=False).sum()
    )
    corner = sc[
        sc["SHOT_ZONE_BASIC"].str.contains("Above the Break 3|Right Corner 3|Left Corner 3|Corner", case=False, na=False)
        & sc["SHOT_ZONE_AREA"].str.contains("Right Side|Left Side", case=False, na=False)
        & sc["SHOT_ZONE_RANGE"].str.contains("24\\+|3PT", case=False, na=False)
    ].groupby("PLAYER_ID").size()
    loc = pd.concat([attempts, rim.rename("rim"), mid.rename("mid"), corner.rename("corner")], axis=1).fillna(0)
    loc["rim_freq"] = loc["rim"] / loc["shot_attempts"].replace(0, np.nan)
    loc["midrange_freq"] = loc["mid"] / loc["shot_attempts"].replace(0, np.nan)
    loc["corner3_freq"] = loc["corner"] / loc["shot_attempts"].replace(0, np.nan)
    loc = loc[["rim_freq", "midrange_freq", "corner3_freq"]].reset_index()
    merged = df[["PLAYER_ID"]].merge(loc, on="PLAYER_ID", how="left")
    for col in ("rim_freq", "midrange_freq", "corner3_freq"):
        df[col] = merged[col].fillna(0.0).to_numpy()


def add_normalized_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    raw_features = PLAYSTYLE_METRIC_KEYS
    for feature in raw_features:
        if feature not in out.columns:
            out[feature] = 0.0

    for season, idx in out.groupby("SEASON").groups.items():
        season_idx = list(idx)
        for feature in raw_features:
            values = pd.to_numeric(out.loc[season_idx, feature], errors="coerce").astype(float)
            mean = values.mean()
            std = values.std(ddof=0)
            z = (values - mean) / std if std and np.isfinite(std) else values * 0.0
            out.loc[season_idx, f"{feature}_z"] = z.fillna(0.0)
            out.loc[season_idx, f"{feature}_pctile"] = values.rank(pct=True).fillna(0.5) * 100.0

    for (season, pos), idx in out.groupby(["SEASON", "POSITION_GROUP"]).groups.items():
        pos_idx = list(idx)
        for feature in raw_features:
            values = pd.to_numeric(out.loc[pos_idx, feature], errors="coerce").astype(float)
            mean = values.mean()
            std = values.std(ddof=0)
            z = (values - mean) / std if std and np.isfinite(std) else values * 0.0
            out.loc[pos_idx, f"{feature}_pos_z"] = z.fillna(0.0)

    return out


def add_career_percentiles(career_df: pd.DataFrame) -> pd.DataFrame:
    out = career_df.copy()
    for feature in PLAYSTYLE_METRIC_KEYS:
        if feature not in out.columns:
            out[feature] = 0.0
        values = pd.to_numeric(out[feature], errors="coerce").astype(float)
        out[f"{feature}_career_pctile"] = values.rank(pct=True).fillna(0.5) * 100.0
    return out


def aggregate_career_features(season_df: pd.DataFrame) -> pd.DataFrame:
    if season_df.empty:
        return pd.DataFrame()

    df = season_df.copy()
    df["weight"] = np.maximum(_num(df["MIN"]), _num(df["GP"]) * 12.0)
    def clean_draft_val(val):
        if pd.isna(val) or val is None or val == "":
            return None
        val_str = str(val).strip()
        if val_str.lower() in ("undrafted", "nan", "none"):
            return "Undrafted"
        try:
            return int(float(val_str))
        except ValueError:
            return val_str

    def _mode_value_or_none(gp: pd.DataFrame, cols: tuple[str, ...]) -> Any:
        for col in cols:
            if col in gp.columns:
                vals = gp[col].dropna()
                if not vals.empty and vals.dtype == object:
                    vals = vals[vals.astype(str).str.strip() != ""]
                if not vals.empty:
                    v = vals.mode().iloc[0]
                    if pd.isna(v) or str(v).strip() == "":
                        return None
                    return v
        return None

    career_rows: list[dict[str, Any]] = []
    for pid, group in df.groupby("PLAYER_ID", sort=False):
        weights = _num(group["weight"]).to_numpy(dtype=float)
        if not np.isfinite(weights).any() or weights.sum() <= 0:
            weights = np.ones(len(group), dtype=float)
        total_games = int(_num(group["GP"]).sum())
        seasons = sorted(str(s) for s in group["SEASON"].unique())
        row: dict[str, Any] = {
            "player_id": int(pid),
            "player_name": str(group["PLAYER_NAME"].dropna().iloc[-1]),
            "career_games": total_games,
            "career_minutes": float(_num(group["MIN"]).sum()),
            "career_span": f"{seasons[0]} to {seasons[-1]}",
            "first_season": seasons[0],
            "last_season": seasons[-1],
            "last_season_start": season_start_year(seasons[-1]),
            "primary_position": _mode_value(group, ("PLAYER_POSITION_BIO", "POSITION", "POSITION_GROUP")),
            "position_group": _mode_value(group, ("POSITION_GROUP",)),
            "career_teams": _career_teams(group),
            "is_active": bool(group.get("ROSTERSTATUS_BIO", pd.Series(dtype=str)).astype(str).str.contains("Active", case=False).any()),
            "height": _mode_value_or_none(group, ("HEIGHT",)),
            "weight": clean_draft_val(_mode_value_or_none(group, ("WEIGHT",))),
            "draft_year": clean_draft_val(_mode_value_or_none(group, ("DRAFT_YEAR",))),
            "draft_position": clean_draft_val(_mode_value_or_none(group, ("DRAFT_NUMBER",))),
        }
        for feature in STYLE_FEATURES:
            values = _col(group, feature).to_numpy(dtype=float)
            row[feature] = float(np.average(values, weights=weights))
        for raw in PLAYSTYLE_METRIC_KEYS:
            values = _col(group, raw).to_numpy(dtype=float)
            row[raw] = float(np.average(values, weights=weights))
        career_rows.append(row)
    return pd.DataFrame(career_rows)


def _mode_value(group: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    for col in candidates:
        if col in group.columns:
            vals = group[col].dropna().astype(str)
            vals = vals[vals != ""]
            if not vals.empty:
                return str(vals.mode().iloc[0])
    return ""


def _career_teams(group: pd.DataFrame) -> list[str]:
    cols = [c for c in ("TEAM_ABBREVIATION", "SEASON") if c in group.columns]
    if not cols:
        return []
    teams = group[cols].drop_duplicates().sort_values("SEASON")
    out: list[str] = []
    for _, row in teams.iterrows():
        abbreviation = str(row.get("TEAM_ABBREVIATION", "")).strip()
        if abbreviation and (not out or out[-1] != abbreviation):
            out.append(abbreviation)
    return out


def apply_player_filter(
    career_df: pd.DataFrame,
    *,
    current_season_start: int,
    config: PlayerFilterConfig | None = None,
) -> pd.DataFrame:
    config = config or PlayerFilterConfig()
    recent_cutoff = current_season_start - config.recent_season_window + 1
    active = career_df["is_active"].astype(bool)
    recent = _num(career_df["last_season_start"]) >= recent_cutoff
    enough_games = _num(career_df["career_games"]) >= config.inactive_min_career_games
    return career_df[active | recent | enough_games].reset_index(drop=True)


def height_to_inches(height_val: Any) -> float:
    if pd.isna(height_val) or not height_val:
        return 0.0
    val_str = str(height_val).strip()
    if not val_str:
        return 0.0
    try:
        return float(val_str)
    except ValueError:
        pass

    for char in ('-', "'", '"', '/'):
        if char in val_str:
            parts = val_str.split(char)
            if len(parts) >= 2:
                try:
                    feet = float(parts[0].strip())
                    inches = float(parts[1].replace('"', '').strip())
                    return feet * 12.0 + inches
                except ValueError:
                    pass
    return 0.0


def clean_weight_to_float(weight_val: Any, default: float = 220.0) -> float:
    if pd.isna(weight_val) or not weight_val:
        return default
    val_str = str(weight_val).strip()
    try:
        return float(val_str)
    except ValueError:
        return default


def build_embedding_features(career_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    # We restrict the similarity engine to the 8 stable features (6 playstyle + height/weight)
    stable_features = [
        "reb_per36_z",
        "ast_per36_z",
        "blk_per36_z",
        "stl_per36_z",
        "fg3a_rate_z",
        "fta_rate_z",
    ]
    feature_cols = [c for c in stable_features if c in career_df.columns]

    if "height" in career_df.columns:
        heights = career_df["height"].apply(height_to_inches)
    else:
        heights = pd.Series(78.0, index=career_df.index)
    valid_heights = heights[heights > 0]
    mean_height = valid_heights.mean() if not valid_heights.empty else 78.0
    heights = heights.replace(0.0, mean_height)

    if "weight" in career_df.columns:
        weights = career_df["weight"].apply(lambda w: clean_weight_to_float(w, default=215.0))
    else:
        weights = pd.Series(215.0, index=career_df.index)
    valid_weights = weights[weights > 0]
    mean_weight = valid_weights.mean() if not valid_weights.empty else 215.0
    weights = weights.replace(0.0, mean_weight)

    X = career_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).copy()
    X["height_inches"] = heights
    X["weight_lbs"] = weights

    all_features = feature_cols + ["height_inches", "weight_lbs"]
    X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=all_features, index=career_df.index)
    
    # Apply calibrated similarity feature weights
    FEATURE_WEIGHTS = {
        "reb_per36_z": 1.7190,
        "ast_per36_z": 1.3206,
        "blk_per36_z": 1.3726,
        "stl_per36_z": 0.8896,
        "fg3a_rate_z": 3.1222,
        "fta_rate_z": 1.3506,
        "height_inches": 1.0926,
        "weight_lbs": 1.1863,
    }
    for col, w in FEATURE_WEIGHTS.items():
        if col in X_scaled.columns:
            X_scaled[col] *= w


    X_scaled.insert(0, "player_id", career_df["player_id"].astype(int).to_numpy())
    return X_scaled, all_features
