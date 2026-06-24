from __future__ import annotations

import json
import logging
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    DATA_STATIC_DIR,
    PLAYER_SEASON_FEATURES_PATH,
    PROSPECTS_JSON_PATH,
    PROSPECT_LINEUP_FEATURES_PATH,
)
from analytics.draft_year import (
    normalize_prospects_payload,
    season_for_draft_class_year,
)
from analytics.player_profiles.features import PLAYSTYLE_METRIC_KEYS

logger = logging.getLogger(__name__)

PROSPECT_ID_BASE = 900_000_000

COLLEGE_METRICS = [
    "pts_per36",
    "reb_per36",
    "ast_per36",
    "blk_per36",
    "stl_per36",
    "ts_pct",
    "efg_pct",
    "fg3a_rate",
    "fta_rate",
    "mpg",
]
FEATURE_COLS = COLLEGE_METRICS + ["fg3_pct_college", "ast_pct_college", "height_in", "weight"]
TARGET_COLS = list(PLAYSTYLE_METRIC_KEYS)
ALL_TARGETS = TARGET_COLS + ["fg3_pct"]
TARGET_SOURCE = {"fg3_pct": "FG3_PCT"}

CLIP = {
    "pts_per36": (0.0, 40.0),
    "reb_per36": (0.0, 20.0),
    "ast_per36": (0.0, 14.0),
    "blk_per36": (0.0, 5.0),
    "stl_per36": (0.0, 4.0),
    "tov_per36": (0.0, 6.0),
    "fg3a_rate": (0.0, 0.95),
    "fta_rate": (0.0, 1.2),
    "ts_pct": (0.35, 0.75),
    "efg_pct": (0.3, 0.75),
    "ast_pct": (0.0, 60.0),
    "mpg": (8.0, 38.0),
    "fg3_pct": (0.20, 0.45),
}

ROLE_TO_POSITION_GROUP = {
    "Playmaker": "G",
    "Secondary Creator": "G",
    "Perimeter Specialist": "W",
    "Designated Scorer": "W",
    "Rim Attacker": "W",
    "Defensive Specialist": "W",
    "Interior Presence": "B",
}


def _norm_name(name: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(name))
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", stripped.lower()).strip()


def _height_inches(value) -> float:
    text = str(value or "").strip()
    if "-" in text:
        parts = text.split("-")
        try:
            return float(parts[0]) * 12.0 + float(parts[1])
        except (ValueError, IndexError):
            pass
    try:
        return float(text)
    except ValueError:
        return 78.0


def _metric_value(prospect: dict, key: str) -> float:
    metric = prospect.get(key)
    if isinstance(metric, dict):
        try:
            return float(metric.get("value", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0
    try:
        return float(metric or 0.0)
    except (TypeError, ValueError):
        return 0.0


FEATURE_CLIP = {
    "pts_per36": (0.0, 40.0),
    "reb_per36": (0.0, 22.0),
    "ast_per36": (0.0, 16.0),
    "blk_per36": (0.0, 7.0),
    "stl_per36": (0.0, 5.0),
    "ts_pct": (0.30, 0.80),
    "efg_pct": (0.25, 0.80),
    "fg3a_rate": (0.0, 0.95),
    "fta_rate": (0.0, 1.4),
    "mpg": (5.0, 40.0),
    "ast_pct_college": (0.0, 1.6),
    "fg3_pct_college": (0.0, 0.60),
    "height_in": (66.0, 90.0),
    "weight": (150.0, 320.0),
}

ROOKIE_MIN_MINUTES = 300.0


def _college_features(prospect: dict) -> dict:
    raw = prospect.get("raw_stats", {}) or {}
    apg = float(raw.get("apg", 0.0) or 0.0)
    fgm = float(raw.get("fgm", 0.0) or 0.0)
    feats = {m: _metric_value(prospect, m) for m in COLLEGE_METRICS}
    feats["ast_pct_college"] = (apg / fgm) if fgm > 0 else 0.0
    feats["fg3_pct_college"] = float(raw.get("fg3_pct", 0.0) or 0.0)
    feats["height_in"] = _height_inches(prospect.get("height"))
    feats["weight"] = float(prospect.get("weight", 0.0) or 0.0) or 215.0
    for key, (low, high) in FEATURE_CLIP.items():
        feats[key] = min(high, max(low, feats[key]))
    return feats


def _read_features() -> pd.DataFrame:
    try:
        return pd.read_parquet(PLAYER_SEASON_FEATURES_PATH)
    except Exception:
        return pd.read_parquet(PLAYER_SEASON_FEATURES_PATH, engine="fastparquet")


def _rookie_index(features: pd.DataFrame) -> dict:
    rookies = (
        features.sort_values("SEASON_START")
        .groupby("PLAYER_ID", as_index=False)
        .first()
    )
    index = {}
    for _, row in rookies.iterrows():
        index.setdefault(_norm_name(row["PLAYER_NAME"]), row)
    return index


def _prospect_classes() -> list[tuple[int, list[dict]]]:
    classes: list[tuple[int, list[dict]]] = []
    if PROSPECTS_JSON_PATH.exists():
        records, meta = normalize_prospects_payload(
            json.loads(PROSPECTS_JSON_PATH.read_text(encoding="utf-8"))
        )
        if meta.get("draft_class_year"):
            classes.append((int(meta["draft_class_year"]), records))
    draft_dir = DATA_STATIC_DIR / "draft"
    if draft_dir.exists():
        for path in sorted(draft_dir.glob("prospects_*.json")):
            try:
                year = int(path.stem.split("_")[-1])
            except ValueError:
                continue
            records, _ = normalize_prospects_payload(
                json.loads(path.read_text(encoding="utf-8"))
            )
            classes.append((year, records))
    return classes


def _fit_model(features: pd.DataFrame, classes: list[tuple[int, list[dict]]]) -> Pipeline:
    rookie_index = _rookie_index(features)
    rows_x, rows_y = [], []
    for _year, records in classes:
        for prospect in records:
            rookie = rookie_index.get(_norm_name(prospect.get("player_name", "")))
            if rookie is None:
                continue
            if float(rookie.get("MIN", 0.0) or 0.0) < ROOKIE_MIN_MINUTES:
                continue
            feats = _college_features(prospect)
            target = [float(rookie.get(TARGET_SOURCE.get(t, t), 0.0) or 0.0) for t in ALL_TARGETS]
            rows_x.append([feats[c] for c in FEATURE_COLS])
            rows_y.append(target)

    X = np.array(rows_x, dtype=float)
    Y = np.array(rows_y, dtype=float)

    model = Pipeline([
        ("scale", StandardScaler()),
        ("reg", MultiOutputRegressor(RidgeCV(alphas=np.logspace(-1, 3.5, 19)))),
    ])

    preds = cross_val_predict(model, X, Y, cv=KFold(n_splits=5, shuffle=True, random_state=7))
    print(f"  College -> NBA rookie model trained on {len(X)} prospect/player pairs.")
    print("  5-fold verification (predicted vs actual rookie metrics):")
    for i, target in enumerate(ALL_TARGETS):
        r2 = r2_score(Y[:, i], preds[:, i])
        mae = mean_absolute_error(Y[:, i], preds[:, i])
        print(f"    {target:<11} R2={r2:+.2f}  MAE={mae:.2f}")
    print(f"    overall      R2={r2_score(Y, preds):+.2f}")

    model.fit(X, Y)
    return model


def _season_norm_stats(features: pd.DataFrame, season: str) -> dict:
    sub = features[features["SEASON"] == season]
    if sub.empty:
        sub = features
    stats = {}
    for metric in ALL_TARGETS:
        column = TARGET_SOURCE.get(metric, metric)
        vals = pd.to_numeric(sub[column], errors="coerce").dropna().astype(float)
        arr = np.sort(vals.to_numpy())
        std = float(vals.std(ddof=0))
        stats[metric] = (float(vals.mean()), std if std > 1e-9 else 1.0, arr)
    return stats


def _percentile(arr: np.ndarray, value: float) -> float:
    if arr.size == 0:
        return 50.0
    return float(np.searchsorted(arr, value, side="right") / arr.size * 100.0)


def _raw_prediction(model: Pipeline, prospect: dict) -> dict:
    feats = _college_features(prospect)
    raw = model.predict(np.array([[feats[c] for c in FEATURE_COLS]], dtype=float))[0]
    return dict(zip(ALL_TARGETS, (float(v) for v in raw)))


def _calibrate(raw: dict, cohort_stats: dict, norm_stats: dict) -> dict:
    out = {}
    for metric in ALL_TARGETS:
        cohort_mean, cohort_std = cohort_stats[metric]
        league_mean, league_std, _arr = norm_stats[metric]
        if cohort_std > 1e-9:
            value = cohort_mean + (raw[metric] - cohort_mean) * (league_std / cohort_std)
        else:
            value = cohort_mean
        low, high = CLIP[metric]
        out[metric] = float(min(high, max(low, value)))
    return out


def _assemble_row(
    prospect: dict,
    predicted: dict,
    season: str,
    year: int,
    player_id: int,
    feature_columns: list[str],
    norm_stats: dict,
) -> dict:
    row = {col: 0.0 for col in feature_columns}

    mpg = predicted["mpg"]
    gp = 70.0
    minutes = mpg * gp

    def total(per36: float) -> float:
        return per36 * minutes / 36.0

    ts = predicted["ts_pct"]
    fta_rate = predicted["fta_rate"]
    fga36 = predicted["pts_per36"] / (2.0 * ts * (1.0 + 0.44 * fta_rate)) if ts > 0.2 else predicted["pts_per36"] / 2.0
    fg3a36 = predicted["fg3a_rate"] * fga36
    fta36 = fta_rate * fga36
    fg3m36 = predicted["fg3_pct"] * fg3a36
    fgm36 = max(0.0, predicted["efg_pct"] * fga36 - 0.5 * fg3m36)
    ftm36 = 0.75 * fta36
    reb36 = predicted["reb_per36"]

    row.update({
        "GP": gp,
        "MIN": minutes,
        "PTS": total(predicted["pts_per36"]),
        "AST": total(predicted["ast_per36"]),
        "REB": total(reb36),
        "OREB": total(0.25 * reb36),
        "DREB": total(0.75 * reb36),
        "STL": total(predicted["stl_per36"]),
        "BLK": total(predicted["blk_per36"]),
        "TOV": total(predicted["tov_per36"]),
        "FGA": total(fga36),
        "FGM": total(fgm36),
        "FG3A": total(fg3a36),
        "FG3M": total(fg3m36),
        "FTA": total(fta36),
        "FTM": total(ftm36),
        "PF": total(2.0),
        "FG_PCT": fgm36 / fga36 if fga36 > 0 else 0.0,
        "FG3_PCT": fg3m36 / fg3a36 if fg3a36 > 0 else 0.0,
        "FT_PCT": ftm36 / fta36 if fta36 > 0 else 0.0,
    })

    for metric in TARGET_COLS:
        value = predicted[metric]
        mean, std, arr = norm_stats[metric]
        row[metric] = value
        row[f"{metric}_z"] = (value - mean) / std
        row[f"{metric}_pctile"] = _percentile(arr, value)
        row[f"{metric}_pos_z"] = (value - mean) / std

    row.update({
        "PLAYER_ID": player_id,
        "PLAYER_NAME": str(prospect.get("player_name", prospect.get("prospect_id", ""))),
        "TEAM_ABBREVIATION": "DRAFT",
        "SEASON": season,
        "SEASON_START": int(season[:4]),
        "POSITION_GROUP": ROLE_TO_POSITION_GROUP.get(str(prospect.get("role", "")), "W"),
        "POSITION": str(prospect.get("role", "")),
        "HEIGHT": str(prospect.get("height", "") or ""),
        "WEIGHT": float(prospect.get("weight", 0.0) or 0.0) or 215.0,
        "DRAFT_YEAR": year,
        "DRAFT_NUMBER": int(prospect.get("pick") or 0),
    })

    counterparts = prospect.get("similar_nba_players") or []
    row["is_prospect"] = True
    row["prospect_id"] = str(prospect.get("prospect_id", ""))
    row["draft_class_year"] = year
    row["counterpart_id"] = int(counterparts[0]["player_id"]) if counterparts else 0
    row["counterpart_name"] = str(counterparts[0].get("player_name", "")) if counterparts else ""
    return row


def build_prospect_lineup_features() -> pd.DataFrame:
    features = _read_features()
    feature_columns = list(features.columns)
    classes = _prospect_classes()
    if not classes:
        print("✓ Prospect lineup features: no prospect classes found.")
        return pd.DataFrame()

    model = _fit_model(features, classes)

    entries: list[tuple[int, str, dict, dict]] = []
    seen: set[tuple[str, str]] = set()
    for year, records in classes:
        season = season_for_draft_class_year(year)
        for prospect in records:
            slug = prospect.get("prospect_id")
            if not slug or (slug, season) in seen:
                continue
            entries.append((year, season, prospect, _raw_prediction(model, prospect)))
            seen.add((slug, season))

    cohort_stats: dict[str, dict] = {}
    for season in {s for _, s, _, _ in entries}:
        raws = [raw for _, s, _, raw in entries if s == season]
        cohort_stats[season] = {
            metric: (
                float(np.mean([r[metric] for r in raws])),
                float(np.std([r[metric] for r in raws])),
            )
            for metric in ALL_TARGETS
        }

    norm_cache: dict[str, dict] = {}
    rows: list[dict] = []
    next_id = PROSPECT_ID_BASE
    for year, season, prospect, raw in entries:
        if season not in norm_cache:
            norm_cache[season] = _season_norm_stats(features, season)
        calibrated = _calibrate(raw, cohort_stats[season], norm_cache[season])
        rows.append(
            _assemble_row(
                prospect, calibrated, season, year, next_id, feature_columns, norm_cache[season]
            )
        )
        next_id += 1

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.astype({"PLAYER_ID": "int64", "counterpart_id": "int64"})

    PROSPECT_LINEUP_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(PROSPECT_LINEUP_FEATURES_PATH, index=False, engine="fastparquet")
    logger.info("Wrote %d prospect lineup feature rows.", len(frame))
    print(f"✓ Prospect lineup features: {len(frame)} rows (college-projected).")
    return frame


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_prospect_lineup_features()
