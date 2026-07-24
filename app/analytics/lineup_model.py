from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupKFold

MODEL_VERSION = "quality_fit_v3"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "data" / "static" / "lineup_synergy_model.json"

QUALITY_FEATURES = [
    "talent_mean",
    "talent_top_two",
    "talent_floor",
    "scoring_mean",
    "scoring_peak",
    "efficiency_mean",
    "efficiency_floor",
    "playmaking_mean",
    "playmaking_peak",
    "rebounding_mean",
    "defense_mean",
    "availability_mean",
    "availability_floor",
]

FIT_FEATURES = [
    "spacing_coverage",
    "spacing_floor",
    "creation_peak",
    "creation_support",
    "scoring_support",
    "perimeter_defense",
    "rim_protection",
    "rebounding_support",
    "two_way_coverage",
    "role_coverage",
    "role_entropy",
    "role_concentration",
    "skill_redundancy",
    "skill_diversity",
    "guard_share",
    "wing_share",
    "big_share",
    "style_spacing",
    "style_paint",
    "style_playmaking",
    "style_defense",
    "style_rebounding",
    "pace_extremity",
    "inside_out_balance",
    "creation_spacing",
    "defense_rebounding",
]

OUTCOME_FEATURES = QUALITY_FEATURES + FIT_FEATURES

FIT_FEATURE_GROUPS = {
    "spacing_coverage": "spacing",
    "spacing_floor": "spacing",
    "creation_peak": "playmaking",
    "creation_support": "playmaking",
    "scoring_support": "playmaking",
    "perimeter_defense": "defense",
    "rim_protection": "defense",
    "rebounding_support": "interior",
    "two_way_coverage": "defense",
    "role_coverage": "overlap",
    "role_entropy": "overlap",
    "role_concentration": "overlap",
    "skill_redundancy": "overlap",
    "skill_diversity": "overlap",
    "guard_share": "overlap",
    "wing_share": "overlap",
    "big_share": "interior",
    "style_spacing": "spacing",
    "style_paint": "interior",
    "style_playmaking": "playmaking",
    "style_defense": "defense",
    "style_rebounding": "interior",
    "pace_extremity": "overlap",
    "inside_out_balance": "spacing",
    "creation_spacing": "playmaking",
    "defense_rebounding": "defense",
}


def _number(value: Any, default: float = 50.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _percentiles(rows: pd.DataFrame, column: str) -> np.ndarray:
    values = rows[column] if column in rows else pd.Series([50.0] * len(rows), index=rows.index)
    return np.asarray([_number(value) for value in values], dtype=float)


def _top(values: np.ndarray, rank: int = 1) -> float:
    if values.size == 0:
        return 50.0
    ordered = np.sort(values)
    return float(ordered[max(0, len(ordered) - rank)])


def _entropy(counts: list[int]) -> float:
    values = np.asarray([value for value in counts if value > 0], dtype=float)
    if values.size <= 1:
        return 0.0
    probabilities = values / values.sum()
    return float(-np.sum(probabilities * np.log(probabilities)) / np.log(7.0))


def _pair_geometry(matrix: np.ndarray) -> tuple[float, float]:
    similarities = []
    distances = []
    for left in range(len(matrix)):
        for right in range(left + 1, len(matrix)):
            a = matrix[left]
            b = matrix[right]
            denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
            similarities.append(float(np.dot(a, b) / denominator) if denominator else 0.0)
            distances.append(float(np.linalg.norm(a - b) / np.sqrt(matrix.shape[1])))
    return float(np.mean(similarities)), float(np.mean(distances))


def build_lineup_model_features(
    lineup_rows: pd.DataFrame,
    player_roles: dict[int, list[str]],
    style_percentiles: dict[str, float],
) -> dict[str, float]:
    if len(lineup_rows) != 5:
        raise ValueError("Lineup model features require exactly five player rows")
    pts = _percentiles(lineup_rows, "pts_per36_pctile")
    ast = _percentiles(lineup_rows, "ast_per36_pctile")
    ast_pct = _percentiles(lineup_rows, "ast_pct_pctile")
    reb = _percentiles(lineup_rows, "reb_per36_pctile")
    stl = _percentiles(lineup_rows, "stl_per36_pctile")
    blk = _percentiles(lineup_rows, "blk_per36_pctile")
    shooting = _percentiles(lineup_rows, "fg3a_rate_pctile")
    rim = _percentiles(lineup_rows, "fta_rate_pctile")
    efficiency = _percentiles(lineup_rows, "ts_pct_pctile")
    availability = _percentiles(lineup_rows, "mpg_pctile")
    creation = 0.55 * ast + 0.45 * ast_pct
    defense = np.maximum(stl, blk)
    talent = (
        0.25 * pts
        + 0.17 * creation
        + 0.13 * reb
        + 0.12 * defense
        + 0.18 * efficiency
        + 0.15 * availability
    )
    skill_matrix = np.column_stack([pts, creation, shooting, rim, reb, stl, blk, efficiency]) / 100.0
    redundancy, diversity = _pair_geometry(skill_matrix)
    role_counts: dict[str, int] = {}
    for roles in player_roles.values():
        for role in set(roles):
            role_counts[role] = role_counts.get(role, 0) + 1
    position_counts = lineup_rows.get("POSITION_GROUP", pd.Series(["W"] * len(lineup_rows))).value_counts()
    style = {key: _number(value) / 100.0 for key, value in style_percentiles.items()}
    spacing = style.get("spacing", 0.5)
    paint = style.get("paint", 0.5)
    playmaking = style.get("playmaking", 0.5)
    lineup_defense = style.get("defense", 0.5)
    rebounding = style.get("rebounding", 0.5)
    pace = style.get("pace", 0.5)
    features = {
        "talent_mean": float(np.mean(talent)),
        "talent_top_two": float(np.mean(np.sort(talent)[-2:])),
        "talent_floor": float(np.min(talent)),
        "scoring_mean": float(np.mean(pts)),
        "scoring_peak": float(np.max(pts)),
        "efficiency_mean": float(np.mean(efficiency)),
        "efficiency_floor": float(np.min(efficiency)),
        "playmaking_mean": float(np.mean(creation)),
        "playmaking_peak": float(np.max(creation)),
        "rebounding_mean": float(np.mean(reb)),
        "defense_mean": float(np.mean(defense)),
        "availability_mean": float(np.mean(availability)),
        "availability_floor": float(np.min(availability)),
        "spacing_coverage": float(np.mean(shooting >= 60.0)),
        "spacing_floor": _top(shooting, 4) / 100.0,
        "creation_peak": float(np.max(creation)) / 100.0,
        "creation_support": _top(creation, 2) / 100.0,
        "scoring_support": _top(pts, 2) / 100.0,
        "perimeter_defense": _top(stl, 2) / 100.0,
        "rim_protection": float(np.max(blk)) / 100.0,
        "rebounding_support": _top(reb, 2) / 100.0,
        "two_way_coverage": float(np.mean((pts + efficiency >= 120.0) & (defense >= 60.0))),
        "role_coverage": len(role_counts) / 7.0,
        "role_entropy": _entropy(list(role_counts.values())),
        "role_concentration": max(role_counts.values(), default=0) / 5.0,
        "skill_redundancy": redundancy,
        "skill_diversity": diversity,
        "guard_share": float(position_counts.get("G", 0)) / 5.0,
        "wing_share": float(position_counts.get("W", 0)) / 5.0,
        "big_share": float(position_counts.get("B", 0)) / 5.0,
        "style_spacing": spacing,
        "style_paint": paint,
        "style_playmaking": playmaking,
        "style_defense": lineup_defense,
        "style_rebounding": rebounding,
        "pace_extremity": abs(pace - 0.5) * 2.0,
        "inside_out_balance": 1.0 - abs(spacing - paint),
        "creation_spacing": float(np.sqrt(max(0.0, playmaking * spacing))),
        "defense_rebounding": float(np.sqrt(max(0.0, lineup_defense * rebounding))),
    }
    return features


def _matrix(rows: list[dict[str, float]], names: list[str]) -> np.ndarray:
    return np.asarray([[row[name] for name in names] for row in rows], dtype=float)


def _fit_ridge(
    matrix: np.ndarray,
    target: np.ndarray,
    sample_weight: np.ndarray,
    alpha: float,
) -> dict[str, Any]:
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (matrix - mean) / scale
    estimator = Ridge(alpha=alpha, solver="lsqr")
    estimator.fit(standardized, target, sample_weight=sample_weight)
    return {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficient": estimator.coef_.tolist(),
        "intercept": float(estimator.intercept_),
        "alpha": float(alpha),
    }


def _predict(model: dict[str, Any], matrix: np.ndarray) -> np.ndarray:
    mean = np.asarray(model["mean"], dtype=float)
    scale = np.asarray(model["scale"], dtype=float)
    coefficient = np.asarray(model["coefficient"], dtype=float)
    standardized = (matrix - mean) / scale
    return np.sum(standardized * coefficient, axis=1) + float(model["intercept"])


def _export_tree(estimator: Any) -> dict[str, Any]:
    tree = estimator.tree_
    return {
        "children_left": tree.children_left.tolist(),
        "children_right": tree.children_right.tolist(),
        "feature": tree.feature.tolist(),
        "threshold": tree.threshold.tolist(),
        "value": tree.value[:, 0, 0].tolist(),
    }


def _fit_gradient_boosting(
    matrix: np.ndarray,
    target: np.ndarray,
    sample_weight: np.ndarray,
) -> dict[str, Any]:
    estimator = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=3,
        min_samples_leaf=20,
        loss="huber",
        random_state=42,
    )
    estimator.fit(matrix, target, sample_weight=sample_weight)
    return {
        "initial": float(estimator.init_.constant_[0]),
        "learning_rate": float(estimator.learning_rate),
        "trees": [_export_tree(tree[0]) for tree in estimator.estimators_],
    }


def _predict_tree(tree: dict[str, Any], row: np.ndarray) -> float:
    node = 0
    while tree["children_left"][node] != tree["children_right"][node]:
        feature = tree["feature"][node]
        node = (
            tree["children_left"][node]
            if row[feature] <= tree["threshold"][node]
            else tree["children_right"][node]
        )
    return float(tree["value"][node])


def _predict_gradient_boosting(model: dict[str, Any], matrix: np.ndarray) -> np.ndarray:
    predictions = np.full(len(matrix), float(model["initial"]), dtype=float)
    learning_rate = float(model["learning_rate"])
    for tree in model["trees"]:
        predictions += learning_rate * np.asarray(
            [_predict_tree(tree, row) for row in matrix],
            dtype=float,
        )
    return predictions


def _cross_validated_gradient_boosting(
    matrix: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    sample_weight: np.ndarray,
) -> np.ndarray:
    splitter = GroupKFold(n_splits=min(6, len(np.unique(groups))))
    predictions = np.zeros_like(target)
    for train, test in splitter.split(matrix, target, groups):
        model = _fit_gradient_boosting(
            matrix[train],
            target[train],
            sample_weight[train],
        )
        predictions[test] = _predict_gradient_boosting(model, matrix[test])
    return predictions


def _cross_validated_ridge(
    matrix: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    sample_weight: np.ndarray,
    alphas: tuple[float, ...],
) -> tuple[float, np.ndarray]:
    unique_groups = np.unique(groups)
    splits = min(6, len(unique_groups))
    splitter = GroupKFold(n_splits=splits)
    best_alpha = alphas[0]
    best_predictions = np.zeros_like(target)
    best_error = np.inf
    for alpha in alphas:
        predictions = np.zeros_like(target)
        for train, test in splitter.split(matrix, target, groups):
            model = _fit_ridge(matrix[train], target[train], sample_weight[train], alpha)
            predictions[test] = _predict(model, matrix[test])
        error = float(np.sqrt(mean_squared_error(target, predictions)))
        if error < best_error:
            best_error = error
            best_alpha = alpha
            best_predictions = predictions
    return best_alpha, best_predictions


def _cross_validated_residual_fit(
    quality_matrix: np.ndarray,
    fit_matrix: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    sample_weight: np.ndarray,
    quality_alpha: float,
    fit_alphas: tuple[float, ...],
) -> tuple[float, np.ndarray, np.ndarray]:
    outer = GroupKFold(n_splits=min(6, len(np.unique(groups))))
    best_alpha = fit_alphas[0]
    best_outcome = np.zeros_like(target)
    best_fit = np.zeros_like(target)
    best_error = np.inf
    for fit_alpha in fit_alphas:
        outcome_predictions = np.zeros_like(target)
        fit_predictions = np.zeros_like(target)
        for train, test in outer.split(quality_matrix, target, groups):
            quality_model = _fit_ridge(
                quality_matrix[train],
                target[train],
                sample_weight[train],
                quality_alpha,
            )
            quality_test = _predict(quality_model, quality_matrix[test])
            train_groups = groups[train]
            inner = GroupKFold(n_splits=min(5, len(np.unique(train_groups))))
            quality_train_oof = np.zeros(len(train), dtype=float)
            for inner_train, inner_test in inner.split(
                quality_matrix[train],
                target[train],
                train_groups,
            ):
                inner_model = _fit_ridge(
                    quality_matrix[train][inner_train],
                    target[train][inner_train],
                    sample_weight[train][inner_train],
                    quality_alpha,
                )
                quality_train_oof[inner_test] = _predict(
                    inner_model,
                    quality_matrix[train][inner_test],
                )
            residual_target = target[train] - quality_train_oof
            fit_model = _fit_ridge(
                fit_matrix[train],
                residual_target,
                sample_weight[train],
                fit_alpha,
            )
            fit_test = _predict(fit_model, fit_matrix[test])
            fit_predictions[test] = fit_test
            outcome_predictions[test] = quality_test + fit_test
        error = float(np.sqrt(mean_squared_error(target, outcome_predictions)))
        if error < best_error:
            best_error = error
            best_alpha = fit_alpha
            best_outcome = outcome_predictions
            best_fit = fit_predictions
    return best_alpha, best_outcome, best_fit


def train_lineup_model(
    feature_rows: list[dict[str, float]],
    targets: list[float],
    seasons: list[str],
) -> dict[str, Any]:
    quality_matrix = _matrix(feature_rows, QUALITY_FEATURES)
    fit_matrix = _matrix(feature_rows, FIT_FEATURES)
    target = np.asarray(targets, dtype=float)
    groups = np.asarray(seasons)
    sample_weight = np.where((target >= 0.55) | (target <= 0.45), 1.5, 0.75)
    alphas = (1.0, 3.0, 10.0, 30.0, 100.0)
    quality_alpha, quality_oof = _cross_validated_ridge(
        quality_matrix, target, groups, sample_weight, alphas
    )
    fit_alphas = (3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
    fit_alpha, outcome_oof, _ = _cross_validated_residual_fit(
        quality_matrix,
        fit_matrix,
        target,
        groups,
        sample_weight,
        quality_alpha,
        fit_alphas,
    )
    outcome_matrix = _matrix(feature_rows, OUTCOME_FEATURES)
    nonlinear_oof = _cross_validated_gradient_boosting(
        outcome_matrix,
        target,
        groups,
        sample_weight,
    )
    blend_candidates = (0.25, 0.5, 0.75)
    nonlinear_weight = min(
        blend_candidates,
        key=lambda weight: mean_squared_error(
            target,
            (1.0 - weight) * outcome_oof + weight * nonlinear_oof,
        ),
    )
    blended_oof = (
        (1.0 - nonlinear_weight) * outcome_oof
        + nonlinear_weight * nonlinear_oof
    )
    quality_model = _fit_ridge(quality_matrix, target, sample_weight, quality_alpha)
    residual_target = target - quality_oof
    fit_model = _fit_ridge(
        fit_matrix,
        residual_target,
        sample_weight,
        fit_alpha,
    )
    quality_full = _predict(quality_model, quality_matrix)
    fit_full = _predict(fit_model, fit_matrix)
    nonlinear_model = _fit_gradient_boosting(
        outcome_matrix,
        target,
        sample_weight,
    )
    nonlinear_full = _predict_gradient_boosting(nonlinear_model, outcome_matrix)
    outcome_full = (
        (1.0 - nonlinear_weight) * (quality_full + fit_full)
        + nonlinear_weight * nonlinear_full
    )
    tail_mask = (target >= 0.55) | (target <= 0.45)
    tail_target = (target[tail_mask] >= 0.55).astype(int)
    tail_prediction = blended_oof[tail_mask]
    residuals = target - blended_oof
    residual_low = float(np.quantile(residuals, 0.10))
    residual_high = float(np.quantile(residuals, 0.90))
    interval_coverage = float(
        np.mean((residuals >= residual_low) & (residuals <= residual_high))
    )
    outcome_rmse = float(np.sqrt(mean_squared_error(target, blended_oof)))
    quality_rmse = float(np.sqrt(mean_squared_error(target, quality_oof)))
    validation = {
        "rmse": round(outcome_rmse, 4),
        "mae": round(float(mean_absolute_error(target, blended_oof)), 4),
        "quality_only_rmse": round(quality_rmse, 4),
        "fit_rmse_improvement": round(quality_rmse - outcome_rmse, 4),
        "tail_auc": round(float(roc_auc_score(tail_target, tail_prediction)), 4),
        "residual_p10": round(residual_low, 4),
        "residual_p90": round(residual_high, 4),
        "interval_coverage": round(interval_coverage, 3),
    }
    return {
        "version": MODEL_VERSION,
        "training": {
            "lineups": len(targets),
            "successful": int(np.sum(target >= 0.55)),
            "unsuccessful": int(np.sum(target <= 0.45)),
            "seasons": len(set(seasons)),
            "first_season": min(seasons),
            "last_season": max(seasons),
            "target_mean": float(np.mean(target)),
        },
        "validation": validation,
        "quality": {
            "features": QUALITY_FEATURES,
            "model": quality_model,
            "reference": sorted(round(float(value), 6) for value in quality_full),
        },
        "fit": {
            "features": FIT_FEATURES,
            "groups": FIT_FEATURE_GROUPS,
            "model": fit_model,
            "reference": sorted(round(float(value), 6) for value in fit_full),
        },
        "outcome": {
            "features": OUTCOME_FEATURES,
            "linear_weight": round(1.0 - nonlinear_weight, 2),
            "nonlinear_weight": round(nonlinear_weight, 2),
            "nonlinear_model": nonlinear_model,
        },
        "outcome_reference": sorted(round(float(value), 6) for value in outcome_full),
    }


def _rank(reference: list[float], value: float) -> float:
    values = np.asarray(reference, dtype=float)
    left = np.searchsorted(values, value, side="left")
    right = np.searchsorted(values, value, side="right")
    return float(100.0 * (left + right) / (2.0 * len(values))) if len(values) else 50.0


def score_lineup_model(features: dict[str, float], artifact: dict[str, Any]) -> dict[str, Any]:
    if artifact.get("version") != MODEL_VERSION:
        raise ValueError(f"Unsupported lineup model version: {artifact.get('version')}")
    quality_names = artifact["quality"]["features"]
    fit_names = artifact["fit"]["features"]
    quality_matrix = _matrix([features], quality_names)
    fit_matrix = _matrix([features], fit_names)
    outcome_matrix = _matrix([features], artifact["outcome"]["features"])
    quality_prediction = float(_predict(artifact["quality"]["model"], quality_matrix)[0])
    fit_prediction = float(_predict(artifact["fit"]["model"], fit_matrix)[0])
    linear_prediction = quality_prediction + fit_prediction
    nonlinear_prediction = float(
        _predict_gradient_boosting(
            artifact["outcome"]["nonlinear_model"],
            outcome_matrix,
        )[0]
    )
    outcome_prediction = (
        float(artifact["outcome"]["linear_weight"]) * linear_prediction
        + float(artifact["outcome"]["nonlinear_weight"]) * nonlinear_prediction
    )
    projected_win_pct = float(np.clip(outcome_prediction, 0.10, 0.90))
    fit_model = artifact["fit"]["model"]
    standardized = (
        fit_matrix[0] - np.asarray(fit_model["mean"], dtype=float)
    ) / np.asarray(fit_model["scale"], dtype=float)
    contributions = standardized * np.asarray(fit_model["coefficient"], dtype=float)
    contribution_by_name = dict(zip(fit_names, contributions))
    channels = {name: 0.0 for name in ("playmaking", "spacing", "interior", "defense", "overlap")}
    for name in fit_names:
        channels[artifact["fit"]["groups"][name]] += float(contribution_by_name[name]) * 100.0
    validation = artifact.get("validation", {})
    projected_low = float(
        np.clip(projected_win_pct + float(validation.get("residual_p10", -0.14)), 0.10, 0.90)
    )
    projected_high = float(
        np.clip(projected_win_pct + float(validation.get("residual_p90", 0.14)), 0.10, 0.90)
    )
    training = artifact.get("training", {})
    return {
        "synergy_score": round(_rank(artifact["outcome_reference"], projected_win_pct), 1),
        "quality_score": round(_rank(artifact["quality"]["reference"], quality_prediction), 1),
        "fit_score": round(_rank(artifact["fit"]["reference"], fit_prediction), 1),
        "projected_win_pct": round(projected_win_pct, 3),
        "projected_win_range": {
            "low": round(projected_low, 3),
            "high": round(projected_high, 3),
        },
        "fit_delta_win_pct": round(fit_prediction, 3),
        "model_context": {
            "comparison_lineups": int(training.get("lineups", 0)),
            "seasons": int(training.get("seasons", 0)),
            "first_season": str(training.get("first_season", "")),
            "last_season": str(training.get("last_season", "")),
            "validation_mae": float(validation.get("mae", 0.0)),
            "interval_coverage": float(validation.get("interval_coverage", 0.8)),
        },
        "quality_prediction": quality_prediction,
        "fit_prediction": fit_prediction,
        "synergy_breakdown": {key: round(value, 1) for key, value in channels.items()},
    }


def save_lineup_model(artifact: dict[str, Any], path: str | Path = DEFAULT_MODEL_PATH) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, separators=(",", ":")))
    load_lineup_model.cache_clear()


@lru_cache(maxsize=4)
def load_lineup_model(path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
