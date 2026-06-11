#!/usr/bin/env python3
"""Bounded weight calibration for the prospect -> NBA similarity engine.

You give per-column floors and ceilings (in ``weight_bounds.json``); this
script searches that box for the weight vector that maximises the **points
metric** on the historical draft classes (2007-2023), then reports how the
winning config compares to the currently-shipped weights and appends the run
to ``weights_history.json`` so every calibration is documented.

The search runs through the *production* code path
(``ProspectsPipeline.compute_similarity_matrix``), so the weights printed here
behave identically when pasted into ``app/pipelines/prospects_pipeline.py``.

Points metric (per prospect, evaluated on the RAW top-7 before name masking):
  +2  if the prospect's own NBA counterpart is in the top-7 similarities.
  +1  for each NBA player in the top-7 whose own top-7 NBA-NBA similars
      (player_profiles.json) contain that counterpart.  Max 9 / prospect.

Quick start:
    python3 app/calibration/calibrate_weights.py            # evaluate baseline only
    python3 app/calibration/calibrate_weights.py --search   # calibrate within bounds

See README.md in this directory for the full workflow.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_APP_ROOT = Path(__file__).resolve().parents[1]
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

from config import DATA_STATIC_DIR, PLAYER_CAREER_FEATURES_PATH, PLAYER_PROFILES_PATH
from pipelines.prospects_pipeline import (
    SIMILARITY_COMP_BANDWIDTH,
    SIMILARITY_COMP_FEATURES,
    SIMILARITY_COMP_SMOOTH_LAMBDA,
    SIMILARITY_COMP_WEIGHTS,
    ProspectsPipeline,
    _era_bucket,
    build_nba_neighbor_index,
)

HERE = Path(__file__).resolve().parent
DEFAULT_BOUNDS_PATH = HERE / "weight_bounds.json"
HISTORY_PATH = HERE / "weights_history.json"

TOP_K = 7
MIN_EVAL_YEAR = 2007
MAX_EVAL_YEAR = 2023
NBA_MIN_GAMES = 200


# --------------------------------------------------------------------------- #
# Evaluation (mirrors scratch/tune_points.py, kept self-contained on purpose)  #
# --------------------------------------------------------------------------- #
def _clean_name(n: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(n))
    cleaned = "".join(c for c in normalized if not unicodedata.combining(c))
    cleaned = re.sub(r"[\.,]", "", cleaned)
    return " ".join(cleaned.lower().split())


def _build_top7_matrix(nba: pd.DataFrame) -> np.ndarray:
    """M7[j, k] = True if pool player k is among the first 7 entries of player
    j's ``similar_players`` list (the NBA-NBA similarity graph)."""
    profiles = json.loads(Path(PLAYER_PROFILES_PATH).read_text(encoding="utf-8"))
    col_by_id = {int(pid): j for j, pid in enumerate(nba["player_id"].astype(int))}
    M7 = np.zeros((len(nba), len(nba)), dtype=bool)
    for pid, j in col_by_id.items():
        for s in profiles.get(str(pid), {}).get("similar_players", [])[:TOP_K]:
            k = col_by_id.get(int(s["player_id"]))
            if k is not None:
                M7[j, k] = True
    return M7


def build_ctx() -> dict:
    """Load the NBA comp pool, the second-order graph, and every eval class."""
    career = pd.read_parquet(PLAYER_CAREER_FEATURES_PATH, engine="fastparquet")
    nba = career[career["career_games"] >= NBA_MIN_GAMES].copy().reset_index(drop=True)
    nba_clean = nba["player_name"].apply(_clean_name)
    nba_idx_by_name = {name: i for i, name in enumerate(nba_clean)}

    neighbor_idx = build_nba_neighbor_index(nba, top_k=TOP_K)
    M7 = _build_top7_matrix(nba)
    era_buckets = [
        _era_bucket(str(nba.iloc[j].get("career_span", ""))) for j in range(len(nba))
    ]

    classes = []
    draft_dir = DATA_STATIC_DIR / "draft"
    for year in range(MIN_EVAL_YEAR, MAX_EVAL_YEAR + 1):
        path = draft_dir / f"prospects_{year}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, engine="fastparquet").reset_index(drop=True)
        clean = df["player_name"].apply(_clean_name)
        rows = np.flatnonzero(clean.isin(nba_idx_by_name).to_numpy())
        if len(rows) == 0:
            continue
        cp = np.array([nba_idx_by_name[clean.iloc[r]] for r in rows], dtype=int)
        classes.append({"year": year, "df": df, "eval_rows": rows, "cp": cp})

    pipe = ProspectsPipeline()
    # Pre-populate the optional "quality" column so it can appear in bounds.
    nba["quality"] = pipe.nba_quality_scores(nba)
    for cls in classes:
        cls["df"]["quality"] = pipe.prospect_quality_scores(cls["df"])

    return {
        "pipe": pipe,
        "nba": nba,
        "neighbor_idx": neighbor_idx,
        "M7": M7,
        "eras": era_buckets,
        "classes": classes,
        "n": int(sum(len(c["eval_rows"]) for c in classes)),
    }


def evaluate(
    ctx: dict,
    *,
    features: list[str],
    weights: np.ndarray,
    bandwidth: float,
    smooth_lambda: float,
) -> dict:
    """Score one config through the production similarity code path."""
    pipe, nba, M7 = ctx["pipe"], ctx["nba"], ctx["M7"]
    first_hits = 0
    second_pts = 0
    second_any = 0
    all_ranks: list[np.ndarray] = []

    for cls in ctx["classes"]:
        sim = pipe.compute_similarity_matrix(
            cls["df"], nba,
            features=features, weights=np.asarray(weights, dtype=float),
            bandwidth=float(bandwidth), smooth_lambda=float(smooth_lambda),
            neighbor_idx=ctx["neighbor_idx"],
        )
        sub = sim[cls["eval_rows"]]
        cp = cls["cp"]
        target = sub[np.arange(len(cp)), cp]
        ranks = (sub > target[:, None]).sum(axis=1) + 1
        first_hits += int((ranks <= TOP_K).sum())
        all_ranks.append(ranks)

        top7 = np.argpartition(-sub, TOP_K, axis=1)[:, :TOP_K]
        for i in range(len(cp)):
            pts = int(M7[top7[i], cp[i]].sum())
            second_pts += pts
            second_any += int(pts > 0)

    ranks = np.concatenate(all_ranks)
    n = ctx["n"]
    points = 2 * first_hits + second_pts
    return {
        "n": n,
        "points": points,
        "points_per_prospect": round(points / n, 4),
        "max_points": 9 * n,
        "points_pct": round(100.0 * points / (9 * n), 2),
        "top7_hits": first_hits,
        "top7_pct": round(100.0 * first_hits / n, 2),
        "second_order_points": second_pts,
        "second_order_any_pct": round(100.0 * second_any / n, 2),
        "mean_rank": round(float(np.mean(ranks)), 2),
        "median_rank": float(np.median(ranks)),
    }


def _key(r: dict) -> tuple:
    """Ranking key: maximise points, then top-7 hit rate, then lower mean rank."""
    return (r["points"], r["top7_pct"], -r["mean_rank"])


# --------------------------------------------------------------------------- #
# Bounds + search                                                             #
# --------------------------------------------------------------------------- #
def load_bounds(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    features = cfg["features"]
    cols = features + ["height", "weight"]
    lo = np.array([cfg["bounds"][c][0] for c in cols], dtype=float)
    hi = np.array([cfg["bounds"][c][1] for c in cols], dtype=float)
    if np.any(hi < lo):
        bad = [c for c, a, b in zip(cols, lo, hi) if b < a]
        raise ValueError(f"ceiling < floor for columns: {bad}")
    return {
        "features": features,
        "columns": cols,
        "lo": lo,
        "hi": hi,
        "bandwidth": cfg.get("bandwidth", {"fixed": SIMILARITY_COMP_BANDWIDTH}),
        "smooth_lambda": cfg.get("smooth_lambda", {"fixed": SIMILARITY_COMP_SMOOTH_LAMBDA}),
        "hw_budget": cfg.get("hw_budget"),
        "search": cfg.get("search", {}),
    }


def _grid(spec: dict, fallback: float) -> list[float]:
    if "grid" in spec:
        return [float(x) for x in spec["grid"]]
    return [float(spec.get("fixed", fallback))]


def _hw_ok(w: np.ndarray, budget: float | None) -> bool:
    if budget is None:
        return True
    play_sq = float(np.sum(w[:-2] ** 2))
    hw_sq = float(w[-2] ** 2 + w[-1] ** 2)
    return hw_sq <= budget * (hw_sq + play_sq) + 1e-9


def _clamp(w: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(w, lo), hi)


def _shipped_in_bounds(bounds: dict) -> np.ndarray | None:
    """The currently-shipped weights, projected into the bounds box, if the
    feature list matches the shipped one (so we never lose to baseline)."""
    if bounds["features"] != list(SIMILARITY_COMP_FEATURES):
        return None
    return _clamp(np.asarray(SIMILARITY_COMP_WEIGHTS, float), bounds["lo"], bounds["hi"])


def calibrate(ctx: dict, bounds: dict, *, verbose: bool = True) -> dict:
    lo, hi = bounds["lo"], bounds["hi"]
    feats = bounds["features"]
    budget = bounds["hw_budget"]
    bw_grid = _grid(bounds["bandwidth"], SIMILARITY_COMP_BANDWIDTH)
    lam_grid = _grid(bounds["smooth_lambda"], SIMILARITY_COMP_SMOOTH_LAMBDA)

    s = bounds["search"]
    n_random = int(s.get("n_random", 400))
    seeds = [int(x) for x in s.get("seeds", [7, 11, 23])]
    cd_rounds = int(s.get("cd_rounds", 12))
    cd_step = float(s.get("cd_step", 0.05))

    def ev(w, bw, lam):
        return evaluate(ctx, features=feats, weights=w, bandwidth=bw, smooth_lambda=lam)

    best: dict = {"points": -1, "top7_pct": -1.0, "mean_rank": 1e9}

    # Seed with the shipped vector (clamped into the box) when feature sets match.
    seed_w = _shipped_in_bounds(bounds)
    if seed_w is not None and _hw_ok(seed_w, budget):
        r = ev(seed_w, bw_grid[0], lam_grid[0])
        best = {**r, "weights": seed_w.tolist(), "bandwidth": bw_grid[0],
                "smooth_lambda": lam_grid[0]}
        if verbose:
            print(f"  seed (shipped-in-box): pts={r['points']} top7={r['top7_pct']}%")

    # Random exploration within the box.
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for _ in range(n_random):
            w = lo + rng.random(len(lo)) * (hi - lo)
            if not _hw_ok(w, budget):
                continue
            bw = float(rng.choice(bw_grid))
            lam = float(rng.choice(lam_grid))
            r = ev(w, bw, lam)
            if _key(r) > _key(best):
                best = {**r, "weights": w.tolist(), "bandwidth": bw, "smooth_lambda": lam}
                if verbose:
                    print(f"  rand seed={seed}: pts={r['points']} "
                          f"top7={r['top7_pct']}% bw={bw} lam={lam}")

    if "weights" not in best:  # box so tight nothing passed the hw filter
        w0 = _clamp((lo + hi) / 2.0, lo, hi)
        r = ev(w0, bw_grid[0], lam_grid[0])
        best = {**r, "weights": w0.tolist(), "bandwidth": bw_grid[0],
                "smooth_lambda": lam_grid[0]}

    # Coordinate descent (respecting per-column bounds).
    weights = np.asarray(best["weights"], dtype=float)
    bw = float(best["bandwidth"])
    lam = float(best["smooth_lambda"])
    for rnd in range(cd_rounds):
        improved = False
        for i in range(len(weights)):
            base = weights[i]
            cand = {lo[i], hi[i], base,
                    base - cd_step, base - cd_step / 2.0,
                    base + cd_step / 2.0, base + cd_step, base + 2 * cd_step}
            for v in sorted(x for x in cand if lo[i] - 1e-9 <= x <= hi[i] + 1e-9):
                if abs(v - base) < 1e-9:
                    continue
                w_try = weights.copy(); w_try[i] = v
                if not _hw_ok(w_try, budget):
                    continue
                r = ev(w_try, bw, lam)
                if _key(r) > _key(best):
                    best = {**r, "weights": w_try.tolist(), "bandwidth": bw,
                            "smooth_lambda": lam}
                    weights = w_try
                    improved = True
        for bw_try in bw_grid:
            if abs(bw_try - bw) < 1e-9:
                continue
            r = ev(weights, bw_try, lam)
            if _key(r) > _key(best):
                best = {**r, "weights": weights.tolist(), "bandwidth": bw_try,
                        "smooth_lambda": lam}
                bw = bw_try; improved = True
        for lam_try in lam_grid:
            if abs(lam_try - lam) < 1e-9:
                continue
            r = ev(weights, bw, lam_try)
            if _key(r) > _key(best):
                best = {**r, "weights": weights.tolist(), "bandwidth": bw,
                        "smooth_lambda": lam_try}
                lam = lam_try; improved = True
        if verbose:
            print(f"  CD round {rnd + 1}: pts={best['points']} "
                  f"top7={best['top7_pct']}% bw={bw} lam={lam}")
        if not improved:
            break

    return best


# --------------------------------------------------------------------------- #
# Reporting + history ledger                                                  #
# --------------------------------------------------------------------------- #
def shipped_metrics(ctx: dict) -> dict:
    return evaluate(
        ctx,
        features=list(SIMILARITY_COMP_FEATURES),
        weights=np.asarray(SIMILARITY_COMP_WEIGHTS, float),
        bandwidth=SIMILARITY_COMP_BANDWIDTH,
        smooth_lambda=SIMILARITY_COMP_SMOOTH_LAMBDA,
    )


def _delta(label: str, new: float, old: float, *, pct: bool = False) -> str:
    d = new - old
    arrow = "▲" if d > 0 else ("▼" if d < 0 else "—")
    unit = "pp" if pct else ""
    return f"  {label:22s} {old:>8.3f} -> {new:>8.3f}   {arrow} {d:+.3f}{unit}"


def print_report(baseline: dict, best: dict, bounds: dict) -> None:
    print("\n" + "=" * 68)
    print("CALIBRATION RESULT  (vs currently-shipped weights)")
    print("=" * 68)
    print(_delta("points", best["points"], baseline["points"]))
    print(_delta("points / prospect", best["points_per_prospect"],
                 baseline["points_per_prospect"]))
    print(_delta("top-7 hit %", best["top7_pct"], baseline["top7_pct"], pct=True))
    print(_delta("median rank", best["median_rank"], baseline["median_rank"]))
    print(_delta("mean rank", best["mean_rank"], baseline["mean_rank"]))
    verdict = "IMPROVEMENT" if _key(best) > _key(baseline) else "DOWNGRADE / no gain"
    print(f"\n  Verdict: {verdict}")

    cols = bounds["columns"]
    w = [round(x, 4) for x in best["weights"]]
    print("\n  Tuned weights:")
    for c, val in zip(cols, w):
        print(f"    {c:16s} {val}")
    print(f"  bandwidth = {best['bandwidth']}   smooth_lambda = {best['smooth_lambda']}")

    print("\n  Paste into app/pipelines/prospects_pipeline.py:")
    print("    SIMILARITY_COMP_WEIGHTS = np.array([")
    for c, val in zip(cols, w):
        print(f"        {val},  # {c}")
    print("    ])")
    print(f"    SIMILARITY_COMP_BANDWIDTH = {best['bandwidth']}")
    print(f"    SIMILARITY_COMP_SMOOTH_LAMBDA = {best['smooth_lambda']}")
    print("=" * 68)


def append_history(baseline: dict, best: dict, bounds: dict, *, note: str) -> None:
    history = []
    if HISTORY_PATH.exists():
        history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": note,
        "eval_window": f"{MIN_EVAL_YEAR}-{MAX_EVAL_YEAR}",
        "n_prospects": best["n"],
        "features": bounds["columns"],
        "weights": [round(x, 4) for x in best["weights"]],
        "bandwidth": best["bandwidth"],
        "smooth_lambda": best["smooth_lambda"],
        "hw_budget": bounds["hw_budget"],
        "metrics": {
            "points": best["points"],
            "points_per_prospect": best["points_per_prospect"],
            "top7_pct": best["top7_pct"],
            "median_rank": best["median_rank"],
            "mean_rank": best["mean_rank"],
        },
        "vs_shipped": {
            "points": best["points"] - baseline["points"],
            "points_per_prospect": round(
                best["points_per_prospect"] - baseline["points_per_prospect"], 4),
            "top7_pct": round(best["top7_pct"] - baseline["top7_pct"], 2),
        },
        "is_improvement": bool(_key(best) > _key(baseline)),
    }
    history.append(entry)
    HISTORY_PATH.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(f"\nAppended run to {HISTORY_PATH.relative_to(_APP_ROOT.parent)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bounds", type=Path, default=DEFAULT_BOUNDS_PATH,
                    help="path to the floors/ceilings JSON (default: weight_bounds.json)")
    ap.add_argument("--search", action="store_true",
                    help="run the calibration search; without it, only the baseline is shown")
    ap.add_argument("--note", default="",
                    help="label stored alongside this run in weights_history.json")
    ap.add_argument("--no-log", action="store_true",
                    help="do not append the result to weights_history.json")
    args = ap.parse_args()

    print(f"Loading eval pools + second-order graph ({MIN_EVAL_YEAR}-{MAX_EVAL_YEAR})...")
    ctx = build_ctx()
    print(f"  eval prospects: {ctx['n']}   nba pool: {len(ctx['nba'])}   "
          f"classes: {len(ctx['classes'])}")

    baseline = shipped_metrics(ctx)
    print(f"\nShipped baseline: points={baseline['points']} "
          f"({baseline['points_per_prospect']}/prospect)  "
          f"top7={baseline['top7_pct']}%  median_rank={baseline['median_rank']:.0f}")

    if not args.search:
        print("\n(no --search: stopping after baseline. Edit weight_bounds.json, "
              "then re-run with --search.)")
        return

    bounds = load_bounds(args.bounds)
    print(f"\nCalibrating within bounds from {args.bounds.name} ...")
    best = calibrate(ctx, bounds)
    print_report(baseline, best, bounds)
    if not args.no_log:
        append_history(baseline, best, bounds,
                       note=args.note or "calibration run")


if __name__ == "__main__":
    main()
