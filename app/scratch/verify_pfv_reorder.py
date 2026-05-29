import sys
import math
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analytics.player_profiles.archetypes import calculate_pfv

def calculate_pfv_old(metrics: dict) -> float:
    # Original order
    keys = [
        "pts_per36",
        "ast_per36",
        "reb_per36",
        "stl_per36",
        "blk_per36",
        "ts_pct",
    ]
    r = []
    for k in keys:
        metric = metrics.get(k, {})
        pct = metric.get("percentile", 0.0) if isinstance(metric, dict) else metric
        r.append(float(pct or 0.0) / 100.0)
    
    n = 6
    sum_product = 0.0
    for i in range(n):
        sum_product += r[i] * r[(i + 1) % n]
    
    sin_term = math.sin(2.0 * math.pi / n)
    A = 0.5 * sin_term * sum_product
    A_max = n * 0.5 * sin_term
    return round(A / A_max, 4)

def run_verification():
    # 1. Specialized Defensive Center
    center_metrics = {
        "pts_per36": {"percentile": 35.0},
        "ast_per36": {"percentile": 10.0},
        "reb_per36": {"percentile": 95.0},
        "stl_per36": {"percentile": 30.0},
        "blk_per36": {"percentile": 95.0},
        "ts_pct": {"percentile": 95.0},
    }

    # 2. Elite Playmaking Guard
    guard_metrics = {
        "pts_per36": {"percentile": 95.0},
        "ast_per36": {"percentile": 85.0},
        "reb_per36": {"percentile": 20.0},
        "stl_per36": {"percentile": 70.0},
        "blk_per36": {"percentile": 10.0},
        "ts_pct": {"percentile": 85.0},
    }

    # 3. Truly Versatile Forward (e.g. LeBron James style)
    versatile_metrics = {
        "pts_per36": {"percentile": 95.0},
        "ast_per36": {"percentile": 90.0},
        "reb_per36": {"percentile": 75.0},
        "stl_per36": {"percentile": 70.0},
        "blk_per36": {"percentile": 50.0},
        "ts_pct": {"percentile": 85.0},
    }

    print("=== Verification of PFV Clockwise Reordering ===")
    
    for label, metrics in [
        ("Defensive Center (Specialized)", center_metrics),
        ("Elite Playmaking Guard", guard_metrics),
        ("Truly Versatile Forward", versatile_metrics)
    ]:
        pfv_old = calculate_pfv_old(metrics)
        pfv_new = calculate_pfv(metrics)
        pct_change = ((pfv_new - pfv_old) / pfv_old) * 100
        print(f"\n{label}:")
        print(f"  Old PFV: {pfv_old:.4f}")
        print(f"  New PFV: {pfv_new:.4f}")
        print(f"  Change:  {pct_change:+.1f}%")

if __name__ == "__main__":
    run_verification()
