                                                          

                                                                              
                                                                               
                                                                                
                                                            

                                                                      
   
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analytics.lineup_synergy import calculate_lineup_synergy              
from parquet_io import read_teams_parquet              

STATIC = os.path.join(os.path.dirname(__file__), "..", "data", "static")
OUT_PATH = os.path.join(STATIC, "lineup_synergy_scores.json")


def main():
    tp = json.load(open(os.path.join(STATIC, "team_profiles.json")))
    sl = json.load(open(os.path.join(STATIC, "starting_lineups.json")))
    tm = json.load(open(os.path.join(STATIC, "team_metadata.json")))
    pp = json.load(open(os.path.join(STATIC, "player_profiles.json")))
    baselines_path = os.path.join(STATIC, "season_lineup_baselines.json")
    sb = json.load(open(baselines_path)) if os.path.exists(baselines_path) else {}
    psf = read_teams_parquet(os.path.join(STATIC, "player_season_features.parquet"))
    teams_df = read_teams_parquet(os.path.join(STATIC, "teams_historical.parquet"))

    scores = {}
    t0 = time.time()
    for i, (key, ld) in enumerate(sl.items()):
        ids = [s["player_id"] for s in ld["starters"]]
        try:
            res = calculate_lineup_synergy(
                ids, ld["season"], psf, teams_df, sl, tp, tm, pp,
                compute_similar=False, season_baselines=sb,
            )
            scores[key] = round(float(res["synergy_score"]), 1)
        except Exception as exc:                
            print(f"  [skip] {key}: {exc}", file=sys.stderr)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(sl)} ({time.time()-t0:.0f}s)", file=sys.stderr)

    with open(OUT_PATH, "w") as f:
        json.dump(scores, f)
    print(f"wrote {len(scores)} synergy scores -> {OUT_PATH} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
