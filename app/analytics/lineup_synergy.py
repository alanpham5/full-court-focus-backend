import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
from typing import List, Dict, Any

from analytics.eras import season_to_decade

ALL_ROLES = [
    "Playmaker",
    "Secondary Creator",
    "Designated Scorer",
    "Perimeter Specialist",
    "Rim Attacker",
    "Interior Presence",
    "Defensive Specialist",
]

STYLE_AXIS_WEIGHTS = np.array([0.75, 1.20, 1.00, 1.15, 1.20, 0.95], dtype=float)


def style_affinity(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    dist = float(np.sqrt(np.average(diff * diff, weights=STYLE_AXIS_WEIGHTS)))
    return float(max(0.0, 1.0 - dist / 0.78))


def synergy_win_pct_target(synergy_score: float) -> float:
    return float(min(0.82, max(0.25, 0.24 + 0.0062 * synergy_score)))


def quality_affinity(target_win_pct: float, historical_win_pct: Any) -> float:
    try:
        win_pct = float(historical_win_pct)
    except (TypeError, ValueError):
        return 0.70
    if not np.isfinite(win_pct):
        return 0.70
    return float(max(0.0, 1.0 - abs(win_pct - target_win_pct) / 0.52))


def select_similar_lineup_candidates(candidates: list[dict], limit: int = 8) -> list[dict]:
    if not candidates:
        return []

    ordered = sorted(candidates, key=lambda x: x["similarity_pct"], reverse=True)
    selected: list[dict] = []
    selected_keys: set[str] = set()

    def add(item: dict) -> bool:
        key = str(item["key"])
        if key in selected_keys or len(selected) >= limit:
            return False
        selected.append(item)
        selected_keys.add(key)
        return True

    for item in ordered:
        if item.get("exact_overlap", 0) >= 4 and item["similarity_pct"] >= 78.0:
            add(item)
            if len(selected) >= min(3, limit):
                break

    for item in ordered:
        if item.get("exact_overlap", 0) >= 3 and item["similarity_pct"] >= 64.0:
            add(item)
            if len(selected) >= min(5, limit):
                break

    for decade in ("1990s", "2000s", "2010s", "2020s"):
        if any(item.get("decade") == decade for item in selected):
            continue
        for item in ordered:
            if item.get("decade") == decade and item["similarity_pct"] >= 58.0:
                add(item)
                break

    while len(selected) < limit:
        best_item = None
        best_score = -np.inf
        team_counts: dict[int, int] = {}
        decade_counts: dict[str, int] = {}
        starter_sets = [set(item.get("starter_ids", [])) for item in selected]
        for item in selected:
            team_counts[item["team_id"]] = team_counts.get(item["team_id"], 0) + 1
            decade = item.get("decade", "")
            decade_counts[decade] = decade_counts.get(decade, 0) + 1

        for item in ordered:
            if str(item["key"]) in selected_keys:
                continue
            score = float(item["similarity_pct"])
            score -= 5.0 * team_counts.get(item["team_id"], 0)
            score -= 3.0 * decade_counts.get(item.get("decade", ""), 0)

            item_starters = set(item.get("starter_ids", []))
            if any(len(item_starters & prev) >= 4 for prev in starter_sets):
                score -= 4.0
            if item.get("exact_overlap", 0) >= 3:
                score += 6.0

            if score > best_score:
                best_score = score
                best_item = item

        if best_item is None:
            break
        add(best_item)

    return sorted(selected, key=lambda x: x["similarity_pct"], reverse=True)

def estimate_paint_fga(row: pd.Series) -> float:
    fga = float(row.get("FGA", 0.0) or 0.0)
    fg3a = float(row.get("FG3A", 0.0) or 0.0)
    non_3pa = fga - fg3a
    
    reb_z = float(row.get("reb_per36_z", 0.0) or 0.0)
    blk_z = float(row.get("blk_per36_z", 0.0) or 0.0)
    
    if reb_z > 0.5 or blk_z > 0.5:
        ratio = 0.80
    elif (fg3a / fga if fga > 0 else 0) > 0.45:
        ratio = 0.30
    else:
        ratio = 0.45
        
    return max(0.0, non_3pa * ratio)

def compute_defense_score(row: pd.Series, roles: Any) -> float:
    min_val = float(row.get("MIN", 0.0) or 0.0)
    if min_val <= 0.0:
        return 0.0
    stl_per36 = float(row.get("STL", 0.0) or 0.0) / min_val * 36.0
    blk_per36 = float(row.get("BLK", 0.0) or 0.0) / min_val * 36.0
    
    if isinstance(roles, str):
        roles = [roles]
    roles = roles or []
    primary_role = roles[0] if roles else ""
    
    role_bonus = 0.0
    if primary_role == "Defensive Specialist":
        role_bonus = 1.5
    elif primary_role == "Interior Presence":
        role_bonus = 1.0
        
    return stl_per36 + 1.2 * blk_per36 + role_bonus

def assign_player_roles_absolute(df: pd.DataFrame) -> Dict[int, List[str]]:
    roles_dict = {}
    for _, r in df.iterrows():
        pid = int(r["PLAYER_ID"])
        pts = float(r.get("pts_per36_z", 0.0) or 0.0)
        ast = float(r.get("ast_per36_z", 0.0) or 0.0)
        reb = float(r.get("reb_per36_z", 0.0) or 0.0)
        stl = float(r.get("stl_per36_z", 0.0) or 0.0)
        blk = float(r.get("blk_per36_z", 0.0) or 0.0)
        fg3 = float(r.get("fg3a_rate_z", 0.0) or 0.0)
        fta = float(r.get("fta_rate_z", 0.0) or 0.0)
        ast_pct = float(r.get("ast_pct_z", 0.0) or 0.0)
        
        roles = []
        
        if ast > 0.8 and ast_pct > 0.6:
            roles.append("Playmaker")
        elif pts > 0.8 and ast_pct < 0.4:
            roles.append("Designated Scorer")
        elif ast > 0.2 and pts > 0.0:
            roles.append("Secondary Creator")
            
        if fg3 > 0.6:
            roles.append("Perimeter Specialist")
        elif fta > 0.5 and pts > 0.0:
            roles.append("Rim Attacker")
            
        if (reb > 0.7 or blk > 0.7) and fg3 < -0.4:
            roles.append("Interior Presence")
            
        if (stl > 0.8 or blk > 0.8) and "Interior Presence" not in roles:
            roles.append("Defensive Specialist")
            
        if not roles:
            if pts > 0.0:
                roles.append("Secondary Creator")
            elif reb > 0.0:
                roles.append("Interior Presence")
            else:
                roles.append("Defensive Specialist")
                
        roles_dict[pid] = roles[:2]
    return roles_dict

def compute_lineup_collective_stats(player_rows: List[pd.Series], player_roles: Dict[int, List[str]]) -> Dict[str, float]:
    fg3a_per36_sum = 0.0
    paint_fga_per36_sum = 0.0
    ast_per36_sum = 0.0
    fgm_per36_sum = 0.0
    reb_per36_sum = 0.0
    blk_per36_sum = 0.0
    def_score_sum = 0.0
    
    for r in player_rows:
        p_min = float(r.get("MIN", 0.0) or 0.0)
        p_id = int(r["PLAYER_ID"])
        roles = player_roles.get(p_id, ["Secondary Creator"])
        
        if p_min > 0:
            fg3a_per36_sum += float(r.get("FG3A", 0.0) or 0.0) / p_min * 36.0
            paint_fga_per36_sum += estimate_paint_fga(r) / p_min * 36.0
            ast_per36_sum += float(r.get("AST", 0.0) or 0.0) / p_min * 36.0
            fgm_per36_sum += float(r.get("FGM", 0.0) or 0.0) / p_min * 36.0
            reb_per36_sum += float(r.get("REB", 0.0) or 0.0) / p_min * 36.0
            blk_per36_sum += float(r.get("BLK", 0.0) or 0.0) / p_min * 36.0
            def_score_sum += compute_defense_score(r, roles)
            
    fg3a_proj = fg3a_per36_sum * 1.3333
    paint_fga_proj = paint_fga_per36_sum * 1.3333
    ast_proj = ast_per36_sum * 1.3333
    fgm_proj = fgm_per36_sum * 1.3333
    ast_pct_proj = (ast_proj / fgm_proj) * 100.0 if fgm_proj > 0 else 0.0
    reb_proj = reb_per36_sum * 1.3333
    blk_proj = blk_per36_sum * 1.3333
    paint_score_proj = reb_proj + 4.0 * blk_proj
    
    return {
        "fg3a": fg3a_proj,
        "paint_fga": paint_fga_proj,
        "paint_score": paint_score_proj,
        "ast": ast_proj,
        "fgm": fgm_proj,
        "ast_pct": ast_pct_proj,
        "reb": reb_proj,
        "blk": blk_proj,
        "def_score": def_score_sum
    }

def cosine_similarity(a, b):
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))

def height_to_inches(height_val: Any) -> float:
    if pd.isna(height_val) or not height_val:
        return 78.0
    val_str = str(height_val).strip()
    if not val_str:
        return 78.0
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
    return 78.0

def compute_lineup_player_similarity(
    player_ids: List[int],
    h_starter_ids: List[int],
    player_profiles: Dict[str, Any]
) -> float:
    if not player_profiles:
        overlap = len(set(player_ids) & set(h_starter_ids))
        return overlap / 5.0

    p_list = list(player_ids)
    h_list = list(h_starter_ids)
    
    exact_matches = 0
    matched_p = set()
    matched_h = set()
    
    for p_id in p_list:
        if p_id in h_list:
            exact_matches += 1
            matched_p.add(p_id)
            matched_h.add(p_id)
            
    remaining_p = [p for p in p_list if p not in matched_p]
    remaining_h = [h for h in h_list if h not in matched_h]
    
    if not remaining_p:
        return 1.0
        
    def get_player_sim(p_id: int, h_id: int) -> float:
        if p_id == h_id:
            return 1.0
        profile = player_profiles.get(str(p_id), {})
        similar_players = profile.get("similar_players", [])
        for sim_p in similar_players:
            if sim_p.get("player_id") == h_id:
                return float(sim_p.get("similarity_score", 0.0)) / 100.0
                
        profile_h = player_profiles.get(str(h_id), {})
        similar_players_h = profile_h.get("similar_players", [])
        for sim_p in similar_players_h:
            if sim_p.get("player_id") == p_id:
                return float(sim_p.get("similarity_score", 0.0)) / 100.0
                
        role_p = profile.get("role")
        role_h = profile_h.get("role")
        if role_p and role_h and role_p == role_h:
            return 0.35
            
        return 0.15

    import itertools
    max_sim_sum = 0.0
    K = len(remaining_p)
    for perm in itertools.permutations(remaining_h):
        sim_sum = sum(get_player_sim(remaining_p[idx], perm[idx]) for idx in range(K))
        if sim_sum > max_sim_sum:
            max_sim_sum = sim_sum
            
    return (exact_matches * 1.0 + max_sim_sum) / 5.0

def calculate_lineup_synergy(
    player_ids: List[int],
    season: str,
    player_season_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    starting_lineups: Dict[str, Any],
    team_profiles: Dict[str, Any],
    team_metadata: Dict[str, Any],
    player_profiles: Dict[str, Any] = None,
) -> Dict[str, Any]:
    season_players = player_season_df[player_season_df["SEASON"] == season]
    if season_players.empty:
        raise ValueError(f"No player data found for season {season}")
        
    lineup_rows = season_players[season_players["PLAYER_ID"].isin(player_ids)]
    found_ids = set(lineup_rows["PLAYER_ID"].unique())
    missing_ids = [pid for pid in player_ids if pid not in found_ids]
    if missing_ids:
        raise ValueError(f"Player IDs {missing_ids} not found in season {season}")
        
    lineup_rows = lineup_rows.set_index("PLAYER_ID").loc[player_ids].reset_index()
    
    from analytics.player_profiles.archetypes import get_player_archetypes
    player_roles = assign_player_roles_absolute(lineup_rows)
    
    player_rows_list = [lineup_rows.iloc[i] for i in range(len(lineup_rows))]
    custom_stats = compute_lineup_collective_stats(player_rows_list, player_roles)
    
    season_lineup_keys = [k for k in starting_lineups.keys() if k.endswith(f":{season}")]
    starting_rebs = []
    starting_defs = []
    starting_paints = []
    starting_blks = []
    starting_asts = []
    starting_ast_pcts = []
    
    for key in season_lineup_keys:
        lineup = starting_lineups[key]
        starter_ids = [s["player_id"] for s in lineup["starters"]]
        s_rows = season_players[season_players["PLAYER_ID"].isin(starter_ids)]
        if len(s_rows) >= 5:
            s_roles = assign_player_roles_absolute(s_rows)
            s_rows_list = [s_rows.iloc[i] for i in range(len(s_rows))]
            s_stats = compute_lineup_collective_stats(s_rows_list, s_roles)
            starting_rebs.append(s_stats["reb"])
            starting_defs.append(s_stats["def_score"])
            starting_paints.append(s_stats["paint_score"])
            starting_blks.append(s_stats["blk"])
            starting_asts.append(s_stats["ast"])
            starting_ast_pcts.append(s_stats["ast_pct"])
            
    mean_starting_reb = np.mean(starting_rebs) if starting_rebs else 1.0
    mean_starting_def = np.mean(starting_defs) if starting_defs else 1.0
    mean_starting_paint = np.mean(starting_paints) if starting_paints else 1.0
    mean_starting_blk = np.mean(starting_blks) if starting_blks else 1.0
    
    season_teams = teams_df[teams_df["SEASON"] == season]
    if season_teams.empty:
        raise ValueError(f"No team historical stats found for season {season}")
        
    team_abbr_to_id = {v["abbreviation"]: int(k) for k, v in team_metadata.items()}
    
    custom_paces = []
    custom_mpgs = []
    for _, r in lineup_rows.iterrows():
        abbr = r.get("TEAM_ABBREVIATION", "")
        mpg = float(r.get("MIN", 0.0) or 0.0) / float(r.get("GP", 1.0) or 1.0)
        p_team_id = team_abbr_to_id.get(abbr)
        p_team_row = season_teams[season_teams["TEAM_ID"] == p_team_id] if p_team_id else pd.DataFrame()
        
        if not p_team_row.empty:
            team_pace = float(p_team_row.iloc[0].get("PACE", 98.0))
        else:
            team_pace = 98.0
        custom_paces.append(team_pace)
        custom_mpgs.append(mpg)
        
    sum_mpg = sum(custom_mpgs)
    custom_pace = sum(p * m for p, m in zip(custom_paces, custom_mpgs)) / sum_mpg if sum_mpg > 0 else 98.0
    
    avg_team_reb_pct = (season_teams["REB_PCT"].mean() or 0.50) * 100.0
    estimated_reb_pct = avg_team_reb_pct + 10.0 * (custom_stats["reb"] - mean_starting_reb) / mean_starting_reb
    estimated_reb_pct = min(60.0, max(40.0, estimated_reb_pct))
    
    avg_team_paint_fga = season_teams["PAINT_FGA"].mean()
    if pd.isna(avg_team_paint_fga) or avg_team_paint_fga <= 0.0:
        avg_team_paint_fga = 44.3
    estimated_paint_fga = avg_team_paint_fga + 15.0 * (custom_stats["paint_score"] - mean_starting_paint) / mean_starting_paint
    estimated_paint_fga = min(70.0, max(25.0, estimated_paint_fga))
    
    avg_team_def_rating = season_teams["DEF_RATING"].mean() or 110.0
    estimated_def_rating = avg_team_def_rating - 15.0 * (custom_stats["def_score"] - mean_starting_def) / mean_starting_def
    
    pace_pct = percentileofscore(season_teams["PACE"].dropna().to_numpy(), custom_pace, kind="rank")
    fg3a_pct = percentileofscore(season_teams["FG3A"].dropna().to_numpy(), custom_stats["fg3a"], kind="rank")
    
    paint_fga_series = season_teams["PAINT_FGA"].dropna()
    if paint_fga_series.empty:
        paint_fga_distribution = np.linspace(35.0, 53.5, 30)
    else:
        paint_fga_distribution = paint_fga_series.to_numpy()
    paint_pct = percentileofscore(paint_fga_distribution, estimated_paint_fga, kind="rank")
    
    if starting_asts:
        ast_vol_pct = percentileofscore(np.array(starting_asts), custom_stats["ast"], kind="rank")
        ast_rate_pct = percentileofscore(np.array(starting_ast_pcts), custom_stats["ast_pct"], kind="rank")
        playmaking_pct = 0.55 * ast_vol_pct + 0.45 * ast_rate_pct
    else:
        playmaking_pct = percentileofscore(
            season_teams["AST_PCT"].dropna().to_numpy() * 100.0,
            custom_stats["ast_pct"],
            kind="rank",
        )
    rebounding_pct = percentileofscore(season_teams["REB_PCT"].dropna().to_numpy() * 100.0, estimated_reb_pct, kind="rank")
    
    defense_pct = percentileofscore(season_teams["DEF_RATING"].dropna().to_numpy(), estimated_def_rating, kind="rank")
    defense_pct = 100.0 - defense_pct
    
    style_vector = {
        "pace": round(min(100.0, max(0.0, pace_pct)), 1),
        "three_point_volume": round(min(100.0, max(0.0, fg3a_pct)), 1),
        "paint": round(min(100.0, max(0.0, paint_pct)), 1),
        "defense": round(min(100.0, max(0.0, defense_pct)), 1),
        "playmaking": round(min(100.0, max(0.0, playmaking_pct)), 1),
        "rebounding": round(min(100.0, max(0.0, rebounding_pct)), 1)
    }
    
    player_ratings = []
    for _, r in lineup_rows.iterrows():
        pts_p = float(r.get("pts_per36_pctile", 50.0) or 50.0)
        ast_p = float(r.get("ast_per36_pctile", 50.0) or 50.0)
        reb_p = float(r.get("reb_per36_pctile", 50.0) or 50.0)
        stl_p = float(r.get("stl_per36_pctile", 50.0) or 50.0)
        blk_p = float(r.get("blk_per36_pctile", 50.0) or 50.0)
        ts_p = float(r.get("ts_pct_pctile", 50.0) or 50.0)
        
        rating = 0.30 * pts_p + 0.15 * ast_p + 0.15 * reb_p + 0.10 * stl_p + 0.10 * blk_p + 0.20 * ts_p
        player_ratings.append(rating)
        
    baseline_talent = 0.88 * np.mean(player_ratings) + 0.12 * np.min(player_ratings)
    
    playmakers_count = sum(1 for pid in player_ids if "Playmaker" in player_roles.get(pid, []))
    creators_count = sum(1 for pid in player_ids if "Secondary Creator" in player_roles.get(pid, []) or "Designated Scorer" in player_roles.get(pid, []))
    
    shooters_count = 0
    for _, r in lineup_rows.iterrows():
        p_min = float(r.get("MIN", 0.0) or 0.0)
        fg3m_per36 = (float(r.get("FG3M", 0.0) or 0.0) / p_min * 36.0) if p_min > 0 else 0.0
        fg3a_pctile = float(r.get("fg3a_rate_pctile", 0.0) or 0.0)
        if "Perimeter Specialist" in player_roles.get(int(r["PLAYER_ID"]), []) or fg3m_per36 >= 1.5 or fg3a_pctile > 65.0:
            shooters_count += 1
            
    bigs_count = sum(1 for pid in player_ids if "Interior Presence" in player_roles.get(pid, []))
    
    defenders_count = 0
    for _, r in lineup_rows.iterrows():
        pid = int(r["PLAYER_ID"])
        roles = player_roles.get(pid, [])
        is_def_spec = "Defensive Specialist" in roles
        stl_pctile = float(r.get("stl_per36_pctile", 0.0) or 0.0)
        blk_pctile = float(r.get("blk_per36_pctile", 0.0) or 0.0)
        if is_def_spec or stl_pctile > 60.0 or blk_pctile > 65.0:
            defenders_count += 1
            
    playmaker_score = playmakers_count + 0.5 * creators_count
    if playmaker_score == 0:
        playmaking_adj = -15.0
    elif playmakers_count >= 3:
        playmaking_adj = -1.0
    elif playmaker_score <= 1.0:
        playmaking_adj = 2.0
    else:
        playmaking_adj = 7.0
        
    if shooters_count == 0:
        spacing_adj = -20.0
    elif shooters_count == 1:
        spacing_adj = -10.0
    elif shooters_count == 2:
        spacing_adj = 4.0
    else:
        spacing_adj = 8.0
        
    reb_ratio = custom_stats["reb"] / mean_starting_reb if mean_starting_reb > 0 else 1.0
    blk_ratio = custom_stats["blk"] / mean_starting_blk if mean_starting_blk > 0 else 1.0
    
    if reb_ratio < 0.88 or blk_ratio < 0.70:
        interior_adj = -15.0
    elif reb_ratio > 1.25 and blk_ratio > 1.60:
        interior_adj = -10.0
    elif reb_ratio >= 1.05 and blk_ratio >= 1.15:
        interior_adj = 5.0
    else:
        interior_adj = 2.0
        
    if defenders_count == 0:
        defense_adj = -8.0
    elif defenders_count == 1:
        defense_adj = 2.0
    else:
        defense_adj = 6.0

    if playmaking_pct >= 60.0:
        if playmaking_adj < 0:
            playmaking_adj = 4.0 if playmakers_count >= 3 else max(playmaking_adj, -2.0)
        else:
            playmaking_adj = max(4.0, playmaking_adj)
    elif playmaking_pct >= 45.0:
        if playmaking_adj < 0 and playmakers_count >= 3:
            playmaking_adj = 1.0
        elif playmaking_adj > 0:
            playmaking_adj = max(1.0, playmaking_adj)
    else:
        playmaking_adj = min(-3.0, playmaking_adj)

    if fg3a_pct < 45.0:
        spacing_adj = min(-3.0, spacing_adj)

    if rebounding_pct < 45.0 or paint_pct < 45.0:
        interior_adj = min(-3.0, interior_adj)

    if defense_pct < 45.0:
        if defenders_count >= 3:
            defense_adj = max(defense_adj, 4.0)
            defense_pct = max(defense_pct, 68.0)
        elif defenders_count >= 2:
            defense_adj = max(defense_adj, 1.0)
            defense_pct = max(defense_pct, 62.0)
        else:
            defense_adj = min(-3.0, defense_adj)
    elif defense_pct >= 60.0 and defense_adj >= 0.0:
        defense_adj = max(4.0, defense_adj)

    style_vector["defense"] = round(min(100.0, max(0.0, defense_pct)), 1)
        
    primary_roles = [player_roles.get(pid)[0] for pid in player_ids if player_roles.get(pid)]
    overlap_adj = 0.0
    for role in set(primary_roles):
        if primary_roles.count(role) >= 3:
            if role in ["Designated Scorer", "Interior Presence", "Playmaker"]:
                overlap_adj = -6.0
                break
            
    synergy_score = baseline_talent + playmaking_adj + spacing_adj + interior_adj + defense_adj + overlap_adj
    synergy_score = round(min(100.0, max(0.0, synergy_score)), 1)
    
    strengths = []
    weaknesses = []
    custom_fg3a = custom_stats["fg3a"]
    sum_ast = sum(float(r.get("ast_per36", 0.0) or 0.0) for _, r in lineup_rows.iterrows())
    sum_stl = sum(float(r.get("stl_per36", 0.0) or 0.0) for _, r in lineup_rows.iterrows())
    sum_pts = sum(float(r.get("pts_per36", 0.0) or 0.0) for _, r in lineup_rows.iterrows())
    
    if playmaking_pct >= 60.0 and playmaking_adj >= 4.0:
        if playmakers_count >= 2 and sum_ast >= 25.0:
            strengths.append(
                "Multiple Initiators - Two or more primary creators drive above-average assist volume, keeping the offense fluid and generating open looks."
            )
        else:
            strengths.append(
                "Fluid Offense - Assist volume and creation rate both rank above comparable NBA starting units."
            )
    elif playmaking_pct >= 50.0 and playmaking_adj >= 0.0:
        strengths.append(
            "Balanced Creation - Solid assist output supports workable ball movement without over-relying on a single handler."
        )

    playmaking_weak = playmaking_pct < 40.0 or (playmaking_adj <= -5.0 and playmaking_pct < 50.0)
    if playmaking_weak:
        if playmakers_count == 0 and playmaker_score <= 1.0:
            weaknesses.append(
                "Missing Initiator - No clear primary playmaker leaves the offense dependent on isolation and scripted sets."
            )
        elif playmakers_count >= 3 and playmaking_pct < 45.0:
            weaknesses.append(
                "Crowded Creation - Several ball-dominant players overlap, and collective assist rates trail comparable NBA units."
            )
        else:
            weaknesses.append(
                "Limited Ball Movement - Assist volume and rates rank below average, raising the risk of stagnant half-court possessions."
            )

    if spacing_adj < 0 and fg3a_pct < 45.0:
        if shooters_count == 0 or custom_fg3a < 22.0:
            weaknesses.append(
                "Clogged Driving Lanes - Fewer than two credible perimeter threats lets defenses shrink the floor and wall off the rim."
            )
        elif shooters_count == 1:
            weaknesses.append(
                "Limited Floor Spacing - Only one reliable outside threat makes it easier for help defenders to sit in the paint."
            )
        else:
            weaknesses.append(
                "Low 3PT Volume - Three-point attempt rate ranks below average, reducing the space available for drivers and post touches."
            )
    elif spacing_adj > 0 and fg3a_pct >= 60.0:
        if shooters_count >= 3 and custom_fg3a >= 32.0:
            strengths.append(
                "Elite Floor Spacing - Multiple perimeter threats pull help defenders out and keep driving lanes open."
            )
        else:
            strengths.append(
                "Stretch-the-Floor Shooting - Above-average three-point volume forces defenses to extend, opening up the interior."
            )

    if interior_adj < 0:
        if interior_adj == -15.0 and (rebounding_pct < 45.0 or paint_pct < 45.0):
            weaknesses.append(
                "Frontcourt Vulnerability - Below-average size and interior metrics leave this group exposed in the paint and on the glass."
            )
        elif interior_adj == -10.0 and paint_pct >= 55.0:
            weaknesses.append(
                "Paint Congestion - Heavy frontcourt usage can crowd the lane and limit driving paths for perimeter creators."
            )

        if rebounding_pct < 45.0 and reb_ratio < 0.92:
            weaknesses.append(
                "Rebounding Vulnerability - Rebounding profile ranks below average, increasing second-chance opportunities for opponents."
            )

        if paint_pct < 40.0:
            weaknesses.append(
                "Limited Paint Pressure - Low interior scoring volume makes the offense easier to defend with a compact shell."
            )
    elif interior_adj > 0:
        if reb_ratio >= 1.05 and rebounding_pct >= 55.0:
            if reb_ratio >= 1.12 and blk_ratio >= 1.20:
                strengths.append(
                    "Elite Interior Presence - Strong rebounding and rim protection give this unit control of the paint on both ends."
                )
            elif rebounding_pct >= 60.0:
                strengths.append(
                    "Glass Control - Above-average rebounding helps this group finish possessions and limit extra opponent looks."
                )
        if paint_pct >= 60.0:
            strengths.append(
                "Paint Scoring Pressure - High interior shot volume keeps rim protectors honest and generates efficient looks at the basket."
            )

    if defense_adj < 0 and defense_pct < 45.0:
        if blk_ratio < 0.70 and defenders_count == 0:
            weaknesses.append(
                "Weak Rim Protection - Without a credible shot-blocking anchor, opponents can finish more comfortably at the basket."
            )
        elif sum_stl < 4.0 and defenders_count <= 1:
            weaknesses.append(
                "Passive Perimeter Defense - Limited steal and disruption activity makes it harder to blow up opposing actions."
            )
        else:
            weaknesses.append(
                "Defensive Limitations - Projected defensive profile ranks below average for a five-man unit."
            )
    elif defense_adj > 0 and defense_pct >= 60.0:
        if blk_ratio >= 1.25:
            strengths.append(
                "Rim Protection - Shot-blocking presence deters attempts at the basket and protects the back line."
            )
        elif sum_stl >= 7.5 and defenders_count >= 2:
            strengths.append(
                "Perimeter Disruption - Active hands and rotations create turnovers and transition opportunities."
            )
        else:
            strengths.append(
                "Solid Defensive Foundation - Above-average defensive projection supports consistent half-court stops."
            )

    if sum_pts >= 95.0 and baseline_talent >= 72.0:
        strengths.append(
            "High-End Scoring - Elite individual scoring profiles give this lineup multiple ways to generate points against set defenses."
        )
    elif (sum_pts < 65.0 or baseline_talent < 40.0) and baseline_talent < 50.0:
        weaknesses.append(
            "Scoring Depth Concerns - Limited high-volume scoring options increase the risk of cold stretches against disciplined defenses."
        )
        
    total_allowed = 7
    if len(strengths) + len(weaknesses) > total_allowed:
        max_each = total_allowed // 2
        if len(strengths) > max_each and len(weaknesses) > max_each:
            strengths = strengths[:4]
            weaknesses = weaknesses[:3]
        elif len(strengths) <= max_each:
            weaknesses = weaknesses[:(total_allowed - len(strengths))]
        else:
            strengths = strengths[:(total_allowed - len(weaknesses))]
        
    custom_role_vector = np.zeros(len(ALL_ROLES))
    for pid in player_ids:
        for role in player_roles.get(pid, []):
            if role in ALL_ROLES:
                custom_role_vector[ALL_ROLES.index(role)] += 1
                
    custom_style_vector = np.array([
        style_vector["pace"],
        style_vector["three_point_volume"],
        style_vector["paint"],
        style_vector["defense"],
        style_vector["playmaking"],
        style_vector["rebounding"]
    ]) / 100.0
    
    similar_teams_list = []
    target_win_pct = synergy_win_pct_target(synergy_score)
    
    for key, h_lineup in starting_lineups.items():
        h_team_id, h_season = key.split(":")
        
        h_starter_ids = [s["player_id"] for s in h_lineup["starters"]]
            
        h_role_vector = np.zeros(len(ALL_ROLES))
        for s in h_lineup["starters"]:
            for r in s.get("roles", []):
                if r in ALL_ROLES:
                    h_role_vector[ALL_ROLES.index(r)] += 1
                    
        h_profile = team_profiles.get(key, {})
        h_style_dict = h_profile.get("style_vector", {})
        if h_style_dict:
            h_style_vector = np.array([
                h_style_dict.get("pace", 50.0),
                h_style_dict.get("three_point_volume", 50.0),
                h_style_dict.get("paint", 50.0),
                h_style_dict.get("defense", 50.0),
                h_style_dict.get("playmaking", 50.0),
                h_style_dict.get("rebounding", 50.0)
            ]) / 100.0
        else:
            h_style_vector = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
            
        role_sim = cosine_similarity(custom_role_vector, h_role_vector)
        style_sim = style_affinity(custom_style_vector, h_style_vector)
        
        player_sim = compute_lineup_player_similarity(player_ids, h_starter_ids, player_profiles)
        exact_overlap = len(set(player_ids) & set(h_starter_ids))
        quality_sim = quality_affinity(target_win_pct, h_profile.get("win_pct"))
        
        w_player = min(0.55, 0.12 + 0.10 * exact_overlap + 0.22 * max(0.0, player_sim - 0.35))
        context_sim = 0.54 * style_sim + 0.26 * role_sim + 0.20 * quality_sim
        base_sim = w_player * player_sim + (1.0 - w_player) * context_sim
        
        if player_sim >= 0.6:
            boost_t = ((player_sim - 0.6) / 0.4) ** 2
            combined_sim = base_sim + boost_t * (player_sim - base_sim)
        else:
            combined_sim = base_sim
        
        if exact_overlap >= 2:
            overlap_bonus = 0.045 * (exact_overlap - 1)
            combined_sim = min(1.0, combined_sim + overlap_bonus)
        
        similarity_pct = round(combined_sim * 100.0, 1)
        
        similar_teams_list.append({
            "key": key,
            "team_id": int(h_team_id),
            "season": h_season,
            "similarity_pct": similarity_pct,
            "decade": season_to_decade(h_season),
            "exact_overlap": exact_overlap,
            "starter_ids": h_starter_ids,
            "starters": h_lineup["starters"],
            "style_vector": h_style_dict or {
                "pace": 50.0, "three_point_volume": 50.0, "paint": 50.0,
                "defense": 50.0, "playmaking": 50.0, "rebounding": 50.0
            }
        })
        
    similar_teams_list = select_similar_lineup_candidates(similar_teams_list, limit=8)
    
    top_similar = []
    for item in similar_teams_list:
        tid_str = str(item["team_id"])
        meta = team_metadata.get(tid_str, {})
        t_profile = team_profiles.get(f"{tid_str}:{item['season']}", {})
        
        starters_out = []
        for s in item["starters"]:
            starters_out.append({
                "player_id": int(s["player_id"]),
                "name": s["name"],
                "roles": s.get("roles", ["Secondary Creator"])
            })
            
        top_similar.append({
            "team_id": item["team_id"],
            "team_name": t_profile.get("team_name", meta.get("name", "Unknown Team")),
            "abbreviation": t_profile.get("abbreviation", meta.get("abbreviation", "")),
            "season": item["season"],
            "similarity_pct": item["similarity_pct"],
            "record": t_profile.get("record", "0-0"),
            "style_vector": item["style_vector"],
            "starters": starters_out
        })
        
    players_payload = []
    for _, r in lineup_rows.iterrows():
        pid = int(r["PLAYER_ID"])
        p_roles = player_roles.get(pid, ["Secondary Creator"])
        p_name = str(r["PLAYER_NAME"])
        
        p_min = float(r.get("MIN", 0.0) or 0.0)
        p_gp = float(r.get("GP", 1.0) or 1.0)
        p_mpg = p_min / p_gp if p_gp > 0 else 0.0
        
        players_payload.append({
            "player_id": pid,
            "name": p_name,
            "role": p_roles[0] if p_roles else "Secondary Creator",
            "archetypes": get_player_archetypes(r),
            "pts_per36": round(float(r.get("pts_per36", 0.0) or 0.0), 1),
            "ast_per36": round(float(r.get("ast_per36", 0.0) or 0.0), 1),
            "reb_per36": round(float(r.get("reb_per36", 0.0) or 0.0), 1),
            "stl_per36": round(float(r.get("stl_per36", 0.0) or 0.0), 1),
            "blk_per36": round(float(r.get("blk_per36", 0.0) or 0.0), 1),
            "fg3a_rate": round(float(r.get("fg3a_rate", 0.0) or 0.0), 3)
        })
        
    return {
        "season": season,
        "players": players_payload,
        "style_vector": style_vector,
        "synergy_score": synergy_score,
        "synergy_breakdown": {
            "playmaking": playmaking_adj,
            "spacing": spacing_adj,
            "interior": interior_adj,
            "defense": defense_adj,
            "overlap": overlap_adj
        },
        "strengths": strengths,
        "weaknesses": weaknesses,
        "similar_teams": top_similar
    }
