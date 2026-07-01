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

STYLE_W_DEFENSE = 0.296
STYLE_W_REBOUNDING = 0.232
STYLE_W_PAINT = 0.049
STYLE_W_THREE = 0.132
STYLE_W_PLAYMAKING = 0.122
STYLE_W_PACE = -0.108
ROLE_ADJ_WEIGHT = 0.35

SOFT_CAP_KNEE = 84.0
SOFT_CAP_CEIL = 103.0

SHOOTER_MIN_3PM_PER36 = 1.8


def three_point_makes_per36(row: pd.Series) -> float:
    p_min = float(row.get("MIN", 0.0) or 0.0)
    if p_min <= 0.0:
        return 0.0
    fg3m = float(row.get("FG3M", 0.0) or 0.0)
    return fg3m / p_min * 36.0

STRENGTH_TRAIT_LABELS = {
    "playmaking": "Playmaking & Ball Movement",
    "spacing": "Floor Spacing & Shooting",
    "rebounding": "Rebounding & Interior Presence",
    "paint_scoring": "Paint Scoring Pressure",
    "defense": "Team Defense",
    "scoring": "Scoring Firepower",
}
WEAKNESS_TRAIT_LABELS = {
    "playmaking": "Limited Playmaking",
    "spacing": "Poor Spacing",
    "rebounding": "Frontcourt / Rebounding Vulnerability",
    "paint_scoring": "Limited Paint Pressure",
    "defense": "Defensive Limitations",
    "scoring": "Scoring Depth Concerns",
}


def _soft_cap_score(raw: float) -> float:
    if raw <= SOFT_CAP_KNEE:
        return raw
    span = SOFT_CAP_CEIL - SOFT_CAP_KNEE
    return SOFT_CAP_KNEE + span * (1.0 - float(np.exp(-(raw - SOFT_CAP_KNEE) / span)))





_PLAYER_SKILL_COLS = {
    "shooting": "fg3a_rate_pctile",
    "scoring": "pts_per36_pctile",
    "efficiency": "ts_pct_pctile",
    "passing": "ast_per36_pctile",
    "creation": "ast_pct_pctile",
    "rebounding": "reb_per36_pctile",
    "rim_protection": "blk_per36_pctile",
    "perimeter_defense": "stl_per36_pctile",
    "rim_pressure": "fta_rate_pctile",
}


def _pctile(row: pd.Series, col: str, default: float = 50.0) -> float:
    val = row.get(col, default)
    try:
        f = float(val)
    except (TypeError, ValueError):
        return default
    return f if np.isfinite(f) else default


def _humanize_names(names: List[str]) -> str:
    names = [n for n in names if n]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def build_player_descriptors(
    lineup_rows: pd.DataFrame, player_roles: Dict[int, List[str]]
) -> List[Dict[str, Any]]:
    descriptors: List[Dict[str, Any]] = []
    for _, r in lineup_rows.iterrows():
        pid = int(r["PLAYER_ID"])
        roles = player_roles.get(pid, ["Secondary Creator"])
        skills = {k: _pctile(r, col) for k, col in _PLAYER_SKILL_COLS.items()}
        makes36 = three_point_makes_per36(r)

        playmaking = max(skills["passing"], skills["creation"])
        defense = max(skills["perimeter_defense"], skills["rim_protection"])
        descriptors.append({
            "id": pid,
            "name": str(r["PLAYER_NAME"]),
            "role": roles[0] if roles else "Secondary Creator",
            "roles": roles,
            "fg3m_per36": makes36,
            "is_shooter": makes36 >= SHOOTER_MIN_3PM_PER36,
            "shooting": skills["shooting"],
            "scoring": skills["scoring"],
            "efficiency": skills["efficiency"],
            "playmaking": playmaking,
            "rebounding": skills["rebounding"],
            "rim_protection": skills["rim_protection"],
            "perimeter_defense": skills["perimeter_defense"],
            "rim_pressure": skills["rim_pressure"],
            "defense": defense,
        })
    return descriptors


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
    makes36_sum = 0.0

    for r in player_rows:
        makes36_sum += three_point_makes_per36(r)
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
        "space": makes36_sum / len(player_rows) if player_rows else 0.0,
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

def _build_players_payload(lineup_rows: pd.DataFrame, player_roles: Dict[int, List[str]]) -> List[Dict[str, Any]]:
    from analytics.player_profiles.archetypes import get_player_archetypes
    players_payload = []
    for _, r in lineup_rows.iterrows():
        pid = int(r["PLAYER_ID"])
        p_roles = player_roles.get(pid, ["Secondary Creator"])
        p_min = float(r.get("MIN", 0.0) or 0.0)
        fg3m_per36 = (float(r.get("FG3M", 0.0) or 0.0) / p_min * 36.0) if p_min > 0 else 0.0
        ip = r.get("is_prospect", False)
        is_prospect = bool(ip) and not pd.isna(ip)
        cid = r.get("counterpart_id")
        cname = r.get("counterpart_name")
        prospect_id = r.get("prospect_id")
        players_payload.append({
            "player_id": pid,
            "name": str(r["PLAYER_NAME"]),
            "role": p_roles[0] if p_roles else "Secondary Creator",
            "archetypes": get_player_archetypes(r),
            "pts_per36": round(float(r.get("pts_per36", 0.0) or 0.0), 1),
            "ast_per36": round(float(r.get("ast_per36", 0.0) or 0.0), 1),
            "reb_per36": round(float(r.get("reb_per36", 0.0) or 0.0), 1),
            "stl_per36": round(float(r.get("stl_per36", 0.0) or 0.0), 1),
            "blk_per36": round(float(r.get("blk_per36", 0.0) or 0.0), 1),
            "fg3a_rate": round(float(r.get("fg3a_rate", 0.0) or 0.0), 3),
            "fg3m_per36": round(fg3m_per36, 1),
            "is_prospect": is_prospect,
            "prospect_id": str(prospect_id) if is_prospect and prospect_id is not None and not pd.isna(prospect_id) else None,
            "counterpart_id": int(cid) if is_prospect and cid is not None and not pd.isna(cid) else None,
            "counterpart_name": str(cname) if is_prospect and cname is not None and not pd.isna(cname) else None,
        })
    return players_payload


_TRAIT_SKILL = {
    "playmaking": "playmaking",
    "spacing": "shooting",
    "rebounding": "rebounding",
    "paint_scoring": "rim_pressure",
    "defense": "defense",
    "scoring": "scoring",
}
_TRAIT_DIM = {
    "playmaking": "playmaking",
    "spacing": "spacing",
    "rebounding": "rebounding",
    "paint_scoring": "paint",
    "defense": "defense",
    "scoring": "scoring",
}
_TRAIT_CHANNEL = {
    "playmaking": "playmaking",
    "spacing": "spacing",
    "rebounding": "interior",
    "paint_scoring": "interior",
    "defense": "defense",
}

_GOOD = 62.0
_STRONG = 72.0
_WEAK = 43.0
_RESCUE = 60.0
_INTERACTION_BASE = 26.0
_CHANNEL_WEIGHT = 1.1
_CHANNEL_WEAKNESS_GATE = -5.0
_DIAGNOSIS_LIMIT = 4


def generate_lineup_diagnoses(
    *,
    descriptors: List[Dict[str, Any]],
    dims: Dict[str, float],
    composition: Dict[str, int],
    synergy_score: float,
    channels: Dict[str, float],
) -> tuple:
    def chan(trait: str) -> float:
        key = _TRAIT_CHANNEL.get(trait)
        return float(channels.get(key, 0.0)) if key else 0.0

    def carriers(skill: str, gate: float = 55.0, n: int = 3) -> List[Dict[str, Any]]:
        elig = [d for d in descriptors if d.get(skill, 0.0) >= gate]
        return sorted(elig, key=lambda d: d.get(skill, 0.0), reverse=True)[:n]

    def best(skill: str) -> float:
        return max((d.get(skill, 0.0) for d in descriptors), default=50.0)

    def names(ds: List[Dict[str, Any]]) -> str:
        return _humanize_names([d["name"] for d in ds])


    assess: Dict[str, Dict[str, Any]] = {}
    for trait, skill in _TRAIT_SKILL.items():
        dim = float(dims.get(_TRAIT_DIM[trait], 50.0))
        support = best(skill)
        assess[trait] = {
            "dim": dim,
            "support": support,
            "signal": 0.5 * dim + 0.5 * support,
            "carriers": carriers(skill),
        }

    def sig(trait: str) -> float:
        return assess[trait]["signal"]

    non_shooters = sorted(
        [d for d in descriptors if not d["is_shooter"]],
        key=lambda d: d.get("scoring", 0.0), reverse=True,
    )

    insights: List[Dict[str, Any]] = []

    def add(category: str, trait, traits: set, weight: float, headline: str, text: str):
        insights.append({
            "category": category, "trait": trait, "traits": traits,
            "weight": float(weight), "headline": headline, "text": text,
        })

    pm_carrier = assess["playmaking"]["carriers"][:1]
    pm_names = names(pm_carrier)
    pm_id = pm_carrier[0]["id"] if pm_carrier else None
    shooter_names = names([d for d in descriptors if d["is_shooter"] and d["id"] != pm_id][:3])


    if sig("playmaking") >= _GOOD and sig("spacing") >= _GOOD - 2:
        add("strength", "playmaking", {"playmaking", "spacing"},
            _INTERACTION_BASE + 0.5 * (sig("playmaking") - 50) + 0.5 * (sig("spacing") - 50),
            "Drive-and-Kick Engine",
            f"{pm_names or 'The lead creator'} works downhill with shooters spaced around the arc; "
            f"defenses are forced to pick their poison between collapsing on the drive and chasing "
            f"{shooter_names or 'kick-out threats'}, and this group makes them pay either way.")


    if sig("paint_scoring") >= _GOOD and sig("spacing") >= _GOOD:
        rim_names = names(assess["paint_scoring"]["carriers"][:2])
        add("strength", "paint_scoring", {"paint_scoring", "spacing"},
            _INTERACTION_BASE + 0.5 * (sig("paint_scoring") - 50) + 0.5 * (sig("spacing") - 50),
            "Inside-Out Pressure",
            f"Interior pressure from {rim_names or 'the frontcourt'} bends the defense inward while "
            f"shooters wait on the perimeter — a collapse-and-kick dynamic that is brutal to defend "
            f"without elite rim protection.")

    if sig("defense") >= _GOOD and sig("rebounding") >= _GOOD:
        def_names = names(assess["defense"]["carriers"][:2])
        add("strength", "defense", {"defense", "rebounding"},
            _INTERACTION_BASE + 0.5 * (sig("defense") - 50) + 0.5 * (sig("rebounding") - 50),
            "Two-Way Wall",
            f"With {def_names or 'this group'} anchoring the back line, the unit contests shots and then "
            f"cleans the glass; opponents get one contested look and rarely a second, which travels in any era.")


    if sig("scoring") >= _STRONG and (sig("playmaking") >= 53 or sig("spacing") >= 53):
        score_names = names(assess["scoring"]["carriers"][:2])
        add("strength", "scoring", {"scoring"},
            _INTERACTION_BASE + 0.6 * (sig("scoring") - 50),
            "Shot-Making Firepower",
            f"{score_names or 'Multiple scorers'} give this unit shot-creation it can lean on when "
            f"possessions break down — there is always someone who can manufacture a bucket late in the clock.")

    if sig("playmaking") >= _GOOD and assess["spacing"]["dim"] <= _WEAK and best("shooting") < _RESCUE + 5:
        clog = names(non_shooters[:2])
        add("weakness", "spacing", {"spacing"},
            _INTERACTION_BASE + 0.5 * (sig("playmaking") - 50) + 0.4 * (50 - assess["spacing"]["dim"]),
            "Cramped Driving Lanes",
            f"{pm_names or 'The lead creator'} has little room to operate — with "
            f"{clog or 'multiple non-shooters'} offering no perimeter threat, help defenders sink into "
            f"the paint and wall off the rim, blunting the very creation that should drive this offense.")

    if sig("defense") >= _GOOD and assess["rebounding"]["dim"] <= _WEAK and best("rebounding") < _RESCUE:
        add("weakness", "rebounding", {"rebounding"},
            _INTERACTION_BASE + 0.5 * (sig("defense") - 50) + 0.4 * (50 - assess["rebounding"]["dim"]),
            "Second-Chance Leak",
            "The unit forces tough misses but lacks the size to close possessions, gifting opponents "
            "offensive rebounds that quietly undo all the good defense.")

    if sig("spacing") >= _GOOD + 2 and assess["paint_scoring"]["dim"] <= 40 and best("rim_pressure") < _RESCUE:
        add("weakness", "paint_scoring", {"paint_scoring"},
            _INTERACTION_BASE + 0.5 * (sig("spacing") - 50) + 0.4 * (50 - assess["paint_scoring"]["dim"]),
            "Jump-Shot Dependent",
            "With little rim pressure to fall back on, the offense lives and dies by the three; on cold "
            "shooting nights there's no reliable way to manufacture easy points at the rim or the line.")


    if composition.get("creators", 0) >= 3 and sig("playmaking") < 48 and best("playmaking") < _RESCUE + 8:
        score_names = names(assess["scoring"]["carriers"][:2])
        add("weakness", "playmaking", {"playmaking"},
            _INTERACTION_BASE + (50 - sig("playmaking")) + 3.0 * composition["creators"],
            "No Lead Initiator",
            f"{score_names or 'Several scorers'} all want touches, but no true table-setter organizes "
            f"the halfcourt; possessions risk devolving into isolation as the ball sticks.")

    _SINGLE_STRENGTH = {
        "playmaking": ("Connective Passing",
            "ball movement that keeps the offense humming and finds the open man before the defense recovers"),
        "spacing": ("Floor Spacing",
            "enough shooting to stretch defenses and keep the paint honest"),
        "rebounding": ("Glass Control",
            "size and rebounding instincts that secure extra possessions and limit opponents to one shot"),
        "paint_scoring": ("Rim Pressure",
            "a steady diet of attempts at the rim and the foul line that pressures opposing bigs"),
        "defense": ("Defensive Bite",
            "the personnel to defend on the ball and protect the rim, making clean looks hard to come by"),
        "scoring": ("Scoring Punch",
            "shot-makers who can get a bucket when the offense bogs down"),
    }
    _SINGLE_WEAKNESS = {
        "playmaking": ("Limited Ball Movement",
            "without a real creator the offense leans on isolation and predictable sets"),
        "spacing": ("Spacing Concerns",
            "thin perimeter shooting lets defenders pack the paint and cut off driving lanes"),
        "rebounding": ("Undersized on the Glass",
            "a lack of interior size leaves the unit vulnerable to being out-rebounded"),
        "paint_scoring": ("Soft Interior Scoring",
            "little pressure at the rim forces the group into a steady diet of jumpers"),
        "defense": ("Defensive Exposure",
            "minus defenders give opponents mismatches they can hunt repeatedly"),
        "scoring": ("Scoring Droughts",
            "a shortage of shot creation makes this unit prone to extended cold stretches"),
    }

    for trait in _TRAIT_SKILL:
        a = assess[trait]
        c = chan(trait)
        carrier_names = names(a["carriers"][:2])

        is_strength = a["signal"] >= _GOOD and (a["dim"] >= 47 or a["support"] >= 82) and c > _CHANNEL_WEAKNESS_GATE
        if is_strength:
            head, body = _SINGLE_STRENGTH[trait]
            lead = f"Led by {carrier_names}, this group brings " if carrier_names else "This group brings "
            add("strength", trait, {trait}, (a["signal"] - 50),
                head, f"{lead}{body}.")
        elif (a["dim"] <= _WEAK and a["support"] < _RESCUE) or c <= _CHANNEL_WEAKNESS_GATE:
            head, body = _SINGLE_WEAKNESS[trait]
            add("weakness", trait, {trait},
                (50 - a["dim"]) + 0.25 * (_RESCUE - a["support"]) + max(0.0, -c),
                head, body[0].upper() + body[1:] + ".")
        elif a["dim"] < 50 and a["support"] < 55:
            head, body = _SINGLE_WEAKNESS[trait]
            add("weakness", trait, {trait}, (50 - a["dim"]) * 0.45,
                f"Minor: {head}", f"To a lesser degree, {body}.")

    overlap_chan = float(channels.get("overlap", 0.0))
    if overlap_chan <= _CHANNEL_WEAKNESS_GATE:
        add("weakness", None, {"__overlap__"}, _INTERACTION_BASE - 4 + max(0.0, -overlap_chan),
            "Redundant Roles",
            "Several players occupy the same role and compete for the same touches and floor space, "
            "so the unit's parts overlap instead of complementing one another.")

    for ins in insights:
        if not ins["trait"]:
            continue
        c = chan(ins["trait"])
        ins["weight"] += _CHANNEL_WEIGHT * c if ins["category"] == "strength" else -_CHANNEL_WEIGHT * c

    if synergy_score >= 70:
        max_str, max_weak = _DIAGNOSIS_LIMIT, 1
    elif synergy_score >= 58:
        max_str, max_weak = 3, 2
    elif synergy_score >= 46:
        max_str, max_weak = 2, 2
    elif synergy_score >= 34:
        max_str, max_weak = 2, 3
    else:
        max_str, max_weak = 1, _DIAGNOSIS_LIMIT

    insights.sort(key=lambda x: x["weight"], reverse=True)
    strengths: List[str] = []
    weaknesses: List[str] = []
    strength_traits: List[str] = []
    weakness_traits: List[str] = []
    used_traits: set = set()

    for ins in insights:
        if ins["traits"] & used_traits:
            continue
        if ins["category"] == "strength" and len(strengths) < max_str:
            strengths.append(f"{ins['headline']} - {ins['text']}")
            if ins["trait"]:
                strength_traits.append(ins["trait"])
            used_traits |= ins["traits"]
        elif ins["category"] == "weakness" and len(weaknesses) < max_weak:
            weaknesses.append(f"{ins['headline']} - {ins['text']}")
            if ins["trait"]:
                weakness_traits.append(ins["trait"])
            used_traits |= ins["traits"]


    if not strengths:
        t = max(_TRAIT_SKILL, key=sig)
        head, body = _SINGLE_STRENGTH[t]
        cn = names(assess[t]["carriers"][:2])
        lead = f"Led by {cn}, this group's most reliable edge is " if cn else "This group's most reliable edge is "
        strengths.append(f"Relative Strength: {head} - {lead}{body}.")
        strength_traits.append(t)
        used_traits.add(t)
    if not weaknesses:
        candidates = [t for t in _TRAIT_SKILL if t not in used_traits]
        t = min(candidates or list(_TRAIT_SKILL), key=lambda x: assess[x]["dim"])
        if assess[t]["dim"] < 50:
            head, body = _SINGLE_WEAKNESS[t]
            weaknesses.append(f"Soft Spot: {head} - {body[0].upper() + body[1:]}.")
            weakness_traits.append(t)
        else:
            weaknesses.append(
                "Well-Rounded - No dimension stands out as a glaring liability; opponents "
                "lack an obvious matchup to attack against this group.")


    pace = float(dims.get("pace", 50.0))
    identities: List[tuple] = []
    if composition.get("shooters", 0) >= 4:
        identities.append(("Five-Out Spacing", sig("spacing") + 10))
    if composition.get("playmakers", 0) == 1 and composition.get("shooters", 0) >= 3:
        identities.append(("Heliocentric Attack", sig("playmaking") + sig("spacing")))
    elif composition.get("playmakers", 0) >= 3:
        identities.append(("Positionless Playmaking", sig("playmaking") + 8))
    if sig("defense") >= _GOOD and sig("scoring") < _GOOD:
        identities.append(("Defense-First Identity", sig("defense") + (60 - min(60, sig("scoring")))))
    if sig("defense") >= _STRONG:
        identities.append(("Lockdown Unit", sig("defense") + 6))
    if pace >= 70 and sig("scoring") >= 55:
        identities.append(("Transition Attack", pace))
    elif pace <= 35:
        identities.append(("Grind-It-Out Halfcourt", 100 - pace))
    if composition.get("bigs", 0) >= 2 and sig("paint_scoring") >= _GOOD:
        identities.append(("Interior-Oriented", sig("paint_scoring") + 6))
    if sig("paint_scoring") >= _GOOD and sig("spacing") >= _GOOD:
        identities.append(("Inside-Out Balance", sig("paint_scoring") + sig("spacing")))
    if not identities:
        identities.append(("Balanced Lineup", 50.0))
    identities.sort(key=lambda x: x[1], reverse=True)
    primary_identity = identities[0][0]
    secondary_identity = identities[1][0] if len(identities) > 1 else None

    st_labels = [STRENGTH_TRAIT_LABELS[t] for t in strength_traits]
    wt_labels = [WEAKNESS_TRAIT_LABELS[t] for t in weakness_traits]

    return strengths, weaknesses, st_labels, wt_labels, primary_identity, secondary_identity


def calculate_lineup_synergy(
    player_ids: List[int],
    season: str,
    player_season_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    starting_lineups: Dict[str, Any],
    team_profiles: Dict[str, Any],
    team_metadata: Dict[str, Any],
    player_profiles: Dict[str, Any] = None,
    player_seasons: List[str] = None,
    compute_similar: bool = True,
    season_baselines: Dict[str, Any] = None,
) -> Dict[str, Any]:
    season_players = player_season_df[player_season_df["SEASON"] == season]
    if player_seasons:
        rows = []
        for pid, p_season in zip(player_ids, player_seasons):
            row = player_season_df[(player_season_df["PLAYER_ID"] == pid) & (player_season_df["SEASON"] == p_season)]
            if not row.empty:
                rows.append(row.iloc[0])
            else:
                                                                                     
                fallback_row = player_season_df[player_season_df["PLAYER_ID"] == pid]
                if not fallback_row.empty:
                    rows.append(fallback_row.iloc[0])
                else:
                    raise ValueError(f"Player ID {pid} not found in database")
        lineup_rows = pd.DataFrame(rows)
    else:
        if season_players.empty:
            raise ValueError(f"No player data found for season {season}")
            
        lineup_rows = season_players[season_players["PLAYER_ID"].isin(player_ids)]
        found_ids = set(lineup_rows["PLAYER_ID"].unique())
        missing_ids = [pid for pid in player_ids if pid not in found_ids]
        if missing_ids:
            raise ValueError(f"Player IDs {missing_ids} not found in season {season}")
            
        lineup_rows = lineup_rows.set_index("PLAYER_ID").loc[player_ids].reset_index()

    player_roles = assign_player_roles_absolute(lineup_rows)
    
    player_rows_list = [lineup_rows.iloc[i] for i in range(len(lineup_rows))]
    custom_stats = compute_lineup_collective_stats(player_rows_list, player_roles)
    
    team_abbr_to_id = {v["abbreviation"]: int(k) for k, v in team_metadata.items()}

    from analytics.season_baselines import (
        compute_lineup_metrics,
        get_season_baseline,
        percentile_in,
    )

    season_baseline = get_season_baseline(
        season, season_baselines, player_season_df, teams_df, starting_lineups, team_abbr_to_id
    )
    custom_metrics = compute_lineup_metrics(
        lineup_rows, player_roles, teams_df, team_abbr_to_id, season
    )

    if season_baseline and season_baseline.get("metrics"):
        bm = season_baseline["metrics"]
        pace_pct = percentile_in(bm["pace"], custom_metrics["pace"])
        space_pct = percentile_in(bm["space"], custom_metrics["space"])
        paint_pct = percentile_in(bm["paint_fga"], custom_metrics["paint_fga"])
        defense_pct = percentile_in(bm["def_score"], custom_metrics["def_score"])
        ast_vol_pct = percentile_in(bm["ast"], custom_metrics["ast"])
        ast_rate_pct = percentile_in(bm["ast_pct"], custom_metrics["ast_pct"])
        playmaking_pct = 0.55 * ast_vol_pct + 0.45 * ast_rate_pct
        rebounding_pct = percentile_in(bm["reb"], custom_metrics["reb"])
        mean_starting_reb = bm["reb"]["mean"] or 1.0
        mean_starting_blk = bm["blk"]["mean"] or 1.0
    else:
        pace_pct = space_pct = paint_pct = defense_pct = playmaking_pct = rebounding_pct = 50.0
        mean_starting_reb = custom_metrics["reb"] or 1.0
        mean_starting_blk = custom_metrics["blk"] or 1.0

    _t_axes = ["pace", "three_point_volume", "paint", "defense", "playmaking", "rebounding"]
    _t_vals = {a: [] for a in _t_axes}
    for _, r in lineup_rows.iterrows():
        abbr = r.get("TEAM_ABBREVIATION", "")
        p_team_id = team_abbr_to_id.get(abbr)
        p_season = r.get("SEASON", season)
        style = team_profiles.get(f"{p_team_id}:{p_season}", {}).get("style_vector", {})
        for a in _t_axes:
            _t_vals[a].append(float(style.get(a, 50.0)) if style else 50.0)
    avg_t = {a: (float(np.mean(v)) if v else 50.0) for a, v in _t_vals.items()}

    pace_pct = 0.40 * pace_pct + 0.60 * avg_t["pace"]
    paint_pct = 0.40 * paint_pct + 0.60 * avg_t["paint"]
    defense_pct = 0.30 * defense_pct + 0.70 * avg_t["defense"]
    playmaking_pct = 0.60 * playmaking_pct + 0.40 * avg_t["playmaking"]
    rebounding_pct = 0.40 * rebounding_pct + 0.60 * avg_t["rebounding"]

    style_vector = {
        "pace": round(min(100.0, max(0.0, pace_pct)), 1),
        "three_point_volume": round(min(100.0, max(0.0, space_pct)), 1),
        "paint": round(min(100.0, max(0.0, paint_pct)), 1),
        "defense": round(min(100.0, max(0.0, defense_pct)), 1),
        "playmaking": round(min(100.0, max(0.0, playmaking_pct)), 1),
        "rebounding": round(min(100.0, max(0.0, rebounding_pct)), 1)
    }
    
    season_cache = {}
    def get_season_arrays(s_val):
        if s_val not in season_cache:
            s_players = player_season_df[player_season_df["SEASON"] == s_val]
                                                                                               
            pool = s_players[s_players["MIN"] >= 1000]
            if pool.empty:
                pool = s_players
            season_cache[s_val] = {
                "pts": pool["PTS"].dropna().to_numpy(),
                "ast": pool["AST"].dropna().to_numpy(),
                "reb": pool["REB"].dropna().to_numpy(),
                "stl": pool["STL"].dropna().to_numpy(),
                "blk": pool["BLK"].dropna().to_numpy(),
                "ts": pool["ts_pct"].dropna().to_numpy(),
                "mpg": pool["mpg"].dropna().to_numpy(),
            }
        return season_cache[s_val]

    player_ratings = []
    for _, r in lineup_rows.iterrows():
        p_season = r.get("SEASON", season)
        arrays = get_season_arrays(p_season)
        
        pts_tot_p = percentileofscore(arrays["pts"], float(r.get("PTS", 0.0) or 0.0), kind="rank")
        ast_tot_p = percentileofscore(arrays["ast"], float(r.get("AST", 0.0) or 0.0), kind="rank")
        reb_tot_p = percentileofscore(arrays["reb"], float(r.get("REB", 0.0) or 0.0), kind="rank")
        stl_tot_p = percentileofscore(arrays["stl"], float(r.get("STL", 0.0) or 0.0), kind="rank")
        blk_tot_p = percentileofscore(arrays["blk"], float(r.get("BLK", 0.0) or 0.0), kind="rank")
        ts_p = percentileofscore(arrays["ts"], float(r.get("ts_pct", 0.0) or 0.0), kind="rank")
        mpg_p = percentileofscore(arrays["mpg"], float(r.get("mpg", 0.0) or 0.0), kind="rank")
        
        rating = 0.25 * pts_tot_p + 0.15 * ast_tot_p + 0.10 * reb_tot_p + 0.05 * stl_tot_p + 0.05 * blk_tot_p + 0.15 * ts_p + 0.25 * mpg_p
        player_ratings.append(rating)
        
    baseline_talent = 0.88 * np.mean(player_ratings) + 0.12 * np.min(player_ratings)
    
    playmakers_count = sum(1 for pid in player_ids if "Playmaker" in player_roles.get(pid, []))
    creators_count = sum(1 for pid in player_ids if "Secondary Creator" in player_roles.get(pid, []) or "Designated Scorer" in player_roles.get(pid, []))
    
    shooters_count = 0
    for _, r in lineup_rows.iterrows():
        if three_point_makes_per36(r) >= SHOOTER_MIN_3PM_PER36:
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

    strong_rebounders = 0
    rim_protectors = 0
    elite_passers = 0
    for _, r in lineup_rows.iterrows():
        gp = float(r.get("GP", 1.0) or 1.0) or 1.0
        rpg = float(r.get("REB", 0.0) or 0.0) / gp
        bpg = float(r.get("BLK", 0.0) or 0.0) / gp
        apg = float(r.get("AST", 0.0) or 0.0) / gp
        reb36 = float(r.get("reb_per36", 0.0) or 0.0)
        blk36 = float(r.get("blk_per36", 0.0) or 0.0)
        if rpg >= 9.5 or reb36 >= 11.5:
            strong_rebounders += 1
        if bpg >= 1.2 or blk36 >= 1.6:
            rim_protectors += 1
        if apg >= 6.0:
            elite_passers += 1

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

    if strong_rebounders >= 2 or (reb_ratio >= 1.05 and blk_ratio >= 1.15):
        interior_adj = 5.0
    elif (strong_rebounders >= 1 and reb_ratio >= 0.95) or rim_protectors >= 2:
        interior_adj = 2.0
    elif reb_ratio > 1.25 and blk_ratio > 1.60:
        interior_adj = -10.0
    elif (reb_ratio < 0.88 or blk_ratio < 0.70) and strong_rebounders == 0 and rim_protectors == 0:
        interior_adj = -15.0
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

    if space_pct < 45.0:
        spacing_adj = min(-3.0, spacing_adj)

    if (rebounding_pct < 45.0 or paint_pct < 45.0) and strong_rebounders < 2 and rim_protectors < 2:
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
            
    defense_chan = ROLE_ADJ_WEIGHT * defense_adj + STYLE_W_DEFENSE * (style_vector["defense"] - 50.0)
    interior_chan = (ROLE_ADJ_WEIGHT * interior_adj
                     + STYLE_W_REBOUNDING * (style_vector["rebounding"] - 50.0)
                     + STYLE_W_PAINT * (style_vector["paint"] - 50.0))
    spacing_chan = ROLE_ADJ_WEIGHT * spacing_adj + STYLE_W_THREE * (style_vector["three_point_volume"] - 50.0)
    playmaking_chan = ROLE_ADJ_WEIGHT * playmaking_adj + STYLE_W_PLAYMAKING * (style_vector["playmaking"] - 50.0)
    overlap_chan = ROLE_ADJ_WEIGHT * overlap_adj + STYLE_W_PACE * (style_vector["pace"] - 50.0)

    synergy_raw = baseline_talent + defense_chan + interior_chan + spacing_chan + playmaking_chan + overlap_chan
    synergy_score = round(min(100.0, max(0.0, _soft_cap_score(synergy_raw))), 1)

    synergy_breakdown = {
        "playmaking": round(playmaking_chan, 1),
        "spacing": round(spacing_chan, 1),
        "interior": round(interior_chan, 1),
        "defense": round(defense_chan, 1),
        "overlap": round(overlap_chan, 1),
    }

    descriptors = build_player_descriptors(lineup_rows, player_roles)
    diag_dims = {
        "playmaking": playmaking_pct,
        "spacing": space_pct,
        "rebounding": rebounding_pct,
        "paint": paint_pct,
        "defense": defense_pct,
        "pace": pace_pct,
        "scoring": baseline_talent,
    }
    composition = {
        "playmakers": playmakers_count,
        "creators": creators_count,
        "shooters": shooters_count,
        "bigs": bigs_count,
        "defenders": defenders_count,
        "strong_rebounders": strong_rebounders,
        "rim_protectors": rim_protectors,
        "elite_passers": elite_passers,
    }
    strengths, weaknesses, strength_traits, weakness_traits, p_identity, s_identity = generate_lineup_diagnoses(
        descriptors=descriptors,
        dims=diag_dims,
        composition=composition,
        synergy_score=synergy_score,
        channels=synergy_breakdown,
    )

    if not compute_similar:
        return {
            "season": season,
            "players": _build_players_payload(lineup_rows, player_roles),
            "style_vector": style_vector,
            "synergy_score": synergy_score,
            "synergy_breakdown": synergy_breakdown,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "strength_traits": strength_traits,
            "weakness_traits": weakness_traits,
            "primary_identity": p_identity,
            "secondary_identity": s_identity,
            "similar_teams": [],
        }

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
        
    return {
        "season": season,
        "players": _build_players_payload(lineup_rows, player_roles),
        "style_vector": style_vector,
        "synergy_score": synergy_score,
        "synergy_breakdown": synergy_breakdown,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "strength_traits": strength_traits,
        "weakness_traits": weakness_traits,
        "primary_identity": p_identity,
        "secondary_identity": s_identity,
        "similar_teams": top_similar,
    }
