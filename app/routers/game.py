import random
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel
import pandas as pd

from analytics.lineup_synergy import (
    calculate_lineup_synergy,
    assign_player_roles_absolute,
    STRENGTH_TRAIT_LABELS,
    WEAKNESS_TRAIT_LABELS,
)

router = APIRouter(prefix="/game", tags=["game"])

STRENGTHS = list(STRENGTH_TRAIT_LABELS.values())
WEAKNESSES = list(WEAKNESS_TRAIT_LABELS.values())

_STRENGTH_LABEL_TO_KEY = {v: k for k, v in STRENGTH_TRAIT_LABELS.items()}
_WEAKNESS_LABEL_TO_KEY = {v: k for k, v in WEAKNESS_TRAIT_LABELS.items()}
_TRAIT_ORDER = list(STRENGTH_TRAIT_LABELS.keys())
_TRAIT_AREA = {
    "playmaking": "playmaking",
    "spacing": "spacing",
    "rebounding": "rebounding",
    "paint_scoring": "paint pressure",
    "defense": "defense",
    "scoring": "scoring depth",
}

CHALLENGE_PERCENTILE = 40.0
PROSPECT_SWAP_CHANCE = 0.25


class LineupOutlook(BaseModel):
    synergy_score: float
    quality_score: float
    fit_score: float
    projected_win_pct: float
    projected_wins: int
    fit_delta_wins: float


class StartGameResponse(BaseModel):
    mode: str
    original_team_id: int
    original_team_name: str
    original_season: str
    original_lineup: List[Dict[str, Any]]
    original_synergy: float
    original_outlook: LineupOutlook
    swap_team_name: str
    swap_team_abbreviation: str
    swap_season: str
    swap_roster: List[Dict[str, Any]]

class EvaluateDiagnosisRequest(BaseModel):
    player_ids: List[int]
    season: str
    selected_traits: List[str]

class EvaluateDiagnosisResponse(BaseModel):
    diagnosis_score: float
    correct_picks: List[str]
    wrong_picks: List[str]
    missed_opportunities: List[str]

class EvaluateSwapRequest(BaseModel):
    original_player_ids: List[int]
    original_season: str
    player_out_id: int
    player_in_id: int
    player_in_season: str
    diagnosis_score: float
    selected_traits: List[str]

class EvaluateSwapResponse(BaseModel):
    original_synergy: float
    new_synergy: float
    synergy_delta: float
    diagnosis_score: float
    synergy_change_score: float
    roster_move_score: float
    final_score: float
    optimization_score: float
    original_outlook: LineupOutlook
    new_outlook: LineupOutlook
    outlook_delta: Dict[str, float]
    breakdown: Dict[str, Any]


def _lineup_df(request: Request):
    df = getattr(request.app.state, "lineup_features_df", None)
    if df is None:
        df = request.app.state.player_season_features_df
    return df


def _join_traits(items: List[str]) -> str:
    items = [str(i) for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _outlook(res: Dict[str, Any]) -> Dict[str, float]:
    projected_win_pct = float(res["projected_win_pct"])
    return {
        "synergy_score": round(float(res["synergy_score"]), 1),
        "quality_score": round(float(res["quality_score"]), 1),
        "fit_score": round(float(res["fit_score"]), 1),
        "projected_win_pct": round(projected_win_pct, 3),
        "projected_wins": round(projected_win_pct * 82),
        "fit_delta_wins": round(float(res["fit_delta_win_pct"]) * 82, 1),
    }


def _outlook_verdict(delta_wins: float) -> str:
    if delta_wins >= 4.0:
        return "a clear upgrade"
    if delta_wins >= 1.5:
        return "a modest upgrade"
    if delta_wins > -0.75:
        return "roughly a wash"
    if delta_wins > -3.0:
        return "a modest step back"
    return "a clear downgrade"


def _trait_keys(labels, mapping):
    return {mapping[l] for l in labels if l in mapping}


def _ordered(keys):
    return [k for k in _TRAIT_ORDER if k in keys]


def _describe_swap(player_out_name, player_in_name, original_res, new_res):
    old_s = _trait_keys(original_res.get("strength_traits", []), _STRENGTH_LABEL_TO_KEY)
    new_s = _trait_keys(new_res.get("strength_traits", []), _STRENGTH_LABEL_TO_KEY)
    old_w = _trait_keys(original_res.get("weakness_traits", []), _WEAKNESS_LABEL_TO_KEY)
    new_w = _trait_keys(new_res.get("weakness_traits", []), _WEAKNESS_LABEL_TO_KEY)

    flips_up = old_w & new_s
    flips_down = old_s & new_w
    gained = (new_s - old_s) - flips_up
    lost = (old_s - new_s) - flips_down
    fixed = (old_w - new_w) - flips_up
    opened = (new_w - old_w) - flips_down

    original_outlook = _outlook(original_res)
    new_outlook = _outlook(new_res)
    win_delta = (
        float(new_res["projected_win_pct"])
        - float(original_res["projected_win_pct"])
    ) * 82
    synergy_delta = float(new_res["synergy_score"]) - float(original_res["synergy_score"])
    quality_delta = float(new_res["quality_score"]) - float(original_res["quality_score"])
    fit_delta = float(new_res["fit_score"]) - float(original_res["fit_score"])
    verdict = _outlook_verdict(win_delta)
    sign = "+" if win_delta >= 0 else ""
    headline = (
        f"Swapping {player_out_name} for {player_in_name} is {verdict} "
        f"({original_outlook['projected_wins']} → {new_outlook['projected_wins']} "
        f"projected wins, {sign}{win_delta:.1f})."
    )

    positives, negatives = [], []
    if quality_delta >= 3.0:
        positives.append("raises the unit's player quality")
    elif quality_delta <= -3.0:
        negatives.append("lowers the unit's player quality")
    if fit_delta >= 3.0:
        positives.append("improves lineup fit")
    elif fit_delta <= -3.0:
        negatives.append("reduces lineup fit")
    for k in _ordered(flips_up):
        positives.append(f"turns its {_TRAIT_AREA[k]} from a weakness into a strength")
    for k in _ordered(gained):
        positives.append(f"adds {STRENGTH_TRAIT_LABELS[k]}")
    for k in _ordered(fixed):
        positives.append(f"shores up its {_TRAIT_AREA[k]}")
    for k in _ordered(flips_down):
        negatives.append(f"flips its {_TRAIT_AREA[k]} from a strength into a weakness")
    for k in _ordered(opened):
        negatives.append(f"opens a hole in its {_TRAIT_AREA[k]}")
    for k in _ordered(lost):
        negatives.append(f"gives up its {STRENGTH_TRAIT_LABELS[k]}")

    if positives and negatives:
        explanation = f"{headline} It {_join_traits(positives)}, but {_join_traits(negatives)}."
    elif positives:
        explanation = f"{headline} It {_join_traits(positives)}."
    elif negatives:
        explanation = f"{headline} It {_join_traits(negatives)}."
    else:
        explanation = f"{headline} The unit's overall profile is largely unchanged."

    effects = {
        "verdict": verdict,
        "synergy_delta": round(float(synergy_delta), 1),
        "projected_wins_delta": round(float(win_delta), 1),
        "quality_delta": round(float(quality_delta), 1),
        "fit_delta": round(float(fit_delta), 1),
        "strengths_gained": [STRENGTH_TRAIT_LABELS[k] for k in _ordered(gained)],
        "strengths_lost": [STRENGTH_TRAIT_LABELS[k] for k in _ordered(lost)],
        "weaknesses_fixed": [WEAKNESS_TRAIT_LABELS[k] for k in _ordered(fixed)],
        "weaknesses_introduced": [WEAKNESS_TRAIT_LABELS[k] for k in _ordered(opened)],
        "improved": [STRENGTH_TRAIT_LABELS[k] for k in _ordered(flips_up)],
        "regressed": [WEAKNESS_TRAIT_LABELS[k] for k in _ordered(flips_down)],
    }
    return explanation, effects


def determine_lineup_traits(player_ids: List[int], season: str, request: Request) -> Dict[str, bool]:
    res = calculate_lineup_synergy(
        player_ids=player_ids,
        season=season,
        player_season_df=_lineup_df(request),
        teams_df=request.app.state.teams_df,
        starting_lineups=request.app.state.starting_lineups,
        team_profiles=request.app.state.team_profiles,
        team_metadata=request.app.state.team_metadata,
        player_profiles=request.app.state.player_profiles,
        compute_similar=False,
        season_baselines=getattr(request.app.state, "season_lineup_baselines", {}),
        synergy_model=getattr(request.app.state, "lineup_synergy_model", None),
    )

    true_traits = set(res.get("strength_traits", [])) | set(res.get("weakness_traits", []))
    return {label: (label in true_traits) for label in (STRENGTHS + WEAKNESSES)}


@router.get("/start", response_model=StartGameResponse)
def start_game(mode: str = Query("current"), request: Request = None):
    if mode not in ("current", "all_time"):
        raise HTTPException(status_code=400, detail="Invalid mode. Must be 'current' or 'all_time'")

    starting_lineups = request.app.state.starting_lineups
    player_season_df = request.app.state.player_season_features_df
    teams_df = request.app.state.teams_df
    team_profiles = request.app.state.team_profiles
    team_metadata = request.app.state.team_metadata
    player_profiles = request.app.state.player_profiles

    if not starting_lineups or player_season_df is None:
        raise HTTPException(status_code=503, detail="Database profiles or player features not loaded.")

                     
    if mode == "current":
                                        
        target_season = "2025-26"
        keys_pool = [k for k in starting_lineups.keys() if k.endswith(f":{target_season}")]
    else:
        keys_pool = list(starting_lineups.keys())

    if not keys_pool:
        raise HTTPException(status_code=500, detail="No lineups found in selected mode pool.")

    synergy_scores = getattr(request.app.state, "lineup_synergy_scores", {}) or {}
    scored_pool = [(k, synergy_scores[k]) for k in keys_pool if k in synergy_scores]

    challenge_keys = keys_pool
    if scored_pool:
        scores_sorted = sorted(s for _, s in scored_pool)
        idx = max(0, int(round((CHALLENGE_PERCENTILE / 100.0) * (len(scores_sorted) - 1))))
        threshold = scores_sorted[idx]
        challenge_keys = [k for k, s in scored_pool if s <= threshold] or [k for k, _ in scored_pool]

    selected_key = None
    original_synergy = 0.0
    selected_data = None
    starter_ids = []
    original_outlook = None

    random.shuffle(challenge_keys)
    for key in challenge_keys[:6]:
        lineup_data = starting_lineups[key]
        candidate_ids = [s["player_id"] for s in lineup_data["starters"]]
        try:
            res = calculate_lineup_synergy(
                player_ids=candidate_ids,
                season=lineup_data["season"],
                player_season_df=player_season_df,
                teams_df=teams_df,
                starting_lineups=starting_lineups,
                team_profiles=team_profiles,
                team_metadata=team_metadata,
                player_profiles=player_profiles,
                compute_similar=False,
                season_baselines=getattr(request.app.state, "season_lineup_baselines", {}),
                synergy_model=getattr(request.app.state, "lineup_synergy_model", None),
            )
            selected_key = key
            original_synergy = res["synergy_score"]
            original_outlook = _outlook(res)
            selected_data = lineup_data
            starter_ids = candidate_ids
            break
        except Exception:
            continue

    if selected_data is None:
        raise HTTPException(status_code=500, detail="Could not select a valid starting lineup from database.")

    original_season = selected_data["season"]
    original_team_id = selected_data["team_id"]
    original_team_name = team_metadata.get(str(original_team_id), {}).get("name", "Unknown Team")

                                     
    original_lineup_players = []
    for p in selected_data["starters"]:
        pid = int(p["player_id"])
        roles = p.get("roles")
        if not roles and p.get("role"):
            roles = [p["role"]]
        roles = list(roles or ["Rotation Piece"])
        
        p_profile = player_profiles.get(str(pid), {})
        draft_year = p_profile.get("draft_year")
        draft_position = p_profile.get("draft_position")
        
                                             
        feat_df = player_season_df[(player_season_df["PLAYER_ID"] == pid) & (player_season_df["SEASON"] == original_season)]
        pts_per36 = 0.0
        ast_per36 = 0.0
        reb_per36 = 0.0
        stl_per36 = 0.0
        blk_per36 = 0.0
        fg3a_rate = 0.0
        fg3m_per36 = 0.0
        if not feat_df.empty:
            r = feat_df.iloc[0]
            pts_per36 = round(float(r.get("pts_per36", 0.0) or 0.0), 1)
            ast_per36 = round(float(r.get("ast_per36", 0.0) or 0.0), 1)
            reb_per36 = round(float(r.get("reb_per36", 0.0) or 0.0), 1)
            stl_per36 = round(float(r.get("stl_per36", 0.0) or 0.0), 1)
            blk_per36 = round(float(r.get("blk_per36", 0.0) or 0.0), 1)
            fg3a_rate = round(float(r.get("fg3a_rate", 0.0) or 0.0), 3)
            r_min = float(r.get("MIN", 0.0) or 0.0)
            fg3m_per36 = round((float(r.get("FG3M", 0.0) or 0.0) / r_min * 36.0) if r_min > 0 else 0.0, 1)

        original_lineup_players.append({
            "player_id": pid,
            "name": str(p["name"]),
            "roles": roles,
            "gp": int(p["gp"]),
            "mpg": float(p["mpg"]),
            "pts_per36": pts_per36,
            "ast_per36": ast_per36,
            "reb_per36": reb_per36,
            "stl_per36": stl_per36,
            "blk_per36": blk_per36,
            "fg3a_rate": fg3a_rate,
            "fg3m_per36": fg3m_per36,
            "draft_year": draft_year,
            "draft_position": draft_position
        })

                               
    swap_team_abbr = None
    swap_season = None
    swap_team_name = "Unknown Team"

    if mode == "current":
        swap_season = "2025-26"
        current_features = player_season_df[player_season_df["SEASON"] == "2025-26"]
        current_teams = list(current_features["TEAM_ABBREVIATION"].dropna().unique())
        
        orig_abbr = team_metadata.get(str(original_team_id), {}).get("abbreviation")
        available_teams = [t for t in current_teams if t != orig_abbr]
        if not available_teams:
            available_teams = current_teams

        swap_team_abbr = random.choice(available_teams)
    else:
        all_keys = [k for k in starting_lineups.keys() if k != selected_key]
        if not all_keys:
            all_keys = list(starting_lineups.keys())
        random_key = random.choice(all_keys)
        random_data = starting_lineups[random_key]
        swap_season = random_data["season"]
        swap_team_abbr = team_metadata.get(str(random_data["team_id"]), {}).get("abbreviation", "ATL")

                                  
    for tid, meta in team_metadata.items():
        if meta.get("abbreviation") == swap_team_abbr:
            swap_team_name = meta.get("name", "Unknown Team")
            break

    roster_df = None
    prospect_pool_df = getattr(request.app.state, "prospect_lineup_features_df", None)
    if prospect_pool_df is not None and not prospect_pool_df.empty:
        pool = prospect_pool_df[prospect_pool_df["SEASON"] == swap_season]
        if not pool.empty and random.random() < PROSPECT_SWAP_CHANCE:
            roster_df = pool
            swap_team_abbr = "DRAFT"
            swap_team_name = f"{int(pool.iloc[0]['draft_class_year'])} Draft Class"

    if roster_df is None:
        roster_df = player_season_df[(player_season_df["SEASON"] == swap_season) & (player_season_df["TEAM_ABBREVIATION"] == swap_team_abbr)]
        if roster_df.empty:
            roster_df = player_season_df[player_season_df["TEAM_ABBREVIATION"] == swap_team_abbr]
            if roster_df.empty:
                roster_df = player_season_df[player_season_df["TEAM_ABBREVIATION"] == "ATL"]
            swap_season = roster_df.iloc[0]["SEASON"]

                             
    roles_dict = assign_player_roles_absolute(roster_df)

    swap_roster = []
    for _, r in roster_df.iterrows():
        pid = int(r["PLAYER_ID"])
        p_roles = roles_dict.get(pid, ["Secondary Creator"])
        
        min_val = float(r.get("MIN", 0.0) or 0.0)
        gp = float(r.get("GP", 1.0) or 1.0)
        mpg = min_val / gp if gp > 0 else 0.0

        p_profile = player_profiles.get(str(pid), {})
        is_prospect = bool(r.get("is_prospect", False)) and not pd.isna(r.get("is_prospect", False))
        counterpart_id = int(r["counterpart_id"]) if is_prospect else None
        swap_roster.append({
            "player_id": pid,
            "name": str(r["PLAYER_NAME"]),
            "team_abbreviation": str(r.get("TEAM_ABBREVIATION", "")),
            "gp": int(gp),
            "mpg": round(mpg, 1),
            "pts_per36": round(float(r.get("pts_per36", 0.0) or 0.0), 1),
            "ast_per36": round(float(r.get("ast_per36", 0.0) or 0.0), 1),
            "reb_per36": round(float(r.get("reb_per36", 0.0) or 0.0), 1),
            "stl_per36": round(float(r.get("stl_per36", 0.0) or 0.0), 1),
            "blk_per36": round(float(r.get("blk_per36", 0.0) or 0.0), 1),
            "fg3a_rate": round(float(r.get("fg3a_rate", 0.0) or 0.0), 3),
            "fg3m_per36": round((float(r.get("FG3M", 0.0) or 0.0) / min_val * 36.0) if min_val > 0 else 0.0, 1),
            "role": p_roles[0] if p_roles else "Secondary Creator",
            "draft_year": p_profile.get("draft_year"),
            "draft_position": p_profile.get("draft_position"),
            "is_prospect": is_prospect,
            "counterpart_id": counterpart_id,
            "counterpart_name": str(r["counterpart_name"]) if is_prospect else None,
        })

                                                     
    swap_roster.sort(key=lambda x: x["mpg"], reverse=True)

    swap_team_display = (
        swap_team_name
        if swap_team_abbr == "DRAFT"
        else f"{swap_season} {swap_team_name}"
    )

    return StartGameResponse(
        mode=mode,
        original_team_id=original_team_id,
        original_team_name=original_team_name,
        original_season=original_season,
        original_lineup=original_lineup_players,
        original_synergy=original_synergy,
        original_outlook=original_outlook,
        swap_team_name=swap_team_display,
        swap_team_abbreviation=swap_team_abbr,
        swap_season=swap_season,
        swap_roster=swap_roster
    )


@router.post("/evaluate-diagnosis", response_model=EvaluateDiagnosisResponse)
def evaluate_diagnosis(body: EvaluateDiagnosisRequest, request: Request):
    player_ids = body.player_ids
    season = body.season
    selected_traits = body.selected_traits

    try:
        traits = determine_lineup_traits(player_ids, season, request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to evaluate lineup traits: {e}")

    correct_picks = []
    wrong_picks = []
    missed_opportunities = []

    for selection in selected_traits:
        if selection in traits:
            if traits[selection]:
                correct_picks.append(selection)
            else:
                wrong_picks.append(selection)

    for trait, is_true in traits.items():
        if is_true and trait not in selected_traits:
            missed_opportunities.append(trait)

                                                                                                       
                                                                        
    total_true_traits = len(correct_picks) + len(missed_opportunities)
    if total_true_traits > 0:
        score = max(0.0, (len(correct_picks) - len(wrong_picks)) / total_true_traits) * 50.0
    else:
        score = 50.0 if len(wrong_picks) == 0 else 0.0
    score = float(score)

    return EvaluateDiagnosisResponse(
        diagnosis_score=score,
        correct_picks=correct_picks,
        wrong_picks=wrong_picks,
        missed_opportunities=missed_opportunities
    )


@router.post("/evaluate-swap", response_model=EvaluateSwapResponse)
def evaluate_swap(body: EvaluateSwapRequest, request: Request):
    original_player_ids = body.original_player_ids
    original_season = body.original_season
    player_out_id = body.player_out_id
    player_in_id = body.player_in_id
    player_in_season = body.player_in_season
    diagnosis_score = max(0.0, min(50.0, float(body.diagnosis_score)))
    selected_traits = body.selected_traits

    player_season_df = _lineup_df(request)
    teams_df = request.app.state.teams_df
    starting_lineups = request.app.state.starting_lineups
    team_profiles = request.app.state.team_profiles
    team_metadata = request.app.state.team_metadata
    player_profiles = request.app.state.player_profiles

    new_player_ids = [pid if pid != player_out_id else player_in_id for pid in original_player_ids]
    player_seasons = [original_season if pid != player_out_id else player_in_season for pid in original_player_ids]

    try:
                                    
        original_res = calculate_lineup_synergy(
            player_ids=original_player_ids,
            season=original_season,
            player_season_df=player_season_df,
            teams_df=teams_df,
            starting_lineups=starting_lineups,
            team_profiles=team_profiles,
            team_metadata=team_metadata,
            player_profiles=player_profiles,
            compute_similar=False,
            season_baselines=getattr(request.app.state, "season_lineup_baselines", {}),
            synergy_model=getattr(request.app.state, "lineup_synergy_model", None),
        )
        original_synergy = original_res["synergy_score"]

                                                            
        new_res = calculate_lineup_synergy(
            player_ids=new_player_ids,
            season=original_season,
            player_season_df=player_season_df,
            teams_df=teams_df,
            starting_lineups=starting_lineups,
            team_profiles=team_profiles,
            team_metadata=team_metadata,
            player_profiles=player_profiles,
            player_seasons=player_seasons,
            compute_similar=False,
            season_baselines=getattr(request.app.state, "season_lineup_baselines", {}),
            synergy_model=getattr(request.app.state, "lineup_synergy_model", None),
        )
        new_synergy = new_res["synergy_score"]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to calculate synergy: {e}")

    synergy_delta = new_synergy - original_synergy
    original_outlook = _outlook(original_res)
    new_outlook = _outlook(new_res)
    projected_win_delta = (
        float(new_res["projected_win_pct"])
        - float(original_res["projected_win_pct"])
    )

    in_row = player_season_df[(player_season_df["PLAYER_ID"] == player_in_id) & (player_season_df["SEASON"] == player_in_season)]
    if not in_row.empty:
        swap_team_abbr = str(in_row.iloc[0]["TEAM_ABBREVIATION"])
        swap_season = player_in_season
    else:
        swap_team_abbr = "ATL"
        swap_season = "2025-26"

    roster_df = player_season_df[(player_season_df["SEASON"] == swap_season) & (player_season_df["TEAM_ABBREVIATION"] == swap_team_abbr)]
    if roster_df.empty:
        roster_df = player_season_df[player_season_df["TEAM_ABBREVIATION"] == swap_team_abbr]
        if roster_df.empty:
            roster_df = player_season_df[player_season_df["TEAM_ABBREVIATION"] == "ATL"]
        swap_season = roster_df.iloc[0]["SEASON"]

    swap_roster_pids = list(roster_df["PLAYER_ID"].dropna().unique())

    max_projected_win_pct = float(original_res["projected_win_pct"])
    for out_pid in original_player_ids:
        for in_pid in swap_roster_pids:
            if in_pid in original_player_ids:
                continue
            candidate_pids = [pid if pid != out_pid else in_pid for pid in original_player_ids]
            candidate_seasons = [original_season if pid != out_pid else swap_season for pid in original_player_ids]
            try:
                cand_res = calculate_lineup_synergy(
                    player_ids=candidate_pids,
                    season=original_season,
                    player_season_df=player_season_df,
                    teams_df=teams_df,
                    starting_lineups=starting_lineups,
                    team_profiles=team_profiles,
                    team_metadata=team_metadata,
                    player_profiles=player_profiles,
                    player_seasons=candidate_seasons,
                    compute_similar=False,
                    season_baselines=getattr(request.app.state, "season_lineup_baselines", {}),
                    synergy_model=getattr(request.app.state, "lineup_synergy_model", None),
                )
                candidate_win_pct = float(cand_res["projected_win_pct"])
                if candidate_win_pct > max_projected_win_pct:
                    max_projected_win_pct = candidate_win_pct
            except Exception:
                continue

    max_projected_win_delta = (
        max_projected_win_pct - float(original_res["projected_win_pct"])
    )
    if max_projected_win_delta > 0.001:
        roster_move_score = max(
            0.0,
            min(50.0, (projected_win_delta / max_projected_win_delta) * 50.0),
        )
    else:
        if projected_win_delta >= -0.001:
            roster_move_score = 50.0
        else:
            roster_move_score = 0.0

    synergy_change_score = float(roster_move_score)
    optimization_score = float(diagnosis_score + roster_move_score)
    final_score = optimization_score

    out_row = player_season_df[(player_season_df["PLAYER_ID"] == player_out_id) & (player_season_df["SEASON"] == original_season)]
    player_out_name = str(out_row.iloc[0]["PLAYER_NAME"]) if not out_row.empty else "Removed Player"

    in_row = player_season_df[(player_season_df["PLAYER_ID"] == player_in_id) & (player_season_df["SEASON"] == player_in_season)]
    player_in_name = str(in_row.iloc[0]["PLAYER_NAME"]) if not in_row.empty else "Added Player"

    explanation, swap_effects = _describe_swap(
        player_out_name,
        player_in_name,
        original_res,
        new_res,
    )

    true_traits = set(original_res.get("strength_traits", [])) | set(original_res.get("weakness_traits", []))
    correct_picks = [s for s in selected_traits if s in true_traits]
    wrong_picks = [s for s in selected_traits if s not in true_traits]
    missed_opportunities = [t for t in (STRENGTHS + WEAKNESSES) if t in true_traits and t not in selected_traits]

    return EvaluateSwapResponse(
        original_synergy=original_synergy,
        new_synergy=new_synergy,
        synergy_delta=synergy_delta,
        diagnosis_score=diagnosis_score,
        synergy_change_score=synergy_change_score,
        roster_move_score=roster_move_score,
        final_score=final_score,
        optimization_score=optimization_score,
        original_outlook=original_outlook,
        new_outlook=new_outlook,
        outlook_delta={
            "synergy_score": round(
                new_outlook["synergy_score"] - original_outlook["synergy_score"],
                1,
            ),
            "quality_score": round(
                new_outlook["quality_score"] - original_outlook["quality_score"],
                1,
            ),
            "fit_score": round(
                new_outlook["fit_score"] - original_outlook["fit_score"],
                1,
            ),
            "projected_win_pct": round(projected_win_delta, 3),
            "projected_wins": round(projected_win_delta * 82, 1),
            "fit_delta_wins": round(
                new_outlook["fit_delta_wins"] - original_outlook["fit_delta_wins"],
                1,
            ),
        },
        breakdown={
            "correct": correct_picks,
            "missed": missed_opportunities,
            "wrong": wrong_picks,
            "explanation": explanation,
            **swap_effects,
        }
    )
