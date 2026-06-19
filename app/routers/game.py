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

CHALLENGE_PERCENTILE = 40.0

                         
class StartGameResponse(BaseModel):
    mode: str
    original_team_id: int
    original_team_name: str
    original_season: str
    original_lineup: List[Dict[str, Any]]
    original_synergy: float
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
    final_score: float
    optimization_score: float
    breakdown: Dict[str, Any]


def determine_lineup_traits(player_ids: List[int], season: str, request: Request) -> Dict[str, bool]:
    res = calculate_lineup_synergy(
        player_ids=player_ids,
        season=season,
        player_season_df=request.app.state.player_season_features_df,
        teams_df=request.app.state.teams_df,
        starting_lineups=request.app.state.starting_lineups,
        team_profiles=request.app.state.team_profiles,
        team_metadata=request.app.state.team_metadata,
        player_profiles=request.app.state.player_profiles,
        compute_similar=False,
        season_baselines=getattr(request.app.state, "season_lineup_baselines", {}),
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
            )
            selected_key = key
            original_synergy = res["synergy_score"]
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
        if not feat_df.empty:
            r = feat_df.iloc[0]
            pts_per36 = round(float(r.get("pts_per36", 0.0) or 0.0), 1)
            ast_per36 = round(float(r.get("ast_per36", 0.0) or 0.0), 1)
            reb_per36 = round(float(r.get("reb_per36", 0.0) or 0.0), 1)
            stl_per36 = round(float(r.get("stl_per36", 0.0) or 0.0), 1)
            blk_per36 = round(float(r.get("blk_per36", 0.0) or 0.0), 1)
            fg3a_rate = round(float(r.get("fg3a_rate", 0.0) or 0.0), 3)
            
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
        draft_year = p_profile.get("draft_year")
        draft_position = p_profile.get("draft_position")

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
            "role": p_roles[0] if p_roles else "Secondary Creator",
            "draft_year": draft_year,
            "draft_position": draft_position
        })

                                                     
    swap_roster.sort(key=lambda x: x["mpg"], reverse=True)

    return StartGameResponse(
        mode=mode,
        original_team_id=original_team_id,
        original_team_name=original_team_name,
        original_season=original_season,
        original_lineup=original_lineup_players,
        original_synergy=original_synergy,
        swap_team_name=f"{swap_season} {swap_team_name}",
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
    diagnosis_score = body.diagnosis_score
    selected_traits = body.selected_traits

    player_season_df = request.app.state.player_season_features_df
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
            season_baselines=getattr(request.app.state, "season_lineup_baselines", {})
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
            season_baselines=getattr(request.app.state, "season_lineup_baselines", {})
        )
        new_synergy = new_res["synergy_score"]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to calculate synergy: {e}")

    synergy_delta = new_synergy - original_synergy

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

    max_new_synergy = original_synergy
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
                    season_baselines=getattr(request.app.state, "season_lineup_baselines", {})
                )
                cand_synergy = cand_res["synergy_score"]
                if cand_synergy > max_new_synergy:
                    max_new_synergy = cand_synergy
            except Exception:
                continue

    max_delta = max_new_synergy - original_synergy
    if max_delta > 0.1:
        synergy_change_score = max(0.0, min(50.0, (synergy_delta / max_delta) * 50.0))
    else:
        if synergy_delta >= -0.1:
            synergy_change_score = 50.0
        else:
            synergy_change_score = 0.0

    optimization_score = float(diagnosis_score + synergy_change_score)
    final_score = optimization_score

    out_row = player_season_df[(player_season_df["PLAYER_ID"] == player_out_id) & (player_season_df["SEASON"] == original_season)]
    player_out_name = str(out_row.iloc[0]["PLAYER_NAME"]) if not out_row.empty else "Removed Player"

    in_row = player_season_df[(player_season_df["PLAYER_ID"] == player_in_id) & (player_season_df["SEASON"] == player_in_season)]
    player_in_name = str(in_row.iloc[0]["PLAYER_NAME"]) if not in_row.empty else "Added Player"

    spacing_change = new_res["synergy_breakdown"]["spacing"] - original_res["synergy_breakdown"]["spacing"]
    playmaking_change = new_res["synergy_breakdown"]["playmaking"] - original_res["synergy_breakdown"]["playmaking"]
    defense_change = new_res["synergy_breakdown"]["defense"] - original_res["synergy_breakdown"]["defense"]
    interior_change = new_res["synergy_breakdown"]["interior"] - original_res["synergy_breakdown"]["interior"]
    overlap_change = new_res["synergy_breakdown"]["overlap"] - original_res["synergy_breakdown"]["overlap"]

    reasons = []
    if spacing_change > 0.5:
        reasons.append(f"boosted spacing (+{spacing_change:.1f})")
    elif spacing_change < -0.5:
        reasons.append(f"clogged spacing ({spacing_change:.1f})")

    if playmaking_change > 0.5:
        reasons.append(f"improved ball movement (+{playmaking_change:.1f})")
    elif playmaking_change < -0.5:
        reasons.append(f"stagnated playmaking ({playmaking_change:.1f})")

    if defense_change > 0.5:
        reasons.append(f"strengthened team defense (+{defense_change:.1f})")
    elif defense_change < -0.5:
        reasons.append(f"weakened team defense ({defense_change:.1f})")

    if interior_change > 0.5:
        reasons.append(f"solidified interior play (+{interior_change:.1f})")
    elif interior_change < -0.5:
        reasons.append(f"exposed frontcourt presence ({interior_change:.1f})")

    if overlap_change > 0.5:
        reasons.append(f"improved role synergy (+{overlap_change:.1f})")
    elif overlap_change < -0.5:
        reasons.append(f"created role redundancies ({overlap_change:.1f})")

    if not reasons:
        explanation = f"Swapping out {player_out_name} for {player_in_name} had minimal net impact on the starting unit's metrics."
    else:
        explanation = f"Swapping out {player_out_name} for {player_in_name} " + ", ".join(reasons[:-1]) + (f" and {reasons[-1]}." if len(reasons) > 1 else f"{reasons[0]}.")

    try:
        traits = determine_lineup_traits(original_player_ids, original_season, request)
    except Exception:
        traits = {}

    correct_picks = []
    missed_opportunities = []
    wrong_picks = []
    for selection in selected_traits:
        if traits.get(selection):
            correct_picks.append(selection)
        else:
            wrong_picks.append(selection)
    for trait, is_true in traits.items():
        if is_true and trait not in selected_traits:
            missed_opportunities.append(trait)

    return EvaluateSwapResponse(
        original_synergy=original_synergy,
        new_synergy=new_synergy,
        synergy_delta=synergy_delta,
        diagnosis_score=diagnosis_score,
        synergy_change_score=synergy_change_score,
        final_score=final_score,
        optimization_score=optimization_score,
        breakdown={
            "correct": correct_picks,
            "missed": missed_opportunities,
            "wrong": wrong_picks,
            "explanation": explanation
        }
    )
