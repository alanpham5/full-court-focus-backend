import re
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request
from rapidfuzz import fuzz

from models.schemas import LineupSearchPlayer, LineupSynergyRequest, LineupSynergyResponse
from analytics.lineup_synergy import calculate_lineup_synergy

router = APIRouter(prefix="/lineup", tags=["lineups"])

@router.get("/search", response_model=list[LineupSearchPlayer])
def search_season_players(
    request: Request,
    season: str = Query(..., min_length=1),
    q: str = Query(..., min_length=1),
    limit: int = Query(8, ge=1, le=25)
):
    features_df = getattr(request.app.state, "player_season_features_df", None)
    if features_df is None:
        raise HTTPException(
            status_code=503,
            detail="Player season features not loaded."
        )
        
    season_df = features_df[features_df["SEASON"] == season]
    if season_df.empty:
        return []
        
    needle = re.sub(r"[^a-z0-9]+", " ", q.lower()).strip()
    if not needle:
        return []
        
    matches = []
    for idx, row in season_df.iterrows():
        name = str(row["PLAYER_NAME"])
        haystack = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
        if not haystack:
            continue
            
        score = 0.0
        if needle == haystack:
            score = 120.0
        else:
            tokens = haystack.split()
            if needle in tokens:
                score = 115.0
            elif any(token.startswith(needle) for token in tokens):
                score = 110.0
            elif haystack.startswith(needle):
                score = 105.0
            elif needle in haystack:
                score = 100.0
            else:
                token_prefix = max((fuzz.ratio(needle, token) for token in tokens), default=0.0)
                full_name = fuzz.ratio(needle, haystack)
                token_set = fuzz.token_set_ratio(needle, haystack)
                score = max(full_name, token_set, token_prefix * 0.95)
                
        if score >= 35:
            matches.append((row, score))
            
    matches.sort(key=lambda x: (-x[1], str(x[0]["PLAYER_NAME"]).lower()))
    top_matches = matches[:limit]
    
    if not top_matches:
        return []
        
    # Create small df to assign roles on the fly
    matched_df = pd.DataFrame([m[0] for m in top_matches])
    from analytics.lineup_synergy import assign_player_roles_absolute
    roles_dict = assign_player_roles_absolute(matched_df)
    
    results = []
    for row, _score in top_matches:
        pid = int(row["PLAYER_ID"])
        roles = roles_dict.get(pid, ["Secondary Creator"])
        
        min_val = float(row.get("MIN", 0.0) or 0.0)
        gp = float(row.get("GP", 1.0) or 1.0)
        mpg = min_val / gp if gp > 0 else 0.0
        
        results.append(
            LineupSearchPlayer(
                player_id=pid,
                name=str(row["PLAYER_NAME"]),
                team_abbreviation=str(row.get("TEAM_ABBREVIATION", "")),
                role=roles[0] if roles else "Secondary Creator",
                pts_per36=round(float(row.get("pts_per36", 0.0) or 0.0), 1),
                mpg=round(mpg, 1)
            )
        )
        
    return results

@router.post("/synergy", response_model=LineupSynergyResponse)
def get_lineup_synergy(
    body: LineupSynergyRequest,
    request: Request
):
    player_ids = body.player_ids
    season = body.season
    
    if len(player_ids) != 5:
        raise HTTPException(
            status_code=400,
            detail="Exactly 5 player IDs must be provided for a lineup."
        )
        
    player_season_df = getattr(request.app.state, "player_season_features_df", None)
    teams_df = getattr(request.app.state, "teams_df", None)
    starting_lineups = getattr(request.app.state, "starting_lineups", {})
    team_profiles = getattr(request.app.state, "team_profiles", {})
    team_metadata = getattr(request.app.state, "team_metadata", {})
    player_profiles = getattr(request.app.state, "player_profiles", {})
    
    if player_season_df is None or teams_df is None:
        raise HTTPException(
            status_code=503,
            detail="Player season features or teams historical data not loaded."
        )
        
    try:
        res = calculate_lineup_synergy(
            player_ids=player_ids,
            season=season,
            player_season_df=player_season_df,
            teams_df=teams_df,
            starting_lineups=starting_lineups,
            team_profiles=team_profiles,
            team_metadata=team_metadata,
            player_profiles=player_profiles
        )
        return LineupSynergyResponse(**res)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
