import json
import re

from fastapi import APIRouter, HTTPException, Query, Request
from rapidfuzz import fuzz

from config import (
    PLAYER_METADATA_PATH,
    PLAYER_PROFILES_PATH,
    PLAYER_SEASON_FEATURES_PATH,
)
from models.schemas import (
    PlayerLeaderItem,
    PlayerLeadersResponse,
    PlayerProfileResponse,
    PlayerSearchSuggestion,
    SimilarPlayer,
)
from analytics.player_profiles.archetypes import (
    calculate_adjusted_pfv,
    calculate_apfv,
    calculate_apfv_batch_by_height,
    calculate_pfv,
    height_bucket,
    remove_mpg_adjustment_from_metrics,
)
from analytics.player_profiles.season_profiles import (
    build_season_bundle,
    normalize_season,
)
from analytics.player_profiles.features import height_to_inches

router = APIRouter(prefix="/players", tags=["players"])

PLAYSTYLE_METRIC_KEYS = (
    "pts_per36",
    "reb_per36",
    "ast_per36",
    "blk_per36",
    "stl_per36",
    "ts_pct",
    "tov_per36",
    "efg_pct",
    "ast_pct",
    "fg3a_rate",
    "fta_rate",
    "mpg",
)


@router.get("/search", response_model=list[PlayerSearchSuggestion])
def search_players(
    request: Request,
    q: str = Query(..., min_length=1),
    limit: int = Query(8, ge=1, le=25),
):
    metadata = getattr(request.app.state, "player_metadata", None)
    if metadata is None:
        if not PLAYER_METADATA_PATH.exists():
            return []
        with PLAYER_METADATA_PATH.open() as f:
            metadata = json.load(f)

    ranked = _rank_player_search(q, metadata, limit)
    return [
        PlayerSearchSuggestion(
            player_id=int(pid),
            player_name=metadata[pid]["player_name"],
            role=str(metadata[pid].get("role", "")),
            career_span=str(metadata[pid].get("career_span", "")),
            archetypes=list(metadata[pid].get("archetypes", [])),
        )
        for pid, _score in ranked
    ]


@router.get("/archetypes/{archetype}", response_model=list[PlayerSearchSuggestion])
def players_by_archetype(
    archetype: str,
    request: Request,
    limit: int = Query(25, ge=1, le=100),
):
    metadata = getattr(request.app.state, "player_metadata", None)
    if metadata is None:
        if not PLAYER_METADATA_PATH.exists():
            return []
        with PLAYER_METADATA_PATH.open() as f:
            metadata = json.load(f)
    needle = archetype.lower().replace("-", " ")
    matches = []
    for item in metadata.values():
        labels = [str(a).lower().replace("-", " ") for a in item.get("archetypes", [])]
        if needle in labels:
            matches.append(
                PlayerSearchSuggestion(
                    player_id=int(item["player_id"]),
                    player_name=item["player_name"],
                    role=str(item.get("role", "")),
                    career_span=str(item.get("career_span", "")),
                    archetypes=list(item.get("archetypes", [])),
                )
            )
    return matches[:limit]


LEADER_SKILL_KEYS = (
    "pts_per36",
    "reb_per36",
    "ast_per36",
    "blk_per36",
    "stl_per36",
    "ts_pct",
)
LEADER_SHAPE_CONTRAST = 0.75
LEADER_SORTABLE_KEYS = {"apfv", "height", "weight", *LEADER_SKILL_KEYS}


@router.get("/leaders", response_model=PlayerLeadersResponse)
def get_player_leaders(
    request: Request,
    season: str | None = Query(None),
    role: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
    sort: str = Query("apfv"),
    order: str = Query("desc"),
):
    if season is not None:
        normalized = normalize_season(season)
        if normalized is None:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid season '{season}'. Use 'YYYY-YY' (e.g. '2024-25') or 'YYYY'.",
            )
        source = _season_bundle(request, normalized).values()
    else:
        source = _profiles(request).values()

    items = [_leader_item(p) for p in source]
    roles = sorted({i.role for i in items if i.role})

    if role:
        items = [i for i in items if i.role == role]

    sort_key = sort if sort in LEADER_SORTABLE_KEYS else "apfv"
    descending = order != "asc"
    if sort_key in LEADER_SKILL_KEYS:
        ratings = {
            i.player_id: _skill_rating(i.skill_percentiles, i.apfv, sort_key)
            for i in items
        }

        def sort_value(item):
            return ratings[item.player_id]
    elif sort_key == "height":
        def sort_value(item):
            return height_to_inches(item.height) if item.height else 0.0
    elif sort_key == "weight":
        def sort_value(item):
            return _weight_value(item.weight)
    else:
        def sort_value(item):
            return item.apfv if item.apfv is not None else -1.0

    items.sort(
        key=lambda i: (sort_value(i), i.apfv if i.apfv is not None else -1.0),
        reverse=descending,
    )

    total = len(items)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = min(page, total_pages)
    start = (page - 1) * page_size
    page_items = items[start : start + page_size]

    return PlayerLeadersResponse(
        players=page_items,
        page=page,
        page_size=page_size,
        total=total,
        total_pages=total_pages,
        seasons=_all_player_seasons(request),
        roles=roles,
        sort=sort_key,
        order="desc" if descending else "asc",
    )


def _weight_value(weight) -> float:
    try:
        return float(weight)
    except (TypeError, ValueError):
        return 0.0


def _skill_rating(percentiles: dict, apfv, key: str) -> float:
    vals = [
        max(0.0, min(100.0, float(percentiles.get(k, 0.0) or 0.0)))
        for k in LEADER_SKILL_KEYS
    ]
    mean = sum(vals) / len(vals) if vals else 0.0
    level = apfv * 100.0 if isinstance(apfv, (int, float)) else mean
    value = vals[LEADER_SKILL_KEYS.index(key)]
    return min(100.0, max(0.0, level + LEADER_SHAPE_CONTRAST * (value - mean)))


def _leader_item(profile: dict) -> PlayerLeaderItem:
    metrics = profile.get("playstyle_metrics", {}) or {}
    return PlayerLeaderItem(
        player_id=int(profile.get("player_id")),
        player_name=str(profile.get("player_name", "")),
        height=_clean_str(profile.get("height")),
        weight=profile.get("weight"),
        role=_clean_str(profile.get("role")),
        archetypes=list(profile.get("archetypes", []) or []),
        apfv=profile.get("apfv"),
        skill_percentiles={
            key: _metric_pctile(metrics.get(key)) for key in LEADER_SKILL_KEYS
        },
    )


def _clean_str(value):
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _metric_pctile(metric) -> float:
    if isinstance(metric, dict):
        try:
            return float(metric.get("percentile", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _all_player_seasons(request: Request) -> list[str]:
    df = _season_features(request)
    if df is None or df.empty or "SEASON" not in df.columns:
        return []
    seasons = {
        s for s in df["SEASON"].astype(str) if normalize_season(s) is not None
    }
    return sorted(seasons, reverse=True)


@router.get("/{player_id}", response_model=PlayerProfileResponse)
def get_player_profile(
    player_id: int,
    request: Request,
    season: str | None = Query(
        None,
        description="Optional season (e.g. '2024-25' or '2024'). When provided, "
        "returns the same response shape with metrics scoped to that season only.",
    ),
):
    if season is not None:
        return PlayerProfileResponse(**_season_profile_payload(player_id, season, request))

    pid_str = str(player_id)
    profile = _profiles(request).get(pid_str)
    if profile is None:
        raise HTTPException(
            status_code=404,
            detail=f"Player profile not found. Build {PLAYER_PROFILES_PATH.name} first.",
        )


    has_missing_bio = (
        profile.get("height") is None
        or profile.get("weight") is None
        or profile.get("draft_year") is None
        or profile.get("draft_position") is None
    )
    
    if has_missing_bio:
        updated_profile, role = _fetch_and_populate_missing_bio(player_id, profile)
        if role:
            _update_cached_files(request, pid_str, updated_profile, role)
            profile = updated_profile

    return PlayerProfileResponse(**_profile_response_payload(profile, request))


@router.get("/{player_id}/seasons")
def get_player_seasons(player_id: int, request: Request):
    """Return the seasons (most recent first) this player has stats for.

    Used by the profile page to populate the season-selection dropdown so it
    only offers seasons the player actually played.
    """
    df = _season_features(request)
    if df.empty or "SEASON" not in df.columns or "PLAYER_ID" not in df.columns:
        return {"seasons": []}

    import pandas as pd

    pids = pd.to_numeric(df["PLAYER_ID"], errors="coerce")
    frame = df[pids == player_id]
    seasons = sorted(
        {s for s in frame["SEASON"].astype(str) if normalize_season(s) is not None},
        reverse=True,
    )
    return {"seasons": seasons}


@router.get("/{player_id}/similar", response_model=list[SimilarPlayer])
def get_similar_players(player_id: int, request: Request):
    profile = _profiles(request).get(str(player_id))
    if profile is None:
        raise HTTPException(status_code=404, detail="Player profile not found.")
    return [SimilarPlayer(**p) for p in profile.get("similar_players", [])]


def _profiles(request: Request) -> dict:
    profiles = getattr(request.app.state, "player_profiles", None)
    if profiles is not None:
        return profiles
    if not PLAYER_PROFILES_PATH.exists():
        return {}
    with PLAYER_PROFILES_PATH.open() as f:
        return json.load(f)


def _season_features(request: Request):
    df = getattr(request.app.state, "player_season_features", None)
    if df is not None:
        return df
    import pandas as pd

    if not PLAYER_SEASON_FEATURES_PATH.exists():
        df = pd.DataFrame()
    else:
        try:
            df = pd.read_parquet(PLAYER_SEASON_FEATURES_PATH, engine="fastparquet")
        except Exception:
            df = pd.read_parquet(PLAYER_SEASON_FEATURES_PATH)
    request.app.state.player_season_features = df
    return df


def _season_bundle(request: Request, season: str) -> dict:
    """Return (and cache) the per-player payloads for a single season."""
    cache = getattr(request.app.state, "season_bundles", None)
    if cache is None:
        cache = {}
        request.app.state.season_bundles = cache
    if season in cache:
        return cache[season]

    df = _season_features(request)
    if df.empty or "SEASON" not in df.columns:
        raise HTTPException(
            status_code=404,
            detail=f"Season features unavailable. Build {PLAYER_SEASON_FEATURES_PATH.name} first.",
        )
    frame = df[df["SEASON"] == season]
    if frame.empty:
        seasons = sorted(df["SEASON"].astype(str).unique())
        raise HTTPException(
            status_code=404,
            detail=f"No data for season '{season}'. Available: {seasons[0]} to {seasons[-1]}.",
        )
    bundle = build_season_bundle(frame, _profiles(request))
    cache[season] = bundle
    return bundle


def _season_profile_payload(player_id: int, season: str, request: Request) -> dict:
    normalized = normalize_season(season)
    if normalized is None:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid season '{season}'. Use 'YYYY-YY' (e.g. '2024-25') or 'YYYY'.",
        )
    bundle = _season_bundle(request, normalized)
    payload = bundle.get(str(player_id))
    if payload is None:
        raise HTTPException(
            status_code=404,
            detail=f"Player {player_id} has no data for season '{normalized}'.",
        )
    payload = dict(payload)
    payload["playstyle_metrics"] = _normalize_playstyle_metrics(payload.get("playstyle_metrics", {}))
    return payload


def _profile_response_payload(profile: dict, request: Request = None) -> dict:
    payload = dict(profile)
    payload["career_teams"] = _collapse_career_teams(payload.get("career_teams", []))
    payload["playstyle_metrics"] = _normalize_playstyle_metrics(payload.get("playstyle_metrics", {}))
    if "pfv" not in payload or payload["pfv"] is None:
        payload["pfv"] = calculate_pfv(remove_mpg_adjustment_from_metrics(payload["playstyle_metrics"]))
    payload.pop("npfv", None)
    if "apfv" not in payload or payload["apfv"] is None:
        all_adjusted_pfvs = getattr(request.app.state, "player_all_adjusted_pfvs", []) if request else []
        all_height_buckets = getattr(request.app.state, "player_all_height_buckets", []) if request else []
        if all_adjusted_pfvs and all_height_buckets:
            h_bucket = height_bucket(payload.get("height"))
            group_vals = [val for val, hb in zip(all_adjusted_pfvs, all_height_buckets) if hb == h_bucket]
            player_adjusted_pfv = calculate_adjusted_pfv(remove_mpg_adjustment_from_metrics(payload["playstyle_metrics"]))
            if player_adjusted_pfv not in group_vals:
                group_vals.append(player_adjusted_pfv)
            payload["apfv"] = calculate_apfv(player_adjusted_pfv, group_vals)
        else:
            payload["apfv"] = 0.0
    return payload


def _normalize_playstyle_metrics(metrics: dict) -> dict:
    normalized = {}
    for key in PLAYSTYLE_METRIC_KEYS:
        item = metrics.get(key, {})
        if isinstance(item, dict):
            normalized[key] = {
                "value": float(item.get("value", 0.0) or 0.0),
                "percentile": float(item.get("percentile", 0.0) or 0.0),
            }
        else:
            normalized[key] = {
                "value": float(item or 0.0),
                "percentile": 0.0,
            }
    return normalized


def _collapse_career_teams(career_teams: list) -> list[str]:
    out: list[str] = []
    for item in career_teams:
        if isinstance(item, dict):
            abbreviation = str(item.get("abbreviation", "")).strip()
        else:
            abbreviation = str(item).strip()
        if abbreviation and abbreviation not in out:
            out.append(abbreviation)
    return out


def _rank_player_search(query: str, metadata: dict, limit: int) -> list[tuple[str, float]]:
    needle = _normalize_name(query)
    if not needle:
        return []

    ranked = []
    for pid, item in metadata.items():
        name = str(item.get("player_name", ""))
        haystack = _normalize_name(name)
        if not haystack:
            continue
        score = _player_search_score(needle, haystack)
        if score >= 35:
            ranked.append((str(pid), score))

    ranked.sort(
        key=lambda match: (
            -match[1],
            str(metadata[match[0]].get("player_name", "")).lower(),
            int(match[0]),
        )
    )
    return ranked[:limit]


def _player_search_score(needle: str, haystack: str) -> float:
    if needle == haystack:
        return 120.0

    tokens = haystack.split()
    if needle in tokens:
        return 115.0
    if any(token.startswith(needle) for token in tokens):
        return 110.0
    if haystack.startswith(needle):
        return 105.0
    if needle in haystack:
        return 100.0

    token_prefix = max((fuzz.ratio(needle, token) for token in tokens), default=0.0)
    full_name = fuzz.ratio(needle, haystack)
    token_set = fuzz.token_set_ratio(needle, haystack)
    return max(full_name, token_set, token_prefix * 0.95)


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _fetch_and_populate_missing_bio(player_id: int, profile: dict):
    from nba_api.stats.endpoints import commonplayerinfo
    
    try:

        endpoint = commonplayerinfo.CommonPlayerInfo(
            player_id=player_id,
            timeout=5,
        )
        df = endpoint.common_player_info.get_data_frame()
        if not df.empty:
            row = df.iloc[0].to_dict()
            height = row.get("HEIGHT")
            weight = row.get("WEIGHT")
            draft_year = row.get("DRAFT_YEAR")
            draft_number = row.get("DRAFT_NUMBER")
            raw_pos = row.get("POSITION")
            
            from analytics.player_profiles.features import position_group
            mapped_pos = position_group(raw_pos)
            if mapped_pos == "G":
                fallback_h, fallback_w = "6-3", 190.0
            elif mapped_pos == "B":
                fallback_h, fallback_w = "6-10", 240.0
            else:
                fallback_h, fallback_w = "6-7", 215.0

            def clean_val(v):
                if v is None or str(v).strip() == "" or str(v).lower() in ("nan", "none"):
                    return None
                val_str = str(v).strip()
                if val_str.lower() == "undrafted":
                    return "Undrafted"
                try:
                    return int(float(val_str))
                except ValueError:
                    return val_str

            clean_weight = clean_val(weight)
            if clean_weight is None:
                clean_weight = fallback_w

            clean_height = height if (height and str(height).strip()) else fallback_h
            clean_draft_year = clean_val(draft_year)
            clean_draft_pos = clean_val(draft_number)

            profile["height"] = clean_height
            profile["weight"] = clean_weight
            profile["draft_year"] = clean_draft_year
            profile["draft_position"] = clean_draft_pos

            legacy_positions = {"", "G", "W", "B", "F", "C", "G-F", "F-G", "F-C", "C-F", "GUARD", "FORWARD", "CENTER"}
            current_role = str(profile.get("role", "")).upper()
            if not profile.get("role") or current_role in legacy_positions:
                profile["role"] = mapped_pos
                ret_pos = mapped_pos
            else:
                ret_pos = profile["role"]

            return profile, ret_pos
    except Exception as e:
        print(f"  [WARN] Failed to fetch on-demand player info for {player_id}: {e}")
    return profile, None


def _update_cached_files(request: Request, player_id: str, profile: dict, role=None):
    profiles = _profiles(request)
    profiles[player_id] = profile
    
    metadata = getattr(request.app.state, "player_metadata", None)
    if metadata is None and PLAYER_METADATA_PATH.exists():
        try:
            with PLAYER_METADATA_PATH.open() as f:
                metadata = json.load(f)
                request.app.state.player_metadata = metadata
        except Exception:
            pass
            
    if metadata is not None and player_id in metadata:
        if role:
            metadata[player_id]["role"] = role
            
    try:
        with PLAYER_PROFILES_PATH.open("w") as f:
            json.dump(profiles, f)
        if metadata is not None:
            with PLAYER_METADATA_PATH.open("w") as f:
                json.dump(metadata, f)
    except Exception as e:
        print(f"  [WARN] Failed to save updated player profiles/metadata: {e}")
