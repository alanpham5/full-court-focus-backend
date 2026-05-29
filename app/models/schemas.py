from __future__ import annotations

from typing import Optional, Union

from pydantic import BaseModel


class Badge(BaseModel):
    id: str
    label: str
    emoji: str
    description: str


class StatLeader(BaseModel):
    player_id: int
    name: str
    value: float


class StatLeaders(BaseModel):
    ppg: StatLeader
    apg: StatLeader
    rpg: StatLeader
    spg: StatLeader
    bpg: StatLeader
    fg3_pct: StatLeader


class SimilarTeam(BaseModel):
    team_id: int
    team_name: str
    abbreviation: str
    season: str
    similarity_pct: float
    record: str


class StyleVector(BaseModel):
    pace: float
    three_point_volume: float
    paint: float
    defense: float
    playmaking: float
    rebounding: float


class TeamProfileResponse(BaseModel):
    team_id: int
    team_name: str
    season: str
    record: str
    win_pct: float
    style_vector: StyleVector
    badges: list[Badge]
    leaders: StatLeaders
    similar_teams: list[SimilarTeam]


class SearchSuggestion(BaseModel):
    team_id: int
    team_name: str
    abbreviation: str


class BadgeExemplarTeam(BaseModel):
    team_id: int
    team_name: str
    abbreviation: str
    record: str
    exemplar_score: float


class BadgeLeaderEntry(BaseModel):
    badge: Badge
    top: Optional[BadgeExemplarTeam] = None
    runner_up: Optional[BadgeExemplarTeam] = None


class SeasonBadgeLeadersResponse(BaseModel):
    season: str
    badges: list[BadgeLeaderEntry]


class EraSimilarTeamsResponse(BaseModel):
    team_id: int
    season: str
    era: str
    similar_teams: list[SimilarTeam]


class LineupPlayer(BaseModel):
    player_id: int
    name: str
    roles: list[str]
    gp: int
    mpg: float


class StartingLineupResponse(BaseModel):
    team_id: int
    season: str
    source: str
    lineup_mpg: Optional[float] = None
    starters: list[LineupPlayer]


class CareerTeam(BaseModel):
    team_id: int
    abbreviation: str
    team_name: str
    season: str


class PlayerMetricValue(BaseModel):
    value: float
    percentile: float


class PlayerPlaystyleMetrics(BaseModel):
    pts_per36: PlayerMetricValue
    ast_per36: PlayerMetricValue
    reb_per36: PlayerMetricValue
    stl_per36: PlayerMetricValue
    blk_per36: PlayerMetricValue
    tov_per36: PlayerMetricValue
    ts_pct: PlayerMetricValue
    efg_pct: PlayerMetricValue
    ast_pct: PlayerMetricValue
    fg3a_rate: PlayerMetricValue
    fta_rate: PlayerMetricValue
    mpg: PlayerMetricValue


class SimilarPlayer(BaseModel):
    player_id: int
    player_name: str
    similarity_score: float
    career_span: str
    explanation: str


class PlayerProfileResponse(BaseModel):
    player_id: int
    player_name: str
    height: Optional[str] = None
    weight: Optional[Union[str, int]] = None
    draft_year: Optional[Union[str, int]] = None
    draft_position: Optional[Union[str, int]] = None
    role: Optional[str] = None
    career_teams: list[str]
    career_span: str
    career_games: int
    archetypes: list[str]
    playstyle_metrics: PlayerPlaystyleMetrics
    pfv: Optional[float] = None
    npfv: Optional[float] = None
    similar_players: list[SimilarPlayer]


class PlayerSearchSuggestion(BaseModel):
    player_id: int
    player_name: str
    role: str
    career_span: str
    archetypes: list[str]
