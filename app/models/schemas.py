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
    """League percentiles for that season (0–100; higher = more of that style)."""

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
