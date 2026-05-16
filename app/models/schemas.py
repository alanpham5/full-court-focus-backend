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
    top: BadgeExemplarTeam | None = None
    runner_up: BadgeExemplarTeam | None = None


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
    lineup_mpg: float | None = None
    starters: list[LineupPlayer]
