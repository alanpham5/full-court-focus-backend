# Full Court Focus Backend

FastAPI-powered basketball analytics service that provides career-level historical player profiles, similar-player matches, team-season profiles, starting lineups, playstyle badges, and historical similarity indices using NBA API data.

---

## Scripts Directory

The analytical pipelines and data scraping workflows are organized as Python scripts in `app/scripts/`.

### 1. `scrape_all_seasons.py`

- **What it does**: Scrapes historical team statistics, normalizes statistics by season, generates similarity indices, indexes starting lineups, builds team playstyle badges, and runs the career-level player profiles pipeline.
- **How to run**:
  - **Full historical build (1996-Present)**:
    ```bash
    python app/scripts/scrape_all_seasons.py
    ```
  - **Current season only (incremental update)**:
    ```bash
    python app/scripts/scrape_all_seasons.py --current
    ```

### 2. `build_player_profiles.py`

- **What it does**: Gathers biographical details (height, weight, draft numbers) for players, standardizes metrics, computes career-level playstyle metric percentiles against all careers from 1996 through the configured end season, computes PCA-based similarity coordinates, and compiles career profiles.
- **How to run**:
  ```bash
  python app/scripts/build_player_profiles.py \
    --start-season 1996 \
    --end-season 2025 \
    --storage-uri app/data/player_profiles \
    --copy-to-static \
    --refresh-raw
  ```

### 3. `build_team_profiles.py`

- **What it does**: Recompiles the team-season stat profiles and historical similarity indices from the offline cache (`teams_historical.parquet`).
- **How to run**:
  ```bash
  python app/scripts/build_team_profiles.py
  ```

### 4. `check_and_scrape.py`

- **What it does**: Checks if local cached files are stale compared to the scheduled season calendar, and executes an incremental scrape only when new games have been played.
- **How to run**:
  ```bash
  python app/scripts/check_and_scrape.py
  ```
  _(To force a scrape execution regardless of status, append `--force`)_

### 5. `scrape_lineups.py`

- **What it does**: Refreshes or scrapes starting lineups and playmaker/scorer roles for selected teams and seasons.
- **How to run**:

  ```bash
  # Scrape all teams for the current season
  python app/scripts/scrape_lineups.py --current

  # Scrape a specific team-season
  python app/scripts/scrape_lineups.py --team-id 1610612743 --season 2023-24
  ```

### 6. `backfill_paint_parquet.py`

- **What it does**: Downloads paints touches (`PTS_PAINT`) and registers `PAINT_FGA` aliases to historical parquet data files for older seasons.
- **How to run**:
  ```bash
  python app/scripts/backfill_paint_parquet.py
  ```

---

## API Endpoints Reference

Start the local server from the `app/` folder using:

```bash
uvicorn main:app --reload
```

### 1. `GET /health`

- **Description**: Simple health status endpoint.
- **Sample Response**:
  ```json
  {
    "status": "ok"
  }
  ```

---

### Player Endpoints

#### `GET /players/search`

- **Description**: Fuzz searches for players by name or abbreviation prefix.
- **Query Parameters**: `q` (string, required), `limit` (integer, default=8)
- **Sample Response**:
  ```json
  [
    {
      "player_id": 201939,
      "player_name": "Stephen Curry",
      "role": "Designated Scorer",
      "career_span": "2009-10 to 2025-26",
      "archetypes": ["high-volume creator", "perimeter scorer"]
    }
  ]
  ```

#### `GET /players/archetypes/{archetype}`

- **Description**: Returns all players belonging to a specific playstyle archetype (e.g. `perimeter-scorer`).
- **Query Parameters**: `limit` (integer, default=25)
- **Sample Response**:
  ```json
  [
    {
      "player_id": 1629012,
      "player_name": "Collin Sexton",
      "role": "Secondary Creator",
      "career_span": "2018-19 to 2025-26",
      "archetypes": ["perimeter scorer", "box-score defender"]
    }
  ]
  ```

#### `GET /players/{player_id}`

- **Description**: Returns detailed biographical stats, player playstyle role, career playstyle metrics, each metric's percentile against all careers from 1996 through 2025, and similar-player lists.
- **Sample Response**:
  ```json
  {
    "player_id": 201939,
    "player_name": "Stephen Curry",
    "height": "6-2",
    "weight": 185.0,
    "draft_year": "2009",
    "draft_position": "7",
    "role": "Designated Scorer",
    "career_teams": ["GSW"],
    "career_span": "2009-10 to 2025-26",
    "career_games": 1069,
    "archetypes": ["high-volume creator", "perimeter scorer"],
    "playstyle_metrics": {
      "pts_per36": { "value": 26.3044, "percentile": 99.1 },
      "ast_per36": { "value": 6.6862, "percentile": 95.2 },
      "reb_per36": { "value": 4.9311, "percentile": 40.3 },
      "stl_per36": { "value": 1.5885, "percentile": 84.1 },
      "blk_per36": { "value": 0.2796, "percentile": 31.6 },
      "tov_per36": { "value": 3.2801, "percentile": 92.3 },
      "ts_pct": { "value": 0.6225, "percentile": 95.1 },
      "efg_pct": { "value": 0.5793, "percentile": 92.5 },
      "ast_pct": { "value": 0.7638, "percentile": 72.7 },
      "fg3a_rate": { "value": 0.5137, "percentile": 84.6 },
      "fta_rate": { "value": 0.2396, "percentile": 47.7 },
      "mpg": { "value": 34.0881, "percentile": 97.1 }
    },
    "similar_players": [
      {
        "player_id": 203081,
        "player_name": "Damian Lillard",
        "similarity_score": 91.7,
        "career_span": "2012-13 to 2025-26",
        "explanation": "Closest match in rebounding, defensive box score, and scoring efficiency."
      }
    ]
  }
  ```

#### `GET /players/{player_id}/similar`

- **Description**: Returns list of similar players computed from playstyle metrics.
- **Sample Response**:
  ```json
  [
    {
      "player_id": 203081,
      "player_name": "Damian Lillard",
      "similarity_score": 91.7,
      "career_span": "2012-13 to 2025-26",
      "explanation": "Closest match in rebounding, defensive box score, and scoring efficiency."
    }
  ]
  ```

---

### Team Endpoints

#### `GET /team/{team_id}/{season}`

- **Description**: Retrieves a team's statistical profile, playstyle vector, badges, leaders, and similar historical teams.
- **Sample Response**:
  ```json
  {
    "team_id": 1610612743,
    "team_name": "Denver Nuggets",
    "season": "2023-24",
    "record": "57-25",
    "win_pct": 0.695,
    "style_vector": {
      "pace": 0.45,
      "spacing": 0.62,
      "ball_movement": 0.88,
      "isolation": 0.12,
      "rim_pressure": 0.74,
      "transition": 0.35,
      "defense": 0.81
    },
    "badges": [
      {
        "id": "playmaker",
        "name": "Elite Playmaking",
        "description": "Top tier passing and ball movement."
      }
    ],
    "leaders": {
      "pts": { "player_id": 203999, "name": "Nikola Jokic", "value": 26.4 },
      "ast": { "player_id": 203999, "name": "Nikola Jokic", "value": 9.0 },
      "reb": { "player_id": 203999, "name": "Nikola Jokic", "value": 12.4 },
      "stl": { "player_id": 203999, "name": "Nikola Jokic", "value": 1.4 },
      "blk": { "player_id": 203999, "name": "Nikola Jokic", "value": 0.9 }
    },
    "similar_teams": [
      {
        "team_id": 1610612744,
        "season": "2015-16",
        "abbreviation": "GSW",
        "similarity": 95.8,
        "record": "73-9"
      }
    ]
  }
  ```

#### `GET /team/{team_id}/{season}/lineup`

- **Description**: Returns starting lineups and player tactical roles.
- **Sample Response**:
  ```json
  {
    "team_id": 1610612743,
    "season": "2023-24",
    "source": "Manual curation",
    "lineup_mpg": 28.5,
    "starters": [
      {
        "player_id": 203999,
        "name": "Nikola Jokic",
        "roles": ["Playmaker", "Designated Scorer"],
        "gp": 79,
        "mpg": 34.6
      }
    ]
  }
  ```

#### `GET /team/{team_id}/{season}/similar-by-era/{era}`

- **Description**: Finds similar historical team profiles filtered by a specific era decade (`1990s`, `2000s`, `2010s`, `2020s`).
- **Sample Response**:
  ```json
  {
    "team_id": 1610612743,
    "season": "2023-24",
    "era": "2010-19",
    "similar_teams": [
      {
        "team_id": 1610612744,
        "season": "2015-16",
        "abbreviation": "GSW",
        "similarity": 95.8,
        "record": "73-9"
      }
    ]
  }
  ```

---

### Search & Badge Endpoints

#### `GET /search/teams`

- **Description**: Returns matching team recommendations matching name/abbreviation.
- **Query Parameters**: `q` (string, required)
- **Sample Response**:
  ```json
  [
    {
      "team_id": 1610612743,
      "team_name": "Denver Nuggets",
      "abbreviation": "DEN"
    }
  ]
  ```

#### `GET /search/seasons/{team_id}`

- **Description**: Lists all historical seasons available for a specific team.
- **Sample Response**:
  ```json
  {
    "seasons": ["2023-24", "2024-25"]
  }
  ```

#### `GET /badges/{season}/leaders`

- **Description**: Retrieves the top team and runner-up for all playstyle badges in a given season.
- **Sample Response**:
  ```json
  {
    "season": "2023-24",
    "badges": [
      {
        "badge": {
          "id": "playmaker",
          "name": "Elite Playmaking",
          "description": "Top tier passing and ball movement."
        },
        "top": {
          "team_id": 1610612743,
          "team_name": "Denver Nuggets",
          "abbreviation": "DEN",
          "value": 0.88,
          "record": "57-25"
        },
        "runner_up": {
          "team_id": 1610612744,
          "team_name": "Golden State Warriors",
          "abbreviation": "GSW",
          "value": 0.85,
          "record": "46-36"
        }
      }
    ]
  }
  ```

---

## Player Role & Archetype Reference

### Player Roles

Each player is assigned exactly **one** role representing their primary function on the court.

| Role                     | Description                                                                                                                    |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| **Playmaker**            | Primary offensive engine who organizes possessions through passing, advantage creation, and ball distribution.                 |
| **Secondary Creator**    | Off-ball or second-side handler who can pass into actions, punish rotations, and create offense without being the main engine. |
| **Designated Scorer**    | Player whose value is self-generated scoring: pull-ups, isolations, tough shot conversion, and late-clock offense.             |
| **Perimeter Specialist** | Spacing-focused scorer who primarily provides catch-and-shoot gravity and punishes help defense from range.                    |
| **Rim Attacker**         | Downhill scorer who pressures the paint, draws fouls, and scores around the basket.                                            |
| **Interior Presence**    | Paint-dominant big who scores at the rim, crashes the glass, and anchors the defense near the basket.                          |
| **Defensive Specialist** | Player whose primary value is on the defensive end — disrupting plays, protecting the rim, or generating steals.               |

### Player Archetypes

Each player can have up to **3 archetypes** drawn from the following list. Archetypes reflect statistical tendencies rather than a single role.

| Archetype                      | Description                                                                                                                     |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| **high-volume creator**        | High scorer who also facilitates — elevated points and assists per 36.                                                          |
| **perimeter scorer**           | Heavy 3-point volume shooter with above-average efficiency from range.                                                          |
| **free-throw pressure scorer** | Gets to the line at a high rate alongside strong scoring volume.                                                                |
| **table setter**               | High-assist, low-turnover distributor who keeps the offense organized.                                                          |
| **efficient finisher**         | Above-average eFG% scorer who does most damage near the basket or on high-percentage looks, without being a high-volume scorer. |
| **rim protection profile**     | Rim-protecting big with elite block rate and strong rebounding presence.                                                        |
| **3-and-D profile**            | Perimeter defender who also contributes from three — elevated steal rate and 3-point volume.                                    |
| **rebounding defender**        | Dominant rebounder with strong defensive box-score contributions.                                                               |
| **event creator**              | Disruptive defender who generates steals or blocks at an elite rate.                                                            |
| **balanced scorer**            | Fallback archetype for players with solid but unremarkable offensive contributions across the board.                            |
| **box-score defender**         | Fallback archetype for players with positive but unspectacular defensive box-score numbers.                                     |
