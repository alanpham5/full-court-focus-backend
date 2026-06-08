# Full Court Focus Backend

FastAPI-powered basketball analytics service that provides career-level historical player profiles, similar-player matches, team-season profiles, starting lineups, playstyle badges, historical similarity indices, and NBA draft prospect analytics using NBA API data.

---

## Analytical Pipelines & Scripts Architecture

The data ingestion, scraping, and analytics workflow is built around a **modular pipeline architecture** located in `app/pipelines/`. Command-line entry points are located in `app/scripts/`.

### Modular Pipeline Stages (`app/pipelines/`)

Each stage in the analytics process is encapsulated in a dedicated pipeline module:

1. **`TeamStatsPipeline`** (`app/pipelines/team_stats_pipeline.py`): Scrapes historical team statistics (Advanced, Shooting, Misc) from the NBA API, builds the historical teams dataset (`teams_historical.parquet`), and updates `team_metadata.json` and `season_index.json`.
2. **`BadgeLeadersPipeline`** (`app/pipelines/badge_leaders_pipeline.py`): Reads the team stats dataset and metadata, calculates the top teams for each playstyle badge, and writes `badge_leaders.json`.
3. **`SimilarTeamsPipeline`** (`app/pipelines/similar_teams_pipeline.py`): Computes a similarity index between historical team-seasons and writes `similar_teams.json`.
4. **`LineupsPipeline`** (`app/pipelines/lineups_pipeline.py`): Scrapes and indexes team starting lineups, updating `starting_lineups.json`.
5. **`TeamProfilesPipeline`** (`app/pipelines/team_profiles_pipeline.py`): Rebuilds team profiles (stat leaders, similarity display, etc.) and updates `team_profiles.json`. Supports incremental updates for specific team-season keys.
6. **`PlayerProfilesPipeline`** (`app/pipelines/player_profiles_pipeline.py`): Gathers player biographical data, calculates playstyle metrics/percentiles, generates similarity coordinates, builds `player_profiles.json`, and copies assets to the static directory.
7. **`ProspectsPipeline`** (`app/pipelines/prospects_pipeline.py`): Scrapes NBA draft prospects from RealGM, standardizes metrics to per-36 features, computes playstyle roles, finds similar NBA counterparts, and writes the prospects datasets.

---

### Central Pipeline Orchestrator (`app/scripts/run_pipeline.py`)

A master runner script coordinates the execution of all stages in their correct dependency order. 

- **Run the full historical pipeline (all stages, full scrape)**:
  ```bash
  python app/scripts/run_pipeline.py
  ```
- **Incremental run for the current season only (updates team/lineup/player data for active season)**:
  ```bash
  python app/scripts/run_pipeline.py --current
  ```
- **Run specific stages only (e.g., rebuild similar teams and team profiles)**:
  ```bash
  python app/scripts/run_pipeline.py --stages similar,profile
  ```

Supported arguments:
- `--current`: Only refresh the current season.
- `--stages`: Comma-separated list of stages to run (`team`, `badge`, `similar`, `lineup`, `profile`, `player`, `prospect` or `all`).
- `--storage-uri`: Storage root path for player profiles (defaults to local `app/data/player_profiles`).
- `--gcs-project`: Google Cloud project for gs:// player profiles storage.
- `--refresh-raw`: Force re-download of raw NBA API player statistics.
- `--rate-limit`: Seconds to sleep between team stats/lineup NBA API requests (default: 1.5).

---

### Standalone Wrapper Scripts (`app/scripts/`)

Each stage can still be run independently using the existing command-line scripts. These scripts serve as thin CLI wrappers that parse arguments and delegate to their respective pipeline modules in `app/pipelines/`.

#### 1. `scrape_all_seasons.py`
- **What it does**: Scrapes team statistics and orchestrates all downstream updates (badge leaders, similarity indices, lineups, team profiles, and player profiles).
- **How to run**:
  - Full build: `python app/scripts/scrape_all_seasons.py`
  - Current season only: `python app/scripts/scrape_all_seasons.py --current`

#### 2. `build_player_profiles.py`
- **What it does**: Builds player career features, similarity index, embeddings, and profiles.
- **How to run**:
  ```bash
  python app/scripts/build_player_profiles.py --storage-uri app/data/player_profiles --copy-to-static
  ```

#### 3. `build_team_profiles.py`
- **What it does**: Recompiles the team-season stat profiles and historical similarity indices from `teams_historical.parquet`.
- **How to run**:
  ```bash
  python app/scripts/build_team_profiles.py
  ```

#### 4. `check_and_scrape.py`
- **What it does**: Checks if local cached files are stale compared to the scheduled season calendar, and executes an incremental scrape only when new games have been played.
- **How to run**:
  ```bash
  python app/scripts/check_and_scrape.py
  ```

#### 5. `scrape_lineups.py`
- **What it does**: Refreshes or scrapes starting lineups for selected team-seasons.
- **How to run**:
  ```bash
  python app/scripts/scrape_lineups.py --current
  ```

#### 6. `build_prospects_dataset.py`
- **What it does**: Scrapes RealGM draft prospect averages, converts stats to per-36 features, compares them against NBA counterparts, and writes `prospects.json`.
- **How to run**:
  ```bash
  python app/scripts/build_prospects_dataset.py
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

- **Description**: Returns detailed biographical stats, player playstyle role, career playstyle metrics, each metric's percentile against all careers from 1996 through 2025, PFV/APFV scores, and similar-player lists.
- **PFV**: Polygonal Feature Value, the normalized area of the six-axis radar chart using raw population percentiles for `pts_per36`, `reb_per36`, `ast_per36`, `blk_per36`, `stl_per36`, and `ts_pct`.
- **APFV**: Adjusted PFV, a population-ranked version of PFV after applying the MPG adjustment once. Higher APFV means the player's radar shape holds up with stronger playing-time context.
- **MPG adjustment**: For `pts_per36`, `ast_per36`, `reb_per36`, `stl_per36`, `blk_per36`, `ts_pct`, `efg_pct`, `fg3a_rate`, and `fta_rate`, only the percentile is multiplied by `(mpg_percentile / 100) ^ 1.5`. The metric `value` remains the raw career value, and `mpg.percentile` remains unadjusted.
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
      "pts_per36": { "value": 26.3044, "percentile": 94.8 },
      "reb_per36": { "value": 4.9311, "percentile": 38.6 },
      "ast_per36": { "value": 6.6862, "percentile": 91.1 },
      "blk_per36": { "value": 0.2796, "percentile": 30.2 },
      "stl_per36": { "value": 1.5885, "percentile": 80.5 },
      "tov_per36": { "value": 3.2801, "percentile": 92.3 },
      "ts_pct": { "value": 0.6225, "percentile": 90.9 },
      "efg_pct": { "value": 0.5793, "percentile": 88.5 },
      "ast_pct": { "value": 0.7638, "percentile": 72.7 },
      "fg3a_rate": { "value": 0.5137, "percentile": 80.9 },
      "fta_rate": { "value": 0.2396, "percentile": 45.7 },
      "mpg": { "value": 34.0881, "percentile": 97.1 }
    },
    "pfv": 0.4715,
    "apfv": 0.9627,
    "similar_players": [
      {
        "player_id": 202681,
        "player_name": "Kyrie Irving",
        "similarity_score": 92.6,
        "career_span": "2011-12 to 2024-25",
        "explanation": "Closest match in rebounding, scoring volume, and defensive box score."
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

### Lineup Synergy Endpoints

#### `GET /lineup/search`

- **Description**: Searches for players active in a specific season by name or abbreviation prefix.
- **Query Parameters**:
  - `season` (string, required): The NBA season, e.g. `2023-24`.
  - `q` (string, required): The search query.
  - `limit` (integer, default=8): Max suggestions to return.
- **Sample Request**: `GET /lineup/search?season=2023-24&q=LeBron`
- **Sample Response**:
  ```json
  [
    {
      "player_id": 2544,
      "name": "LeBron James",
      "team_abbreviation": "LAL",
      "role": "Playmaker",
      "pts_per36": 26.2,
      "mpg": 35.3
    }
  ]
  ```

#### `POST /lineup/synergy`

- **Description**: Measures the strengths, weaknesses, projected style vector percentiles, and synergy score for a 5-player lineup within a specific season. Also returns matching similar historical team lineups.
- **Request Body**:
  - `season` (string): The NBA season, e.g. `2023-24`.
  - `player_ids` (array of 5 integers): Exactly 5 player IDs.
- **Sample Request**:
  `POST /lineup/synergy`
  ```json
  {
    "season": "2023-24",
    "player_ids": [2544, 201939, 201142, 203507, 203999]
  }
  ```
- **Sample Response**:
  ```json
  {
    "season": "2023-24",
    "players": [
      {
        "player_id": 2544,
        "name": "LeBron James",
        "role": "Playmaker",
        "archetypes": ["high-volume creator"],
        "pts_per36": 26.2,
        "ast_per36": 8.5,
        "reb_per36": 7.5,
        "stl_per36": 1.3,
        "blk_per36": 0.5,
        "fg3a_rate": 0.287
      },
      {
        "player_id": 201939,
        "name": "Stephen Curry",
        "role": "Designated Scorer",
        "archetypes": ["perimeter scorer"],
        "pts_per36": 28.1,
        "ast_per36": 5.4,
        "reb_per36": 4.8,
        "stl_per36": 0.8,
        "blk_per36": 0.4,
        "fg3a_rate": 0.582
      },
      {
        "player_id": 201142,
        "name": "Kevin Durant",
        "role": "Designated Scorer",
        "archetypes": ["balanced scorer"],
        "pts_per36": 27.5,
        "ast_per36": 5.2,
        "reb_per36": 6.8,
        "stl_per36": 0.9,
        "blk_per36": 1.2,
        "fg3a_rate": 0.285
      },
      {
        "player_id": 203507,
        "name": "Giannis Antetokounmpo",
        "role": "Rim Attacker",
        "archetypes": ["free-throw pressure scorer"],
        "pts_per36": 31.0,
        "ast_per36": 6.6,
        "reb_per36": 11.7,
        "stl_per36": 1.2,
        "blk_per36": 1.1,
        "fg3a_rate": 0.089
      },
      {
        "player_id": 203999,
        "name": "Nikola Jokic",
        "role": "Playmaker",
        "archetypes": ["high-volume creator"],
        "pts_per36": 27.5,
        "ast_per36": 9.4,
        "reb_per36": 12.9,
        "stl_per36": 1.4,
        "blk_per36": 0.9,
        "fg3a_rate": 0.161
      }
    ],
    "style_vector": {
      "pace": 60.0,
      "three_point_volume": 83.3,
      "paint": 90.0,
      "defense": 40.0,
      "playmaking": 96.7,
      "rebounding": 100.0
    },
    "synergy_score": 93.4,
    "synergy_breakdown": {
      "playmaking": 10.0,
      "spacing": 12.0,
      "interior": 5.0,
      "defense": -10.0,
      "overlap": -6.0
    },
    "strengths": [
      "Elite Floor Spacing - With multiple perimeter specialist threats, this lineup will pull defenders out and open up driving lanes.",
      "High-Volume Scoring Engine - Elite scoring capacity across multiple positions makes this lineup extremely difficult to contain.",
      "Elite Playmaking & Synergy - Above-average playmaking percentiles indicate exceptional ball movement and high-quality shot creation.",
      "Dominant Rebounding - Above-average rebounding metrics ensure the lineup can limit opponents to single-shot possessions and control the glass.",
      "High-Volume Paint Scoring - Active paint scorers put relentless pressure on the rim and excel at finishing inside."
    ],
    "weaknesses": [
      "Vulnerable Defensive Shell - Below-average defensive rating indicates this lineup lacks the collective stops to halt elite offenses."
    ],
    "similar_teams": [
      {
        "team_id": 1610612766,
        "team_name": "Charlotte Hornets",
        "abbreviation": "CHA",
        "season": "2025-26",
        "similarity_pct": 89.3,
        "record": "44-38",
        "style_vector": {
          "pace": 13.3,
          "three_point_volume": 100.0,
          "paint": 50.0,
          "defense": 65.0,
          "playmaking": 55.0,
          "rebounding": 96.7
        },
        "starters": [
          {
            "player_id": 1628970,
            "name": "Miles Bridges",
            "roles": ["Designated Scorer", "Rim Attacker"]
          },
          {
            "player_id": 1629630,
            "name": "LaMelo Ball",
            "roles": ["Playmaker"]
          },
          {
            "player_id": 1630625,
            "name": "Moussa Diabaté",
            "roles": ["Interior Presence"]
          },
          {
            "player_id": 1641706,
            "name": "Brandon Miller",
            "roles": ["Designated Scorer"]
          },
          {
            "player_id": 1642258,
            "name": "Kon Knueppel",
            "roles": ["Perimeter Specialist"]
          }
        ]
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

### Draft Endpoints

Requires `app/data/static/prospects.json` to be generated first via `build_prospects_dataset.py`.

#### `GET /draft/prospects`

- **Description**: Lists every prospect in the current draft class with their `prospect_id`, display name, college/team, physical metadata (`height`, `weight`), playstyle role (`role`), and flat raw counting stats.
- **Sample Response**:
  ```json
  [
    {
      "prospect_id": "a-j-dybantsa",
      "player_name": "A.J. Dybantsa",
      "team": "BYU",
      "height": "6-9",
      "weight": "210",
      "role": "Designated Scorer",
      "raw_stats": {
        "gp": 35,
        "mpg": 34.8,
        "ppg": 25.5,
        "fgm": 8.8,
        "fga": 17.3,
        "fg_pct": 0.51,
        "fg3m": 1.4,
        "fg3a": 4.2,
        "fg3_pct": 0.331,
        "ftm": 6.5,
        "fta": 8.5,
        "ft_pct": 0.774,
        "rpg": 6.8,
        "apg": 3.7,
        "spg": 1.1,
        "bpg": 0.3,
        "pfv": 0.276,
        "apfv": 0.9515
      }
    }
  ]
  ```

#### `GET /draft/{prospect_id}`

- **Description**: Returns the complete dataset for a single prospect, including physical metadata and playstyle role. Non-raw stats include both the raw value and the prospect's MPG-adjusted percentile rank within the current draft class (0–100). RealGM profile URLs are not included in API responses.
- **Path Parameters**: `prospect_id` (string) — slugified name, e.g. `a-j-dybantsa`
- **Sample Response**:
  ```json
  {
    "prospect_id": "a-j-dybantsa",
    "player_name": "A.J. Dybantsa",
    "team": "BYU",
    "height": "6-9",
    "weight": "210",
    "role": "Designated Scorer",
    "gp": 35,
    "mpg":      { "value": 34.8,    "percentile": 92.2  },
    "pts_per36": { "value": 26.3793, "percentile": 88.5  },
    "reb_per36": { "value": 7.0345,  "percentile": 49.3  },
    "ast_per36": { "value": 3.8276,  "percentile": 66.2  },
    "blk_per36": { "value": 0.3103,  "percentile": 23.9  },
    "stl_per36": { "value": 1.1379,  "percentile": 36.9  },
    "ts_pct":    { "value": 0.606,   "percentile": 51.6  },
    "efg_pct":   { "value": 0.5491,  "percentile": 40.8  },
    "fg3a_rate": { "value": 0.2428,  "percentile": 27.7  },
    "fta_rate":  { "value": 0.4913,  "percentile": 67.7  },
    "raw_stats": {
      "gp": 35, "mpg": 34.8, "ppg": 25.5,
      "fgm": 8.8, "fga": 17.3, "fg_pct": 0.51,
      "fg3m": 1.4, "fg3a": 4.2, "fg3_pct": 0.331,
      "ftm": 6.5, "fta": 8.5, "ft_pct": 0.774,
      "rpg": 6.8, "apg": 3.7, "spg": 1.1, "bpg": 0.3,
      "pfv": 0.276, "apfv": 0.9515
    },
    "similar_nba_players": [
      {
        "player_id": 2546,
        "player_name": "Carmelo Anthony",
        "similarity_score": 91.9,
        "career_span": "2003-04 to 2021-22",
        "position_group": "W",
        "role": "Designated Scorer"
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
| **rim protection**             | Rim-protecting big with elite block rate and strong rebounding presence.                                                        |
| **3-and-D profile**            | Perimeter defender who also contributes from three — elevated steal rate and 3-point volume.                                    |
| **rebounding defender**        | Dominant rebounder with strong defensive box-score contributions.                                                               |
| **event creator**              | Disruptive defender who generates steals or blocks at an elite rate.                                                            |
| **balanced scorer**            | Fallback archetype for players with solid but unremarkable offensive contributions across the board.                            |
| **box-score defender**         | Fallback archetype for players with positive but unspectacular defensive box-score numbers.                                     |
