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

### Standalone Utility Scripts (`app/scripts/`)

For specific tasks, you can run the following standalone utility scripts:

#### 1. `check_and_scrape.py`
- **What it does**: Checks if local cached files are stale compared to the scheduled season calendar, and executes an incremental scrape (updating `team`, `badge`, `similar`, `lineup`, `profile`, and `player` stages for the current season) only when new games have been played.
- **How to run**:
  ```bash
  python app/scripts/check_and_scrape.py
  ```
  Use `--force` to trigger the incremental scrape regardless of cached state.

#### 2. `build_historical_prospects.py`
- **What it does**: Scrapes historical RealGM NBA draft classes since 2007, standardizes statistics to per-36 features, compares prospects against NBA counterparts, and writes separate JSON and Parquet datasets for each draft class year into the `app/data/static/draft/` directory. If `PLAYER_PROFILES_STORAGE_URI` is configured as a GCS URL (starts with `gs://`), it automatically uploads both JSON and Parquet outputs to GCS.
- **How to run**:
  - Run all years since 2007:
    ```bash
    python app/scripts/build_historical_prospects.py --all
    ```
  - Run a specific year:
    ```bash
    python app/scripts/build_historical_prospects.py --year 2023
    ```
  - Force overwrite cache:
    ```bash
    python app/scripts/build_historical_prospects.py --year 2023 --force
    ```

#### 3. `prospect_tuning_cli.py`
- **What it does**: Manages **versioned tunings** of the prospect → NBA comp model. A version stores not only coefficients (feature weights, bandwidth, smoothing, display-selection knobs including the one-sided scoring-volume affinity `τ`) but the **algorithmic foundation** itself — `feature_norm` (`percentile`/`zscore`/`hybrid`) and `kernel` (`laplacian`/`gaussian`). Versions live as named JSON under `app/data/tuning/versions/`, with `app/data/tuning/active.json` naming the live one. `ProspectsPipeline` overlays the active version on the code defaults at import, so **switching the entire foundation — not just weights — is a pointer change, not a code edit**. See `ALGORITHMS.md` §3.2, §3.5, and §3.8 for the math.
- **Shipped versions**: `v1_baseline` → `v2_scoring_affinity` → `v3_tau067` (all percentile/laplacian) → `v4_hybrid` (**active**; `feature_norm=hybrid`, `kernel=gaussian` — concatenates percentile and z-score feature blocks into one Gaussian metric, strictly beating the percentile engine on counterpart recall, points, and median rank). Revert to the percentile engine with `activate v3_tau067 && regenerate`.
- **How to run**:
  - List versions (the active one is marked `*`):
    ```bash
    python app/scripts/prospect_tuning_cli.py list
    ```
  - Inspect or compare parameter sets:
    ```bash
    python app/scripts/prospect_tuning_cli.py show v2_scoring_affinity
    python app/scripts/prospect_tuning_cli.py diff v1_baseline v2_scoring_affinity
    ```
  - **Revert** to the previous tuning and rewrite the board comps:
    ```bash
    python app/scripts/prospect_tuning_cli.py activate v1_baseline
    python app/scripts/prospect_tuning_cli.py regenerate
    ```
  - **Run a specific version for one run only** (without moving the active pointer) to compare outputs:
    ```bash
    python app/scripts/prospect_tuning_cli.py regenerate --version v1_baseline
    ```
  - **Create a new tuning** from the active one with overridden parameters (e.g. a tighter affinity), then activate it:
    ```bash
    python app/scripts/prospect_tuning_cli.py snapshot v3_tau15 \
      --set scoring_affinity_tau=0.15 --description "tighter one-sided affinity" --activate
    python app/scripts/prospect_tuning_cli.py regenerate
    ```
- **What `regenerate` covers**: by default it recomputes **both** the current 2026 board (`prospects.json`/`.parquet`) **and every stored historical draft class** (`app/data/static/draft/`), then renormalizes prospect APFV across the combined population and **uploads the results to GCS** (if `PLAYER_PROFILES_STORAGE_URI` is a `gs://` URI). It recomputes from the already-stored prospect data — **no scraping** — so it is the single command to apply a tuning change everywhere. Flags:
  - `--scope current|historical|all` (default `all`) — limit to one base.
  - `--years 2020,2021` — restrict the historical classes recomputed.
  - `--force-scrape` — re-fetch historical raw data instead of using the local cache.
  - `--no-upload` — skip the GCS upload; `--upload` — force it and warn if no `gs://` URI is configured.

  The GCS upload runs **after** recompute + APFV normalization so the pushed files reflect the final on-disk state (prospect files live under the `draft/` prefix). The historical loop and the uploader are shared with `build_historical_prospects.py` (which still handles initial scraping and `--recompute`); `regenerate` is the tuning-driven, cache-only entry point.

#### 4. `build_season_lineup_baselines.py`
- **What it does**: Builds the **per-season starting-lineup baselines** that calibrate the lineup synergy engine. For every season it summarises each team's starting lineup into collective metrics (`pace`, `fg3a`, `paint_fga`, `def_score`, `ast`, `ast_pct`, `reb`, `blk`) and stores the season-wide distribution of each metric (`mean`, `std`, `min`, `max`, sorted `values`). A lineup's **style vector, synergy breakdown, and strengths/weaknesses** are graded by where its metrics fall (percentile / z-score) within its season's distribution, so both the synergy page and the Lineup IQ game share one calibration. Output is written to `app/data/static/season_lineup_baselines.json` locally and, when `PLAYER_PROFILES_STORAGE_URI` is a `gs://` URI, mirrored to GCS at `static/season_lineup_baselines.json`. The same metric function is used to build the baselines and to grade a live lineup, so they can never drift.
- **How to run**:
  - Rebuild every season:
    ```bash
    python app/scripts/build_season_lineup_baselines.py --all
    ```
  - Update only the most recent season (merges into the existing artifact):
    ```bash
    python app/scripts/build_season_lineup_baselines.py --latest
    ```
  - Update a single season:
    ```bash
    python app/scripts/build_season_lineup_baselines.py --season 2025-26
    ```
- **When to run**: after a fresh data scrape adds games to the current season (pair with `--latest`), or after retuning the synergy engine (`--all`).

#### 5. `build_lineup_synergy_scores.py`
- **What it does**: Precomputes the synergy score of every stored starting lineup into `app/data/static/lineup_synergy_scores.json`. The Lineup IQ game draws its challenge lineups from the **lower 40th percentile** of these scores, so this artifact lets `/game/start` select a genuinely flawed unit without scoring the whole pool on every request. Rebuild it after retuning the synergy engine or rebuilding the season baselines.
- **How to run**:
  ```bash
  python app/scripts/build_lineup_synergy_scores.py
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

- **Description**: Returns detailed biographical stats, player playstyle role, career playstyle metrics, each metric's percentile against all careers from 1996 through 2025, PFV/APFV scores, the six `skill_ratings` (0–100, computed server-side via `analytics/skill_ratings.py` and used by the radar), and similar-player lists.
- **Optional `season` query parameter** (e.g. `?season=2024-25`, or `?season=2024`): returns the identical response shape but with every metric scoped to that single season — playstyle metrics and percentiles (ranked within that season), role, archetypes, PFV/APFV (ranked within the season pool), and a season-specific similar-player list. `career_span` is the season, `career_teams`/`career_games` reflect that season, and bio fields (height, weight, draft) are unchanged. Returns `400` for a malformed season, `404` if the season has no data or the player did not play that season.
- **PFV**: Polygonal Feature Value, the normalized area of the six-axis radar chart using raw population percentiles for `pts_per36`, `reb_per36`, `ast_per36`, `blk_per36`, `stl_per36`, and `ts_pct`.
- **APFV**: Adjusted PFV, a population-ranked version of PFV after applying the MPG adjustment once. Higher APFV means the player's radar shape holds up with stronger playing-time context.
- **Display percentiles**: Each `playstyle_metrics` percentile is a **global** (cross-position) rank against all careers, shrunk toward the median by sample-size credibility — the same basis as PFV/APFV. This avoids overstating skills a player is not known for (e.g. a small guard's blocks/rebounds rank against the whole league, not just other guards). `tov_per36` is inverted so a higher percentile is always better; the metric `value` remains the raw career value.
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

#### `GET /players/{player_id}/seasons`

- **Description**: Returns the seasons (most recent first) a player has stats for. Powers the season selector on the player profile.
- **Sample Response**: `{ "seasons": ["2024-25", "2023-24", "2022-23"] }`

#### `GET /players/leaders`

- **Description**: Paginated leaderboard of all players ranked by APFV (descending), used by the player table page. Like the prospects list, each item carries two parallel stat views for the Box Score / Fingerprint toggle:
  - `box_score`: per-36 and shooting-rate metrics, each as `{ value, percentile }` (league-wide percentiles, for color-coding).
  - `skill_ratings`: the six fingerprint skill ratings (0–100), computed server-side from the playstyle percentiles and APFV (`analytics/skill_ratings.py`) — the same values shown on the player profile radar.
  - `apfv`: the player's APFV (0–1).

  Career figures by default; pass `season` for a single-season view.
- **Qualification**: players below a minimum sample size (career/season games × MPG < ~300 total minutes) are excluded so a handful of garbage-time minutes can't top the per-36 columns. The bar is deliberately low — a full rookie season clears it easily.
- **Query Params**:
  - `season` (optional): `YYYY-YY` or `YYYY`; omit for career.
  - `role` (optional): one of the assigned role labels (e.g. `Playmaker`, `Interior Presence`).
  - `page` (default `1`), `page_size` (default `25`, max `100`).
  - `sort` (default `apfv`): `apfv`, `height`, `weight`, a skill axis (`pts_per36` … `ts_pct`, sorts by **rating**), or a box-score key prefixed with `box_` (e.g. `box_pts_per36`, sorts by **value**).
  - `order` (default `desc`): `asc` or `desc`.
- **Sample Request**: `GET /players/leaders?season=2023-24&role=Playmaker&sort=box_ast_per36&page=1&page_size=25`
- **Sample Response**:
  ```json
  {
    "players": [
      {
        "player_id": 2544,
        "player_name": "LeBron James",
        "height": "6-9",
        "weight": 250,
        "role": "Designated Scorer",
        "archetypes": ["balanced scorer", "box-score defender"],
        "apfv": 0.979,
        "box_score": {
          "pts_per36": { "value": 25.6, "percentile": 99.0 },
          "ast_per36": { "value": 7.1, "percentile": 96.0 },
          "ts_pct": { "value": 0.585, "percentile": 84.0 }
        },
        "skill_ratings": {
          "pts_per36": 100, "reb_per36": 85.6, "ast_per36": 100,
          "blk_per36": 84.6, "stl_per36": 93.9, "ts_pct": 100
        }
      }
    ],
    "page": 1,
    "page_size": 25,
    "total": 1950,
    "total_pages": 78,
    "seasons": ["2025-26", "2024-25"],
    "roles": ["Defensive Specialist", "Designated Scorer", "Playmaker"]
  }
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

- **Description**: Measures the strengths, weaknesses, projected style vector percentiles, and synergy score for a 5-player lineup within a specific season. Also returns matching similar historical team lineups. The style vector, synergy breakdown, and strengths/weaknesses are graded against that season's **starting-lineup baselines** (see `build_season_lineup_baselines.py`): each axis is the percentile of the lineup's collective metric within the season distribution, blended with the players' teams' real style attributes. A trait reads as a strength when its axis lands in the upper tail of the distribution and a weakness when it lands in the lower tail, so high-record lineups skew strength-heavy and low-record lineups weakness-heavy. The response also includes `strength_traits` and `weakness_traits` — the canonical trait labels (the same taxonomy the Lineup IQ game's diagnosis checklist uses).
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

### Lineup IQ Game Endpoints

#### `GET /game/start`
- **Description**: Initializes a new Lineup IQ game session. Selects a challenge starting lineup from the **lower 40th percentile** of synergy scores (a genuinely flawed unit worth optimising), using the precomputed `lineup_synergy_scores.json` artifact, and extracts the target swap team's roster. Each starter in `original_lineup` carries its per-36 stats (`pts_per36`, `ast_per36`, `reb_per36`, `stl_per36`, `blk_per36`, `fg3a_rate`) so the client can render full stat cards.
- **Query Parameters**: `mode` (string, optional, default `"current"`): Either `"current"` or `"all_time"`.
- **Sample Response**:
  ```json
  {
    "mode": "current",
    "original_team_id": 1610612762,
    "original_team_name": "Utah Jazz",
    "original_season": "2025-26",
    "original_lineup": [],
    "original_synergy": 44.2,
    "swap_team_name": "2025-26 New Orleans Pelicans",
    "swap_team_abbreviation": "NOP",
    "swap_season": "2025-26",
    "swap_roster": []
  }
  ```

#### `POST /game/evaluate-diagnosis`
- **Description**: Evaluates the user's checklist selections. Scores them where every incorrect pick cancels out a correct pick (floored at 0).
- **Request Body**:
  - `player_ids` (array of integers)
  - `season` (string)
  - `selected_traits` (array of strings)
- **Sample Response**:
  ```json
  {
    "diagnosis_score": 83.3,
    "correct_picks": ["Playmaking & Ball Movement", "Team Defense"],
    "wrong_picks": ["Scoring Firepower"],
    "missed_opportunities": ["Floor Spacing & Shooting"]
  }
  ```

#### `POST /game/evaluate-swap`
- **Description**: Evaluates the substitution, computes new synergy, and calculates the final Manager IQ Rating (weighing swap synergy 80% and diagnosis 20%).
- **Request Body**:
  - `original_player_ids` (array of integers)
  - `original_season` (string)
  - `player_out_id` (integer)
  - `player_in_id` (integer)
  - `player_in_season` (string)
  - `diagnosis_score` (float)
  - `selected_traits` (array of strings)
- **Sample Response**:
  ```json
  {
    "original_synergy": 44.2,
    "new_synergy": 45.7,
    "synergy_delta": 1.5,
    "diagnosis_score": 83.3,
    "final_score": 53.22,
    "breakdown": {
      "correct": ["Playmaking & Ball Movement", "Team Defense"],
      "missed": ["Floor Spacing & Shooting"],
      "wrong": ["Scoring Firepower"],
      "explanation": "Swapping out Jusuf Nurkić for Trey Murphy III improved team spacing but created role redundancies (-6.0)."
    }
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

The draft endpoints support both the current year's draft prospects and historical draft classes dating back to 2007.

- **Datasets**: Current-year prospects are loaded from `app/data/static/prospects.json`. Historical prospects datasets are stored by year under the `app/data/static/draft/` directory (e.g., `prospects_2023.json`).
- **Startup Indexing**: On application startup, the server scans the `app/data/static/draft/` directory and builds an in-memory lookup map linking each historical `prospect_id` to its corresponding draft year file.
- **Prospect APFV Calculations**: Adjusted Polygonal Feature Value (APFV) scores are computed against the combined pool of current prospects and all ingested historical draft classes, with height-bucket normalization applied across that full prospect population.

---

Requires `app/data/static/prospects.json` to be generated first via `run_pipeline.py --stages prospect`, or historical draft year datasets to be generated under `app/data/static/draft/` via `build_historical_prospects.py`.

#### `GET /draft/meta`

- **Description**: Metadata for the current draft class. Returns `{ "draft_class_year": <int> }`, the upcoming class year persisted in `prospects.json` (advanced on scrape; see the prospects pipeline). The frontend reads this instead of guessing the year from the calendar.

#### `GET /draft/prospects` (Current Prospects List)

- **Description**: Lists every prospect in the current draft class with their `prospect_id`, display name, college/team, physical metadata, playstyle `role`, and two parallel stat views used by the table's Box Score / Fingerprint toggle:
  - `box_score`: per-game counting stats, each as `{ value, percentile }`. Percentiles rank the stat **within the listed class population** so cells can be color-coded.
  - `skill_ratings`: the six fingerprint skill ratings (0–100) computed server-side from the playstyle percentiles and APFV (`analytics/skill_ratings.py`), the same values shown on the prospect profile radar.
  - `apfv`: the prospect's APFV (0–1), ranked against the combined current-plus-historical population.
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
      "pick": null,
      "apfv": 0.9515,
      "box_score": {
        "ppg": { "value": 25.5, "percentile": 100.0 },
        "rpg": { "value": 6.8, "percentile": 70.4 },
        "apg": { "value": 3.7, "percentile": 88.7 },
        "spg": { "value": 1.1, "percentile": 60.0 },
        "bpg": { "value": 0.3, "percentile": 33.0 },
        "mpg": { "value": 34.8, "percentile": 96.5 },
        "gp": { "value": 35.0, "percentile": 78.3 },
        "fg_pct": { "value": 0.51, "percentile": 65.2 },
        "fg3_pct": { "value": 0.331, "percentile": 55.7 },
        "ft_pct": { "value": 0.774, "percentile": 60.9 }
      },
      "skill_ratings": {
        "pts_per36": 100.0,
        "reb_per36": 91.9,
        "ast_per36": 100.0,
        "blk_per36": 72.0,
        "stl_per36": 86.5,
        "ts_pct": 100.0
      }
    }
  ]
  ```

#### `GET /draft/prospects?year={year}` (Historical Prospects List)

- **Description**: Same response shape as the current list for a specific historical draft class since 2007 (box-score percentiles are ranked within that class).
- **Query Parameters**:
  - `year` (integer, required): The draft year to fetch (e.g., `2023`). Returns a `404` error if the specified year has not been ingested.
- **Differences in Response Shape**:
  - Each item additionally carries a `"pick"` attribute (integer) for the player's draft selection number.

#### `GET /draft/{prospect_id}`

- **Description**: Returns the complete dataset for a single prospect from the current draft class, including physical metadata and playstyle role. Non-raw stats include both the raw value and percentile rank. APFV is ranked against the combined current-plus-historical prospect population. RealGM profile URLs are not included in API responses. Returns a `404` error if the prospect is not found in the current or historical draft classes.
- **Path Parameters**: `prospect_id` (string) — slugified name, e.g., `a-j-dybantsa`
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

#### `GET /draft/{historical_prospect_id}` (Historical Detail Lookup Fallback)

- **Description**: If a requested `prospect_id` is not found in the current draft class, the server automatically queries the historical prospects index. If a matching player is found in a historical draft class since 2007, it resolves their draft year, calculates prospect-population APFV rankings, and returns their full profile.
- **Differences in Response Shape**:
  - Includes a `"pick"` attribute (integer) representing the selection number in their draft class.
  - The `"raw_stats.apfv"` value is computed against the combined current and historical prospect population.
  - The `"similar_nba_players"` matches are computed using the historical database and will reflect raw coordinates matching their pre-draft profile.
- **Example Path**: `GET /draft/victor-wembanyama`
- **Example Response**:
  ```json
  {
    "prospect_id": "victor-wembanyama",
    "player_name": "Victor Wembanyama",
    "team": "All Teams",
    "height": "7-4",
    "weight": "235",
    "role": "Designated Scorer",
    "gp": 44,
    "mpg": {
      "value": 32.2,
      "percentile": 69.0
    },
    "pts_per36": {
      "value": 23.3665,
      "percentile": 98.3
    },
    "reb_per36": {
      "value": 11.5155,
      "percentile": 96.6
    },
    "ast_per36": {
      "value": 2.6832,
      "percentile": 60.3
    },
    "blk_per36": {
      "value": 3.354,
      "percentile": 98.3
    },
    "stl_per36": {
      "value": 0.8944,
      "percentile": 25.9
    },
    "ts_pct": {
      "value": 0.5715,
      "percentile": 44.8
    },
    "efg_pct": {
      "value": 0.5096,
      "percentile": 34.5
    },
    "fg3a_rate": {
      "value": 0.3013,
      "percentile": 37.9
    },
    "fta_rate": {
      "value": 0.391,
      "percentile": 81.0
    },
    "pick": 1,
    "similar_nba_players": [
      {
        "player_id": 1885,
        "player_name": "Lamar Odom",
        "similarity_score": 35.5,
        "career_span": "1999-00 to 2012-13",
        "position_group": "W",
        "role": "Secondary Creator"
      },
      {
        "player_id": 1717,
        "player_name": "Dirk Nowitzki",
        "similarity_score": 26.1,
        "career_span": "1998-99 to 2018-19",
        "position_group": "W",
        "role": "Designated Scorer"
      },
      {
        "player_id": 2200,
        "player_name": "Pau Gasol",
        "similarity_score": 24.5,
        "career_span": "2001-02 to 2018-19",
        "position_group": "W",
        "role": "Designated Scorer"
      },
      {
        "player_id": 934,
        "player_name": "Derrick Coleman",
        "similarity_score": 20.6,
        "career_span": "1996-97 to 2004-05",
        "position_group": "W",
        "role": "Rim Attacker"
      }
    ],
    "raw_stats": {
      "gp": 44,
      "mpg": 32.2,
      "ppg": 20.9,
      "fgm": 7.3,
      "fga": 15.6,
      "fg_pct": 0.468,
      "fg3m": 1.3,
      "fg3a": 4.7,
      "fg3_pct": 0.272,
      "ftm": 5.0,
      "fta": 6.1,
      "ft_pct": 0.818,
      "rpg": 10.3,
      "apg": 2.4,
      "spg": 0.8,
      "bpg": 3.0,
      "pfv": 0.4893,
      "apfv": 0.8581
    }
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
