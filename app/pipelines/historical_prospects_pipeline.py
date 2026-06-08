from __future__ import annotations

import json
import logging
import re
import time
import random
from pathlib import Path
from io import StringIO
import pandas as pd
from bs4 import BeautifulSoup

from config import DATA_STATIC_DIR, PLAYER_CAREER_FEATURES_PATH
from pipelines.prospects_pipeline import ProspectsPipeline, fetch_html, add_prospect_features

logger = logging.getLogger(__name__)

# Cache locations
HISTORICAL_DIR = Path(__file__).resolve().parents[1] / "data" / "historical"
PLAYERS_CACHE_DIR = HISTORICAL_DIR / "players"

def fetch_html_with_retry(url: str, retries: int = 3, backoff: float = 2.0) -> str:
    """Wrapper around fetch_html that adds exponential backoff and jitter."""
    for attempt in range(retries + 1):
        try:
            return fetch_html(url)
        except Exception as e:
            if attempt == retries:
                logger.error("Failed to fetch %s after %d retries: %s", url, retries, e)
                raise
            sleep_time = backoff * (2 ** attempt) + random.uniform(0.1, 1.0)
            logger.warning("Error fetching %s (attempt %d/%d): %s. Retrying in %.2fs...", 
                           url, attempt + 1, retries + 1, e, sleep_time)
            time.sleep(sleep_time)
    raise RuntimeError("Unreachable")

def parse_season_end_year(season_str: str) -> int | None:
    """Parses season string like '2022-23' or '2023' to end year (2023)."""
    cleaned = re.sub(r'[^0-9\-]', '', str(season_str)).strip()
    if '-' in cleaned:
        parts = cleaned.split('-')
        if len(parts) == 2:
            try:
                start = int(parts[0])
                if len(parts[1]) == 2:
                    end = (start // 100) * 100 + int(parts[1])
                else:
                    end = int(parts[1])
                return end
            except ValueError:
                pass
    else:
        try:
            return int(cleaned)
        except ValueError:
            pass
    return None

class HistoricalProspectsPipeline(ProspectsPipeline):
    def __init__(
        self,
        career_features_path: Path | str = PLAYER_CAREER_FEATURES_PATH,
        json_output_path: Path | str | None = None,
        parquet_output_path: Path | str | None = None,
        similar_count: int = 4,
    ):
        super().__init__(
            source_url="",
            career_features_path=career_features_path,
            json_output_path=json_output_path or "",
            parquet_output_path=parquet_output_path or "",
            similar_count=similar_count
        )
        # Ensure directories exist
        HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)
        PLAYERS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def crawl_draft_class_players(self, year: int, force: bool = False) -> list[dict]:
        """Crawl the list of drafted players for the given year."""
        raw_cache_path = HISTORICAL_DIR / f"draft_class_{year}_raw.json"
        if raw_cache_path.exists() and not force:
            logger.info("Loading draft class %d from cache: %s", year, raw_cache_path.name)
            with raw_cache_path.open(encoding="utf-8") as f:
                return json.load(f)

        url = f"https://basketball.realgm.com/nba/draft/past_drafts/{year}"
        logger.info("Fetching draft class page: %s", url)
        html = fetch_html_with_retry(url)
        
        soup = BeautifulSoup(html, "html.parser")
        tables = soup.find_all("table")
        prospects = []
        
        for idx, table in enumerate(tables):
            headers = [th.get_text(strip=True) for th in table.find_all("th")]
            if not ("Pick" in headers and "Player" in headers and "Team" in headers):
                continue
                
            logger.info("Parsing draft table %d with headers: %s", idx, headers)
            
            try:
                pick_idx = headers.index("Pick")
                player_idx = headers.index("Player")
                team_idx = headers.index("Team")
                pos_idx = headers.index("Pos") if "Pos" in headers else -1
                ht_idx = headers.index("HT") if "HT" in headers else -1
                wt_idx = headers.index("WT") if "WT" in headers else -1
                age_idx = headers.index("Age") if "Age" in headers else -1
                predraft_idx = headers.index("Pre-Draft Team") if "Pre-Draft Team" in headers else -1
                nat_idx = headers.index("Nationality") if "Nationality" in headers else -1
            except ValueError as e:
                logger.error("Skipping table %d due to missing header: %s", idx, e)
                continue
                
            rows = table.find("tbody").find_all("tr") if table.find("tbody") else table.find_all("tr")
            for row in rows:
                cells = row.find_all("td")
                if not cells or len(cells) < len(headers):
                    continue
                    
                player_cell = cells[player_idx]
                a_tag = player_cell.find("a", href=True)
                if not a_tag:
                    continue
                player_link = a_tag["href"]
                player_name = a_tag.text.strip()
                
                pick_val = cells[pick_idx].text.strip()
                team_val = cells[team_idx].text.strip()
                pos_val = cells[pos_idx].text.strip() if pos_idx != -1 else ""
                ht_val = cells[ht_idx].text.strip() if ht_idx != -1 else ""
                wt_val = cells[wt_idx].text.strip() if wt_idx != -1 else ""
                age_val = cells[age_idx].text.strip() if age_idx != -1 else ""
                predraft_val = cells[predraft_idx].text.strip() if predraft_idx != -1 else ""
                nat_val = cells[nat_idx].text.strip() if nat_idx != -1 else ""
                
                prospects.append({
                    "player_name": player_name,
                    "profile_link": player_link,
                    "draft_year": year,
                    "draft_position": int(pick_val) if pick_val.isdigit() else pick_val,
                    "draft_team": team_val,
                    "position": pos_val,
                    "height": ht_val,
                    "weight": wt_val,
                    "age": int(age_val) if age_val.isdigit() else age_val,
                    "pre_draft_team": predraft_val,
                    "nationality": nat_val
                })
                
        # Save cache
        with raw_cache_path.open("w", encoding="utf-8") as f:
            json.dump(prospects, f, indent=2)
            
        logger.info("Saved draft class raw players list to cache: %s", raw_cache_path.name)
        return prospects

    def fetch_player_pre_draft_stats(self, profile_link: str, draft_year: int, force: bool = False) -> dict | None:
        """Fetch pre-draft season stats row from the player's profile page."""
        # Extract player ID
        # e.g., /player/Victor-Wembanyama/Summary/136048 -> 136048
        player_id = profile_link.split("/")[-1]
        player_cache_path = PLAYERS_CACHE_DIR / f"{player_id}.json"
        
        if player_cache_path.exists() and not force:
            with player_cache_path.open(encoding="utf-8") as f:
                return json.load(f)

        url = f"https://basketball.realgm.com{profile_link}"
        logger.info("Fetching player profile: %s", url)
        try:
            html = fetch_html_with_retry(url)
        except Exception as e:
            logger.error("Failed to fetch player profile %s: %s", url, e)
            return None
            
        soup = BeautifulSoup(html, "html.parser")
        dom_tables = soup.find_all("table")
        tables = pd.read_html(StringIO(html))
        
        # 1. Identify all eligible regular season stats tables
        eligible_tables = []
        for idx, table in enumerate(dom_tables):
            prev_h2 = table.find_previous(["h2", "h3", "h1"])
            title = prev_h2.text.strip() if prev_h2 else "None"
            parent_id = table.parent.get("id") or ""
            
            title_lower = title.lower()
            if any(x in title_lower for x in ["nba", "summer league", "preseason", "play-in", "playoff", "rising stars", "all-star", "awards", "transactions", "experience", "camp", "high school", "prep"]):
                continue
            
            headers = [str(th.get_text(strip=True)) for th in table.find_all("th")]
            if not ("Season" in headers or "Year" in headers):
                continue
            if not ("GP" in headers and "MIN" in headers):
                continue
                
            eligible_tables.append((idx, title, parent_id, headers, tables[idx]))
            
        # 2. Find the final pre-draft season end year <= draft_year
        best_season_end_year = -1
        best_season_label = ""
        
        for idx, title, parent_id, headers, df in eligible_tables:
            season_col = "Season" if "Season" in df.columns else "Year"
            for _, row in df.iterrows():
                season_val = row[season_col]
                if pd.isna(season_val) or str(season_val).upper() in ["CAREER", "TOTAL"]:
                    continue
                end_year = parse_season_end_year(season_val)
                if end_year and end_year <= draft_year:
                    if end_year > best_season_end_year:
                        best_season_end_year = end_year
                        best_season_label = str(season_val)
                        
        if best_season_end_year == -1:
            logger.warning("No pre-draft season found for %s", profile_link)
            # Cache empty dict to avoid re-fetching
            with player_cache_path.open("w", encoding="utf-8") as f:
                json.dump({}, f)
            return None
            
        # 3. Gather all candidate rows matching the best pre-draft season
        candidate_rows = []
        for idx, title, parent_id, headers, df in eligible_tables:
            season_col = "Season" if "Season" in df.columns else "Year"
            if "FGM" not in df.columns or "FGA" not in df.columns:
                continue
            
            for _, row in df.iterrows():
                season_val = row[season_col]
                if pd.isna(season_val):
                    continue
                cleaned_val = re.sub(r'[^0-9\-]', '', str(season_val)).strip()
                cleaned_best = re.sub(r'[^0-9\-]', '', best_season_label).strip()
                if cleaned_val == cleaned_best:
                    # Safely parse PTS
                    pts_val = 0.0
                    try:
                        pts_str = str(row.get("PTS", 0.0)).strip()
                        if pts_str and pts_str != "-":
                            pts_val = float(pts_str)
                    except ValueError:
                        pass

                    # Safely parse MIN
                    min_val = str(row.get("MIN", "0")).strip()
                    if not min_val or min_val == "-":
                        min_float = 0.0
                    elif ':' in min_val:
                        try:
                            m, s = map(float, min_val.split(':'))
                            min_float = m + s/60.0
                        except ValueError:
                            min_float = 0.0
                    else:
                        try:
                            min_float = float(min_val)
                        except ValueError:
                            min_float = 0.0
                    
                    # Safely parse GP
                    gp_val = 0.0
                    try:
                        gp_str = str(row.get("GP", 0.0)).strip()
                        if gp_str and gp_str != "-":
                            gp_val = float(gp_str)
                    except ValueError:
                        gp_val = 0.0

                    if gp_val > 0 and min_float > 0:
                        is_totals = min_float > 50.0  # Totals usually has MIN in hundreds
                        if not is_totals:
                            # Convert series to dict
                            row_dict = row.to_dict()
                            # Convert numpy values to standard python types for JSON serialization
                            serializable_dict = {}
                            for k, v in row_dict.items():
                                if pd.isna(v):
                                    serializable_dict[k] = None
                                elif hasattr(v, "item"):
                                    serializable_dict[k] = v.item()
                                else:
                                    serializable_dict[k] = v
                            candidate_rows.append(serializable_dict)

        if not candidate_rows:
            logger.warning("No stats rows found matching season %s for %s", best_season_label, profile_link)
            with player_cache_path.open("w", encoding="utf-8") as f:
                json.dump({}, f)
            return None
            
        # 4. Selection priority
        selected_row = None
        # Look for 'All Teams'
        for row in candidate_rows:
            team_col = "School" if "School" in row else ("Team" if "Team" in row else "Event")
            team_val = str(row.get(team_col, ""))
            if "All Teams" in team_val:
                selected_row = row
                break
                
        if selected_row is None:
            # Sort by MIN descending
            def get_min_float(row):
                min_val = str(row.get("MIN", "0"))
                if ':' in min_val:
                    try:
                        m, s = map(float, min_val.split(':'))
                        return m + s/60.0
                    except ValueError:
                        return 0.0
                try:
                    return float(min_val)
                except ValueError:
                    return 0.0
            candidate_rows.sort(key=get_min_float, reverse=True)
            selected_row = candidate_rows[0]

        # Resolve real team name if the selected row uses 'All Teams'
        if selected_row:
            team_col = "School" if "School" in selected_row else ("Team" if "Team" in selected_row else "Event")
            team_val = str(selected_row.get(team_col, ""))
            if "All Teams" in team_val:
                best_other_team = None
                max_gp = -1
                for row in candidate_rows:
                    t_val = str(row.get(team_col, ""))
                    if "All Teams" not in t_val:
                        try:
                            gp_val = float(str(row.get("GP", 0.0)).strip().replace("-", "0") or 0.0)
                        except ValueError:
                            gp_val = 0.0
                        if gp_val > max_gp:
                            max_gp = gp_val
                            best_other_team = t_val
                if best_other_team:
                    selected_row = dict(selected_row)
                    selected_row[team_col] = best_other_team

        # Cache the selected row
        with player_cache_path.open("w", encoding="utf-8") as f:
            json.dump(selected_row, f, indent=2)
            
        return selected_row

    def process_draft_class(self, year: int, force: bool = False, sleep_time: float = 1.5):
        """Crawls, scrapes, and processes a draft class year."""
        logger.info("--------------------------------------------------")
        logger.info("Processing Draft Class Year: %d", year)
        
        # 1. Crawl players list
        players = self.crawl_draft_class_players(year, force=force)
        logger.info("Found %d players in draft class %d.", len(players), year)
        
        # 2. Scrape pre-draft stats for each player
        mapped_records = []
        for idx, player in enumerate(players):
            name = player["player_name"]
            link = player["profile_link"]
            logger.info("[%d/%d] Scraping stats for %s...", idx + 1, len(players), name)
            
            stats_row = None
            if link:
                player_id = link.split("/")[-1]
                player_cache_path = PLAYERS_CACHE_DIR / f"{player_id}.json"
                is_cached = player_cache_path.exists()
                
                stats_row = self.fetch_player_pre_draft_stats(link, year, force=force)
                
                if force or not is_cached:
                    time.sleep(sleep_time) # Respect rate limit
                
            # Map row to the normalized schema input
            mapped = self.map_prospect_row(player, stats_row)
            mapped_records.append(mapped)
            
        # 3. Create DataFrame
        df = pd.DataFrame(mapped_records)
        
        # 4. Add features
        prospects = add_prospect_features(df)
        
        # 5. Read career features for NBA counterparts
        if not self.career_features_path.exists():
            raise FileNotFoundError(
                f"Career features parquet not found: {self.career_features_path}. "
                "Ensure player profiles pipeline has run first."
            )
        career = self.read_parquet_compat(self.career_features_path)
        
        # 6. Add similar NBA players & roles
        logger.info("Adding similar NBA players and playstyle roles...")
        prospects = self.add_similar_nba_players(prospects, career)
        prospects = self.add_prospect_roles(prospects)
        
        # 7. Write outputs
        logger.info("Writing finalized outputs for %d draft class...", year)
        self.write_outputs(prospects)
        logger.info("Successfully processed draft class %d. Output saved.", year)
        print(f"✓ Draft year {year} completed: {len(prospects)} prospects processed.")

    def map_prospect_row(self, player: dict, stats_row: dict | None) -> dict:
        """Maps crawled player details and scraped stats to standard DataFrame columns."""
        def parse_mpg(val):
            if pd.isna(val) or val is None:
                return 0.0
            val_str = str(val).strip()
            if ':' in val_str:
                try:
                    m, s = map(float, val_str.split(':'))
                    return m + s/60.0
                except ValueError:
                    return 0.0
            try:
                return float(val_str)
            except ValueError:
                return 0.0

        def safe_float(val, default=0.0):
            if pd.isna(val) or val is None:
                return default
            val_str = str(val).strip()
            if not val_str or val_str == "-":
                return default
            try:
                return float(val_str)
            except ValueError:
                return default

        def safe_int(val, default=0):
            if pd.isna(val) or val is None:
                return default
            val_str = str(val).strip()
            if not val_str or val_str == "-":
                return default
            try:
                return int(float(val_str))
            except ValueError:
                return default

        if stats_row is None:
            stats_row = {}

        pre_draft_team = player.get("pre_draft_team") or ""
        team_val = stats_row.get("School") or stats_row.get("Team") or stats_row.get("Event") or pre_draft_team
        rpg_val = stats_row.get("TRB") or stats_row.get("REB") or stats_row.get("RPG") or 0.0

        return {
            "Player": player["player_name"],
            "Team": str(team_val).strip(),
            "GP": safe_int(stats_row.get("GP")),
            "MPG": parse_mpg(stats_row.get("MIN") or stats_row.get("MPG")),
            "PPG": safe_float(stats_row.get("PTS") or stats_row.get("PPG")),
            "FGM": safe_float(stats_row.get("FGM")),
            "FGA": safe_float(stats_row.get("FGA")),
            "FG%": safe_float(stats_row.get("FG%")),
            "3PM": safe_float(stats_row.get("3PM")),
            "3PA": safe_float(stats_row.get("3PA")),
            "3P%": safe_float(stats_row.get("3P%")),
            "FTM": safe_float(stats_row.get("FTM")),
            "FTA": safe_float(stats_row.get("FTA")),
            "FT%": safe_float(stats_row.get("FT%")),
            "RPG": safe_float(rpg_val),
            "APG": safe_float(stats_row.get("AST") or stats_row.get("APG")),
            "SPG": safe_float(stats_row.get("STL") or stats_row.get("SPG")),
            "BPG": safe_float(stats_row.get("BLK") or stats_row.get("BPG")),
            "height": player.get("height", ""),
            "weight": player.get("weight", ""),
            "profile_link": player.get("profile_link", ""),
            "pick": int(player.get("draft_position", 0)) if str(player.get("draft_position", "")).isdigit() else player.get("draft_position", 0)
        }

    def write_outputs(self, prospects: pd.DataFrame) -> None:
        """First call ProspectsPipeline's write_outputs to write locally, then sync to GCS."""
        super().write_outputs(prospects)
        
        import os
        gcs_uri = os.getenv("PLAYER_PROFILES_STORAGE_URI")
        if gcs_uri and gcs_uri.startswith("gs://"):
            logger.info("PLAYER_PROFILES_STORAGE_URI is configured for GCS: %s. Initiating prospects upload...", gcs_uri)
            self.upload_file_to_gcs(self.json_output_path, gcs_uri)
            self.upload_file_to_gcs(self.parquet_output_path, gcs_uri)

    def upload_file_to_gcs(self, local_path: Path, gcs_uri: str) -> None:
        """Uploads a local file to GCS under both draft/ and prefix/draft/ keys."""
        try:
            from google.cloud import storage as gcs_storage
        except ModuleNotFoundError:
            logger.warning("google-cloud-storage not installed; skipping GCS upload.")
            return

        import os
        rest = gcs_uri[5:].strip("/")
        bucket_name, _, prefix = rest.partition("/")
        
        project = (
            os.getenv("GOOGLE_CLOUD_PROJECT")
            or os.getenv("GCLOUD_PROJECT")
        )
        try:
            client = gcs_storage.Client(project=project)
            bucket = client.bucket(bucket_name)
        except Exception as e:
            logger.warning("Failed to initialize GCS client: %s", e)
            return

        file_name = local_path.name
        content_type = (
            "application/json" if local_path.suffix == ".json" 
            else "application/vnd.apache.parquet"
        )

        paths_to_upload = []
        if prefix:
            paths_to_upload.append(f"{prefix}/draft/{file_name}")
        paths_to_upload.append(f"draft/{file_name}")

        for blob_name in paths_to_upload:
            try:
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(str(local_path), content_type=content_type)
                logger.info("✓ Uploaded %s to gs://%s/%s", local_path.name, bucket_name, blob_name)
            except Exception as e:
                logger.error("Failed to upload %s to gs://%s/%s: %s", local_path.name, bucket_name, blob_name, e)
