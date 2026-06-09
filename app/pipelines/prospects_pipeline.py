from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from config import PROSPECTS_JSON_PATH, PLAYER_CAREER_FEATURES_PATH
from analytics.similarity_scoring import similarity_pct
from analytics.player_profiles.archetypes import (
    calculate_adjusted_pfv,
    calculate_apfv_batch_by_height,
    calculate_pfv,
    height_bucket,
)

logger = logging.getLogger(__name__)

DEFAULT_SOURCE_URL = "https://basketball.realgm.com/nba/draft/prospects/stats"
DEFAULT_JSON_OUTPUT = PROSPECTS_JSON_PATH
DEFAULT_PARQUET_OUTPUT = PROSPECTS_JSON_PATH.with_suffix(".parquet")
DEFAULT_CAREER_FEATURES = PLAYER_CAREER_FEATURES_PATH

_fetch_lock = threading.Lock()
_fetch_subprocess_python: str | None = None
_fetch_subprocess_python_initialized = False

PROSPECT_TABLE_REQUIRED_COLUMNS = {
    "Player",
    "Team",
    "GP",
    "MPG",
    "PPG",
    "FGM",
    "FGA",
    "3PM",
    "3PA",
    "FTM",
    "FTA",
    "RPG",
    "APG",
    "SPG",
    "BPG",
}

SIMILARITY_FEATURES = [
    "pts_per36",
    "reb_per36",
    "ast_per36",
    "blk_per36",
    "stl_per36",
    "ts_pct",
    "efg_pct",
    "fg3a_rate",
    "fta_rate",
    "mpg",
]

RAW_STAT_JSON_KEYS = {
    "GP": "gp",
    "MPG": "mpg",
    "PPG": "ppg",
    "FGM": "fgm",
    "FGA": "fga",
    "FG%": "fg_pct",
    "3PM": "fg3m",
    "3PA": "fg3a",
    "3P%": "fg3_pct",
    "FTM": "ftm",
    "FTA": "fta",
    "FT%": "ft_pct",
    "RPG": "rpg",
    "APG": "apg",
    "SPG": "spg",
    "BPG": "bpg",
}

_PERCENTILE_COLS = [
    "mpg",
    "pts_per36",
    "reb_per36",
    "ast_per36",
    "blk_per36",
    "stl_per36",
    "ts_pct",
    "efg_pct",
    "fg3a_rate",
    "fta_rate",
]


def find_openssl_python() -> str | None:
    import os
    import subprocess
    common_paths = [
        "/opt/anaconda3/bin/python3",
        "/opt/anaconda3/bin/python",
        "/opt/homebrew/bin/python3",
        "/usr/local/bin/python3",
    ]
    for path in common_paths:
        if Path(path).exists():
            try:
                out = subprocess.check_output(
                    [path, "-c", "import ssl; print('OpenSSL' in ssl.OPENSSL_VERSION)"],
                    timeout=2,
                ).decode().strip()
                if out == "True":
                    return path
            except Exception:
                pass

    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    checked = set()
    for directory in path_dirs:
        if not directory:
            continue
        for name in ["python3", "python"]:
            full_path = os.path.join(directory, name)
            if os.path.isfile(full_path) and os.access(full_path, os.X_OK):
                real_path = os.path.realpath(full_path)
                if real_path in checked or real_path == os.path.realpath(sys.executable):
                    continue
                checked.add(real_path)
                try:
                    out = subprocess.check_output(
                        [real_path, "-c", "import ssl; print('OpenSSL' in ssl.OPENSSL_VERSION)"],
                        timeout=2,
                    ).decode().strip()
                    if out == "True":
                        return real_path
                except Exception:
                    pass
    return None


def _get_fetch_subprocess_python() -> str | None:
    global _fetch_subprocess_python, _fetch_subprocess_python_initialized
    if _fetch_subprocess_python_initialized:
        return _fetch_subprocess_python

    with _fetch_lock:
        if _fetch_subprocess_python_initialized:
            return _fetch_subprocess_python

        import ssl

        if "LibreSSL" in ssl.OPENSSL_VERSION:
            openssl_python = find_openssl_python()
            if openssl_python:
                logger.info(
                    "Current python uses LibreSSL; falling back to %s for fetching.",
                    openssl_python,
                )
                _fetch_subprocess_python = openssl_python

        _fetch_subprocess_python_initialized = True
        return _fetch_subprocess_python


def fetch_html(url: str) -> str:
    import subprocess

    openssl_python = _get_fetch_subprocess_python()
    if openssl_python:
        fetch_script = (
            "import urllib.request\n"
            "import sys\n"
            "url = sys.argv[1]\n"
            "headers = {\n"
            "    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',\n"
            "    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',\n"
            "    'Accept-Language': 'en-US,en;q=0.9',\n"
            "    'Cache-Control': 'max-age=0',\n"
            "    'Connection': 'keep-alive',\n"
            "    'Upgrade-Insecure-Requests': '1'\n"
            "}\n"
            "req = urllib.request.Request(url, headers=headers)\n"
            "with urllib.request.urlopen(req, timeout=30) as resp:\n"
            "    sys.stdout.buffer.write(resp.read())\n"
        )
        try:
            out = subprocess.check_output(
                [openssl_python, "-c", fetch_script, url],
                timeout=30,
            )
            return out.decode("utf-8", errors="replace")
        except Exception as exc:
            logger.warning("Subprocess fetch failed: %s. Falling back to local urllib fetch.", exc)

    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "max-age=0",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        },
    )
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def scrape_prospect_stats(html: str) -> pd.DataFrame:
    tables = pd.read_html(StringIO(html))
    target_table_idx = -1
    for idx, table in enumerate(tables):
        columns = {str(col).strip() for col in table.columns}
        if PROSPECT_TABLE_REQUIRED_COLUMNS.issubset(columns):
            target_table_idx = idx
            break

    if target_table_idx == -1:
        raise RuntimeError("Could not find the RealGM draft prospect stats table.")

    out = tables[target_table_idx].copy()
    out.columns = [str(col).strip() for col in out.columns]

    soup = BeautifulSoup(html, "lxml")
    dom_tables = soup.find_all("table")
    dom_table = None
    for table in dom_tables:
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        if "Player" in headers and "GP" in headers:
            dom_table = table
            break

    if not dom_table:
        raise RuntimeError("Could not find the RealGM draft prospect stats table using BeautifulSoup.")

    rows = dom_table.find("tbody").find_all("tr")
    profile_links = []
    for row in rows:
        cells = row.find_all("td")
        if not cells:
            continue
        player_link = ""
        for cell in cells:
            a_tag = cell.find("a", href=True)
            if a_tag and "/player/" in a_tag['href'] and "/Summary/" in a_tag['href']:
                player_link = a_tag['href']
                break
        profile_links.append(player_link)

    if len(profile_links) != len(out):
        logger.warning(
            "Mismatch in row counts: pandas has %d rows, BeautifulSoup extracted %d links",
            len(out),
            len(profile_links),
        )
        if len(profile_links) < len(out):
            profile_links.extend([""] * (len(out) - len(profile_links)))
        else:
            profile_links = profile_links[:len(out)]

    out["profile_link"] = profile_links
    return out


def fetch_heights_weights(prospects: pd.DataFrame) -> pd.DataFrame:
    out = prospects.copy()
    heights = {}
    weights = {}
    teams = {}

    unique_links = {link for link in out["profile_link"] if link}
    logger.info("Fetching profiles for %d prospects to parse height, weight, and team...", len(unique_links))

    def fetch_one(link: str) -> tuple[str, str, str, str]:
        url = f"https://basketball.realgm.com{link}"
        try:
            html = fetch_html(url)
            h_match = re.search(r'<strong>Height:</strong>\s*([0-9]+-[0-9]+)', html, re.IGNORECASE)
            w_match = re.search(r'<strong>Weight:</strong>\s*([0-9]+)', html, re.IGNORECASE)
            h = h_match.group(1).strip() if h_match else ""
            w = w_match.group(1).strip() if w_match else ""

            # Extract full team name
            team = ""
            soup = BeautifulSoup(html, "html.parser")
            for label in ["Current Team:", "Current School:", "College:", "High School:", "Prep/High School:"]:
                for p in soup.find_all("p"):
                    strong = p.find("strong")
                    if strong and label in strong.get_text(strip=True):
                        a_tag = p.find("a")
                        if a_tag:
                            val = a_tag.get_text(strip=True)
                        else:
                            val = p.get_text(strip=True).replace(strong.get_text(strip=True), "").strip()
                        val = re.sub(r'\s*\(\d{4}\)', '', val).strip()
                        if val:
                            team = val
                            break
                if team:
                    break

            return link, h, w, team
        except Exception as e:
            logger.warning("Error fetching/parsing profile %s: %s", url, e)
            return link, "", "", ""

    from tqdm import tqdm
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_one, link): link for link in unique_links}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching prospect bio details", unit="prospect"):
            link, h, w, t = future.result()
            heights[link] = h
            weights[link] = w
            teams[link] = t

    out["height"] = out["profile_link"].map(heights).fillna("")
    out["weight"] = out["profile_link"].map(weights).fillna("")

    full_teams = out["profile_link"].map(teams).fillna("")
    out["Team"] = np.where(full_teams != "", full_teams, out["Team"])

    return out



def per36(values: pd.Series, mpg: pd.Series) -> pd.Series:
    return (values / mpg * 36.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator / denominator.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def slugify_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")
    return slug or "unknown"


def add_prospect_features(prospects: pd.DataFrame) -> pd.DataFrame:
    out = prospects.copy()
    if "pick" not in out.columns:
        out["pick"] = None
    
    # Coerce all numeric columns to numeric, converting non-numbers to NaN
    numeric_cols = [
        "GP", "MPG", "PPG", "FGM", "FGA", "FG%", 
        "3PM", "3PA", "3P%", "FTM", "FTA", "FT%", 
        "RPG", "APG", "SPG", "BPG"
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    mpg = out["MPG"].replace(0, np.nan)
    out["prospect_id"] = out["Player"].apply(slugify_name)
    out["player_name"] = out["Player"].astype(str)
    out["team"] = out["Team"].fillna("").astype(str)
    out["gp"] = out["GP"].fillna(0).astype(int)
    out["mpg"] = out["MPG"].fillna(0.0)
    out["pts_per36"] = per36(out["PPG"], mpg)
    out["ast_per36"] = per36(out["APG"], mpg)
    out["reb_per36"] = per36(out["RPG"], mpg)
    out["stl_per36"] = per36(out["SPG"], mpg)
    out["blk_per36"] = per36(out["BPG"], mpg)
    out["fg3a_rate"] = safe_div(out["3PA"], out["FGA"])
    out["fta_rate"] = safe_div(out["FTA"], out["FGA"])
    out["ts_pct"] = safe_div(out["PPG"], 2.0 * (out["FGA"] + 0.44 * out["FTA"]))
    out["efg_pct"] = safe_div(out["FGM"] + 0.5 * out["3PM"], out["FGA"])

    fill_vals = {
        "GP": 0,
        "MPG": 0.0,
        "PPG": 0.0,
        "FGM": 0.0,
        "FGA": 0.0,
        "FG%": 0.0,
        "3PM": 0.0,
        "3PA": 0.0,
        "3P%": 0.0,
        "FTM": 0.0,
        "FTA": 0.0,
        "FT%": 0.0,
        "RPG": 0.0,
        "APG": 0.0,
        "SPG": 0.0,
        "BPG": 0.0,
        "pts_per36": 0.0,
        "ast_per36": 0.0,
        "reb_per36": 0.0,
        "stl_per36": 0.0,
        "blk_per36": 0.0,
        "fg3a_rate": 0.0,
        "fta_rate": 0.0,
        "ts_pct": 0.0,
        "efg_pct": 0.0,
    }
    
    if "pfv" in out.columns:
        fill_vals["pfv"] = 0.0
    if "apfv" in out.columns:
        fill_vals["apfv"] = 0.0

    return out.replace([np.inf, -np.inf], np.nan).fillna(fill_vals)


def _height_to_inches(height_val: Any) -> float:
    if pd.isna(height_val) or not height_val:
        return 0.0
    val_str = str(height_val).strip()
    if not val_str:
        return 0.0
    try:
        return float(val_str)
    except ValueError:
        pass
    for sep in ("-", "'", '"', "/"):
        if sep in val_str:
            parts = val_str.split(sep)
            if len(parts) >= 2:
                try:
                    feet = float(parts[0].strip())
                    inches = float(parts[1].replace('"', "").strip())
                    return feet * 12.0 + inches
                except ValueError:
                    pass
    return 0.0


def _clean_weight(weight_val: Any, default: float = 215.0) -> float:
    if pd.isna(weight_val) or not weight_val:
        return default
    try:
        return float(str(weight_val).strip())
    except ValueError:
        return default


def _era_bucket(career_span: str) -> str:
    try:
        parts = career_span.split(" to ")
        start_year = int(parts[0].split("-")[0])
        end_year = int(parts[-1].split("-")[0])
        mid = (start_year + end_year) / 2.0
    except Exception:
        return "2010s"
    if mid < 2001:
        return "late-90s"
    elif mid < 2006:
        return "early-2000s"
    elif mid < 2011:
        return "late-2000s"
    elif mid < 2016:
        return "early-2010s"
    elif mid < 2021:
        return "late-2010s"
    else:
        return "2020s"


def _select_era_diverse(
    ranked_indices: np.ndarray,
    era_buckets: list[str],
    count: int,
    max_per_era: int = 2,
) -> list[int]:
    selected: list[int] = []
    era_counts: dict[str, int] = {}
    for idx in ranked_indices:
        if len(selected) >= count:
            break
        era = era_buckets[int(idx)]
        if era_counts.get(era, 0) < max_per_era:
            selected.append(int(idx))
            era_counts[era] = era_counts.get(era, 0) + 1

    if len(selected) < count:
        for idx in ranked_indices:
            if len(selected) >= count:
                break
            if int(idx) not in selected:
                selected.append(int(idx))

    return selected


def _percentile_of_score(arr: np.ndarray, score: float) -> float:
    n = len(arr)
    if n == 0:
        return 0.0
    rank = float(np.sum(arr <= score))
    return round(rank / n * 100.0, 1)


class ProspectsPipeline:
    def __init__(
        self,
        source_url: str = DEFAULT_SOURCE_URL,
        career_features_path: Path | str = DEFAULT_CAREER_FEATURES,
        json_output_path: Path | str = DEFAULT_JSON_OUTPUT,
        parquet_output_path: Path | str = DEFAULT_PARQUET_OUTPUT,
        similar_count: int = 4,
    ):
        self.source_url = source_url
        self.career_features_path = Path(career_features_path)
        self.json_output_path = Path(json_output_path)
        self.parquet_output_path = Path(parquet_output_path)
        self.similar_count = similar_count

    def read_parquet_compat(self, path: Path) -> pd.DataFrame:
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            logger.warning("pyarrow could not read %s (%s); retrying with fastparquet", path, exc)
            return pd.read_parquet(path, engine="fastparquet")

    def add_similar_nba_players(self, prospects: pd.DataFrame, career: pd.DataFrame) -> pd.DataFrame:
        QUALITY_SIGMA = 0.23

        # Filter the career pool to players with career_games >= 200
        nba = career[career["career_games"] >= 200].copy().reset_index(drop=True)

        nba_pct_cols = [f"{col}_career_pctile" for col in SIMILARITY_FEATURES]
        missing = [col for col in nba_pct_cols if col not in nba.columns]
        if missing:
            raise RuntimeError(f"Career feature table is missing required percentile columns: {missing}")

        # Parse prospect height/weight to numeric
        prospect_heights = prospects["height"].apply(_height_to_inches).to_numpy(dtype=float).copy()
        prospect_weights = prospects["weight"].apply(
            lambda w: _clean_weight(w, default=215.0)
        ).to_numpy(dtype=float).copy()

        # Fill missing prospect heights with class mean
        valid_p_h = prospect_heights[prospect_heights > 0]
        mean_p_h = float(valid_p_h.mean()) if len(valid_p_h) > 0 else 78.0
        prospect_heights[prospect_heights == 0] = mean_p_h

        prospect_weights[prospect_weights == 0] = (
            float(prospect_weights[prospect_weights > 0].mean()) if (prospect_weights > 0).any() else 215.0
        )

        # Parse NBA height/weight
        nba_heights = nba["height"].apply(_height_to_inches).to_numpy(dtype=float).copy()
        nba_weights = nba["weight"].apply(
            lambda w: _clean_weight(w, default=215.0)
        ).to_numpy(dtype=float).copy()
        valid_n_h = nba_heights[nba_heights > 0]
        mean_n_h = float(valid_n_h.mean()) if len(valid_n_h) > 0 else 78.0
        nba_heights[nba_heights == 0] = mean_n_h
        valid_n_w = nba_weights[nba_weights > 0]
        mean_n_w = float(valid_n_w.mean()) if len(valid_n_w) > 0 else 215.0
        nba_weights[nba_weights == 0] = mean_n_w

        # We use 8 stable features to prevent noise and role change distortion
        STABLE_FEATURES = [
            "reb_per36",
            "ast_per36",
            "blk_per36",
            "stl_per36",
            "fg3a_rate",
            "fta_rate",
        ]

        p_raw = prospects[STABLE_FEATURES].to_numpy(dtype=float)
        n_raw = nba[STABLE_FEATURES].to_numpy(dtype=float)

        # Clip prospect raw stats to 1st and 99th percentile of NBA pool
        n_1 = np.percentile(n_raw, 1, axis=0)
        n_99 = np.percentile(n_raw, 99, axis=0)
        p_raw_clipped = np.clip(p_raw, n_1, n_99)

        prospect_features = np.column_stack([p_raw_clipped, prospect_heights, prospect_weights])
        nba_features = np.column_stack([n_raw, nba_heights, nba_weights])

        # Use independent standard scaling to align draft cohorts to the NBA pool correctly
        p_scaled = StandardScaler().fit_transform(prospect_features)
        n_scaled = StandardScaler().fit_transform(nba_features)

        # Tuned playstyle-dominated weights with controlled overlap
        weights = np.array([
            1.7190,  # reb_per36
            1.3206,  # ast_per36
            1.3726,  # blk_per36
            0.8896,  # stl_per36
            3.1222,  # fg3a_rate
            1.3506,  # fta_rate
            1.0926,  # height_inches
            1.1863,  # weight_lbs
        ])
        p_weighted = p_scaled * weights
        n_weighted = n_scaled * weights

        # Compute Euclidean distance and map to display similarity score [0, 1]
        from sklearn.metrics.pairwise import euclidean_distances
        dists = euclidean_distances(p_weighted, n_weighted)
        playstyle_scores = np.exp(-dists / 50.0)

        pfv_keys = ["pts_per36", "reb_per36", "ast_per36", "blk_per36", "stl_per36", "ts_pct"]
        pfv_pct_arrays = {}
        for col in pfv_keys:
            if col in prospects.columns:
                pfv_pct_arrays[col] = prospects[col].to_numpy(dtype=float)
        adjusted_pfvs = []
        for i in range(len(prospects)):
            metrics = {}
            for col in pfv_keys:
                if col in pfv_pct_arrays:
                    val = float(pfv_pct_arrays[col][i])
                    pct = _percentile_of_score(pfv_pct_arrays[col], val)
                    metrics[col] = {"value": val, "percentile": pct}
            mpg_val = float(prospects.iloc[i]["mpg"])
            mpg_arr = prospects["mpg"].to_numpy(dtype=float)
            metrics["mpg"] = {"value": mpg_val, "percentile": _percentile_of_score(mpg_arr, mpg_val)}
            adjusted_pfvs.append(calculate_adjusted_pfv(metrics, is_prospect=True))

        prospect_height_buckets = [height_bucket(h) for h in prospects["height"]]
        prospect_quality = np.array(calculate_apfv_batch_by_height(adjusted_pfvs, prospect_height_buckets))

        nba_adjusted_pfvs = []
        for j in range(len(nba)):
            row_j = nba.iloc[j]
            metrics_j = {}
            for col in SIMILARITY_FEATURES:
                val = float(row_j.get(col, 0.0))
                pct = float(row_j.get(f"{col}_career_pctile", 0.0))
                metrics_j[col] = {"value": val, "percentile": pct}
            nba_adjusted_pfvs.append(calculate_adjusted_pfv(metrics_j, is_prospect=False))

        nba_height_buckets = [height_bucket(h) for h in nba["height"]]
        nba_quality = np.array(calculate_apfv_batch_by_height(nba_adjusted_pfvs, nba_height_buckets))

        era_buckets = [
            _era_bucket(str(nba.iloc[j].get("career_span", "")))
            for j in range(len(nba))
        ]

        import unicodedata
        import re
        def clean_name(n: str) -> str:
            normalized = unicodedata.normalize('NFKD', str(n))
            cleaned = "".join(c for c in normalized if not unicodedata.combining(c))
            cleaned = re.sub(r'[\.,]', '', cleaned)
            return " ".join(cleaned.lower().split())

        nba_names_clean = nba["player_name"].apply(clean_name)

        similar_payloads = []
        similar_names = []

        from tqdm import tqdm
        for i in tqdm(range(len(prospects)), desc="Calculating similar NBA players for prospects", unit="prospect"):
            # Quality affinity is disabled to prioritize true playstyle similarity
            composite = playstyle_scores[i]
            
            # Name match filter: exclude NBA counterpart with the exact same name (diacritic-insensitive)
            prospect_name_raw = str(prospects.iloc[i].get("Player") or prospects.iloc[i].get("player_name", ""))
            prospect_name_clean = clean_name(prospect_name_raw)
            name_mask = nba_names_clean == prospect_name_clean
            composite[name_mask] = -1.0

            ranked = np.argsort(composite)[::-1]
            top_pool = ranked[: self.similar_count * 5]
            selected = _select_era_diverse(top_pool, era_buckets, self.similar_count)

            matches = []
            names = []
            for idx in selected:
                player = nba.iloc[idx]
                names.append(str(player["player_name"]))
                matches.append(
                    {
                        "player_id": int(player["player_id"]),
                        "player_name": str(player["player_name"]),
                        "similarity_score": similarity_pct(float(composite[idx])),
                        "career_span": str(player.get("career_span", "")),
                        "position_group": str(player.get("position_group", "")),
                        "role": str(player.get("role", "")),
                    }
                )
            similar_payloads.append(matches)
            similar_names.append(names)

        out = prospects.copy()
        out["similar_nba_players"] = similar_payloads
        out["similar_nba_player_names"] = similar_names
        return out

    def add_prospect_roles(self, prospects: pd.DataFrame) -> pd.DataFrame:
        out = prospects.copy()
        profile_features = [
            "pts_per36",
            "reb_per36",
            "ast_per36",
            "blk_per36",
            "stl_per36",
            "ts_pct",
            "efg_pct",
            "fg3a_rate",
            "fta_rate",
        ]
        feat_df = out[profile_features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        means = feat_df.mean()
        stds = feat_df.std().replace(0, 1.0)
        scaled = (feat_df - means) / stds

        roles = []
        for idx, row in out.iterrows():
            pts = scaled.at[idx, "pts_per36"]
            ast = scaled.at[idx, "ast_per36"]
            reb = scaled.at[idx, "reb_per36"]
            stl = scaled.at[idx, "stl_per36"]
            blk = scaled.at[idx, "blk_per36"]
            efg = scaled.at[idx, "efg_pct"]
            fg3 = scaled.at[idx, "fg3a_rate"]
            fta = scaled.at[idx, "fta_rate"]

            is_big = (reb > 0.8 or blk > 0.8) and fg3 < -0.6

            if ast > 1.0 and not is_big:
                role = "Playmaker"
            elif pts > 0.8 and ast < 0.5:
                role = "Designated Scorer"
            elif ast > 0.3 and (pts > 0.1 or ast > 0.3):
                role = "Secondary Creator"
            elif fg3 > 0.8 and efg > -0.5:
                role = "Perimeter Specialist"
            elif fta > 0.5 and pts > 0.0:
                role = "Rim Attacker"
            elif (stl > 0.5 or blk > 0.5) and pts < -0.2:
                role = "Defensive Specialist"
            elif is_big and fg3 < -0.2:
                role = "Interior Presence"
            elif blk > 1.0 and reb > 0.5 and fg3 < 0.0:
                role = "Interior Presence"
            elif is_big:
                role = "Interior Presence"
            elif fg3 > 0.3:
                role = "Perimeter Specialist"
            elif ast > 0.0:
                role = "Secondary Creator"
            elif pts > 0.0:
                role = "Rim Attacker"
            else:
                role = "Defensive Specialist"

            roles.append(role)

        out["role"] = roles
        return out

    def write_outputs(self, prospects: pd.DataFrame) -> None:
        self.json_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.parquet_output_path.parent.mkdir(parents=True, exist_ok=True)

        pct_arrays = {
            col: prospects[col].to_numpy(dtype=float)
            for col in _PERCENTILE_COLS
            if col in prospects.columns
        }

        json_cols = [
            "prospect_id",
            "player_name",
            "team",
            "height",
            "weight",
            "role",
            "gp",
            "mpg",
            "pts_per36",
            "reb_per36",
            "ast_per36",
            "blk_per36",
            "stl_per36",
            "ts_pct",
            "efg_pct",
            "fg3a_rate",
            "fta_rate",
            "pick",
            "similar_nba_players",
        ]
        records = prospects[json_cols].to_dict(orient="records")
        raw_cols = [
            "GP",
            "MPG",
            "PPG",
            "FGM",
            "FGA",
            "FG%",
            "3PM",
            "3PA",
            "3P%",
            "FTM",
            "FTA",
            "FT%",
            "RPG",
            "APG",
            "SPG",
            "BPG",
        ]
        pfvs = []
        for record, raw in zip(records, prospects[raw_cols].to_dict(orient="records")):
            record["raw_stats"] = {RAW_STAT_JSON_KEYS[key]: value for key, value in raw.items()}
            for col in _PERCENTILE_COLS:
                if col not in record:
                    continue
                val = float(record[col])
                pct = _percentile_of_score(pct_arrays[col], val) if col in pct_arrays else 0.0
                record[col] = {"value": round(val, 4), "percentile": pct}

            pfv_metrics = {
                col: dict(record[col])
                for col in _PERCENTILE_COLS
                if col in record and isinstance(record[col], dict)
            }
            pfv_val = calculate_pfv(pfv_metrics)
            pfvs.append(pfv_val)

            record["pfv"] = pfv_val

        prospects["pfv"] = pfvs

        for record in records:
            record["raw_stats"]["pfv"] = record.pop("pfv", 0.0)

        self.json_output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

        parquet_cols = [
            "prospect_id",
            "player_name",
            "team",
            "height",
            "weight",
            "role",
            "GP",
            "MPG",
            "PPG",
            "FGM",
            "FGA",
            "FG%",
            "3PM",
            "3PA",
            "3P%",
            "FTM",
            "FTA",
            "FT%",
            "RPG",
            "APG",
            "SPG",
            "BPG",
            *SIMILARITY_FEATURES,
            "similar_nba_player_names",
            "pfv",
            "pick",
        ]
        available_cols = [col for col in parquet_cols if col in prospects.columns]
        prospects[available_cols].to_parquet(self.parquet_output_path, index=False, engine="fastparquet")

    def run(self, *, html_input_path: Path | str | None = None) -> pd.DataFrame:
        logger.info("Starting prospects pipeline...")
        if html_input_path is not None:
            html = Path(html_input_path).read_text(encoding="utf-8")
        else:
            logger.info("Fetching RealGM prospect stats from %s", self.source_url)
            html = fetch_html(self.source_url)

        prospects = scrape_prospect_stats(html)
        prospects = fetch_heights_weights(prospects)
        prospects = add_prospect_features(prospects)

        if not self.career_features_path.exists():
            raise FileNotFoundError(
                f"Career features parquet not found: {self.career_features_path}. "
                "Ensure player profiles pipeline has run first."
            )
        career = self.read_parquet_compat(self.career_features_path)

        prospects = self.add_similar_nba_players(prospects, career)
        prospects = self.add_prospect_roles(prospects)

        self.write_outputs(prospects)
        logger.info("Saved %s prospects datasets.", len(prospects))

        n_prospects = len(prospects)
        print(f"✓ Prospects completed: {n_prospects}/{n_prospects} prospects processed.")

        return prospects
