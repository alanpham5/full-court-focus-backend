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
    usage: dict[int, int] | None = None,
    max_usage: int | None = None,
) -> list[int]:
    """Pick `count` comps from a ranked candidate list.

    Constraints: at most `max_per_era` per era bucket, and (if `usage` and
    `max_usage` are given) at most `max_usage` appearances of any one NBA
    player across the whole run — prospects processed earlier in the run get
    first claim, later ones fall through to the next-best distinct comp.
    Era constraint is relaxed before the usage cap if we run short.
    """
    def _capped(idx: int) -> bool:
        return (
            usage is not None
            and max_usage is not None
            and usage.get(idx, 0) >= max_usage
        )

    selected: list[int] = []
    era_counts: dict[str, int] = {}
    for idx in ranked_indices:
        if len(selected) >= count:
            break
        idx = int(idx)
        if _capped(idx):
            continue
        era = era_buckets[idx]
        if era_counts.get(era, 0) < max_per_era:
            selected.append(idx)
            era_counts[era] = era_counts.get(era, 0) + 1

    if len(selected) < count:
        for idx in ranked_indices:
            if len(selected) >= count:
                break
            idx = int(idx)
            if idx not in selected and not _capped(idx):
                selected.append(idx)

    if usage is not None:
        for idx in selected:
            usage[idx] = usage.get(idx, 0) + 1
    return selected


# ---- prospect -> NBA similarity config (tuned; see scratch/tune_points.py)
SIMILARITY_COMP_FEATURES = [
    "pts_per36",
    "reb_per36",
    "ast_per36",
    "blk_per36",
    "stl_per36",
    "fg3a_rate",
    "fta_rate",
    "ts_pct",
    "ast_pct",
    "mpg",
    "stocks",        # stl_per36 + blk_per36
    "scoring_load",  # pts_per36 * (1 - ast_pct)
]
SIMILARITY_COMP_WEIGHTS = np.array([
    0.0,    # pts_per36
    0.34,   # reb_per36
    0.05,   # ast_per36
    0.4,    # blk_per36
    0.3,    # stl_per36
    0.29,   # fg3a_rate
    0.31,   # fta_rate
    0.0,    # ts_pct
    0.35,   # ast_pct
    0.1,    # mpg
    0.13,   # stocks
    0.15,   # scoring_load
    0.294,  # height_inches (h/w capped at 25% of squared-weight budget)
    0.392,  # weight_lbs    (h/w capped at 25% of squared-weight budget)
])
SIMILARITY_COMP_BANDWIDTH = 0.08
SIMILARITY_COMP_SMOOTH_LAMBDA = 0.8  # (1-l)*own score + l*best neighbor score
SIMILARITY_COMP_SMOOTH_TOPK = 7  # smooth over each player's top-7 NBA-NBA similars
# Display gamma for prospect comps: composite scores live well below 1.0
# (euclidean-exp scale, not cosine), so the shared 5.5 gamma crushes them.
# 0.2 maps the stored-comp distribution to ~55-79 (median ~69).
PROSPECT_SIMILARITY_GAMMA = 0.2
# Anti-spam: candidates are RANKED by score * (1 - penalty * their mean
# score across every prospect in the run), demoting "universal attractor"
# players who look similar to everyone; displayed scores stay raw. The
# penalty is multiplicative so a dissimilar player can never outrank a
# similar one. Additionally no NBA player may appear as a comp more than
# COMP_MAX_APPEARANCES times in one run (one class / one current board).
COMP_POPULARITY_PENALTY = 1.0
COMP_MAX_APPEARANCES = 2
# Career-establishment prior (selection-stage only; the similarity matrix
# and the points metric are untouched): comp RANKING is multiplied by
# (floor + (1-floor) * establishment)^alpha, where establishment is the
# NBA player's mean percentile of career minutes and career APFV quality.
# Among similarly-shaped candidates this prefers the player with the more
# substantial career, fixing the systematic pessimism where strong
# prospects drew bench-player comps. Data-driven, NBA-side only — no
# draft position and no post-draft prospect information.
COMP_ESTABLISHMENT_FLOOR = 0.35
COMP_ESTABLISHMENT_ALPHA = 1.0


def build_nba_neighbor_index(nba: pd.DataFrame, top_k: int | None = None) -> np.ndarray | None:
    """Padded NBA-NBA neighbor index from player_profiles.json.

    Row j lists the columns of player j's `similar_players` (first `top_k`
    if given), padded with j itself (a neutral self-reference under
    max-aggregation). Returns None if player_profiles.json is missing
    (first-ever run) or has no usable edges.
    """
    from config import PLAYER_PROFILES_PATH
    try:
        profiles = json.loads(Path(PLAYER_PROFILES_PATH).read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning(
            "player_profiles.json not found; skipping NBA neighborhood smoothing."
        )
        return None
    col_by_id = {int(pid): j for j, pid in enumerate(nba["player_id"].astype(int))}
    neighbor_idx = []
    for pid in nba["player_id"].astype(int):
        sims = profiles.get(str(int(pid)), {}).get("similar_players", [])
        if top_k is not None:
            sims = sims[:top_k]
        cols = [
            col_by_id[int(s["player_id"])]
            for s in sims
            if int(s["player_id"]) in col_by_id
        ]
        neighbor_idx.append(cols)
    max_deg = max((len(c) for c in neighbor_idx), default=0)
    if max_deg == 0:
        return None
    return np.array(
        [cols + [j] * (max_deg - len(cols)) for j, cols in enumerate(neighbor_idx)],
        dtype=np.int64,
    )


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
        """Prospect -> NBA similarity in PEER-PERCENTILE SPACE.

        Predictive design:
        - Each NBA player is described by their career percentile (within the
          NBA pool) on each playstyle feature. Each prospect is described by
          their class percentile (within their draft class). Both live in
          [0, 1] and answer the same question: "where does this player rank
          relative to their peers on this dimension?" — the unit that
          transfers best across the college -> NBA gap.
        - Height & weight enter as a softly weighted size prior: they cannot
          dominate (combined squared weight is ~20 % of the budget) so the
          algorithm cannot collapse to a trivial "same person has identical
          h/w" identity match.
        - Quality affinity (APFV) further gates matches so a marginal prospect
          isn't paired with a star and vice versa.

        Tuned on the historical 2007–2023 prospect cohorts (n=490 with NBA
        counterparts in the pool) against a points objective evaluated on the
        raw top-7 before name filtering: +2 if the counterpart is in the
        top-7, +1 per top-7 player whose own top-7 NBA-NBA similars contain
        the counterpart. The tuner (scratch/tune_points.py) calls
        compute_similarity_matrix directly so the eval space and production
        space cannot drift. No post-draft information or draft position is
        used, and height/weight carry a capped share of the weight budget.
        """
        # ---- pool selection
        nba = career[career["career_games"] >= 200].copy().reset_index(drop=True)
        playstyle_scores = self.compute_similarity_matrix(prospects, nba)

        return self._build_similarity_payload(prospects, nba, playstyle_scores)

    def compute_similarity_matrix(
        self,
        prospects: pd.DataFrame,
        nba: pd.DataFrame,
        *,
        features: list[str] | None = None,
        weights: np.ndarray | None = None,
        bandwidth: float | None = None,
        smooth_lambda: float | None = None,
        neighbor_idx: np.ndarray | None = None,
    ) -> np.ndarray:
        """Composite similarity matrix (n_prospects, n_nba) for one class.

        `nba` must already be the filtered comp pool (career_games >= 200).
        Keyword overrides exist so scratch/tune_points.py can tune through
        this exact code path; production uses the module-level constants.
        """
        FEATURES = features if features is not None else SIMILARITY_COMP_FEATURES
        WEIGHTS = weights if weights is not None else SIMILARITY_COMP_WEIGHTS
        BANDWIDTH = bandwidth if bandwidth is not None else SIMILARITY_COMP_BANDWIDTH
        SMOOTH_LAMBDA = (
            smooth_lambda if smooth_lambda is not None else SIMILARITY_COMP_SMOOTH_LAMBDA
        )

        # ast_pct may be missing on older prospect parquet rows; compute proxy.
        if "ast_pct" not in prospects.columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                prospects = prospects.copy()
                prospects["ast_pct"] = (
                    prospects["APG"] / prospects["FGM"].replace(0, np.nan)
                ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Engineered archetype features (computed identically on both pools
        # from pre-draft / per-36 stats only).
        prospects = prospects.copy()
        for df in (prospects, nba):
            if "stocks" not in df.columns:
                df["stocks"] = df["stl_per36"] + df["blk_per36"]
            if "scoring_load" not in df.columns:
                df["scoring_load"] = df["pts_per36"] * (
                    1.0 - df["ast_pct"].clip(0.0, 1.0)
                )
            if "playmake_load" not in df.columns:
                df["playmake_load"] = (
                    df["ast_per36"] + 0.5 * df["ast_pct"].clip(0.0, 1.0)
                )
            if "perimeter_skill" not in df.columns:
                df["perimeter_skill"] = df["fg3a_rate"] * df["ts_pct"]
            if "interior_load" not in df.columns:
                df["interior_load"] = df["reb_per36"] + 2.0 * df["blk_per36"]

        # Quality (APFV) as an optional similarity dimension: prospect side
        # is computed from pre-draft stats only; NBA side from career stats.
        # Both become within-pool percentiles downstream, so a top-quality
        # prospect matches players who had top-quality careers — a
        # performance prior with no draft-position input.
        if "quality" in FEATURES:
            if "quality" not in prospects.columns:
                prospects["quality"] = self.prospect_quality_scores(prospects)
            if "quality" not in nba.columns:
                nba["quality"] = self.nba_quality_scores(nba)

        # ---- height & weight (NBA & prospect), filled with pool means
        prospect_heights = prospects["height"].apply(_height_to_inches).to_numpy(dtype=float).copy()
        prospect_weights = prospects["weight"].apply(
            lambda w: _clean_weight(w, default=215.0)
        ).to_numpy(dtype=float).copy()
        valid_p_h = prospect_heights[prospect_heights > 0]
        prospect_heights[prospect_heights == 0] = float(valid_p_h.mean()) if len(valid_p_h) > 0 else 78.0
        prospect_weights[prospect_weights == 0] = (
            float(prospect_weights[prospect_weights > 0].mean()) if (prospect_weights > 0).any() else 215.0
        )

        nba_heights = nba["height"].apply(_height_to_inches).to_numpy(dtype=float).copy()
        nba_weights = nba["weight"].apply(
            lambda w: _clean_weight(w, default=215.0)
        ).to_numpy(dtype=float).copy()
        valid_n_h = nba_heights[nba_heights > 0]
        nba_heights[nba_heights == 0] = float(valid_n_h.mean()) if len(valid_n_h) > 0 else 78.0
        valid_n_w = nba_weights[nba_weights > 0]
        nba_weights[nba_weights == 0] = float(valid_n_w.mean()) if len(valid_n_w) > 0 else 215.0

        # ---- raw playstyle matrices (for prospect-class percentile ranking)
        p_raw = prospects[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        n_raw = nba[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)

        # ---- NBA: percentile within NBA career pool (precomputed when possible)
        n_feat = np.empty_like(n_raw, dtype=float)
        for j, col in enumerate(FEATURES):
            pct_col = f"{col}_career_pctile"
            if pct_col in nba.columns:
                n_feat[:, j] = nba[pct_col].to_numpy(dtype=float) / 100.0
            else:
                ref = np.sort(n_raw[:, j])
                n_feat[:, j] = np.searchsorted(ref, n_raw[:, j], side="right") / max(len(ref), 1)

        # ---- prospect: percentile within ALL current prospects (class-relative).
        # If multiple draft classes are mixed (historical mode), the caller
        # passes them as separate runs — so we treat this dataframe as one class.
        p_feat = np.empty_like(p_raw, dtype=float)
        for j in range(p_raw.shape[1]):
            ref = np.sort(p_raw[:, j])
            p_feat[:, j] = np.searchsorted(ref, p_raw[:, j], side="right") / max(len(ref), 1)

        # ---- height/weight in shared NBA-anchored standardization
        all_h = np.concatenate([prospect_heights, nba_heights])
        all_w = np.concatenate([prospect_weights, nba_weights])
        h_mu, h_sd = float(all_h.mean()), max(float(all_h.std()), 1e-6)
        w_mu, w_sd = float(all_w.mean()), max(float(all_w.std()), 1e-6)
        p_h_n = (prospect_heights - h_mu) / h_sd
        p_w_n = (prospect_weights - w_mu) / w_sd
        n_h_n = (nba_heights - h_mu) / h_sd
        n_w_n = (nba_weights - w_mu) / w_sd

        p_mat = np.column_stack([p_feat, p_h_n, p_w_n]) * WEIGHTS
        n_mat = np.column_stack([n_feat, n_h_n, n_w_n]) * WEIGHTS

        # ---- weighted Euclidean -> exp(-d / bandwidth) playstyle score
        diff_sq = (
            (p_mat ** 2).sum(axis=1)[:, None]
            + (n_mat ** 2).sum(axis=1)[None, :]
            - 2.0 * p_mat @ n_mat.T
        )
        diff_sq = np.maximum(diff_sq, 0.0)
        dists = np.sqrt(diff_sq)
        playstyle_scores = np.exp(-dists / BANDWIDTH)

        # ---- neighborhood smoothing over the NBA-NBA similarity graph
        # score'(p, j) = (1-l)*score(p, j) + l*max_{k in similar_players(j)} score(p, k)
        # Comps whose neighborhoods resemble the prospect rank higher. NBA-side
        # information only — no post-draft prospect information enters.
        if SMOOTH_LAMBDA > 0.0:
            if neighbor_idx is None:
                neighbor_idx = build_nba_neighbor_index(
                    nba, top_k=SIMILARITY_COMP_SMOOTH_TOPK
                )
            if neighbor_idx is not None:
                neighbor_best = playstyle_scores[:, neighbor_idx].max(axis=2)
                playstyle_scores = (
                    (1.0 - SMOOTH_LAMBDA) * playstyle_scores
                    + SMOOTH_LAMBDA * neighbor_best
                )

        return playstyle_scores

    def prospect_quality_scores(self, prospects: pd.DataFrame) -> np.ndarray:
        """Stats-only prospect quality (APFV, height-bucket-normalized).

        Pre-draft information only: per-36 production, efficiency, minutes.
        No draft position, no post-draft data.
        """
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
            gp_val = float(prospects.iloc[i].get("gp", prospects.iloc[i].get("GP", 0)) or 0)
            metrics["gp"] = {"value": gp_val, "percentile": 0.0}
            metrics["team"] = str(prospects.iloc[i].get("team", "") or prospects.iloc[i].get("Team", "") or "")
            adjusted_pfvs.append(calculate_adjusted_pfv(metrics, is_prospect=True))

        prospect_height_buckets = [height_bucket(h) for h in prospects["height"]]
        return np.array(calculate_apfv_batch_by_height(adjusted_pfvs, prospect_height_buckets))

    def nba_quality_scores(self, nba: pd.DataFrame) -> np.ndarray:
        """NBA career quality (APFV, height-bucket-normalized)."""
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
        return np.array(calculate_apfv_batch_by_height(nba_adjusted_pfvs, nba_height_buckets))

    def _build_similarity_payload(
        self,
        prospects: pd.DataFrame,
        nba: pd.DataFrame,
        playstyle_scores: np.ndarray,
    ) -> pd.DataFrame:
        prospect_quality = self.prospect_quality_scores(prospects)
        nba_quality = self.nba_quality_scores(nba)

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

        # Anti-spam ranking: demote "universal attractor" players whose score
        # is high for everyone in the run. Ranking uses the penalized score;
        # the displayed similarity stays the raw composite.
        popularity = playstyle_scores.mean(axis=0)

        # Career-establishment prior: among similarly-shaped candidates,
        # prefer the one with the more substantial NBA career.
        career_minutes = pd.to_numeric(
            nba["career_minutes"], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=float)
        ref_m = np.sort(career_minutes)
        mins_pct = np.searchsorted(ref_m, career_minutes, side="right") / max(len(ref_m), 1)
        ref_q = np.sort(nba_quality)
        qual_pct = np.searchsorted(ref_q, nba_quality, side="right") / max(len(ref_q), 1)
        establishment = 0.5 * mins_pct + 0.5 * qual_pct
        estab_prior = (
            COMP_ESTABLISHMENT_FLOOR
            + (1.0 - COMP_ESTABLISHMENT_FLOOR) * establishment
        ) ** COMP_ESTABLISHMENT_ALPHA

        rank_scores = (
            playstyle_scores
            * (1.0 - COMP_POPULARITY_PENALTY * popularity[None, :])
            * estab_prior[None, :]
        )
        comp_usage: dict[int, int] = {}

        from tqdm import tqdm
        for i in tqdm(range(len(prospects)), desc="Calculating similar NBA players for prospects", unit="prospect"):
            # Composite = playstyle similarity in peer-percentile space.
            # Caliber is already implicitly captured by the percentile features
            # (a top-decile rebounder/shooter in their pool matches a top-decile
            # NBA rebounder/shooter). A separate APFV gate was tested and
            # consistently hurt predictive accuracy because the same person's
            # prospect-APFV and NBA-APFV often diverge.
            composite = playstyle_scores[i].copy()
            ranking = rank_scores[i].copy()

            # Name match filter: exclude NBA counterpart with the exact same name (diacritic-insensitive)
            prospect_name_raw = str(prospects.iloc[i].get("Player") or prospects.iloc[i].get("player_name", ""))
            prospect_name_clean = clean_name(prospect_name_raw)
            name_mask = nba_names_clean == prospect_name_clean
            composite[name_mask] = -1.0
            ranking[name_mask] = -np.inf

            ranked = np.argsort(ranking)[::-1]
            top_pool = ranked[: self.similar_count * 10]
            selected = _select_era_diverse(
                top_pool, era_buckets, self.similar_count,
                usage=comp_usage, max_usage=COMP_MAX_APPEARANCES,
            )

            matches = []
            names = []
            for idx in selected:
                player = nba.iloc[idx]
                names.append(str(player["player_name"]))
                matches.append(
                    {
                        "player_id": int(player["player_id"]),
                        "player_name": str(player["player_name"]),
                        "similarity_score": similarity_pct(
                            float(composite[idx]), gamma=PROSPECT_SIMILARITY_GAMMA
                        ),
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

    def _collect_global_metric_values(self, prospects_df: pd.DataFrame) -> dict[str, np.ndarray]:
        from config import DATA_STATIC_DIR
        draft_dir = DATA_STATIC_DIR / "draft"
        
        global_vals = {
            col: prospects_df[col].tolist()
            for col in _PERCENTILE_COLS
            if col in prospects_df.columns
        }
        
        if draft_dir.exists():
            for path in sorted(draft_dir.glob("prospects_*.json")):
                try:
                    with path.open(encoding="utf-8") as f:
                        records = json.load(f)
                    for record in records:
                        for col in _PERCENTILE_COLS:
                            val = None
                            if col in record:
                                metric = record[col]
                                if isinstance(metric, dict):
                                    val = metric.get("value")
                                else:
                                    val = metric
                            if val is None:
                                val = record.get("raw_stats", {}).get(RAW_STAT_JSON_KEYS.get(col, col))
                            if val is not None:
                                try:
                                    global_vals.setdefault(col, []).append(float(val))
                                except (TypeError, ValueError):
                                    pass
                except Exception as e:
                    logger.warning("Failed to load historical prospects for global metrics from %s: %s", path.name, e)
                    
        return {col: np.array(vals, dtype=float) for col, vals in global_vals.items()}

    def _collect_global_adjusted_pfvs(
        self,
        current_adjusted_pfvs: list[float],
        current_height_buckets: list[str],
    ) -> tuple[list[float], list[str], slice]:
        from config import DATA_STATIC_DIR
        draft_dir = DATA_STATIC_DIR / "draft"
        
        global_adjusted_pfvs = list(current_adjusted_pfvs)
        global_height_buckets = list(current_height_buckets)
        current_slice = slice(0, len(current_adjusted_pfvs))
        
        if draft_dir.exists():
            for path in sorted(draft_dir.glob("prospects_*.json")):
                try:
                    with path.open(encoding="utf-8") as f:
                        records = json.load(f)
                    for record in records:
                        pfv_metrics = {
                            col: dict(record[col])
                            for col in _PERCENTILE_COLS
                            if col in record and isinstance(record[col], dict)
                        }
                        if pfv_metrics:
                            gp_val = float(record.get("gp", record.get("raw_stats", {}).get("gp", 0)) or 0)
                            pfv_metrics["gp"] = {"value": gp_val, "percentile": 0.0}
                            pfv_metrics["team"] = str(record.get("team", "") or "")
                            adj_pfv = calculate_adjusted_pfv(pfv_metrics, is_prospect=True)
                            global_adjusted_pfvs.append(adj_pfv)
                            global_height_buckets.append(height_bucket(record.get("height", "")))
                except Exception as e:
                    logger.warning("Failed to load historical prospects for APFV from %s: %s", path.name, e)
                    
        return global_adjusted_pfvs, global_height_buckets, current_slice

    def write_outputs(self, prospects: pd.DataFrame) -> None:
        self.json_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.parquet_output_path.parent.mkdir(parents=True, exist_ok=True)

        pct_arrays = self._collect_global_metric_values(prospects)

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
        current_adjusted_pfvs = []
        current_height_buckets = []
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
            gp_val = float(record.get("gp", record["raw_stats"].get("gp", 0)) or 0)
            pfv_metrics["gp"] = {"value": gp_val, "percentile": 0.0}
            pfv_metrics["team"] = str(record.get("team", "") or "")
            current_adjusted_pfvs.append(calculate_adjusted_pfv(pfv_metrics, is_prospect=True))
            current_height_buckets.append(height_bucket(record.get("height", "")))

        prospects["pfv"] = pfvs

        global_adjusted_pfvs, global_height_buckets, current_slice = self._collect_global_adjusted_pfvs(
            current_adjusted_pfvs, current_height_buckets
        )
        # Prospect APFV: steeper curve + raw-magnitude anchor so a player
        # cannot reach 0.99 just by being best-of-a-weak-bucket. The anchor
        # 0.55 sits near the top observed adjusted_pfv for star prospects
        # (Zion-class) after sample-size/efficiency/competition dampers.
        global_apfvs = calculate_apfv_batch_by_height(
            global_adjusted_pfvs, global_height_buckets,
            curve_exponent=2.2, raw_anchor=0.50,
        )
        current_apfvs = global_apfvs[current_slice]

        prospects["apfv"] = current_apfvs

        for record, apfv_val in zip(records, current_apfvs):
            record["apfv"] = apfv_val

        for record in records:
            record["raw_stats"]["pfv"] = record.pop("pfv", 0.0)
            record["raw_stats"]["apfv"] = record.pop("apfv", 0.0)

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
            "apfv",
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
