from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
import json
import logging
import re
import sys
import threading
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
_APP_ROOT = Path(__file__).resolve().parents[1]
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

from analytics.similarity_scoring import cross_pool_cosine_similarity, similarity_pct

DEFAULT_SOURCE_URL = "https://basketball.realgm.com/nba/draft/prospects/stats"
DEFAULT_STATIC_DIR = _APP_ROOT / "data" / "static"
DEFAULT_CAREER_FEATURES = DEFAULT_STATIC_DIR / "player_career_features.parquet"
DEFAULT_JSON_OUTPUT = DEFAULT_STATIC_DIR / "prospects.json"
DEFAULT_PARQUET_OUTPUT = DEFAULT_STATIC_DIR / "prospects.parquet"

logger = logging.getLogger(__name__)

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
    "ast_per36",
    "reb_per36",
    "stl_per36",
    "blk_per36",
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

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape RealGM NBA draft prospect stats, convert available counting stats "
            "to per-36 features, and find four similar NBA career counterparts."
        )
    )
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument(
        "--html-input",
        type=Path,
        default=None,
        help="Optional saved RealGM HTML file. Useful when RealGM blocks non-browser clients.",
    )
    parser.add_argument("--career-features", type=Path, default=DEFAULT_CAREER_FEATURES)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--parquet-output", type=Path, default=DEFAULT_PARQUET_OUTPUT)
    parser.add_argument("--similar-count", type=int, default=4)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    html = args.html_input.read_text(encoding="utf-8") if args.html_input else fetch_html(args.source_url)
    prospects = scrape_prospect_stats(html)
    prospects = fetch_heights_weights(prospects)
    prospects = add_prospect_features(prospects)

    career = read_parquet_compat(args.career_features)
    prospects = add_similar_nba_players(
        prospects,
        career,
        similar_count=max(1, int(args.similar_count)),
    )
    prospects = add_prospect_roles(prospects)

    write_outputs(prospects, args.json_output, args.parquet_output)
    logger.info(
        "Wrote %s prospects to %s and %s",
        len(prospects),
        args.json_output,
        args.parquet_output,
    )


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
    """Resolve once whether HTTPS fetches should use a subprocess Python."""
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

    unique_links = {link for link in out["profile_link"] if link}

    logger.info("Fetching profiles for %d prospects to parse height and weight...", len(unique_links))

    def fetch_one(link: str) -> tuple[str, str, str]:
        url = f"https://basketball.realgm.com{link}"
        try:
            html = fetch_html(url)
            h_match = re.search(r'<strong>Height:</strong>\s*([0-9]+-[0-9]+)', html, re.IGNORECASE)
            w_match = re.search(r'<strong>Weight:</strong>\s*([0-9]+)', html, re.IGNORECASE)
            h = h_match.group(1).strip() if h_match else ""
            w = w_match.group(1).strip() if w_match else ""
            return link, h, w
        except Exception as e:
            logger.warning("Error fetching/parsing profile %s: %s", url, e)
            return link, "", ""

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_one, link): link for link in unique_links}
        for future in as_completed(futures):
            link, h, w = future.result()
            heights[link] = h
            weights[link] = w

    out["height"] = out["profile_link"].map(heights).fillna("")
    out["weight"] = out["profile_link"].map(weights).fillna("")
    return out


def add_prospect_features(prospects: pd.DataFrame) -> pd.DataFrame:
    out = prospects.copy()
    for col in PROSPECT_TABLE_REQUIRED_COLUMNS - {"Player", "Team"}:
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

    return out.replace([np.inf, -np.inf], np.nan).fillna(
        {
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
    )


def add_similar_nba_players(
    prospects: pd.DataFrame,
    career: pd.DataFrame,
    *,
    similar_count: int,
) -> pd.DataFrame:
    missing = [col for col in SIMILARITY_FEATURES if col not in career.columns]
    if missing:
        raise RuntimeError(f"Career feature table is missing required columns: {missing}")

    nba = career.copy()
    nba_features = nba[SIMILARITY_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    prospect_features = prospects[SIMILARITY_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    scores = cross_pool_cosine_similarity(
        prospect_features.to_numpy(dtype=float),
        nba_features.to_numpy(dtype=float),
    )

    similar_payloads: list[list[dict[str, Any]]] = []
    similar_names: list[list[str]] = []
    for row_scores in scores:
        candidate_indices = np.argsort(row_scores)[::-1][:similar_count]
        matches = []
        names = []
        for idx in candidate_indices:
            player = nba.iloc[int(idx)]
            names.append(str(player["player_name"]))
            matches.append(
                {
                    "player_id": int(player["player_id"]),
                    "player_name": str(player["player_name"]),
                    "similarity_score": similarity_pct(float(row_scores[int(idx)])),
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


def add_prospect_roles(prospects: pd.DataFrame) -> pd.DataFrame:
    out = prospects.copy()
    profile_features = [
        "pts_per36",
        "ast_per36",
        "reb_per36",
        "stl_per36",
        "blk_per36",
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
        sim_players = row.get("similar_nba_players", [])
        pos_group = "W"
        if isinstance(sim_players, list) and len(sim_players) > 0:
            pos_group = sim_players[0].get("position_group", "W")

        pts = scaled.at[idx, "pts_per36"]
        ast = scaled.at[idx, "ast_per36"]
        reb = scaled.at[idx, "reb_per36"]
        stl = scaled.at[idx, "stl_per36"]
        blk = scaled.at[idx, "blk_per36"]
        ts = scaled.at[idx, "ts_pct"]
        efg = scaled.at[idx, "efg_pct"]
        fg3 = scaled.at[idx, "fg3a_rate"]
        fta = scaled.at[idx, "fta_rate"]

        if ast > 1.0 and pos_group in ("G", "W"):
            role = "Playmaker"
        elif pos_group == "B" and fg3 < -0.2:
            role = "Interior Presence"
        elif blk > 1.0 and reb > 0.5 and fg3 < 0.0:
            role = "Interior Presence"
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
        else:
            if pos_group == "B":
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


_PERCENTILE_COLS = [
    "mpg",
    "pts_per36",
    "ast_per36",
    "reb_per36",
    "stl_per36",
    "blk_per36",
    "ts_pct",
    "efg_pct",
    "fg3a_rate",
    "fta_rate",
]


def _percentile_of_score(arr: np.ndarray, score: float) -> float:
    """Return the percentile (0-100) of *score* within *arr* (higher is better).

    Uses the same 'rank / n * 100' approach as scipy percentileofscore with
    kind='rank' so that the top value in the class earns 100 and the bottom
    earns a value > 0 (not 0).
    """
    n = len(arr)
    if n == 0:
        return 0.0
    rank = float(np.sum(arr <= score))
    return round(rank / n * 100.0, 1)


def write_outputs(prospects: pd.DataFrame, json_output: Path, parquet_output: Path) -> None:
    json_output.parent.mkdir(parents=True, exist_ok=True)
    parquet_output.parent.mkdir(parents=True, exist_ok=True)

    pct_arrays: dict[str, np.ndarray] = {
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
        "ast_per36",
        "reb_per36",
        "stl_per36",
        "blk_per36",
        "ts_pct",
        "efg_pct",
        "fg3a_rate",
        "fta_rate",
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
    for record, raw in zip(records, prospects[raw_cols].to_dict(orient="records")):
        record["raw_stats"] = {RAW_STAT_JSON_KEYS[key]: value for key, value in raw.items()}
        for col in _PERCENTILE_COLS:
            if col not in record:
                continue
            val = float(record[col])
            pct = _percentile_of_score(pct_arrays[col], val) if col in pct_arrays else 0.0
            record[col] = {"value": round(val, 4), "percentile": pct}
    json_output.write_text(json.dumps(records, indent=2), encoding="utf-8")

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
    ]
    available_cols = [col for col in parquet_cols if col in prospects.columns]
    prospects[available_cols].to_parquet(parquet_output, index=False, engine="fastparquet")


def read_parquet_compat(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        logger.warning("pyarrow could not read %s (%s); retrying with fastparquet", path, exc)
        return pd.read_parquet(path, engine="fastparquet")


def per36(values: pd.Series, mpg: pd.Series) -> pd.Series:
    return (values / mpg * 36.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator / denominator.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def slugify_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")
    return slug or "unknown"


if __name__ == "__main__":
    main()
