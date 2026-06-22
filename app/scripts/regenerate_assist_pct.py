"""Recompute `ast_pct` (assist percentage) and every artifact derived from it.

`ast_pct` used to be AST / own-FGM — an unbounded ratio that exceeded 100% for
pass-first guards (e.g. Trae Young). It is now the standard NBA assist percentage
(`features.assist_percentage`): the share of *teammates'* made field goals a player
assisted while on the floor, a 0-1 fraction clamped to <= 100%.

Because `ast_pct` feeds the within-season z/percentile columns, career aggregates,
the similarity embeddings, the radar `playstyle_metrics`, and the impact/versatility
PFV/APFV, this rebuilds the whole player-profile artifact set from local data
(no NBA API calls — bios are already present in the cached season tables):

  season_player_tables/*  ->  player_seasons.parquet  ->  features/* parquets
  ->  similarity embeddings/index  ->  player_profiles.json + player_metadata.json

Outputs are written to the pipeline storage root and copied into data/static (what
the API serves). Run from the `app/` directory:  python scripts/regenerate_assist_pct.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_APP_ROOT = Path(__file__).resolve().parents[1]
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

from analytics.player_profiles.archetypes import (  # noqa: E402
    add_archetypes,
    calculate_impact_apfv_batch,
    profile_payload,
)
from analytics.player_profiles.features import (  # noqa: E402
    CANONICAL_SEASON_PLAYER_COLUMNS,
    PlayerFilterConfig,
    add_career_percentiles,
    add_normalized_features,
    aggregate_career_features,
    apply_player_filter,
    assist_percentage,
    build_embedding_features,
)
from analytics.player_profiles.pipeline import copy_processed_outputs_to_static  # noqa: E402
from analytics.player_profiles.season_profiles import career_apfv_from_seasons  # noqa: E402
from analytics.player_profiles.similarity import (  # noqa: E402
    build_similarity_embeddings,
    build_similarity_index,
)
from analytics.player_profiles.storage import ProfileStorage, StorageConfig  # noqa: E402
from parquet_io import read_parquet, write_teams_parquet  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("regenerate_assist_pct")

STORAGE_ROOT = str(_APP_ROOT / "data" / "player_profiles")
STATIC_DIR = _APP_ROOT / "data" / "static"
SEASON_TABLE_GLOB = "processed/season_player_tables/season=*/players.parquet"
K_SIMILAR = 10


def _fix_cached_season_tables(storage_root: Path) -> None:
    """Recompute ast_pct in each cached per-season table so a future pipeline run
    (which reuses these when the schema is unchanged) keeps the corrected metric."""
    tables = sorted((storage_root).glob(SEASON_TABLE_GLOB))
    for path in tables:
        df = read_parquet(path)
        df["ast_pct"] = assist_percentage(df)
        df = df[CANONICAL_SEASON_PLAYER_COLUMNS]
        write_teams_parquet(df, path)
    logger.info("Recomputed ast_pct in %s cached season tables", len(tables))


def _build_profiles_json(
    career: pd.DataFrame,
    similar: dict[str, list[dict[str, Any]]],
    season_features: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Mirror of PlayerProfilePipeline._build_profiles_json (payload + apfv)."""
    profiles: dict[str, dict[str, Any]] = {}
    pids: list[str] = []
    pfvs: list[float] = []
    mpg_pctiles: list[float] = []
    career_minutes: list[float] = []
    for _, row in career.iterrows():
        pid = str(int(row["player_id"]))
        payload = profile_payload(row, similar.get(pid, []))
        profiles[pid] = payload
        pids.append(pid)
        pfvs.append(float(payload["pfv"]))
        mpg_pctiles.append(float(row.get("mpg_global_pctile", 0.0) or 0.0))
        career_minutes.append(float(row.get("career_minutes", 0.0) or 0.0))

    career_apfvs = career_apfv_from_seasons(season_features)
    if pfvs:
        fallback = dict(zip(pids, calculate_impact_apfv_batch(pfvs, mpg_pctiles, career_minutes)))
        for pid in pids:
            profiles[pid]["apfv"] = career_apfvs.get(pid, fallback[pid])
    return profiles


def _build_metadata(career: pd.DataFrame) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    for _, row in career.iterrows():
        pid = str(int(row["player_id"]))
        meta[pid] = {
            "player_id": int(row["player_id"]),
            "player_name": str(row["player_name"]),
            "role": str(row.get("role", "")),
            "career_span": str(row.get("career_span", "")),
            "archetypes": list(row.get("archetypes", [])),
        }
    return meta


def main() -> None:
    storage = ProfileStorage(StorageConfig.from_uri(STORAGE_ROOT))
    root = Path(STORAGE_ROOT)

    # 1. Fix cached per-season tables, then rebuild the concatenated table from them.
    _fix_cached_season_tables(root)
    tables = [read_parquet(p) for p in sorted(root.glob(SEASON_TABLE_GLOB))]
    all_seasons = pd.concat(tables, ignore_index=True, sort=False)[CANONICAL_SEASON_PLAYER_COLUMNS]
    storage.write_parquet("processed/player_seasons.parquet", all_seasons)
    over_100 = int((all_seasons["ast_pct"] > 1.0).sum())
    logger.info(
        "player_seasons rebuilt: %s rows, ast_pct max=%.3f, rows>100%%=%s",
        len(all_seasons), all_seasons["ast_pct"].max(), over_100,
    )

    # 2. Within-season normalized features (z / percentile / pos_z).
    normalized = add_normalized_features(all_seasons)
    storage.write_parquet("features/player_season_features.parquet", normalized)

    # 3. Career aggregates + percentiles, filtered population, archetypes.
    current_season_start = int(max(all_seasons["SEASON_START"]))
    career = aggregate_career_features(normalized)
    career = add_career_percentiles(career)
    career = apply_player_filter(
        career,
        current_season_start=current_season_start,
        config=PlayerFilterConfig(),
    )
    career = add_archetypes(career)
    storage.write_parquet("features/player_career_features.parquet", career)

    # 4. Similarity embeddings + index (ast_pct_z is one of the embedding features).
    embedding_features, _cols = build_embedding_features(career)
    storage.write_parquet("features/player_similarity_features.parquet", embedding_features)
    embeddings, _pca = build_similarity_embeddings(career, embedding_features)
    storage.write_parquet("embeddings/player_similarity_embeddings.parquet", embeddings)
    similar = build_similarity_index(career, embeddings, embedding_features, k=K_SIMILAR)
    storage.write_json("embeddings/player_similarity_index.json", similar)

    # 5. Profiles + metadata JSON (radar metrics, pfv, apfv, similar_players).
    profiles = _build_profiles_json(career, similar, normalized)
    storage.write_json("processed/player_profiles.json", profiles)
    metadata = _build_metadata(career)
    storage.write_json("processed/player_metadata.json", metadata)

    # 6. Copy served artifacts into data/static, preserving each file's on-disk
    #    format: player_profiles.json is pretty-printed (regenerate_player_{radar,pfv}.py
    #    convention), player_metadata.json is compact.
    copy_processed_outputs_to_static(STORAGE_ROOT, STATIC_DIR)
    profiles_path = STATIC_DIR / "player_profiles.json"
    profiles_path.write_text(json.dumps(json.loads(profiles_path.read_text()), indent=2))
    metadata_path = STATIC_DIR / "player_metadata.json"
    metadata_path.write_text(json.dumps(json.loads(metadata_path.read_text())))
    logger.info("Regenerated %s profiles and copied artifacts to %s", len(profiles), STATIC_DIR)


if __name__ == "__main__":
    main()
