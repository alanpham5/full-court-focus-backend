from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from analytics.player_profiles.archetypes import (
    add_archetypes,
    calculate_adjusted_pfv,
    calculate_apfv_batch_by_height,
    height_bucket,
    profile_payload,
    style_summary,
)
from analytics.player_profiles.features import (
    CANONICAL_SEASON_PLAYER_COLUMNS,
    PlayerFilterConfig,
    add_career_percentiles,
    add_normalized_features,
    aggregate_career_features,
    apply_player_filter,
    build_embedding_features,
    build_season_player_table,
    normalize_traditional_totals,
)
from analytics.player_profiles.nba_client import EndpointResult, NbaStatsClient
from analytics.player_profiles.seasons import current_season_start, seasons_since_1996
from analytics.player_profiles.similarity import build_similarity_embeddings, build_similarity_index
from analytics.player_profiles.storage import ProfileStorage, StorageConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlayerProfilePipelineConfig:
    seasons: list[str]
    current_season_start_year: int
    refresh_raw: bool = False
    k_similar: int = 10
    checkpoint_key: str = "processed/player_profile_checkpoint.json"

    @classmethod
    def default(cls) -> "PlayerProfilePipelineConfig":
        current = current_season_start()
        return cls(
            seasons=seasons_since_1996(current),
            current_season_start_year=current,
        )


class PlayerProfilePipeline:
    def __init__(
        self,
        *,
        storage: ProfileStorage,
        client: NbaStatsClient | None = None,
        config: PlayerProfilePipelineConfig | None = None,
    ):
        self.storage = storage
        self.client = client or NbaStatsClient()
        self.config = config or PlayerProfilePipelineConfig.default()

    def run(self) -> dict[str, Any]:
        logger.info("Starting player profile build for %s seasons", len(self.config.seasons))
        season_tables = []
        checkpoint = self._load_checkpoint()
        for season in self.config.seasons:
            logger.info("Building season player table: %s", season)
            table_key = f"processed/season_player_tables/season={season}/players.parquet"
            if self.storage.exists(table_key) and not self.config.refresh_raw:
                cached = self.storage.read_parquet(table_key)
                if _has_canonical_season_schema(cached):
                    season_tables.append(cached[CANONICAL_SEASON_PLAYER_COLUMNS])
                    continue
                logger.info("Cached season table for %s has an older schema; rebuilding", season)
            season_table = self._build_one_season(season)
            if season_table.empty:
                logger.warning("Season %s returned no player rows", season)
                continue
            self.storage.write_parquet(table_key, season_table)
            checkpoint["completed_seasons"] = sorted(set(checkpoint.get("completed_seasons", []) + [season]))
            self._write_checkpoint(checkpoint)
            season_tables.append(season_table)

        if not season_tables:
            raise RuntimeError("No player season data was gathered.")

        all_seasons = pd.concat(season_tables, ignore_index=True, sort=False)
        self.storage.write_parquet("processed/player_seasons.parquet", all_seasons)

        normalized = add_normalized_features(all_seasons)
        self.storage.write_parquet("features/player_season_features.parquet", normalized)

        career = aggregate_career_features(normalized)
        career = add_career_percentiles(career)
        career = apply_player_filter(
            career,
            current_season_start=self.config.current_season_start_year,
            config=PlayerFilterConfig(),
        )
        career = add_archetypes(career)

        has_missing = career["height"].isna() | career["weight"].isna()
        missing_pids = career.loc[has_missing, "player_id"].unique().tolist()
        if missing_pids:
            for col in ["draft_year", "draft_position"]:
                if col in career.columns:
                    career[col] = career[col].astype(object)
            logger.info("Scraping missing bio info for %s players...", len(missing_pids))
            
            from collections import deque
            import time
            import random
            from nba_api.stats.endpoints import commonplayerinfo
            from analytics.player_profiles.nba_client import NBA_HEADERS

            queue = deque(missing_pids)
            attempts = {pid: 0 for pid in missing_pids}
            max_attempts = getattr(self.client, "retries", 3)
            base_sleep = getattr(self.client, "base_sleep", 1.5)
            jitter = getattr(self.client, "jitter", 0.35)
            timeout_val = getattr(self.client, "timeout", 15)

            while queue:
                pid = queue.popleft()
                attempts[pid] += 1
                try:
                    logger.info(
                        "Fetching on-demand bio for missing player %s (attempt %s/%s)",
                        pid,
                        attempts[pid],
                        max_attempts,
                    )
                    # Pass headers for reliability and use client timeout configuration
                    endpoint = commonplayerinfo.CommonPlayerInfo(
                        player_id=int(pid),
                        timeout=timeout_val,
                        headers=NBA_HEADERS,
                    )
                    df_bio = endpoint.common_player_info.get_data_frame()
                    if not df_bio.empty:
                        row = df_bio.iloc[0].to_dict()
                        h = row.get("HEIGHT")
                        w = row.get("WEIGHT")
                        dy = row.get("DRAFT_YEAR")
                        dn = row.get("DRAFT_NUMBER")
                        pos = row.get("POSITION")

                        from analytics.player_profiles.features import position_group
                        pg = position_group(pos)
                        if pg == "G":
                            fallback_h, fallback_w = "6-3", 190.0
                        elif pg == "B":
                            fallback_h, fallback_w = "6-10", 240.0
                        else:
                            fallback_h, fallback_w = "6-7", 215.0

                        def clean_val(v):
                            if v is None or str(v).strip() == "" or str(v).lower() in ("nan", "none"):
                                return None
                            val_str = str(v).strip()
                            if val_str.lower() == "undrafted":
                                return "Undrafted"
                            try:
                                return int(float(val_str))
                            except ValueError:
                                return val_str

                        h_val = h if (h and str(h).strip()) else fallback_h
                        w_val = clean_val(w)
                        if w_val is None:
                            w_val = fallback_w

                        idx = career[career["player_id"] == pid].index
                        if not idx.empty:
                            career.loc[idx, "height"] = h_val
                            career.loc[idx, "weight"] = w_val
                            career.loc[idx, "draft_year"] = clean_val(dy)
                            career.loc[idx, "draft_position"] = clean_val(dn)
                            if pos:
                                career.loc[idx, "primary_position"] = pg

                    # Sleep with extra cushion to avoid timeouts and rate limits
                    sleep_time = base_sleep + 1.0 + random.random() * (jitter + 0.5)
                    logger.info("Successfully fetched player %s bio. Sleeping for %.2fs", pid, sleep_time)
                    time.sleep(sleep_time)

                except Exception as e:
                    logger.warning(
                        "Failed to fetch bio for player %s during build (attempt %s/%s): %s",
                        pid,
                        attempts[pid],
                        max_attempts,
                        e,
                    )
                    if attempts[pid] < max_attempts:
                        logger.info("Adding player %s to the queue to retry", pid)
                        queue.append(pid)
                        # Sleep longer after a failure to allow rate limits to cool down
                        sleep_time = (base_sleep * 2.0) + (random.random() * 2.0)
                        logger.info("Sleeping for %.2fs before retrying next player", sleep_time)
                        time.sleep(sleep_time)
                    else:
                        logger.error(
                            "Exceeded maximum attempts (%s) to fetch bio for player %s. Moving on.",
                            max_attempts,
                            pid,
                        )

        def force_clean_draft_val(x):
            if pd.isna(x) or x is None:
                return None
            val_str = str(x).strip()
            if not val_str or val_str.lower() in ("nan", "none"):
                return None
            if val_str.lower() == "undrafted":
                return "Undrafted"
            try:
                return str(int(float(val_str)))
            except ValueError:
                return val_str

        for col in ["draft_year", "draft_position"]:
            if col in career.columns:
                career[col] = career[col].apply(force_clean_draft_val)

        self.storage.write_parquet("features/player_career_features.parquet", career)

        embedding_features, feature_cols = build_embedding_features(career)
        self.storage.write_parquet("features/player_similarity_features.parquet", embedding_features)
        embeddings, _pca = build_similarity_embeddings(career, embedding_features)
        self.storage.write_parquet("embeddings/player_similarity_embeddings.parquet", embeddings)
        similar = build_similarity_index(
            career,
            embeddings,
            embedding_features,
            k=self.config.k_similar,
        )
        self.storage.write_json("embeddings/player_similarity_index.json", similar)

        profiles = self._build_profiles_json(career, similar)
        self.storage.write_json("processed/player_profiles.json", profiles)
        metadata = self._build_metadata(career)
        self.storage.write_json("processed/player_metadata.json", metadata)

        return {
            "players": len(profiles),
            "seasons": len(self.config.seasons),
            "feature_columns": feature_cols,
            "storage": self.storage.config.root,
            "gcs_bucket": self.storage.config.gcs_bucket,
        }

    def _build_one_season(self, season: str) -> pd.DataFrame:
        results: dict[str, EndpointResult] = {}
        endpoint_plan = {
            "traditional": (lambda: self._fetch_traditional_totals(season), True, "traditional_totals"),
            "bio": (lambda: self._fetch_player_index(season), False, "player_index"),
        }
        for name, (fetch, required, raw_name) in endpoint_plan.items():
            raw_key = f"raw/season={season}/{raw_name}.json"
            if self.storage.exists(raw_key) and not self.config.refresh_raw:
                raw = self.storage.read_json(raw_key)
                frames = _frames_from_raw_result(name, raw)
                results[name] = EndpointResult(name=name, params={}, raw=raw, frames=frames)
                continue
            try:
                result = fetch()
                self.storage.write_json(raw_key, result.raw)
                results[name] = result
            except Exception as exc:
                if required:
                    raise
                logger.warning("Optional endpoint %s unavailable for %s: %s", name, season, exc)
                raw = {"error": str(exc), "resultSets": []}
                self.storage.write_json(raw_key, raw)
                results[name] = EndpointResult(name=name, params={}, raw=raw, frames={})

        return build_season_player_table(
            season=season,
            traditional=normalize_traditional_totals(_first_frame(results["traditional"])),
            advanced=pd.DataFrame(),
            misc=pd.DataFrame(),
            scoring=pd.DataFrame(),
            usage=pd.DataFrame(),
            bio=_first_frame(results["bio"]),
            drives=pd.DataFrame(),
            touches=pd.DataFrame(),
            passing=pd.DataFrame(),
            pullup=pd.DataFrame(),
            catch_shoot=pd.DataFrame(),
            hustle=pd.DataFrame(),
            defense=pd.DataFrame(),
            shot_chart=pd.DataFrame(),
        )

    def _fetch_traditional_totals(self, season: str) -> EndpointResult:
        result = self.client.league_leaders_totals(season)
        logger.info("Fetched %s traditional totals using LeagueLeaders", season)
        return result

    def _fetch_player_index(self, season: str) -> EndpointResult:
        result = self.client.player_index(season)
        logger.info("Fetched %s player index stats", season)
        return result


    def _build_profiles_json(
        self,
        career: pd.DataFrame,
        similar: dict[str, list[dict[str, Any]]],
    ) -> dict[str, dict[str, Any]]:
        profiles = {}
        adjusted_pfvs = []
        height_buckets = []
        for _, row in career.iterrows():
            pid = str(int(row["player_id"]))
            profiles[pid] = profile_payload(row, similar.get(pid, []))
            raw_metrics = style_summary(row, adjust_for_mpg=False)
            adjusted_pfvs.append(calculate_adjusted_pfv(raw_metrics))
            height_buckets.append(height_bucket(row.get("height")))

        if adjusted_pfvs:
            apfvs = calculate_apfv_batch_by_height(adjusted_pfvs, height_buckets)
            for p, apfv_val in zip(profiles.values(), apfvs):
                p["apfv"] = apfv_val
        return profiles

    def _build_metadata(self, career: pd.DataFrame) -> dict[str, dict[str, Any]]:
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

    def _load_checkpoint(self) -> dict[str, Any]:
        if self.storage.exists(self.config.checkpoint_key):
            return self.storage.read_json(self.config.checkpoint_key)
        return {"completed_seasons": []}

    def _write_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.storage.write_json(self.config.checkpoint_key, checkpoint)


def copy_processed_outputs_to_static(storage_root: str, static_dir: Path, gcs_project: str | None = None) -> None:
    storage = ProfileStorage(StorageConfig.from_uri(storage_root, gcs_project=gcs_project))
    static_dir.mkdir(parents=True, exist_ok=True)
    
    for name in ("player_profiles.json", "player_metadata.json"):
        key = f"processed/{name}"
        if storage.is_gcs:
            blob = storage._bucket.blob(storage._blob_name(key))
            (static_dir / name).write_text(blob.download_as_text())
        else:
            src = Path(storage_root)
            (static_dir / name).write_text((src / key).read_text())
            
    for rel in (
        "features/player_career_features.parquet",
        "features/player_season_features.parquet",
        "features/player_similarity_features.parquet",
        "embeddings/player_similarity_embeddings.parquet",
    ):
        target = static_dir / Path(rel).name
        if storage.is_gcs:
            blob = storage._bucket.blob(storage._blob_name(rel))
            blob.download_to_filename(str(target))
        else:
            source = Path(storage_root) / rel
            target.write_bytes(source.read_bytes())


def _first_frame(result: EndpointResult) -> pd.DataFrame:
    if not result.frames:
        return pd.DataFrame()
    return next(iter(result.frames.values())).copy()


def _has_canonical_season_schema(frame: pd.DataFrame) -> bool:
    return list(frame.columns) == CANONICAL_SEASON_PLAYER_COLUMNS


def _frames_from_raw_result(name: str, raw: dict[str, Any]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    result_sets = raw.get("resultSets") or raw.get("resultSet") or []
    if isinstance(result_sets, dict):
        result_sets = [result_sets]
    for i, result in enumerate(result_sets):
        headers = result.get("headers") or []
        rows = result.get("rowSet") or []
        frames[f"{name}_{i}"] = pd.DataFrame(rows, columns=headers)
    return frames
