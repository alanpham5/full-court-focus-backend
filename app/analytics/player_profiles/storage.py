from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

from parquet_io import read_parquet, write_teams_parquet
from env_defaults import configure_gcloud_warnings

configure_gcloud_warnings()

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StorageConfig:
    root: str
    gcs_bucket: str | None = None
    gcs_project: str | None = None

    @classmethod
    def from_uri(cls, uri: str, *, gcs_project: str | None = None) -> "StorageConfig":
        if uri.startswith("gs://"):
            rest = uri[5:].strip("/")
            bucket, _, prefix = rest.partition("/")
            return cls(root=prefix, gcs_bucket=bucket, gcs_project=gcs_project)
        return cls(root=uri)


class ProfileStorage:
    def __init__(self, config: StorageConfig):
        self.config = config
        self._bucket = None
        if config.gcs_bucket:
            try:
                from google.cloud import storage as gcs_storage
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "google-cloud-storage is required for gs:// player profile storage. "
                    "Install it with: app/.venv/bin/python -m pip install -r app/requirements.txt"
                ) from exc

            project = (
                config.gcs_project
                or os.getenv("GOOGLE_CLOUD_PROJECT")
                or os.getenv("GCLOUD_PROJECT")
            )
            client = gcs_storage.Client(project=project)
            self._bucket = client.bucket(config.gcs_bucket)

    @property
    def is_gcs(self) -> bool:
        return self._bucket is not None

    def _local_path(self, key: str) -> Path:
        return Path(self.config.root) / key

    def _blob_name(self, key: str) -> str:
        prefix = self.config.root.strip("/")
        return f"{prefix}/{key}".strip("/")

    def exists(self, key: str) -> bool:
        if self._bucket is not None:
            return self._bucket.blob(self._blob_name(key)).exists()
        return self._local_path(key).exists()

    def write_json(self, key: str, payload: Any) -> None:
        data = json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")
        if self._bucket is not None:
            blob = self._bucket.blob(self._blob_name(key))
            blob.upload_from_string(data, content_type="application/json")
            return
        path = self._local_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def read_json(self, key: str) -> Any:
        if self._bucket is not None:
            return json.loads(self._bucket.blob(self._blob_name(key)).download_as_text())
        return json.loads(self._local_path(key).read_text())

    def write_parquet(self, key: str, df: pd.DataFrame) -> None:
        if self._bucket is not None:
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "data.parquet"
                write_teams_parquet(df, path)
                data = path.read_bytes()
            blob = self._bucket.blob(self._blob_name(key))
            blob.upload_from_string(
                data,
                content_type="application/vnd.apache.parquet",
            )
            return
        path = self._local_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_teams_parquet(df, path)

    def read_parquet(self, key: str) -> pd.DataFrame:
        if self._bucket is not None:
            data = self._bucket.blob(self._blob_name(key)).download_as_bytes()
            return read_parquet(BytesIO(data))
        return read_parquet(self._local_path(key))
