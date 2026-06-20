from __future__ import annotations

import os
import warnings
from pathlib import Path

_GCLOUD_WARNINGS_CONFIGURED = False


def configure_gcloud_warnings() -> None:
    global _GCLOUD_WARNINGS_CONFIGURED
    if _GCLOUD_WARNINGS_CONFIGURED:
        return

    quota_project = os.getenv("GOOGLE_CLOUD_QUOTA_PROJECT")
    if not quota_project:
        project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCLOUD_PROJECT")
        if project:
            os.environ.setdefault("GOOGLE_CLOUD_QUOTA_PROJECT", project)

    warnings.filterwarnings(
        "ignore",
        message=".*authenticated using end user credentials from Google Cloud SDK without a quota project.*",
        category=UserWarning,
    )
    _GCLOUD_WARNINGS_CONFIGURED = True


def load_env_defaults(app_root: Path | None = None) -> None:
    root = app_root or Path(__file__).resolve().parent
    env_path = root / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
    configure_gcloud_warnings()
