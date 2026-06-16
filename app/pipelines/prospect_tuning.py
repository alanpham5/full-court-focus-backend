"""Versioned tuning store for the prospect -> NBA comp model.

All tunable parameters of the comp engine (feature weights, bandwidth,
smoothing, and the display-selection knobs including the one-sided scoring
affinity) live here as named JSON versions so a tuning can be saved, diffed,
reverted, and re-run.

Layout (under config.PROSPECT_TUNING_DIR):
    versions/<name>.json   one frozen parameter set per file
    active.json            {"active": "<name>"}

`prospects_pipeline` calls `resolve_active(defaults)` at import to overlay the
active version on the code defaults, and exposes `apply_tuning` so a one-off run
can use any version without changing the active pointer.

This module is pure-JSON (no numpy / pipeline imports) so it is safe to import
from anywhere, including the CLI.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import PROSPECT_TUNING_DIR

# Parameter keys that make up a tuning version. Anything outside this set is
# ignored on load (and dropped on save) so the schema stays explicit.
PARAM_KEYS = (
    "features",
    "weights",
    "bandwidth",
    "smooth_lambda",
    "smooth_topk",
    "similarity_gamma",
    "popularity_penalty",
    "max_appearances",
    "establishment_floor",
    "establishment_alpha",
    "quality_bucket",
    "scoring_affinity_tau",
)


def _versions_dir() -> Path:
    return Path(PROSPECT_TUNING_DIR) / "versions"


def _active_path() -> Path:
    return Path(PROSPECT_TUNING_DIR) / "active.json"


def version_path(name: str) -> Path:
    return _versions_dir() / f"{name}.json"


def list_versions() -> list[str]:
    d = _versions_dir()
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.json"))


def active_name() -> str | None:
    p = _active_path()
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("active")
    except (json.JSONDecodeError, OSError):
        return None


def load_version(name: str) -> dict[str, Any]:
    p = version_path(name)
    if not p.exists():
        raise FileNotFoundError(
            f"Tuning version '{name}' not found at {p}. "
            f"Available: {', '.join(list_versions()) or '(none)'}"
        )
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {k: raw[k] for k in PARAM_KEYS if k in raw}


def save_version(name: str, params: dict[str, Any], *,
                 description: str = "", overwrite: bool = False) -> Path:
    p = version_path(name)
    if p.exists() and not overwrite:
        raise FileExistsError(
            f"Tuning version '{name}' already exists at {p}. Pass overwrite=True."
        )
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"name": name, "description": description}
    payload.update({k: params[k] for k in PARAM_KEYS if k in params})
    missing = [k for k in PARAM_KEYS if k not in params]
    if missing:
        raise ValueError(f"Cannot save version '{name}': missing params {missing}")
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def set_active(name: str) -> None:
    if name not in list_versions():
        raise FileNotFoundError(
            f"Cannot activate unknown tuning version '{name}'. "
            f"Available: {', '.join(list_versions()) or '(none)'}"
        )
    p = _active_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"active": name}, indent=2), encoding="utf-8")


def resolve_active(defaults: dict[str, Any]) -> dict[str, Any]:
    """Return code `defaults` overlaid with the active version (if any).

    Falls back silently to `defaults` when no store / active pointer exists, so
    the pipeline still runs on a fresh checkout before any version is written.
    """
    merged = {k: defaults[k] for k in PARAM_KEYS if k in defaults}
    name = active_name()
    if name:
        try:
            merged.update(load_version(name))
        except FileNotFoundError:
            pass
    return merged
