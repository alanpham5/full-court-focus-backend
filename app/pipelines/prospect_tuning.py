                                                             

                                                                            
                                                                       
                                                                             
                                                                        
                                 

                                                                             
                                                                            

                                                                            
   
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import PROSPECT_TUNING_DIR

PARAM_KEYS = (
    "feature_norm",
    "kernel",
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

LEGACY_PARAM_DEFAULTS = {
    "feature_norm": "percentile",
    "kernel": "laplacian",
}


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
    params = {k: raw[k] for k in PARAM_KEYS if k in raw}
    for key, default in LEGACY_PARAM_DEFAULTS.items():
        params.setdefault(key, default)
    return params


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
                                                                        

                                                                               
                                                                              
       
    merged = {k: defaults[k] for k in PARAM_KEYS if k in defaults}
    name = active_name()
    if name:
        try:
            merged.update(load_version(name))
        except FileNotFoundError:
            pass
    return merged
