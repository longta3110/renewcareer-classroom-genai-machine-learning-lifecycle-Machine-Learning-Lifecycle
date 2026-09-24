from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "src" / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load YAML configuration for the lifecycle pipeline."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a possibly relative path from the project root."""
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def ensure_parent_dir(path: str | Path) -> Path:
    """Create the parent directory for a path and return the resolved path."""
    resolved = resolve_project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    """Write a dictionary as pretty JSON."""
    resolved = ensure_parent_dir(path)
    with resolved.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
    return resolved
