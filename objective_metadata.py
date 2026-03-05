"""Metadata management for objective functions.

This module manages metadata for generated objective functions, storing
descriptions and creation information separately from the executable code.
Metadata is stored in objectives/metadata.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Dict, Optional

from database.schema import create_timestamp
import workspace_config

_metadata_lock = Lock()


def _get_metadata_file() -> Path:
    """Get the metadata file path using workspace configuration."""
    return workspace_config.get_objectives_dir() / "metadata.json"


def _read_metadata() -> Dict[str, dict]:
    """Read metadata from file (thread-safe).

    Returns:
        Dictionary mapping objective_id (as string) to metadata dict
    """
    metadata_file = _get_metadata_file()
    with _metadata_lock:
        if not metadata_file.exists():
            return {}

        try:
            with metadata_file.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}


def _write_metadata(metadata: Dict[str, dict]) -> None:
    """Write metadata to file (thread-safe).

    Args:
        metadata: Dictionary mapping objective_id to metadata dict
    """
    metadata_file = _get_metadata_file()
    with _metadata_lock:
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with metadata_file.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)


def save_objective_metadata(
    objective_id: int,
    description: str,
    round_id: int,
) -> None:
    """Save metadata for an objective function.

    Args:
        objective_id: Unique identifier for the objective
        description: Human-readable description of what the objective measures
        round_id: Objective round ID that created this objective
    """
    metadata = _read_metadata()

    metadata[str(objective_id)] = {
        "objective_id": objective_id,
        "description": description,
        "created_round": round_id,
        "created_timestamp": create_timestamp(),
    }

    _write_metadata(metadata)


def load_objective_metadata(objective_id: int) -> Optional[dict]:
    """Load metadata for a specific objective.

    Args:
        objective_id: The objective ID to look up

    Returns:
        Metadata dict if found, None otherwise
    """
    metadata = _read_metadata()
    return metadata.get(str(objective_id))


def list_objective_metadata() -> Dict[str, dict]:
    """Get metadata for all objectives.

    Returns:
        Dictionary mapping objective_id (as string) to metadata dict
    """
    return _read_metadata()


def get_objective_description(objective_id: int, default: str = "No description available") -> str:
    """Get the description for an objective.

    Args:
        objective_id: The objective ID to look up
        default: Default value if description not found

    Returns:
        Description string or default value
    """
    metadata = load_objective_metadata(objective_id)
    if metadata:
        return metadata.get("description", default)
    return default


__all__ = [
    "save_objective_metadata",
    "load_objective_metadata",
    "list_objective_metadata",
    "get_objective_description",
]
