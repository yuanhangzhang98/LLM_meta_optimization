"""Metadata management for schedule functions.

This module manages metadata for generated schedule functions, storing
descriptions and creation information separately from the executable code.
Metadata is stored in schedules/metadata.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Dict, Literal, Optional

from database.schema import create_timestamp
import workspace_config

_metadata_lock = Lock()

FidelityLevel = Literal["low", "medium", "high"]


def _get_metadata_file() -> Path:
    """Get the metadata file path using workspace configuration."""
    return workspace_config.get_schedules_dir() / "metadata.json"


def _read_metadata() -> Dict[str, dict]:
    """Read metadata from file (thread-safe).

    Returns:
        Dictionary mapping schedule_id (as string) to metadata dict
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
        metadata: Dictionary mapping schedule_id to metadata dict
    """
    metadata_file = _get_metadata_file()
    with _metadata_lock:
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with metadata_file.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)


def save_schedule_metadata(
    schedule_id: int,
    description: str,
    fidelities: list[str],
    round_id: int,
) -> None:
    """Save metadata for a schedule function.

    Args:
        schedule_id: Unique identifier for the schedule
        description: Human-readable description of the schedule
        fidelities: List of available fidelity levels (e.g., ["low", "medium", "high"])
        round_id: Objective round ID that created this schedule
    """
    metadata = _read_metadata()

    metadata[str(schedule_id)] = {
        "schedule_id": schedule_id,
        "description": description,
        "fidelities": fidelities,
        "created_round": round_id,
        "created_timestamp": create_timestamp(),
    }

    _write_metadata(metadata)


def load_schedule_metadata(schedule_id: int) -> Optional[dict]:
    """Load metadata for a specific schedule.

    Args:
        schedule_id: The schedule ID to look up

    Returns:
        Metadata dict if found, None otherwise
    """
    metadata = _read_metadata()
    return metadata.get(str(schedule_id))


def list_schedule_metadata() -> Dict[str, dict]:
    """Get metadata for all schedules.

    Returns:
        Dictionary mapping schedule_id (as string) to metadata dict
    """
    return _read_metadata()


def get_schedule_description(schedule_id: int, default: str = "No description available") -> str:
    """Get the description for a schedule.

    Args:
        schedule_id: The schedule ID to look up
        default: Default value if description not found

    Returns:
        Description string or default value
    """
    metadata = load_schedule_metadata(schedule_id)
    if metadata:
        return metadata.get("description", default)
    return default


__all__ = [
    "save_schedule_metadata",
    "load_schedule_metadata",
    "list_schedule_metadata",
    "get_schedule_description",
]
