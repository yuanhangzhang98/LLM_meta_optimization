"""State management for meta-agent.

This module manages persistent state for the meta-agent, including research
assessments, objective directions, and weight adjustments. State is stored
in meta_agent[_workspace]/state.json with workspace suffix when configured.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

from database.schema import create_timestamp
import workspace_config


@dataclass
class MetaAgentState:
    """State output from a meta-agent round.

    Attributes:
        round_id: Meta-agent round number
        timestamp: ISO format timestamp
        research_phase: Current phase (exploring, converging, stuck, breakthrough_needed, refining)
        research_assessment: Analysis of current research progress
        objective_directions: Guidance for next objective generation
        weight_adjustments: Dict mapping objective_id to weight multiplier
    """
    round_id: int
    timestamp: str
    research_phase: str
    research_assessment: str
    objective_directions: str
    weight_adjustments: Dict[int, float]


_state_lock = Lock()


def _get_state_dir() -> Path:
    """Get the meta-agent state directory using workspace configuration."""
    return workspace_config.get_meta_agent_dir()


def _get_state_file() -> Path:
    """Get the meta-agent state file path."""
    return _get_state_dir() / "state.json"


def _read_state_data() -> dict:
    """Read state data from file (thread-safe).

    Returns:
        Dictionary with 'current' and 'history' keys
    """
    state_file = _get_state_file()
    with _state_lock:
        if not state_file.exists():
            return {"current": None, "history": []}

        try:
            with state_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
                # Ensure expected structure
                if "current" not in data:
                    data["current"] = None
                if "history" not in data:
                    data["history"] = []
                return data
        except (json.JSONDecodeError, IOError):
            return {"current": None, "history": []}


def _write_state_data(data: dict) -> None:
    """Write state data to file (thread-safe).

    Args:
        data: Dictionary with 'current' and 'history' keys
    """
    state_file = _get_state_file()
    with _state_lock:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        with state_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def _state_to_dict(state: MetaAgentState) -> dict:
    """Convert MetaAgentState to serializable dict.

    Ensures weight_adjustments keys are strings for JSON compatibility.
    """
    d = asdict(state)
    # JSON requires string keys
    d["weight_adjustments"] = {str(k): v for k, v in state.weight_adjustments.items()}
    return d


def _dict_to_state(d: dict) -> MetaAgentState:
    """Convert dict to MetaAgentState.

    Converts weight_adjustments keys back to integers.
    """
    # Convert string keys back to integers
    weight_adjustments = {int(k): v for k, v in d.get("weight_adjustments", {}).items()}
    return MetaAgentState(
        round_id=d["round_id"],
        timestamp=d["timestamp"],
        research_phase=d["research_phase"],
        research_assessment=d["research_assessment"],
        objective_directions=d["objective_directions"],
        weight_adjustments=weight_adjustments,
    )


def save_meta_agent_state(state: MetaAgentState) -> None:
    """Save meta-agent state, adding to history.

    Args:
        state: MetaAgentState to save
    """
    data = _read_state_data()

    state_dict = _state_to_dict(state)

    # Add current to history if it exists
    if data["current"] is not None:
        data["history"].append(data["current"])

    # Set new current
    data["current"] = state_dict

    _write_state_data(data)


def load_meta_agent_state() -> Optional[MetaAgentState]:
    """Load the current meta-agent state.

    Returns:
        Current MetaAgentState or None if no state exists
    """
    data = _read_state_data()
    if data["current"] is None:
        return None
    return _dict_to_state(data["current"])


def list_meta_agent_history() -> List[MetaAgentState]:
    """Get all historical meta-agent states.

    Returns:
        List of MetaAgentState in chronological order (oldest first)
    """
    data = _read_state_data()
    history = [_dict_to_state(d) for d in data.get("history", [])]
    # Add current if exists
    if data["current"] is not None:
        history.append(_dict_to_state(data["current"]))
    return history


def get_latest_weight_adjustments() -> Dict[int, float]:
    """Get weight adjustments from the latest meta-agent state.

    Returns:
        Dict mapping objective_id to weight multiplier, empty if no state
    """
    state = load_meta_agent_state()
    if state is None:
        return {}
    return state.weight_adjustments


def get_latest_objective_directions() -> Optional[str]:
    """Get objective directions from the latest meta-agent state.

    Returns:
        Objective directions string or None if no state
    """
    state = load_meta_agent_state()
    if state is None:
        return None
    return state.objective_directions


def get_effective_meta_weights() -> Dict[int, float]:
    """Get effective meta weights from the latest state.

    Returns:
        Dict mapping objective_id to weight multiplier
    """
    state = load_meta_agent_state()
    if state is None:
        return {}

    return dict(state.weight_adjustments)


def create_meta_agent_state(
    round_id: int,
    research_phase: str,
    research_assessment: str,
    objective_directions: str,
    weight_adjustments: Dict[int, float],
) -> MetaAgentState:
    """Create a new MetaAgentState with current timestamp.

    Args:
        round_id: Meta-agent round number
        research_phase: Current research phase
        research_assessment: Analysis of research progress
        objective_directions: Guidance for next objective
        weight_adjustments: Dict of objective_id to weight multiplier

    Returns:
        New MetaAgentState instance
    """
    return MetaAgentState(
        round_id=round_id,
        timestamp=create_timestamp(),
        research_phase=research_phase,
        research_assessment=research_assessment,
        objective_directions=objective_directions,
        weight_adjustments=weight_adjustments,
    )


__all__ = [
    "MetaAgentState",
    "save_meta_agent_state",
    "load_meta_agent_state",
    "list_meta_agent_history",
    "get_latest_weight_adjustments",
    "get_latest_objective_directions",
    "get_effective_meta_weights",
    "create_meta_agent_state",
]
