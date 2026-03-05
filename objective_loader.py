"""Dynamic objective loading utilities."""

import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

import workspace_config


def load_objective(objective_id: int) -> Callable:
    """Load objective function by ID.

    Args:
        objective_id: ID of the objective to load (corresponds to objectives/objective_N.py)

    Returns:
        Callable objective function with signature: objective(experiment_results: List[Dict]) -> float

    Raises:
        FileNotFoundError: If objective file doesn't exist
        ImportError: If objective module cannot be imported
        AttributeError: If objective function is not defined in module
    """
    objectives_dir = workspace_config.get_objectives_dir()
    objective_path = objectives_dir / f"objective_{objective_id}.py"

    if not objective_path.exists():
        raise FileNotFoundError(
            f"Objective file not found: {objective_path}\n"
            f"Available objectives: {list_available_objectives()}"
        )

    # Load module directly from file path (avoids module name conflicts with workspace)
    module_name = f"objective_{objective_id}"

    # Create a unique module name to avoid conflicts
    full_module_name = f"_loaded_objectives.objective_{objective_id}"

    # Always reload to support dynamic updates
    spec = importlib.util.spec_from_file_location(full_module_name, objective_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load objective module from {objective_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[full_module_name] = module
    spec.loader.exec_module(module)

    # Get the objective function
    if not hasattr(module, "objective"):
        raise AttributeError(
            f"Objective module {objective_path} must define an 'objective' function"
        )

    objective_fn = module.objective
    return objective_fn


def list_available_objectives() -> List[int]:
    """List all available objective IDs.

    Returns:
        Sorted list of objective IDs found in objectives/ directory
    """
    objectives_dir = workspace_config.get_objectives_dir()
    if not objectives_dir.exists():
        return []

    objective_ids = []
    for objective_file in objectives_dir.glob("objective_*.py"):
        try:
            objective_id = int(objective_file.stem.split("_")[1])
            objective_ids.append(objective_id)
        except (IndexError, ValueError):
            # Skip files that don't match objective_N.py pattern
            continue

    return sorted(objective_ids)


def get_next_objective_id() -> int:
    """Get the next available objective ID for creating new objectives.

    Returns:
        Next unused objective ID (max existing ID + 1, or 1 if none exist)
    """
    existing = list_available_objectives()
    if not existing:
        return 1
    return max(existing) + 1


def validate_objective(objective_fn: Callable) -> bool:
    """Validate that an objective function has the correct signature.

    Args:
        objective_fn: Function to validate

    Returns:
        True if valid, False otherwise
    """
    import inspect

    # Check it's callable
    if not callable(objective_fn):
        return False

    # Check signature accepts experiment_results parameter
    sig = inspect.signature(objective_fn)
    params = list(sig.parameters.keys())

    # Should have at least experiment_results parameter (may have default)
    if not params or params[0] != "experiment_results":
        return False

    return True


__all__ = [
    "load_objective",
    "list_available_objectives",
    "get_next_objective_id",
    "validate_objective",
]
