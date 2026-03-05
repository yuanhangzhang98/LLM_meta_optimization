"""Dynamic schedule loading utilities."""

import importlib.util
import sys
from pathlib import Path
from typing import Callable, List

import workspace_config


def _load_schedule_module(schedule_id: int):
    """Load a schedule module by ID using direct file loading.

    Args:
        schedule_id: ID of the schedule to load

    Returns:
        Loaded module object

    Raises:
        FileNotFoundError: If schedule file doesn't exist
        ImportError: If module cannot be loaded
    """
    schedules_dir = workspace_config.get_schedules_dir()
    schedule_path = schedules_dir / f"schedule_{schedule_id}.py"

    if not schedule_path.exists():
        raise FileNotFoundError(
            f"Schedule file not found: {schedule_path}\n"
            f"Available schedules: {list_available_schedules()}"
        )

    # Load module directly from file path (avoids module name conflicts with workspace)
    full_module_name = f"_loaded_schedules.schedule_{schedule_id}"

    # Always reload to support dynamic updates
    spec = importlib.util.spec_from_file_location(full_module_name, schedule_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load schedule module from {schedule_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[full_module_name] = module
    spec.loader.exec_module(module)

    return module


def load_schedule(schedule_id: int) -> Callable:
    """Load schedule function by ID.

    Args:
        schedule_id: ID of the schedule to load (corresponds to schedules/schedule_M.py)

    Returns:
        Callable schedule function with signature: schedule(design_id: int) -> List[Dict]

    Raises:
        FileNotFoundError: If schedule file doesn't exist
        ImportError: If schedule module cannot be imported
        AttributeError: If schedule function is not defined in module
    """
    module = _load_schedule_module(schedule_id)

    # Get the schedule function
    if not hasattr(module, "schedule"):
        raise AttributeError(
            f"Schedule module schedule_{schedule_id} must define a 'schedule' function"
        )

    schedule_fn = module.schedule
    return schedule_fn


def load_schedule_fidelities(schedule_id: int) -> dict:
    """Load all fidelity schedule functions by ID.

    Args:
        schedule_id: ID of the schedule to load (corresponds to schedules/schedule_M.py)

    Returns:
        Dict with keys 'low', 'medium', 'high' mapping to callable schedule functions.
        Each function has signature: schedule_X(design_id: int) -> List[Dict]

    Raises:
        FileNotFoundError: If schedule file doesn't exist
        ImportError: If schedule module cannot be imported
        AttributeError: If any fidelity function is missing
    """
    module = _load_schedule_module(schedule_id)

    # Get all three fidelity functions
    fidelity_fns = {}
    for fidelity in ['low', 'medium', 'high']:
        fn_name = f"schedule_{fidelity}"
        if not hasattr(module, fn_name):
            raise AttributeError(
                f"Schedule module schedule_{schedule_id} must define a '{fn_name}' function"
            )
        fn = getattr(module, fn_name)
        if not validate_schedule(fn):
            raise AttributeError(
                f"Schedule function '{fn_name}' must be callable and accept 'design_id' as first parameter"
            )
        fidelity_fns[fidelity] = fn

    return fidelity_fns


def list_available_schedules() -> List[int]:
    """List all available schedule IDs.

    Returns:
        Sorted list of schedule IDs found in schedules/ directory
    """
    schedules_dir = workspace_config.get_schedules_dir()
    if not schedules_dir.exists():
        return []

    schedule_ids = []
    for schedule_file in schedules_dir.glob("schedule_*.py"):
        try:
            schedule_id = int(schedule_file.stem.split("_")[1])
            schedule_ids.append(schedule_id)
        except (IndexError, ValueError):
            # Skip files that don't match schedule_N.py pattern
            continue

    return sorted(schedule_ids)


def get_next_schedule_id() -> int:
    """Get the next available schedule ID for creating new schedules.

    Returns:
        Next unused schedule ID (max existing ID + 1, or 1 if none exist)
    """
    existing = list_available_schedules()
    if not existing:
        return 1
    return max(existing) + 1


def validate_schedule(schedule_fn: Callable) -> bool:
    """Validate that a schedule function has the correct signature.

    Args:
        schedule_fn: Function to validate

    Returns:
        True if valid, False otherwise
    """
    import inspect

    # Check it's callable
    if not callable(schedule_fn):
        return False

    # Check signature accepts design_id parameter
    sig = inspect.signature(schedule_fn)
    params = list(sig.parameters.keys())

    # Should have at least design_id parameter (may have default)
    if not params or params[0] != "design_id":
        return False

    return True


__all__ = [
    "load_schedule",
    "load_schedule_fidelities",
    "list_available_schedules",
    "get_next_schedule_id",
    "validate_schedule",
]
