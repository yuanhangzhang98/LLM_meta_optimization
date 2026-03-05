import argparse
import importlib
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, List, Dict, Any

from solver_writer import extract_solver_id_from_path
from schedule_loader import load_schedule


def import_callable(module_path: str, workspace: Path):
    if "." not in module_path:
        raise ValueError(f"Entry '{module_path}' must be in module.function format.")
    module_name, func_name = module_path.rsplit(".", 1)

    workspace_path = Path(workspace).resolve()
    workspace_str = str(workspace_path)

    inserted = False
    if workspace_str not in sys.path:
        sys.path.insert(0, workspace_str)
        inserted = True

    try:
        module = importlib.import_module(module_name)
    finally:
        if inserted and sys.path and sys.path[0] == workspace_str:
            sys.path.pop(0)

    if not hasattr(module, func_name):
        raise AttributeError(f"Module '{module_name}' does not define '{func_name}'.")
    return module, getattr(module, func_name)


def execute_schedule(
    design_id: int,
    schedule_id: int
) -> List[Dict[str, Any]]:
    """
    Execute a schedule for a design (NEW MULTI-FIDELITY APPROACH).

    Dynamically loads the schedule function and executes it.
    The schedule function returns a list of experiment results.

    Args:
        design_id: ID of the design to evaluate
        schedule_id: ID of the schedule to use

    Returns:
        List of experiment results, each containing parameters and outcomes.
        Structure: [{'N': 10, 'batch': 100, 'median_step': 25, ...}, ...]
    """
    # Load the schedule function
    schedule_fn = load_schedule(schedule_id)

    # Execute the schedule
    print(f"Executing schedule_{schedule_id} for design_{design_id}...")
    experiment_results = schedule_fn(design_id=design_id)

    print(f"Schedule completed: {len(experiment_results)} experiments executed")
    return experiment_results
