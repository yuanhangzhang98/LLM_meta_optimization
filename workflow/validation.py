"""Sandbox validation for generated code.

Provides functions to test schedule and objective code with dummy/mock data
before using them in production. This catches runtime errors early.
"""

from __future__ import annotations

from typing import Callable, Tuple, Any, Dict, List

from domain_knowledge.experiment_schema import (
    validate_experiment_results,
    get_mock_experiment_data,
)


def validate_schedule_sandbox(
    schedule_fn: Callable,
    test_design_id: int = 0,
) -> Tuple[bool, str]:
    """Test a schedule function with a dummy design_id.

    This runs the schedule function to verify it executes without errors
    and returns properly formatted experiment results.

    Note: This actually executes the schedule, which may run experiments.
    For true sandbox testing without execution, use validate_schedule_structure().

    Args:
        schedule_fn: The schedule function to test
        test_design_id: Design ID to pass to the schedule (default: 0)

    Returns:
        (is_valid, error_message) tuple
    """
    try:
        results = schedule_fn(design_id=test_design_id)
    except TypeError as e:
        # Check if it's a signature issue
        return False, f"Schedule function signature error: {e}"
    except Exception as e:
        return False, f"Schedule execution error: {e}"

    # Validate the returned results
    is_valid, msg = validate_experiment_results(results)
    if not is_valid:
        return False, f"Schedule returned invalid results: {msg}"

    return True, ""


def validate_schedule_structure(schedule_fn: Callable) -> Tuple[bool, str]:
    """Validate schedule function structure without executing it.

    This is a lightweight check that verifies the function is callable
    and has the expected signature, without actually running experiments.

    Args:
        schedule_fn: The schedule function to validate

    Returns:
        (is_valid, error_message) tuple
    """
    import inspect

    if not callable(schedule_fn):
        return False, "Schedule must be callable"

    # Check signature
    try:
        sig = inspect.signature(schedule_fn)
        params = list(sig.parameters.keys())

        # Must accept design_id as first parameter
        if not params or params[0] != "design_id":
            return False, "Schedule function must accept 'design_id' as first parameter"

    except (ValueError, TypeError) as e:
        return False, f"Could not inspect schedule signature: {e}"

    return True, ""


def validate_objective_sandbox(objective_fn: Callable) -> Tuple[bool, str]:
    """Test an objective function with mock experiment data.

    This runs the objective function with predefined mock data to verify
    it executes without errors and returns a valid numeric result.

    Args:
        objective_fn: The objective function to test

    Returns:
        (is_valid, error_message) tuple
    """
    mock_data = get_mock_experiment_data()

    try:
        result = objective_fn(mock_data)
    except TypeError as e:
        return False, f"Objective function signature error: {e}"
    except KeyError as e:
        return False, f"Objective function missing field access: {e}"
    except Exception as e:
        return False, f"Objective execution error: {e}"

    # Validate return type
    if not isinstance(result, (int, float)):
        return False, f"Objective must return a number, got {type(result).__name__}"

    # Check for NaN/Inf
    import math
    if math.isnan(result):
        return False, "Objective returned NaN"
    if math.isinf(result):
        return False, "Objective returned Inf"

    return True, ""


def validate_objective_structure(objective_fn: Callable) -> Tuple[bool, str]:
    """Validate objective function structure without executing it.

    This is a lightweight check that verifies the function is callable
    and has the expected signature.

    Args:
        objective_fn: The objective function to validate

    Returns:
        (is_valid, error_message) tuple
    """
    import inspect

    if not callable(objective_fn):
        return False, "Objective must be callable"

    # Check signature
    try:
        sig = inspect.signature(objective_fn)
        params = list(sig.parameters.keys())

        # Must accept experiment_results as first parameter
        if not params or params[0] != "experiment_results":
            return False, "Objective function must accept 'experiment_results' as first parameter"

    except (ValueError, TypeError) as e:
        return False, f"Could not inspect objective signature: {e}"

    return True, ""


__all__ = [
    "validate_schedule_sandbox",
    "validate_schedule_structure",
    "validate_objective_sandbox",
    "validate_objective_structure",
]
