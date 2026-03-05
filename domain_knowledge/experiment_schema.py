"""Fixed experiment result schema for multi-fidelity optimization.

All schedules MUST return experiment dicts conforming to this schema.
This ensures consistent data contracts between schedules and objectives.
"""

from typing import Dict, List, Tuple, Any

# Required fields that all experiment results must contain
REQUIRED_FIELDS = ["N", "batch", "median_step", "unsolved_fraction", "max_steps"]

# Optional fields that may be present
OPTIONAL_FIELDS = ["mean_step", "std_step", "ratio"]


def validate_experiment_result(result: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate that a single experiment result has all required fields.

    Args:
        result: A single experiment result dict

    Returns:
        (is_valid, error_message) tuple
    """
    if not isinstance(result, dict):
        return False, f"Result must be a dict, got {type(result).__name__}"

    missing = [field for field in REQUIRED_FIELDS if field not in result]
    if missing:
        return False, f"Missing required fields: {missing}"

    return True, ""


def validate_experiment_results(results: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """Validate a list of experiment results.

    Args:
        results: List of experiment result dicts

    Returns:
        (is_valid, error_message) tuple
    """
    if not isinstance(results, list):
        return False, f"Results must be a list, got {type(results).__name__}"

    if len(results) == 0:
        return False, "Results list is empty"

    for i, result in enumerate(results):
        is_valid, msg = validate_experiment_result(result)
        if not is_valid:
            return False, f"Result {i}: {msg}"

    return True, ""


def get_mock_experiment_data() -> List[Dict[str, Any]]:
    """Get mock experiment data for sandbox validation.

    Returns:
        List of mock experiment results conforming to the schema
    """
    return [
        {
            "N": 10,
            "batch": 100,
            "median_step": 25,
            "unsolved_fraction": 0.0,
            "max_steps": 50,
        },
        {
            "N": 20,
            "batch": 100,
            "median_step": 40,
            "unsolved_fraction": 0.05,
            "max_steps": 100,
        },
        {
            "N": 40,
            "batch": 100,
            "median_step": 75,
            "unsolved_fraction": 0.1,
            "max_steps": 200,
        },
    ]


__all__ = [
    "REQUIRED_FIELDS",
    "OPTIONAL_FIELDS",
    "validate_experiment_result",
    "validate_experiment_results",
    "get_mock_experiment_data",
]
