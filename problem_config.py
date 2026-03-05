"""Problem configuration loader for research automation system.

This module provides a problem-agnostic interface to load problem-specific
configuration from domain_knowledge/problem_config.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SolverComponent:
    """Specification for a solver component (variable, hyperparameter space, function, etc.)."""
    name: str
    type: str  # "dict", "function", etc.
    description: str
    required: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SolverComponent:
        return cls(
            name=data["name"],
            type=data["type"],
            description=data["description"],
            required=data.get("required", True)
        )


@dataclass
class ProblemConfig:
    """Configuration for a research problem.

    This configuration defines the interface between problem-specific code
    (in domain_knowledge/) and problem-agnostic orchestration code.

    Attributes:
        problem_name: Name of the research problem
        problem_description: Description of the problem
        solver_components: List of solver component specifications
        required_imports: List of required Python imports
        metric_entry_point: Entry point for metric evaluation
        tunable_component_name: Name of the component to be tuned
        baseline_filename: Filename of baseline solver implementation
        framework_module: Module name for the solver framework
        default_schedule_id: Default schedule ID for multi-fidelity optimization (default: 0)
        default_objective_id: Default objective ID for multi-fidelity optimization (default: 0)
        scheduler_frequency: Run scheduler/objective agents every N iterations (default: 5)
    """
    problem_name: str
    problem_description: str
    solver_components: List[SolverComponent]
    required_imports: List[str]
    metric_entry_point: str
    tunable_component_name: str
    baseline_filename: str
    framework_module: str
    default_schedule_id: int = 0
    default_objective_id: int = 0
    scheduler_frequency: int = 5

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> ProblemConfig:
        """Load problem configuration from JSON file.

        Args:
            config_path: Path to problem_config.json. If None, uses
                        domain_knowledge/problem_config.json in project root.

        Returns:
            ProblemConfig instance
        """
        if config_path is None:
            # Default to domain_knowledge/problem_config.json
            root = Path(__file__).parent
            config_path = root / "domain_knowledge" / "problem_config.json"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Problem configuration not found at {config_path}. "
                f"Please ensure domain_knowledge/problem_config.json exists."
            )

        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls(
            problem_name=data["problem_name"],
            problem_description=data["problem_description"],
            solver_components=[
                SolverComponent.from_dict(comp)
                for comp in data["solver_components"]
            ],
            required_imports=data["required_imports"],
            metric_entry_point=data["metric_entry_point"],
            tunable_component_name=data["tunable_component_name"],
            baseline_filename=data.get("baseline_filename", "solver_baseline.py"),
            framework_module=data.get("framework_module", "sat_solver"),
            default_schedule_id=data.get("default_schedule_id", 0),
            default_objective_id=data.get("default_objective_id", 0),
            scheduler_frequency=data.get("scheduler_frequency", 5)
        )

    def get_component_names(self) -> List[str]:
        """Get list of all solver component names."""
        return [comp.name for comp in self.solver_components]

    def get_required_component_names(self) -> List[str]:
        """Get list of required solver component names."""
        return [comp.name for comp in self.solver_components if comp.required]

    def get_component_by_name(self, name: str) -> Optional[SolverComponent]:
        """Get component specification by name."""
        for comp in self.solver_components:
            if comp.name == name:
                return comp
        return None

    def get_import_patterns(self) -> List[str]:
        """Get list of import patterns to check for (e.g., 'import torch', 'from torch')."""
        patterns = []
        for imp in self.required_imports:
            patterns.append(f"import {imp}")
            patterns.append(f"from {imp}")
        return patterns

    @property
    def num_components(self) -> int:
        """Get number of solver components."""
        return len(self.solver_components)


# Singleton instance - loaded once per process
_config_instance: Optional[ProblemConfig] = None


def get_problem_config(reload: bool = False) -> ProblemConfig:
    """Get singleton ProblemConfig instance.

    Args:
        reload: If True, reload configuration from disk

    Returns:
        ProblemConfig instance
    """
    global _config_instance

    if _config_instance is None or reload:
        _config_instance = ProblemConfig.load()

    return _config_instance
