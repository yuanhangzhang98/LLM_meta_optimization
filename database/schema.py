"""
Database schema definitions for the LLM automated discovery system.

This module defines the structure of the unified database.json file.
The schema is problem-agnostic and supports multi-fidelity experiments
with dynamic objective functions.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class ReferenceWeight:
    """Reference to a parent design with normalized weight."""
    design_id: int
    weight: float

    def __post_init__(self):
        if not 0 <= self.weight <= 1:
            raise ValueError(f"Weight must be between 0 and 1, got {self.weight}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "design_id": self.design_id,
            "weight": self.weight
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReferenceWeight":
        return cls(
            design_id=data["design_id"],
            weight=data["weight"]
        )


@dataclass
class Design:
    """
    A design represents a solver architecture created by the designer agent.

    Each design can have multiple experiments at different fidelity levels,
    which are added incrementally as the scheduler agent schedules them.
    """
    design_id: int
    planner_round_id: int
    reference_weights: List[ReferenceWeight]
    short_name: str
    model_name: str
    code_path: str  # Path to solver file (e.g., "solvers/solver_5.py")
    planner_log: str  # Path to planner LLM conversation log
    designer_log: str  # Path to designer LLM conversation log
    timestamp: str  # ISO format timestamp (UTC)
    experiments: List[Dict[str, Any]] = field(default_factory=list)
    schedule_id: int = 0  # Which schedule was used to generate experiments
    objective_id: int = 0  # Which objective was used to evaluate this design
    scheduler_log: Optional[str] = None  # Path to scheduler decision log (if applicable)

    def __post_init__(self):
        # Validate reference weights sum to 1.0 (within tolerance)
        if self.reference_weights:
            total_weight = sum(rw.weight for rw in self.reference_weights)
            if not (0.99 <= total_weight <= 1.01):
                raise ValueError(
                    f"Reference weights must sum to 1.0, got {total_weight}"
                )

    def add_experiment(self, experiment_data: Dict[str, Any]) -> None:
        """
        Add an experiment result to this design.

        Args:
            experiment_data: Problem-specific experiment data. The structure
                           is flexible and determined by the problem domain.
                           Common fields might include:
                           - fidelity_level: int
                           - batch: int
                           - Any raw statistics collected during the experiment
        """
        self.experiments.append(experiment_data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "design_id": self.design_id,
            "planner_round_id": self.planner_round_id,
            "reference_weights": [rw.to_dict() for rw in self.reference_weights],
            "short_name": self.short_name,
            "model_name": self.model_name,
            "code_path": self.code_path,
            "planner_log": self.planner_log,
            "designer_log": self.designer_log,
            "timestamp": self.timestamp,
            "experiments": self.experiments,
            "schedule_id": self.schedule_id,
            "objective_id": self.objective_id,
        }
        if self.scheduler_log is not None:
            result["scheduler_log"] = self.scheduler_log
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Design":
        """Create Design from dictionary."""
        reference_weights = [
            ReferenceWeight.from_dict(rw) for rw in data["reference_weights"]
        ]
        return cls(
            design_id=data["design_id"],
            planner_round_id=data["planner_round_id"],
            reference_weights=reference_weights,
            short_name=data["short_name"],
            model_name=data["model_name"],
            code_path=data["code_path"],
            planner_log=data["planner_log"],
            designer_log=data["designer_log"],
            timestamp=data["timestamp"],
            experiments=data.get("experiments", []),
            schedule_id=data.get("schedule_id", 0),
            objective_id=data.get("objective_id", 0),
            scheduler_log=data.get("scheduler_log")
        )


@dataclass
class Database:
    """
    The unified database structure.

    Contains all designs and their associated experiments.
    Replaces the previous multi-file JSONL approach.
    """
    designs: List[Design] = field(default_factory=list)

    def get_design(self, design_id: int) -> Optional[Design]:
        """Get a design by ID."""
        for design in self.designs:
            if design.design_id == design_id:
                return design
        return None

    def get_designs_by_planner_round(self, planner_round_id: int) -> List[Design]:
        """Get all designs from a specific planner round."""
        return [
            design for design in self.designs
            if design.planner_round_id == planner_round_id
        ]

    def get_designs_by_objective(self, objective_id: int) -> List[Design]:
        """Get all designs evaluated with a specific objective."""
        return [
            design for design in self.designs
            if design.objective_id == objective_id
        ]

    def get_designs_by_schedule(self, schedule_id: int) -> List[Design]:
        """Get all designs executed with a specific schedule."""
        return [
            design for design in self.designs
            if design.schedule_id == schedule_id
        ]

    def get_top_designs(self, n: int, objective_id: Optional[int] = None) -> List[Design]:
        """Get top N designs, optionally filtered by objective.

        Note: This returns designs with experiments, but doesn't rank them.
        Ranking should be done using the MCGS module with the objective function.

        Args:
            n: Number of top designs to return
            objective_id: If provided, only return designs with this objective_id

        Returns:
            List of designs with experiments (not ranked, just filtered)
        """
        designs = self.designs
        if objective_id is not None:
            designs = [d for d in designs if d.objective_id == objective_id]

        # Filter to designs that have experiments
        designs_with_experiments = [d for d in designs if d.experiments]

        # Return first n (caller should rank them using MCGS + objective)
        return designs_with_experiments[:n]

    def get_max_design_id(self) -> int:
        """Get the maximum design ID (for generating new IDs)."""
        if not self.designs:
            return -1
        return max(design.design_id for design in self.designs)

    def add_design(self, design: Design) -> None:
        """Add a new design to the database."""
        # Check for duplicate design_id
        if self.get_design(design.design_id) is not None:
            raise ValueError(f"Design with ID {design.design_id} already exists")
        self.designs.append(design)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "designs": [design.to_dict() for design in self.designs]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Database":
        """Create Database from dictionary."""
        designs = [Design.from_dict(d) for d in data.get("designs", [])]
        return cls(designs=designs)


def create_timestamp() -> str:
    """Create ISO format timestamp in UTC."""
    return datetime.utcnow().isoformat() + "Z"


def validate_reference_weights(weights: List[ReferenceWeight]) -> bool:
    """
    Validate that reference weights sum to 1.0.

    Returns:
        True if valid (or empty list), raises ValueError otherwise.
    """
    if not weights:
        return True

    total = sum(w.weight for w in weights)
    if not (0.99 <= total <= 1.01):
        raise ValueError(f"Reference weights must sum to 1.0, got {total}")

    return True
