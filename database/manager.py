"""
Database manager for the unified database.json file.

Provides thread-safe operations for reading and writing the database,
replacing the previous scattered JSONL operations.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from threading import RLock

from database.schema import Database, Design, ReferenceWeight, create_timestamp


class DatabaseManager:
    """
    Manager for the unified database.json file.

    Provides atomic read/write operations and high-level database management.
    Thread-safe for concurrent access.
    """

    def __init__(self, database_path: Path):
        """
        Initialize the database manager.

        Args:
            database_path: Path to database.json file
        """
        self.database_path = Path(database_path)
        self._lock = RLock()
        self._db: Optional[Database] = None

        # Ensure database directory exists
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> Database:
        """
        Load the database from disk.

        Returns:
            Database object

        Creates an empty database if file doesn't exist.
        """
        with self._lock:
            if not self.database_path.exists():
                self._db = Database()
                return self._db

            with open(self.database_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self._db = Database.from_dict(data)
            return self._db

    def save(self, db: Optional[Database] = None) -> None:
        """
        Save the database to disk atomically.

        Args:
            db: Database object to save. If None, saves the cached instance.

        Uses temp file + rename pattern for atomic writes.
        """
        with self._lock:
            if db is not None:
                self._db = db
            elif self._db is None:
                raise ValueError("No database to save. Call load() first or provide db parameter.")

            # Write to temporary file first
            temp_path = self.database_path.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(self._db.to_dict(), f, indent=2, ensure_ascii=False)

            # Atomic rename
            temp_path.replace(self.database_path)

    def add_design(
        self,
        design_id: int,
        planner_round_id: int,
        reference_weights: List[ReferenceWeight],
        short_name: str,
        model_name: str,
        code_path: str,
        planner_log: str,
        designer_log: str,
        timestamp: Optional[str] = None,
        schedule_id: int = 0,
        objective_id: int = 0,
        scheduler_log: Optional[str] = None
    ) -> Design:
        """
        Add a new design to the database and save.

        Args:
            design_id: Unique design identifier
            planner_round_id: ID of the planner round that created this design
            reference_weights: List of parent designs with weights
            short_name: Brief name for the design
            model_name: LLM model used
            code_path: Path to solver file
            planner_log: Path to planner conversation log
            designer_log: Path to designer conversation log
            timestamp: ISO timestamp (UTC), auto-generated if None
            schedule_id: Schedule used to generate experiments (default: 0)
            objective_id: Objective used to evaluate design (default: 0)
            scheduler_log: Path to scheduler decision log (optional)

        Returns:
            The created Design object
        """
        if timestamp is None:
            timestamp = create_timestamp()

        design = Design(
            design_id=design_id,
            planner_round_id=planner_round_id,
            reference_weights=reference_weights,
            short_name=short_name,
            model_name=model_name,
            code_path=code_path,
            planner_log=planner_log,
            designer_log=designer_log,
            timestamp=timestamp,
            experiments=[],
            schedule_id=schedule_id,
            objective_id=objective_id,
            scheduler_log=scheduler_log
        )

        with self._lock:
            if self._db is None:
                self.load()

            self._db.add_design(design)
            self.save()

        return design

    def add_experiment_to_design(
        self,
        design_id: int,
        experiment_data: Dict[str, Any]
    ) -> None:
        """
        Add an experiment result to an existing design and save.

        Args:
            design_id: ID of the design to add experiment to
            experiment_data: Problem-specific experiment data
        """
        with self._lock:
            if self._db is None:
                self.load()

            design = self._db.get_design(design_id)
            if design is None:
                raise ValueError(f"Design with ID {design_id} not found")

            design.add_experiment(experiment_data)
            self.save()

    def update_design_metadata(
        self,
        design_id: int,
        schedule_id: Optional[int] = None,
        objective_id: Optional[int] = None,
        scheduler_log: Optional[str] = None
    ) -> None:
        """
        Update metadata for an existing design and save.

        Args:
            design_id: ID of the design to update
            schedule_id: New schedule ID (if provided)
            objective_id: New objective ID (if provided)
            scheduler_log: New scheduler log path (if provided)
        """
        with self._lock:
            if self._db is None:
                self.load()

            design = self._db.get_design(design_id)
            if design is None:
                raise ValueError(f"Design with ID {design_id} not found")

            if schedule_id is not None:
                design.schedule_id = schedule_id
            if objective_id is not None:
                design.objective_id = objective_id
            if scheduler_log is not None:
                design.scheduler_log = scheduler_log

            self.save()

    def get_design(self, design_id: int) -> Optional[Design]:
        """
        Get a design by ID.

        Args:
            design_id: Design identifier

        Returns:
            Design object or None if not found
        """
        with self._lock:
            if self._db is None:
                self.load()
            return self._db.get_design(design_id)

    def get_all_designs(self) -> List[Design]:
        """
        Get all designs in the database.

        Returns:
            List of Design objects
        """
        with self._lock:
            if self._db is None:
                self.load()
            return list(self._db.designs)

    def get_designs_by_planner_round(self, planner_round_id: int) -> List[Design]:
        """
        Get all designs from a specific planner round.

        Args:
            planner_round_id: Planner round identifier

        Returns:
            List of Design objects
        """
        with self._lock:
            if self._db is None:
                self.load()
            return self._db.get_designs_by_planner_round(planner_round_id)

    def get_designs_by_objective(self, objective_id: int) -> List[Design]:
        """
        Get all designs evaluated with a specific objective.

        Args:
            objective_id: Objective identifier

        Returns:
            List of Design objects
        """
        with self._lock:
            if self._db is None:
                self.load()
            return self._db.get_designs_by_objective(objective_id)

    def get_designs_by_schedule(self, schedule_id: int) -> List[Design]:
        """
        Get all designs executed with a specific schedule.

        Args:
            schedule_id: Schedule identifier

        Returns:
            List of Design objects
        """
        with self._lock:
            if self._db is None:
                self.load()
            return self._db.get_designs_by_schedule(schedule_id)

    def get_top_designs(self, n: int, objective_id: Optional[int] = None) -> List[Design]:
        """
        Get top N designs, optionally filtered by objective.

        Note: This returns designs with experiments, but doesn't rank them.
        Ranking should be done using the MCGS module with the objective function.

        Args:
            n: Number of top designs to return
            objective_id: If provided, only return designs with this objective_id

        Returns:
            List of designs with experiments (not ranked, just filtered)
        """
        with self._lock:
            if self._db is None:
                self.load()
            return self._db.get_top_designs(n, objective_id)

    def get_max_design_id(self) -> int:
        """
        Get the maximum design ID (for generating new IDs).

        Returns:
            Max design ID, or -1 if database is empty
        """
        with self._lock:
            if self._db is None:
                self.load()
            return self._db.get_max_design_id()

    def get_next_design_id(self) -> int:
        """
        Get the next available design ID.

        Returns:
            Next design ID (max_id + 1)
        """
        return self.get_max_design_id() + 1

    def filter_designs(
        self,
        predicate: Callable[[Design], bool]
    ) -> List[Design]:
        """
        Filter designs using a custom predicate.

        Args:
            predicate: Function that takes a Design and returns bool

        Returns:
            List of Design objects that match the predicate

        Example:
            # Get designs with more than 2 experiments
            designs = db.filter_designs(lambda d: len(d.experiments) > 2)
        """
        with self._lock:
            if self._db is None:
                self.load()
            return [d for d in self._db.designs if predicate(d)]

    def count_designs(self) -> int:
        """
        Count total number of designs.

        Returns:
            Number of designs in database
        """
        with self._lock:
            if self._db is None:
                self.load()
            return len(self._db.designs)

    def export_to_dict(self) -> Dict[str, Any]:
        """
        Export the entire database as a dictionary.

        Returns:
            Dictionary representation of the database
        """
        with self._lock:
            if self._db is None:
                self.load()
            return self._db.to_dict()

    def import_from_dict(self, data: Dict[str, Any]) -> None:
        """
        Import database from dictionary and save.

        Args:
            data: Dictionary representation of the database

        Warning: This replaces the entire database!
        """
        with self._lock:
            self._db = Database.from_dict(data)
            self.save()

    def clear(self) -> None:
        """
        Clear the entire database.

        Warning: This deletes all data!
        """
        with self._lock:
            self._db = Database()
            self.save()


def _read_json(path: Path) -> Dict[str, Any]:
    """Helper function to read JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    """Helper function to write JSON file."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
