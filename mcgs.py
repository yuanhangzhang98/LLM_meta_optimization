"""Monte Carlo Graph Search utilities for experiment scheduling.

Refactored to support dynamic objective functions and rebuilt from designs on-demand.
No longer persists to mcgs_graph.json - graph is rebuilt each time with the current objective.
"""

from __future__ import annotations

import math
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any

from database.schema import Design


@dataclass
class GraphNode:
    """
    Node in the Monte Carlo Graph representing a design.

    Note: Fields renamed from experiment_id to design_id for consistency.
    """
    design_id: int
    objective: Optional[float] = None
    visit_count: float = 1.0
    timestamp: Optional[str] = None
    short_name: Optional[str] = None
    parent_edges: List[dict] = field(default_factory=list)  # {"parent_id": int, "weight": float}
    children: List[int] = field(default_factory=list)
    rank_score: float = 0.0
    last_puct: float = 0.0

    def to_dict(self) -> dict:
        return {
            "design_id": self.design_id,
            "objective": self.objective,
            "visit_count": self.visit_count,
            "timestamp": self.timestamp,
            "short_name": self.short_name,
            "parent_edges": self.parent_edges,
            "children": self.children,
            "rank_score": self.rank_score,
            "last_puct": self.last_puct,
        }


class MonteCarloGraph:
    """
    Monte Carlo Graph Search for design ranking and selection.

    Key changes from original:
    - No persistence (no save() method, no mcgs_graph.json)
    - Builds from Design objects instead of JSONL records
    - Supports dynamic objective functions
    - Uses design_id instead of experiment_id
    - Supports objective-specific filtering to only rank designs evaluated
      with the same objective (important for multi-objective optimization)
    """

    def __init__(
        self,
        *,
        c_puct: float = 0.1,
        decay_factor: float = 0.9,
        min_contribution: float = 1e-4,
    ) -> None:
        """
        Initialize Monte Carlo Graph.

        Args:
            c_puct: Exploration constant for PUCT scoring
            decay_factor: Decay factor for visit propagation
            min_contribution: Minimum contribution threshold for propagation
        """
        self.c_puct = c_puct
        self.decay_factor = decay_factor
        self.min_contribution = min_contribution
        self._nodes: Dict[int, GraphNode] = {}

    # ------------------------------------------------------------------ #
    # Graph Building                                                     #
    # ------------------------------------------------------------------ #
    def build_from_designs(
        self,
        designs: List[Design],
        objective_fn: Callable[[List[Dict[str, Any]]], float],
        objective_id: Optional[int] = None
    ) -> None:
        """
        Build the graph from a list of designs with a dynamic objective function.

        Args:
            designs: List of Design objects from the database
            objective_fn: Function that takes experiments list and returns a objective.
                         This allows dynamic objective functions that can change
                         based on research goals.
            objective_id: Optional objective ID to filter designs. If provided, only
                         designs with matching objective_id will be included in the graph.
                         This allows ranking designs evaluated with the same objective.
                         If None, all designs are included (backward compatible).

        Example:
            def my_objective(experiments):
                # Extract raw stats and compute custom objective
                if not experiments:
                    return 0.0
                last_exp = experiments[-1]
                return last_exp.get('scaling_fit', {}).get('power_law_b', 0.0)

            # Include all designs (backward compatible)
            graph.build_from_designs(designs, my_objective)

            # Only include designs evaluated with objective 2
            graph.build_from_designs(designs, my_objective, objective_id=2)
        """
        self._nodes.clear()

        # Filter designs by objective_id if provided
        if objective_id is not None:
            designs = [d for d in designs if d.objective_id == objective_id]

            # Early return if no designs match the objective_id
            if not designs:
                warnings.warn(
                    f"No designs found with objective_id={objective_id}. "
                    "Graph will be empty.",
                    UserWarning
                )
                return

        for design in designs:
            # Compute objective using the objective function
            objective = objective_fn(design.experiments) if design.experiments else None

            # Extract reference weights as parent edges
            references = [
                (rw.design_id, rw.weight)
                for rw in design.reference_weights
            ]

            # Create or update node
            node = GraphNode(
                design_id=design.design_id,
                objective=objective,
                visit_count=1.0,
                timestamp=design.timestamp,
                short_name=design.short_name,
            )
            self._nodes[design.design_id] = node
            self._update_parent_edges(node, references)

        # Build children relationships and propagate visits
        self._refresh_children()
        for design in designs:
            references = [(rw.design_id, rw.weight) for rw in design.reference_weights]
            if references:
                self._propagate_visit_counts(design.design_id, references)

    def add_design(
        self,
        design: Design,
        objective_fn: Callable[[List[Dict[str, Any]]], float]
    ) -> None:
        """
        Add a single design to the existing graph.

        Args:
            design: Design object to add
            objective_fn: Function to compute objective from experiments
        """
        objective = objective_fn(design.experiments) if design.experiments else None
        references = [(rw.design_id, rw.weight) for rw in design.reference_weights]

        node = self._nodes.get(design.design_id)
        if node is None:
            node = GraphNode(design_id=design.design_id)
            self._nodes[design.design_id] = node

        node.objective = objective
        node.timestamp = design.timestamp
        node.short_name = design.short_name
        node.visit_count = max(node.visit_count, 1.0)

        self._update_parent_edges(node, references)
        self._refresh_children()
        self._propagate_visit_counts(design.design_id, references)

    def get_node(self, design_id: int) -> Optional[GraphNode]:
        """Get a node by design ID."""
        return self._nodes.get(int(design_id))

    # ------------------------------------------------------------------ #
    # Ranking                                                            #
    # ------------------------------------------------------------------ #
    def get_ranked_nodes(self, limit: Optional[int] = None) -> List[dict]:
        """
        Get designs ranked by PUCT score (exploitation + exploration).

        Args:
            limit: Maximum number of nodes to return

        Returns:
            List of dicts with design_id, objective, visit_count, rank_score, puct
        """
        scores = self._compute_puct_scores()
        if limit is not None:
            scores = scores[:limit]
        return scores

    def get_node_context(self, design_id: int) -> dict:
        """Get context information for a specific design."""
        node = self._nodes.get(int(design_id))
        if not node:
            return {}
        return {
            "design_id": node.design_id,
            "objective": node.objective,
            "visit_count": node.visit_count,
            "rank_score": node.rank_score,
            "puct": node.last_puct,
            "timestamp": node.timestamp,
        }

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _update_parent_edges(
        self,
        node: GraphNode,
        references: List[Tuple[int, float]]
    ) -> None:
        """Update parent edges for a node."""
        node.parent_edges = []
        for parent_id, weight in references:
            # Ensure parent node exists (even if empty)
            if parent_id not in self._nodes:
                self._nodes[parent_id] = GraphNode(design_id=parent_id)
            node.parent_edges.append({
                "parent_id": parent_id,
                "weight": float(weight)
            })

    def _refresh_children(self) -> None:
        """Rebuild children lists from parent edges."""
        # Clear all children
        for node in self._nodes.values():
            node.children.clear()

        # Rebuild from parent edges
        for node in self._nodes.values():
            for edge in node.parent_edges:
                parent_id = edge["parent_id"]
                parent = self._nodes.get(parent_id)
                if parent is None:
                    continue
                if node.design_id not in parent.children:
                    parent.children.append(node.design_id)

    def _propagate_visit_counts(
        self,
        node_id: int,
        references: List[Tuple[int, float]]
    ) -> None:
        """
        Distribute visit counts upwards using weighted, decayed contributions.

        This implements the key MCGS idea: frequently referenced designs
        get higher visit counts, making them less likely to be explored again
        (favoring exploitation of less-explored promising designs).
        """
        node = self._nodes.get(node_id)
        if not node:
            return

        visited: Dict[int, float] = {}
        queue: deque[Tuple[int, float, int]] = deque(
            (parent_id, weight, 0)
            for parent_id, weight in references
            if weight > 0
        )

        while queue:
            current, weight, depth = queue.popleft()
            if weight <= 0:
                continue

            visited[current] = visited.get(current, 0.0) + weight
            ancestor = self._nodes.get(current)
            if ancestor is None:
                continue

            next_depth = depth + 1
            for edge in ancestor.parent_edges:
                parent_id = edge["parent_id"]
                combined_weight = weight * edge["weight"] * (self.decay_factor ** next_depth)
                if combined_weight < self.min_contribution:
                    continue
                queue.append((parent_id, combined_weight, next_depth))

        # Apply accumulated contributions
        for ancestor_id, contribution in visited.items():
            ancestor_node = self._nodes.get(ancestor_id)
            if ancestor_node:
                ancestor_node.visit_count += contribution

    def _compute_puct_scores(self) -> List[dict]:
        """
        Compute PUCT (Predictor + Upper Confidence Bound) scores for all nodes.

        PUCT = exploitation + exploration
             = rank_score + c_puct * prior * sqrt(total_visits) / (1 + node_visits)

        Higher scores indicate designs that should be examined more closely.
        """
        # Compute rank scores (normalized objectives)
        objectives = [
            (node.objective if node.objective is not None else float("-inf"), node.design_id)
            for node in self._nodes.values()
        ]
        valid_objectives = [item for item in objectives if item[0] != float("-inf")]

        rank_map: Dict[int, float] = {}
        if len(valid_objectives) <= 1:
            # All same rank if ≤1 valid objective
            for _, node_id in objectives:
                rank_map[node_id] = 1.0
        else:
            # Normalize to [0, 1] based on sorted position (descending for minimization)
            # Lower objective → higher rank_score (better)
            sorted_objectives = sorted(valid_objectives, key=lambda x: x[0], reverse=True)
            total = len(sorted_objectives)
            for idx, (_, node_id) in enumerate(sorted_objectives, start=1):
                rank_map[node_id] = (idx - 1) / (total - 1)

        # Assign 0.0 to invalid objectives
        for objective_value, node_id in objectives:
            if objective_value == float("-inf"):
                rank_map.setdefault(node_id, 0.0)

        # Compute PUCT scores
        total_nodes = max(len(self._nodes), 1)
        total_visits = sum(node.visit_count for node in self._nodes.values())
        if total_visits <= 0:
            total_visits = 1.0

        prior = 1.0
        scored: List[dict] = []

        for node in self._nodes.values():
            node.rank_score = rank_map.get(node.design_id, 0.0)
            exploration = self.c_puct * prior * math.sqrt(total_visits) / (1.0 + node.visit_count)
            node.last_puct = node.rank_score + exploration

            scored.append({
                "design_id": node.design_id,
                "objective": node.objective,
                "visit_count": node.visit_count,
                "rank_score": node.rank_score,
                "puct": node.last_puct,
            })

        # Sort by PUCT (descending)
        scored.sort(key=lambda d: d["puct"], reverse=True)
        return scored


def compute_default_objective(experiments: List[Dict[str, Any]]) -> float:
    """
    Placeholder for default objective computation.
    """

    raise NotImplementedError("Default objective function is not implemented.")


__all__ = ["MonteCarloGraph", "GraphNode", "compute_default_objective"]
