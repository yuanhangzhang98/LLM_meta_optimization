"""Utilities for rendering prompt templates with project context.

Refactored to use DatabaseManager and support dynamic objective functions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, List, Sequence, Callable

from database.manager import DatabaseManager
from database.schema import Design
from mcgs import MonteCarloGraph, compute_default_objective
from problem_config import get_problem_config
import workspace_config


_PLACEHOLDER_PATTERN = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _safe_format(template: str, context: Mapping[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in context:
            raise KeyError(key)
        return str(context[key])

    formatted = _PLACEHOLDER_PATTERN.sub(replace, template)
    return formatted.replace("{{", "{").replace("}}", "}")


def _indent(text: str, prefix: str) -> str:
    return "\n".join(f"{prefix}{line}" for line in text.splitlines() or [""])


def _read_text(path: Path, default: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return default


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    path: Path

    def render(self, context: Mapping[str, Any]) -> str:
        template = _read_text(self.path)
        try:
            return _safe_format(template, context)
        except KeyError as exc:
            missing = exc.args[0]
            raise KeyError(f"Missing placeholder '{missing}' for prompt '{self.name}'") from exc


class PromptRepository:
    """Load prompt templates from disk."""

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        if not self._base_dir.exists():
            raise FileNotFoundError(f"Prompt directory not found: {base_dir}")

    def load(self, relative_path: str) -> PromptTemplate:
        path = (self._base_dir / relative_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        return PromptTemplate(name=relative_path, path=path)


class PromptFormatter:
    """Render prompts with pre-populated research context.

    Refactored to use DatabaseManager instead of JSONL files.
    Supports dynamic objective functions for MCGS ranking.
    """

    def __init__(
        self,
        prompt_dir: Path,
        knowledge_dir: Path,
        database_manager: DatabaseManager,
        *,
        objective_fn: Optional[Callable[[List[Dict[str, Any]]], float]] = None,
        current_objective_id: int = 0,
        num_designers: int = 2,
        max_design_summaries: int = 50,
        max_design_detail: int = 6,
        max_designer_references: int = 3,
    ) -> None:
        """
        Initialize prompt formatter.

        Args:
            prompt_dir: Directory containing prompt templates
            knowledge_dir: Directory containing domain knowledge files
            database_manager: DatabaseManager instance
            objective_fn: Function to compute metrics from experiments (for MCGS)
            current_objective_id: Current objective ID being used (default: 0)
            num_designers: Number of designer agents per planner round
            max_design_summaries: Maximum designs to include in summaries
            max_design_detail: Maximum designs to show in detail
            max_designer_references: Maximum reference designs per designer
        """
        self._repo = PromptRepository(prompt_dir)
        self._knowledge_dir = knowledge_dir
        self._db = database_manager
        self._project_root = prompt_dir.resolve().parent
        self._objective_fn = objective_fn or compute_default_objective
        self._current_objective_id = current_objective_id
        self._num_designers = num_designers
        self._max_design_summaries = max_design_summaries
        self._max_design_detail = max_design_detail
        self._max_designer_references = max_designer_references
        self._recent_summary_count = 5

        # Build MCGS graph from current database
        self._graph = MonteCarloGraph()
        self._rebuild_graph()

        # Build default context
        self._defaults = self._build_defaults()

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def format(self, relative_path: str, extra_context: Optional[Mapping[str, Any]] = None) -> str:
        """Render a prompt template with context."""
        template = self._repo.load(relative_path)
        context = dict(self._defaults)
        if extra_context:
            context.update(extra_context)
        return template.render(context)

    def base_context(self) -> Dict[str, Any]:
        """Return a copy of the default context (useful for inspection/tests)."""
        return dict(self._defaults)

    def rebuild_context(
        self,
        objective_fn: Optional[Callable[[List[Dict[str, Any]]], float]] = None,
        new_objective_id: Optional[int] = None,
        filter_by_objective: bool = False
    ) -> None:
        """
        Rebuild the context with a new objective function.

        Useful when the objective function changes during research.

        Args:
            objective_fn: New objective function to use (if provided)
            new_objective_id: New objective ID to track (if provided)
            filter_by_objective: If True and new_objective_id is provided, filter MCGS graph
                                to only include designs from that objective (default: False)
        """
        if objective_fn is not None:
            self._objective_fn = objective_fn
        if new_objective_id is not None:
            self._current_objective_id = new_objective_id

        # Rebuild graph with optional filtering
        if filter_by_objective and new_objective_id is not None:
            self._rebuild_graph(objective_id=new_objective_id)
        else:
            self._rebuild_graph()

        self._defaults = self._build_defaults()


    # ------------------------------------------------------------------ #
    # Context construction                                               #
    # ------------------------------------------------------------------ #
    def _rebuild_graph(self, objective_id: Optional[int] = None) -> None:
        """
        Rebuild MCGS graph from database with current objective function.

        Args:
            objective_id: If provided, filter designs to only those associated with this objective.
                         If None, uses all designs (default behavior).
        """
        designs = self._db.get_all_designs()
        # Use MCGS built-in filtering by objective_id
        self._graph.build_from_designs(designs, self._objective_fn, objective_id=objective_id)

    def _build_defaults(self) -> Dict[str, Any]:
        """Build default context dict for prompt templates."""
        config = get_problem_config()
        main_research_context = _read_text(self._knowledge_dir / "main_research_context.md")
        base_solver_code = _read_text(self._knowledge_dir / "solver_baseline.py")
        baseline_objective_code = _read_text(self._knowledge_dir / "objective_baseline.py")
        baseline_schedule_code = _read_text(self._knowledge_dir / "schedule_baseline.py")
        experiment_code = _read_text(self._knowledge_dir / "experiment.py")
        reference_designs_text, metrics = self._summarise_designs()

        # Build component-related context from config
        component_names = config.get_component_names()
        num_components = config.num_components
        component_descriptions = "\n".join(
            f"- **{comp.name}** ({comp.type}): {comp.description}"
            for comp in config.solver_components
        )

        defaults: Dict[str, Any] = {
            # Core research context
            "main_research_context": main_research_context.strip(),
            "base_solver_code": base_solver_code.strip(),
            "baseline_objective_code": baseline_objective_code.strip(),
            "baseline_schedule_code": baseline_schedule_code.strip(),
            "experiment_code": experiment_code.strip(),

            # Planner context (filled dynamically by workflow)
            "planner_context": "",

            # Design/experiment summaries and details
            "reference_experiments": reference_designs_text,
            "experiment_summaries": reference_designs_text,
            "design_details": "No design details requested yet.",

            # Metrics and performance
            "total_experiments": metrics.get("total_designs", 0),
            "best_performance": metrics.get("best_metric"),
            "recent_summary": metrics.get("recent_summary", "No designs logged."),

            # Agent configuration
            "DESIGNER_AGENT_COUNT": self._num_designers,
            "MAX_EXPERIMENT_SUMMARY": self._max_design_summaries,
            "MAX_EXPERIMENT_DETAIL": self._max_design_detail,
            "MAX_DESIGNER_REFERENCE": self._max_designer_references,

            # Error handling (filled dynamically)
            "error_message": "",
            "full_traceback": "",

            # Component configuration
            "num_components": num_components,
            "component_names": ", ".join(component_names),
            "component_descriptions": component_descriptions,
            "baseline_filename": config.baseline_filename,
            "framework_module": config.framework_module,

            # Objective and schedule IDs (filled dynamically by workflow)
            "current_objective_id": self._current_objective_id,
            "current_schedule_id": 0,  # Default schedule ID

            # Objective-specific summaries (filled dynamically by objective workflow)
            "existing_objective_summary": "",
            "existing_schedule_summary": "",
            "recent_experiments_objectives_summary": "",
            "objective_description": "",
            "schedule_description": "",

            # Designer analysis context (filled dynamically)
            "current_fidelity": "",
            "experiment_results": "",
            "objective_value": "",
        }
        return defaults

    def _summarise_designs(self) -> tuple[str, Dict[str, Any]]:
        """Generate summary of all designs ranked by MCGS PUCT score."""
        designs = self._db.get_all_designs()

        if not designs:
            return "No prior designs recorded.", {
                "total_designs": 0,
                "best_metric": None,
                "recent_summary": "No designs logged.",
            }

        ranked = self._graph.get_ranked_nodes(self._max_design_summaries)
        summaries: List[str] = []
        best_metric = None

        for entry in ranked:
            design_id = entry["design_id"]
            metric_value = entry.get("objective")

            design = self._db.get_design(design_id)
            if design is None:
                continue

            short_name = design.short_name or f"Design {design_id}"
            visit_count = entry.get("visit_count", 0.0) or 0.0
            puct_value = entry.get("puct", 0.0) or 0.0

            summaries.append(
                f"- Design {design_id}: short_name='{short_name}' | "
                f"metric={metric_value} | visits={visit_count:.2f} | ucb={puct_value:.3f}"
            )

            if metric_value is not None:
                if best_metric is None or metric_value < best_metric:  # Minimization: lower is better
                    best_metric = metric_value

        # Recent summary
        recent_parts = []
        recent_designs = designs[-self._recent_summary_count:]
        for design in recent_designs:
            design_id = design.design_id
            metric_value = self._objective_fn(design.experiments) if design.experiments else None
            context = self._graph.get_node_context(design_id)
            ucb_value = context.get("puct", 0.0)
            short_name = design.short_name or f"Design {design_id}"
            recent_parts.append(f"{design_id}:{short_name} (metric={metric_value}, ucb={ucb_value:.3f})")

        recent_summary = ", ".join(recent_parts) if recent_parts else "No designs logged."

        return "\n".join(summaries), {
            "total_designs": len(designs),
            "best_metric": best_metric,
            "recent_summary": recent_summary,
        }

    def _load_metadata(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Load objective and schedule metadata from JSON files."""
        obj_path = workspace_config.get_objectives_dir() / "metadata.json"
        sched_path = workspace_config.get_schedules_dir() / "metadata.json"

        obj_meta = _load_json(obj_path) if obj_path.is_file() else {}
        sched_meta = _load_json(sched_path) if sched_path.is_file() else {}

        return obj_meta, sched_meta

    def render_design_details(self, design_ids: Sequence[int]) -> str:
        """
        Render detailed information for specific designs.
        """
        if not design_ids:
            return "No design details requested."

        obj_metadata, sched_metadata = self._load_metadata()

        # Show current objective once at the top (same for all designs)
        obj_desc = obj_metadata.get(str(self._current_objective_id), {}).get("description", "Unknown")
        lines: List[str] = [
            f"Current objective (id={self._current_objective_id}): {obj_desc}",
            ""
        ]

        node_scores = {entry["design_id"]: entry for entry in self._graph.get_ranked_nodes()}

        for design_id in design_ids:
            design = self._db.get_design(design_id)
            if design is None:
                lines.append(f"- Design {design_id}: not found in database.")
                continue

            lines.append(f"- Design {design_id}:")

            # Compute metric from experiments
            metric_value = self._objective_fn(design.experiments) if design.experiments else None
            node_context = self._graph.get_node_context(design_id)
            ucb_value = (node_scores.get(design_id) or {}).get("puct")
            visit_count = node_context.get("visit_count")

            lines.append(f"  short_name: {design.short_name}")
            lines.append(f"  metric: {metric_value}")
            lines.append(f"  ucb: {ucb_value}")
            lines.append(f"  visit_count: {visit_count}")
            lines.append(f"  timestamp: {design.timestamp}")
            lines.append(f"  model_name: {design.model_name}")
            lines.append(f"  planner_round_id: {design.planner_round_id}")

            # Schedule info
            lines.append(f"  schedule_id: {design.schedule_id}")
            sched_desc = sched_metadata.get(str(design.schedule_id), {}).get("description", "Unknown")
            lines.append(f"  schedule_description: {sched_desc}")

            # Reference weights
            if design.reference_weights:
                ref_strs = [f"{rw.design_id} (weight={rw.weight:.2f})" for rw in design.reference_weights]
                lines.append(f"  reference_weights: {', '.join(ref_strs)}")
            else:
                lines.append("  reference_weights: none")

            # Experiments
            if design.experiments:
                lines.append(f"  experiments ({len(design.experiments)} total):")
                for i, exp in enumerate(design.experiments):
                    lines.append(f"    experiment[{i}]:")
                    for key, value in exp.items():
                        if isinstance(value, dict):
                            lines.append(f"      {key}:")
                            for sub_key, sub_value in value.items():
                                lines.append(f"        {sub_key}: {sub_value}")
                        elif isinstance(value, list):
                            lines.append(f"      {key}: {value}")
                        else:
                            lines.append(f"      {key}: {value}")
            else:
                lines.append("  experiments: none")

            # Solver code
            solver_file = Path(design.code_path)
            if not solver_file.is_absolute():
                solver_file = self._project_root / solver_file

            if solver_file.is_file():
                try:
                    solver_code = _read_text(solver_file)
                    lines.append(f"  code_path: {design.code_path}")
                    lines.append("  solver_code:")
                    lines.append(_indent(solver_code, "    "))
                except FileNotFoundError:
                    lines.append(f"  code_path: {design.code_path} (file not found)")
            else:
                lines.append(f"  code_path: {design.code_path} (file not found)")

        return "\n".join(lines)


__all__ = ["PromptFormatter", "PromptTemplate", "PromptRepository"]
