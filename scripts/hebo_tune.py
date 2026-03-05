"""Hyper-parameter optimisation using HEBO integrated with the new database system.

Updated for new database structure with DatabaseManager and multi-fidelity support.
Prepared for dynamic objective functions in objectives/objective_M.py.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

import numpy as np
import pandas as pd
import torch
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import workspace_config
from consensus_objective import create_consensus_objective
from database.manager import DatabaseManager
from database.schema import ReferenceWeight, create_timestamp
from llm_client import LLMClient
from problem_config import get_problem_config
from prompt_formatter import PromptFormatter
from solver_writer import read_solver_code, write_solver_file, load_solver_module
import importlib

GPU_AVAILABLE = torch.cuda.is_available()
DEFAULT_TENSOR_TYPE = torch.cuda.FloatTensor if GPU_AVAILABLE else torch.FloatTensor
CPU_TENSOR_TYPE = torch.FloatTensor
torch.set_default_tensor_type(DEFAULT_TENSOR_TYPE)


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------- #
# Schedule loading                                                            #
# --------------------------------------------------------------------------- #
def _load_fidelity_schedule(schedule_id: int, fidelity: str = "medium") -> Callable:
    """Load a specific fidelity schedule function.

    Args:
        schedule_id: Schedule module ID
        fidelity: Fidelity level ('low', 'medium', 'high')

    Returns:
        Schedule function with signature: (design_id: int) -> List[Dict]
    """
    from schedule_loader import load_schedule_fidelities
    fidelities = load_schedule_fidelities(schedule_id)
    return fidelities[fidelity]


# --------------------------------------------------------------------------- #
# Solver file operations                                                      #
# --------------------------------------------------------------------------- #
def load_solver_code(parent_solver_id: int, solvers_dir: Path = None) -> str:
    """Load solver code from parent solver file."""
    if solvers_dir is None:
        solvers_dir = workspace_config.get_solvers_dir()
    return read_solver_code(parent_solver_id, solvers_dir)


def import_hyper_space_from_solver(parent_solver_id: int, solvers_dir: Path = None) -> Dict[str, Dict[str, Any]]:
    """Import tunable component (e.g., HYPER_SPACE) from parent solver module."""
    config = get_problem_config()
    if solvers_dir is None:
        solvers_dir = workspace_config.get_solvers_dir()
    solver_module = load_solver_module(parent_solver_id, solvers_dir)
    tunable_component = getattr(solver_module, config.tunable_component_name)
    return tunable_component


def update_hyper_space_in_code(solver_code: str, new_params: Dict[str, Any]) -> str:
    """
    Update tunable component defaults in solver source code using regex.

    Preserves all other code exactly as-is (comments, formatting, etc.).

    Args:
        solver_code: Original solver source code
        new_params: Dict of parameter name -> new default value

    Returns:
        Updated solver code with new tunable component defaults
    """
    config = get_problem_config()
    tunable_name = config.tunable_component_name

    # Find tunable component block (e.g., HYPER_SPACE = {...})
    component_pattern = rf'({tunable_name}\s*=\s*\{{)(.*?)(\n\}})'
    match = re.search(component_pattern, solver_code, re.DOTALL)

    if not match:
        raise ValueError(f"{tunable_name} not found in solver code")

    prefix, hyper_space_body, suffix = match.groups()

    # Update each parameter's default value
    for param_name, new_value in new_params.items():
        # Pattern: "param": dict(...default=OLD_VALUE...)
        # Matches: "alpha": dict(type="...", default=1.5, low=...)
        param_pattern = rf'("{param_name}":\s*dict\([^)]*\bdefault\s*=\s*)([^,\)]+)'

        def replace_default(m):
            # Format value appropriately
            if isinstance(new_value, float):
                value_str = f"{new_value}"
            elif isinstance(new_value, int):
                value_str = f"{new_value}"
            elif isinstance(new_value, str):
                value_str = f'"{new_value}"'
            else:
                value_str = str(new_value)
            return f'{m.group(1)}{value_str}'

        hyper_space_body = re.sub(param_pattern, replace_default, hyper_space_body)

    # Reconstruct code
    new_hyper_space = prefix + hyper_space_body + suffix
    updated_code = solver_code[:match.start()] + new_hyper_space + solver_code[match.end():]

    return updated_code


def save_tuned_solver(
    parent_solver_code: str,
    best_params: Dict[str, Any],
    new_solver_id: int,
    solvers_dir: Path = None
) -> Path:
    """
    Save tuned solver file with updated tunable component defaults.

    Returns path to new solver file.
    """
    if solvers_dir is None:
        solvers_dir = workspace_config.get_solvers_dir()

    # Update tunable component defaults in code
    tuned_code = update_hyper_space_in_code(parent_solver_code, best_params)

    # Write to solvers directory with new ID
    solver_path = write_solver_file(
        solver_code=tuned_code,
        solver_id=new_solver_id,
        solvers_dir=solvers_dir,
        validate=True,
        overwrite=False  # Should not overwrite - new solver ID
    )

    return solver_path


# --------------------------------------------------------------------------- #
# HEBO optimization utilities                                                 #
# --------------------------------------------------------------------------- #
def convert_to_design_space(hyper_space: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    space_cfg: List[Dict[str, Any]] = []
    for name, cfg in hyper_space.items():
        entry: Dict[str, Any] = {"name": name}
        hp_type = cfg.get("type")
        if hp_type in {"uniform", "num"}:
            entry.update({"type": "num", "lb": float(cfg["low"]), "ub": float(cfg["high"])})
        elif hp_type == "log_uniform":
            entry.update({"type": "pow", "lb": float(cfg["low"]), "ub": float(cfg["high"])})
        elif hp_type in {"int", "integer"}:
            entry.update({"type": "int", "lb": int(cfg["low"]), "ub": int(cfg["high"])})
        elif hp_type == "categorical":
            entry.update({"type": "cat", "categories": list(cfg.get("choices") or cfg.get("categories"))})
        elif hp_type == "bool":
            entry.update({"type": "bool"})
        else:
            raise ValueError(f"Unsupported hyper-parameter type: {hp_type}")
        space_cfg.append(entry)
    return space_cfg


def row_to_params(row: pd.Series, hyper_space: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for name, cfg in hyper_space.items():
        hp_type = cfg.get("type")
        value = row[name]
        if hp_type in {"uniform", "num", "log_uniform"}:
            params[name] = float(value)
        elif hp_type in {"int", "integer"}:
            params[name] = int(round(value))
        elif hp_type == "categorical":
            params[name] = value
        elif hp_type == "bool":
            params[name] = bool(value)
        else:
            params[name] = value
    return params


def run_metric_with_solver(
    parent_solver_id: int,
    run_index: int,
    hp: Dict[str, Any],
    temp_solvers_dir: Path,
    db: DatabaseManager,
    schedule_id: int = 0,
    promote_medium_threshold: float = 0.5,
    promote_high_threshold: float = 0.1,
) -> tuple[List[Dict[str, Any]], Path, str]:
    """
    Run multi-fidelity experiment evaluation with dynamic promotion.

    Creates temporary solver with updated tunable component defaults, then executes
    using progressive fidelity levels with consensus-based promotion.

    Promotion criteria:
    - Start at low fidelity
    - Promote to medium if consensus_objective < promote_medium_threshold (top 50%)
    - Promote to high if consensus_objective < promote_high_threshold (top 10%)

    Returns:
        (experiment_results, solver_path, final_fidelity) - List of experiment dicts,
        solver path, and final fidelity level reached
    """
    # Read parent solver code
    parent_code = read_solver_code(parent_solver_id, solvers_dir=workspace_config.get_solvers_dir())

    # Update tunable component defaults in code
    updated_code = update_hyper_space_in_code(parent_code, hp)

    # Write temporary solver file with run_index as ID
    temp_solver_path = write_solver_file(
        solver_code=updated_code,
        solver_id=run_index,
        solvers_dir=temp_solvers_dir,
        validate=True,
        overwrite=True
    )

    # Multi-fidelity loop with promotion
    fidelity_sequence = ['low', 'medium', 'high']
    all_experiments = []
    final_fidelity = 'low'

    for fidelity_idx, fidelity_name in enumerate(fidelity_sequence):
        # Load and execute schedule at current fidelity
        schedule_fn = _load_fidelity_schedule(schedule_id, fidelity_name)
        prior = all_experiments if fidelity_idx > 0 else None
        fid_experiments = schedule_fn(design_id=run_index, prior_experiments=prior)

        # Tag experiments with fidelity level
        for exp in fid_experiments:
            exp['fidelity'] = fidelity_name
        all_experiments.extend(fid_experiments)
        final_fidelity = fidelity_name

        # Check promotion (skip if at highest fidelity)
        if fidelity_idx < len(fidelity_sequence) - 1:
            consensus_fn, _ = create_consensus_objective(db)
            consensus_score = consensus_fn(all_experiments)

            threshold = promote_medium_threshold if fidelity_name == 'low' else promote_high_threshold
            if consensus_score >= threshold:
                # Don't promote - not in top percentile
                break

    return all_experiments, temp_solver_path, final_fidelity


# --------------------------------------------------------------------------- #
# Main HEBO tuning function                                                   #
# --------------------------------------------------------------------------- #
def tune(
    parent_solver_id: int,
    max_iter: int,
    suggestions: int,
    db: DatabaseManager,
    keep_temp_solvers: bool = False,
    schedule_id: int = 0,
    promote_medium_threshold: float = 0.5,
    promote_high_threshold: float = 0.1,
    parent_metric: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Tune hyperparameters using HEBO optimization with dynamic fidelity promotion.

    Uses consensus objective for both fidelity promotion decisions and metric computation.

    Args:
        parent_solver_id: ID of parent solver to tune
        max_iter: Number of HEBO iterations
        suggestions: Number of suggestions per iteration
        db: Database manager (required for consensus objective evaluation)
        keep_temp_solvers: Whether to keep temporary solver files for inspection
        schedule_id: Schedule module ID
        promote_medium_threshold: Threshold to promote low→medium (default 0.5, top 50%)
        promote_high_threshold: Threshold to promote medium→high (default 0.1, top 10%)

    Returns:
        Summary dict with best params, metric, experiment data, paths, etc.
    """

    # 1. Load parent solver code
    parent_code = load_solver_code(parent_solver_id, solvers_dir=workspace_config.get_solvers_dir())

    # 2. Import tunable component from parent solver
    hyper_space = import_hyper_space_from_solver(parent_solver_id, solvers_dir=workspace_config.get_solvers_dir())

    # 3. Convert to HEBO design space
    space_cfg = convert_to_design_space(hyper_space)
    space = DesignSpace().parse(space_cfg)

    # 4. Move space tensors to CPU for HEBO
    cpu_device = torch.device("cpu")
    for attr in ("opt_lb", "opt_ub", "opt_lb_raw", "opt_ub_raw"):
        if hasattr(space, attr):
            tensor = getattr(space, attr)
            try:
                setattr(space, attr, tensor.to(cpu_device))
            except AttributeError:
                pass
    try:
        space.sample_device = cpu_device
    except AttributeError:
        pass

    optimizer = HEBO(space, rand_sample=False)

    # 4b. Warm-start optimizer with parent's default params and metric
    if parent_metric is not None:
        default_params = {name: cfg["default"] for name, cfg in hyper_space.items()}
        warmstart_df = pd.DataFrame([default_params])
        warmstart_obj = np.array([[parent_metric]])
        optimizer.observe(warmstart_df, warmstart_obj)
        print(f"[HEBO] Warm-started with parent metric: {parent_metric:.4f}")

    # 5. Use main solvers directory for trial solver files
    # (Required so framework can import from solvers.solver_{id})
    temp_solvers_dir = workspace_config.get_solvers_dir()

    history: List[Dict[str, Any]] = []
    best_metric = float("inf")  # Minimize: start with +inf
    best_params: Dict[str, Any] = {}
    best_experiment_data: Optional[List[Dict[str, Any]]] = None
    best_run_index: Optional[int] = None

    run_counter = int(time.time())
    temp_solver_files: List[Path] = []  # Track temp solver files for cleanup

    try:
        # 6. HEBO optimization loop
        for iteration in range(max_iter):
            torch.set_default_tensor_type(CPU_TENSOR_TYPE)
            try:
                suggestions_df = optimizer.suggest(n_suggestions=suggestions)
            finally:
                torch.set_default_tensor_type(DEFAULT_TENSOR_TYPE)

            metrics: List[float] = []
            for _, row in suggestions_df.iterrows():
                params = row_to_params(row, hyper_space)
                run_counter += 1

                print(f"HEBO iteration {iteration + 1}/{max_iter}, trial {run_counter}: {params}")

                # Run experiment with temporary solver (multi-fidelity with promotion)
                experiment_results, temp_solver_path, final_fidelity = run_metric_with_solver(
                    parent_solver_id=parent_solver_id,
                    run_index=run_counter,
                    hp=params,
                    temp_solvers_dir=temp_solvers_dir,
                    db=db,
                    schedule_id=schedule_id,
                    promote_medium_threshold=promote_medium_threshold,
                    promote_high_threshold=promote_high_threshold,
                )

                # Track temp solver file for cleanup
                temp_solver_files.append(temp_solver_path)

                # Compute metric from experiment results using consensus objective
                if experiment_results:
                    consensus_fn, _ = create_consensus_objective(db)
                    metric_value = consensus_fn(experiment_results)
                else:
                    metric_value = 1000.0

                print(f"  Metric: {metric_value} (fidelity: {final_fidelity}, {len(experiment_results) if experiment_results else 0} experiments)")

                metrics.append(metric_value)
                history.append({
                    "iteration": iteration,
                    "run_index": run_counter,
                    "params": params,
                    "metric": metric_value,
                    "num_experiments": len(experiment_results) if experiment_results else 0,
                    "final_fidelity": final_fidelity,
                })

                if metric_value < best_metric:  # Minimize: lower is better
                    best_metric = metric_value
                    best_params = params
                    best_experiment_data = experiment_results
                    best_run_index = run_counter
                    print(f"  New best! Metric: {best_metric}")

            objective = np.array(metrics).reshape(-1, 1)
            optimizer.observe(suggestions_df, objective)

    except Exception:
        if not keep_temp_solvers:
            # Clean up temp solver files on error
            for solver_file in temp_solver_files:
                try:
                    if solver_file.exists():
                        solver_file.unlink()
                except Exception:
                    pass  # Ignore cleanup errors during exception handling
        raise

    # 7. Generate timestamp for filenames
    timestamp = int(time.time())

    # 8. Save summary
    summary_path = workspace_config.get_llm_log_dir() / f"hebo_summary_{timestamp}.json"
    summary_payload = {
        "best_metric": best_metric,
        "best_params": best_params,
        "best_experiment_data": best_experiment_data,
        "history": history,
        "best_run_index": best_run_index,
        "promote_medium_threshold": promote_medium_threshold,
        "promote_high_threshold": promote_high_threshold,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # 9. Clean up temporary solver files
    if not keep_temp_solvers:
        for solver_file in temp_solver_files:
            try:
                if solver_file.exists():
                    solver_file.unlink()
                # Also remove __pycache__ entry if exists
                pycache_dir = solver_file.parent / "__pycache__"
                if pycache_dir.exists():
                    pattern = f"{solver_file.stem}*.pyc"
                    for pyc_file in pycache_dir.glob(pattern):
                        pyc_file.unlink()
            except Exception as e:
                print(f"Warning: Failed to clean up {solver_file}: {e}")
        print(f"Cleaned up {len(temp_solver_files)} temporary solver files")
    else:
        print(f"Temporary solvers kept in: {temp_solvers_dir} ({len(temp_solver_files)} files)")

    summary = {
        "best_metric": best_metric,
        "best_params": best_params,
        "best_experiment_data": best_experiment_data,
        "history": history,
        "summary_path": _relative_path(summary_path),
        "timestamp": timestamp,
        "parent_solver_id": parent_solver_id,
        "parent_solver_code": parent_code,  # Include for saving later
    }
    return summary


# --------------------------------------------------------------------------- #
# Database registration (NEW: uses DatabaseManager)                          #
# --------------------------------------------------------------------------- #
def tune_and_register(
    parent_solver_id: int,
    db: DatabaseManager,
    llm: LLMClient,
    max_iter: int = 50,
    suggestions: int = 1,
    solvers_dir: Path = Path("solvers"),
    llm_log_dir: Path = Path("llm_log"),
    schedule_id: int = 0,
    promote_medium_threshold: float = 0.5,
    promote_high_threshold: float = 0.1,
) -> int:
    """
    Run HEBO tuning with dynamic fidelity promotion and register result in database.

    Uses consensus objective for metric computation (consistent with fidelity promotion).

    Args:
        parent_solver_id: ID of parent solver (also parent design_id)
        db: Database manager
        llm: LLM client
        max_iter: Maximum HEBO iterations
        suggestions: Number of suggestions per iteration
        solvers_dir: Directory containing solver files
        llm_log_dir: Directory for LLM logs
        schedule_id: Schedule module ID
        promote_medium_threshold: Threshold to promote low→medium (default 0.5, top 50%)
        promote_high_threshold: Threshold to promote medium→high (default 0.1, top 10%)

    Returns:
        Design ID of the tuned solver
    """
    # Get parent design
    parent_design = db.get_design(parent_solver_id)
    if parent_design is None:
        raise ValueError(f"Parent design {parent_solver_id} not found in database")

    # Compute parent metric using consensus objective
    consensus_fn, _ = create_consensus_objective(db)
    print(f"\n[HEBO] Tuning parent design {parent_solver_id}: {parent_design.short_name}")
    parent_metric = consensus_fn(parent_design.experiments) if parent_design.experiments else None
    print(f"[HEBO] Parent metric: {parent_metric}")

    # Run HEBO optimization with dynamic fidelity promotion
    summary = tune(
        parent_solver_id=parent_solver_id,
        max_iter=max_iter,
        suggestions=suggestions,
        db=db,
        keep_temp_solvers=False,
        schedule_id=schedule_id,
        promote_medium_threshold=promote_medium_threshold,
        promote_high_threshold=promote_high_threshold,
        parent_metric=parent_metric,
    )

    # Get new design ID
    new_design_id = db.get_next_design_id()

    # Save tuned solver
    tuned_solver_path = save_tuned_solver(
        parent_solver_code=summary["parent_solver_code"],
        best_params=summary["best_params"],
        new_solver_id=new_design_id,
        solvers_dir=solvers_dir
    )

    print(f"[HEBO] Tuned solver saved: {_relative_path(tuned_solver_path)}")

    # Compute improvement message
    improvement = ""
    if parent_metric is not None and summary["best_metric"] is not None:
        delta = summary["best_metric"] - parent_metric
        pct = (delta / parent_metric * 100) if parent_metric != 0 else 0
        improvement = f" ({delta:+.1f}, {pct:+.1f}%)"

    # Register in database
    db.add_design(
        design_id=new_design_id,
        planner_round_id=parent_design.planner_round_id,
        reference_weights=[ReferenceWeight(design_id=parent_solver_id, weight=1.0)],
        short_name=f"HEBO_tuned_{parent_design.short_name}",
        model_name=llm.model,
        code_path=str(tuned_solver_path),
        planner_log=parent_design.planner_log,
        designer_log=str(llm_log_dir / f"hebo_tune_{new_design_id}.json"),
        timestamp=create_timestamp(),
    )

    # Add experiment data if available (experiments already have fidelity tagged)
    if summary["best_experiment_data"]:
        for exp_data in summary["best_experiment_data"]:
            db.add_experiment_to_design(new_design_id, exp_data)

    # Save HEBO log
    hebo_log = {
        "design_id": new_design_id,
        "parent_design_id": parent_solver_id,
        "method": "HEBO",
        "iterations": len(summary["history"]),
        "best_metric": summary["best_metric"],
        "parent_metric": parent_metric,
        "improvement": improvement,
        "best_params": summary["best_params"],
        "summary_path": summary["summary_path"],
        "timestamp": create_timestamp(),
    }
    hebo_log_path = llm_log_dir / f"hebo_tune_{new_design_id}.json"
    hebo_log_path.parent.mkdir(parents=True, exist_ok=True)
    hebo_log_path.write_text(json.dumps(hebo_log, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[HEBO] Registered design {new_design_id}")
    print(f"[HEBO] Metric: {summary['best_metric']:.2f}{improvement}")

    return new_design_id


def maybe_run_tuning(
    db: DatabaseManager,
    llm: LLMClient,
    formatter: PromptFormatter,
    llm_log_dir: Path,
    solvers_dir: Path,
    max_iter: int = 50,
    suggestions: int = 1,
    max_tuning_ratio: float = 0.1,
    schedule_id: int = 0,
    promote_medium_threshold: float = 0.5,
    promote_high_threshold: float = 0.1,
) -> Optional[int]:
    """
    Maybe run HEBO tuning with dynamic fidelity promotion on the best untuned design.

    Selects the highest-ranked untuned design based on consensus objective.

    Args:
        db: Database manager
        llm: LLM client
        formatter: Prompt formatter
        llm_log_dir: LLM log directory
        solvers_dir: Solvers directory
        max_iter: Maximum HEBO iterations
        suggestions: Number of suggestions per iteration
        max_tuning_ratio: Max ratio of tuned designs to total designs
        schedule_id: Schedule module ID
        promote_medium_threshold: Threshold to promote low→medium (default 0.5, top 50%)
        promote_high_threshold: Threshold to promote medium→high (default 0.1, top 10%)

    Returns:
        Design ID of tuned design, or None if no tuning performed
    """
    all_designs = db.get_all_designs()
    if not all_designs:
        return None

    # Count existing tuned designs
    tuned_count = sum(
        1 for d in all_designs
        if "tuned" in d.short_name.lower() or "hebo" in d.short_name.lower()
    )

    total_count = len(all_designs)
    max_tuned = int(total_count * max_tuning_ratio)

    if tuned_count >= max_tuned:
        print(f"[HEBO] Skipping: already have {tuned_count} tuned designs (max {max_tuned})")
        return None

    # Rebuild MCGS and get ranked designs
    formatter._rebuild_graph()
    ranked = formatter._graph.get_ranked_nodes()

    if not ranked:
        return None

    # Find the best untuned design (ranked list is already sorted by objective)
    parent_id = None
    parent_metric = None
    for entry in ranked:
        design_id = entry["design_id"]
        design = db.get_design(design_id)
        if design and "tuned" not in design.short_name.lower() and "hebo" not in design.short_name.lower():
            metric = entry.get("objective")
            if metric is not None:
                parent_id = design_id
                parent_metric = metric
                break

    if parent_id is None:
        print(f"[HEBO] No suitable untuned designs found")
        return None

    print(f"\n[HEBO] Running hyperparameter optimization")
    print(f"[HEBO] Parent design: {parent_id} (metric={parent_metric:.2f})")

    try:
        tuned_id = tune_and_register(
            parent_solver_id=parent_id,
            db=db,
            llm=llm,
            max_iter=max_iter,
            suggestions=suggestions,
            solvers_dir=solvers_dir,
            llm_log_dir=llm_log_dir,
            schedule_id=schedule_id,
            promote_medium_threshold=promote_medium_threshold,
            promote_high_threshold=promote_high_threshold,
        )

        print(f"[HEBO] Created tuned design: {tuned_id}")
        return tuned_id

    except Exception as e:
        print(f"[HEBO] Tuning failed: {e}")
        return None

__all__ = ["tune", "tune_and_register", "save_tuned_solver", "maybe_run_tuning"]
