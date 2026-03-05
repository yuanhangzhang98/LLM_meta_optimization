"""Unified objective+scheduler workflow for multi-fidelity optimization.

This module generates proxy objective functions and multi-fidelity experiment
schedules using a single LLM call, following the workflow defined in:
- prompts/objective_plan.md
- prompts/objective_error_fix.md

Key features:
- Single-pass generation of objective + 3 schedule functions (low, medium, high)
- Retry/error-fix logic using prompt templates
- Sandbox validation of objective functions
- Context builders for existing objectives, schedules, and recent experiments
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from database.manager import DatabaseManager
from database.schema import create_timestamp
from llm_client import LLMClient, Message
from prompt_formatter import PromptFormatter
from prompt_schemas import (
    ObjectiveSchedulePlan,
    ObjectiveErrorFix,
    ObjectiveOnlyPlan,
    ObjectiveOnlyErrorFix,
)
import objective_loader
import objective_metadata
from consensus_objective import create_consensus_objective
import schedule_loader
import schedule_metadata
from workflow.validation import validate_objective_sandbox
import workspace_config


@dataclass(frozen=True)
class ObjectiveScheduleResult:
    """Result from an objective+schedule round."""
    new_objective_id: Optional[int]  # None if no new objective created
    new_schedule_id: Optional[int]  # Single schedule ID or None
    objective_fn: Callable  # The objective function (loaded)


# --------------------------------------------------------------------------- #
# File I/O helpers                                                            #
# --------------------------------------------------------------------------- #

def _write_json(path: Path, data: dict) -> None:
    """Write JSON to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def _write_objective_file(objective_id: int, code: str) -> Path:
    """Write objective function code to file.

    Args:
        objective_id: ID for the new objective
        code: Python code defining the objective function

    Returns:
        Path to the written objective file
    """
    objectives_dir = workspace_config.get_objectives_dir()
    objectives_dir.mkdir(parents=True, exist_ok=True)
    objective_path = objectives_dir / f"objective_{objective_id}.py"

    with objective_path.open("w", encoding="utf-8") as fh:
        fh.write(code)

    return objective_path


def _write_schedule_file(schedule_id: int, code: str) -> Path:
    """Write schedule code to file.

    Args:
        schedule_id: ID for the schedule
        code: Python code containing schedule functions

    Returns:
        Path to the written schedule file
    """
    schedules_dir = workspace_config.get_schedules_dir()
    schedules_dir.mkdir(parents=True, exist_ok=True)
    schedule_path = schedules_dir / f"schedule_{schedule_id}.py"

    with schedule_path.open("w", encoding="utf-8") as fh:
        fh.write(code)

    return schedule_path


def _build_existing_objective_summary(db: DatabaseManager, current_objective_id: int) -> str:
    """Build summary of existing objectives with descriptions and full code.

    This populates {existing_objective_summary} in objective_plan.md.

    Args:
        db: Database manager
        current_objective_id: ID of the currently active objective

    Returns:
        Formatted string describing existing objectives with their descriptions and code
    """
    available_objectives = objective_loader.list_available_objectives()

    if not available_objectives:
        return "No objectives defined yet. This will be objective_0.py."

    # Load all objective metadata
    all_metadata = objective_metadata.list_objective_metadata()
    objectives_dir = workspace_config.get_objectives_dir()

    lines = []
    for obj_id in available_objectives:
        metadata = all_metadata.get(str(obj_id), {})
        description = metadata.get("description", "No description available")

        if obj_id == current_objective_id:
            lines.append(f"**Objective {obj_id} (CURRENT):** {description}")
        else:
            lines.append(f"- Objective {obj_id}: {description}")

        # Read and include full objective code
        objective_path = objectives_dir / f"objective_{obj_id}.py"
        if objective_path.exists():
            code = objective_path.read_text(encoding="utf-8")
            lines.append("```python")
            lines.append(code.rstrip())
            lines.append("```")

        lines.append("")  # Blank line between objectives

    return "\n".join(lines)


def _build_existing_schedule_summary() -> str:
    """Build summary of existing schedules with descriptions.

    This populates {existing_schedule_summary} in objective_plan.md.

    Returns:
        Formatted string describing existing schedules with their descriptions
    """
    available_schedules = schedule_loader.list_available_schedules()

    if not available_schedules:
        return "No schedules defined yet. This will create the first schedule."

    # Load all schedule metadata
    all_metadata = schedule_metadata.list_schedule_metadata()

    lines = []
    for sched_id in sorted(available_schedules):
        metadata = all_metadata.get(str(sched_id), {})
        description = metadata.get("description", "No description available")
        fidelities = metadata.get("fidelities", ["low", "medium", "high"])

        lines.append(f"- Schedule {sched_id} ({'/'.join(fidelities)}): {description}")
        lines.append("")  # Blank line

    return "\n".join(lines)


def _build_recent_experiments_summary(db: DatabaseManager, current_objective_id: int, limit: int = 10) -> str:
    """Build summary of recent experiments with consensus objective metric.

    This populates {recent_experiments_objectives_summary} in objective_plan.md.

    Args:
        db: Database manager
        current_objective_id: ID of the current objective (used to select designs)
        limit: Maximum number of recent designs to include

    Returns:
        Formatted string describing recent experiments with full experiment JSON
    """
    designs = db.get_designs_by_objective(current_objective_id)

    if not designs:
        return "No designs evaluated with current objective yet."

    # Sort by timestamp (most recent first)
    sorted_designs = sorted(designs, key=lambda d: d.timestamp, reverse=True)[:limit]

    # Get consensus objective for metric computation
    consensus_fn, _ = create_consensus_objective(db)

    # Load all available objective functions for detailed breakdown
    available_objectives = objective_loader.list_available_objectives()
    objective_fns = {}
    for obj_id in available_objectives:
        try:
            objective_fns[obj_id] = objective_loader.load_objective(obj_id)
        except Exception:
            pass  # Skip objectives that fail to load

    lines = [f"Recent {len(sorted_designs)} designs evaluated:"]

    for design in sorted_designs:
        metric = "N/A"
        if design.experiments:
            # Compute metric using consensus objective
            metric = f"{consensus_fn(design.experiments):.4f}"

        lines.append(f"- Design {design.design_id} ({design.short_name}): consensus_objective={metric}")

        # Show detailed objective values for each objective
        if design.experiments and objective_fns:
            lines.append("  Objective values:")
            for obj_id in sorted(objective_fns.keys()):
                try:
                    obj_value = objective_fns[obj_id](design.experiments)
                    if obj_value is None or obj_value != obj_value:  # None or NaN
                        obj_value_str = "N/A"
                    else:
                        obj_value_str = f"{obj_value:.6f}"
                except Exception as e:
                    obj_value_str = f"Error: {e}"
                lines.append(f"    objective_{obj_id}: {obj_value_str}")

        # Include full experiment result JSON
        if design.experiments:
            experiments_json = json.dumps(design.experiments, indent=2)
            lines.append(f"  Experiments:\n{experiments_json}")
        else:
            lines.append("  Experiments: None")
        lines.append("")  # Blank line between designs

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Error fixing helpers                                                        #
# --------------------------------------------------------------------------- #

def _request_objective_schedule_error_fix(
    llm: LLMClient,
    formatter: PromptFormatter,
    objective_code: str,
    schedule_code: str,
    error_message: str,
    error_traceback: str,
    objective_id: int,
    attempt: int,
    messages: List[Message],
    llm_log_dir: Path,
) -> tuple[str, str, List[Message]]:
    """Request LLM to fix objective/schedule code errors by continuing conversation.

    Maintains conversation history by appending error fix request to existing messages,
    following the same pattern as Designer error fixes in iteration.py.

    Args:
        llm: LLM client
        formatter: Prompt formatter
        objective_code: Current objective code (may have errors)
        schedule_code: Current schedule code (may have errors)
        error_message: Error message from validation
        error_traceback: Full traceback
        objective_id: Objective ID being generated
        attempt: Current retry attempt number
        messages: Existing conversation history
        llm_log_dir: Directory for logs

    Returns:
        (fixed_objective_code, fixed_schedule_code, updated_messages) tuple
    """
    # Build error fix context
    error_context = formatter.base_context()
    error_context.update({
        "error_message": error_message,
        "full_traceback": error_traceback,
    })

    # Format error prompt using template
    error_prompt = formatter.format("objective_error_fix.md", error_context)

    # Append to existing conversation (not fresh start)
    new_messages = messages + [
        Message(role="user", content=error_prompt),
    ]

    fix = llm.request_structured(
        messages=new_messages,
        response_model=ObjectiveErrorFix,
    )

    # Add assistant response to conversation history
    fix_json = json.dumps(fix.model_dump(), ensure_ascii=False, indent=2)
    updated_messages = new_messages + [
        Message(role="assistant", content=fix_json),
    ]

    # Log error fix attempt
    log_dir = llm_log_dir / "objective_scheduler"
    log_dir.mkdir(parents=True, exist_ok=True)

    _write_json(log_dir / f"objective_error_fix_{objective_id}_try{attempt + 1}.json", {
        "objective_id": objective_id,
        "attempt": attempt + 1,
        "error_message": error_message,
        "error_traceback": error_traceback,
        "fix": fix.model_dump(),
        "timestamp": create_timestamp(),
    })

    # Return both codes (use original if fix is blank)
    fixed_objective = fix.objective_code if fix.objective_code.strip() else objective_code
    fixed_schedule = fix.schedule_code if fix.schedule_code.strip() else schedule_code

    return fixed_objective, fixed_schedule, updated_messages


def _request_objective_only_error_fix(
    llm: LLMClient,
    formatter: PromptFormatter,
    objective_code: str,
    error_message: str,
    error_traceback: str,
    objective_id: int,
    attempt: int,
    messages: List[Message],
    llm_log_dir: Path,
) -> tuple[str, List[Message]]:
    """Request LLM to fix objective code errors (objective-only mode).

    Args:
        llm: LLM client
        formatter: Prompt formatter
        objective_code: Current objective code (may have errors)
        error_message: Error message from validation
        error_traceback: Full traceback
        objective_id: Objective ID being generated
        attempt: Current retry attempt number
        messages: Existing conversation history
        llm_log_dir: Directory for logs

    Returns:
        (fixed_objective_code, updated_messages) tuple
    """
    # Build error fix context
    error_context = formatter.base_context()
    error_context.update({
        "error_message": error_message,
        "full_traceback": error_traceback,
    })

    # Format error prompt using objective-only template
    error_prompt = formatter.format("objective_only_error_fix.md", error_context)

    # Append to existing conversation (not fresh start)
    new_messages = messages + [
        Message(role="user", content=error_prompt),
    ]

    fix = llm.request_structured(
        messages=new_messages,
        response_model=ObjectiveOnlyErrorFix,
    )

    # Add assistant response to conversation history
    fix_json = json.dumps(fix.model_dump(), ensure_ascii=False, indent=2)
    updated_messages = new_messages + [
        Message(role="assistant", content=fix_json),
    ]

    # Log error fix attempt
    log_dir = llm_log_dir / "objective_scheduler"
    log_dir.mkdir(parents=True, exist_ok=True)

    _write_json(log_dir / f"objective_only_error_fix_{objective_id}_try{attempt + 1}.json", {
        "objective_id": objective_id,
        "attempt": attempt + 1,
        "error_message": error_message,
        "error_traceback": error_traceback,
        "fix": fix.model_dump(),
        "timestamp": create_timestamp(),
    })

    return fix.objective_code, updated_messages


def _write_objective_only_with_retries(
    llm: LLMClient,
    formatter: PromptFormatter,
    objective_code: str,
    objective_id: int,
    messages: List[Message],
    llm_log_dir: Path,
    max_retries: int = 3,
) -> Tuple[bool, Optional[Path], Optional[Callable], Optional[str], List[Message]]:
    """Write objective file and validate with retries on errors (objective-only mode).

    Args:
        llm: LLM client for error fixes
        formatter: Prompt formatter for error fix templates
        objective_code: Objective code to write
        objective_id: ID for the objective
        messages: Existing conversation history
        llm_log_dir: Directory for LLM logs
        max_retries: Maximum retry attempts

    Returns:
        (success, objective_path, objective_fn, error_info, updated_messages)
    """
    current_messages = messages  # Track messages through retries

    for attempt in range(max_retries):
        try:
            # Write and validate objective
            objective_path = _write_objective_file(objective_id, objective_code)
            objective_fn = objective_loader.load_objective(objective_id)
            is_valid, error_msg = validate_objective_sandbox(objective_fn)
            if not is_valid:
                raise ValueError(f"Objective validation failed: {error_msg}")

            return True, objective_path, objective_fn, None, current_messages

        except Exception as e:
            error_message = str(e)
            error_tb = traceback.format_exc()

            print(f"Objective {objective_id} attempt {attempt + 1}/{max_retries} failed: {error_message}")

            if attempt < max_retries - 1:
                # Request error fix for objective only (with conversation history)
                objective_code, current_messages = _request_objective_only_error_fix(
                    llm=llm,
                    formatter=formatter,
                    objective_code=objective_code,
                    error_message=error_message,
                    error_traceback=error_tb,
                    objective_id=objective_id,
                    attempt=attempt,
                    messages=current_messages,
                    llm_log_dir=llm_log_dir,
                )
            else:
                return False, None, None, error_message, current_messages

    return False, None, None, "Max retries exceeded", current_messages


def _write_objective_schedule_with_retries(
    llm: LLMClient,
    formatter: PromptFormatter,
    objective_code: str,
    schedule_code: str,
    objective_id: int,
    schedule_id: int,
    messages: List[Message],
    llm_log_dir: Path,
    max_retries: int = 3,
) -> Tuple[bool, Optional[Path], Optional[Path], Optional[Callable], Optional[str], List[Message]]:
    """Write objective and schedule files and validate with retries on errors.

    Maintains conversation history through retries, following the same pattern
    as Designer error fixes in iteration.py.

    Args:
        llm: LLM client for error fixes
        formatter: Prompt formatter for error fix templates
        objective_code: Objective code to write
        schedule_code: Schedule code to write (contains 3 functions)
        objective_id: ID for the objective
        schedule_id: ID for the schedule
        messages: Existing conversation history
        llm_log_dir: Directory for LLM logs
        max_retries: Maximum retry attempts

    Returns:
        (success, objective_path, schedule_path, objective_fn, error_info, updated_messages)
    """
    current_messages = messages  # Track messages through retries

    for attempt in range(max_retries):
        try:
            # Write and validate objective
            objective_path = _write_objective_file(objective_id, objective_code)
            objective_fn = objective_loader.load_objective(objective_id)
            is_valid, error_msg = validate_objective_sandbox(objective_fn)
            if not is_valid:
                raise ValueError(f"Objective validation failed: {error_msg}")

            # Write single schedule file with all 3 functions
            schedule_path = _write_schedule_file(schedule_id, schedule_code)
            _ = schedule_loader.load_schedule_fidelities(schedule_id)  # Validate all 3 fidelity functions load

            return True, objective_path, schedule_path, objective_fn, None, current_messages

        except Exception as e:
            error_message = str(e)
            error_tb = traceback.format_exc()

            print(f"Objective {objective_id} attempt {attempt + 1}/{max_retries} failed: {error_message}")

            if attempt < max_retries - 1:
                # Request error fix for BOTH objective and schedule (with conversation history)
                objective_code, schedule_code, current_messages = _request_objective_schedule_error_fix(
                    llm=llm,
                    formatter=formatter,
                    objective_code=objective_code,
                    schedule_code=schedule_code,
                    error_message=error_message,
                    error_traceback=error_tb,
                    objective_id=objective_id,
                    attempt=attempt,
                    messages=current_messages,
                    llm_log_dir=llm_log_dir,
                )
            else:
                return False, None, None, None, error_message, current_messages

    return False, None, None, None, "Max retries exceeded", current_messages


# --------------------------------------------------------------------------- #
# Main workflow                                                               #
# --------------------------------------------------------------------------- #

def run_objective_schedule_round(
    llm: LLMClient,
    db: DatabaseManager,
    formatter: PromptFormatter,
    current_objective_id: int,
    objective_round_id: int,
    *,
    llm_log_dir: Path = None,
    max_retries: int = 3,
    generate_schedule: bool = True,
    meta_directions: Optional[str] = None,
) -> ObjectiveScheduleResult:
    """
    Run a unified objective+schedule round to create new objective and optionally schedules.

    Generates new objective function and optionally schedules based on recent experiments,
    following the workflow defined in prompts/objective_plan.md or objective_only_plan.md.

    Args:
        llm: LLM client
        db: Database manager
        formatter: Prompt formatter
        current_objective_id: ID of the current objective function
        objective_round_id: ID for this objective analysis round
        llm_log_dir: Directory for LLM conversation logs
        max_retries: Maximum retry attempts for code generation
        generate_schedule: If True, generate both objective and schedules.
            If False, only generate objective (use baseline schedule).
        meta_directions: Optional strategic guidance from meta-agent for
            objective generation. If provided, included in the prompt context.

    Returns:
        ObjectiveScheduleResult with new objective/schedules and full message history
    """
    if llm_log_dir is None:
        llm_log_dir = workspace_config.get_llm_log_dir()

    # Load current objective function
    current_objective_fn = objective_loader.load_objective(current_objective_id)

    # Build context with all required keys
    stage1_context = formatter.base_context()
    stage1_context.update({
        # Required context keys for objective prompts
        "existing_objective_summary": _build_existing_objective_summary(db, current_objective_id),
        "existing_schedule_summary": _build_existing_schedule_summary(),
        "recent_experiments_objectives_summary": _build_recent_experiments_summary(db, current_objective_id),
        # Meta-agent guidance (if provided)
        "meta_agent_directions": meta_directions or "No specific guidance provided.",
        # baseline_objective_code, baseline_schedule_code, experiment_code already in base_context()
    })

    # Format prompts - use appropriate template based on generate_schedule flag
    system_prompt = formatter.format("objective_system_prompt.md", stage1_context)
    if generate_schedule:
        stage1_prompt = formatter.format("objective_plan.md", stage1_context)
    else:
        stage1_prompt = formatter.format("objective_only_plan.md", stage1_context)

    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=stage1_prompt),
    ]

    # Request plan using appropriate schema
    if generate_schedule:
        plan = llm.request_structured(
            messages=messages,
            response_model=ObjectiveSchedulePlan,
        )
    else:
        plan = llm.request_structured(
            messages=messages,
            response_model=ObjectiveOnlyPlan,
        )

    # Add plan response to conversation history
    plan_json = json.dumps(plan.model_dump(), ensure_ascii=False, indent=2)
    messages = messages + [
        Message(role="assistant", content=plan_json),
    ]

    # Get next objective ID
    new_objective_id = objective_loader.get_next_objective_id()

    if generate_schedule:
        # Full mode: write and validate both objective and schedule
        new_schedule_id = schedule_loader.get_next_schedule_id()

        success, objective_path, schedule_path, objective_fn, error_info, messages = _write_objective_schedule_with_retries(
            llm=llm,
            formatter=formatter,
            objective_code=plan.objective_code,
            schedule_code=plan.schedule_code,
            objective_id=new_objective_id,
            schedule_id=new_schedule_id,
            messages=messages,
            llm_log_dir=llm_log_dir,
            max_retries=max_retries,
        )

        if success:
            print(f"✓ Created objective {new_objective_id}: {objective_path}")
            print(f"✓ Created schedule {new_schedule_id}: {schedule_path}")

            # Save objective metadata
            objective_metadata.save_objective_metadata(
                objective_id=new_objective_id,
                description=plan.objective_description,
                round_id=objective_round_id,
            )

            # Save schedule metadata
            schedule_metadata.save_schedule_metadata(
                schedule_id=new_schedule_id,
                description=plan.schedule_description,
                fidelities=["low", "medium", "high"],
                round_id=objective_round_id,
            )

            final_objective_fn = objective_fn
            final_objective_id = new_objective_id
            final_schedule_id = new_schedule_id
        else:
            print(f"✗ Failed to create objective/schedules: {error_info}")
            final_objective_fn = current_objective_fn
            final_objective_id = None
            final_schedule_id = None
    else:
        # Objective-only mode: write and validate only objective
        success, objective_path, objective_fn, error_info, messages = _write_objective_only_with_retries(
            llm=llm,
            formatter=formatter,
            objective_code=plan.objective_code,
            objective_id=new_objective_id,
            messages=messages,
            llm_log_dir=llm_log_dir,
            max_retries=max_retries,
        )

        if success:
            print(f"✓ Created objective {new_objective_id}: {objective_path}")
            print(f"  (Schedule generation disabled, using baseline schedule)")

            # Save objective metadata
            objective_metadata.save_objective_metadata(
                objective_id=new_objective_id,
                description=plan.objective_description,
                round_id=objective_round_id,
            )

            final_objective_fn = objective_fn
            final_objective_id = new_objective_id
            final_schedule_id = None  # No new schedule created
        else:
            print(f"✗ Failed to create objective: {error_info}")
            final_objective_fn = current_objective_fn
            final_objective_id = None
            final_schedule_id = None

    # Save conversation log
    log_data = {
        "objective_round_id": objective_round_id,
        "model_name": llm.model,
        "current_objective_id": current_objective_id,
        "new_objective_id": final_objective_id,
        "new_schedule_id": final_schedule_id,
        "generate_schedule": generate_schedule,
        "meta_directions": meta_directions,
        "plan": plan.model_dump(),
        "messages": [msg.as_dict() for msg in messages],
        "timestamp": create_timestamp(),
    }
    _write_json(llm_log_dir / "objective_scheduler" / f"round_{objective_round_id}.json", log_data)

    return ObjectiveScheduleResult(
        new_objective_id=final_objective_id,
        new_schedule_id=final_schedule_id,
        objective_fn=final_objective_fn,
    )


__all__ = [
    "ObjectiveScheduleResult",
    "run_objective_schedule_round",
]
