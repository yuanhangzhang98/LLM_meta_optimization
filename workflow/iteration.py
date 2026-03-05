"""Single-iteration planner and designer workflows.

Refactored to use DatabaseManager, renamed agents (supervisor→planner, researcher→designer),
and removed rounds.jsonl persistence (planner messages stored in LLM logs only).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Callable, Any
import traceback

from database.manager import DatabaseManager
from database.schema import ReferenceWeight, create_timestamp
from executor import execute_schedule
from llm_client import LLMClient, Message
from problem_config import get_problem_config
from prompt_formatter import PromptFormatter
from prompt_schemas import (
    DesignerAnalysis,
    DesignerPlan,
    DesignerErrorFix,
    FidelityPromotionDecision,
    PlannerLookup,
    PlannerRoundSummary,
)
from solver_writer import write_solver_file, read_solver_code
import schedule_metadata
from consensus_objective import create_consensus_objective, get_consensus_description
import workspace_config
from schedule_loader import load_schedule_fidelities


def _load_fidelity_schedules(schedule_id: int) -> Dict[str, Callable]:
    """Load fidelity-specific schedule functions from a schedule module.

    Args:
        schedule_id: ID of the schedule to load

    Returns:
        Dict with keys 'low', 'medium', 'high' mapping to schedule functions

    Raises:
        ImportError: If schedule module cannot be loaded
        AttributeError: If required fidelity functions are missing
    """
    # Use schedule_loader which handles workspace-aware paths
    return load_schedule_fidelities(schedule_id)


def _schedule_accepts_prior_experiments(schedule_fn: Callable) -> bool:
    """Check if schedule function accepts prior_experiments parameter.

    Used for backward compatibility with schedules that don't support skipping
    already-successful experiments.
    """
    import inspect
    sig = inspect.signature(schedule_fn)
    return 'prior_experiments' in sig.parameters


@dataclass(frozen=True)
class PlannerResult:
    """Result from a planner round."""
    round_id: int
    lookup: PlannerLookup
    summary: PlannerRoundSummary
    messages: List[Message]  # Full conversation for reference


# --------------------------------------------------------------------------- #
# JSON helpers                                                                #
# --------------------------------------------------------------------------- #

def _write_json(path: Path, data: Dict) -> None:
    """Write JSON to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def _next_planner_round_id(db: DatabaseManager) -> int:
    """Get next planner round ID."""
    designs = db.get_all_designs()
    if not designs:
        return 0
    max_round = max(d.planner_round_id for d in designs)
    return max_round + 1


# --------------------------------------------------------------------------- #
# Formatting helpers                                                          #
# --------------------------------------------------------------------------- #

def _build_planner_context(summary: PlannerRoundSummary, index: int, references: Sequence[int]) -> str:
    """Build context string for designer agent from planner summary."""
    direction = summary.research_directions[index]
    rationale = summary.strategy_rationales[index] if index < len(summary.strategy_rationales) else ""
    focus = summary.focus_areas[index] if index < len(summary.focus_areas) else []
    avoid = summary.avoid_areas[index] if index < len(summary.avoid_areas) else []

    if isinstance(focus, str):
        focus = [focus]
    if isinstance(avoid, str):
        avoid = [avoid]

    focus_text = "\n".join(f"- {item}" for item in focus) if focus else "- (unspecified)"
    avoid_text = "\n".join(f"- {item}" for item in avoid) if avoid else "- (none)"
    ref_text = ", ".join(str(r) for r in references) if references else "none"

    return (
        f"Research direction: {direction}\n"
        f"Rationale: {rationale}\n"
        f"Focus areas:\n{focus_text}\n"
        f"Avoid:\n{avoid_text}\n"
        f"Reference design IDs: {ref_text}"
    )


# --------------------------------------------------------------------------- #
# Planner workflow                                                            #
# --------------------------------------------------------------------------- #

def run_planner_round(
    llm: LLMClient,
    db: DatabaseManager,
    formatter: PromptFormatter,
    *,
    llm_log_dir: Path = None,
    num_designers: int = 2,
) -> PlannerResult:
    """
    Run a planner round to analyze designs and provide research directions.

    Renamed from run_supervisor_round. No longer writes to rounds.jsonl.
    Full conversation stored in LLM logs only.

    Args:
        llm: LLM client
        db: Database manager
        formatter: Prompt formatter
        llm_log_dir: Directory for LLM conversation logs
        num_designers: Number of designer agents to spawn

    Returns:
        PlannerResult with round info and full message history
    """
    if llm_log_dir is None:
        llm_log_dir = workspace_config.get_llm_log_dir()

    round_id = _next_planner_round_id(db)

    # Stage 1: Lookup relevant designs
    system_prompt = formatter.format("planner_system_prompt.md")
    stage1_prompt = formatter.format("planner_analysis_1.md")

    stage1_messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=stage1_prompt),
    ]

    lookup = llm.request_structured(
        messages=stage1_messages,
        response_model=PlannerLookup,
    )

    # Stage 2: Provide strategic directions
    details_text = formatter.render_design_details(lookup.lookup_design_ids)
    stage2_context = formatter.base_context()
    stage2_context.update({"design_details": details_text})  # Legacy key
    stage2_prompt = formatter.format("planner_analysis_2.md", stage2_context)

    lookup_json = json.dumps(lookup.model_dump(), ensure_ascii=False, indent=2)
    stage2_messages = stage1_messages + [
        Message(role="assistant", content=lookup_json),
        Message(role="user", content=stage2_prompt),
    ]

    summary = llm.request_structured(
        messages=stage2_messages,
        response_model=PlannerRoundSummary,
    )

    # Save full conversation to LLM log
    conversation_log = {
        "round_id": round_id,
        "model_name": llm.model,
        "lookup": lookup.model_dump(),
        "summary": summary.model_dump(),
        "messages": [msg.as_dict() for msg in stage2_messages],
        "timestamp": create_timestamp(),
    }
    _write_json(llm_log_dir / f"planner_{round_id}.json", conversation_log)

    return PlannerResult(
        round_id=round_id,
        lookup=lookup,
        summary=summary,
        messages=stage2_messages
    )


# --------------------------------------------------------------------------- #
# Designer workflow                                                           #
# --------------------------------------------------------------------------- #

def run_designer_iteration(
    llm: LLMClient,
    db: DatabaseManager,
    formatter: PromptFormatter,
    planner: PlannerResult,
    *,
    llm_log_dir: Path = None,
    solvers_dir: Path = None,
    max_retries: int = 3,
    schedule_id: int = 0,
    objective_id: int = 0,
    promote_medium_threshold: float = 0.5,
    promote_high_threshold: float = 0.1,
) -> List[int]:
    """
    Run designer iterations based on planner directions.

    Renamed from run_researcher_iteration. Uses DatabaseManager instead of JSONL.
    Each design is created with designer plan, executed, and analyzed.

    Args:
        llm: LLM client
        db: Database manager
        formatter: Prompt formatter
        planner: Planner result with research directions
        llm_log_dir: Directory for LLM logs
        solvers_dir: Directory for solver files
        max_retries: Maximum retry attempts for execution errors
        schedule_id: ID of schedule to use
        objective_id: ID of objective function to use
        promote_medium_threshold: Consensus objective threshold to promote low→medium (default 0.5)
        promote_high_threshold: Consensus objective threshold to promote medium→high (default 0.1)

    Returns:
        List of created design IDs
    """
    if llm_log_dir is None:
        llm_log_dir = workspace_config.get_llm_log_dir()
    if solvers_dir is None:
        solvers_dir = workspace_config.get_solvers_dir()

    completed_ids: List[int] = []
    directions = planner.summary.research_directions

    for index, direction in enumerate(directions):
        design_id = db.get_next_design_id()

        # Get reference designs from planner
        references: Sequence[int] = []
        if planner.summary.reference_design_ids:
            if index < len(planner.summary.reference_design_ids):
                references = planner.summary.reference_design_ids[index]

        # Build planner context for this direction
        planner_context = _build_planner_context(planner.summary, index, references)

        # Stage 1: Generate design plan
        design_plan, solver_code, messages = _run_designer_plan_stage(
            llm=llm,
            formatter=formatter,
            planner_context=planner_context,
            references=references,
            design_id=design_id,
            llm_log_dir=llm_log_dir,
            objective_id=objective_id,
            schedule_id=schedule_id,
        )

        # Initialize conversation history with plan stage
        plan_response_json = json.dumps(design_plan.model_dump(), ensure_ascii=False, indent=2)

        designer_messages = messages + [
            Message(role="assistant", content=plan_response_json),
        ]

        # Stage 2: Multi-fidelity execution and analysis
        all_experiments = []
        final_analysis = None

        # NEW: Multi-fidelity progressive execution
        try:
            fidelity_schedules = _load_fidelity_schedules(schedule_id)
            fidelity_sequence = ['low', 'medium', 'high']

            from objective_loader import load_objective
            objective_fn = load_objective(objective_id)

            for fidelity_idx, fidelity_name in enumerate(fidelity_sequence):
                print(f"\n=== Design {design_id}: {fidelity_name} fidelity ===")

                # Execute this fidelity level
                # Pass prior experiments to skip already-successful experiments at higher fidelities
                fid_success, fid_experiments, fid_error, designer_messages = _execute_single_fidelity(
                    llm=llm,
                    formatter=formatter,
                    solver_code=solver_code,
                    design_id=design_id,
                    solvers_dir=solvers_dir,
                    llm_log_dir=llm_log_dir,
                    schedule_fn=fidelity_schedules[fidelity_name],
                    fidelity_name=fidelity_name,
                    messages=designer_messages,
                    max_retries=max_retries,
                    prior_experiments=all_experiments if fidelity_idx > 0 else None,
                )

                if not fid_success:
                    # Execution failed - stop progression
                    print(f"Design {design_id} failed at {fidelity_name} fidelity: {fid_error}")
                    break

                # Tag experiments with fidelity
                for exp in fid_experiments:
                    exp['fidelity'] = fidelity_name

                all_experiments.extend(fid_experiments)

                # Compute objective value
                obj_value = objective_fn(fid_experiments)

                print(f"Design {design_id} {fidelity_name} fidelity: "
                        f"{len(fid_experiments)} experiments, objective={obj_value:.6f}")

                # Append experiment results to conversation (no LLM response)
                designer_messages = _render_experiment_results(
                    formatter=formatter,
                    current_fidelity=fidelity_name,
                    current_experiments=fid_experiments,
                    objective_value=obj_value,
                    messages=designer_messages,
                )

                # Check if we should promote to next fidelity (rule-based)
                is_highest_fidelity = (fidelity_idx == len(fidelity_sequence) - 1)

                if not is_highest_fidelity:
                    promote, consensus_score, reasoning = _rule_based_promotion_decision(
                        db=db,
                        experiments=all_experiments,
                        current_fidelity=fidelity_name,
                        medium_threshold=promote_medium_threshold,
                        high_threshold=promote_high_threshold,
                    )

                    # Log the rule-based decision
                    _write_json(llm_log_dir / f"designer_promotion_{design_id}_{fidelity_name}.json", {
                        "design_id": design_id,
                        "current_fidelity": fidelity_name,
                        "consensus_objective": consensus_score,
                        "threshold": promote_medium_threshold if fidelity_name == 'low' else promote_high_threshold,
                        "promote_fidelity": promote,
                        "reasoning": reasoning,
                        "timestamp": create_timestamp(),
                    })

                    # Add promotion decision to conversation for context
                    decision_text = f"Fidelity promotion decision: {reasoning} → {'PROMOTE' if promote else 'CONCLUDE'}"
                    designer_messages = designer_messages + [
                        Message(role="user", content=decision_text),
                    ]

                    print(f"Design {design_id} at {fidelity_name}: {reasoning} → {'PROMOTE' if promote else 'CONCLUDE'}")

                    if not promote:
                        break
                else:
                    print(f"Design {design_id}: Reached highest fidelity, concluding")

        except Exception as e:
            # Fidelity progression failed
            print(f"Design {design_id}: Fidelity progression error: {str(e)}")
            traceback.print_exc()

        # Run final analysis ONCE after fidelity loop
        # Conversation contains all context (plan, experiment results, promotion decisions)
        final_analysis, designer_messages = _run_designer_analysis_stage(
            llm=llm,
            formatter=formatter,
            design_id=design_id,
            messages=designer_messages,
            llm_log_dir=llm_log_dir,
        )

        # Save final conversation log
        if final_analysis is not None:
            _write_json(llm_log_dir / f"designer_{design_id}.json", {
                "design_id": design_id,
                "model_name": llm.model,
                "messages": [msg.as_dict() for msg in designer_messages],
                "final_analysis": final_analysis.model_dump(),
                "timestamp": create_timestamp(),
            })

        # Stage 3: Create design in database
        reference_weights = []
        if final_analysis and final_analysis.reference_weights:
            reference_weights = [
                ReferenceWeight(design_id=rw.design_id, weight=rw.weight)
                for rw in final_analysis.reference_weights
            ]

        db.add_design(
            design_id=design_id,
            planner_round_id=planner.round_id,
            reference_weights=reference_weights,
            short_name=final_analysis.short_name if final_analysis else f"failed_{design_id}",
            model_name=llm.model,
            code_path=str(solvers_dir / f"solver_{design_id}.py"),
            planner_log=str(llm_log_dir / f"planner_{planner.round_id}.json"),
            designer_log=str(llm_log_dir / f"designer_{design_id}.json"),
            timestamp=create_timestamp(),
        )

        # Add all experiment results if execution succeeded
        if all_experiments:
            for experiment_data in all_experiments:
                db.add_experiment_to_design(design_id, experiment_data)

        # Update design metadata with schedule_id and objective_id
        if schedule_id > 0 or objective_id > 0:
            db.update_design_metadata(
                design_id=design_id,
                schedule_id=schedule_id if schedule_id > 0 else None,
                objective_id=objective_id if objective_id > 0 else None,
            )

        completed_ids.append(design_id)

    return completed_ids


def _run_designer_plan_stage(
    llm: LLMClient,
    formatter: PromptFormatter,
    planner_context: str,
    references: Sequence[int],
    design_id: int,
    llm_log_dir: Path,
    objective_id: int = 0,
    schedule_id: int = 0,
) -> tuple[DesignerPlan, str]:
    """Run designer plan stage (generate solver code)."""
    # Fetch reference design details
    details_text = formatter.render_design_details(references)
    context = formatter.base_context()
    context.update({
        "planner_context": planner_context,
        "reference_experiments": details_text,  # Populate reference_experiments with specific designs
        "objective_description": get_consensus_description(formatter._db, objective_id),
        "schedule_description": schedule_metadata.get_schedule_description(schedule_id),
    })

    system_prompt = formatter.format("designer_system_prompt.md", context)
    stage1_prompt = formatter.format("designer_analysis_1.md", context)

    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=stage1_prompt),
    ]

    plan = llm.request_structured(
        messages=messages,
        response_model=DesignerPlan,
    )

    # Save plan to log
    _write_json(llm_log_dir / f"designer_plan_{design_id}.json", {
        "design_id": design_id,
        "model_name": llm.model,
        "plan": plan.model_dump(),
        "messages": [msg.as_dict() for msg in messages],
        "timestamp": create_timestamp(),
    })

    return plan, plan.solver_code, messages


def _execute_single_fidelity(
    llm: LLMClient,
    formatter: PromptFormatter,
    solver_code: str,
    design_id: int,
    solvers_dir: Path,
    llm_log_dir: Path,
    schedule_fn: Callable,
    fidelity_name: str,
    messages: List[Message],
    max_retries: int = 3,
    prior_experiments: Optional[List[Dict[str, Any]]] = None,
) -> tuple[bool, Optional[List[Dict[str, Any]]], Optional[str], List[Message]]:
    """
    Execute solver at a single fidelity level with retries on errors.

    Args:
        llm: LLM client for error fixes
        formatter: Prompt formatter for error fix prompts
        solver_code: Solver code to execute (will be updated on fixes)
        design_id: ID of the design
        solvers_dir: Directory for solver files
        llm_log_dir: Directory for LLM logs
        schedule_fn: Fidelity-specific schedule function to execute
        fidelity_name: Name of fidelity level ('low', 'medium', 'high') for logging
        messages: Conversation history from previous stages
        max_retries: Maximum retry attempts
        prior_experiments: Optional prior experiment results from lower fidelity levels.
            If provided and schedule supports it, skips already-successful experiments.

    Returns:
        (success, experiment_results, error_info, updated_messages)
        where experiment_results is a list of experiment dicts from this fidelity
    """
    current_messages = messages  # Track messages through retry loop

    for attempt in range(max_retries):
        try:
            # Write solver file
            solver_path = write_solver_file(
                solver_code=solver_code,
                solver_id=design_id,
                solvers_dir=solvers_dir,
                validate=True,
                overwrite=True,
            )

            # Execute the provided schedule function
            # Pass prior_experiments if schedule supports it (for skipping already-successful experiments)
            print(f"Design {design_id}: Executing {fidelity_name} fidelity...")
            if prior_experiments and _schedule_accepts_prior_experiments(schedule_fn):
                experiment_results = schedule_fn(design_id=design_id, prior_experiments=prior_experiments)
            else:
                experiment_results = schedule_fn(design_id=design_id)
            print(f"Design {design_id}: {fidelity_name} fidelity completed with {len(experiment_results)} experiments")

            return True, experiment_results, None, current_messages

        except Exception as e:
            error_message = str(e)
            error_traceback = traceback.format_exc()

            print(f"Design {design_id} {fidelity_name} fidelity attempt {attempt + 1}/{max_retries} failed: {error_message}")

            if attempt < max_retries - 1:
                # Try to fix the error
                solver_code, current_messages = _request_error_fix(
                    llm=llm,
                    formatter=formatter,
                    solver_code=solver_code,
                    error_message=error_message,
                    error_traceback=error_traceback,
                    design_id=design_id,
                    attempt=attempt,
                    messages=current_messages,
                    llm_log_dir=llm_log_dir,
                )
            else:
                # Final attempt failed
                return False, None, error_message, current_messages

    return False, None, "Max retries exceeded", current_messages


def _request_error_fix(
    llm: LLMClient,
    formatter: PromptFormatter,
    solver_code: str,
    error_message: str,
    error_traceback: str,
    design_id: int,
    attempt: int,
    messages: List[Message],
    llm_log_dir: Path,
) -> tuple[str, List[Message]]:
    """Request LLM to fix solver code error.

    Returns:
        (fixed_solver_code, updated_messages) - Fixed code and conversation history
    """
    # Get problem config for component info
    config = get_problem_config()
    required_components = config.get_required_component_names()

    # Format error fix prompt using formatter
    context = formatter.base_context()
    context.update({
        'error_message': error_message,
        'full_traceback': error_traceback,
        'num_components': len(required_components),
        'component_names': ', '.join(required_components),
    })

    error_prompt = formatter.format("designer_error_fix.md", context)

    # Append to existing conversation
    new_messages = messages + [
        Message(role="user", content=error_prompt),
    ]

    fix = llm.request_structured(
        messages=new_messages,
        response_model=DesignerErrorFix,
    )

    # Add assistant response to conversation history
    fix_json = json.dumps(fix.model_dump(), ensure_ascii=False, indent=2)
    updated_messages = new_messages + [
        Message(role="assistant", content=fix_json),
    ]

    # Log error fix attempt
    _write_json(llm_log_dir / f"designer_error_fix_{design_id}_try{attempt + 1}.json", {
        "design_id": design_id,
        "attempt": attempt + 1,
        "error_message": error_message,
        "error_traceback": error_traceback,
        "fix": fix.model_dump(),
        "timestamp": create_timestamp(),
    })

    return fix.solver_code, updated_messages


def _rule_based_promotion_decision(
    db: DatabaseManager,
    experiments: List[Dict[str, Any]],
    current_fidelity: str,
    medium_threshold: float,
    high_threshold: float,
) -> tuple[bool, float, str]:
    """Rule-based fidelity promotion using consensus objective.

    Args:
        db: Database manager
        experiments: List of experiment results from current fidelity
        current_fidelity: Current fidelity level ('low', 'medium', 'high')
        medium_threshold: Threshold to promote low→medium (default 0.5)
        high_threshold: Threshold to promote medium→high (default 0.1)

    Returns:
        (promote, consensus_score, reasoning) - whether to promote, score, and explanation
    """
    consensus_fn, _ = create_consensus_objective(db)
    consensus_score = consensus_fn(experiments)

    # Determine threshold based on target fidelity
    if current_fidelity == 'low':
        threshold = medium_threshold
        target = 'medium'
    elif current_fidelity == 'medium':
        threshold = high_threshold
        target = 'high'
    else:
        return False, float(consensus_score), "Already at highest fidelity"

    # Convert NumPy types to native Python types for JSON serialization
    promote = bool(consensus_score < threshold)
    reasoning = f"consensus={consensus_score:.3f}, threshold={threshold} for {target}"
    return promote, float(consensus_score), reasoning


def _run_designer_promotion_stage(
    llm: LLMClient,
    formatter: PromptFormatter,
    design_id: int,
    current_fidelity: str,
    messages: List[Message],
    llm_log_dir: Path,
) -> tuple[FidelityPromotionDecision, List[Message]]:
    """Run designer promotion decision stage.

    Returns:
        (decision, updated_messages) - decision and full conversation history
    """
    context = formatter.base_context()
    context.update({
        'current_fidelity': current_fidelity,
    })

    promotion_prompt = formatter.format("designer_analysis_3.md", context)

    # Append to existing conversation
    new_messages = messages + [
        Message(role="user", content=promotion_prompt),
    ]

    decision = llm.request_structured(
        messages=new_messages,
        response_model=FidelityPromotionDecision,
    )

    # Add assistant response to conversation history
    decision_json = json.dumps(decision.model_dump(), ensure_ascii=False, indent=2)
    updated_messages = new_messages + [
        Message(role="assistant", content=decision_json),
    ]

    # Log decision
    _write_json(llm_log_dir / f"designer_promotion_{design_id}_{current_fidelity}.json", {
        "design_id": design_id,
        "current_fidelity": current_fidelity,
        "decision": decision.model_dump(),
        "timestamp": create_timestamp(),
    })

    print(f"Design {design_id} at {current_fidelity} fidelity: "
          f"{'PROMOTE' if decision.promote_fidelity else 'CONCLUDE'}")

    return decision, updated_messages


def _render_experiment_results(
    formatter: PromptFormatter,
    current_fidelity: str,
    current_experiments: List[Dict[str, Any]],
    objective_value: float,
    messages: List[Message],
) -> List[Message]:
    """Append experiment results to conversation (no LLM response).

    This renders designer_analysis_2.md which shows experiment results
    as information for the designer to consider in subsequent decisions.

    Returns:
        Updated messages with experiment results appended
    """
    # Format experiment data for prompt
    if not current_experiments:
        experiments_text = "No experiments completed (execution failed)"
        objective_text = "N/A"
    else:
        experiments_text = json.dumps(current_experiments, indent=2)
        objective_text = f"{objective_value:.6f}"

    context = formatter.base_context()
    context.update({
        "current_fidelity": current_fidelity,
        "experiment_results": experiments_text,
        "objective_value": objective_text,
    })

    results_prompt = formatter.format("designer_analysis_2.md", context)

    # Append to conversation as user message (no response expected)
    updated_messages = messages + [
        Message(role="user", content=results_prompt),
    ]

    return updated_messages


def _run_designer_analysis_stage(
    llm: LLMClient,
    formatter: PromptFormatter,
    design_id: int,
    messages: List[Message],
    llm_log_dir: Path,
) -> tuple[DesignerAnalysis, List[Message]]:
    """Run final comprehensive analysis using designer_analysis_4.md.

    This is called once after all fidelity levels complete to produce
    the final DesignerAnalysis with insights and reference weights.

    Returns:
        (analysis, updated_messages) - analysis and full conversation history
    """
    context = formatter.base_context()
    analysis_prompt = formatter.format("designer_analysis_4.md", context)

    # Append to existing conversation
    new_messages = messages + [
        Message(role="user", content=analysis_prompt),
    ]

    analysis = llm.request_structured(
        messages=new_messages,
        response_model=DesignerAnalysis,
    )

    # Add assistant response to conversation history
    analysis_json = json.dumps(analysis.model_dump(), ensure_ascii=False, indent=2)
    updated_messages = new_messages + [
        Message(role="assistant", content=analysis_json),
    ]

    # Save analysis to log
    _write_json(llm_log_dir / f"designer_analysis_{design_id}.json", {
        "design_id": design_id,
        "model_name": llm.model,
        "analysis": analysis.model_dump(),
        "timestamp": create_timestamp(),
    })

    return analysis, updated_messages


# --------------------------------------------------------------------------- #
# Combined iteration (planner + designers)                                    #
# --------------------------------------------------------------------------- #

def run_iteration(
    llm: LLMClient,
    db: DatabaseManager,
    formatter: PromptFormatter,
    *,
    num_designers: int = 2,
    llm_log_dir: Path = None,
    solvers_dir: Path = None,
    schedule_id: int = 0,
    objective_id: int = 0,
) -> tuple[PlannerResult, List[int]]:
    """
    Run a complete iteration: planner round + designer iterations.

    Combines planner and designer workflows into a single iteration.

    Args:
        llm: LLM client
        db: Database manager
        formatter: Prompt formatter
        num_designers: Number of designer agents per planner round
        llm_log_dir: Directory for LLM logs
        solvers_dir: Directory for solver files
        schedule_id: ID of schedule to use (0 for legacy single execution)
        objective_id: ID of objective function to use (0 for legacy)

    Returns:
        (PlannerResult, List of design IDs)
    """
    if llm_log_dir is None:
        llm_log_dir = workspace_config.get_llm_log_dir()
    if solvers_dir is None:
        solvers_dir = workspace_config.get_solvers_dir()

    # Run planner round
    planner_result = run_planner_round(
        llm=llm,
        db=db,
        formatter=formatter,
        llm_log_dir=llm_log_dir,
        num_designers=num_designers,
    )

    # Run designer iterations
    design_ids = run_designer_iteration(
        llm=llm,
        db=db,
        formatter=formatter,
        planner=planner_result,
        llm_log_dir=llm_log_dir,
        solvers_dir=solvers_dir,
        schedule_id=schedule_id,
        objective_id=objective_id,
    )

    return planner_result, design_ids


__all__ = [
    "PlannerResult",
    "run_planner_round",
    "run_designer_iteration",
    "run_iteration",
]
