"""Meta-agent workflow for research oversight and objective guidance.

The meta-agent oversees research progress, evaluates objective functions,
and provides strategic guidance for objective generation. It runs before
objective rounds to adjust weights and provide directions.

Key responsibilities:
- Analyze research progress and identify stuck/converging phases
- Evaluate objective function effectiveness via correlation analysis
- Assign weight multipliers to amplify/suppress objectives
- Provide strategic directions for next objective generation
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from database.manager import DatabaseManager
from database.schema import create_timestamp
from llm_client import LLMClient, Message
from prompt_formatter import PromptFormatter
from prompt_schemas import MetaAgentAnalysis
import objective_loader
import objective_metadata
from consensus_objective import (
    evaluate_all_objectives_on_designs,
    build_ranking_matrix,
    compute_kendall_tau_matrix,
    compute_objective_weights,
    build_consensus_ranking,
)
from meta_agent_state import (
    MetaAgentState,
    save_meta_agent_state,
    load_meta_agent_state,
    list_meta_agent_history,
    create_meta_agent_state,
    get_latest_weight_adjustments,
)
import workspace_config


@dataclass(frozen=True)
class MetaAgentResult:
    """Result from a meta-agent round."""
    round_id: int
    research_assessment: str
    objective_directions: str
    weight_adjustments: Dict[int, float]
    research_phase: str


# --------------------------------------------------------------------------- #
# File I/O helpers                                                            #
# --------------------------------------------------------------------------- #

def _write_json(path: Path, data: dict) -> None:
    """Write JSON to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Context builders                                                            #
# --------------------------------------------------------------------------- #

def _build_objective_summary_with_code(db: DatabaseManager) -> str:
    """Build summary of all objectives with their descriptions and full code.

    Args:
        db: Database manager

    Returns:
        Formatted string with objective details and code
    """
    available_objectives = objective_loader.list_available_objectives()

    if not available_objectives:
        return "No objectives defined yet."

    all_metadata = objective_metadata.list_objective_metadata()
    objectives_dir = workspace_config.get_objectives_dir()

    lines = []
    for obj_id in sorted(available_objectives):
        metadata = all_metadata.get(str(obj_id), {})
        description = metadata.get("description", "No description available")
        created_round = metadata.get("created_round", "?")

        lines.append(f"### Objective {obj_id}")
        lines.append(f"- Description: {description}")

        # Read and include full objective code
        objective_path = objectives_dir / f"objective_{obj_id}.py"
        if objective_path.exists():
            code = objective_path.read_text(encoding="utf-8")
            lines.append("```python")
            lines.append(code.rstrip())
            lines.append("```")

        lines.append("")

    return "\n".join(lines)


def _build_objective_correlation_matrix(db: DatabaseManager) -> str:
    """Build Kendall tau correlation matrix between objectives.

    Args:
        db: Database manager

    Returns:
        Formatted correlation matrix string
    """
    score_matrix, objective_ids, design_ids = evaluate_all_objectives_on_designs(db)

    if len(objective_ids) < 2:
        return "Not enough objectives for correlation analysis."

    if len(design_ids) < 2:
        return "Not enough designs for correlation analysis."

    # Build ranking and tau matrices
    ranking_matrix = build_ranking_matrix(score_matrix)
    tau_matrix = compute_kendall_tau_matrix(ranking_matrix, objective_ids)

    # Format as table
    lines = ["| Objective |"]
    header_row = "|---|"
    for obj_id in sorted(objective_ids):
        lines[0] += f" {obj_id} |"
        header_row += "---|"
    lines.append(header_row)

    for obj_i in sorted(objective_ids):
        row = f"| {obj_i} |"
        for obj_j in sorted(objective_ids):
            if obj_i == obj_j:
                row += " 1.00 |"
            else:
                tau = tau_matrix.get((obj_i, obj_j), 0.0)
                row += f" {tau:.2f} |"
        lines.append(row)

    return "\n".join(lines)


def _build_objective_agreement_summary(db: DatabaseManager) -> str:
    """Build summary of objective agreement metrics.

    Args:
        db: Database manager

    Returns:
        Formatted agreement summary
    """
    score_matrix, objective_ids, design_ids = evaluate_all_objectives_on_designs(db)

    if len(objective_ids) < 2:
        return "Not enough objectives for agreement analysis."

    # Build ranking and tau matrices
    ranking_matrix = build_ranking_matrix(score_matrix)
    tau_matrix = compute_kendall_tau_matrix(ranking_matrix, objective_ids)

    # Compute weights (which include agreement metrics)
    weights = compute_objective_weights(tau_matrix, objective_ids)

    lines = ["Objective agreement and weights:"]
    for obj_id in sorted(objective_ids):
        # Compute median tau with others
        tau_values = [
            tau_matrix.get((obj_id, other_id), 0.0)
            for other_id in objective_ids
            if other_id != obj_id
        ]

        if tau_values:
            import numpy as np
            median_tau = np.median(tau_values)
            min_tau = min(tau_values)
            max_tau = max(tau_values)
        else:
            median_tau = min_tau = max_tau = 0.0

        weight = weights.get(obj_id, 0.0)

        lines.append(
            f"- Objective {obj_id}: default_weight={weight:.3f}, "
            f"median_tau={median_tau:.2f}, range=[{min_tau:.2f}, {max_tau:.2f}]"
        )

    return "\n".join(lines)


def _build_recent_progress_summary(db: DatabaseManager, limit: int = 10) -> str:
    """Build summary of recent research progress.

    Args:
        db: Database manager
        limit: Maximum designs to include

    Returns:
        Formatted progress summary with raw experiment data
    """
    designs = db.get_all_designs()

    if not designs:
        return "No designs yet."

    # Sort by timestamp (most recent first)
    sorted_designs = sorted(designs, key=lambda d: d.timestamp, reverse=True)[:limit]

    lines = [f"Recent {len(sorted_designs)} designs:"]

    for design in sorted_designs:
        lines.append(f"\n### Design {design.design_id}: {design.short_name}")
        lines.append(f"Objective: {design.objective_id}")
        lines.append("Experiments:")
        lines.append(_format_experiments(design.experiments or []))

    return "\n".join(lines)


def _format_experiments(experiments: List[Dict[str, Any]], max_entries: int = 20) -> str:
    """Format experiment results in a readable way.

    Args:
        experiments: List of experiment dicts
        max_entries: Maximum number of experiments to show

    Returns:
        Formatted experiment string
    """
    if not experiments:
        return "  No experiments"

    lines = []
    if len(experiments) > max_entries:
        lines.append(f"  ... {len(experiments) - max_entries} earlier experiments omitted")

    # Show the final experiments (skip initial ones if too many)
    final_exps = experiments[-max_entries:]
    for exp in final_exps:
        filtered = {k: v for k, v in exp.items() if k != "design_id"}
        parts = []
        for key, val in filtered.items():
            if isinstance(val, float):
                parts.append(f"{key}={val:.3g}")
            else:
                parts.append(f"{key}={val}")
        lines.append("  - " + ", ".join(parts))

    return "\n".join(lines)


def _build_top_designs_summary(db: DatabaseManager) -> str:
    """Build summary of top-performing designs across all objectives.

    Identifies top 3 designs for each objective function and for the
    consensus objective. Lists unique designs ordered by consensus score,
    showing objective values and experiment details for each.

    Args:
        db: Database manager

    Returns:
        Formatted string with top design details
    """
    score_matrix, objective_ids, design_ids = evaluate_all_objectives_on_designs(db)

    if not design_ids:
        return "No designs with experiments yet."
    if not objective_ids:
        return "No objectives defined yet."

    ranking_matrix = build_ranking_matrix(score_matrix)

    # Top 3 per objective
    top_per_objective: Dict[int, List[int]] = {}
    for obj_id in objective_ids:
        if obj_id not in ranking_matrix:
            continue
        rankings = ranking_matrix[obj_id]
        sorted_designs = sorted(rankings.items(), key=lambda x: x[1])
        top_per_objective[obj_id] = [d_id for d_id, _ in sorted_designs[:3]]

    # Consensus ranking with meta weights applied
    tau_matrix = compute_kendall_tau_matrix(ranking_matrix, objective_ids)
    weights = compute_objective_weights(tau_matrix, objective_ids)

    # Apply meta-agent weight multipliers from previous round
    meta_weights = get_latest_weight_adjustments()
    if meta_weights:
        for obj_id in list(weights.keys()):
            multiplier = meta_weights.get(obj_id, 1.0)
            weights[obj_id] *= multiplier
        # Re-normalize to sum to 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

    consensus_ranks = build_consensus_ranking(ranking_matrix, weights, design_ids)

    sorted_by_consensus = sorted(consensus_ranks.items(), key=lambda x: x[1])
    top_consensus_ids = [d_id for d_id, _ in sorted_by_consensus[:3]]

    # Collect unique designs ordered by consensus
    all_top_ids = set(top_consensus_ids)
    for top_list in top_per_objective.values():
        all_top_ids.update(top_list)

    ordered_design_ids = sorted(
        all_top_ids,
        key=lambda d_id: consensus_ranks.get(d_id, float("inf"))
    )

    # Build output
    designs = db.get_all_designs()
    design_map = {d.design_id: d for d in designs}

    lines = [
        f"Top designs across {len(objective_ids)} objectives "
        f"({len(ordered_design_ids)} unique):"
    ]
    lines.append("")

    for design_id in ordered_design_ids:
        design = design_map.get(design_id)
        if design is None:
            continue

        consensus_score = consensus_ranks.get(design_id, float("inf"))
        lines.append(f"### Design {design_id}: {design.short_name}")
        lines.append(f"Consensus score: {consensus_score:.3f}")

        # Which top lists this appears in
        appearances = []
        if design_id in top_consensus_ids:
            appearances.append(f"consensus (#{top_consensus_ids.index(design_id) + 1})")
        for obj_id, top_list in sorted(top_per_objective.items()):
            if design_id in top_list:
                appearances.append(f"obj_{obj_id} (#{top_list.index(design_id) + 1})")
        lines.append(f"Top in: {', '.join(appearances)}")

        # Objective scores
        lines.append("Objective scores:")
        for obj_id in sorted(objective_ids):
            if obj_id in score_matrix and design_id in score_matrix[obj_id]:
                score = score_matrix[obj_id][design_id]
                if score == float("inf"):
                    lines.append(f"  - objective_{obj_id}: inf (failed)")
                else:
                    lines.append(f"  - objective_{obj_id}: {score:.4g}")

        # Experiments
        lines.append("Experiments:")
        lines.append(_format_experiments(design.experiments))
        lines.append("")

    return "\n".join(lines)


def _build_previous_meta_guidance(current_round_id: int) -> str:
    """Build summary of previous meta-agent guidance.

    Args:
        current_round_id: Current meta-agent round ID

    Returns:
        Formatted previous guidance summary
    """
    history = list_meta_agent_history()

    if not history:
        return "No previous meta-agent guidance."

    # Get last few rounds
    recent = history[-3:] if len(history) >= 3 else history

    lines = ["Previous meta-agent assessments:"]
    for state in recent:
        lines.append(f"\n**Round {state.round_id}** ({state.research_phase}):")
        lines.append(f"Assessment: {state.research_assessment[:200]}...")
        lines.append(f"Directions: {state.objective_directions[:200]}...")

        # Show non-default weight adjustments
        non_default = {k: v for k, v in state.weight_adjustments.items() if v != 1.0}
        if non_default:
            lines.append(f"Weight adjustments: {non_default}")

    return "\n".join(lines)


def _load_research_goal() -> str:
    """Load research goal from domain_knowledge/research_goal.txt.

    Returns:
        Research goal string or default message
    """
    goal_path = workspace_config.get_knowledge_dir() / "research_goal.txt"

    if goal_path.exists():
        return goal_path.read_text(encoding="utf-8").strip()

    return "No specific research goal provided. Focus on improving algorithm performance."


# --------------------------------------------------------------------------- #
# Main workflow                                                               #
# --------------------------------------------------------------------------- #

def run_meta_agent_round(
    llm: LLMClient,
    db: DatabaseManager,
    formatter: PromptFormatter,
    meta_round_id: int,
    *,
    research_goal: Optional[str] = None,
    llm_log_dir: Optional[Path] = None,
) -> MetaAgentResult:
    """Run a meta-agent round to analyze research and guide objectives.

    Args:
        llm: LLM client
        db: Database manager
        formatter: Prompt formatter
        meta_round_id: ID for this meta-agent round
        research_goal: Optional override for research goal (otherwise loads from file)
        llm_log_dir: Directory for LLM conversation logs

    Returns:
        MetaAgentResult with assessments, directions, and weight adjustments
    """
    if llm_log_dir is None:
        llm_log_dir = workspace_config.get_llm_log_dir()

    # Load research goal
    if research_goal is None:
        research_goal = _load_research_goal()

    # Build context
    context = formatter.base_context()
    context.update({
        "high_level_research_goal": research_goal,
        "objective_summary_with_code": _build_objective_summary_with_code(db),
        "objective_correlation_matrix": _build_objective_correlation_matrix(db),
        "objective_agreement_summary": _build_objective_agreement_summary(db),
        "recent_progress_summary": _build_recent_progress_summary(db),
        "top_designs_summary": _build_top_designs_summary(db),
        "previous_meta_guidance": _build_previous_meta_guidance(meta_round_id),
    })

    # Format prompts
    system_prompt = formatter.format("meta_agent_system_prompt.md", context)
    analysis_prompt = formatter.format("meta_agent_analysis.md", context)

    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=analysis_prompt),
    ]

    # Request structured analysis
    analysis = llm.request_structured(
        messages=messages,
        response_model=MetaAgentAnalysis,
    )

    # Build weight adjustments from analysis
    weight_adjustments = {}
    for obj_assessment in analysis.objective_analysis:
        weight_adjustments[obj_assessment.objective_id] = obj_assessment.weight_multiplier

    # Create and save state
    state = create_meta_agent_state(
        round_id=meta_round_id,
        research_phase=analysis.research_phase.value,
        research_assessment=analysis.research_assessment,
        objective_directions=analysis.objective_directions,
        weight_adjustments=weight_adjustments,
    )
    save_meta_agent_state(state)

    # Log conversation
    analysis_json = json.dumps(analysis.model_dump(), ensure_ascii=False, indent=2)
    messages_with_response = messages + [
        Message(role="assistant", content=analysis_json),
    ]

    log_data = {
        "meta_round_id": meta_round_id,
        "model_name": llm.model,
        "research_goal": research_goal,
        "analysis": analysis.model_dump(),
        "messages": [msg.as_dict() for msg in messages_with_response],
        "timestamp": create_timestamp(),
    }

    log_dir = llm_log_dir / "meta_agent"
    log_dir.mkdir(parents=True, exist_ok=True)
    _write_json(log_dir / f"round_{meta_round_id}.json", log_data)

    print(f"Meta-agent round {meta_round_id}:")
    print(f"  Research phase: {analysis.research_phase.value}")
    print(f"  Assessment: {analysis.research_assessment[:100]}...")
    print(f"  Directions: {analysis.objective_directions[:100]}...")

    return MetaAgentResult(
        round_id=meta_round_id,
        research_assessment=analysis.research_assessment,
        objective_directions=analysis.objective_directions,
        weight_adjustments=weight_adjustments,
        research_phase=analysis.research_phase.value,
    )


__all__ = [
    "MetaAgentResult",
    "run_meta_agent_round",
]
