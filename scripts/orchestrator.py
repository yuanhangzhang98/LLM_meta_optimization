"""High-level orchestrator for planner/designer iterations with multi-fidelity optimization.

Simplified design:
- Objective evolution is periodic (every N iterations), not agent-decided
- Fidelity levels map to predefined schedules: low→0, medium→1, high→2
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Optional, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import workspace_config  # type: ignore
from database.manager import DatabaseManager  # type: ignore
from llm_client import LLMClient  # type: ignore
from prompt_formatter import PromptFormatter  # type: ignore
from workflow.iteration import run_iteration, run_planner_round, run_designer_iteration  # type: ignore
from workflow.objective_scheduler import run_objective_schedule_round  # type: ignore
from workflow.meta_agent import run_meta_agent_round  # type: ignore
from scripts.hebo_tune import maybe_run_tuning  # type: ignore
from consensus_objective import create_consensus_objective, get_consensus_summary  # type: ignore
from meta_agent_state import get_effective_meta_weights, get_latest_objective_directions  # type: ignore
import objective_metadata  # type: ignore
import schedule_metadata  # type: ignore


# --------------------------------------------------------------------------- #
# Baseline Benchmark                                                          #
# --------------------------------------------------------------------------- #

def benchmark_baseline(db: DatabaseManager, schedule_id: int = 0, objective_id: int = 0) -> None:
    """Benchmark the baseline solver and log results to database.

    Only runs if no designs exist in the database. Creates design_id=0
    with high-fidelity experiment results from the baseline solver.

    Args:
        db: Database manager
        schedule_id: Schedule ID to use (default: 0)
        objective_id: Objective ID to record (default: 0)
    """
    # Skip if designs already exist
    if db.count_designs() > 0:
        print(f"Database has {db.count_designs()} designs, skipping baseline benchmark")
        return

    print("\n" + "="*70)
    print("BASELINE BENCHMARK")
    print("="*70)
    print("Running high-fidelity benchmark on baseline solver (design_id=0)...")

    # Import schedule function
    from domain_knowledge.schedule_baseline import schedule_high

    # Run high-fidelity schedule
    experiment_results = schedule_high(design_id=0)

    print(f"Baseline benchmark complete: {len(experiment_results)} experiments")
    for exp in experiment_results:
        print(f"  N={exp['N']:4d}: median_step={exp['median_step']:6d}, unsolved={exp['unsolved_fraction']:.2f}")

    # Add baseline design to database
    db.add_design(
        design_id=0,
        planner_round_id=0,  # Round 0 = baseline
        reference_weights=[],  # No parents
        short_name="baseline",
        model_name="baseline",  # Not LLM-generated
        code_path="solvers/solver_0.py",
        planner_log="",  # No log
        designer_log="",  # No log
        schedule_id=schedule_id,
        objective_id=objective_id,
    )

    # Add experiment results (flat format, matching designer workflow)
    for exp in experiment_results:
        exp['fidelity'] = 'high'
        db.add_experiment_to_design(design_id=0, experiment_data=exp)

    print(f"Baseline design added to database (design_id=0)")
    print("="*70 + "\n")


# --------------------------------------------------------------------------- #
# Workspace Initialization                                                    #
# --------------------------------------------------------------------------- #

def initialize_workspace(db: DatabaseManager) -> None:
    """Initialize workspace with baseline files and experiments.

    Copies:
    1. Baseline code files from domain_knowledge/ to workspace directories
    2. Baseline experiment results from database/database.json

    This avoids re-running expensive baseline experiments when creating
    a new workspace.
    """
    # Skip if database already has designs
    if db.count_designs() > 0:
        return

    print("\n" + "="*70)
    print("WORKSPACE INITIALIZATION")
    print("="*70)

    # Get workspace paths
    solvers_dir = workspace_config.get_solvers_dir()
    objectives_dir = workspace_config.get_objectives_dir()
    schedules_dir = workspace_config.get_schedules_dir()
    knowledge_dir = workspace_config.get_knowledge_dir()

    # Create directories
    objectives_dir.mkdir(parents=True, exist_ok=True)
    schedules_dir.mkdir(parents=True, exist_ok=True)

    # Copy baseline files
    shutil.copy(knowledge_dir / "solver_baseline.py", solvers_dir / "solver_0.py")
    shutil.copy(knowledge_dir / "objective_baseline.py", objectives_dir / "objective_0.py")
    shutil.copy(knowledge_dir / "schedule_baseline.py", schedules_dir / "schedule_0.py")
    print(f"Copied baseline files to workspace:")
    print(f"  - {solvers_dir / 'solver_0.py'}")
    print(f"  - {objectives_dir / 'objective_0.py'}")
    print(f"  - {schedules_dir / 'schedule_0.py'}")

    # Initialize baseline metadata (load descriptions from domain_knowledge/)
    objective_desc = (knowledge_dir / "objective_baseline_description.txt").read_text().strip()
    schedule_desc = (knowledge_dir / "schedule_baseline_description.txt").read_text().strip()
    objective_metadata.save_objective_metadata(
        objective_id=0,
        description=objective_desc,
        round_id=0,
    )
    schedule_metadata.save_schedule_metadata(
        schedule_id=0,
        description=schedule_desc,
        fidelities=["low", "medium", "high"],
        round_id=0,
    )
    print(f"Initialized baseline metadata:")
    print(f"  - {objectives_dir / 'metadata.json'}")
    print(f"  - {schedules_dir / 'metadata.json'}")

    # Copy baseline experiments from default database
    default_db_path = workspace_config.get_root() / "database" / "database.json"
    if default_db_path.exists():
        with open(default_db_path) as f:
            default_data = json.load(f)

        # Find baseline design (design_id=0)
        for design in default_data.get("designs", []):
            if design.get("design_id") == 0:
                # Add baseline design to workspace database
                db.add_design(
                    design_id=0,
                    planner_round_id=0,
                    reference_weights=[],
                    short_name="baseline",
                    model_name="baseline",
                    code_path="solvers/solver_0.py",
                    planner_log="",
                    designer_log="",
                    schedule_id=0,
                    objective_id=0,
                )
                # Add experiments
                experiments = design.get("experiments", [])
                for exp in experiments:
                    db.add_experiment_to_design(design_id=0, experiment_data=exp)
                print(f"Copied baseline design with {len(experiments)} experiments from database/database.json")
                break
        else:
            print("Warning: No baseline design (design_id=0) found in database/database.json")
    else:
        print(f"Warning: Default database not found at {default_db_path}")

    print("="*70 + "\n")


# --------------------------------------------------------------------------- #
# Main Orchestrator                                                           #
# --------------------------------------------------------------------------- #

def main():
    """Main orchestration loop with simplified multi-fidelity workflow.

    Flow:
    - Regular iterations: Planner → Designer → Execute with fidelity-mapped schedule
    - Objective iterations (periodic): Same + Objective round to create new objective
    """
    args = parse_args()

    # Set workspace for all path resolution
    workspace_config.set_workspace(args.workspace)

    # Setup paths using workspace_config
    database_path = workspace_config.get_database_path()
    prompt_dir = workspace_config.get_prompts_dir()
    knowledge_dir = workspace_config.get_knowledge_dir()
    llm_log_dir = workspace_config.get_llm_log_dir()
    solvers_dir = workspace_config.get_solvers_dir()

    # Create directories
    database_path.parent.mkdir(parents=True, exist_ok=True)
    llm_log_dir.mkdir(parents=True, exist_ok=True)
    solvers_dir.mkdir(parents=True, exist_ok=True)

    # Initialize database
    db = DatabaseManager(database_path)
    db.load()

    # Initialize workspace with baseline files (if new workspace)
    initialize_workspace(db)

    # Fallback: run benchmark if initialize_workspace didn't populate baseline
    # (e.g., if database/database.json doesn't exist)
    benchmark_baseline(db, schedule_id=args.initial_schedule_id, objective_id=args.initial_objective_id)

    # Determine schedule generation setting
    # --enable-schedule-generation overrides --disable-schedule-generation
    generate_schedule = args.enable_schedule_generation and not args.disable_schedule_generation

    # Raise exception if schedule generation is enabled (feature deferred)
    if generate_schedule:
        raise NotImplementedError(
            "Schedule generation is disabled in this version. "
            "The meta-agent currently only supports objective guidance. "
            "Schedule generation will be implemented in a future version. "
            "Use --disable-schedule-generation (default) to proceed."
        )

    # Initialize multi-fidelity tracking
    current_objective_id = args.initial_objective_id
    current_schedule_id = args.initial_schedule_id
    objective_round_counter = 0
    meta_round_counter = 0

    # Determine meta-agent frequency
    meta_agent_frequency = args.meta_agent_frequency or args.objective_frequency

    print(f"Starting orchestrator with {args.iterations} iterations")
    if args.workspace:
        print(f"Workspace: {args.workspace}")
    print(f"Database: {database_path}")
    print(f"Current designs: {db.count_designs()}")
    print(f"Objective frequency: every {args.objective_frequency} iterations")
    print(f"Initial objective_id: {current_objective_id}")
    print(f"Schedule generation: disabled (using baseline)")
    print(f"Using consensus objective (aggregates all objectives with Kendall tau weighting)")
    print(f"Promotion thresholds: medium<{args.promote_medium_threshold}, high<{args.promote_high_threshold}")
    if args.enable_meta_agent:
        print(f"Meta-agent: enabled (frequency: every {meta_agent_frequency} iterations)")
        if args.research_goal:
            print(f"Research goal: {args.research_goal[:80]}...")
        else:
            print(f"Research goal: from domain_knowledge/research_goal.txt")

    # Initialize LLM clients
    llm_clients = [LLMClient(model=model) for model in args.models]
    if not llm_clients:
        llm_clients = [LLMClient(model="gpt-5.2")]

    # Main iteration loop
    for iteration in range(1, args.iterations + 1):
        print(f"\n{'='*70}")
        print(f"Iteration {iteration}/{args.iterations}")
        print(f"{'='*70}")

        # Rotate through LLM models
        llm = llm_clients[(iteration - 1) % len(llm_clients)]
        print(f"Using model: {llm.model}")
        print(f"Current objective_id: {current_objective_id}")

        # Get meta weights if meta-agent is enabled
        meta_weights = None
        if args.enable_meta_agent:
            meta_weights = get_effective_meta_weights()
            if meta_weights:
                print(f"Applying meta-agent weights: {len(meta_weights)} objectives adjusted")

        # Create consensus objective (recomputed each iteration)
        consensus_fn, consensus_stats = create_consensus_objective(
            db, age_decay_base=0.9, meta_weights=meta_weights
        )
        print(get_consensus_summary(consensus_stats))

        # Create prompt formatter with current database state and consensus objective
        formatter = PromptFormatter(
            prompt_dir=prompt_dir,
            knowledge_dir=knowledge_dir,
            database_manager=db,
            objective_fn=consensus_fn,
            num_designers=args.num_designers,
        )

        # Check if this is an objective round (periodic)
        is_objective_round = (iteration % args.objective_frequency == 0) and (iteration > 0)

        # Step 1: Run planner round
        print(f"\n[PLANNER] Running planner round")
        planner_result = run_planner_round(
            llm=llm,
            db=db,
            formatter=formatter,
            llm_log_dir=llm_log_dir,
            num_designers=args.num_designers,
        )
        print(f"Planner round: {planner_result.round_id}")
        print(f"Research phase: {planner_result.summary.current_phase}")

        # Step 2: Run designer iterations
        print(f"\n[DESIGNER] Running designer iterations")
        design_ids = run_designer_iteration(
            llm=llm,
            db=db,
            formatter=formatter,
            planner=planner_result,
            llm_log_dir=llm_log_dir,
            solvers_dir=solvers_dir,
            max_retries=3,
            schedule_id=current_schedule_id,
            objective_id=current_objective_id,
            promote_medium_threshold=args.promote_medium_threshold,
            promote_high_threshold=args.promote_high_threshold,
        )
        print(f"Created designs: {design_ids}")

        # HEBO tuning (uses same promotion thresholds as designer)
        maybe_run_tuning(
            db=db,
            llm=llm,
            formatter=formatter,
            llm_log_dir=llm_log_dir,
            solvers_dir=solvers_dir,
            max_iter=args.hebo_max_iter,
            suggestions=args.hebo_suggestions,
            max_tuning_ratio=args.hebo_max_tuning_ratio,
            schedule_id=current_schedule_id,
            promote_medium_threshold=args.promote_medium_threshold,
            promote_high_threshold=args.promote_high_threshold,
        )

        # Step 3: Meta-agent round (before objective round, if enabled)
        meta_directions = None
        is_meta_round = (iteration % meta_agent_frequency == 0) and (iteration > 0)

        if is_meta_round and args.enable_meta_agent:
            meta_round_counter += 1
            print(f"\n[META-AGENT] Running meta-agent round {meta_round_counter}")

            # Rebuild formatter with latest state
            consensus_fn, consensus_stats = create_consensus_objective(
                db, age_decay_base=0.9, meta_weights=meta_weights
            )
            formatter = PromptFormatter(
                prompt_dir=prompt_dir,
                knowledge_dir=knowledge_dir,
                database_manager=db,
                objective_fn=consensus_fn,
                num_designers=args.num_designers,
            )

            meta_result = run_meta_agent_round(
                llm=llm,
                db=db,
                formatter=formatter,
                meta_round_id=meta_round_counter,
                research_goal=args.research_goal,
                llm_log_dir=llm_log_dir,
            )

            # Get directions for objective generation
            meta_directions = meta_result.objective_directions

            # Update meta weights for next iteration
            # (weights will be applied in next iteration's consensus)
            print(f"Meta-agent phase: {meta_result.research_phase}")

        # Step 4: Periodic objective round
        if is_objective_round:
            objective_round_counter += 1
            print(f"\n[OBJECTIVE] Running periodic round {objective_round_counter}")

            # Get meta directions if available (from current or previous run)
            if meta_directions is None and args.enable_meta_agent:
                meta_directions = get_latest_objective_directions()

            # Rebuild formatter with latest state using consensus objective
            consensus_fn, consensus_stats = create_consensus_objective(
                db, age_decay_base=0.9, meta_weights=meta_weights
            )
            formatter = PromptFormatter(
                prompt_dir=prompt_dir,
                knowledge_dir=knowledge_dir,
                database_manager=db,
                objective_fn=consensus_fn,
                num_designers=args.num_designers,
            )

            objective_result = run_objective_schedule_round(
                llm=llm,
                db=db,
                formatter=formatter,
                current_objective_id=current_objective_id,
                objective_round_id=objective_round_counter,
                llm_log_dir=llm_log_dir,
                generate_schedule=False,  # Schedule generation disabled
                meta_directions=meta_directions,
            )

            if objective_result.new_objective_id is not None:
                print(f"[OBJECTIVE] New objective created: objective_{objective_result.new_objective_id}")
                # Update current objective ID (consensus will include it in next iteration)
                current_objective_id = objective_result.new_objective_id

                # Only update schedule_id if a new schedule was created
                if objective_result.new_schedule_id is not None:
                    print(f"[SCHEDULE] New schedule created: schedule_{objective_result.new_schedule_id}")
                    current_schedule_id = objective_result.new_schedule_id
            else:
                print(f"[OBJECTIVE] Failed to create new objective")

        print(f"\nCompleted iteration {iteration}")
        print(f"Total designs in database: {db.count_designs()}")


    print(f"\n{'='*70}")
    print(f"Orchestration complete!")
    print(f"Total designs: {db.count_designs()}")
    print(f"Final objective_id: {current_objective_id}")
    print(f"Database saved to: {database_path}")
    print(f"{'='*70}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Orchestrate planner/designer iterations with multi-fidelity optimization"
    )

    # Workspace configuration
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Workspace name for run isolation (e.g., 'run_22'). "
             "Prefixes output directories: database_run_22/, solvers_run_22/, etc. "
             "If not specified, uses default paths for backwards compatibility.",
    )

    # Basic iteration settings
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of planner/designer iteration rounds (default: 10)",
    )
    parser.add_argument(
        "--num-designers",
        type=int,
        default=2,
        help="Number of designer agents per planner round (default: 2)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-5.2"],
        help="LLM models to use (rotates through them). Default: gpt-5.2",
    )

    # Multi-fidelity settings
    parser.add_argument(
        "--objective-frequency",
        type=int,
        default=10,
        help="Create new objective every N iterations (default: 10)",
    )
    parser.add_argument(
        "--initial-objective-id",
        type=int,
        default=0,
        help="Initial objective function ID (default: 0)",
    )
    parser.add_argument(
        "--initial-schedule-id",
        type=int,
        default=0,
        help="Initial schedule ID (default: 0)",
    )
    parser.add_argument(
        "--disable-schedule-generation",
        action="store_true",
        default=True,
        help="Disable schedule generation, always use baseline schedule (default: True)",
    )
    parser.add_argument(
        "--enable-schedule-generation",
        action="store_true",
        help="Enable schedule generation (overrides --disable-schedule-generation)",
    )

    # Meta-agent settings
    parser.add_argument(
        "--enable-meta-agent",
        action="store_true",
        default=True,
        help="Enable meta-agent for objective guidance and weight adjustment (default: True)",
    )
    parser.add_argument(
        "--research-goal",
        type=str,
        default=None,
        help="High-level research goal for meta-agent guidance. "
             "If not provided, reads from domain_knowledge/research_goal.txt",
    )
    parser.add_argument(
        "--meta-agent-frequency",
        type=int,
        default=None,
        help="Run meta-agent every N iterations (default: same as objective-frequency)",
    )

    # Fidelity promotion thresholds
    parser.add_argument(
        "--promote-medium-threshold",
        type=float,
        default=0.5,
        help="Consensus objective threshold to promote low→medium (default: 0.5, top 50%%)",
    )
    parser.add_argument(
        "--promote-high-threshold",
        type=float,
        default=0.1,
        help="Consensus objective threshold to promote medium→high (default: 0.1, top 10%%)",
    )

    # HEBO tuning settings
    parser.add_argument(
        "--hebo-max-iter",
        type=int,
        default=50,
        help="Maximum HEBO optimization iterations (default: 50)",
    )
    parser.add_argument(
        "--hebo-suggestions",
        type=int,
        default=1,
        help="Number of HEBO suggestions per iteration (default: 1)",
    )
    parser.add_argument(
        "--hebo-max-tuning-ratio",
        type=float,
        default=0.1,
        help="Maximum ratio of tuned designs to total designs (default: 0.1)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nOrchestrator failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
