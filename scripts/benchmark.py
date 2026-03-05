"""Benchmark script to test a solver on a schedule.

Usage:
    python scripts/benchmark.py --solver-id 0 --schedule-id 0 --fidelity low
    python scripts/benchmark.py --solver-id 0 --schedule-id 0 --timeout 120 --n-cap 640
    python scripts/benchmark.py --solver-id 0 --schedule-id 0 --fidelity high --output results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import workspace_config
from schedule_loader import load_schedule, load_schedule_fidelities, list_available_schedules


def validate_solver(solver_id: int) -> bool:
    """Check if solver file exists."""
    solver_path = workspace_config.get_solvers_dir() / f"solver_{solver_id}.py"
    return solver_path.exists()


def list_available_solvers() -> list[int]:
    """List all available solver IDs."""
    solvers_dir = workspace_config.get_solvers_dir()
    if not solvers_dir.exists():
        return []

    solver_ids = []
    for solver_file in solvers_dir.glob("solver_*.py"):
        try:
            solver_id = int(solver_file.stem.split("_")[1])
            solver_ids.append(solver_id)
        except (IndexError, ValueError):
            continue

    return sorted(solver_ids)


def print_results_table(experiments: list[dict], solver_id: int, schedule_id: int, fidelity: str | None) -> None:
    """Print experiment results as a formatted table."""
    fidelity_str = f" ({fidelity} fidelity)" if fidelity else ""
    print(f"\nBenchmark: Solver {solver_id} on Schedule {schedule_id}{fidelity_str}")
    print("=" * 60)
    print(f"{'N':>6} | {'max_steps':>9} | {'unsolved':>8} | {'median_step':>11}")
    print("-" * 6 + "-+-" + "-" * 9 + "-+-" + "-" * 8 + "-+-" + "-" * 11)

    for exp in experiments:
        N = exp.get('N', 0)
        max_steps = exp.get('max_steps', 0)
        unsolved = exp.get('unsolved_fraction', 0.0)
        median = exp.get('median_step', 0)
        print(f"{N:>6} | {max_steps:>9} | {unsolved:>8.2f} | {median:>11}")

    print("=" * 60)

    if experiments:
        max_N = max(e.get('N', 0) for e in experiments)
        min_unsolved = min(e.get('unsolved_fraction', 1.0) for e in experiments)
        print(f"Summary: Reached N={max_N}, {len(experiments)} experiments, best unsolved={min_unsolved:.2f}")


def save_results_json(experiments: list[dict], solver_id: int, schedule_id: int,
                      fidelity: str | None, output_path: Path) -> None:
    """Save experiment results to JSON file."""
    result = {
        "solver_id": solver_id,
        "schedule_id": schedule_id,
        "fidelity": fidelity,
        "experiments": experiments,
        "summary": {
            "max_N": max(e.get('N', 0) for e in experiments) if experiments else 0,
            "num_experiments": len(experiments),
            "min_unsolved": min(e.get('unsolved_fraction', 1.0) for e in experiments) if experiments else 1.0,
        }
    }

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark a solver on a schedule",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/benchmark.py --solver-id 0 --schedule-id 0 --fidelity low
  python scripts/benchmark.py --solver-id 0 --schedule-id 0 --timeout 120 --n-cap 640
  python scripts/benchmark.py --solver-id 0 --schedule-id 0 --fidelity high --output results.json
        """
    )

    parser.add_argument(
        "--solver-id",
        type=int,
        required=True,
        help="Design ID of the solver to benchmark"
    )
    parser.add_argument(
        "--schedule-id",
        type=int,
        default=99999,
        help="Schedule ID to use (default: 99999)"
    )
    parser.add_argument(
        "--fidelity",
        type=str,
        choices=["low", "medium", "high"],
        default=None,
        help="Fidelity level. If not specified, uses custom params via --timeout and --n-cap"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=86400,
        help="Timeout in seconds (default: 86400, only used without --fidelity)"
    )
    parser.add_argument(
        "--n-cap",
        type=int,
        default=2560,
        help="Maximum problem size N (default: 2560, only used without --fidelity)"
    )
    parser.add_argument(
        "--max-steps-start",
        type=int,
        default=500,
        help="Maximum steps to run (default: 500, only used without --fidelity)"
    )
    parser.add_argument(
        "--max-steps-cap",
        type=int,
        default=10000000,
        help="Maximum steps to run (default: 10000000, only used without --fidelity)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for JSON results (optional)"
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Workspace name for run isolation (e.g., 'run_22')"
    )

    args = parser.parse_args()

    # Set workspace for all path resolution
    workspace_config.set_workspace(args.workspace)

    # Validate solver exists
    if not validate_solver(args.solver_id):
        available = list_available_solvers()
        print(f"Error: Solver {args.solver_id} not found.")
        print(f"Available solvers: {available}")
        sys.exit(1)

    # Validate schedule exists
    available_schedules = list_available_schedules()
    if args.schedule_id not in available_schedules:
        print(f"Error: Schedule {args.schedule_id} not found.")
        print(f"Available schedules: {available_schedules}")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        results_dir = workspace_config.get_results_dir()
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / f"benchmark_solver_{args.solver_id}_schedule_{args.schedule_id}.json"

    # Load and run schedule
    if args.fidelity:
        fidelities = load_schedule_fidelities(args.schedule_id)
        schedule_fn = fidelities[args.fidelity]
        print(f"Running {args.fidelity} fidelity schedule {args.schedule_id} on solver {args.solver_id}...")
        results = schedule_fn(design_id=args.solver_id)
    else:
        schedule_fn = load_schedule(args.schedule_id)
        print(f"Running schedule {args.schedule_id} on solver {args.solver_id} (timeout={args.timeout}s, N_cap={args.n_cap})...")
        results = schedule_fn(
            design_id=args.solver_id,
            N_cap=args.n_cap,
            timeout=args.timeout,
            max_steps_start=args.max_steps_start,
            max_steps_cap=args.max_steps_cap
        )

    # Consume results — supports both generators (streaming) and lists
    from types import GeneratorType
    if isinstance(results, GeneratorType):
        fidelity_str = f" ({args.fidelity} fidelity)" if args.fidelity else ""
        print(f"\nBenchmark: Solver {args.solver_id} on Schedule {args.schedule_id}{fidelity_str}")
        print("=" * 60)
        print(f"{'N':>6} | {'max_steps':>9} | {'unsolved':>8} | {'median_step':>11}")
        print("-" * 6 + "-+-" + "-" * 9 + "-+-" + "-" * 8 + "-+-" + "-" * 11)

        experiments = []
        for exp in results:
            experiments.append(exp)
            N = exp.get('N', 0)
            max_steps = exp.get('max_steps', 0)
            unsolved = exp.get('unsolved_fraction', 0.0)
            median = exp.get('median_step', 0)
            print(f"{N:>6} | {max_steps:>9} | {unsolved:>8.2f} | {median:>11}")
            save_results_json(experiments, args.solver_id, args.schedule_id, args.fidelity, output_path)

        print("=" * 60)
        if experiments:
            max_N = max(e.get('N', 0) for e in experiments)
            min_unsolved = min(e.get('unsolved_fraction', 1.0) for e in experiments)
            print(f"Summary: Reached N={max_N}, {len(experiments)} experiments, best unsolved={min_unsolved:.2f}")
    else:
        experiments = results
        print_results_table(experiments, args.solver_id, args.schedule_id, args.fidelity)
        save_results_json(experiments, args.solver_id, args.schedule_id, args.fidelity, output_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
