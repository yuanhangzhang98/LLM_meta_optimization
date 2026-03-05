"""Helper script: Read database, compute all objectives for all designs, write CSV and summary."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import workspace_config
from database.manager import DatabaseManager
from consensus_objective import (
    evaluate_all_objectives_on_designs,
    create_consensus_objective,
    get_consensus_summary,
)
import objective_metadata


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate all objectives on all designs and write summary"
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Workspace name for run isolation (e.g., 'run_22')"
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Set workspace for all path resolution
    workspace_config.set_workspace(args.workspace)

    # Load database
    database_path = workspace_config.get_database_path()
    db = DatabaseManager(database_path)
    db.load()

    designs = db.get_all_designs()
    print(f"Loaded {len(designs)} designs from database")

    if not designs:
        print("No designs in database!")
        return

    # Evaluate all objectives on all designs
    print("\nEvaluating all objectives on all designs...")
    score_matrix, objective_ids, design_ids = evaluate_all_objectives_on_designs(db)
    print(f"  Objectives: {objective_ids}")
    print(f"  Designs with experiments: {len(design_ids)}")

    # Create consensus objective
    print("\nCreating consensus objective...")
    consensus_fn, stats = create_consensus_objective(db, age_decay_base=0.9)
    print(get_consensus_summary(stats))

    # Prepare output directory
    results_dir = workspace_config.get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)

    # Write CSV file
    csv_path = results_dir / "objectives_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header
        header = [
            "design_id",
            "short_name",
            "planner_round_id",
            "num_experiments",
        ] + [f"objective_{obj_id}" for obj_id in objective_ids] + ["consensus_score"]
        writer.writerow(header)

        # Rows
        for design in designs:
            row = [
                design.design_id,
                design.short_name,
                design.planner_round_id,
                len(design.experiments),
            ]

            # Add objective scores
            for obj_id in objective_ids:
                score = score_matrix.get(obj_id, {}).get(design.design_id, float("inf"))
                row.append(f"{score:.4f}" if score != float("inf") else "inf")

            # Add consensus score
            if design.experiments:
                consensus_score = consensus_fn(design.experiments)
                row.append(f"{consensus_score:.4f}" if consensus_score != float("inf") else "inf")
            else:
                row.append("inf")

            writer.writerow(row)

    print(f"\nCSV written to: {csv_path}")

    # Write summary report
    summary_path = results_dir / "objectives_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("OBJECTIVES EVALUATION SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        # Statistics
        f.write(f"Total designs: {len(designs)}\n")
        f.write(f"Designs with experiments: {len(design_ids)}\n")
        f.write(f"Available objectives: {len(objective_ids)}\n\n")

        # Consensus summary
        f.write(get_consensus_summary(stats))
        f.write("\n\n")

        # Objective descriptions
        f.write("-" * 70 + "\n")
        f.write("OBJECTIVE DESCRIPTIONS\n")
        f.write("-" * 70 + "\n\n")
        for obj_id in objective_ids:
            desc = objective_metadata.get_objective_description(obj_id, default="No description")
            f.write(f"  objective_{obj_id}: {desc}\n")
        f.write("\n")

        # Rankings
        f.write("-" * 70 + "\n")
        f.write("DESIGN RANKINGS (by consensus objective, lower is better)\n")
        f.write("-" * 70 + "\n\n")

        # Compute consensus scores for all designs
        design_scores = []
        for design in designs:
            if design.experiments:
                score = consensus_fn(design.experiments)
                design_scores.append((design.design_id, design.short_name, score))

        # Sort by score (lower is better)
        design_scores.sort(key=lambda x: x[2] if x[2] != float("inf") else float("inf"))

        for rank, (did, name, score) in enumerate(design_scores, 1):
            score_str = f"{score:.4f}" if score != float("inf") else "inf"
            f.write(f"  {rank:3d}. Design {did:3d} ({name:30s}): {score_str}\n")

    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
