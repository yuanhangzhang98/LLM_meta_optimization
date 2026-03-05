"""Initialize metadata for baseline objectives and schedules.

This script creates initial metadata.json files for objective 0 and schedule 0
based on the baseline files in domain_knowledge/.

Run this once during setup or when metadata files need to be regenerated.
"""

from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import objective_metadata
import schedule_metadata


def initialize_baseline_metadata():
    """Initialize metadata for baseline objective and schedules."""

    print("Initializing baseline metadata...")

    # Initialize objective 0 metadata
    objective_metadata.save_objective_metadata(
        objective_id=0,
        description="Baseline objective - minimizes the primary evaluation metric",
        round_id=0,
    )
    print("✓ Created metadata for objective 0")

    # Initialize schedule 0 metadata (single baseline schedule with all fidelities)
    schedule_metadata.save_schedule_metadata(
        schedule_id=0,
        description="Baseline schedule with adaptive multi-fidelity evaluation",
        fidelities=["low", "medium", "high"],
        round_id=0,
    )
    print("✓ Created metadata for schedule 0 (low/medium/high)")

    print("\nMetadata initialization complete!")
    print(f"  - Objective metadata: {objective_metadata.METADATA_FILE}")
    print(f"  - Schedule metadata: {schedule_metadata.METADATA_FILE}")


if __name__ == "__main__":
    initialize_baseline_metadata()
