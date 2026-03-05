import os, argparse, json, time
import numpy as np
from pathlib import Path


results_dir = Path.cwd() / "experiment_log"
os.makedirs(results_dir, exist_ok=True)


def metric(experiment_id=0, hp=None, solver_id=None):
    # Set SOLVER_ID environment variable if provided
    # This allows sat_solver.py to import from the correct solver module
    if solver_id is not None:
        os.environ['SOLVER_ID'] = str(solver_id)

        # Force reload of sat_solver to pick up new solver components
        import sys
        import importlib

        # Remove cached solver modules (both old package-style and new direct-loaded)
        modules_to_remove = [k for k in sys.modules.keys() if 'sat_solver' in k or k.startswith('solvers.solver_') or k.startswith('_solver_') or '_loaded_solvers.' in k]
        for mod in modules_to_remove:
            del sys.modules[mod]

    # Import after setting environment variable so sat_solver picks up correct components
    # Try relative import first (when used as module), fall back to absolute (when run directly)
    try:
        from .sat_solver import run_solver
    except ImportError:
        from sat_solver import run_solver

    # New scaling scheme:
    # N doubles each iteration: 10, 20, 40, 80, 160, 320, ...
    # max_steps: 50, 200, 800, 3200, 10000, 10000, ... (4x each time, capped at 10000)
    # Run each experiment until median is found or max_steps is reached
    # metric = projected max solved N (same definition as before)

    batch = 100
    ratio = 4.3

    Ns = []
    medians = []
    max_solved_N = 0

    # Initial values
    N = 10
    max_steps = 50

    # Store first run results for logging
    unsolved_fraction_0 = None
    median_step_0 = None

    # Run scaling experiments with doubling N and 4x max_steps
    while True:
        unsolved_fraction, median_step = run_solver(
            batch=batch,
            N=N,
            ratio=ratio,
            run_till_median=True,  # Always run until median or max_steps
            max_steps=max_steps,
            verbose=True,
            log_interval=max(100, max_steps // 10),  # Log ~10 times per run
            hp=hp
        )

        Ns.append(N)

        # Store first run results
        if unsolved_fraction_0 is None:
            unsolved_fraction_0 = unsolved_fraction
            median_step_0 = median_step

        if unsolved_fraction < 0.5:
            # Found median solution
            print(f"N={N}  Median steps to solve: {median_step}")
            max_solved_N = N
            medians.append(median_step)

            # Double N and increase max_steps by 4x (capped at 10000)
            N = N * 2
            max_steps = min(max_steps * 4, 10000)
        else:
            # Reached max_steps without finding median
            projected_median = int(max_steps * 0.5 / (1.0 - unsolved_fraction))
            medians.append(projected_median)
            print(f"N={N}  Reached maximum steps ({max_steps}) without finding a median solution. "
                  f"Projected median step: {projected_median}.")
            break

    # Compute power law statistics if we have multiple data points
    if len(Ns) > 1:
        slope = np.polyfit(np.log10(np.array(Ns)), np.log10(np.array(medians)), 1)[0]
        concavity = np.polyfit(np.log10(np.array(Ns)), np.log10(np.array(medians)), 2)[0]

        print(f"Power law exponent (lower is better): {slope:.2f}, "
              f"Concavity in log-log space (lower is better): {concavity:.2f}")
    else:
        slope = None
        concavity = None

    # Project maximum solved N using same formula as before
    projected_max_solved_N = np.round(max_solved_N + (1.0 - unsolved_fraction) / 0.5 * (N - max_solved_N))

    # Save results
    result_path = results_dir / f"experiment_results_{experiment_id}.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump({
            "batch": batch,
            "ratio": ratio,
            "unsolved_fraction": unsolved_fraction_0,
            "median_step": median_step_0,
            "Ns": Ns,
            "medians": medians,
            "slope": slope,
            "concavity": concavity,
            "metric": projected_max_solved_N
        }, f, indent=2, ensure_ascii=False)

    # Clean up environment variable
    if solver_id is not None:
        os.environ.pop('SOLVER_ID', None)

    return projected_max_solved_N


if __name__ == "__main__":
    t0 = time.time()
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=int, default=0, help="Index for the run (for logging purposes)")
    args = p.parse_args()
    metric_value = metric(experiment_id=args.index)
