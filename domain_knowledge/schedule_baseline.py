"""
**unsolved_fraction=0.49 is an intended behavior!!! DO NOT penalize it.**
**Each experiment concludes when unsolved_fraction < 0.5 or max_steps is exceeded**

Baseline experiment schedule for adaptive multi-fidelity evaluation.

This module defines how to progressively test a design across increasing problem scales.
The schedule adaptively increases problem difficulty based on performance at each level.
"""

from utils.timeout import DeferredTimeout
from domain_knowledge.experiment import experiment


def schedule(design_id=0, N_start=10, max_steps_start=50, N_cap=5120, max_steps_cap=10000, timeout=60, prior_experiments=None):
    """
    Adaptive experiment schedule that progressively scales problem difficulty.
    Move on to the next N when unsolved_fraction < 0.5 for current N. 

    Args:
        design_id: Identifier for the design being evaluated
        N_start: Initial problem size
        max_steps_start: Initial step budget
        N_cap: Maximum problem size to test
        max_steps_cap: Maximum step budget allowed
        timeout: Maximum wall-clock time in seconds
        prior_experiments: Optional list of prior experiment results from lower fidelity.
            If provided, skips experiments at N values that already succeeded (unsolved_fraction < 0.5).

    Returns:
        List of experiment results
    """
    batch = 100
    ratio = 4.3
    run_till_median = True

    # Determine starting point based on prior successful experiments
    if prior_experiments:
        successful = [e for e in prior_experiments if e.get('unsolved_fraction', 1.0) < 0.5]

        if successful:
            max_successful_N = max(e['N'] for e in successful)

            # Find experiment_count for max_successful_N by iteration (robust to rounding)
            success_count = 0
            while int(round(N_start * (2 ** (success_count / 2)))) < max_successful_N:
                success_count += 1

            # Start from the next experiment (skip the last successful one)
            experiment_count = success_count + 1
            N = int(round(N_start * (2 ** (experiment_count / 2))))

            # Default max_steps based on experiment_count
            max_steps = max_steps_start * (2 ** experiment_count)

            # If there was a failed experiment at this N, use double its max_steps
            failed_at_N = [e for e in prior_experiments
                           if e['N'] == N and e.get('unsolved_fraction', 1.0) >= 0.5]
            if failed_at_N:
                prior_max_steps = max(e.get('max_steps', max_steps_start) for e in failed_at_N)
                max_steps = max(max_steps, prior_max_steps * 2)

            max_steps = min(max_steps, max_steps_cap)
        else:
            experiment_count = 0
            N = N_start
            max_steps = max_steps_start
    else:
        experiment_count = 0
        N = N_start
        max_steps = max_steps_start

    experiment_results = []

    with DeferredTimeout(timeout) as timer:
        while N <= N_cap:
            kwargs = {
                'design_id': design_id,
                'batch': batch,
                'N': N,
                'ratio': ratio,
                'run_till_median': run_till_median,
                'max_steps': max_steps
            }

            result_i = experiment(**kwargs)

            experiment_results.append(kwargs | result_i)

            # Check timeout AFTER experiment completes
            if timer.timed_out:
                break

            unsolved_fraction = result_i['unsolved_fraction']

            # Adaptive scaling: continue if performance is good (< 50% unsolved)
            if unsolved_fraction < 0.5:
                experiment_count += 1
                N = int(round(N_start * (2 ** (experiment_count/2))))
                max_steps = min(max_steps * 2, max_steps_cap)
            else:
                break

    return experiment_results


def schedule_low(design_id, prior_experiments=None):
    return schedule(design_id=design_id, N_start=10, max_steps_start=50, N_cap=640, max_steps_cap=int(1e4), timeout=60, prior_experiments=prior_experiments)

def schedule_medium(design_id, prior_experiments=None):
    return schedule(design_id=design_id, N_start=10, max_steps_start=50, N_cap=1280, max_steps_cap=int(1e5), timeout=300, prior_experiments=prior_experiments)

def schedule_high(design_id, prior_experiments=None):
    return schedule(design_id=design_id, N_start=10, max_steps_start=50, N_cap=2560, max_steps_cap=int(1e6), timeout=1800, prior_experiments=prior_experiments)
