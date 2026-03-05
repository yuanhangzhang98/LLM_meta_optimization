"""
Generator-based experiment schedule for streaming benchmark results.

Identical logic to schedule_0, but yields each experiment result as it completes
instead of collecting and returning all at once. This allows the caller to
print/save partial results incrementally.
"""

from utils.timeout import DeferredTimeout
from domain_knowledge.experiment import experiment


def schedule(design_id=0, N_start=10, max_steps_start=50, N_cap=5120, max_steps_cap=10000, timeout=60, prior_experiments=None):
    """
    Generator version of the adaptive experiment schedule.
    Yields one experiment result dict at a time.
    """
    batch = 100
    ratio = 4.3
    run_till_median = True

    # Determine starting point based on prior successful experiments
    if prior_experiments:
        successful = [e for e in prior_experiments if e.get('unsolved_fraction', 1.0) < 0.5]

        if successful:
            max_successful_N = max(e['N'] for e in successful)

            success_count = 0
            while int(round(N_start * (2 ** (success_count / 2)))) < max_successful_N:
                success_count += 1

            experiment_count = success_count + 1
            N = int(round(N_start * (2 ** (experiment_count / 2))))

            max_steps = max_steps_start * (2 ** experiment_count)

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

            yield kwargs | result_i

            if timer.timed_out:
                break

            unsolved_fraction = result_i['unsolved_fraction']

            if unsolved_fraction < 0.5:
                experiment_count += 1
                N = int(round(N_start * (2 ** (experiment_count/2))))
                max_steps = min(max_steps * 2, max_steps_cap)
            else:
                break


def schedule_low(design_id, prior_experiments=None):
    yield from schedule(design_id=design_id, N_start=10, max_steps_start=50, N_cap=320, max_steps_cap=int(1e4), timeout=60, prior_experiments=prior_experiments)

def schedule_medium(design_id, prior_experiments=None):
    yield from schedule(design_id=design_id, N_start=10, max_steps_start=50, N_cap=640, max_steps_cap=int(1e5), timeout=300, prior_experiments=prior_experiments)

def schedule_high(design_id, prior_experiments=None):
    yield from schedule(design_id=design_id, N_start=10, max_steps_start=50, N_cap=1280, max_steps_cap=int(1e6), timeout=1800, prior_experiments=prior_experiments)
