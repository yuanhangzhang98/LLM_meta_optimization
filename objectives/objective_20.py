import math
import numpy as np
from scipy.stats import linregress


def _first_failure_index(results):
    for i, r in enumerate(results):
        if r.get('unsolved_fraction', 1.0) >= 0.5:
            return i
    return None


def _successful_prefix(results):
    # ignore any post-failure points (schedule usually stops at first failure, but be robust)
    ff = _first_failure_index(results)
    prefix = results if ff is None else results[:ff]
    return [r for r in prefix if r.get('unsolved_fraction', 1.0) < 0.5]


def objective_reach(results):
    """Primary 'reach' objective: maximize largest successful N; hard penalize failure before target(s).

    Returns value to MINIMIZE.
    """
    if not results:
        return 1e6

    results = sorted(results, key=lambda r: r.get('N', 0))
    succ = _successful_prefix(results)

    if not succ:
        # no success at all
        return 1e5

    N_max = max(r.get('N', 0) for r in succ)
    r_at_max = max((r for r in succ if r.get('N', 0) == N_max), key=lambda r: r.get('max_steps', 0), default=None)
    med_at_max = float(r_at_max.get('median_step', 1.0)) if r_at_max else 1.0

    # Determine which target(s) to enforce.
    # Always enforce 640; enforce 1280 only if evaluator progressed to medium/high fidelity.
    fidelities = {str(r.get('fidelity', '')).lower() for r in results}
    progressed = any(f in {'medium', 'high'} for f in fidelities)

    penalties = 0.0

    # Hard penalty if fail before 640
    if N_max < 640:
        # bounded-but-dominant constant + mild distance term
        penalties += 200.0 + 20.0 * math.log(max(640.0 / max(N_max, 1.0), 1.0))

    # Additional hard penalty if fail before 1280 when appropriate
    if progressed and N_max < 1280:
        penalties += 50.0 + 10.0 * math.log(max(1280.0 / max(N_max, 1.0), 1.0))

    # Reward reaching larger N (lower is better)
    reach_term = -math.log2(max(N_max, 1.0))

    # Tie-breaker: prefer fewer steps at the largest successful N
    tie_term = math.log10(max(med_at_max, 1.0))

    return penalties + reach_term + 0.1 * tie_term


def objective_tail_scaling(results):
    """Secondary 'tail-scaling' objective: slope of log(median_step) vs log(N) over top successes + slack penalty.

    Computed only on successful prefix; returns value to MINIMIZE.
    """
    if not results:
        return 1e3

    results = sorted(results, key=lambda r: r.get('N', 0))
    succ = _successful_prefix(results)

    if len(succ) < 2:
        return 100.0

    # Use top 3 successful Ns
    succ_sorted = sorted(succ, key=lambda r: r.get('N', 0))
    top = succ_sorted[-min(3, len(succ_sorted)) :]

    Ns = np.array([max(float(r.get('N', 1.0)), 1.0) for r in top], dtype=float)
    meds = np.array([max(float(r.get('median_step', 1.0)), 1.0) for r in top], dtype=float)

    # Tail slope (penalize steep growth)
    if len(top) >= 2:
        fit = linregress(np.log(Ns), np.log(meds))
        slope = float(fit.slope)
    else:
        slope = 10.0

    # Slack penalty when median steps approach max_steps (little budget margin)
    slack_terms = []
    for r in top:
        m = max(float(r.get('median_step', 1.0)), 1.0)
        ms = float(r.get('max_steps', max(m, 1.0)))
        ms = max(ms, 1.0)
        frac = m / ms
        frac = min(max(frac, 0.0), 0.999)
        # ~0 until close to 1, then grows fast
        slack_terms.append(-math.log(max(1.0 - frac, 1e-6)))

    slack_pen = float(np.mean(slack_terms)) if slack_terms else 0.0

    return slope + 0.5 * slack_pen


def objective(experiment_results):
    """Estimate the research goal from experiment results.

    Returns a combined score to MINIMIZE.
    """
    reach = objective_reach(experiment_results)
    tail = objective_tail_scaling(experiment_results)

    # Reach dominates; tail breaks ties among similarly-reaching designs.
    return reach + 0.3 * tail
