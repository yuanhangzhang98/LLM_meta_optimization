import math
import numpy as np
from scipy.stats import theilslopes


def objective(experiment_results):
    """Estimate the research goal from experiment results.

    Minimizes a composite score:
      1) Reach term: log2(2000 / max_solved_N) (floored at 0 if max_solved_N>=2000)
      2) Scaling term: slope of log(median_step) vs log(N) on successful points only
      3) Step term: log10(predicted median_step at N=2000) from the same fit
      4) Censor-consistency: penalize overly-optimistic fit if the first truly-failed N exists

    Notes:
      - Uses ONLY the threshold unsolved_fraction < 0.5 (does not penalize 0.49).
      - Treats an N as solved if ANY run at that N succeeded.

    Args:
        experiment_results: List of dicts with experiment details
    Returns:
        Float to be MINIMIZED
    """
    if not experiment_results:
        return 1e6

    N_target = 2000.0
    K_censor = 2.0

    # Group runs by N
    byN = {}
    for e in experiment_results:
        N = e.get("N", None)
        if N is None:
            continue
        byN.setdefault(int(N), []).append(e)

    if not byN:
        return 1e6

    # Successful points (unique N): use fastest successful median_step for that N
    success_points = []
    for N, runs in byN.items():
        succ = [r for r in runs if r.get("unsolved_fraction", 1.0) < 0.5]
        if not succ:
            continue
        steps = [r.get("median_step", None) for r in succ]
        steps = [s for s in steps if s is not None and np.isfinite(s) and s > 0]
        if not steps:
            continue
        success_points.append((float(N), float(min(steps))))

    if success_points:
        max_solved_N = max(N for N, _ in success_points)
    else:
        max_solved_N = 1.0

    # First truly failed level: smallest N (above max_solved_N) with no successful run at that N
    N_block = None
    max_steps_block = None
    for N in sorted(byN.keys()):
        if float(N) <= float(max_solved_N):
            continue
        runs = byN[N]
        any_success = any(r.get("unsolved_fraction", 1.0) < 0.5 for r in runs)
        if not any_success:
            N_block = float(N)
            ms = [r.get("max_steps", None) for r in runs]
            ms = [m for m in ms if m is not None and np.isfinite(m) and m > 0]
            max_steps_block = float(max(ms)) if ms else None
            break

    # (1) Reach term (primary)
    if max_solved_N >= N_target:
        term_reach = 0.0
    else:
        term_reach = math.log(max(N_target / max_solved_N, 1.0), 2)
        term_reach = min(term_reach, 50.0)

    # (2)(3) Scaling fit on successful points only
    success_points.sort(key=lambda t: t[0])
    num_succ = len(success_points)

    if num_succ >= 2:
        xs = np.log(np.array([p[0] for p in success_points], dtype=float))
        ys = np.log(np.array([max(p[1], 1e-12) for p in success_points], dtype=float))

        if num_succ >= 3:
            fit = theilslopes(ys, xs)
            slope = float(fit.slope)
            intercept = float(fit.intercept)
        else:
            slope, intercept = np.polyfit(xs, ys, deg=1)
            slope = float(slope)
            intercept = float(intercept)

        slope = float(np.clip(slope, -0.5, 8.0))
        term_slope = slope

        pred2000 = math.exp(intercept + slope * math.log(N_target))
        term_pred = math.log10(max(pred2000, 1e-12))
        term_pred = float(np.clip(term_pred, -6.0, 30.0))

        # (4) Censor-consistency: if we failed at N_block with budget max_steps_block,
        # penalize fits that predict it should have solved much faster than that budget.
        penalty_censor = 0.0
        if N_block is not None and max_steps_block is not None:
            censor_steps = K_censor * max_steps_block
            pred_block = math.exp(intercept + slope * math.log(N_block))
            # penalty if pred_block < censor_steps (too optimistic)
            penalty_censor = max(0.0, math.log10(max(censor_steps, 1e-12)) - math.log10(max(pred_block, 1e-12)))
            penalty_censor = float(np.clip(penalty_censor, 0.0, 30.0))
    else:
        # Not enough successes for a meaningful scaling fit
        term_slope = 8.0
        if num_succ == 1:
            N_last, step_last = success_points[0]
            pred2000 = step_last * (N_target / max(N_last, 1e-9)) ** 2
            term_pred = math.log10(max(pred2000, 1.0))
        else:
            ms = [e.get("max_steps", None) for e in experiment_results]
            ms = [m for m in ms if m is not None and np.isfinite(m) and m > 0]
            term_pred = math.log10(K_censor * max(ms) if ms else 1e6)
        term_pred = float(np.clip(term_pred, -6.0, 30.0))
        penalty_censor = 0.0

    # Mild penalty for too few successful points (encourages stable scaling evidence)
    penalty_few = 0.0 if num_succ >= 3 else (3 - num_succ) * 1.5

    # Weighted composite (lexicographic-ish: reaching larger N dominates)
    obj = (
        10.0 * term_reach
        + 1.0 * term_slope
        + 0.25 * term_pred
        + 2.0 * penalty_censor
        + penalty_few
    )

    return float(obj)
