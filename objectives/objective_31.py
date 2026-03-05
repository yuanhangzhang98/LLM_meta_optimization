import math
import numpy as np
from scipy.stats import theilslopes


def _finite(x):
    return x is not None and np.isfinite(x)


def _finite_pos(x):
    return _finite(x) and x > 0


def objective_area_under_passed_levels(experiment_results, pass_reward=1.0, headroom_lambda=0.08):
    """Secondary proxy (maximize): sum over tested Ns of reward for passing (uf<0.5)
    minus a small penalty proportional to log10(median_step/max_steps) to prefer decisive passes.

    Returns a value to MINIMIZE (negative of the reward).
    Missing/untested levels are ignored (failed-censored, not extrapolated).
    """
    if not experiment_results:
        return 0.0

    byN = {}
    for e in experiment_results:
        N = e.get("N", None)
        if N is None:
            continue
        byN.setdefault(int(N), []).append(e)

    total = 0.0
    for N in sorted(byN.keys()):
        runs = byN[N]
        # pass if any run passes
        succ = [r for r in runs if r.get("unsolved_fraction", 1.0) < 0.5]
        if not succ:
            continue

        # take best (fastest) successful median_step at this N
        steps = [r.get("median_step", None) for r in succ]
        steps = [float(s) for s in steps if _finite_pos(s)]
        if steps:
            step = min(steps)
        else:
            step = None

        ms_list = [r.get("max_steps", None) for r in runs]
        ms_list = [float(m) for m in ms_list if _finite_pos(m)]
        max_steps = max(ms_list) if ms_list else None

        # base pass reward
        contrib = float(pass_reward)

        # headroom preference (log10(step/max_steps) <= 0 for passes with headroom)
        if step is not None and max_steps is not None:
            ratio = max(step / max(max_steps, 1e-12), 1e-12)
            contrib -= headroom_lambda * math.log10(ratio)

        total += contrib

    return float(-total)


def objective(experiment_results):
    """Estimate the research goal from experiment results.

    Lexicographic (implemented via scale separation):
      (i) maximize max_solved_N (any run at N has uf<0.5)
      (ii) minimize log10(best median_step) at max_solved_N
      (iii) minimize robust slope of log10(median_step) vs log10(N) on successful points
      (iv) add censor penalty using first truly-failed N and its max_steps

    Returns a float to MINIMIZE.
    """
    if not experiment_results:
        return 1e6

    # Group runs by N
    byN = {}
    for e in experiment_results:
        N = e.get("N", None)
        if N is None:
            continue
        byN.setdefault(int(N), []).append(e)
    if not byN:
        return 1e6

    # Build success frontier data: for each N, keep best successful median_step
    succ_points = []  # (N, best_step)
    best_step_at = {}
    max_steps_at = {}
    best_uf_at = {}

    for N, runs in byN.items():
        ufs = [r.get("unsolved_fraction", None) for r in runs]
        ufs = [float(u) for u in ufs if _finite(u)]
        best_uf_at[N] = min(ufs) if ufs else 1.0

        ms_list = [r.get("max_steps", None) for r in runs]
        ms_list = [float(m) for m in ms_list if _finite_pos(m)]
        max_steps_at[N] = max(ms_list) if ms_list else None

        succ = [r for r in runs if r.get("unsolved_fraction", 1.0) < 0.5 and _finite_pos(r.get("median_step", None))]
        if succ:
            best = min(succ, key=lambda r: float(r["median_step"]))
            step = float(best["median_step"])
            best_step_at[N] = step
            succ_points.append((float(N), step))

    max_solved_N = max((int(N) for N, _ in succ_points), default=0)

    # Secondary: log10(best step) at the frontier (max_solved_N)
    if max_solved_N > 0 and max_solved_N in best_step_at:
        term_front_logstep = math.log10(max(best_step_at[max_solved_N], 1e-12))
    else:
        # no successes: use smallest-N max_steps as a weak placeholder
        N0 = min(byN.keys())
        ms0 = max_steps_at.get(N0, None)
        term_front_logstep = math.log10(max(float(ms0) if _finite_pos(ms0) else 1e6, 1.0))
    term_front_logstep = float(np.clip(term_front_logstep, -6.0, 30.0))

    # Tertiary: robust slope on successful points (log-log)
    succ_points.sort(key=lambda t: t[0])
    if len(succ_points) >= 2:
        xs = np.log10(np.array([p[0] for p in succ_points], dtype=float))
        ys = np.log10(np.array([max(p[1], 1e-12) for p in succ_points], dtype=float))
        if len(succ_points) >= 3:
            fit = theilslopes(ys, xs)
            slope = float(fit.slope)
            intercept = float(fit.intercept)
        else:
            slope, intercept = np.polyfit(xs, ys, deg=1)
            slope = float(slope)
            intercept = float(intercept)
        slope = float(np.clip(slope, -0.5, 8.0))
    else:
        slope = 8.0
        intercept = None

    # Censor penalty from first truly failed N above max_solved_N
    N_fail = None
    max_steps_fail = None
    for N in sorted(byN.keys()):
        if N <= max_solved_N:
            continue
        # truly failed means no successful run at this N
        if N not in best_step_at:
            N_fail = int(N)
            msf = max_steps_at.get(N_fail, None)
            max_steps_fail = float(msf) if _finite_pos(msf) else None
            break

    censor_pen = 0.0
    if N_fail is not None and max_steps_fail is not None and len(succ_points) >= 2 and intercept is not None:
        # Predict at an OBSERVED (tested) failed level; penalize over-optimism vs available budget
        pred_fail = 10 ** (intercept + slope * math.log10(max(float(N_fail), 1e-12)))
        # allow some slack: failure despite budget suggests median likely > ~2*budget
        slack = 2.0 * max_steps_fail
        if pred_fail > 0:
            censor_pen = max(0.0, math.log10(max(slack, 1e-12)) - math.log10(max(pred_fail, 1e-12)))
    censor_pen = float(np.clip(censor_pen, 0.0, 30.0))

    # Area-under-passed-levels (optional tie-breaker)
    auc_obj = objective_area_under_passed_levels(experiment_results)

    # Lexicographic scalarization via scale separation (primary dominates by construction)
    obj = (
        -float(max_solved_N)           # primary: larger N -> smaller objective
        + 1e-3 * term_front_logstep    # secondary
        + 1e-4 * float(slope)          # tertiary
        + 1e-4 * float(censor_pen)     # censor
        + 1e-5 * float(auc_obj)        # tiny tie-break
    )

    return float(obj)
