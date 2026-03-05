import math
import numpy as np


def _first_failure_index(results):
    for i, r in enumerate(results):
        if r.get("unsolved_fraction", 1.0) >= 0.5:
            return i
    return None


def _successful_prefix(results):
    if not results:
        return []
    results = sorted(results, key=lambda r: float(r.get("N", 0)))
    ff = _first_failure_index(results)
    pref = results if ff is None else results[:ff]
    succ = [r for r in pref if r.get("unsolved_fraction", 1.0) < 0.5]

    # Deduplicate by N: keep the run with largest max_steps (most reliable median)
    byN = {}
    for r in succ:
        N = int(r.get("N", 0))
        if N not in byN or float(r.get("max_steps", 0)) > float(byN[N].get("max_steps", 0)):
            byN[N] = r
    return [byN[N] for N in sorted(byN.keys())]


def _schedule_Ns(N_start=10, N_cap=5120, max_len=64):
    Ns = []
    for exp_count in range(max_len):
        N = int(round(N_start * (2 ** (exp_count / 2))))
        if Ns and N == Ns[-1]:
            continue
        Ns.append(N)
        if N >= N_cap:
            break
    return Ns


def _infer_cap_from_fidelity(results):
    # Conservative caps consistent with provided schedules
    fset = {str(r.get("fidelity", "")).lower() for r in results}
    if "high" in fset:
        return 1e6
    if "medium" in fset:
        return 1e5
    return 1e4


def _softplus(x):
    # stable softplus
    if x > 50:
        return x
    if x < -50:
        return math.exp(x)
    return math.log1p(math.exp(x))


def objective(experiment_results):
    """Return a scalar to MINIMIZE; uses only successful-prefix medians."""
    if not experiment_results:
        return 1e3

    results = sorted(experiment_results, key=lambda r: float(r.get("N", 0)))
    succ = _successful_prefix(results)
    if not succ:
        # no success at all (smooth but clearly bad)
        return 200.0

    Ns = np.array([max(float(r.get("N", 1.0)), 1.0) for r in succ], dtype=float)
    meds = np.array([max(float(r.get("median_step", 1.0)), 1.0) for r in succ], dtype=float)
    msteps = np.array([max(float(r.get("max_steps", 1.0)), 1.0) for r in succ], dtype=float)

    N_max = float(Ns[-1])
    med_at_max = float(meds[-1])
    ms_at_max = float(msteps[-1])

    # (A) Reach (dominant): bigger N => lower objective
    reach_term = -math.log2(max(N_max, 1.0))

    # Tie-breaker: steps at largest successful N
    tie_term = math.log10(max(med_at_max, 1.0))

    # (B) Budget margin on successful prefix tail (median slack over last 3 successes)
    k = int(min(3, len(succ)))
    slack = np.clip(meds[-k:] / np.maximum(msteps[-k:], 1.0), 0.0, 0.999999)
    slack_med = float(np.median(slack))
    # near-budget penalty: ~0 when slack<<0.6, rises smoothly as slack->1
    slack_pen = _softplus((slack_med - 0.60) / 0.06) + 0.15 * (-math.log(max(1.0 - slack_med, 1e-6)))

    # (C) Tail robustness: log-log slope + positive acceleration (curvature) on last up to 5 successes
    t = int(min(5, len(succ)))
    logN = np.log(np.maximum(Ns[-t:], 1.0))
    logM = np.log(np.maximum(meds[-t:], 1.0))

    if t >= 2:
        dlogN = np.diff(logN)
        dlogM = np.diff(logM)
        slopes = dlogM / np.maximum(dlogN, 1e-12)
        slope_tail = float(np.median(slopes))
        # penalize only increasing slope (helps small-N, hurts tail)
        if len(slopes) >= 2:
            accel = np.diff(slopes)
            accel_pos = float(np.mean(np.maximum(accel, 0.0)))
        else:
            accel_pos = 0.0
    else:
        slope_tail, accel_pos = 5.0, 0.0

    tail_pen = 0.25 * slope_tail + 0.8 * accel_pos

    # (D) Next-scheduled-N clearance margin (soft): predict steps at next N and compare to next budget
    sched = _schedule_Ns(N_start=10, N_cap=5120)
    # find next scheduled N after N_max (robust to rounding)
    idx = None
    for i, N in enumerate(sched):
        if N >= int(round(N_max)):
            idx = i
            break
    nextN = None
    if idx is not None and idx + 1 < len(sched):
        nextN = float(sched[idx + 1])

    next_pen = 0.0
    if nextN is not None:
        # conservative next budget = doubled last max_steps, capped by current fidelity cap
        cap = _infer_cap_from_fidelity(results)
        next_budget = min(ms_at_max * 2.0, cap)

        # power-law fit on last 2-3 successes: log(m) = a + b log(N)
        f = int(min(3, len(succ)))
        X = np.log(np.maximum(Ns[-f:], 1.0))
        Y = np.log(np.maximum(meds[-f:], 1.0))
        if f >= 2:
            b, a = np.polyfit(X, Y, 1)
            pred = float(math.exp(a + b * math.log(max(nextN, 1.0))))
        else:
            pred = float(med_at_max)

        # penalty for predicted over-budget; small reward if comfortably under
        margin = pred / max(next_budget, 1.0)
        next_pen = (1.0 / 3.0) * _softplus(3.0 * math.log(max(margin, 1e-12)))

    # Combined (reach primary; others discriminate among same N_max)
    return reach_term + 0.05 * tie_term + 0.35 * slack_pen + 0.30 * tail_pen + 0.50 * next_pen
