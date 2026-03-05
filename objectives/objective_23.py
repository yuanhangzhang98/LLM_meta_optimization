import math
import numpy as np


def _first_failure_index(results):
    for i, r in enumerate(results):
        if float(r.get("unsolved_fraction", 1.0)) >= 0.5:
            return i
    return None


def _successful_prefix_dedup(results):
    if not results:
        return []
    results = sorted(results, key=lambda r: float(r.get("N", 0.0)))
    ff = _first_failure_index(results)
    pref = results if ff is None else results[:ff]
    succ = [r for r in pref if float(r.get("unsolved_fraction", 1.0)) < 0.5]

    # Dedup by N: keep the run with largest max_steps
    byN = {}
    for r in succ:
        N = int(r.get("N", 0))
        if N not in byN or float(r.get("max_steps", 0.0)) > float(byN[N].get("max_steps", 0.0)):
            byN[N] = r
    return [byN[N] for N in sorted(byN.keys())]


def _schedule_budget_dict(N_start=10, max_steps_start=50, N_cap=5120, cap=1e4, max_len=128):
    byN = {}
    for count in range(max_len):
        N = int(round(N_start * (2 ** (count / 2))))
        if N > N_cap:
            break
        ms = float(max_steps_start * (2 ** count))
        ms = float(min(ms, cap))
        byN[N] = max(byN.get(N, 0.0), ms)  # collisions due to rounding
        if N == N_cap:
            break
    return byN


_SCHED_BUDGETS = {
    1e4: _schedule_budget_dict(cap=1e4),
    1e5: _schedule_budget_dict(cap=1e5),
    1e6: _schedule_budget_dict(cap=1e6),
}


def _sched_budget(N, cap):
    d = _SCHED_BUDGETS.get(float(cap))
    if d is None:
        d = _schedule_budget_dict(cap=cap)
    N = int(N)
    if N in d:
        return float(d[N])
    # fallback: nearest available N
    keys = np.array(sorted(d.keys()), dtype=int)
    j = int(np.argmin(np.abs(keys - N)))
    return float(d[int(keys[j])])


def _softplus(x):
    x = np.asarray(x, dtype=float)
    return np.where(x > 50.0, x, np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0))


def objective(experiment_results):
    if not experiment_results:
        return 1e3

    succ = _successful_prefix_dedup(experiment_results)
    if not succ:
        return 200.0

    Ns = np.array([max(float(r.get("N", 1.0)), 1.0) for r in succ], dtype=float)
    meds = np.array([max(float(r.get("median_step", 1.0)), 1.0) for r in succ], dtype=float)
    msteps = np.array([max(float(r.get("max_steps", 1.0)), 1.0) for r in succ], dtype=float)

    N_max = float(Ns[-1])

    # (A) Reach dominates (lower is better)
    reach_term = -math.log2(max(N_max, 1.0))

    # (B) Tail slack penalty: strongly dislike median/max_steps >~ 0.7 on last successes
    k_tail = int(min(3, len(succ)))
    tail_ratio = np.clip(meds[-k_tail:] / np.maximum(msteps[-k_tail:], 1.0), 1e-12, 0.999999)
    tail_med = float(np.median(tail_ratio))
    tail_slack_pen = float(_softplus((tail_med - 0.70) / 0.05) + 0.15 * (-math.log(max(1.0 - tail_med, 1e-6))))

    # (C) Uncertainty-averse clearance at N=1280 (medium budget) and optionally 2560 (high budget)
    safety = 0.70  # require comfortable under-budget
    targets = [(1280.0, 1e5), (2560.0, 1e6)]

    # Fit log(ratio) vs log(N) on several tail windows; score WORST predicted clearance
    logN_all = np.log(np.maximum(Ns, 1.0))
    logR_all = np.log(np.maximum(meds / np.maximum(msteps, 1.0), 1e-12))

    windows = [2, 3, 5]
    worst_log_margin = -1e9
    worst_slope = 0.0

    for w in windows:
        if len(succ) >= w:
            X = logN_all[-w:]
            Y = logR_all[-w:]
            if w >= 2:
                b, a = np.polyfit(X, Y, 1)
                b = float(b)
                a = float(a)
            else:
                a = float(Y[-1])
                b = 0.0
        else:
            # fallback: constant model from last point
            a = float(logR_all[-1])
            b = 0.0

        worst_slope = max(worst_slope, b)

        for Nt, cap in targets:
            # only score targets beyond current reach weakly (keeps reach dominant but still guides headroom)
            gate = 1.0 if Nt <= max(N_max, 1.0) else 0.6

            pred_logR = a + b * math.log(max(Nt, 1.0))
            # convert from ratio-to-current-budget to ratio-to-target-budget via schedule budgets
            # (assume scaling learned in ratio space transfers; adjust by budget change between last point and target)
            bud_last = _sched_budget(N_max, cap=float(msteps[-1])) if False else float(msteps[-1])
            bud_t = _sched_budget(Nt, cap=float(cap))
            # If budgets differ, translate: log(med/bud_t) = log(med/bud_last) + log(bud_last/bud_t)
            pred_logR_to_target = pred_logR + math.log(max(bud_last, 1.0) / max(bud_t, 1.0))

            # margin > 0 means over (safety*budget)
            log_margin = gate * (pred_logR_to_target - math.log(max(safety, 1e-12)))
            worst_log_margin = max(worst_log_margin, float(log_margin))

    clearance_pen = float(_softplus(worst_log_margin / 0.25) * 0.25)

    # (D) Penalize accelerating/steepening trend in ratio space (positive slope is bad)
    slope_pen = float(_softplus((worst_slope - 0.10) / 0.10))

    # Small tie-breaker: fewer steps at largest successful N
    tie_term = math.log10(max(float(meds[-1]), 1.0))

    return reach_term + 1.10 * clearance_pen + 0.80 * tail_slack_pen + 0.25 * slope_pen + 0.02 * tie_term
