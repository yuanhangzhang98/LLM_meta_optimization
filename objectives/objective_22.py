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


def _infer_step_cap(results):
    fset = {str(r.get("fidelity", "")).lower() for r in (results or [])}
    if "high" in fset:
        return 1e6
    if "medium" in fset:
        return 1e5
    return 1e4


def _schedule_budgets(N_start=10, max_steps_start=50, N_cap=5120, max_len=64, cap=1e4):
    # Mirrors the provided schedule: N = round(N_start * 2**(count/2)), max_steps = min(max_steps_start * 2**count, cap)
    byN = {}
    for count in range(max_len):
        N = int(round(N_start * (2 ** (count / 2))))
        ms = float(max_steps_start * (2 ** count))
        ms = float(min(ms, cap))
        if N > N_cap:
            break
        # if rounding collides, keep the larger budget
        byN[N] = max(byN.get(N, 0.0), ms)
        if N == N_cap:
            break
    Ns = sorted(byN.keys())
    budgets = np.array([byN[n] for n in Ns], dtype=float)
    return np.array(Ns, dtype=float), budgets


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

    # --- (1) Clearance N*: largest scheduled N where predicted median_step <= safety * schedule_budget ---
    cap = _infer_step_cap(experiment_results)
    sched_Ns, sched_budget = _schedule_budgets(cap=cap)

    k = int(min(5, len(succ)))
    k = max(k, 1)
    X = np.log(np.maximum(Ns[-k:], 1.0))
    Y = np.log(np.maximum(meds[-k:], 1.0))
    if k >= 2:
        b, a = np.polyfit(X, Y, 1)
    else:
        a, b = float(Y[-1]), 0.0

    safety = 0.7
    pred = np.exp(a + b * np.log(np.maximum(sched_Ns, 1.0)))
    ok = pred <= (safety * np.maximum(sched_budget, 1.0))
    if np.any(ok):
        N_star = float(np.max(sched_Ns[ok]))
    else:
        # if even N=10 fails safety, fall back to smallest tested N
        N_star = float(np.min(Ns))

    clearance_term = -math.log2(max(N_star, 1.0))

    # --- (2) Tail headroom integral: penalize running near budget on successful prefix ---
    # softplus(log(median/max_steps) + margin) with margin set so ratio=0.7 gives ~0 argument
    margin = -math.log(0.7)
    ratio = np.clip(meds / np.maximum(msteps, 1.0), 1e-12, 1.0)
    x = np.log(ratio) + margin
    tau = 0.25
    per_point = tau * _softplus(x / tau)

    L = len(succ)
    w = np.ones(L, dtype=float)
    if L >= 1:
        w[-1] *= 8.0
    if L >= 2:
        w[-2] *= 4.0
    if L >= 3:
        w[-3] *= 2.0
    tail_integral = float(np.sum(w * per_point) / np.sum(w))

    # Small tie-breaker among equal headroom: prefer fewer steps at largest successful N
    tie_term = math.log10(max(float(meds[-1]), 1.0))

    return clearance_term + 0.80 * tail_integral + 0.03 * tie_term
