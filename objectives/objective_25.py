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

    byN = {}
    for r in succ:
        N = int(r.get("N", 0))
        if N not in byN or float(r.get("max_steps", 0.0)) > float(byN[N].get("max_steps", 0.0)):
            byN[N] = r
    return [byN[N] for N in sorted(byN.keys())]


def _infer_caps(results):
    fset = {str(r.get("fidelity", "")).lower() for r in (results or [])}
    if "high" in fset:
        return 2560, 1e6
    if "medium" in fset:
        return 1280, 1e5
    # fallback: infer from max_steps if fidelity absent
    ms = max([float(r.get("max_steps", 0.0)) for r in (results or [])] + [0.0])
    if ms > 1e5:
        return 2560, 1e6
    if ms > 1e4:
        return 1280, 1e5
    return 640, 1e4


def _schedule_grid(N_start=10, max_steps_start=50, N_cap=640, cap=1e4, max_len=256):
    byN = {}
    for count in range(max_len):
        N = int(round(N_start * (2 ** (count / 2))))
        if N > N_cap:
            break
        ms = float(min(max_steps_start * (2 ** count), cap))
        byN[N] = max(byN.get(N, 0.0), ms)  # handle rounding collisions
        if N == N_cap:
            break
    Ns = np.array(sorted(byN.keys()), dtype=float)
    budgets = np.array([byN[int(n)] for n in Ns], dtype=float)
    return Ns, budgets


def _softplus(x):
    x = float(x)
    if x > 50.0:
        return x
    if x < -50.0:
        return math.exp(x)
    return math.log1p(math.exp(x))


def _worst_pred_ratio_at(N_query, logN, logR, windows=(2, 3, 5)):
    # conservative: worst (max) predicted ratio over tail-window fits in log-log ratio space
    worst = 0.0
    for w in windows:
        ww = min(int(w), len(logN))
        if ww >= 2:
            X = logN[-ww:]
            Y = logR[-ww:]
            b, a = np.polyfit(X, Y, 1)
            pred_logR = float(a + b * math.log(max(float(N_query), 1.0)))
        else:
            pred_logR = float(logR[-1])
        pred = math.exp(pred_logR)
        # ratio should be in (0,1]; clip for numerical safety
        pred = min(max(pred, 1e-12), 0.999999)
        worst = max(worst, pred)
    return float(worst)


def objective(experiment_results):
    if not experiment_results:
        return 1e3

    succ = _successful_prefix_dedup(experiment_results)
    if not succ:
        return 200.0

    Ns = np.array([max(float(r.get("N", 1.0)), 1.0) for r in succ], dtype=float)
    meds = np.array([max(float(r.get("median_step", 1.0)), 1.0) for r in succ], dtype=float)
    msteps = np.array([max(float(r.get("max_steps", 1.0)), 1.0) for r in succ], dtype=float)

    ratios = np.clip(meds / np.maximum(msteps, 1.0), 1e-12, 0.999999)
    N_max = float(Ns[-1])

    N_cap, step_cap = _infer_caps(experiment_results)
    sched_Ns, _sched_budgets = _schedule_grid(N_cap=int(N_cap), cap=float(step_cap))

    logN = np.log(np.maximum(Ns, 1.0))
    logR = np.log(np.maximum(ratios, 1e-12))

    safety = 0.60

    # --- (1) Step-ratio clearance N*: largest scheduled N where worst-window predicted ratio <= 0.60 ---
    worst_preds = np.array([
        _worst_pred_ratio_at(n, logN, logR, windows=(2, 3, 5)) for n in sched_Ns
    ], dtype=float)
    ok = worst_preds <= safety
    if np.any(ok):
        N_star = float(np.max(sched_Ns[ok]))
    else:
        N_star = float(sched_Ns[0])

    clearance_term = -math.log2(max(N_star, 1.0))

    # --- (2) Guaranteed headroom across the next two scheduled jumps (continuous penalty) ---
    # Find next two scheduled Ns after current N_max.
    idx = int(np.searchsorted(sched_Ns, N_max, side="left"))
    future = []
    for j in range(idx + 1, min(idx + 3, len(sched_Ns))):
        future.append(float(sched_Ns[j]))
    if not future:
        future = [float(sched_Ns[min(idx, len(sched_Ns) - 1)])]

    future_ratios = [
        _worst_pred_ratio_at(n, logN, logR, windows=(2, 3, 5)) for n in future
    ]
    # penalize log-overshoot above safety (smooth, scale ~0 when comfortably below)
    tau = 0.12
    headroom_pen = float(np.mean([
        tau * _softplus(math.log(max(r / safety, 1e-12)) / tau) for r in future_ratios
    ]))

    # --- (3) Tail hazard: suppress sudden accelerations near the tail ---
    t = int(min(5, len(succ)))
    if t >= 2:
        logN_t = np.log(np.maximum(Ns[-t:], 1.0))
        logM_t = np.log(np.maximum(meds[-t:], 1.0))
        logR_t = np.log(np.maximum(ratios[-t:], 1e-12))

        dlogN = np.diff(logN_t)
        slopes_M = np.diff(logM_t) / np.maximum(dlogN, 1e-12)  # d log(median) / d log(N)
        max_slope_M = float(np.max(slopes_M))

        dlogR = np.diff(logR_t)  # local jump in log(ratio)
        max_dlogR = float(np.max(dlogR))
    else:
        max_slope_M = 10.0
        max_dlogR = 10.0

    hazard_pen = (
        _softplus((max_slope_M - 1.3) / 0.3) +
        0.8 * _softplus((max_dlogR - 0.12) / 0.05)
    )

    # --- (4) Small tie-breaker: last observed ratio itself (prefer more slack than 0.60) ---
    last_ratio = float(ratios[-1])
    last_ratio_pen = 0.25 * _softplus((last_ratio - safety) / 0.04)

    return clearance_term + 0.90 * headroom_pen + 0.45 * hazard_pen + 0.20 * last_ratio_pen
