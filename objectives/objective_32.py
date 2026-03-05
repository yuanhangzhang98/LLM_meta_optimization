import math
import numpy as np


def _softplus(x: float) -> float:
    if x > 60.0:
        return x
    if x < -60.0:
        return math.exp(x)
    return math.log1p(math.exp(x))


def _first_failure_index(results):
    for i, r in enumerate(results):
        if float(r.get("unsolved_fraction", 1.0)) >= 0.5:
            return i
    return None


def _successful_prefix_dedup(results):
    """Only successes (uf<0.5) strictly before first failure; dedup by N keeping largest max_steps."""
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


def _infer_cap_from_results(results):
    fset = {str(r.get("fidelity", "")).lower() for r in (results or [])}
    if "high" in fset:
        return 1e6
    if "medium" in fset:
        return 1e5
    if "low" in fset:
        return 1e4
    # fallback by observed budgets
    mx = max([float(r.get("max_steps", 0.0)) for r in (results or [])] + [0.0])
    if mx >= 1e6:
        return 1e6
    if mx >= 1e5:
        return 1e5
    return 1e4


def _schedule_Ns(N_start=10, N_cap=5120, max_len=256):
    Ns = []
    for c in range(max_len):
        N = int(round(N_start * (2 ** (c / 2))))
        if Ns and N == Ns[-1]:
            continue
        Ns.append(N)
        if N >= N_cap:
            break
    return Ns


def _schedule_count_map(N_start=10, N_cap=5120, max_len=256):
    m = {}
    for c in range(max_len):
        N = int(round(N_start * (2 ** (c / 2))))
        if N > N_cap:
            break
        m[N] = max(m.get(N, -10**9), c)  # handle rounding collisions
        if N == N_cap:
            break
    return m


_COUNT_MAP = _schedule_count_map()


def _budget_for_N(N: int, cap: float) -> float:
    # max_steps = min(50 * 2**count, cap)
    N = int(N)
    c = _COUNT_MAP.get(N)
    if c is None:
        keys = np.array(sorted(_COUNT_MAP.keys()), dtype=int)
        j = int(np.argmin(np.abs(keys - N)))
        c = _COUNT_MAP[int(keys[j])]
    return float(min(50.0 * (2.0 ** float(c)), float(cap)))


def _worst_window_pred_ratio(N_query, Ns, ratios, windows=(2, 3, 5)):
    """Conservative: max predicted ratio among tail-window log-log fits of ratio vs N."""
    Ns = np.asarray(Ns, dtype=float)
    ratios = np.asarray(ratios, dtype=float)
    logN = np.log(np.maximum(Ns, 1.0))
    logR = np.log(np.maximum(ratios, 1e-12))

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
        pred = min(max(pred, 1e-12), 0.999999)
        worst = max(worst, pred)
    return float(worst)


def objective(experiment_results):
    if not experiment_results:
        return 1e3

    results = sorted(experiment_results, key=lambda r: float(r.get("N", 0.0)))
    succ = _successful_prefix_dedup(results)
    if not succ:
        return 300.0

    Ns = np.array([max(float(r.get("N", 1.0)), 1.0) for r in succ], dtype=float)
    meds = np.array([max(float(r.get("median_step", 1.0)), 1.0) for r in succ], dtype=float)
    msteps = np.array([max(float(r.get("max_steps", 1.0)), 1.0) for r in succ], dtype=float)

    ratios = np.clip(meds / np.maximum(msteps, 1.0), 1e-12, 0.999999)
    N_max = float(Ns[-1])

    # --- headroom overshoot for next 1–2 scheduled Ns ---
    safety = 0.68
    tau = 0.10

    cap = _infer_cap_from_results(results)
    sched = _schedule_Ns(N_start=10, N_cap=5120)
    # next 1–2 scheduled Ns after current max success
    idx = int(np.searchsorted(np.array(sched, dtype=float), N_max, side="left"))
    future = []
    for j in range(idx + 1, min(idx + 3, len(sched))):
        future.append(int(sched[j]))

    # if we are already at end of schedule, just score tail
    if not future:
        future = []

    # use a short tail for local fits (prevents small-N from dominating)
    tfit = int(min(7, len(succ)))
    Ns_tail = Ns[-tfit:]
    ratios_tail = ratios[-tfit:]

    future_pen_terms = []
    for Nt in future:
        r_pred = _worst_window_pred_ratio(Nt, Ns_tail, ratios_tail, windows=(2, 3, 5))
        # overshoot in multiplicative/log space
        future_pen_terms.append(tau * _softplus(math.log(max(r_pred / safety, 1e-12)) / tau))

    future_pen = float(np.mean(future_pen_terms)) if future_pen_terms else 0.0

    # --- tail observed headroom (last 2–5 successes): ratio itself, but only above safety ---
    ktail = int(min(5, len(ratios)))
    tail = ratios[-ktail:]
    tail_overs = [tau * _softplus(math.log(max(float(r) / safety, 1e-12)) / tau) for r in tail]
    tail_pen = float(np.mean(tail_overs)) if tail_overs else 0.0

    # --- tail trend over last 2–5 successes: penalize positive slope in log-log ratio space ---
    trend_pen = 0.0
    if len(ratios) >= 3:
        ktr = int(min(5, len(ratios)))
        logN = np.log(np.maximum(Ns[-ktr:], 1.0))
        logR = np.log(np.maximum(ratios[-ktr:], 1e-12))
        dN = np.diff(logN)
        slopes = np.diff(logR) / np.maximum(dN, 1e-12)
        # penalize only positive (ratio worsening with N)
        s_pos = float(np.median(np.maximum(slopes, 0.0)))
        trend_pen = 0.30 * _softplus((s_pos - 0.00) / 0.06)

    # --- explicit budget-cliff penalty: sudden increase in log(ratio) at the end ---
    cliff_pen = 0.0
    if len(ratios) >= 2:
        dlog = float(math.log(ratios[-1]) - math.log(ratios[-2]))
        # tolerate small; penalize sharp worsening
        cliff_pen = 0.70 * _softplus((dlog - 0.12) / 0.05)

    # --- single first-failure censor constraint (if any): predicted ratio at N_fail must not be too optimistic ---
    ff = _first_failure_index(results)
    censor_pen = 0.0
    if ff is not None:
        fail = results[ff]
        Nf = float(fail.get("N", 0.0))
        msf = float(fail.get("max_steps", 1.0))
        medf = float(fail.get("median_step", msf))
        # lower bound on ratio implied by censoring; clamp into [0.85,0.99]
        lb = float(np.clip(medf / max(msf, 1.0), 0.85, 0.99))
        r_pred_fail = _worst_window_pred_ratio(Nf, Ns_tail, ratios_tail, windows=(2, 3, 5))
        # penalize if prediction is below censor LB (too optimistic)
        censor_pen = 0.80 * (tau * _softplus(math.log(max(lb / max(r_pred_fail, 1e-12), 1e-12)) / tau))

    # --- mild reach tie-break (secondary; headroom dominates) ---
    reach_term = -0.06 * math.log2(max(N_max, 1.0))

    return float(reach_term + 6.0 * future_pen + 3.0 * tail_pen + trend_pen + cliff_pen + censor_pen)
