import math
import numpy as np


def _softplus(x: float) -> float:
    # stable scalar softplus
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
    if not results:
        return []
    results = sorted(results, key=lambda r: float(r.get("N", 0.0)))
    ff = _first_failure_index(results)
    pref = results if ff is None else results[:ff]
    succ = [r for r in pref if float(r.get("unsolved_fraction", 1.0)) < 0.5]

    # Dedup by N: keep run with largest max_steps
    byN = {}
    for r in succ:
        N = int(r.get("N", 0))
        if N not in byN or float(r.get("max_steps", 0.0)) > float(byN[N].get("max_steps", 0.0)):
            byN[N] = r
    return [byN[N] for N in sorted(byN.keys())]


def _schedule_Ns(N_start=10, N_cap=1280, max_len=256):
    Ns = []
    for count in range(max_len):
        N = int(round(N_start * (2 ** (count / 2))))
        if Ns and N == Ns[-1]:
            continue
        Ns.append(N)
        if N >= N_cap:
            break
    return Ns


def _worst_window_pred_ratio(N_query, Ns, ratios, windows=(2, 3, 5)):
    # Conservative: max predicted ratio among window fits in log-log ratio space.
    logN = np.log(np.maximum(np.asarray(Ns, dtype=float), 1.0))
    logR = np.log(np.maximum(np.asarray(ratios, dtype=float), 1e-12))

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

    succ = _successful_prefix_dedup(experiment_results)
    if not succ:
        return 200.0

    Ns = np.array([max(float(r.get("N", 1.0)), 1.0) for r in succ], dtype=float)
    meds = np.array([max(float(r.get("median_step", 1.0)), 1.0) for r in succ], dtype=float)
    msteps = np.array([max(float(r.get("max_steps", 1.0)), 1.0) for r in succ], dtype=float)

    ratios = np.clip(meds / np.maximum(msteps, 1.0), 1e-12, 0.999999)
    N_max = float(Ns[-1])
    r_last = float(ratios[-1])

    # --- Gate: only meaningfully evaluate headroom once we have real tail (N_max >= 453) ---
    gate_N = 453.0
    # Smooth large penalty when below gate; ~0 when above.
    gate_pen = 35.0 * _softplus(math.log(gate_N / max(N_max, 1.0)) / 0.25)

    # --- Headroom-to-1280 using only tail data (worst-window predicted ratios) ---
    safety = 0.68
    tau = 0.12

    # Use only a short tail for fitting (prevents small-N data from dominating)
    t = int(min(7, len(succ)))
    Ns_tail = Ns[-t:]
    ratios_tail = ratios[-t:]

    targets = [905.0, 1280.0]
    sched = _schedule_Ns(N_start=10, N_cap=1280)
    succ_set = {int(n) for n in Ns.tolist()}

    headroom_terms = []
    extrap_pen = 0.0

    for Nt in targets:
        allow = True
        if Nt > 2.0 * N_max:
            # Avoid >2x extrapolation unless all intermediate scheduled points were actually succeeded.
            inter = [n for n in sched if int(round(N_max)) < n < int(round(Nt))]
            if any(int(n) not in succ_set for n in inter):
                allow = False

        if allow:
            pred = _worst_window_pred_ratio(Nt, Ns_tail, ratios_tail, windows=(2, 3, 5))
        else:
            pred = max(r_last, 0.95)
            extrap_pen += 2.0 + 2.0 * _softplus(math.log(Nt / max(2.0 * N_max, 1.0)) / 0.20)

        # penalize only overshoot above safety
        headroom_terms.append(tau * _softplus(math.log(max(pred / safety, 1e-12)) / tau))

    headroom_pen = float(np.mean(headroom_terms)) if headroom_terms else 0.0

    # Extra penalty for being too close to budget at the tail
    near_budget_pen = 1.5 * _softplus((r_last - 0.75) / 0.03)

    # --- Tail-only slope+curvature from last 3 successes ---
    slope_curv_pen = 0.0
    if len(succ) < 3:
        slope_curv_pen += 8.0
    else:
        Ns3 = Ns[-3:]
        R3 = ratios[-3:]
        # Must include at least one of these tail anchors, else penalize
        anchors = {226, 320, 453, 640}
        if not any(int(round(n)) in anchors for n in Ns3.tolist()):
            slope_curv_pen += 6.0

        logN3 = np.log(np.maximum(Ns3, 1.0))
        logR3 = np.log(np.maximum(R3, 1e-12))

        b, a = np.polyfit(logN3, logR3, 1)
        slope = float(b)

        # local curvature (change in secant slopes)
        s1 = float((logR3[1] - logR3[0]) / max(logN3[1] - logN3[0], 1e-12))
        s2 = float((logR3[2] - logR3[1]) / max(logN3[2] - logN3[1], 1e-12))
        curv_pos = max(0.0, s2 - s1)

        # Penalize positive slope/curvature; don't overly reward negative.
        slope_curv_pen += 0.20 * _softplus((slope - 0.00) / 0.08)
        slope_curv_pen += 0.35 * _softplus((curv_pos - 0.00) / 0.08)

    # Mild reach tie-breaker among gated designs
    reach_term = -0.15 * math.log2(max(N_max, 1.0))

    # Core score (lower is better)
    return (
        gate_pen
        + reach_term
        + 2.0 * r_last
        + 5.0 * headroom_pen
        + near_budget_pen
        + slope_curv_pen
        + extrap_pen
    )
