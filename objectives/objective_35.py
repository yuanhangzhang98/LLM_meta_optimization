import math
import numpy as np


def _softplus(x: float) -> float:
    if x > 60.0:
        return x
    if x < -60.0:
        return math.exp(x)
    return math.log1p(math.exp(x))


def _sigmoid(x: float) -> float:
    # stable-ish scalar sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _infer_caps(results):
    fset = {str(r.get("fidelity", "")).lower() for r in (results or [])}
    if "high" in fset:
        return 2560, 1e6
    if "medium" in fset:
        return 1280, 1e5
    if "low" in fset:
        return 640, 1e4

    # fallback: infer from observed max_steps
    mx = 0.0
    for r in results or []:
        try:
            mx = max(mx, float(r.get("max_steps", 0.0) or 0.0))
        except Exception:
            pass
    if mx >= 1e6:
        return 2560, 1e6
    if mx >= 1e5:
        return 1280, 1e5
    return 640, 1e4


def _schedule_grid(N_start=10, max_steps_start=50, N_cap=2560, max_steps_cap=1e6, max_len=512):
    Ns = []
    count_map = {}
    for c in range(max_len):
        N = int(round(N_start * (2 ** (c / 2))))
        if N > N_cap:
            break
        count_map[N] = max(count_map.get(N, -10**9), c)
        Ns.append(N)
        if N == N_cap:
            break
    Ns = sorted(set(Ns))
    return Ns, count_map


def _budget_for_N(N: int, count_map, max_steps_start=50.0, max_steps_cap=1e6):
    N = int(N)
    if N not in count_map:
        keys = np.array(sorted(count_map.keys()), dtype=int)
        j = int(np.argmin(np.abs(keys - N)))
        N = int(keys[j])
    c = float(count_map[N])
    return float(min(max_steps_start * (2.0 ** c), float(max_steps_cap)))


def _worst_window_pred_logy(N_query, Ns, ys, windows=(2, 3, 4)):
    """Conservative: max predicted y among tail-window log-log fits (y>0)."""
    Ns = np.asarray(Ns, dtype=float)
    ys = np.asarray(ys, dtype=float)
    logN = np.log(np.maximum(Ns, 1.0))
    logY = np.log(np.maximum(ys, 1e-12))

    worst = -1e100
    for w in windows:
        ww = min(int(w), len(logN))
        if ww >= 2:
            X = logN[-ww:]
            Y = logY[-ww:]
            b, a = np.polyfit(X, Y, 1)
            pred = float(a + b * math.log(max(float(N_query), 1.0)))
        else:
            pred = float(logY[-1])
        worst = max(worst, pred)
    return float(worst)


def _overshoot_pen_ratio(ratio, safety=0.70, tau=0.12):
    """Smooth penalty ~0 when ratio<=safety; grows with log(ratio/safety)."""
    ratio = float(np.clip(ratio, 1e-12, 1e12))
    return float(tau * _softplus(math.log(max(ratio / safety, 1e-12)) / tau))


def objective(experiment_results):
    if not experiment_results:
        return 1e6

    results = sorted(experiment_results, key=lambda r: float(r.get("N", 0.0) or 0.0))
    N_cap, max_steps_cap = _infer_caps(results)
    sched_Ns, count_map = _schedule_grid(N_cap=int(N_cap), max_steps_cap=float(max_steps_cap))

    # group runs by N
    byN = {}
    for r in results:
        try:
            N = int(r.get("N"))
        except Exception:
            continue
        byN.setdefault(N, []).append(r)

    # schedule-faithful cleared prefix: pass iff uf<0.5; do not score uf value when passing
    cleared = []  # list of (N, med_step_pass_median)
    for N in sched_Ns:
        runs = byN.get(int(N))
        if runs is None:
            break
        pass_meds = []
        for rr in runs:
            try:
                uf = float(rr.get("unsolved_fraction", 1.0))
                ms = float(rr.get("median_step", np.nan))
            except Exception:
                continue
            if uf < 0.5 and np.isfinite(ms) and ms > 0:
                pass_meds.append(ms)
        if not pass_meds:
            break
        med = float(np.median(np.asarray(pass_meds, dtype=float)))
        cleared.append((float(N), max(1.0, med)))

    if not cleared:
        # no cleared level: keep clearly bad
        N0 = float(min(byN.keys())) if byN else 1.0
        return float(1e6 + 1e2 * math.log1p(N0))

    Ns = np.array([t[0] for t in cleared], dtype=float)
    meds = np.array([t[1] for t in cleared], dtype=float)
    N_max = float(Ns[-1])

    # schedule budgets (NOT run-provided max_steps)
    budgets = np.array([
        _budget_for_N(int(n), count_map, max_steps_cap=float(max_steps_cap)) for n in Ns
    ], dtype=float)
    ratios = np.clip(meds / np.maximum(budgets, 1.0), 1e-12, 1e12)

    # --- (1) Reach dominates (lexicographic via scale separation) ---
    reach_term = -N_max

    # --- (2) Tail-heavy observed ratio headroom over last 2–4 successes ---
    K = int(min(4, len(ratios)))
    tail_r = ratios[-K:]
    # tail weights: emphasize very end
    w = np.array([2 ** i for i in range(K)], dtype=float)
    w = w / (w.sum() + 1e-12)

    # absolute ratio preference (uses log-ratio so being <<1 helps, but stays small magnitude)
    tail_log_abs = float(np.sum(w * np.log(np.maximum(tail_r, 1e-12))))

    # overshoot above safety (schedule-budget clearance proxy)
    safety_obs = 0.70
    tail_over = float(np.sum(w * np.array([
        _overshoot_pen_ratio(r, safety=safety_obs, tau=0.12) for r in tail_r
    ], dtype=float)))

    # --- conservative tail fits (worst-of-window) ---
    tfit = int(min(7, len(Ns)))
    Ns_t = Ns[-tfit:]
    meds_t = meds[-tfit:]
    ratios_t = ratios[-tfit:]

    def pred_ratio(Nq: float) -> float:
        # worst-window fit in log-log ratio space
        logR = _worst_window_pred_logy(Nq, Ns_t, ratios_t, windows=(2, 3, 4))
        r = float(math.exp(logR))
        return float(np.clip(r, 1e-12, 1e12))

    def pred_med(Nq: float) -> float:
        logM = _worst_window_pred_logy(Nq, Ns_t, meds_t, windows=(2, 3, 4))
        m = float(math.exp(logM))
        return float(np.clip(m, 1.0, 1e30))

    # --- (3) Predicted post-1280 bottleneck ratios at next scheduled Ns (1810, 2560) ---
    # smooth gate so far-extrapolation can't dominate when we haven't reached the regime
    g_post = _sigmoid(math.log(max(N_max, 1.0) / 1280.0) / 0.35)  # ~0 below 1280, ~1 above

    targets = [1810.0, 2560.0]
    safety_pred = 0.72
    pred_terms = []
    for Nt in targets:
        if Nt <= float(N_cap):
            rNt = pred_ratio(Nt)
            pred_terms.append(_overshoot_pen_ratio(rNt, safety=safety_pred, tau=0.12))
    pred_pen = float(np.mean(pred_terms)) if pred_terms else 0.0

    # --- (4) Explicit budget-cliff hazard (905→1280 observed/pred, and 1280→1810 predicted) ---
    # use observed ratios when present; otherwise predicted
    def ratio_at(Nq: float) -> float:
        idx = np.where(np.isclose(Ns, Nq))[0]
        if idx.size:
            return float(ratios[int(idx[-1])])
        return pred_ratio(Nq)

    def med_at(Nq: float) -> float:
        idx = np.where(np.isclose(Ns, Nq))[0]
        if idx.size:
            return float(meds[int(idx[-1])])
        return pred_med(Nq)

    hazard = 0.0
    # ratio-cliff thresholds
    thr_r, scl_r = 0.12, 0.05
    # median-cliff thresholds
    thr_m, scl_m = 0.85, 0.20

    r905 = ratio_at(905.0)
    r1280 = ratio_at(1280.0)
    r1810 = pred_ratio(1810.0) if 1810.0 <= float(N_cap) else r1280

    m905 = med_at(905.0)
    m1280 = med_at(1280.0)
    m1810 = pred_med(1810.0) if 1810.0 <= float(N_cap) else m1280

    dlog_r_905_1280 = math.log(max(r1280, 1e-12)) - math.log(max(r905, 1e-12))
    dlog_r_1280_1810 = math.log(max(r1810, 1e-12)) - math.log(max(r1280, 1e-12))
    dlog_m_905_1280 = math.log(max(m1280, 1.0)) - math.log(max(m905, 1.0))
    dlog_m_1280_1810 = math.log(max(m1810, 1.0)) - math.log(max(m1280, 1.0))

    hazard += 0.35 * _softplus((dlog_r_905_1280 - thr_r) / scl_r)
    hazard += 0.55 * _softplus((dlog_r_1280_1810 - thr_r) / scl_r)
    hazard += 0.10 * _softplus((dlog_m_905_1280 - thr_m) / scl_m)
    hazard += 0.20 * _softplus((dlog_m_1280_1810 - thr_m) / scl_m)
    hazard *= g_post

    # --- (5) Single censor-consistency from first uf>=0.5 (ignore failures otherwise) ---
    censor_pen = 0.0
    first_fail = None
    for r in results:
        try:
            uf = float(r.get("unsolved_fraction", 1.0))
        except Exception:
            continue
        if uf >= 0.5:
            first_fail = r
            break

    if first_fail is not None:
        try:
            Nf = float(first_fail.get("N", 0.0) or 0.0)
        except Exception:
            Nf = 0.0
        if Nf > 0:
            # lower bound on ratio implied by censoring at that scheduled budget
            Bf = _budget_for_N(int(round(Nf)), count_map, max_steps_cap=float(max_steps_cap))
            try:
                medf = float(first_fail.get("median_step", Bf) or Bf)
            except Exception:
                medf = Bf
            lb = float(np.clip(max(medf, Bf) / max(Bf, 1.0), 1.0, 1e6))
            rpred = pred_ratio(Nf)
            # penalize only if prediction is too optimistic (below the LB)
            censor_pen = 0.30 * _overshoot_pen_ratio(lb / max(rpred, 1e-12), safety=1.0, tau=0.18)

    # --- combine: reach dominates; tie-breakers emphasize post-1280 tail/bottleneck ---
    obj = (
        reach_term
        + 45.0 * tail_over
        + 70.0 * (g_post * pred_pen)
        + 55.0 * hazard
        + 6.0 * (g_post * tail_log_abs)  # smaller (more negative) is better
        + 25.0 * censor_pen
    )

    if not math.isfinite(obj):
        return 1e6
    return float(obj)
