import os
import math
import numpy as np


def _softplus(x: float) -> float:
    if x > 60.0:
        return x
    if x < -60.0:
        return math.exp(x)
    return math.log1p(math.exp(x))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _logsumexp(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    m = float(np.max(a))
    return m + float(np.log(np.sum(np.exp(a - m))))


def _smoothmax(vals, tau: float = 0.20) -> float:
    v = np.asarray(list(vals), dtype=float)
    if v.size == 0:
        return 0.0
    tau = float(max(tau, 1e-12))
    return float(tau * _logsumexp(v / tau))


def _schedule_count_map(N_start=10, N_cap=5120, max_len=512):
    m = {}
    for c in range(max_len):
        N = int(round(N_start * (2.0 ** (c / 2.0))))
        if N > N_cap:
            break
        m[N] = max(m.get(N, -10**9), c)  # handle rounding collisions
        if N == N_cap:
            break
    return m


_COUNT_MAP = _schedule_count_map()


def schedule_budget(N: int, cap: float = 1e6, max_steps_start: float = 50.0) -> float:
    """Budget implied by the provided schedule; MUST be used for headroom (not run max_steps)."""
    N = int(N)
    c = _COUNT_MAP.get(N)
    if c is None:
        keys = np.array(sorted(_COUNT_MAP.keys()), dtype=int)
        j = int(np.argmin(np.abs(keys - N)))
        c = _COUNT_MAP[int(keys[j])]
    return float(min(max_steps_start * (2.0 ** float(c)), float(cap)))


def _schedule_Ns(N_start=10, N_cap=5120, max_len=512):
    Ns = []
    for c in range(max_len):
        N = int(round(N_start * (2.0 ** (c / 2.0))))
        if Ns and N == Ns[-1]:
            continue
        Ns.append(N)
        if N >= N_cap:
            break
    return Ns


def _group_by_N(results):
    byN = {}
    for r in results or []:
        if not isinstance(r, dict) or 'N' not in r:
            continue
        try:
            N = int(r['N'])
        except Exception:
            continue
        byN.setdefault(N, []).append(r)
    return byN


def _is_pass(r) -> bool:
    # schedule-faithful: pass iff uf < 0.5; never grade uf values when passing
    try:
        return float(r.get('unsolved_fraction', 1.0)) < 0.5
    except Exception:
        return False


def _pass_medians(runs_at_N):
    meds = []
    for rr in runs_at_N:
        if not _is_pass(rr):
            continue
        try:
            m = float(rr.get('median_step', np.nan))
        except Exception:
            continue
        if np.isfinite(m) and m > 0:
            meds.append(float(m))
    return np.asarray(meds, dtype=float)


def _infer_current_N_cap(results):
    fset = {str(r.get('fidelity', '')).lower() for r in (results or []) if isinstance(r, dict)}
    if 'high' in fset:
        return 2560
    if 'medium' in fset:
        return 1280
    return 640


def _worst_window_pred_log_med(Nq: float, Ns, meds, windows=(2, 3, 5)) -> float:
    """Conservative: max predicted log(med) among tail-window log-log fits; fallback is quadratic-in-N."""
    Ns = np.asarray(Ns, dtype=float)
    meds = np.asarray(meds, dtype=float)
    logN = np.log(np.maximum(Ns, 1.0))
    logM = np.log(np.maximum(meds, 1.0))

    worst = -1e300
    for w in windows:
        ww = min(int(w), len(logN))
        if ww >= 2:
            X = logN[-ww:]
            Y = logM[-ww:]
            b, a = np.polyfit(X, Y, 1)
            pred = float(a + b * math.log(max(float(Nq), 1.0)))
        else:
            # conservative single-point fallback: k=2 scaling
            pred = float(logM[-1] + 2.0 * (math.log(max(float(Nq), 1.0)) - logN[-1]))
        worst = max(worst, pred)
    return float(worst)


def objective(experiment_results):
    """Estimate the research goal from experiment results.

    Returns:
        Float to be MINIMIZED.
    """
    if not experiment_results:
        return 1e6

    variant = int(os.environ.get('OBJ_VARIANT', '1'))  # 1/2/3

    byN = _group_by_N(experiment_results)
    if not byN:
        return 1e6

    # --- schedule-faithful cleared prefix under current fidelity cap ---
    N_cap_cur = int(_infer_current_N_cap(experiment_results))
    sched_cur = _schedule_Ns(N_cap=N_cap_cur)

    cleared = []  # (N, median_of_passing_medians)
    for N in sched_cur:
        runs = byN.get(int(N))
        if runs is None:
            break
        pm = _pass_medians(runs)
        if pm.size == 0:
            break
        cleared.append((float(N), float(np.median(pm))))

    if not cleared:
        N0 = float(min(byN.keys()))
        return float(5e5 + 100.0 * math.log1p(N0))

    Ns = np.array([t[0] for t in cleared], dtype=float)
    meds = np.array([max(1.0, t[1]) for t in cleared], dtype=float)
    N_cleared = float(Ns[-1])

    # (1) Reach dominates
    reach_term = -N_cleared

    # Headroom always uses schedule_budget(..., cap=1e6)
    cap_hi = 1e6

    # tail for fits
    tfit = int(min(7, len(Ns)))
    Ns_t = Ns[-tfit:]
    meds_t = meds[-tfit:]

    def _pred_med(Nq: float) -> float:
        pl = _worst_window_pred_log_med(float(Nq), Ns_t, meds_t, windows=(2, 3, 5))
        return float(np.clip(math.exp(pl), 1.0, 1e30))

    def _ratio_hi(Nq: float) -> float:
        idx = np.where(np.isclose(Ns, float(Nq)))[0]
        med = float(meds[int(idx[-1])]) if idx.size else _pred_med(Nq)
        bud = schedule_budget(int(round(Nq)), cap=cap_hi)
        return float(np.clip(med / max(bud, 1.0), 1e-12, 1e12))

    # (2) Smooth-max bottleneck of log(headroom ratio) over last 2–3 cleared + preds@1810/2560
    lastk = int(min(3, len(Ns)))
    log_ratios = []
    for i in range(-lastk, 0):
        bud = schedule_budget(int(round(Ns[i])), cap=cap_hi)
        r = float(np.clip(meds[i] / max(bud, 1.0), 1e-12, 1e12))
        log_ratios.append(math.log(r))

    for Nt in (1810.0, 2560.0):
        log_ratios.append(math.log(_ratio_hi(Nt)))

    smmax_log_ratio = _smoothmax(log_ratios, tau=0.20)

    # (3) Budget-cliff hazard on log-ratio jumps: 905→1280→1810→2560 (pred if unobserved)
    r905 = _ratio_hi(905.0)
    r1280 = _ratio_hi(1280.0)
    r1810 = _ratio_hi(1810.0)
    r2560 = _ratio_hi(2560.0)

    d1 = math.log(max(r1280, 1e-12)) - math.log(max(r905, 1e-12))
    d2 = math.log(max(r1810, 1e-12)) - math.log(max(r1280, 1e-12))
    d3 = math.log(max(r2560, 1e-12)) - math.log(max(r1810, 1e-12))

    thr, scl = (0.10, 0.06) if variant != 3 else (0.08, 0.05)
    # gates so hazards don't dominate pre-regime
    g1 = _sigmoid(math.log(max(N_cleared, 1.0) / 905.0) / 0.35)
    g2 = _sigmoid(math.log(max(N_cleared, 1.0) / 1280.0) / 0.35)
    g3 = _sigmoid(math.log(max(N_cleared, 1.0) / 1810.0) / 0.35)

    hazard = (
        g1 * 1.0 * _softplus((d1 - thr) / scl)
        + g2 * 1.2 * _softplus((d2 - thr) / scl)
        + g3 * 1.4 * _softplus((d3 - thr) / scl)
    )

    # (4) Repeat robustness (variant 2/3): mixed pass/fail at same N + MAD of log headroom among passing repeats
    stab_pen = 0.0
    if variant in (2, 3):
        num = 0.0
        den = 0.0
        for N, runs in byN.items():
            if len(runs) < 2:
                continue
            passes = np.array([1.0 if _is_pass(r) else 0.0 for r in runs], dtype=float)
            p = float(np.mean(passes))
            mix = 4.0 * p * (1.0 - p)  # 0..1

            bud = schedule_budget(int(N), cap=cap_hi)
            lrs = []
            for r in runs:
                if not _is_pass(r):
                    continue
                try:
                    m = float(r.get('median_step', np.nan))
                except Exception:
                    continue
                if np.isfinite(m) and m > 0:
                    rr = float(np.clip(m / max(bud, 1.0), 1e-12, 1e12))
                    lrs.append(math.log(rr))
            if len(lrs) >= 2:
                lrs = np.asarray(lrs, dtype=float)
                mad = float(np.median(np.abs(lrs - np.median(lrs))))
            else:
                mad = 0.0

            wN = float(math.log1p(max(N, 1.0)))  # larger N weighted more
            num += wN * (1.2 * mix + 0.8 * mad)
            den += wN

        stab_pen = float(num / max(den, 1e-12))

    # weights: keep tie-break O(1..100) so reach dominates
    if variant == 2:
        w_sm, w_h, w_s = 30.0, 26.0, 18.0
    elif variant == 3:
        w_sm, w_h, w_s = 28.0, 32.0, 22.0
    else:
        w_sm, w_h, w_s = 28.0, 24.0, 0.0

    obj = float(reach_term + w_sm * smmax_log_ratio + w_h * hazard + w_s * stab_pen)
    if not math.isfinite(obj):
        return 1e6
    return obj
