import os
import math
import numpy as np


def _softplus(x: float) -> float:
    if x > 60.0:
        return x
    if x < -60.0:
        return math.exp(x)
    return math.log1p(math.exp(x))


def _logsumexp(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    m = float(np.max(a))
    return m + float(np.log(np.sum(np.exp(a - m))))


def _smoothmax(vals, tau: float = 0.15) -> float:
    """Smooth max of real numbers (tau in same units)."""
    v = np.asarray(list(vals), dtype=float)
    if v.size == 0:
        return 0.0
    tau = float(max(tau, 1e-9))
    return float(tau * _logsumexp(v / tau))


def _schedule_count_map(N_start=10, N_cap=5120, max_len=512):
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


def schedule_budget(N: int, cap: float, max_steps_start: float = 50.0) -> float:
    """Explicitly mirrors schedule_budget(N): min(max_steps_start*2**count(N), cap)."""
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
        N = int(round(N_start * (2 ** (c / 2))))
        if Ns and N == Ns[-1]:
            continue
        Ns.append(N)
        if N >= N_cap:
            break
    return Ns


def _infer_current_caps(results):
    # Current evaluation cap (schedule_low/medium/high)
    fset = {str(r.get('fidelity', '')).lower() for r in (results or [])}
    if 'high' in fset:
        return 2560, 1e6
    if 'medium' in fset:
        return 1280, 1e5
    return 640, 1e4


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
    try:
        return float(r.get('unsolved_fraction', 1.0)) < 0.5
    except Exception:
        return False


def _best_pass_medians(runs_at_N):
    """Return array of median_step for passing runs only (schedule-faithful)."""
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


def _worst_window_pred_logy(N_query, Ns, ys, windows=(2, 3, 5)):
    """Conservative: max predicted log(y) among tail-window log-log fits."""
    Ns = np.asarray(Ns, dtype=float)
    ys = np.asarray(ys, dtype=float)
    logN = np.log(np.maximum(Ns, 1.0))
    logY = np.log(np.maximum(ys, 1e-12))

    worst = -1e300
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


def _overshoot_pen(ratio: float, safety: float = 0.70, tau: float = 0.12) -> float:
    """0 when ratio<=safety; smooth in log-space when above."""
    ratio = float(np.clip(ratio, 1e-12, 1e12))
    z = math.log(max(ratio / max(safety, 1e-12), 1e-12))
    return float(tau * _softplus(z / max(tau, 1e-12)))


def _obj_v1(results):
    """Balanced: smooth-max readiness + cliff hazard + repeat stability."""
    byN = _group_by_N(results)
    if not byN:
        return 1e6

    # Current schedule prefix we can clear (schedule-faithful)
    N_cap_cur, cap_cur = _infer_current_caps(results)
    sched_cur = _schedule_Ns(N_cap=N_cap_cur)

    cleared = []  # list of dicts: N, med (median of passing meds), bud_cur
    for N in sched_cur:
        runs = byN.get(int(N))
        if runs is None:
            break
        pass_meds = _best_pass_medians(runs)
        if pass_meds.size == 0:
            break
        med = float(np.median(pass_meds))
        bud = schedule_budget(N, cap=cap_cur)
        cleared.append({'N': float(N), 'med': max(1.0, med), 'bud': max(1.0, bud)})

    if not cleared:
        # no cleared level: clearly bad but smooth-ish
        N0 = float(min(byN.keys()))
        return float(5e5 + 1e2 * math.log1p(N0))

    Ns = np.array([c['N'] for c in cleared], dtype=float)
    meds = np.array([c['med'] for c in cleared], dtype=float)
    buds_cur = np.array([c['bud'] for c in cleared], dtype=float)

    N_max = float(Ns[-1])

    # Reach-dominant term
    reach_term = -N_max

    # --- tail fit support (use only cleared successes) ---
    tfit = int(min(7, len(Ns)))
    Ns_t = Ns[-tfit:]
    meds_t = meds[-tfit:]

    def pred_med(Nq: float) -> float:
        logm = _worst_window_pred_logy(Nq, Ns_t, meds_t, windows=(2, 3, 5))
        return float(np.clip(math.exp(logm), 1.0, 1e30))

    # --- Build smooth-max set: last 3 cleared ratios (current budget) + future readiness ratios ---
    lastk = int(min(3, len(Ns)))
    ratios_last = []
    for i in range(-lastk, 0):
        ratios_last.append(float(meds[i] / max(buds_cur[i], 1.0)))

    # next two scheduled Ns in *full* schedule (up to 2560), scored vs high-cap budgets
    cap_hi = 1e6
    sched_full = _schedule_Ns(N_cap=2560)
    idx = int(np.searchsorted(np.array(sched_full, dtype=float), N_max, side='left'))
    next2 = []
    for j in range(idx + 1, min(idx + 3, len(sched_full))):
        next2.append(float(sched_full[j]))

    # explicitly include 1810 and 2560 if ahead (post-1280 readiness)
    for crit in (1810.0, 2560.0):
        if crit > N_max and crit not in next2:
            next2.append(crit)
    # keep only two most relevant (prefer (1810,2560) if present)
    next2 = sorted(set(next2))
    # prioritize largest two targets (tends to pick 1810/2560)
    next2 = next2[-2:] if len(next2) > 2 else next2

    ratios_future = []
    for Nt in next2:
        bud = schedule_budget(int(round(Nt)), cap=cap_hi)
        ratios_future.append(pred_med(Nt) / max(bud, 1.0))

    # Smooth-max in log-ratio space (minimize worst bottleneck)
    log_ratios = [math.log(max(r, 1e-12)) for r in (ratios_last + ratios_future)]
    smmax_log_ratio = _smoothmax(log_ratios, tau=0.18)

    # --- Budget-cliff hazard: focus on 905→1280 (observed/pred) and 1280→1810 (pred) in *high-budget* ratio ---
    def ratio_hi(Nq: float) -> float:
        bud = schedule_budget(int(round(Nq)), cap=cap_hi)
        # if we have cleared this N, use observed med; else predicted
        w = np.where(np.isclose(Ns, float(Nq)))[0]
        m = float(meds[int(w[-1])]) if w.size else pred_med(Nq)
        return float(np.clip(m / max(bud, 1.0), 1e-12, 1e12))

    r905 = ratio_hi(905.0)
    r1280 = ratio_hi(1280.0)
    r1810 = ratio_hi(1810.0)

    dlog_905_1280 = math.log(r1280) - math.log(r905)
    dlog_1280_1810 = math.log(r1810) - math.log(r1280)

    # penalize upward jumps in log(ratio)
    hazard = (
        0.9 * _softplus((dlog_905_1280 - 0.10) / 0.05)
        + 1.4 * _softplus((dlog_1280_1810 - 0.10) / 0.05)
    )

    # --- Repeat-run stability: only pass/fail + headroom ratio ---
    # For each N with repeats, penalize (i) mixed pass/fail, (ii) high dispersion of log headroom among passes.
    stab = 0.0
    wsum = 0.0
    for N, runs in byN.items():
        if len(runs) < 2:
            continue
        bud = schedule_budget(int(N), cap=cap_hi)  # use high-cap headroom for stability tie-break
        passes = np.array([1.0 if _is_pass(r) else 0.0 for r in runs], dtype=float)
        p = float(np.mean(passes))
        mix = 4.0 * p * (1.0 - p)  # 0..1

        # headroom from passing runs only
        med_pass = []
        for r in runs:
            if not _is_pass(r):
                continue
            try:
                m = float(r.get('median_step', np.nan))
            except Exception:
                continue
            if np.isfinite(m) and m > 0:
                med_pass.append(float(m))
        if len(med_pass) >= 2:
            lr = np.log(np.maximum(np.asarray(med_pass, dtype=float) / max(bud, 1.0), 1e-12))
            mad = float(np.median(np.abs(lr - np.median(lr))))
        else:
            mad = 0.0

        # emphasize bigger N repeats
        wN = float(math.log1p(max(N, 1.0)))
        stab += wN * (1.2 * mix + 0.8 * mad)
        wsum += wN

    stab_pen = float(stab / max(wsum, 1e-12))

    # --- Combine (reach dominates by magnitude; tie-breakers sharpen post-1280 readiness) ---
    obj = (
        reach_term
        + 220.0 * smmax_log_ratio
        + 140.0 * hazard
        + 80.0 * stab_pen
    )
    return float(obj)


def _obj_v2(results):
    """More aggressive post-1280 readiness: heavier smooth-max on (1810,2560) ratios and hazard."""
    v = float(_obj_v1(results))
    # amplify tail readiness when already past 640
    byN = _group_by_N(results)
    N_cap_cur, cap_cur = _infer_current_caps(results)
    sched_cur = _schedule_Ns(N_cap=N_cap_cur)
    N_max = 0.0
    for N in sched_cur:
        runs = byN.get(int(N))
        if runs is None or _best_pass_medians(runs).size == 0:
            break
        N_max = float(N)
    g = 1.0 / (1.0 + math.exp(-(math.log(max(N_max, 1.0) / 640.0) / 0.35)))
    return float(v + 120.0 * g)


def _obj_v3(results):
    """Robustness-forward: upweight repeat stability penalties."""
    base = _obj_v1(results)
    # add extra stability weight if repeats exist
    byN = _group_by_N(results)
    rep = any(len(v) >= 2 for v in byN.values())
    return float(base + (120.0 if rep else 0.0))


def objective(experiment_results):
    """Estimate the research goal from experiment results.

    Args:
        experiment_results: List of dicts with experiment details
    Returns:
        Float to be MINIMIZED
    """
    if not experiment_results:
        return 1e6

    variant = int(os.environ.get('OBJ_VARIANT', '1'))
    if variant == 2:
        return _obj_v2(experiment_results)
    if variant == 3:
        return _obj_v3(experiment_results)
    return _obj_v1(experiment_results)
