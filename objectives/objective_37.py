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


def _smoothmax(vals, tau: float = 0.25) -> float:
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


def _infer_caps(results):
    fset = {str(r.get('fidelity', '')).lower() for r in (results or []) if isinstance(r, dict)}
    if 'high' in fset:
        return 2560, 1e6
    if 'medium' in fset:
        return 1280, 1e5
    if 'low' in fset:
        return 640, 1e4

    # fallback by observed max_steps
    mx = 0.0
    for r in results or []:
        try:
            mx = max(mx, float(r.get('max_steps', 0.0) or 0.0))
        except Exception:
            pass
    if mx >= 1e6:
        return 2560, 1e6
    if mx >= 1e5:
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


def _worst_window_pred_log_med(Nq: float, Ns, meds, windows=(2, 3, 5)) -> float:
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
            # single-point: assume quadratic scaling (conservative but not explosive)
            pred = float(logM[-1] + 2.0 * (math.log(max(float(Nq), 1.0)) - logN[-1]))
        worst = max(worst, pred)
    return float(worst)


def objective(experiment_results):
    """Estimate the research goal from experiment results.

    Args:
        experiment_results: List of dicts with experiment details
    Returns:
        Float to be MINIMIZED
    """
    if not experiment_results:
        return 1e6

    byN = _group_by_N(experiment_results)
    if not byN:
        return 1e6

    N_cap_cur, cap_cur = _infer_caps(experiment_results)
    sched_cur = _schedule_Ns(N_cap=N_cap_cur)

    # --- schedule-faithful cleared prefix (pass iff uf<0.5; do not grade uf below 0.5) ---
    cleared = []  # (N, med_pass)
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

    # primary reach (dominant)
    reach_term = -N_cleared

    # --- ratios on last-3 clears (use schedule budget of current fidelity) ---
    k_last = int(min(3, len(Ns)))
    logs = []
    for i in range(-k_last, 0):
        bud = schedule_budget(int(round(Ns[i])), cap=cap_cur)
        r = float(meds[i] / max(bud, 1.0))
        r = float(np.clip(r, 0.05, 5.0))
        logs.append(math.log(r))

    # --- conservative predicted ratios at next two scheduled Ns (prioritize 1280→1810→2560) ---
    cap_hi = 1e6
    sched_full = _schedule_Ns(N_cap=2560)

    # choose next2 with priority on critical Ns
    crit = [1280.0, 1810.0, 2560.0]
    cand = [c for c in crit if c > N_cleared]
    next2 = cand[:2]
    if len(next2) < 2:
        for n in sched_full:
            if float(n) > N_cleared and float(n) not in next2:
                next2.append(float(n))
            if len(next2) >= 2:
                break

    # local tail fit uses only cleared successes
    tfit = int(min(7, len(Ns)))
    Ns_t = Ns[-tfit:]
    meds_t = meds[-tfit:]

    # gate: don't over-weight far-future when we're still below 640
    g_future = 1.0 / (1.0 + math.exp(-(math.log(max(N_cleared, 1.0) / 640.0) / 0.35)))

    for Nt in next2:
        pred_logm = _worst_window_pred_log_med(Nt, Ns_t, meds_t, windows=(2, 3, 5))
        pred_med = float(np.clip(math.exp(pred_logm), 1.0, 1e30))
        bud = schedule_budget(int(round(Nt)), cap=cap_hi)
        r = float(pred_med / max(bud, 1.0))
        r = float(np.clip(r, 0.05, 5.0))
        logs.append(math.log(r))

    smmax_log_ratio = _smoothmax(logs, tau=0.25)

    # --- hazard: penalize sharp increases in log(ratio) across 905→1280 and 1280→1810 ---
    def _ratio_hi_at(Nq: float) -> float:
        # use observed if cleared contains Nq; else conservative prediction
        idx = np.where(np.isclose(Ns, float(Nq)))[0]
        if idx.size:
            m = float(meds[int(idx[-1])])
        else:
            pred_logm = _worst_window_pred_log_med(Nq, Ns_t, meds_t, windows=(2, 3, 5))
            m = float(np.clip(math.exp(pred_logm), 1.0, 1e30))
        bud = schedule_budget(int(round(Nq)), cap=cap_hi)
        return float(np.clip(m / max(bud, 1.0), 0.05, 5.0))

    r905 = _ratio_hi_at(905.0)
    r1280 = _ratio_hi_at(1280.0)
    r1810 = _ratio_hi_at(1810.0)

    d1 = math.log(r1280) - math.log(r905)
    d2 = math.log(r1810) - math.log(r1280)

    thr, scl = 0.10, 0.06
    hazard = (
        1.0 * _softplus((d1 - thr) / scl)
        + 1.4 * _softplus((d2 - thr) / scl)
    )
    hazard *= g_future

    # --- repeat stability (tiny): mixed pass/fail + MAD of passing headroom (log ratio) ---
    stab_num = 0.0
    stab_den = 0.0
    for N, runs in byN.items():
        if len(runs) < 2:
            continue
        passes = np.array([1.0 if _is_pass(r) else 0.0 for r in runs], dtype=float)
        p = float(np.mean(passes))
        mix = 4.0 * p * (1.0 - p)  # 0..1

        bud = schedule_budget(int(N), cap=cap_hi)
        lr = []
        for r in runs:
            if not _is_pass(r):
                continue
            try:
                m = float(r.get('median_step', np.nan))
            except Exception:
                continue
            if np.isfinite(m) and m > 0:
                rr = float(np.clip(m / max(bud, 1.0), 0.05, 5.0))
                lr.append(math.log(rr))
        if len(lr) >= 2:
            lr = np.asarray(lr, dtype=float)
            mad = float(np.median(np.abs(lr - np.median(lr))))
        else:
            mad = 0.0

        wN = float(math.log1p(max(N, 1.0)))
        stab_num += wN * (0.8 * mix + 0.6 * mad)
        stab_den += wN

    stab_pen = float(stab_num / max(stab_den, 1e-12))

    # combine (reach dominates; tie-breaks are O(1–10))
    obj = (
        reach_term
        + 25.0 * smmax_log_ratio
        + 8.0 * hazard
        + 1.5 * stab_pen
        + 15.0 * g_future  # small bias toward moving into 640+ regime
    )

    if not math.isfinite(obj):
        return 1e6
    return float(obj)
