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


def _smoothmax(vals, tau: float = 0.22) -> float:
    v = np.asarray(list(vals), dtype=float)
    if v.size == 0:
        return 0.0
    tau = float(max(tau, 1e-9))
    return float(tau * _logsumexp(v / tau))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _schedule_count_map(N_start=10, N_cap=5120, max_len=512):
    m = {}
    for c in range(max_len):
        N = int(round(N_start * (2.0 ** (c / 2.0))))
        if N > N_cap:
            break
        m[N] = max(m.get(N, -10**9), c)  # keep max count on rounding collisions
        if N == N_cap:
            break
    return m


_COUNT_MAP = _schedule_count_map()


def schedule_budget(N: int, cap_steps: float, max_steps_start: float = 50.0) -> float:
    """Mirrors the schedule's intended budget at N: min(50*2**count(N), cap_steps)."""
    N = int(N)
    c = _COUNT_MAP.get(N)
    if c is None:
        keys = np.array(sorted(_COUNT_MAP.keys()), dtype=int)
        j = int(np.argmin(np.abs(keys - N)))
        c = _COUNT_MAP[int(keys[j])]
    return float(min(max_steps_start * (2.0 ** float(c)), float(cap_steps)))


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


def _infer_current_caps(results):
    fset = {str(r.get('fidelity', '')).lower() for r in (results or []) if isinstance(r, dict)}
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


def _worst_window_pred_log_med(Nq: float, Ns, meds, windows=(2, 3, 5)) -> float:
    """Conservative: max predicted log(med) among tail-window log-log fits."""
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
            # conservative single-point fallback: quadratic scaling in N
            pred = float(logM[-1] + 2.0 * (math.log(max(float(Nq), 1.0)) - logN[-1]))
        worst = max(worst, pred)
    return float(worst)


def objective(experiment_results):
    """Schedule-faithful objective (lower is better)."""
    if not experiment_results:
        return 1e6

    variant = int(os.environ.get('OBJ_VARIANT', '1'))  # 1/2/3

    byN = _group_by_N(experiment_results)
    if not byN:
        return 1e6

    N_cap_cur, cap_cur = _infer_current_caps(experiment_results)
    sched_cur = _schedule_Ns(N_cap=N_cap_cur)

    # --- cleared scheduled prefix (pass iff uf<0.5; do not grade uf below 0.5) ---
    cleared = []  # list of (N, med_pass)
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

    # For headroom/projection, always use the HIGH schedule budget function (cap=1e6)
    cap_hi = 1e6

    def _ratio_at(Nq: float) -> float:
        # use observed if present in cleared; else conservative prediction
        idx = np.where(np.isclose(Ns, float(Nq)))[0]
        if idx.size:
            med = float(meds[int(idx[-1])])
        else:
            tfit = int(min(7, len(Ns)))
            pred_log = _worst_window_pred_log_med(float(Nq), Ns[-tfit:], meds[-tfit:], windows=(2, 3, 5))
            med = float(np.clip(math.exp(pred_log), 1.0, 1e30))
        bud = schedule_budget(int(round(Nq)), cap_steps=cap_hi)
        r = med / max(bud, 1.0)
        return float(np.clip(r, 1e-6, 1e6))

    # --- smooth-max bottleneck of log(headroom ratio) over last-3 clears + predictions ---
    lastk = int(min(3, len(Ns)))
    log_ratios = []
    for i in range(-lastk, 0):
        bud = schedule_budget(int(round(Ns[i])), cap_steps=cap_hi)
        r = float(np.clip(meds[i] / max(bud, 1.0), 1e-6, 1e6))
        log_ratios.append(math.log(r))

    # include predictions at 2560, and at 1810 if not yet cleared
    log_ratios.append(math.log(_ratio_at(2560.0)))
    if N_cleared < 1810.0:
        log_ratios.append(math.log(_ratio_at(1810.0)))

    smmax_log_ratio = _smoothmax(log_ratios, tau=0.22)

    # --- explicit budget-cliff hazard on log-ratio jumps (use predictions if unobserved) ---
    # 905→1280→1810 and 1810→2560
    r905 = _ratio_at(905.0)
    r1280 = _ratio_at(1280.0)
    r1810 = _ratio_at(1810.0)
    r2560 = _ratio_at(2560.0)

    d1 = math.log(r1280) - math.log(r905)
    d2 = math.log(r1810) - math.log(r1280)
    d3 = math.log(r2560) - math.log(r1810)

    # gates so hazard doesn't matter too much before entering the regime
    g1 = _sigmoid(math.log(max(N_cleared, 1.0) / 905.0) / 0.35)
    g2 = _sigmoid(math.log(max(N_cleared, 1.0) / 1280.0) / 0.35)
    g3 = _sigmoid(math.log(max(N_cleared, 1.0) / 1810.0) / 0.35)

    thr, scl = (0.10, 0.06) if variant != 3 else (0.08, 0.05)
    hazard = (
        g1 * 1.0 * _softplus((d1 - thr) / scl)
        + g2 * 1.2 * _softplus((d2 - thr) / scl)
        + g3 * 1.4 * _softplus((d3 - thr) / scl)
    )

    # weights (keep penalties O(1–100) so reach dominates)
    if variant == 2:
        w_sm, w_h = 32.0, 24.0
    elif variant == 3:
        w_sm, w_h = 26.0, 28.0
    else:
        w_sm, w_h = 28.0, 22.0

    obj = float(reach_term + w_sm * smmax_log_ratio + w_h * hazard)
    if not math.isfinite(obj):
        return 1e6
    return obj
