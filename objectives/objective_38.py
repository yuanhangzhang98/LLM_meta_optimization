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


def _smoothmax(vals, tau: float = 0.18) -> float:
    v = np.asarray(list(vals), dtype=float)
    if v.size == 0:
        return 0.0
    tau = float(max(tau, 1e-9))
    return float(tau * _logsumexp(v / tau))


def _schedule_count_map(N_start=10, N_cap=5120, max_len=512):
    # map scheduled N -> experiment_count (keep max count on rounding collisions)
    m = {}
    for c in range(max_len):
        N = int(round(N_start * (2.0 ** (c / 2.0))))
        if N > N_cap:
            break
        m[N] = max(m.get(N, -10**9), c)
        if N == N_cap:
            break
    return m


_COUNT_MAP = _schedule_count_map()


def schedule_budget(N: int, cap_steps: float, max_steps_start: float = 50.0) -> float:
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


def _infer_caps(results):
    # infer *current* schedule cap from fidelity (preferred) else from observed max_steps
    fset = {str(r.get('fidelity', '')).lower() for r in (results or []) if isinstance(r, dict)}
    if 'high' in fset:
        return 2560, 1e6
    if 'medium' in fset:
        return 1280, 1e5
    if 'low' in fset:
        return 640, 1e4

    mx = 0.0
    for r in results or []:
        if not isinstance(r, dict):
            continue
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


def _worst_window_pred_log_med(Nq: float, Ns, meds, windows=(2, 3, 4)) -> float:
    # conservative: max predicted log(med) among tail-window log-log fits
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
            pred = float(logM[-1])
        worst = max(worst, pred)
    return float(worst)


def _first_failure_run_schedule_order(byN, sched_Ns):
    # failures only used for ONE censor-consistency check
    for N in sched_Ns:
        runs = byN.get(int(N))
        if not runs:
            continue
        # if any run at this N fails (uf>=0.5), treat as failure point
        for r in runs:
            try:
                uf = float(r.get('unsolved_fraction', 1.0))
            except Exception:
                uf = 1.0
            if uf >= 0.5:
                return int(N), r
    return None, None


def objective(experiment_results):
    if not experiment_results:
        return 1e6

    variant = int(os.environ.get('OBJ_VARIANT', '1'))  # 1/2/3

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

    # (1) reach (dominant, schedule-faithful)
    reach_term = -N_cleared

    # --- headroom ratios use schedule budgets (NOT run max_steps) ---
    def ratio_sched(Nq: float, med_q: float, cap_steps: float) -> float:
        bud = schedule_budget(int(round(Nq)), cap_steps=float(cap_steps))
        return float(np.clip(med_q / max(bud, 1.0), 1e-12, 1e12))

    # last-3 cleared ratios @ current cap
    last3 = int(min(3, len(Ns)))
    ratios_last = []
    for i in range(-last3, 0):
        ratios_last.append(ratio_sched(Ns[i], meds[i], cap_steps=cap_cur))

    # conservative prediction at 2560 under HIGH schedule budget
    cap_hi = 1e6
    tfit = int(min(7, len(Ns)))
    Ns_t = Ns[-tfit:]
    meds_t = meds[-tfit:]

    pred_logm_2560 = _worst_window_pred_log_med(2560.0, Ns_t, meds_t, windows=(2, 3, 4))
    pred_med_2560 = float(np.clip(math.exp(pred_logm_2560), 1.0, 1e30))
    ratio_2560 = ratio_sched(2560.0, pred_med_2560, cap_steps=cap_hi)

    # smooth-max of log ratios over {last3 cleared, predicted at 2560}
    logset = [math.log(max(r, 1e-12)) for r in ratios_last] + [math.log(max(ratio_2560, 1e-12))]
    smmax_log_ratio = _smoothmax(logset, tau=0.18)

    # --- tail-local exponent constraint: k = d log(med) / d log(N) on last 2–4 cleared levels ---
    t = int(min(5, len(Ns)))
    logN = np.log(np.maximum(Ns[-t:], 1.0))
    logM = np.log(np.maximum(meds[-t:], 1.0))
    if t >= 2:
        dlogN = np.diff(logN)
        dlogM = np.diff(logM)
        slopes = dlogM / np.maximum(dlogN, 1e-12)
        kk = slopes[-min(4, slopes.size):]  # last 2–4 local slopes
        k0, tauk = 2.0, 0.22
        k_pen = float(np.mean([tauk * _softplus((float(k) - k0) / tauk) for k in kk]))
    else:
        k_pen = 10.0

    # --- post-1280 budget-cliff hazard: emphasize 905→1280 and 1280→1810 and 1810→2560 (pred) ---
    # use observed med if present in cleared, else conservative tail prediction
    def med_at_or_pred(Nq: float) -> float:
        idx = np.where(np.isclose(Ns, float(Nq)))[0]
        if idx.size:
            return float(meds[int(idx[-1])])
        pl = _worst_window_pred_log_med(float(Nq), Ns_t, meds_t, windows=(2, 3, 4))
        return float(np.clip(math.exp(pl), 1.0, 1e30))

    r905 = ratio_sched(905.0, med_at_or_pred(905.0), cap_steps=cap_hi)
    r1280 = ratio_sched(1280.0, med_at_or_pred(1280.0), cap_steps=cap_hi)
    r1810 = ratio_sched(1810.0, med_at_or_pred(1810.0), cap_steps=cap_hi)
    r2560 = ratio_2560

    d1 = math.log(max(r1280, 1e-12)) - math.log(max(r905, 1e-12))
    d2 = math.log(max(r1810, 1e-12)) - math.log(max(r1280, 1e-12))
    d3 = math.log(max(r2560, 1e-12)) - math.log(max(r1810, 1e-12))

    thr, scl = 0.10, 0.06
    hazard = (
        1.2 * _softplus((d1 - thr) / scl)
        + 1.8 * _softplus((d2 - thr) / scl)
        + 1.6 * _softplus((d3 - thr) / scl)
    )

    # --- single censor-consistency lower-bound check from first uf>=0.5 at any scheduled N ---
    N_fail, fail_run = _first_failure_run_schedule_order(byN, _schedule_Ns(N_cap=2560))
    censor_pen = 0.0
    if N_fail is not None and fail_run is not None:
        try:
            medf = float(fail_run.get('median_step', np.nan))
        except Exception:
            medf = np.nan
        Bf = schedule_budget(int(N_fail), cap_steps=float(cap_hi))
        lb = float(max(Bf, medf if (np.isfinite(medf) and medf > 0) else Bf))
        pred_logm_f = _worst_window_pred_log_med(float(N_fail), Ns_t, meds_t, windows=(2, 3, 4))
        pred_med_f = float(np.clip(math.exp(pred_logm_f), 1.0, 1e30))
        # penalize only if model is too optimistic vs the lower bound
        z = math.log(max(lb / max(pred_med_f, 1e-12), 1e-12))
        censor_pen = 0.25 * _softplus(z / 0.20)

    # --- variant weighting (2/3 are stronger tail-focused alternatives) ---
    if variant == 2:
        w_sm, w_k, w_h, w_c = 34.0, 10.0, 16.0, 10.0
    elif variant == 3:
        w_sm, w_k, w_h, w_c = 28.0, 18.0, 12.0, 10.0
    else:
        w_sm, w_k, w_h, w_c = 30.0, 12.0, 14.0, 10.0

    # final (lower is better)
    obj = float(reach_term + w_sm * smmax_log_ratio + w_k * k_pen + w_h * hazard + w_c * censor_pen)
    if not math.isfinite(obj):
        return 1e6
    return obj
