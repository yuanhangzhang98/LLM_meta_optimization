import os
import math
import numpy as np


def _softplus(x: float) -> float:
    if x > 60.0:
        return x
    if x < -60.0:
        return math.exp(x)
    return math.log1p(math.exp(x))


def _first_failure(results_sorted_byN):
    for r in results_sorted_byN:
        if float(r.get("unsolved_fraction", 1.0)) >= 0.5:
            return r
    return None


def _infer_caps(results):
    # N_cap and max_steps_cap implied by schedule fidelity
    fset = {str(r.get("fidelity", "")).lower() for r in (results or [])}
    if "high" in fset:
        return 2560, 1e6
    if "medium" in fset:
        return 1280, 1e5
    if "low" in fset:
        return 640, 1e4

    mx = max([float(r.get("max_steps", 0.0) or 0.0) for r in (results or [])] + [0.0])
    if mx >= 1e6:
        return 2560, 1e6
    if mx >= 1e5:
        return 1280, 1e5
    return 640, 1e4


def _schedule_grid(N_start=10, max_steps_start=50, N_cap=2560, max_steps_cap=1e6, max_len=512):
    # Mirrors provided schedule rounding; returns sorted unique Ns and count-map
    Ns = []
    count_map = {}
    for c in range(max_len):
        N = int(round(N_start * (2 ** (c / 2))))
        if N > N_cap:
            break
        if Ns and N == Ns[-1]:
            # rounding collision: keep the larger count
            count_map[N] = max(count_map.get(N, -10**9), c)
            continue
        Ns.append(N)
        count_map[N] = max(count_map.get(N, -10**9), c)
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


def _passes(r):
    return float(r.get("unsolved_fraction", 1.0)) < 0.5


def _best_pass_run(runs_at_N):
    # schedule-faithful: only pass/fail threshold; among passes prefer larger budget, then smaller median
    cand = [r for r in runs_at_N if _passes(r)]
    if not cand:
        return None

    def key(r):
        ms = float(r.get("max_steps", 0.0) or 0.0)
        med = float(r.get("median_step", 1e30) or 1e30)
        return (ms, -med)

    return max(cand, key=key)


def _tail_fit_pred_med(N_query, Ns, meds, windows=(2, 3, 4)):
    # Conservative: worst (largest) predicted median among tail-window log-log fits.
    Ns = np.asarray(Ns, dtype=float)
    meds = np.asarray(meds, dtype=float)
    logN = np.log(np.maximum(Ns, 1.0))
    logM = np.log(np.maximum(meds, 1.0))

    worst = 0.0
    for w in windows:
        ww = min(int(w), len(logN))
        if ww >= 2:
            X = logN[-ww:]
            Y = logM[-ww:]
            b, a = np.polyfit(X, Y, 1)
            pred = float(math.exp(float(a) + float(b) * math.log(max(float(N_query), 1.0))))
        else:
            pred = float(meds[-1])
        worst = max(worst, pred)
    return float(worst)


def _overshoot_pen(ratio, safety=0.70, tau=0.10):
    # zero when ratio<=safety, smooth growth in log-space when above
    ratio = float(np.clip(ratio, 1e-12, 1e12))
    return float(tau * _softplus(math.log(max(ratio / safety, 1e-12)) / tau))


# -------------------- Objective variants (select via OBJ_VARIANT env var: 1/2/3) --------------------

def _objective_v1(results):
    """Reach (cleared N) dominates; tie-break by tail headroom, next-N clearance vs schedule budget, and budget cliffs."""
    results = sorted(results, key=lambda r: float(r.get("N", 0.0)))
    N_cap, max_steps_cap = _infer_caps(results)
    sched_Ns, count_map = _schedule_grid(N_cap=int(N_cap), max_steps_cap=float(max_steps_cap))

    # group by N
    byN = {}
    for r in results:
        if "N" not in r:
            continue
        try:
            N = int(r["N"])
        except Exception:
            continue
        byN.setdefault(N, []).append(r)

    # cleared scheduled prefix
    cleared = []
    for N in sched_Ns:
        runs = byN.get(int(N))
        if runs is None:
            break
        best = _best_pass_run(runs)
        if best is None:
            break
        med = float(best.get("median_step", 1.0) or 1.0)
        med = max(med, 1.0)
        cleared.append((float(N), med))

    if not cleared:
        # smooth-but-bad: no pass at smallest tested N
        N0 = float(min(byN.keys())) if byN else 1.0
        return 1e6 + 1e2 * math.log1p(N0)

    Ns = np.array([t[0] for t in cleared], dtype=float)
    meds = np.array([t[1] for t in cleared], dtype=float)
    N_max = float(Ns[-1])
    med_at_max = float(meds[-1])

    # schedule budgets (not run budgets) for headroom computation
    budgets = np.array([_budget_for_N(int(n), count_map, max_steps_cap=float(max_steps_cap)) for n in Ns], dtype=float)
    ratios = np.clip(meds / np.maximum(budgets, 1.0), 1e-12, 1e6)

    # (1) Reach dominates
    reach_term = -N_max

    # (i) tail ratio headroom over last 2–4 successes
    K = int(min(4, len(ratios)))
    tail = ratios[-K:]
    tail_ratio_med = float(np.median(tail))
    tail_headroom_pen = _overshoot_pen(tail_ratio_med, safety=0.70, tau=0.10)

    # (ii) conservative worst-window next scheduled N clearance vs schedule budget
    # evaluate next 2 scheduled jumps AND critical (1810,2560) if within cap
    sched_arr = np.array(sched_Ns, dtype=float)
    idx = int(np.searchsorted(sched_arr, N_max, side="left"))
    future = []
    for j in range(idx + 1, min(idx + 3, len(sched_arr))):
        future.append(float(sched_arr[j]))
    for crit in (1810.0, 2560.0):
        if crit <= float(N_cap) and crit > N_max and crit not in future:
            future.append(crit)
    future = sorted(set(future))

    safety = 0.72  # within guidance 0.65–0.75
    fut_pens = []
    for Nt in future:
        pred_med = _tail_fit_pred_med(Nt, Ns, meds, windows=(2, 3, 4))
        bud = _budget_for_N(int(round(Nt)), count_map, max_steps_cap=float(max_steps_cap))
        fut_pens.append(_overshoot_pen(pred_med / max(bud, 1.0), safety=safety, tau=0.10))
    next_clear_pen = float(np.mean(fut_pens)) if fut_pens else 0.0

    # (iii) explicit budget-cliff penalties at the tail (log ratio and log median)
    cliff_pen = 0.0
    if len(ratios) >= 2:
        dlog_r = float(math.log(ratios[-1]) - math.log(ratios[-2]))
        dlog_m = float(math.log(meds[-1]) - math.log(meds[-2]))
        # tolerate mild drift; penalize sharp worsening
        cliff_pen += 0.10 * _softplus((dlog_r - 0.12) / 0.05)
        cliff_pen += 0.06 * _softplus((dlog_m - 0.85) / 0.20)

    # optional: tail acceleration (log median slope steepening)
    accel_pen = 0.0
    if len(meds) >= 3:
        logN3 = np.log(np.maximum(Ns[-3:], 1.0))
        logM3 = np.log(np.maximum(meds[-3:], 1.0))
        s1 = float((logM3[1] - logM3[0]) / max(logN3[1] - logN3[0], 1e-12))
        s2 = float((logM3[2] - logM3[1]) / max(logN3[2] - logN3[1], 1e-12))
        accel_pos = max(0.0, s2 - s1)
        accel_pen = 0.08 * _softplus(accel_pos / 0.10)

    # single censor-consistency check from first true failure (uf>=0.5)
    censor_pen = 0.0
    ff = _first_failure(results)
    if ff is not None:
        Nf = float(ff.get("N", 0.0) or 0.0)
        if Nf > 0 and Nf > N_max:
            msf = float(ff.get("max_steps", 1.0) or 1.0)
            medf = float(ff.get("median_step", msf) or msf)
            lb = max(msf, medf)  # lower bound consistent with observed censoring
            pred_med_f = _tail_fit_pred_med(Nf, Ns, meds, windows=(2, 3, 4))
            # penalize only if prediction is too optimistic vs lower bound
            censor_pen = 0.10 * _overshoot_pen(lb / max(pred_med_f, 1e-12), safety=1.0, tau=0.12)

    # tiny frontier tie-breaker
    tie = 0.002 * math.log10(max(med_at_max, 1.0))

    return float(
        reach_term
        + 1.2 * tail_headroom_pen
        + 2.0 * next_clear_pen
        + 1.0 * cliff_pen
        + accel_pen
        + censor_pen
        + tie
    )


def _objective_v2(results):
    """Reach-dominant via robust clearance N*: largest scheduled N with worst-window predicted median <= safety*budget."""
    results = sorted(results, key=lambda r: float(r.get("N", 0.0)))
    N_cap, max_steps_cap = _infer_caps(results)
    sched_Ns, count_map = _schedule_grid(N_cap=int(N_cap), max_steps_cap=float(max_steps_cap))

    byN = {}
    for r in results:
        if "N" not in r:
            continue
        try:
            N = int(r["N"])
        except Exception:
            continue
        byN.setdefault(N, []).append(r)

    # collect successful scheduled prefix (pass only)
    cleared = []
    for N in sched_Ns:
        runs = byN.get(int(N))
        if runs is None:
            break
        best = _best_pass_run(runs)
        if best is None:
            break
        med = float(best.get("median_step", 1.0) or 1.0)
        med = max(med, 1.0)
        cleared.append((float(N), med))

    if not cleared:
        N0 = float(min(byN.keys())) if byN else 1.0
        return 1e6 + 1e2 * math.log1p(N0)

    Ns = np.array([t[0] for t in cleared], dtype=float)
    meds = np.array([t[1] for t in cleared], dtype=float)
    N_max = float(Ns[-1])

    safety = 0.70

    # compute N* over the schedule up to cap
    okNs = []
    for Nt in sched_Ns:
        pred_med = _tail_fit_pred_med(Nt, Ns, meds, windows=(2, 3, 5))
        bud = _budget_for_N(int(Nt), count_map, max_steps_cap=float(max_steps_cap))
        if pred_med <= safety * bud:
            okNs.append(float(Nt))
    N_star = float(max(okNs)) if okNs else float(sched_Ns[0])

    # reach-dominant objective
    reach_term = -N_star

    # tie-breakers: tail ratio + cliff
    budgets = np.array([_budget_for_N(int(n), count_map, max_steps_cap=float(max_steps_cap)) for n in Ns], dtype=float)
    ratios = np.clip(meds / np.maximum(budgets, 1.0), 1e-12, 1e6)
    K = int(min(4, len(ratios)))
    tail_ratio_med = float(np.median(ratios[-K:]))
    tail_pen = _overshoot_pen(tail_ratio_med, safety=0.70, tau=0.10)

    cliff = 0.0
    if len(ratios) >= 2:
        dlog_r = float(math.log(ratios[-1]) - math.log(ratios[-2]))
        cliff = 0.08 * _softplus((dlog_r - 0.12) / 0.05)

    # mild preference to also truly clear current max success
    cur_bud = _budget_for_N(int(round(N_max)), count_map, max_steps_cap=float(max_steps_cap))
    cur_pen = _overshoot_pen(float(meds[-1]) / max(cur_bud, 1.0), safety=0.75, tau=0.12)

    return float(reach_term + 0.8 * tail_pen + cliff + 0.3 * cur_pen)


def _objective_v3(results):
    """Reach-dominant; extra emphasis on budget-cliff hazard and overshoot at critical next Ns (1280/1810/2560)."""
    results = sorted(results, key=lambda r: float(r.get("N", 0.0)))
    N_cap, max_steps_cap = _infer_caps(results)
    sched_Ns, count_map = _schedule_grid(N_cap=int(N_cap), max_steps_cap=float(max_steps_cap))

    byN = {}
    for r in results:
        if "N" not in r:
            continue
        try:
            N = int(r["N"])
        except Exception:
            continue
        byN.setdefault(N, []).append(r)

    cleared = []
    for N in sched_Ns:
        runs = byN.get(int(N))
        if runs is None:
            break
        best = _best_pass_run(runs)
        if best is None:
            break
        med = float(best.get("median_step", 1.0) or 1.0)
        med = max(med, 1.0)
        cleared.append((float(N), med))

    if not cleared:
        N0 = float(min(byN.keys())) if byN else 1.0
        return 1e6 + 1e2 * math.log1p(N0)

    Ns = np.array([t[0] for t in cleared], dtype=float)
    meds = np.array([t[1] for t in cleared], dtype=float)
    N_max = float(Ns[-1])

    reach_term = -N_max

    # budgets + ratios
    budgets = np.array([_budget_for_N(int(n), count_map, max_steps_cap=float(max_steps_cap)) for n in Ns], dtype=float)
    ratios = np.clip(meds / np.maximum(budgets, 1.0), 1e-12, 1e6)

    # tail cliff hazards: max of last 3 deltas
    dlog_r_max = 0.0
    dlog_m_max = 0.0
    if len(ratios) >= 2:
        k = int(min(4, len(ratios)))
        dlog_r = np.diff(np.log(np.maximum(ratios[-k:], 1e-12)))
        dlog_m = np.diff(np.log(np.maximum(meds[-k:], 1.0)))
        dlog_r_max = float(np.max(dlog_r))
        dlog_m_max = float(np.max(dlog_m))

    cliff_pen = (
        0.12 * _softplus((dlog_r_max - 0.10) / 0.04)
        + 0.07 * _softplus((dlog_m_max - 0.85) / 0.20)
    )

    # critical overshoot checks (only if within schedule cap)
    safety = 0.70
    crit = []
    for Nt in (1280.0, 1810.0, 2560.0):
        if Nt <= float(N_cap) and Nt > 0:
            crit.append(Nt)
    # score those >= current max success more strongly
    pens = []
    for Nt in crit:
        pred_med = _tail_fit_pred_med(Nt, Ns, meds, windows=(2, 3, 4))
        bud = _budget_for_N(int(round(Nt)), count_map, max_steps_cap=float(max_steps_cap))
        w = 1.0 if Nt >= N_max else 0.35
        pens.append(w * _overshoot_pen(pred_med / max(bud, 1.0), safety=safety, tau=0.10))
    crit_pen = float(np.mean(pens)) if pens else 0.0

    # tail ratio headroom (median of last 2–4)
    K = int(min(4, len(ratios)))
    tail_ratio_med = float(np.median(ratios[-K:]))
    tail_pen = _overshoot_pen(tail_ratio_med, safety=0.70, tau=0.10)

    return float(reach_term + 1.4 * tail_pen + 2.4 * crit_pen + 1.2 * cliff_pen)


def objective(experiment_results):
    """Estimate the research goal from experiment results.

    Args:
        experiment_results: List of dicts with experiment details
    Returns:
        Float to be MINIMIZED
    """
    if not experiment_results:
        return 1e6

    variant = int(os.environ.get("OBJ_VARIANT", "1"))
    if variant == 2:
        return _objective_v2(experiment_results)
    if variant == 3:
        return _objective_v3(experiment_results)
    return _objective_v1(experiment_results)
