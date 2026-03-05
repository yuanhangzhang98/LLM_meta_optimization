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

    # Dedup by N: keep the run with largest max_steps
    byN = {}
    for r in succ:
        N = int(r.get("N", 0))
        if N not in byN or float(r.get("max_steps", 0.0)) > float(byN[N].get("max_steps", 0.0)):
            byN[N] = r
    return [byN[N] for N in sorted(byN.keys())]


def _infer_N_cap(results):
    fset = {str(r.get("fidelity", "")).lower() for r in (results or [])}
    if "high" in fset:
        return 2560
    if "medium" in fset:
        return 1280
    return 640


def _schedule_Ns(N_start=10, N_cap=640, max_len=128):
    Ns = []
    for exp_count in range(max_len):
        N = int(round(N_start * (2 ** (exp_count / 2))))
        if Ns and N == Ns[-1]:
            continue
        Ns.append(N)
        if N >= N_cap:
            break
    # ensure sorted unique
    Ns = sorted(set(Ns))
    return Ns


def _softplus(x):
    # scalar stable softplus
    if x > 50.0:
        return x
    if x < -50.0:
        return math.exp(x)
    return math.log1p(math.exp(x))


def _predicted_failure_N(Ns_succ, ratios_succ, N_max, N_cap, safety=0.70, windows=(2, 3, 5)):
    """Conservative: return min_w earliest scheduled N>=N_max where pred_ratio > safety."""
    sched = _schedule_Ns(N_start=10, N_cap=N_cap)

    # start scan at first scheduled N >= N_max (robust to rounding)
    start_idx = 0
    for i, N in enumerate(sched):
        if N >= int(round(N_max)):
            start_idx = i
            break

    logN_all = np.log(np.maximum(Ns_succ, 1.0))
    logR_all = np.log(np.maximum(ratios_succ, 1e-12))

    fail_candidates = []
    for w in windows:
        if len(Ns_succ) >= 2:
            ww = min(int(w), len(Ns_succ))
            X = logN_all[-ww:]
            Y = logR_all[-ww:]
            if len(X) >= 2:
                b, a = np.polyfit(X, Y, 1)
                a = float(a)
                b = float(b)
            else:
                a = float(Y[-1])
                b = 0.0
        else:
            # cannot fit; assume flat at last observed ratio
            a = float(logR_all[-1])
            b = 0.0

        failN = None
        for N in sched[start_idx:]:
            pred_logR = a + b * math.log(max(float(N), 1.0))
            if math.exp(pred_logR) > safety:
                failN = float(N)
                break
        if failN is None:
            failN = float(2 * N_cap)  # "no predicted failure within cap" sentinel
        fail_candidates.append(failN)

    return float(min(fail_candidates))


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

    # (1) Reach dominates
    reach_term = -math.log2(max(N_max, 1.0))

    # (2) Conservative predicted failure point in ratio space
    N_cap = int(_infer_N_cap(experiment_results))
    N_fail_pred = _predicted_failure_N(Ns, ratios, N_max=N_max, N_cap=N_cap, safety=0.70, windows=(2, 3, 5))
    fail_term = -math.log2(max(N_fail_pred, 1.0))

    # (3) Budget-cliff penalty: big positive jump in log(ratio) between last two successes
    cliff_pen = 0.0
    if len(ratios) >= 2:
        dlog = float(math.log(ratios[-1]) - math.log(ratios[-2]))
        # tolerate small changes; penalize sharp worsening
        cliff_pen = 0.08 * _softplus((dlog - 0.12) / 0.05)

    # tiny tie-breaker: last ratio itself (prefer more headroom)
    last_ratio_pen = 0.03 * _softplus((float(ratios[-1]) - 0.70) / 0.05)

    # Combine (reach primary; failure distance secondary)
    return reach_term + 0.10 * fail_term + cliff_pen + last_ratio_pen
