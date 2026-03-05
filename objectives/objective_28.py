import math
import numpy as np


def _softplus(x: float) -> float:
    # stable scalar softplus
    if x > 60.0:
        return x
    if x < -60.0:
        return math.exp(x)
    return math.log1p(math.exp(x))


def _first_failure(results):
    for r in results:
        if float(r.get("unsolved_fraction", 1.0)) >= 0.5:
            return r
    return None


def _successful_prefix_dedup(results):
    if not results:
        return []
    results = sorted(results, key=lambda r: float(r.get("N", 0.0)))
    fail = _first_failure(results)
    if fail is not None:
        # keep only points strictly before the first failure N
        Nf = float(fail.get("N", 0.0))
        results = [r for r in results if float(r.get("N", 0.0)) < Nf]
    succ = [r for r in results if float(r.get("unsolved_fraction", 1.0)) < 0.5]

    # Dedup by N: keep run with largest max_steps
    byN = {}
    for r in succ:
        N = int(r.get("N", 0))
        if N not in byN or float(r.get("max_steps", 0.0)) > float(byN[N].get("max_steps", 0.0)):
            byN[N] = r
    return [byN[N] for N in sorted(byN.keys())]


def _schedule_count_map(N_start=10, N_cap=5120, max_len=256):
    # map scheduled N -> experiment_count (handles rounding collisions by keeping max count)
    m = {}
    for c in range(max_len):
        N = int(round(N_start * (2 ** (c / 2))))
        if N > N_cap:
            break
        m[N] = max(m.get(N, -10**9), c)
        if N == N_cap:
            break
    return m


_COUNT_MAP = _schedule_count_map()


def _budget_for_N(N: int, cap: float) -> float:
    # schedule: max_steps = min(50 * 2**count, cap)
    c = _COUNT_MAP.get(int(N))
    if c is None:
        # fallback to nearest scheduled N
        keys = np.array(sorted(_COUNT_MAP.keys()), dtype=int)
        j = int(np.argmin(np.abs(keys - int(N))))
        c = _COUNT_MAP[int(keys[j])]
    return float(min(50.0 * (2.0 ** float(c)), float(cap)))


def objective(experiment_results):
    if not experiment_results:
        return 1e3

    results = sorted(experiment_results, key=lambda r: float(r.get("N", 0.0)))
    succ = _successful_prefix_dedup(results)
    if not succ:
        return 400.0

    # Gate activates meaningfully once we have tail around 640+
    gateN = 640.0

    Ns = np.array([max(float(r.get("N", 1.0)), 1.0) for r in succ], dtype=float)
    meds = np.array([max(float(r.get("median_step", 1.0)), 1.0) for r in succ], dtype=float)
    msteps = np.array([max(float(r.get("max_steps", 1.0)), 1.0) for r in succ], dtype=float)

    N_max = float(Ns[-1])
    r_last = float(min(max(meds[-1] / max(msteps[-1], 1.0), 1e-12), 0.999999))

    # smooth penalty when below the gate (prevents early-failing / short-reaching designs ranking high)
    gate_pen = 30.0 * _softplus(math.log(gateN / max(N_max, 1.0)) / 0.25)

    # explicit failure penalty if any failure point exists anywhere
    fail = _first_failure(results)
    fail_pen = 0.0
    if fail is not None:
        Nf = max(float(fail.get("N", 1.0)), 1.0)
        # larger penalty if failure happens before 640
        fail_pen = 6.0 + 6.0 * _softplus(math.log(gateN / Nf) / 0.35)

    # If not at/above 640, return mostly gate/failure shaping + mild reach tie-break
    reach_tie = -0.03 * math.log2(max(N_max, 1.0))
    if N_max < gateN:
        return float(gate_pen + fail_pen + reach_tie + 2.0 * r_last)

    # --- Tail-local exponent k from last 2-3 successful points (schedule-local) ---
    t = int(min(3, len(succ)))
    logN = np.log(np.maximum(Ns[-t:], 1.0))
    logM = np.log(np.maximum(meds[-t:], 1.0))
    if t >= 2:
        slopes = np.diff(logM) / np.maximum(np.diff(logN), 1e-12)
        k_local = float(np.median(slopes))
    else:
        k_local = 10.0

    # penalize super-quadratic implied scaling (k>2)
    k0 = 2.0
    tau_k = 0.20
    k_pen = tau_k * _softplus((k_local - k0) / tau_k)

    # --- Two-jump headroom: predicted median/max_steps at 905 and 1280 ---
    # High-fidelity cap implied by reaching 640 in this schedule
    cap = 1e6
    targets = [905, 1280]
    safety = 0.65
    tau_r = 0.10

    ratio_terms = []
    for Nt in targets:
        # Use actual if present, else predict from last point with k_local
        idx = None
        for i, n in enumerate(Ns.tolist()):
            if int(round(n)) == int(Nt):
                idx = i
                break
        if idx is not None:
            med_pred = float(meds[idx])
        else:
            med_pred = float(meds[-1] * (float(Nt) / max(N_max, 1.0)) ** k_local)

        bud = _budget_for_N(int(Nt), cap=cap)
        r_pred = float(min(max(med_pred / max(bud, 1.0), 1e-12), 0.999999))

        # penalize overshoot above safety in a smooth, multiplicative way
        ratio_terms.append(tau_r * _softplus(math.log(max(r_pred / safety, 1e-12)) / tau_r))

    twojump_pen = float(np.mean(ratio_terms)) if ratio_terms else 0.0

    # include last-point ratio directly (as requested), but do NOT penalize unsolved_fraction~0.49
    last_ratio_term = 2.0 * r_last

    return float(
        gate_pen
        + fail_pen
        + reach_tie
        + last_ratio_term
        + 7.0 * twojump_pen
        + 3.0 * k_pen
    )
