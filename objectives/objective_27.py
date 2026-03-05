import math
import numpy as np


def _softplus(x):
    x = np.asarray(x, dtype=float)
    return np.where(x > 60.0, x, np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0))


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

    # Dedup by N: keep run with largest max_steps (more reliable median)
    byN = {}
    for r in succ:
        N = int(r.get("N", 0))
        if N not in byN or float(r.get("max_steps", 0.0)) > float(byN[N].get("max_steps", 0.0)):
            byN[N] = r
    return [byN[N] for N in sorted(byN.keys())]


def objective(experiment_results):
    if not experiment_results:
        return 1e3

    succ = _successful_prefix_dedup(experiment_results)
    if not succ:
        return 200.0

    Ns = np.array([max(float(r.get("N", 1.0)), 1.0) for r in succ], dtype=float)
    meds = np.array([max(float(r.get("median_step", 1.0)), 1.0) for r in succ], dtype=float)
    msteps = np.array([max(float(r.get("max_steps", 1.0)), 1.0) for r in succ], dtype=float)

    N_max = float(Ns[-1])

    # --- (1) Tail-local exponent + acceleration (d log(median_step) / d log(N)) ---
    t = int(min(5, len(succ)))
    logN = np.log(np.maximum(Ns[-t:], 1.0))
    logM = np.log(np.maximum(meds[-t:], 1.0))

    if t >= 2:
        dlogN = np.diff(logN)
        dlogM = np.diff(logM)
        slopes = dlogM / np.maximum(dlogN, 1e-12)
        slope_med = float(np.median(slopes))
        slope_max = float(np.max(slopes))

        # Penalize slopes above ~2 (since budget doubles while N grows ~sqrt(2))
        k0 = 2.0
        tau_k = 0.25
        slope_pen = (
            0.7 * float(_softplus((slope_med - k0) / tau_k)) * tau_k
            + 1.0 * float(_softplus((slope_max - k0) / tau_k)) * tau_k
        )

        # Positive acceleration (steepening) penalty
        if len(slopes) >= 2:
            accel = np.diff(slopes)
            accel_pos = np.maximum(accel, 0.0)
            tau_a = 0.20
            accel_pen = float(np.mean(_softplus(accel_pos / tau_a)) * tau_a)
        else:
            accel_pen = 0.0
    else:
        slope_pen = 6.0
        accel_pen = 2.0

    # --- (2) Pure budget-clearance margin (no fitting): log(max_steps/median_step) ---
    k = int(min(3, len(succ)))
    margins = np.log(np.maximum(msteps[-k:] / np.maximum(meds[-k:], 1.0), 1e-12))
    margin_min = float(np.min(margins))
    margin_med = float(np.median(margins))

    # Penalize low margins; target roughly corresponds to median/max_steps <= ~0.70
    m0 = math.log(1.0 / 0.70)  # ~0.356
    tau_m = 0.18
    margin_pen = (
        1.1 * float(_softplus((m0 - margin_min) / tau_m)) * tau_m
        + 0.7 * float(_softplus((m0 - margin_med) / tau_m)) * tau_m
    )

    # Budget cliff: sharp drop in margin between last two successes
    cliff_pen = 0.0
    if len(margins) >= 2:
        drop = float(margins[-1] - margins[-2])
        cliff = max(0.0, -drop)
        cliff_pen = 0.9 * float(_softplus((cliff - 0.12) / 0.06)) * 0.06

    # Mild tie-breakers among similarly-stable tails
    reach_term = -0.04 * math.log2(max(N_max, 1.0))
    tie_steps = 0.01 * math.log10(max(float(meds[-1]), 1.0))

    return float(reach_term + tie_steps + 1.0 * slope_pen + 1.2 * accel_pen + 1.3 * margin_pen + 1.0 * cliff_pen)
