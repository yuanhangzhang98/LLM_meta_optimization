import math
import numpy as np


def objective(experiment_results):
    """Estimate the research goal from experiment results.

    Args:
        experiment_results: List of dicts with experiment details
    Returns:
        Float to be MINIMIZED
    """
    if not experiment_results:
        return 1000.0

    # ---- helpers ----
    def _get(r, k, default):
        v = r.get(k, default)
        try:
            return float(v)
        except Exception:
            return float(default)

    def _softplus(x):
        # stable softplus
        x = np.asarray(x, dtype=np.float64)
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

    # ---- collect ----
    res = sorted(experiment_results, key=lambda r: _get(r, 'N', 0.0))
    Ns = np.array([max(0.0, _get(r, 'N', 0.0)) for r in res], dtype=np.float64)
    u = np.clip(np.array([_get(r, 'unsolved_fraction', 1.0) for r in res], dtype=np.float64), 0.0, 1.0)
    ms = np.array([max(0.0, _get(r, 'median_step', 0.0)) for r in res], dtype=np.float64)
    mx = np.array([max(1.0, _get(r, 'max_steps', 1.0)) for r in res], dtype=np.float64)

    # Weights emphasize high N, but remain normalized
    w = np.sqrt(np.maximum(Ns, 1.0))
    w = w / (w.sum() + 1e-12)

    # ---- robust caps (discrete) ----
    def maxN(th):
        ok = Ns[u <= th]
        return float(ok.max()) if ok.size else 0.0

    N30 = maxN(0.30)
    N20 = maxN(0.20)

    cap_reward = 200.0 * math.log1p(N30) + 100.0 * math.log1p(N20)

    # ---- continuous robustness penalty across all tested N ----
    tau = 0.05
    pen20 = _softplus((u - 0.20) / tau) * tau
    pen30 = _softplus((u - 0.30) / tau) * tau
    robust_pen = float(np.sum(w * (2.0 * pen20 + 1.0 * pen30)))

    # ---- brittleness penalty (punish sitting near the 0.5 gate) ----
    brittle = 1.0 / (1.0 + np.exp(-(u - 0.40) / 0.03))
    brittle_pen = float(np.sum(w * brittle))

    # ---- efficiency penalty (unsolved + normalized steps), high-N weighted ----
    step_frac = np.clip(ms / mx, 0.0, 1.0)
    lam = 0.2
    eff = u + lam * step_frac
    eff_pen = float(np.sum(w * eff))

    # ---- scaling exponent proxy: slope of log(steps) vs log(N) on top-K ----
    K = int(min(5, len(Ns)))
    if K >= 3:
        top_idx = np.argsort(Ns)[-K:]
        x = np.log(Ns[top_idx] + 1.0)
        ms_clip = np.minimum(ms[top_idx], 2.0 * mx[top_idx])
        y = np.log(ms_clip + 1.0)
        slope = float(np.polyfit(x, y, 1)[0])

        # extra penalty if top-N points are effectively failing
        high_fail = float(np.mean(np.clip((u[top_idx] - 0.45) / 0.05, 0.0, None)))
        scale_pen = slope + 3.0 * high_fail
    else:
        scale_pen = 0.0

    # ---- final (minimize) ----
    score = (
        -cap_reward
        + 400.0 * robust_pen
        + 50.0 * eff_pen
        + 30.0 * brittle_pen
        + 40.0 * scale_pen
    )

    # Guard against NaNs/infs
    if not np.isfinite(score):
        return 1000.0
    return float(score)
