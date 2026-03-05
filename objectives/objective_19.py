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
        return 1e6

    def _f(x, d):
        try:
            return float(x)
        except Exception:
            return float(d)

    def _sigmoid(x):
        x = np.asarray(x, dtype=np.float64)
        x = np.clip(x, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-x))

    def _softplus(x):
        x = np.asarray(x, dtype=np.float64)
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

    res = sorted(experiment_results, key=lambda r: _f(r.get('N', 0), 0.0))

    Ns = np.array([max(1.0, _f(r.get('N', 1.0), 1.0)) for r in res], dtype=np.float64)
    u = np.clip(np.array([_f(r.get('unsolved_fraction', 1.0), 1.0) for r in res], dtype=np.float64), 0.0, 1.0)
    ms = np.array([max(0.0, _f(r.get('median_step', 0.0), 0.0)) for r in res], dtype=np.float64)
    mx = np.array([max(1.0, _f(r.get('max_steps', 1.0), 1.0)) for r in res], dtype=np.float64)
    step_frac = np.clip(ms / mx, 0.0, 2.0)

    L = float(len(res))
    largest_N = float(np.max(Ns))

    # --- soft-pass area in the observed regime u≈0.45–0.55 ---
    tau_u = 0.03
    pass_soft = _sigmoid((0.5 - u) / tau_u)  # 0..1, ~0.5 at u=0.5
    logN = np.log1p(Ns)

    # reward both more levels and higher N (sum, not mean)
    area_reward = float(np.sum(logN * pass_soft))

    # penalize wasting budget, especially when near/above the pass boundary
    step_pen = float(np.sum(logN * (0.3 + 0.7 * (1.0 - pass_soft)) * np.clip(step_frac, 0.0, 1.0)))

    # smooth penalty for failing the u<0.5 gate (operates near 0.5)
    fail_pen = float(np.sum(logN * (_softplus((u - 0.5) / tau_u) * tau_u)))

    # --- top-level term (largest tested N only) ---
    u_top = float(u[-1])
    sf_top = float(np.clip(step_frac[-1], 0.0, 1.0))
    top_reward = float(math.log1p(largest_N) * (1.0 / (1.0 + math.exp(-(0.5 - u_top) / tau_u))))
    top_fail_pen = float((_softplus((u_top - 0.5) / tau_u) * tau_u))

    # --- scaling on successes only: slope of log(median_step) vs log(N) for u<0.5 ---
    ok = (u < 0.5) & (ms > 0)
    ok_idx = np.where(ok)[0]
    if ok_idx.size >= 3:
        x = np.log(Ns[ok_idx])
        y = np.log(ms[ok_idx] + 1.0)
        slope = float(np.polyfit(x, y, 1)[0])
        slope_pen = max(0.0, slope)  # lower is better; negative slope not penalized
        insufficient_pen = 0.0
    else:
        slope_pen = 2.0  # fixed penalty so failure/short runs can't look good
        insufficient_pen = 2.0

    # --- anti-truncation (timeouts/early stops) ---
    # penalize too few completed levels and too-small largest_N (smooth)
    target_levels = 9.0
    target_N = 320.0
    trunc_pen = float(
        3.0 * (_softplus((target_levels - L) / 1.0))
        + 3.0 * (_softplus((target_N - largest_N) / 80.0))
    )

    # monotone reward for pushing N upward (keeps objectives aligned)
    largest_reward = math.log1p(largest_N)

    # --- final score (minimize) ---
    score = (
        -60.0 * area_reward
        -120.0 * top_reward
        -40.0 * largest_reward
        +180.0 * fail_pen
        +35.0 * step_pen
        +70.0 * (top_fail_pen + 0.3 * sf_top)
        +60.0 * slope_pen
        +80.0 * insufficient_pen
        +120.0 * trunc_pen
    )

    if not np.isfinite(score):
        return 1e6
    return float(score)
