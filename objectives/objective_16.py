import math
import numpy as np


def _softplus(x):
    # numerically stable softplus
    x = float(x)
    if x > 50:
        return x
    if x < -50:
        return math.exp(x)
    return math.log1p(math.exp(x))


def objective(experiment_results):
    """Estimate the research goal from experiment results.

    Args:
        experiment_results: List of dicts with experiment details
    Returns:
        Float to be MINIMIZED
    """
    if not experiment_results:
        return 1e6

    # Sort by size
    rs = sorted(experiment_results, key=lambda r: float(r.get('N', 0)))

    N = np.array([float(r.get('N', 0.0)) for r in rs], dtype=np.float64)
    u = np.array([float(r.get('unsolved_fraction', 1.0)) for r in rs], dtype=np.float64)
    max_steps = np.array([float(r.get('max_steps', 1.0)) for r in rs], dtype=np.float64)
    steps = np.array([float(r.get('median_step', max_steps[i])) for i, r in enumerate(rs)], dtype=np.float64)

    max_steps = np.maximum(max_steps, 1.0)
    steps = np.clip(steps, 0.0, max_steps)

    N_last = float(np.max(N)) if N.size else 0.0

    # Reference N for extrapolation: modestly beyond what was tested, but not too far
    N_ref = min(5120.0, max(640.0, 2.0 * max(N_last, 10.0)))

    # Per-experiment weights emphasize larger N
    denom = math.log1p(max(N_last, 1.0))
    w = (np.log1p(N) / max(denom, 1e-9)) ** 2
    w = np.maximum(w, 1e-6)

    # Penalize unsolved_fraction, especially near the schedule cutoff (0.5)
    u_target, u_tau = 0.30, 0.08
    pen_u = float(np.sum(w * np.vectorize(_softplus)((u - u_target) / u_tau)) / np.sum(w))

    # Penalize saturating the step budget (censoring/timeout-like behavior)
    s = steps / max_steps
    s_target, s_tau = 0.75, 0.10
    pen_budget = float(np.sum(w * np.vectorize(_softplus)((s - s_target) / s_tau)) / np.sum(w))

    # Additional penalty if the schedule stopped due to failure
    u_last = float(u[-1])
    pen_fail = 0.0
    if u_last >= 0.5:
        # Strong but smooth penalty; larger if failure happens at smaller N
        scale = 1.0 + math.log((N_ref + 1.0) / (N_last + 1.0))
        pen_fail = _softplus((u_last - 0.5) / 0.02) * scale

    # Fit polynomial scaling: log(steps+1) ~ a + p*log(N+1)
    x = np.log1p(N)
    y = np.log1p(steps)
    margin = np.maximum(0.0, 0.5 - u)
    w_fit = w * (margin + 0.05)  # avoid vanishing weights near the cutoff

    # Weighted ridge regression for stability
    W = np.diag(w_fit)
    X = np.stack([np.ones_like(x), x], axis=1)
    ridge = 1e-6
    XtWX = X.T @ W @ X + ridge * np.eye(2)
    XtWy = X.T @ W @ y
    try:
        a, p = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        a, p = float(np.mean(y)), 2.0

    y_pred = float(a + p * math.log1p(N_ref))

    # Reward reaching larger N, but only if success/efficiency terms support it
    reward_N = math.log1p(N_last)

    # Final objective (lower is better)
    obj = (
        y_pred
        + 0.30 * float(p)
        + 1.50 * pen_u
        + 0.80 * pen_budget
        + 2.00 * pen_fail
        - 0.70 * reward_N
    )

    # Guardrails
    if not math.isfinite(obj):
        return 1e6
    return float(obj)
