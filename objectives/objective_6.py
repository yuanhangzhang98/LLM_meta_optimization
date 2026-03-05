import numpy as np


def _wls_fit(X, y, w):
    """Weighted least squares fit: y ~ X @ b, returns b and SSE."""
    w = np.asarray(w, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)

    w = np.clip(w, 0.0, np.inf)
    if np.all(w == 0):
        w = np.ones_like(w)

    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw
    b, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = y - X @ b
    sse = float(np.sum(w * resid * resid))
    return b, sse


def objective(experiment_results):
    """Estimate the research goal from experiment results.

    Returns a smooth proxy for large-N performance:
      - converts (unsolved_fraction, median_step) into expected steps with restarts
      - fits polynomial and exponential-in-N models in log space
      - predicts expected steps at N=2000
      - adds mild penalties for poor scaling, low coverage, and low success at largest N

    Args:
        experiment_results: List of dicts with experiment details
    Returns:
        Float to be MINIMIZED
    """
    if not experiment_results:
        return 1e3
    if len(experiment_results) < 2:
        return 5e2

    # Extract arrays
    Ns = np.array([float(r.get('N', np.nan)) for r in experiment_results], dtype=float)
    us = np.array([float(r.get('unsolved_fraction', 1.0)) for r in experiment_results], dtype=float)
    ms = np.array([float(r.get('median_step', np.nan)) for r in experiment_results], dtype=float)
    max_steps = np.array([float(r.get('max_steps', np.nan)) for r in experiment_results], dtype=float)

    ok = np.isfinite(Ns) & np.isfinite(us) & np.isfinite(ms)
    if np.sum(ok) < 2:
        return 5e2

    Ns, us, ms, max_steps = Ns[ok], us[ok], ms[ok], max_steps[ok]

    # Convert to expected work with restarts.
    # Use a small floor on success prob so objective remains smooth.
    p = 1.0 - np.clip(us, 0.0, 1.0)
    p_eff = np.clip(p + 0.05, 0.05, 1.05)  # smooth, avoids blow-ups near p=0

    # Censor at max_steps if provided (smoothly by min())
    ms_eff = ms
    finite_cap = np.isfinite(max_steps)
    ms_eff = np.where(finite_cap, np.minimum(ms_eff, max_steps), ms_eff)

    eff_steps = (np.maximum(ms_eff, 1.0) + 1.0) / p_eff
    y = np.log(eff_steps)

    N_target = 2000.0

    # Weights: emphasize larger N (more indicative of scaling)
    w = (Ns / np.max(Ns)) ** 1.5

    # Model 1: polynomial scaling  log(eff) ~ a + b*log(N)
    x_poly = np.log(np.maximum(Ns, 2.0))
    X_poly = np.column_stack([np.ones_like(x_poly), x_poly])
    b_poly, sse_poly = _wls_fit(X_poly, y, w)
    yhat_poly = float(b_poly[0] + b_poly[1] * np.log(N_target))

    # Model 2: exponential scaling log(eff) ~ a + b*(N/N_target)
    x_exp = Ns / N_target
    X_exp = np.column_stack([np.ones_like(x_exp), x_exp])
    b_exp, sse_exp = _wls_fit(X_exp, y, w)
    yhat_exp = float(b_exp[0] + b_exp[1] * 1.0)

    # AIC-style model averaging (stable even when R^2 is weird)
    n = float(len(y))
    eps = 1e-12
    k = 2.0
    aic_poly = n * np.log(max(sse_poly / n, eps)) + 2.0 * k
    aic_exp = n * np.log(max(sse_exp / n, eps)) + 2.0 * k
    aic_min = min(aic_poly, aic_exp)
    w_poly = np.exp(-0.5 * (aic_poly - aic_min))
    w_exp = np.exp(-0.5 * (aic_exp - aic_min))
    w_sum = w_poly + w_exp
    w_poly, w_exp = w_poly / w_sum, w_exp / w_sum

    # Predicted log10 expected steps at N_target
    yhat = w_poly * yhat_poly + w_exp * yhat_exp
    pred_log10 = yhat / np.log(10.0)

    # Mild penalties
    # 1) Penalize super-quadratic estimated polynomial exponent (encourage poly-time)
    b = float(b_poly[1])
    slope_penalty = 0.15 * np.log1p(np.exp(b - 2.0))  # softplus(b-2)

    # 2) Penalize limited coverage (ending at small N makes extrapolation unreliable)
    N_max = float(np.max(Ns))
    coverage = np.log(max(N_target / max(N_max, 1.0), 1.0))
    coverage_penalty = 0.30 * coverage

    # 3) Penalize low success probability at largest tested N
    idx_max = int(np.argmax(Ns))
    p_last = float(p[idx_max])
    success_penalty = 0.50 * np.log1p(np.exp((0.5 - p_last) / 0.05))  # softplus

    return float(pred_log10 + slope_penalty + coverage_penalty + success_penalty)
