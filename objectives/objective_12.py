import numpy as np

def _softplus(x):
    # numerically stable softplus
    x = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _wls_fit(x, y, w):
    """Weighted least squares for y ~ b0 + b1*x."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    w = np.clip(w, 1e-12, None)

    X = np.stack([np.ones_like(x), x], axis=1)
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw

    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    sse = float(np.sum(w * resid * resid))
    return beta, yhat, sse


def _aicc_from_sse(sse, n, k):
    # Gaussian likelihood with MLE sigma^2 = sse/n
    n = int(n)
    k = int(k)
    if n <= k + 1:
        return float("inf")
    sse = max(float(sse), 1e-12)
    sigma2 = sse / n
    loglik = -0.5 * n * (np.log(2.0 * np.pi * sigma2) + 1.0)
    aic = 2.0 * k - 2.0 * loglik
    aicc = aic + (2.0 * k * (k + 1)) / (n - k - 1)
    return float(aicc)


def objective(experiment_results):
    """Estimate the research goal from experiment results.

    Returns a smooth proxy for large-N performance:
    - Expected restart-steps (accounts for failures via unsolved_fraction)
    - AICc model-averaged extrapolation to N=2000 (power vs exponential)
    - Smooth penalties for weak reliability margin and super-quadratic scaling
    """
    if not experiment_results:
        return 1e3

    # Extract and sort by N
    rows = []
    for r in experiment_results:
        if 'N' not in r or 'median_step' not in r or 'unsolved_fraction' not in r:
            continue
        N = float(r['N'])
        u = float(r.get('unsolved_fraction', 1.0))
        med = float(r.get('median_step', np.nan))
        ms = float(r.get('max_steps', np.nan))
        if not np.isfinite(N) or N <= 0:
            continue
        if not np.isfinite(u):
            u = 1.0
        u = float(np.clip(u, 0.0, 1.0))
        if not np.isfinite(ms) or ms <= 0:
            ms = max(50.0, med if np.isfinite(med) else 50.0)
        if not np.isfinite(med) or med <= 0:
            med = ms
        # Expected steps per successful solve with geometric restarts:
        # E = (p*med + (1-p)*max_steps)/p = med + u*max_steps/(1-u)
        p = float(np.clip(1.0 - u, 1e-3, 1.0))
        restart_steps = (med + u * ms) / p
        rows.append((N, u, med, ms, restart_steps, p))

    if len(rows) < 2:
        return 5e2

    rows.sort(key=lambda t: t[0])
    N = np.array([t[0] for t in rows], dtype=float)
    u = np.array([t[1] for t in rows], dtype=float)
    y = np.array([t[4] for t in rows], dtype=float)
    p = np.array([t[5] for t in rows], dtype=float)

    # Work in ln-space
    lnN = np.log(N)
    lny = np.log(np.clip(y, 1e-12, None))

    # Weights: emphasize reliable + larger-N points
    w = (p ** 2) * (N / np.max(N)) ** 0.5
    w = np.clip(w, 1e-6, None)

    # Fit power law: ln y = a + b ln N
    beta_pow, _, sse_pow = _wls_fit(lnN, lny, w)
    a_pow, b_pow = beta_pow
    aicc_pow = _aicc_from_sse(sse_pow, len(N), k=2)

    # Fit exponential: ln y = a + b N
    beta_exp, _, sse_exp = _wls_fit(N, lny, w)
    a_exp, b_exp = beta_exp
    aicc_exp = _aicc_from_sse(sse_exp, len(N), k=2)

    # AICc weights with a mild prior preference for power-law (goal is polynomial)
    prior_pow = 0.0
    prior_exp = 2.0
    a_pow_eff = aicc_pow + prior_pow
    a_exp_eff = aicc_exp + prior_exp
    amin = min(a_pow_eff, a_exp_eff)
    wp = np.exp(-0.5 * (a_pow_eff - amin))
    we = np.exp(-0.5 * (a_exp_eff - amin))
    s = wp + we
    wp, we = (wp / s, we / s) if s > 0 else (0.5, 0.5)

    N_target = 2000.0
    pred_pow = float(np.exp(a_pow + b_pow * np.log(N_target)))
    pred_exp = float(np.exp(a_exp + b_exp * N_target))
    pred = wp * pred_pow + we * pred_exp
    base = float(np.log10(np.clip(pred, 1e-12, None)))

    # Reliability margin penalty near schedule threshold (u<0.5 continues)
    k = min(3, len(u))
    u_top = u[-k:]
    margin = float(np.min(0.5 - u_top))  # negative means already failing
    # Want a comfortable margin: u <= 0.38 (margin >= 0.12)
    m0, tau_m = 0.12, 0.05
    pen_margin = float(_softplus((m0 - margin) / tau_m))

    # Scaling penalty for super-quadratic power exponent (goal: polynomial, ideally small)
    # b_pow is exponent in y ~ N^b
    b0, tau_b = 2.0, 0.25
    pen_slope = float(_softplus((b_pow - b0) / tau_b))

    # Mild penalty if exponential model dominates (encourage polynomial-like fits)
    pen_exp = float(1.0 - wp)

    # Combine (tuned to keep penalties smooth, not dominating)
    obj = base + 0.35 * pen_margin + 0.25 * pen_slope + 0.10 * pen_exp
    return float(obj)
