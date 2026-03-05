import math
import numpy as np


def _softplus(x):
    # numerically stable softplus
    x = float(x)
    if x > 40:
        return x
    if x < -40:
        return math.exp(x)
    return math.log1p(math.exp(x))


def _wls_fit(X, y, w, ridge=1e-8):
    """Weighted least squares with tiny ridge for stability."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    w = np.clip(w, 1e-12, np.inf)

    # Solve (X^T W X + ridge I) beta = X^T W y
    WX = X * w[:, None]
    A = X.T @ WX
    A = A + ridge * np.eye(A.shape[0])
    b = X.T @ (w * y)
    beta = np.linalg.solve(A, b)

    yhat = X @ beta
    resid = y - yhat
    sse = float(np.sum(w * resid * resid))
    n = int(len(y))
    k = int(X.shape[1])

    # Gaussian AIC up to constants; guard sse
    sse = max(sse, 1e-12)
    aic = n * math.log(sse / max(n, 1)) + 2 * k

    # Prediction variance at a new point x*: sigma^2 * x*^T (X^T W X)^-1 x*
    # Use unweighted sigma^2 proxy from weighted SSE / (n-k)
    dof = max(n - k, 1)
    sigma2 = sse / dof
    cov = np.linalg.inv(A)

    return beta, yhat, resid, sse, aic, sigma2, cov


def objective(experiment_results):
    """Estimate the research goal from experiment results.

    Returns a smooth proxy for log10 expected solve-steps at N=2000 that:
      - accounts for partial failures via a restart-style expected-steps proxy
      - penalizes being near the schedule threshold (unsolved_fraction ~ 0.5)
      - model-averages polynomial vs exponential scaling using AIC + a poly prior
      - adds an extrapolation/coverage penalty when max tested N is small

    Args:
        experiment_results: List of dicts with experiment details
    Returns:
        Float to be MINIMIZED
    """
    if not experiment_results:
        return 1e6

    # Extract and sanitize
    rows = []
    for r in experiment_results:
        if 'N' not in r:
            continue
        N = float(r['N'])
        if not np.isfinite(N) or N <= 0:
            continue
        u = float(r.get('unsolved_fraction', 1.0))
        u = float(np.clip(u, 0.0, 1.0))
        s = 1.0 - u
        med = float(r.get('median_step', np.nan))
        if not np.isfinite(med) or med <= 0:
            med = float(r.get('max_steps', 1.0))
        max_steps = float(r.get('max_steps', max(1.0, med)))
        max_steps = max(1.0, max_steps)

        # Expected-steps proxy under independent restarts:
        # E[T] ≈ observed_median + (failure/success)*budget
        eps = 1e-6
        steps_eff = med + (u / max(s, eps)) * max_steps
        steps_eff = float(np.clip(steps_eff, 1.0, 1e20))
        log_steps = math.log10(steps_eff)

        # Reliability margin penalty (smooth): prefer u well below 0.5
        u_target = 0.20
        tau = 0.05
        lam = 1.0
        rel_pen = lam * _softplus((u - u_target) / tau)

        y = log_steps + rel_pen

        # Weight bigger N more (focus on scaling), but keep small-N informative
        w = (N ** 0.5)
        rows.append((N, y, w))

    if len(rows) < 2:
        return 5e5

    Ns = np.array([t[0] for t in rows], dtype=float)
    ys = np.array([t[1] for t in rows], dtype=float)
    ws = np.array([t[2] for t in rows], dtype=float)

    # Light robustification: downweight outliers by one IRLS step with Huber-like rule
    # Initial fit (poly in log10 N)
    x_log10 = np.log10(Ns)
    Xp = np.column_stack([np.ones_like(x_log10), x_log10])
    beta0, yhat0, resid0, sse0, aic0, sigma20, cov0 = _wls_fit(Xp, ys, ws)

    mad = np.median(np.abs(resid0 - np.median(resid0)))
    scale = max(1e-6, 1.4826 * mad)
    c = 1.5 * scale
    w_rob = 1.0 / np.maximum(1.0, np.abs(resid0) / c)
    w2 = ws * w_rob

    # Refit with robust weights
    beta_p, yhat_p, resid_p, sse_p, aic_p, sigma2_p, cov_p = _wls_fit(Xp, ys, w2)

    # Exponential-in-N model on log10 steps: y = a + b*N
    Xe = np.column_stack([np.ones_like(Ns), Ns])
    beta_e, yhat_e, resid_e, sse_e, aic_e, sigma2_e, cov_e = _wls_fit(Xe, ys, w2)

    # AIC model averaging with a mild prior favoring polynomial scaling
    # (prevents small-N data from spuriously preferring exponential)
    prior_poly = 0.8
    prior_exp = 0.2
    aics = np.array([aic_p, aic_e], dtype=float)
    da = aics - np.min(aics)
    w_aic = np.exp(-0.5 * da)
    w_aic = w_aic / np.sum(w_aic)
    w_post = np.array([prior_poly, prior_exp], dtype=float) * w_aic
    w_post = w_post / np.sum(w_post)

    # Predict at target N
    N_target = 2000.0
    xpt = np.array([1.0, math.log10(N_target)], dtype=float)
    xet = np.array([1.0, N_target], dtype=float)

    pred_p = float(xpt @ beta_p)
    pred_e = float(xet @ beta_e)

    # Add a small uncertainty premium (upper-confidence-like) to reduce over-optimistic extrapolation
    # pred_var = sigma^2 * x^T cov x
    var_p = float(sigma2_p * (xpt @ cov_p @ xpt))
    var_e = float(sigma2_e * (xet @ cov_e @ xet))
    std_p = math.sqrt(max(var_p, 0.0))
    std_e = math.sqrt(max(var_e, 0.0))
    eta = 0.5

    pred = w_post[0] * (pred_p + eta * std_p) + w_post[1] * (pred_e + eta * std_e)

    # Coverage penalty: if we never reached large N, discourage optimistic extrapolation
    maxN = float(np.max(Ns))
    cover = max(0.0, math.log10(N_target / max(maxN, 1.0)))
    cover_pen = 0.75 * cover

    # Additional smooth penalty for poly exponent > ~1.2 (push toward near-linear steps)
    # In y=log10(steps)=a+p*log10(N), slope==p
    p_est = float(beta_p[1])
    p_pen = 0.5 * _softplus((p_est - 1.2) / 0.15)

    objective_value = float(pred + cover_pen + p_pen)
    return objective_value
