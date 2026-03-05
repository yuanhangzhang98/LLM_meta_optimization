import numpy as np
from scipy.optimize import minimize


def _softplus(z):
    # stable softplus
    z = np.asarray(z)
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0)


def _fit_censored_model(x, y, censored, lb, w, kind):
    """Fit y ~ f(theta, x) with squared loss + smooth hinge for censored points.

    For censored points: we only know y >= lb.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    censored = np.asarray(censored, dtype=bool)
    lb = np.asarray(lb, dtype=float)
    w = np.asarray(w, dtype=float)

    if kind == "power":
        # y = a + b*x
        def pred(theta):
            a, b = theta
            return a + b * x

        # init from uncensored OLS if possible
        if np.any(~censored):
            X = np.vstack([np.ones(np.sum(~censored)), x[~censored]]).T
            theta0, *_ = np.linalg.lstsq(X, y[~censored], rcond=None)
            theta0 = np.array(theta0)
        else:
            theta0 = np.array([np.median(lb), 2.0])

        bounds = [(None, None), (0.0, 50.0)]

    elif kind == "exp":
        # y = a + b*x
        def pred(theta):
            a, b = theta
            return a + b * x

        if np.any(~censored):
            X = np.vstack([np.ones(np.sum(~censored)), x[~censored]]).T
            theta0, *_ = np.linalg.lstsq(X, y[~censored], rcond=None)
            theta0 = np.array(theta0)
        else:
            theta0 = np.array([np.median(lb), 5.0])

        bounds = [(None, None), (0.0, 200.0)]

    else:
        raise ValueError("Unknown model kind")

    hinge_scale = 0.2  # in log-space

    def loss(theta):
        p = pred(theta)
        # uncensored SSE
        r = y[~censored] - p[~censored]
        sse = np.sum(w[~censored] * r * r) if r.size else 0.0
        # censored: penalize only if prediction violates lower bound
        if np.any(censored):
            v = (lb[censored] - p[censored]) / hinge_scale
            pen = _softplus(v)  # ~max(0, lb-p)
            sse += np.sum(w[censored] * pen * pen)
        return float(sse)

    res = minimize(loss, theta0, method="L-BFGS-B", bounds=bounds)
    theta_hat = res.x
    sse = max(loss(theta_hat), 1e-12)
    return theta_hat, sse


def objective(experiment_results):
    """Estimate the research goal from experiment results.

    Returns an AIC-weighted, censor-aware extrapolation of log10(restart-steps) at N=2000,
    with smooth penalties for limited scale-coverage and super-polynomial power slope.
    """
    if not experiment_results:
        return 1000.0

    # Sort by N and extract
    ers = sorted(experiment_results, key=lambda d: d.get("N", 0))
    N = np.array([float(d.get("N", np.nan)) for d in ers], dtype=float)
    med = np.array([float(d.get("median_step", np.nan)) for d in ers], dtype=float)
    max_steps = np.array([float(d.get("max_steps", np.nan)) for d in ers], dtype=float)
    uf = np.array([float(d.get("unsolved_fraction", 1.0)) for d in ers], dtype=float)

    ok = np.isfinite(N) & np.isfinite(med) & np.isfinite(max_steps) & (N > 0) & (med > 0) & (max_steps > 0)
    if np.sum(ok) < 2:
        return 500.0

    N = N[ok]
    med = med[ok]
    max_steps = max_steps[ok]
    uf = np.clip(uf[ok], 0.0, 1.0)

    # Success indicator: schedule uses unsolved_fraction < 0.5 to continue
    censored = uf >= 0.5

    # Restart-adjusted steps (approx expected steps until a solve via restarts)
    p_solve = np.clip(1.0 - uf, 1e-3, 1.0)
    eff_steps = np.where(censored, max_steps, med) / p_solve

    # Work in log-space
    y = np.log(np.clip(eff_steps, 1e-12, None))
    lb = np.log(np.clip((max_steps / p_solve), 1e-12, None))  # lower bound when censored

    # Emphasize larger N smoothly
    w = (N / np.max(N)) ** 0.5

    N_target = 2000.0

    # Model 1: power law in N => y = a + b*log(N)
    x_pow = np.log(N)
    x_pow_t = np.log(N_target)
    theta_pow, sse_pow = _fit_censored_model(x_pow, y, censored, lb, w, kind="power")
    yhat_pow_t = theta_pow[0] + theta_pow[1] * x_pow_t
    pred_pow = float(np.exp(yhat_pow_t))

    # Model 2: exponential in N => y = a + b*(N/1000)
    x_exp = N / 1000.0
    x_exp_t = N_target / 1000.0
    theta_exp, sse_exp = _fit_censored_model(x_exp, y, censored, lb, w, kind="exp")
    yhat_exp_t = theta_exp[0] + theta_exp[1] * x_exp_t
    pred_exp = float(np.exp(yhat_exp_t))

    # AIC-style weights (treating sse as proxy for -loglik)
    n = len(N)
    def aic_like(sse, k):
        sigma2 = max(sse / max(n, 1), 1e-12)
        return 2.0 * k + n * np.log(sigma2)

    aic_pow = aic_like(sse_pow, k=2)
    aic_exp = aic_like(sse_exp, k=2)
    aics = np.array([aic_pow, aic_exp], dtype=float)
    da = aics - np.min(aics)
    weights = np.exp(-0.5 * da)
    weights = weights / np.sum(weights)

    pred = weights[0] * pred_pow + weights[1] * pred_exp
    base = np.log10(np.clip(pred, 1e-12, None))

    # Smooth coverage penalty: prefer reaching larger successful N
    success = ~censored
    if np.any(success):
        N_best = float(np.max(N[success]))
    else:
        N_best = float(np.min(N))
    coverage = np.log1p(N_best) / np.log1p(N_target)
    pen_cov = 2.0 * (1.0 - coverage) ** 2

    # Mild penalty for steep power exponent (encourages polynomial-ish scaling)
    b_pow = float(theta_pow[1])
    pen_slope = 0.10 * max(0.0, b_pow - 3.0) ** 2

    return float(base + pen_cov + pen_slope)
