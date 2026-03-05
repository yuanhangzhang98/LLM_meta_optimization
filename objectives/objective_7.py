import numpy as np
from scipy.special import logsumexp


def _wls_fit(x, y, w):
    """Weighted least squares for y ~ b0 + b1*x."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    X = np.stack([np.ones_like(x), x], axis=1)
    sw = np.sqrt(np.clip(w, 1e-12, None))
    Xw = X * sw[:, None]
    yw = y * sw
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    rss = float(np.sum(w * resid * resid))
    return beta, rss


def _aic_gaussian(rss, n, k):
    rss = max(rss, 1e-12)
    n = max(int(n), 1)
    return n * np.log(rss / n) + 2 * k


def _softmax_max(values, tau=0.05):
    """Smooth max in log10-space; tau in log10 units."""
    v = np.asarray(values, dtype=float)
    tau = float(max(tau, 1e-6))
    return tau * (logsumexp(v / tau))


def objective(experiment_results):
    """Estimate the research goal from experiment results.

    Args:
        experiment_results: List of dicts with experiment details
    Returns:
        Float to be MINIMIZED
    """
    if not experiment_results:
        return 1000.0

    # Extract / validate
    rows = []
    for r in experiment_results:
        try:
            N = float(r["N"])
            med = float(r["median_step"])
            ms = float(r.get("max_steps", np.nan))
            u = float(r.get("unsolved_fraction", 1.0))
        except Exception:
            continue
        if not (np.isfinite(N) and np.isfinite(med) and N > 0 and med > 0):
            continue
        if not np.isfinite(ms) or ms <= 0:
            ms = max(med, 1.0)
        rows.append((N, med, ms, u))

    if not rows:
        return 1000.0

    Ns = np.array([t[0] for t in rows], dtype=float)
    med = np.array([t[1] for t in rows], dtype=float)
    max_steps = np.array([t[2] for t in rows], dtype=float)
    uns = np.clip(np.array([t[3] for t in rows], dtype=float), 0.0, 1.0)

    N_target = 2000.0

    # Censor-aware effective step:
    # - If unsolved_fraction < 0.5, treat median_step as meaningful.
    # - If unsolved_fraction >= 0.5, treat as time-censored; impute > max_steps smoothly.
    fail = uns >= 0.5
    k_fail = 9.0  # up to 10x max_steps when unsolved_fraction=1
    step_eff = med.copy()
    step_eff[fail] = max_steps[fail] * (1.0 + k_fail * (uns[fail] - 0.5) / 0.5)
    step_eff = np.clip(step_eff, 1.0, None)

    # If only one point, use a conservative polynomial extrapolation.
    if len(step_eff) < 2:
        N0 = float(Ns[0])
        pred = float(step_eff[0]) * (N_target / max(N0, 1.0)) ** 2
        return float(np.log10(max(pred, 1.0)))

    # Weighted regression emphasizing larger N
    N_max = float(np.max(Ns))
    w = np.sqrt(np.clip(Ns / max(N_max, 1.0), 1e-6, 1.0))

    y = np.log(step_eff)

    # Model 1: power law  step ~ N^b  => log(step) ~ a + b*log(N)
    x_poly = np.log(Ns)
    beta_poly, rss_poly = _wls_fit(x_poly, y, w)
    aic_poly = _aic_gaussian(rss_poly, n=len(y), k=2)
    ypred_poly = beta_poly[0] + beta_poly[1] * np.log(N_target)
    pred_poly = float(np.exp(ypred_poly))

    # Model 2: exponential  step ~ exp(b*N) => log(step) ~ a + b*(N/N_target)
    x_exp = Ns / N_target
    beta_exp, rss_exp = _wls_fit(x_exp, y, w)
    aic_exp = _aic_gaussian(rss_exp, n=len(y), k=2)
    ypred_exp = beta_exp[0] + beta_exp[1] * 1.0
    pred_exp = float(np.exp(ypred_exp))

    # AIC model averaging
    aics = np.array([aic_poly, aic_exp], dtype=float)
    daic = aics - np.min(aics)
    weights = np.exp(-0.5 * daic)
    weights = weights / np.sum(weights)
    pred = weights[0] * pred_poly + weights[1] * pred_exp
    pred_log10 = float(np.log10(max(pred, 1.0)))

    # If we ever failed at some N, impose a smooth lower bound on scaling beyond that point
    # (cannot be faster than ">= max_steps at N_fail" with at least linear growth).
    if np.any(fail):
        min_exp = 1.0
        lb_logs = []
        for Ni, msi in zip(Ns[fail], max_steps[fail]):
            Ni = float(max(Ni, 1.0))
            msi = float(max(msi, 1.0))
            lb_logs.append(np.log10(msi) + min_exp * np.log10(N_target / Ni))
        lb_log = float(_softmax_max(lb_logs, tau=0.05))
        pred_log10 = float(_softmax_max([pred_log10, lb_log], tau=0.05))

    # Mild penalty for not reaching large N (encourages designs that scale up the schedule)
    gap_doublings = max(0.0, np.log(max(N_target, 1.0) / max(N_max, 1.0)) / np.log(2.0))
    coverage_penalty = 0.15 * gap_doublings

    return float(pred_log10 + coverage_penalty)
