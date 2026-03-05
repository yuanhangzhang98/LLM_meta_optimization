import math
import numpy as np
from scipy.special import logsumexp


def _aicc(rss, n, k):
    """AICc for Gaussian residuals; lower is better."""
    rss = float(max(rss, 1e-12))
    n = int(n)
    k = int(k)
    aic = n * math.log(rss / n) + 2 * k
    # small-sample correction
    if n > (k + 1):
        aic += (2 * k * (k + 1)) / (n - k - 1)
    else:
        aic += 1e6
    return aic


def _linfit(x, y):
    """Fit y = a + b x; returns (a,b,rss)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.stack([np.ones_like(x), x], axis=1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    rss = float(np.sum((y - yhat) ** 2))
    return float(coef[0]), float(coef[1]), rss


def objective(experiment_results):
    """Estimate the research goal from experiment results.

    Returns a smooth proxy for log10 expected steps at N=2000, adjusted for
    reliability (unsolved_fraction) and penalizing early failure to scale.

    Args:
        experiment_results: List of dicts with experiment details
    Returns:
        Float to be MINIMIZED
    """
    if not experiment_results:
        return 1e3

    # Sort by N and extract
    results = sorted(experiment_results, key=lambda d: d.get('N', 0))

    Ns = np.array([max(2, int(r.get('N', 2))) for r in results], dtype=float)
    med = np.array([max(1.0, float(r.get('median_step', 1.0))) for r in results], dtype=float)
    u = np.array([float(r.get('unsolved_fraction', 1.0)) for r in results], dtype=float)
    u = np.clip(u, 0.0, 0.999)

    max_steps = np.array([float(r.get('max_steps', max(1.0, m))) for r, m in zip(results, med)], dtype=float)
    max_steps = np.maximum(max_steps, 1.0)

    # Reliability-adjusted "expected steps" via geometric restarts: cost ~ steps / (1-u)
    # Also censor by max_steps if median_step is weirdly above budget.
    eps = 1e-6
    med_c = np.minimum(med, max_steps)
    eff = med_c / np.maximum(1.0 - u, eps)
    y = np.log(eff)

    N_target = 2000.0

    # If only 1 point, do conservative polynomial extrapolation with exponent 3.
    if len(results) == 1:
        p = 3.0
        pred_log = float(y[0] + p * (math.log(N_target) - math.log(Ns[0])))
        # Coverage penalty for not demonstrating scaling
        N_max = float(Ns[0])
        coverage_pen = 0.6 * max(0.0, math.log(N_target / N_max))
        return (pred_log / math.log(10.0)) + coverage_pen

    # Model 1: polynomial scaling in log-log
    x_poly = np.log(Ns)
    a1, b1, rss1 = _linfit(x_poly, y)
    pred1 = a1 + b1 * math.log(N_target)

    # Model 2: exponential scaling (log steps linear in N)
    x_exp = Ns
    a2, b2, rss2 = _linfit(x_exp, y)
    pred2 = a2 + b2 * N_target

    # AICc weights (more stable than R^2 under tiny n)
    n = len(results)
    aic1 = _aicc(rss1, n, k=2)
    aic2 = _aicc(rss2, n, k=2)
    logw = -0.5 * np.array([aic1, aic2], dtype=float)
    logw = logw - float(np.max(logw))
    w = np.exp(logw)
    w = w / float(np.sum(w))

    # Mix in log-space (geometric mixture) for stability
    pred_mix = float(logsumexp(np.log(w) + np.array([pred1, pred2], dtype=float)))

    # Local-slope extrapolation (guards against optimistic global fits)
    i0, i1 = -2, -1
    slope_local = (y[i1] - y[i0]) / max(eps, (math.log(Ns[i1]) - math.log(Ns[i0])))
    pred_local = float(y[i1] + slope_local * (math.log(N_target) - math.log(Ns[i1])))

    pred_log = max(pred_mix, pred_local)

    # Penalize lack of demonstrated scaling (early stop at small N looks deceptively good)
    N_max = float(np.max(Ns))
    coverage_pen = 0.6 * max(0.0, math.log(N_target / N_max))

    # Mild penalty for super-high fitted polynomial degree (pushes toward polynomial-time)
    poly_degree_pen = 0.15 * max(0.0, b1 - 4.0) ** 2

    return (pred_log / math.log(10.0)) + coverage_pen + poly_degree_pen
