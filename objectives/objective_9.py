import math
import numpy as np
from scipy.stats import linregress


def _safe_log(x, eps=1e-12):
    return np.log(np.maximum(x, eps))


def _aicc_from_rss(rss, n, k):
    # Gaussian errors, constant-variance; compare models via AICc
    rss = float(max(rss, 1e-18))
    aic = n * math.log(rss / n) + 2 * k
    if n <= (k + 1):
        return float("inf")
    return aic + (2 * k * (k + 1)) / (n - k - 1)


def objective(experiment_results):
    """Estimate the research goal from experiment results.

    Returns a smooth proxy for large-N performance:
    - Converts each experiment (N, median_step, unsolved_fraction, max_steps)
      into an *expected steps to solve with geometric restarts* under that budget.
    - Fits scaling to log(expected_steps) using AICc-weighted model averaging
      (power-law vs exponential), then predicts at N=2000.
    - Adds smooth penalties for poor coverage (didn't reach large N) and
      implausible negative scaling.

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
        N = float(r.get("N", 0) or 0)
        if N <= 0:
            continue
        max_steps = float(r.get("max_steps", 0) or 0)
        if max_steps <= 0:
            max_steps = 1.0
        med = r.get("median_step", None)
        med = float(med) if (med is not None and float(med) > 0) else max_steps
        uns = r.get("unsolved_fraction", 1.0)
        uns = float(uns) if uns is not None else 1.0
        # Smooth clipping to avoid infinities
        p = 1.0 - np.clip(uns, 0.0, 1.0)  # solved fraction
        p_floor = 1e-3
        p_eff = max(float(p), p_floor)

        # Attempt cost: spend ~median on solved runs and max_steps on unsolved runs
        attempt_cost = p * med + (1.0 - p) * max_steps
        # Expected cost to solve with i.i.d. restarts
        exp_steps = attempt_cost / p_eff
        exp_steps = max(exp_steps, 1.0)

        rows.append((N, exp_steps, max_steps, p))

    if len(rows) < 2:
        # Not enough data: return a conservative, failure-aware score
        (_, exp_steps, _, p) = rows[0]
        return float(np.log10(exp_steps) + 1.0 * (1.0 - float(p)))

    Ns = np.array([t[0] for t in rows], dtype=float)
    y = _safe_log(np.array([t[1] for t in rows], dtype=float))  # log expected steps

    N_target = 2000.0

    # Model 1: power law -> y = a + b*log(N)
    x1 = _safe_log(Ns)
    fit1 = linregress(x1, y)
    yhat1 = fit1.intercept + fit1.slope * x1
    rss1 = float(np.sum((y - yhat1) ** 2))
    aicc1 = _aicc_from_rss(rss1, n=len(y), k=2)
    pred1 = float(math.exp(fit1.intercept + fit1.slope * math.log(N_target)))

    # Model 2: exponential -> y = a + b*N
    x2 = Ns
    fit2 = linregress(x2, y)
    yhat2 = fit2.intercept + fit2.slope * x2
    rss2 = float(np.sum((y - yhat2) ** 2))
    aicc2 = _aicc_from_rss(rss2, n=len(y), k=2)
    pred2 = float(math.exp(fit2.intercept + fit2.slope * N_target))

    # AICc weights (lower AICc is better)
    aiccs = np.array([aicc1, aicc2], dtype=float)
    if np.any(~np.isfinite(aiccs)):
        w = np.array([0.5, 0.5], dtype=float)
    else:
        d = aiccs - np.min(aiccs)
        w = np.exp(-0.5 * d)
        w = w / np.sum(w)

    pred = max(1.0, float(w[0] * pred1 + w[1] * pred2))

    # Coverage penalty: if schedule stopped early, be conservative
    maxN = float(np.max(Ns))
    coverage_pen = 0.35 * (max(0.0, math.log(N_target / maxN)) ** 2)

    # Negative-scaling penalty (implausible: faster as N grows)
    neg_pen = 0.0
    neg_pen += 0.25 * max(0.0, -float(fit1.slope))  # power-law slope < 0
    neg_pen += 0.00015 * max(0.0, -float(fit2.slope)) * N_target  # exp slope < 0

    objective_value = float(np.log10(pred) + coverage_pen + neg_pen)
    return objective_value
