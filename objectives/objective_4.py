import math
import numpy as np
from scipy.stats import linregress


def _softplus(x):
    # numerically stable softplus
    x = float(x)
    if x > 40:
        return x
    if x < -40:
        return math.exp(x)
    return math.log1p(math.exp(x))


def _safe_log(x, eps=1e-12):
    return math.log(max(float(x), eps))


def _aicc_from_rss(rss, n, k):
    rss = float(max(rss, 1e-12))
    n = int(n)
    k = int(k)
    aic = n * math.log(rss / n) + 2 * k
    if n <= (k + 1):
        return float("inf")
    return aic + (2 * k * (k + 1)) / (n - k - 1)


def objective(experiment_results):
    """Estimate the research goal from experiment results.

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
        N = float(r.get("N", np.nan))
        med = float(r.get("median_step", np.nan))
        u = float(r.get("unsolved_fraction", 1.0))
        ms = float(r.get("max_steps", np.nan))
        if not (np.isfinite(N) and np.isfinite(med)):
            continue
        rows.append((N, max(med, 1.0), min(max(u, 0.0), 1.0), ms if np.isfinite(ms) else np.nan))

    if len(rows) < 2:
        return 1e5

    rows.sort(key=lambda t: t[0])
    Ns = np.array([t[0] for t in rows], dtype=float)
    meds = np.array([t[1] for t in rows], dtype=float)
    us = np.array([t[2] for t in rows], dtype=float)
    max_steps = np.array([t[3] for t in rows], dtype=float)

    N_target = 2000.0

    # Base scaling target: predict steps@2000 via AICc model averaging (power-law vs exponential)
    y = np.log(np.maximum(meds, 1.0))

    # Model 1: log(med) = a + b*log(N)
    x1 = np.log(np.maximum(Ns, 2.0))
    fit1 = linregress(x1, y)
    yhat1 = fit1.intercept + fit1.slope * x1
    rss1 = float(np.sum((y - yhat1) ** 2))
    aicc1 = _aicc_from_rss(rss1, n=len(y), k=2)
    pred1 = math.exp(fit1.intercept + fit1.slope * math.log(N_target))

    # Model 2: log(med) = a + b*N (=> exponential in N)
    x2 = Ns
    fit2 = linregress(x2, y)
    yhat2 = fit2.intercept + fit2.slope * x2
    rss2 = float(np.sum((y - yhat2) ** 2))
    aicc2 = _aicc_from_rss(rss2, n=len(y), k=2)
    pred2 = math.exp(fit2.intercept + fit2.slope * N_target)

    # AICc weights
    aiccs = np.array([aicc1, aicc2], dtype=float)
    m = float(np.min(aiccs))
    w = np.exp(-0.5 * (aiccs - m))
    if not np.isfinite(w).all() or float(np.sum(w)) <= 0:
        w = np.array([0.5, 0.5], dtype=float)
    else:
        w = w / np.sum(w)

    pred = w[0] * pred1 + w[1] * pred2
    pred = float(np.clip(pred, 1.0, 1e18))
    base = math.log10(pred)

    # Smooth reliability penalty at the largest N tested (discourage hovering near 0.5)
    idx_max = int(np.argmax(Ns))
    u_max = float(us[idx_max])

    u_good = 0.15
    u_scale = 0.05
    rel_weight = 2.0
    rel_pen = rel_weight * _softplus((u_max - u_good) / u_scale)

    # Smooth "censoring" penalty when median approaches step cap at largest N
    cap_pen = 0.0
    ms = float(max_steps[idx_max])
    if np.isfinite(ms) and ms > 0:
        cap_ratio = float(meds[idx_max] / ms)
        cap_good = 0.80
        cap_scale = 0.05
        cap_weight = 1.0
        cap_pen = cap_weight * _softplus((cap_ratio - cap_good) / cap_scale)

    # Encourage polynomial-like scaling by penalizing large power exponent (but softly)
    slope_pen = 0.0
    b = float(fit1.slope)  # exponent in power-law fit
    b_good = 2.0
    b_scale = 0.5
    b_weight = 0.4
    slope_pen = b_weight * _softplus((b - b_good) / b_scale)

    # Mild extrapolation penalty if we never reached moderately large N
    N_max = float(np.max(Ns))
    # ~0 when N_max>=640, rises smoothly when N_max<640
    extrap_weight = 0.6
    extrap_pen = extrap_weight * _softplus((math.log(640.0) - _safe_log(N_max)) / 0.5)

    return float(base + rel_pen + cap_pen + slope_pen + extrap_pen)
