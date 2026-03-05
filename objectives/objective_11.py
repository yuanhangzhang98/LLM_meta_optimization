import math
import numpy as np
from scipy.stats import theilslopes


def _softplus(x):
    x = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def _aicc_from_rss(n, k, rss):
    # AICc up to an additive constant (constants cancel in model weights)
    rss = float(max(rss, 1e-12))
    aic = n * math.log(rss / n) + 2.0 * k
    if n > (k + 1):
        aic += (2.0 * k * (k + 1)) / (n - k - 1)
    else:
        aic += 1e6  # heavy penalty when AICc undefined
    return aic


def _robust_line_fit(x, y):
    # Returns (intercept, slope) for y ~= intercept + slope * x
    # theilslopes returns slope, intercept
    slope, intercept, _, _ = theilslopes(y, x)
    return float(intercept), float(slope)


def _ridge_fit_intercept_slope(x, y, w=None, ridge=1e-2):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if w is None:
        w = np.ones_like(x)
    w = np.asarray(w, dtype=float)

    X = np.column_stack([np.ones_like(x), x])
    W = np.diag(w)
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ y
    reg = ridge * np.eye(2)
    beta = np.linalg.solve(XtWX + reg, XtWy)
    return float(beta[0]), float(beta[1])


def objective(experiment_results):
    """Estimate the research goal from experiment results.

    Proxy: model-average (AICc) extrapolation of restart-adjusted effort
    to N=2000, plus a smooth penalty encouraging a safety margin in
    success probability.

    Args:
        experiment_results: List of dicts with experiment details
    Returns:
        Float to be MINIMIZED
    """
    if not experiment_results:
        return 1000.0

    # Prepare per-N observations
    rows = []
    for r in experiment_results:
        N = r.get("N", None)
        ms = r.get("median_step", None)
        u = r.get("unsolved_fraction", None)
        max_steps = r.get("max_steps", None)
        if N is None or ms is None or u is None:
            continue
        N = float(N)
        ms = float(ms)
        u = float(u)
        max_steps = float(max_steps) if max_steps is not None else float(max(ms, 1.0))

        p = np.clip(1.0 - u, 1e-3, 1.0 - 1e-6)

        # Restart-adjusted typical work (proxy for expected steps-to-solution with restarts)
        effort = ms / p

        # Smooth inflation when the budget is tight (suggests censoring / heavy tail)
        tight = np.clip(ms / max(max_steps, 1.0), 0.0, 2.0)
        inflate = 1.0 + 0.6 * (tight ** 6)  # near 0 unless ms ~ max_steps

        rows.append((N, effort * inflate, p))

    if len(rows) == 0:
        return 1000.0

    # Aggregate duplicates by median in log-space (robust)
    byN = {}
    for N, eff, p in rows:
        byN.setdefault(N, {"eff": [], "p": []})
        byN[N]["eff"].append(eff)
        byN[N]["p"].append(p)

    Ns = np.array(sorted(byN.keys()), dtype=float)
    effs = np.array([np.exp(np.median(np.log(byN[N]["eff"]))) for N in Ns], dtype=float)
    ps = np.array([float(np.median(byN[N]["p"])) for N in Ns], dtype=float)

    n = len(Ns)
    N_target = 2000.0

    # If too few points, fall back to last observed effort with mild scaling penalty
    if n < 3:
        last = float(np.log10(effs[-1] + 1e-12))
        scale_pen = 0.15 * max(0.0, math.log(N_target / Ns[-1], 2.0))  # gentle
        p_last = float(ps[-1])
        margin_pen = float(_softplus((0.65 - p_last) / 0.05)) * 0.25
        return last + scale_pen + margin_pen

    y = np.log(effs + 1e-12)

    # Model 1: polynomial (log-log)
    x1 = np.log(Ns)
    a1, b1 = _robust_line_fit(x1, y)
    yhat1 = a1 + b1 * x1
    rss1 = float(np.sum((y - yhat1) ** 2))
    aicc1 = _aicc_from_rss(n=n, k=2, rss=rss1)
    ypred1 = a1 + b1 * math.log(N_target)

    # Model 2: exponential in N (log effort ~ a + b N)
    x2 = Ns
    a2, b2 = _robust_line_fit(x2, y)
    yhat2 = a2 + b2 * x2
    rss2 = float(np.sum((y - yhat2) ** 2))
    aicc2 = _aicc_from_rss(n=n, k=2, rss=rss2)
    ypred2 = a2 + b2 * N_target

    # AICc weights
    aiccs = np.array([aicc1, aicc2], dtype=float)
    da = aiccs - np.min(aiccs)
    w = np.exp(-0.5 * da)
    w = w / np.sum(w)

    ypred = float(w[0] * ypred1 + w[1] * ypred2)
    pred_log10_effort = ypred / math.log(10.0)

    # Reliability margin: fit logit(p) ~ A + B log N (weakly regularized)
    logit_p = np.log(np.clip(ps, 1e-3, 1 - 1e-3) / np.clip(1 - ps, 1e-3, 1.0))
    xr = np.log(Ns)
    wr = (Ns / Ns.max()) ** 2  # emphasize largest N observed
    A, B = _ridge_fit_intercept_slope(xr, logit_p, w=wr, ridge=1e-2)
    p_target = float(_sigmoid(A + B * math.log(N_target)))

    # Encourage p_target >= 0.65 smoothly (avoid hard cliffs)
    margin_pen = float(_softplus((0.65 - p_target) / 0.05)) * 0.35

    # Slightly discourage being close to the schedule cutoff at any tested N
    cutoff_pen = float(np.mean(_softplus((0.50 - ps) / 0.02))) * 0.10

    return float(pred_log10_effort + margin_pen + cutoff_pen)
