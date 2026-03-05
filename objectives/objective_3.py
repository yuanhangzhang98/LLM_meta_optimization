import math
import numpy as np


def _softplus(x: float) -> float:
    # stable softplus
    if x > 40:
        return x
    if x < -40:
        return math.exp(x)
    return math.log1p(math.exp(x))


def _wls_fit(x, y, w):
    """Weighted least squares for y = a + b x. Returns (a, b, rss)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    X = np.stack([np.ones_like(x), x], axis=1)
    W = w
    XtW = X.T * W
    A = XtW @ X
    bvec = XtW @ y

    # small ridge for numerical stability
    A = A + 1e-12 * np.eye(2)
    coef = np.linalg.solve(A, bvec)
    yhat = X @ coef
    rss = float(np.sum(W * (y - yhat) ** 2))
    return float(coef[0]), float(coef[1]), rss


def _aic_from_rss(rss: float, n: int, k: int = 2) -> float:
    eps = 1e-12
    return 2.0 * k + n * math.log(max(rss / max(n, 1), eps))


def objective(experiment_results):
    """Estimate the research goal from experiment results.

    Args:
        experiment_results: List of dicts with experiment details
    Returns:
        Float to be MINIMIZED
    """
    if not experiment_results:
        return 1e3

    # Deduplicate by N by keeping the latest entry for each N (schedule can revisit N)
    byN = {}
    for r in experiment_results:
        if 'N' in r:
            byN[int(r['N'])] = r
    results = [byN[n] for n in sorted(byN.keys())]

    if len(results) < 2:
        return 5e2

    N_target = 2000.0

    Ns = np.array([float(r['N']) for r in results], dtype=float)
    steps = np.array([max(1.0, float(r.get('median_step', 1.0))) for r in results], dtype=float)
    max_steps = np.array([max(1.0, float(r.get('max_steps', s))) for r, s in zip(results, steps)], dtype=float)
    u = np.array([float(r.get('unsolved_fraction', 1.0)) for r in results], dtype=float)

    # Treat u as mainly a failure indicator (run_till_median often pins u ~ 0.49 on success).
    # Smooth failure severity increases when u crosses 0.5 or when step hits the step budget.
    over_u = np.array([_softplus((ui - 0.5) / 0.02) for ui in u], dtype=float)
    budget_ratio = np.clip(steps / max_steps, 0.0, 2.0)
    near_budget = np.array([_softplus((br - 0.98) / 0.02) for br in budget_ratio], dtype=float)

    # Effective cost: base steps plus smooth penalties for being censored / near budget
    eff_steps = steps * (1.0 + 0.5 * near_budget) + max_steps * (2.0 * over_u + 0.5 * near_budget)
    eff_steps = np.maximum(eff_steps, 1.0)

    # Weights emphasize larger N (closer to the scaling regime)
    w = (Ns / np.max(Ns)) ** 1.5
    w = np.clip(w, 1e-6, None)

    y = np.log(eff_steps)

    # Model 1: polynomial scaling (power law): log(steps) = a + b*log(N)
    x_poly = np.log(Ns)
    a_poly, b_poly, rss_poly = _wls_fit(x_poly, y, w)
    pred_poly = a_poly + b_poly * math.log(N_target)

    # Model 2: exponential scaling: log(steps) = a + b*N
    x_exp = Ns
    a_exp, b_exp, rss_exp = _wls_fit(x_exp, y, w)
    pred_exp = a_exp + b_exp * N_target

    # AIC mixing (prefers model with better explanatory power, but stays smooth)
    n = len(results)
    aic_poly = _aic_from_rss(rss_poly, n, k=2)
    aic_exp = _aic_from_rss(rss_exp, n, k=2)
    # weights ~ exp(-0.5*AIC)
    m = min(aic_poly, aic_exp)
    wp = math.exp(-0.5 * (aic_poly - m))
    we = math.exp(-0.5 * (aic_exp - m))
    s = wp + we
    wp, we = wp / s, we / s

    # Mix in linear step space
    steps_pred = wp * math.exp(pred_poly) + we * math.exp(pred_exp)
    base_obj = math.log10(max(1.0, steps_pred))

    # Penalties: encourage broad scaling evidence and strongly discourage exponential fits
    maxN = float(np.max(Ns))
    coverage = min(1.0, math.log(maxN + 1.0) / math.log(N_target + 1.0))
    penalty_cov = 1.0 * (1.0 - coverage)

    # If exponential model is favored and has positive rate, penalize (proxy for super-polynomial)
    penalty_exp = we * _softplus(b_exp) * (N_target / 500.0)

    # Penalize steep polynomial slopes gently (prefer near-linear-ish growth)
    penalty_slope = 0.25 * _softplus(b_poly - 1.5)

    # Penalize if the largest-N experiment looks failed/censored
    last_over_u = float(over_u[-1])
    last_near_budget = float(near_budget[-1])
    penalty_last = 1.0 * last_over_u + 0.5 * last_near_budget

    objective_value = base_obj + penalty_cov + penalty_exp + penalty_slope + penalty_last
    return float(objective_value)
