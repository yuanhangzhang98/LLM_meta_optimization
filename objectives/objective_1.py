import numpy as np
from scipy.stats import theilslopes, linregress


def _sigmoid(x):
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _softplus(x):
    # smooth ReLU
    return np.logaddexp(0.0, x)


def objective(experiment_results):
    """Estimate the research goal from experiment results.

    Args:
        experiment_results: List of dicts with experiment details
    Returns:
        Float to be MINIMIZED
    """
    if not experiment_results:
        return 1e3

    rows = []
    for r in experiment_results:
        if 'N' not in r or 'median_step' not in r:
            continue
        N = float(r['N'])
        med = float(r['median_step'])
        uns = float(r.get('unsolved_fraction', 1.0))
        ms = float(r.get('max_steps', max(1.0, med)))
        if not np.isfinite(N) or not np.isfinite(med) or N <= 0 or med <= 0:
            continue
        rows.append((N, med, uns, ms))

    if len(rows) < 2:
        return 5e2

    rows.sort(key=lambda t: t[0])
    Ns = np.array([t[0] for t in rows], dtype=float)
    med = np.array([t[1] for t in rows], dtype=float)
    uns = np.clip(np.array([t[2] for t in rows], dtype=float), 0.0, 1.0)
    max_steps = np.maximum(1.0, np.array([t[3] for t in rows], dtype=float))

    # Expected steps proxy: steps inflated by failure probability and censoring near max_steps.
    p_solve = np.clip(1.0 - uns, 1e-3, 1.0)
    eff_steps = med / p_solve

    # Smooth penalty when median hits the step cap (censoring)
    censor = _sigmoid((med - 0.90 * max_steps) / (0.05 * max_steps + 1.0))
    eff_steps = eff_steps * (1.0 + 5.0 * censor)

    # Robust power-law fit: log(eff_steps) = a + b*log(N)
    x = np.log(Ns)
    y = np.log(np.maximum(eff_steps, 1.0))

    try:
        slope, intercept, _, _ = theilslopes(y, x)
    except Exception:
        fit = linregress(x, y)
        slope, intercept = float(fit.slope), float(fit.intercept)

    N_target = 2000.0
    log_pred = intercept + slope * np.log(N_target)
    pred = float(np.exp(np.clip(log_pred, -50.0, 50.0)))

    # Prefer polynomial-like scaling: penalize large exponents smoothly
    b_thresh = 2.0
    scale_pen = 1.0 * _softplus((slope - b_thresh) / 0.30)

    # Prefer robust solving well below the schedule cutoff (0.5)
    w = (Ns / Ns.max()) ** 2
    reliab_pen = 2.0 * float(np.average(_sigmoid((uns - 0.45) / 0.03), weights=w))

    # If the run stopped on a hard failure (unsolved >= 0.5 at max N), penalize not reaching N_target
    idx_max = int(np.argmax(Ns))
    stopped_hard = _sigmoid((uns[idx_max] - 0.50) / 0.02)
    reach_pen = 2.0 * stopped_hard * np.log10(1.0 + (N_target / Ns[idx_max]))

    objective_value = np.log10(max(pred, 1.0)) + scale_pen + reliab_pen + reach_pen
    return float(objective_value)
