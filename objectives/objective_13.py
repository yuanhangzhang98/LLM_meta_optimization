import numpy as np
from scipy.stats import theilslopes, linregress


def _softplus(z):
    # stable softplus
    z = np.asarray(z, dtype=float)
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)


def objective(experiment_results):
    """Estimate the research goal from experiment results.

    Proxy: model scaling of an expected-restart compute proxy
      steps_eff = (0.7*median_step + 0.3*max_steps) / p**k,
    where p = solved_fraction.

    Return: predicted log10(steps_eff) at N=2000 plus smooth penalties for
    (i) not reaching large N and (ii) too-steep power-law exponent.
    """

    if not experiment_results:
        return 1e3

    # Collect and sanitize
    rows = []
    for r in experiment_results:
        if 'N' not in r or 'median_step' not in r:
            continue
        N = float(r['N'])
        if N <= 0:
            continue
        med = float(max(1.0, r.get('median_step', 1.0)))
        max_steps = float(max(med, r.get('max_steps', med)))
        u = float(r.get('unsolved_fraction', 1.0))
        u = float(np.clip(u, 0.0, 1.0))
        p = float(np.clip(1.0 - u, 1e-3, 1.0 - 1e-6))

        k = 1.5  # reliability amplification
        steps_eff = (0.7 * med + 0.3 * max_steps) / (p ** k)
        rows.append((N, steps_eff, p, max_steps))

    if len(rows) == 0:
        return 1e3

    rows.sort(key=lambda t: t[0])
    Ns = np.array([t[0] for t in rows], dtype=float)
    steps_eff = np.array([t[1] for t in rows], dtype=float)

    N_target = 2000.0
    x = np.log(Ns)
    y = np.log(steps_eff)

    # Robust power-law fit: log(steps_eff) = a + b*log(N)
    if len(rows) >= 2 and np.ptp(x) > 1e-12:
        try:
            b, a, *_ = theilslopes(y, x)  # y ~ a + b x
        except Exception:
            b, a = np.polyfit(x, y, 1)
    else:
        # single-point fallback: assume mild growth
        a = y[-1]
        b = 1.0

    y_pred = a + b * np.log(N_target)
    pred_log10 = float(y_pred / np.log(10.0))

    # Penalty 1: insufficient N coverage (smooth, 0 if N_max >= N_target)
    N_max = float(np.max(Ns))
    z_cov = np.log(N_target / max(N_max, 1.0))
    cov_pen = float(_softplus(5.0 * z_cov) / 5.0)

    # Penalty 2: discourage very steep polynomial exponent (allow up to quadratic)
    slope_thr = 2.0
    slope_pen = float(_softplus(5.0 * (b - slope_thr)) / 5.0)

    # Penalty 3: mild evidence-of-exponential-growth penalty
    exp_pen = 0.0
    if len(rows) >= 3 and np.ptp(Ns) > 1e-12:
        try:
            fit_exp = linregress(Ns, y)      # y ~ a + b*N
            fit_pow = linregress(x, y)       # y ~ a + b*logN
            r2_exp = float(fit_exp.rvalue ** 2)
            r2_pow = float(fit_pow.rvalue ** 2)
            # only penalize when exp fit is clearly better and growth slope positive
            exp_adv = max(0.0, r2_exp - r2_pow - 0.05)
            exp_growth = max(0.0, fit_exp.slope)
            exp_pen = float(exp_adv * (exp_growth * (N_target / 2000.0)))
        except Exception:
            exp_pen = 0.0

    # Combine (lower is better)
    return pred_log10 + 0.30 * cov_pen + 0.15 * slope_pen + 0.10 * exp_pen
