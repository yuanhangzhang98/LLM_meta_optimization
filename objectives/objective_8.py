import math
import numpy as np


def _softplus(x, k=20.0):
    # smooth approx of max(0, x)
    return np.log1p(np.exp(k * x)) / k


def objective(experiment_results):
    """Estimate the research goal from experiment results.

    Proxy: predict log10 expected steps at N=2000 using a reliability-adjusted
    runtime and a weighted log-log scaling fit. Adds smooth penalties for
    early stop (not reaching large N) and for failing experiments.

    Args:
        experiment_results: List of dicts with experiment details
    Returns:
        Float to be MINIMIZED
    """
    if not experiment_results:
        return 1000.0

    # Extract
    Ns, ys, ws, uns = [], [], [], []
    eta = 1.0  # how strongly to inflate steps by failure probability (retry cost)

    for r in experiment_results:
        N = float(r.get("N", 0.0))
        if N <= 0:
            continue

        u = float(r.get("unsolved_fraction", 1.0))
        u = float(np.clip(u, 0.0, 1.0))
        p = float(np.clip(1.0 - u, 1e-3, 1.0))

        s = float(r.get("median_step", np.nan))
        if not np.isfinite(s) or s <= 0:
            continue

        budget = float(r.get("max_steps", s))
        budget = max(budget, s)

        # Reliability-adjusted expected runtime (roughly: expected steps with restarts)
        t_eff = s / (p ** eta)

        # If we're near step cap or the run is failing, treat as right-censored lower bound
        near_cap = (s >= 0.95 * budget) or (u >= 0.5)
        if near_cap:
            t_eff = max(t_eff, budget / (p ** eta))

        y = math.log10(max(t_eff, 1.0))

        # Weight larger N more; also downweight unreliable points
        w = (math.log10(N) + 1.0) ** 2 * (p ** 0.5)

        Ns.append(N)
        ys.append(y)
        ws.append(w)
        uns.append(u)

    if len(Ns) == 0:
        return 1000.0

    Ns = np.asarray(Ns, dtype=float)
    ys = np.asarray(ys, dtype=float)
    ws = np.asarray(ws, dtype=float)
    uns = np.asarray(uns, dtype=float)

    N_target = 2000.0
    x = np.log10(Ns)
    x_t = math.log10(N_target)

    # Weighted log-log fit: y = a + b*x
    if len(Ns) >= 2 and np.sum(ws) > 0:
        w = ws / (np.sum(ws) + 1e-12)
        x_bar = np.sum(w * x)
        y_bar = np.sum(w * ys)
        cov = np.sum(w * (x - x_bar) * (ys - y_bar))
        var = np.sum(w * (x - x_bar) ** 2)
        b = cov / (var + 1e-12)
        a = y_bar - b * x_bar
        y_pred = a + b * x_t
    else:
        # Single point fallback
        b = 0.0
        y_pred = float(ys[-1])

    # Smooth penalty for early stop / not reaching large N
    N_max = float(np.max(Ns))
    cov_pen = 0.6 * (math.log10(max(N_target / max(N_max, 1.0), 1.0))) ** 2

    # Smooth penalty for failures anywhere (focus on >0.5 unsolved)
    fail_pen = 3.0 * float(np.mean(_softplus(uns - 0.5, k=25.0)))

    # Mild penalty for very large scaling exponent (still polynomial but discourages huge k)
    exp_pen = 0.08 * float(max(0.0, b - 2.5) ** 2)

    return float(y_pred + cov_pen + fail_pen + exp_pen)
