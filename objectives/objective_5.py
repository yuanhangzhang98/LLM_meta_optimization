import math
import numpy as np


def _softplus(x):
    # numerically stable softplus
    if x > 50:
        return x
    if x < -50:
        return math.exp(x)
    return math.log1p(math.exp(x))


def objective(experiment_results):
    """Estimate the research goal from experiment results.

    Args:
        experiment_results: List of dicts with experiment details
    Returns:
        Float to be MINIMIZED
    """
    if not experiment_results:
        return 1e3

    # Keep valid rows and (lightly) dedupe by N keeping the last occurrence
    byN = {}
    for r in experiment_results:
        if not isinstance(r, dict):
            continue
        if 'N' not in r or 'median_step' not in r or 'unsolved_fraction' not in r:
            continue
        byN[int(r['N'])] = r

    if len(byN) < 2:
        return 5e2

    rows = [byN[k] for k in sorted(byN.keys())]

    N_target = 2000.0

    Ns = np.array([max(1, int(r['N'])) for r in rows], dtype=float)
    u = np.array([float(r.get('unsolved_fraction', 1.0)) for r in rows], dtype=float)
    steps = np.array([max(1.0, float(r.get('median_step', 1.0))) for r in rows], dtype=float)
    caps = np.array([max(1.0, float(r.get('max_steps', steps[i]))) for i, r in enumerate(rows)], dtype=float)

    u = np.clip(u, 0.0, 1.0)
    solved = np.clip(1.0 - u, 1e-3, 1.0)

    # Reliability-adjusted cost (smoothly penalizes marginal solved rates)
    # Using 1/solved (not hard thresholded) strongly discourages u≈0.5 behavior.
    eff = steps / solved

    # Cap-pressure penalty: if median is close to cap, likely many timeouts / heavy tails
    cap_ratio = np.clip(steps / caps, 0.0, 2.0)
    cap_pressure = np.array([_softplus((cr - 0.7) / 0.08) for cr in cap_ratio], dtype=float)
    eff = eff * (1.0 + 0.15 * cap_pressure)

    # Fit log-log scaling: log10(eff) = a + b*log10(N)
    x = np.log10(Ns)
    y = np.log10(np.maximum(eff, 1e-9))

    # Weight larger N more (where asymptotics matter)
    w = Ns / (Ns.max() + 1e-9)
    w = np.clip(w, 0.1, 1.0)

    xbar = np.sum(w * x) / np.sum(w)
    ybar = np.sum(w * y) / np.sum(w)
    denom = np.sum(w * (x - xbar) ** 2)
    b = float(np.sum(w * (x - xbar) * (y - ybar)) / (denom + 1e-12))
    a = float(ybar - b * xbar)

    y_pred = a + b * math.log10(N_target)
    pred_eff = 10 ** y_pred

    # Uncertainty penalty from weighted residuals
    resid = y - (a + b * x)
    resid_std = float(math.sqrt(np.sum(w * resid ** 2) / (np.sum(w) + 1e-12)))

    # Reliability margin penalties (focus on hardest reached point)
    u_max = float(u[-1])
    # Penalize being close to the schedule continuation threshold (0.5)
    margin_pen = _softplus((u_max - 0.30) / 0.05)  # starts increasing around 30% unsolved

    # Average reliability penalty across scales (lighter)
    avg_rel_pen = float(np.mean([_softplus((ui - 0.25) / 0.07) for ui in u]))

    # Coverage: if the schedule never reached large N, don't over-trust extrapolation
    N_max = float(Ns.max())
    coverage_pen = 0.0
    if N_max < N_target:
        coverage_pen = 1.2 * math.log10(N_target / (N_max + 1e-9))

    # Superlinear penalty (polynomial-time target)
    # Penalize b above ~1.2 (gentle near boundary, grows smoothly)
    superlin_pen = 0.6 * _softplus((b - 1.2) / 0.15)

    obj = (
        math.log10(max(pred_eff, 1.0))
        + 0.35 * resid_std
        + 1.0 * margin_pen
        + 0.25 * avg_rel_pen
        + coverage_pen
        + superlin_pen
    )

    if not math.isfinite(obj):
        return 1e3
    return float(obj)
