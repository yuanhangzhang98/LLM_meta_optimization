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

    Proxy: predict log10(expected restart-steps) at N=2000 using a weighted
    log-log scaling fit on *p-adjusted* median steps, plus smooth penalties for:
      - being too close to the schedule gate (unsolved_fraction ~ 0.5)
      - censoring (median_step near max_steps)
      - limited scale coverage (max N reached)
      - superlinear scaling exponent

    Args:
        experiment_results: List of dicts with experiment details
    Returns:
        Float to be MINIMIZED
    """
    if not experiment_results:
        return 1e3

    # Extract arrays (robust to missing keys)
    Ns, steps, unsolveds, budgets = [], [], [], []
    for r in experiment_results:
        N = r.get("N", None)
        s = r.get("median_step", None)
        u = r.get("unsolved_fraction", None)
        B = r.get("max_steps", None)
        if N is None or s is None or u is None:
            continue
        if N <= 0 or s <= 0:
            continue
        if B is None or B <= 0:
            B = max(s, 1.0)
        Ns.append(float(N))
        steps.append(float(s))
        unsolveds.append(float(u))
        budgets.append(float(B))

    if len(Ns) == 0:
        return 1e3

    Ns = np.asarray(Ns)
    steps = np.asarray(steps)
    unsolveds = np.asarray(unsolveds)
    budgets = np.asarray(budgets)

    # Core transformation: p-adjusted (restart-like) median steps + censor penalty
    p = np.clip(1.0 - unsolveds, 1e-3, 1.0)  # solved probability
    censor = np.clip(steps / np.maximum(budgets, 1.0), 0.0, 2.0)  # can exceed 1 if weird logs

    # Expected restart steps approximation: steps / p^k (k>1 emphasizes reliability)
    k_restart = 1.3
    log_s = np.log(np.maximum(steps, 1.0)) + k_restart * np.log(1.0 / p)

    # Smooth censor penalty when median approaches the step budget
    # (encourages designs whose median is well within max_steps)
    censor_r0, censor_tau, censor_w = 0.70, 0.08, 1.4
    log_s = log_s + censor_w * np.vectorize(_softplus)((censor - censor_r0) / censor_tau)

    # Weights: prefer larger N, but downweight censored / low-reliability points
    Nw = (Ns / np.max(Ns)) ** 0.7
    pw = p ** 2.0
    cw = (1.0 - np.clip(censor, 0.0, 1.0)) ** 1.5 + 1e-3
    w = Nw * pw * cw

    # Fit log_s ~ a + b*log(N) with ridge for stability
    x = np.log(Ns)
    X = np.stack([np.ones_like(x), x], axis=1)

    # If insufficient points, fall back to a conservative default slope
    N_target = 2000.0
    if len(Ns) < 2 or np.all(w <= 1e-8):
        y0 = float(np.median(log_s))
        # conservative polynomial exponent prior
        b_prior = 2.0
        a_prior = y0 - b_prior * float(np.median(x))
        y_pred = a_prior + b_prior * math.log(N_target)
        base = y_pred
        b_hat = b_prior
    else:
        sw = np.sqrt(np.clip(w, 1e-12, None))
        Xw = X * sw[:, None]
        yw = log_s * sw

        # Ridge towards modest polynomial scaling
        # (keeps BO smooth when only a few points exist)
        lam_a, lam_b = 0.2, 0.6
        # weak prior anchored near the first point
        idx0 = int(np.argmin(Ns))
        b_prior = 1.8
        a_prior = float(log_s[idx0] - b_prior * x[idx0])

        A = np.vstack([
            Xw,
            [math.sqrt(lam_a), 0.0],
            [0.0, math.sqrt(lam_b)],
        ])
        bvec = np.concatenate([
            yw,
            [math.sqrt(lam_a) * a_prior],
            [math.sqrt(lam_b) * b_prior],
        ])

        coef, _, _, _ = np.linalg.lstsq(A, bvec, rcond=None)
        a_hat, b_hat = float(coef[0]), float(coef[1])
        y_pred = a_hat + b_hat * math.log(N_target)
        base = y_pred

    # Smooth penalties
    # 1) Reliability margin: avoid skating at the schedule gate (u≈0.5)
    # Target u<=0.30 on larger N.
    u_target, u_tau, u_w = 0.30, 0.05, 0.35
    rel_pen = np.sum(Nw * np.vectorize(_softplus)((unsolveds - u_target) / u_tau)) / (np.sum(Nw) + 1e-9)

    # 2) Coverage: encourage reaching larger N
    maxN = float(np.max(Ns))
    cov = math.log(maxN + 1.0) / math.log(N_target + 1.0)
    cov_pen = 0.55 * (1.0 - cov) ** 2

    # 3) Scaling exponent penalty: encourage polynomial with small exponent
    b0, b_tau, b_w = 1.6, 0.25, 0.25
    slope_pen = b_w * _softplus((b_hat - b0) / b_tau)

    # Convert to log10 steps
    obj = (base / math.log(10.0)) + u_w * rel_pen + cov_pen + slope_pen

    # Guardrails
    if not np.isfinite(obj):
        return 1e3
    return float(obj)
