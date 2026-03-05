import math
import numpy as np


def _sigmoid(x):
    # numerically stable sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def _weighted_linreg(x, y, w):
    """Return slope, intercept for y ~ a + b x."""
    w = np.asarray(w, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    wsum = float(w.sum())
    if not np.isfinite(wsum) or wsum <= 1e-12:
        return None
    xm = float((w * x).sum() / wsum)
    ym = float((w * y).sum() / wsum)
    xv = float((w * (x - xm) ** 2).sum())
    if xv <= 1e-12:
        return None
    cov = float((w * (x - xm) * (y - ym)).sum())
    b = cov / xv
    a = ym - b * xm
    return b, a


def objective(experiment_results):
    """Estimate the research goal from experiment results (lower is better)."""
    if not experiment_results:
        return 1e6

    # Aggregate by N, keeping the highest-budget run (largest max_steps).
    byN = {}
    for r in experiment_results:
        try:
            N = int(r.get("N"))
        except Exception:
            continue
        u = float(r.get("unsolved_fraction", 1.0))
        s = float(r.get("median_step", r.get("max_steps", 1.0)))
        ms = float(r.get("max_steps", 1.0))
        b = float(r.get("batch", 100.0))

        # keep record with larger max_steps; tie-breakers: lower u then lower s
        prev = byN.get(N)
        cand = (ms, -u, -s, u, s, ms, b)
        if prev is None or cand > prev["key"]:
            byN[N] = {
                "key": cand,
                "N": float(N),
                "u": u,
                "s": max(1.0, s),
                "ms": max(1.0, ms),
                "b": max(1.0, b),
            }

    pts = sorted(byN.values(), key=lambda d: d["N"])
    if not pts:
        return 1e6

    Ns = np.array([p["N"] for p in pts], dtype=float)
    u = np.array([p["u"] for p in pts], dtype=float)
    s = np.array([p["s"] for p in pts], dtype=float)
    b = np.array([p["b"] for p in pts], dtype=float)

    N_max = float(np.max(Ns))
    N_min = float(np.min(Ns))

    # Uncertainty-aware unsolved (pessimistic): u_eff = u + 1*SE_binom
    se = np.sqrt(np.clip(u * (1.0 - u), 0.0, 0.25) / np.maximum(b, 1.0))
    u_eff = np.clip(u + 1.0 * se, 0.0, 1.0)

    # --- (i) Effective solved size Neff at a stricter threshold than schedule ---
    u_thresh = 0.40
    tau_u = 0.05

    # Hard Neff
    ok = u_eff <= u_thresh
    Neff_hard = float(np.max(Ns[ok])) if np.any(ok) else 0.0

    # Soft Neff (keeps signal when nothing reaches threshold)
    w_soft = np.array([_sigmoid((u_thresh - ue) / tau_u) for ue in u_eff], dtype=float)
    # emphasize larger N but avoid instability
    w_soft = w_soft * (Ns / max(N_min, 1.0)) ** 0.5
    Neff_soft = float((w_soft * Ns).sum() / (w_soft.sum() + 1e-12))

    Neff = 0.7 * Neff_hard + 0.3 * Neff_soft

    # --- (ii) Smooth penalty for exceeding threshold (N-weighted) ---
    denom = max(1e-6, 0.5 - u_thresh)  # normalize relative to schedule boundary
    viol = np.maximum(0.0, (u_eff - u_thresh) / denom)
    wN = (Ns / max(N_max, 1.0)) ** 1.0
    penalty_unsolved = float(np.mean(wN * viol ** 2))

    # --- (iii-a) TTS surrogate from success probability, using median_step as per-run cost ---
    p_solve = np.clip(1.0 - u_eff, 1e-6, 1.0 - 1e-6)
    runs99 = math.log(0.01) / np.log(1.0 - p_solve)  # positive
    tts = s * runs99
    log_tts = np.log1p(np.maximum(tts, 0.0))
    avg_log_tts = float((wN * log_tts).sum() / (wN.sum() + 1e-12))

    # --- (iii-b) Scaling exponent p from weighted log-log fit of steps vs N ---
    # Use soft inclusion around the schedule pass/fail boundary so fit is always defined.
    u_fit = 0.50
    tau_fit = 0.06
    w_fit = np.array([_sigmoid((u_fit - ue) / tau_fit) for ue in u_eff], dtype=float)
    w_fit = w_fit * (Ns / max(N_max, 1.0)) ** 0.5
    x = np.log(np.maximum(Ns, 2.0))
    y = np.log(np.maximum(s, 1.0))

    reg = _weighted_linreg(x, y, w_fit)
    if reg is None:
        p = 5.0
    else:
        p = float(reg[0])
        if not np.isfinite(p):
            p = 5.0

    # Composite objective in "N-equivalent" units (baseline is ~ -largest_N)
    # Lower is better.
    obj = (
        -Neff
        + (N_max * 2.0) * penalty_unsolved
        + 15.0 * avg_log_tts
        + 25.0 * max(0.0, p)  # minimize scaling exponent
    )

    # Safety for pathological outputs
    if not np.isfinite(obj):
        return 1e6
    return float(obj)
