import math
import numpy as np
from scipy.stats import theilslopes


def objective(experiment_results):
    """Estimate the research goal from experiment results (lower is better)."""
    if not experiment_results:
        return 1e6

    N_target = 2000.0
    K_censor = 2.0

    # Group runs by N
    byN = {}
    for e in experiment_results:
        N = e.get("N", None)
        if N is None:
            continue
        byN.setdefault(int(N), []).append(e)
    if not byN:
        return 1e6

    def _finite_pos(x):
        return x is not None and np.isfinite(x) and x > 0

    # Collect successful points (unique N): fastest successful median_step for that N
    success_points = []  # (N, best_step, best_max_steps)
    for N, runs in byN.items():
        succ = [r for r in runs if r.get("unsolved_fraction", 1.0) < 0.5 and _finite_pos(r.get("median_step", None))]
        if not succ:
            continue
        # pick run with minimum median_step
        best = min(succ, key=lambda r: float(r.get("median_step")))
        ms = best.get("max_steps", None)
        if not _finite_pos(ms):
            # fall back to max over runs at that N
            ms_list = [r.get("max_steps", None) for r in runs]
            ms_list = [m for m in ms_list if _finite_pos(m)]
            ms = float(max(ms_list)) if ms_list else float(best.get("median_step"))
        success_points.append((float(N), float(best["median_step"]), float(ms)))

    max_solved_N = max((N for N, _, _ in success_points), default=1.0)

    # First truly failed level above max_solved_N (no successful run)
    N_block = None
    max_steps_block = None
    uf_block_best = None
    for N in sorted(byN.keys()):
        if float(N) <= float(max_solved_N):
            continue
        runs = byN[N]
        any_success = any(r.get("unsolved_fraction", 1.0) < 0.5 for r in runs)
        if not any_success:
            N_block = float(N)
            ufs = [r.get("unsolved_fraction", None) for r in runs]
            ufs = [u for u in ufs if u is not None and np.isfinite(u)]
            uf_block_best = float(min(ufs)) if ufs else 1.0
            ms = [r.get("max_steps", None) for r in runs]
            ms = [m for m in ms if _finite_pos(m)]
            max_steps_block = float(max(ms)) if ms else None
            break

    # --- Primary: reach term ---
    if max_solved_N >= N_target:
        term_reach = 0.0
    else:
        term_reach = math.log(max(N_target / max_solved_N, 1.0), 2)
        term_reach = float(min(term_reach, 50.0))

    # --- Scaling fit on successful points only ---
    success_points.sort(key=lambda t: t[0])
    num_succ = len(success_points)

    if num_succ >= 2:
        xs = np.log(np.array([p[0] for p in success_points], dtype=float))
        ys = np.log(np.array([max(p[1], 1e-12) for p in success_points], dtype=float))

        if num_succ >= 3:
            fit = theilslopes(ys, xs)
            slope = float(fit.slope)
            intercept = float(fit.intercept)
        else:
            slope, intercept = np.polyfit(xs, ys, deg=1)
            slope = float(slope)
            intercept = float(intercept)

        slope = float(np.clip(slope, -0.5, 8.0))
        term_slope = slope

        pred2000 = math.exp(intercept + slope * math.log(N_target))
        term_pred = float(np.clip(math.log10(max(pred2000, 1e-12)), -6.0, 30.0))

        penalty_censor = 0.0
        if N_block is not None and max_steps_block is not None:
            censor_steps = K_censor * max_steps_block
            pred_block = math.exp(intercept + slope * math.log(N_block))
            penalty_censor = max(0.0, math.log10(max(censor_steps, 1e-12)) - math.log10(max(pred_block, 1e-12)))
            penalty_censor = float(np.clip(penalty_censor, 0.0, 30.0))
    else:
        term_slope = 8.0
        if num_succ == 1:
            N_last, step_last, _ = success_points[0]
            pred2000 = step_last * (N_target / max(N_last, 1e-9)) ** 2
            term_pred = float(np.clip(math.log10(max(pred2000, 1.0)), -6.0, 30.0))
        else:
            ms = [e.get("max_steps", None) for e in experiment_results]
            ms = [m for m in ms if _finite_pos(m)]
            term_pred = float(np.clip(math.log10(K_censor * max(ms) if ms else 1e6), -6.0, 30.0))
        penalty_censor = 0.0

    penalty_few = 0.0 if num_succ >= 3 else (3 - num_succ) * 1.5

    # --- Secondary: headroom on successful levels (do NOT penalize 0.49) ---
    headroom_term = 0.0
    if num_succ > 0:
        hrs = []
        for _, step, ms in success_points:
            ratio = max(step / max(ms, 1e-12), 1e-12)
            hrs.append(float(np.clip(math.log10(ratio), -6.0, 1.0)))  # lower (more negative) is better
        headroom_term = float(np.median(hrs))

    # --- Secondary: frontier robustness near threshold at max_solved_N and N_block ---
    # At max_solved_N: reward margin below threshold (bigger is better)
    margin_succ = 0.0
    headroom_front = 0.0
    if num_succ > 0:
        runs = byN.get(int(max_solved_N), [])
        ufs = [r.get("unsolved_fraction", None) for r in runs]
        ufs = [u for u in ufs if u is not None and np.isfinite(u)]
        uf_best = float(min(ufs)) if ufs else 0.49
        margin_succ = max(0.0, 0.5 - uf_best)

        # headroom specifically at the frontier (best successful run)
        frontier = [p for p in success_points if p[0] == float(max_solved_N)]
        if frontier:
            _, step_f, ms_f = frontier[0]
            headroom_front = float(np.clip(math.log10(max(step_f / max(ms_f, 1e-12), 1e-12)), -6.0, 1.0))

    # At N_block: penalize excess above threshold (smaller is better)
    excess_fail = 0.0
    if N_block is not None:
        excess_fail = max(0.0, float(uf_block_best) - 0.5)

    # Mild reward for reaching larger N (keep small; reach term dominates)
    term_N_mild = -0.05 * math.log(max_solved_N + 1.0, 2)

    # Composite
    obj = (
        10.0 * term_reach
        + 1.0 * term_slope
        + 0.25 * term_pred
        + 2.0 * penalty_censor
        + penalty_few
        + 0.30 * headroom_term
        + 0.50 * excess_fail
        - 0.50 * margin_succ
        + 0.10 * headroom_front
        + term_N_mild
    )

    return float(obj)
