import math
import numpy as np


def _softplus(x: float) -> float:
    if x > 60.0:
        return x
    if x < -60.0:
        return math.exp(x)
    return math.log1p(math.exp(x))


def _schedule_Ns(N_start=10, N_cap=5120, max_len=256):
    Ns = []
    for c in range(max_len):
        N = int(round(N_start * (2 ** (c / 2))))
        if Ns and N == Ns[-1]:
            continue
        Ns.append(N)
        if N >= N_cap:
            break
    return Ns


def _infer_N_cap(results):
    fset = {str(r.get("fidelity", "")).lower() for r in (results or [])}
    if "high" in fset:
        return 2560
    if "medium" in fset:
        return 1280
    if "low" in fset:
        return 640
    # fallback from observed max_steps
    mx = max([float(r.get("max_steps", 0.0)) for r in (results or [])] + [0.0])
    if mx >= 1e6:
        return 2560
    if mx >= 1e5:
        return 1280
    return 640


def _passes(run):
    return float(run.get("unsolved_fraction", 1.0)) < 0.5


def _best_pass_run(runs_at_N):
    # among passing runs: prefer larger max_steps, then smaller median_step
    cand = [r for r in runs_at_N if _passes(r)]
    if not cand:
        return None
    def key(r):
        ms = float(r.get("max_steps", 0.0) or 0.0)
        med = float(r.get("median_step", 1e30) or 1e30)
        return (ms, -med)
    return max(cand, key=key)


def _worst_window_pred_ratio(N_query, Ns, ratios, windows=(2, 3, 4)):
    Ns = np.asarray(Ns, dtype=float)
    ratios = np.asarray(ratios, dtype=float)
    logN = np.log(np.maximum(Ns, 1.0))
    logR = np.log(np.maximum(ratios, 1e-12))

    worst = 0.0
    for w in windows:
        ww = min(int(w), len(logN))
        if ww >= 2:
            X = logN[-ww:]
            Y = logR[-ww:]
            b, a = np.polyfit(X, Y, 1)
            pred_logR = float(a + b * math.log(max(float(N_query), 1.0)))
        else:
            pred_logR = float(logR[-1])
        pred = math.exp(pred_logR)
        pred = min(max(pred, 1e-12), 0.999999)
        worst = max(worst, pred)
    return float(worst)


def objective(experiment_results):
    if not experiment_results:
        return 1e6

    # group by N
    byN = {}
    for r in experiment_results:
        if "N" not in r:
            continue
        try:
            N = int(r["N"])
        except Exception:
            continue
        byN.setdefault(N, []).append(r)

    if not byN:
        return 1e6

    N_cap = int(_infer_N_cap(experiment_results))
    sched = _schedule_Ns(N_start=10, N_cap=N_cap)

    # schedule-faithful reach: last cleared scheduled N (must have at least one passing run)
    cleared = []  # list of dicts with N, median_step, max_steps
    for N in sched:
        runs = byN.get(int(N), None)
        if runs is None:
            break
        best = _best_pass_run(runs)
        if best is None:
            break
        med = float(best.get("median_step", 1.0))
        ms = float(best.get("max_steps", max(med, 1.0)))
        med = max(med, 1.0)
        ms = max(ms, 1.0)
        cleared.append({"N": float(N), "med": med, "ms": ms})

    if not cleared:
        # no cleared level: keep smooth-ish but clearly bad
        N0 = float(min(byN.keys()))
        return 5e5 + 1e2 * math.log1p(N0)

    Ns = np.array([c["N"] for c in cleared], dtype=float)
    meds = np.array([c["med"] for c in cleared], dtype=float)
    msteps = np.array([c["ms"] for c in cleared], dtype=float)

    N_max = float(Ns[-1])
    ratios = np.clip(meds / np.maximum(msteps, 1.0), 1e-12, 0.999999)

    # (1) Reach (dominant)
    reach_term = -N_max

    # (2) Tail headroom tie-break: last 2–4 successful Ns
    K = int(min(4, len(ratios)))
    tail = ratios[-K:]

    safety = 0.70
    tau = 0.10

    # absolute headroom preference (smaller ratio is better), plus overshoot above safety
    tail_ratio_med = float(np.median(tail))
    tail_overs = [tau * _softplus(math.log(max(float(r) / safety, 1e-12)) / tau) for r in tail]
    tail_overs_pen = float(np.mean(tail_overs)) if tail_overs else 0.0

    # (3) Budget-cliff penalty: sharp worsening in ratio near the end
    cliff_pen = 0.0
    if len(tail) >= 2:
        dlogs = []
        for i in range(1, len(tail)):
            dlogs.append(math.log(float(tail[i])) - math.log(float(tail[i - 1])))
        # penalize positive jumps beyond a small tolerance
        cliff0, cliff_tau = 0.12, 0.05
        cliff_pen = float(np.mean([cliff_tau * _softplus((dl - cliff0) / cliff_tau) for dl in dlogs]))

    # (4) Conservative next scheduled N clearance: worst-window prediction in log-ratio space
    # (scores next 2 scheduled jumps; explicitly hits 1280→1810→2560 when applicable)
    idx = int(np.searchsorted(np.array(sched, dtype=float), N_max, side="left"))
    future = []
    for j in range(idx + 1, min(idx + 3, len(sched))):
        future.append(float(sched[j]))

    tfit = int(min(6, len(ratios)))
    Ns_tail = Ns[-tfit:]
    ratios_tail = ratios[-tfit:]

    future_terms = []
    for Nt in future:
        r_pred = _worst_window_pred_ratio(Nt, Ns_tail, ratios_tail, windows=(2, 3, 4))
        future_terms.append(tau * _softplus(math.log(max(r_pred / safety, 1e-12)) / tau))
    future_pen = float(np.mean(future_terms)) if future_terms else 0.0

    # (5) Tail trend penalty (worsening ratios with N)
    trend_pen = 0.0
    if len(ratios_tail) >= 3:
        logN = np.log(np.maximum(Ns_tail, 1.0))
        logR = np.log(np.maximum(ratios_tail, 1e-12))
        d = np.diff(logN)
        slopes = np.diff(logR) / np.maximum(d, 1e-12)
        s_pos = float(np.median(np.maximum(slopes, 0.0)))
        trend_pen = 0.08 * _softplus(s_pos / 0.06)

    # Lexicographic scalarization via scale separation (reach dominates by construction)
    # Tail/headroom terms are strong tie-breakers among equal N_max.
    obj = (
        reach_term
        + 40.0 * tail_ratio_med
        + 120.0 * tail_overs_pen
        + 90.0 * future_pen
        + 120.0 * cliff_pen
        + 40.0 * trend_pen
    )

    if not math.isfinite(obj):
        return 1e6
    return float(obj)
