import torch

# ------------------------------------------------------------------ #
# 1. VARIABLES_SPEC
# ------------------------------------------------------------------ #

VARIABLES_SPEC = {
    "v":  dict(init=lambda s: 2 * torch.rand(s) - 1,
               shape=("B", "N"),
               bounds=(-1.0, 1.0)),

    "xl": dict(init=lambda s: torch.ones(s),
               shape=("B", "M"),
               bounds=(1.0, 1e6)),

    "xs": dict(init=lambda s: torch.zeros(s),
               shape=("B", "M"),
               bounds=(0.0, 1.0)),
}


# ------------------------------------------------------------------ #
# 2. HYPER_SPACE
# ------------------------------------------------------------------ #

HYPER_SPACE = {
    # baseline / 208-family defaults (from 262)
    "alpha":   dict(type="log_uniform", default=5.628430483159852,      low=0.5,  high=50.0),
    "beta":    dict(type="log_uniform", default=40.43498719903352,      low=2.0,  high=200.0),
    "gamma":   dict(type="uniform",     default=0.21765848650443836,    low=0.01, high=0.50),
    "delta":   dict(type="uniform",     default=0.09869929008180485,    low=0.01, high=0.50),
    "epsilon": dict(type="log_uniform", default=0.0009807063473973208,  low=1e-4, high=1e-2),
    "zeta":    dict(type="log_uniform", default=0.0010498238170223528,  low=1e-4, high=1.0),

    # asymmetric xs knobs (208)
    "eta":      dict(type="uniform",    default=0.3794440949062971,     low=0.0,  high=3.0),
    "kappa":    dict(type="uniform",    default=0.019776372159554443,   low=0.0,  high=0.25),
    "eta_hold": dict(type="uniform",    default=0.2999995351252482,     low=0.0,  high=3.0),

    # satisfied-side gate (208)
    "tau_sat": dict(type="log_uniform", default=0.010000370298734167,   low=1e-5, high=2e-1),
    "p_sat":   dict(type="uniform",     default=0.9842825158415519,     low=0.5,  high=3.0),

    # violated-side boundary reactivation strength (208)
    "kappa_on": dict(type="uniform",    default=0.04999601539141226,    low=0.0,  high=0.25),

    # tie-only parameters (used ONLY in dxs)
    "g0_tie":    dict(type="log_uniform", default=0.0034,              low=1e-4, high=5e-2),
    "kappa_tie": dict(type="log_uniform", default=0.010,               low=1e-4, high=5e-2),

    # NEW (single change vs 262): bounded, low-upward-bias tie estimator via negative power mean
    # p_tie -> -inf approaches u1; p_tie -> 0- approaches geometric mean.
    "p_tie":     dict(type="uniform",     default=-4.0,                low=-16.0, high=-0.5),

    "lr":      dict(type="log_uniform", default=1.7694676919282437,     low=1e-1, high=3.0),
}


# ------------------------------------------------------------------ #
# 3. _grad_single
# ------------------------------------------------------------------ #

def _grad_single(vars, idx, sgn, hp):
    v  = vars["v"]
    xl = vars["xl"]
    xs = vars["xs"]

    eps = 1e-12

    # ===== CLAUSE EVALUATION (hard, baseline) =====
    v_lit = v[idx] * sgn  # (M,3)
    top_val, top_idx = torch.topk(v_lit, 2, dim=-1)
    unsat_top = 0.5 * (1.0 - top_val)  # (M,2)
    c_hard = unsat_top[:, 0]           # (M,)

    # ===== GRADIENT TERM (baseline) =====
    G = c_hard.unsqueeze(-1).repeat(1, 3)
    G.scatter_(1, top_idx[:, 0:1], unsat_top[:, 1:2])
    G *= (xl * xs).unsqueeze(-1)

    # ===== RIGIDITY TERM (baseline) =====
    R = torch.zeros_like(G)
    R.scatter_(1, top_idx[:, 0:1], c_hard.unsqueeze(-1))
    R *= ((1.0 + hp.zeta * xl) * (1.0 - xs)).unsqueeze(-1)

    # ===== ACCUMULATE v GRADIENT (baseline) =====
    dv_clause = (G + R) * sgn
    dv = torch.zeros_like(v).scatter_add_(0, idx.reshape(-1), dv_clause.reshape(-1))

    # ===== AUXILIARY VARIABLE GRADIENTS =====
    dxl = hp.alpha * (c_hard - hp.delta)

    # --- tie-only smoothed monitor for xs switching (used ONLY in dxs) ---
    u = 0.5 * (1.0 - v_lit)  # per-literal unsat in [0,1]
    u12, _ = torch.topk(u, 2, dim=-1, largest=False)  # (M,2): u1<=u2
    u1 = u12[:, 0]
    gap = (u12[:, 1] - u12[:, 0]).clamp(min=0.0)

    # C∞ compact-support bump tie window (as in 262)
    t_raw = (gap - hp.g0_tie) / (hp.kappa_tie + eps)
    t = t_raw.clamp(0.0, 1.0)
    a = torch.exp(-1.0 / (t + eps))
    b = torch.exp(-1.0 / (1.0 - t + eps))
    s = a / (a + b)              # ~0 at t=0, ~1 at t=1
    w_mid = 1.0 - s              # ~1 at t=0, ~0 at t=1
    w = torch.where(t_raw <= 0.0, torch.ones_like(w_mid), w_mid)
    w = torch.where(t_raw >= 1.0, torch.zeros_like(w), w)
    w = w.clamp(0.0, 1.0)

    # SINGLE CHANGE vs 262: bounded negative power-mean over top-2 (in [u1,u2])
    p = torch.as_tensor(hp.p_tie, device=v.device, dtype=v.dtype)
    p = p.clamp(max=-0.05)  # ensure negative (avoid instability near 0)
    u_safe = u12.clamp(min=eps, max=1.0)
    mean_pow = u_safe.pow(p).mean(dim=-1).clamp(min=eps)
    c_pm2 = mean_pow.pow(1.0 / p).clamp(0.0, 1.0)  # in [u1,u2]

    c_mem = ((1.0 - w) * u1 + w * c_pm2).clamp(0.0, 1.0)

    # xs update (Design 208), driven by c_mem
    pos = torch.relu(c_mem - hp.gamma)
    neg = torch.relu(hp.gamma - c_mem)

    tau_sat = torch.as_tensor(hp.tau_sat, device=v.device, dtype=v.dtype).clamp(min=eps)
    p_sat   = torch.as_tensor(hp.p_sat,   device=v.device, dtype=v.dtype).clamp(min=0.5)

    frac = tau_sat / (neg + tau_sat)
    gate_sat = 1.0 - torch.pow(frac.clamp(0.0, 1.0), p_sat)

    beta_on   = hp.beta
    beta_off  = hp.beta * (1.0 + hp.eta)
    beta_hold = hp.beta * (1.0 + hp.eta_hold)

    y = pos / tau_sat
    bump_base = torch.square(y / (1.0 + y)) * torch.exp(-y)
    bump = xs * bump_base

    on_drive  = pos + hp.kappa_on * bump
    off_drive = (beta_off * neg) + (beta_hold * (hp.kappa * gate_sat))

    dxs = beta_on * (1.0 - xs) * on_drive - (xs + hp.epsilon) * off_drive

    # ===== GRADIENT SCALING (baseline style) =====
    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))

    return {
        "v":  -dv  * scale,
        "xl": -dxl * scale,
        "xs": -dxs * scale,
    }
