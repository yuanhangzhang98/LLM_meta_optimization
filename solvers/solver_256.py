import torch

# ------------------------------------------------------------------ #
# 1.  VARIABLES_SPEC
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
# 2.  HYPER_SPACE
# ------------------------------------------------------------------ #

HYPER_SPACE = {
    # Baseline (defaults taken from a stable tuned family)
    "alpha":   dict(type="log_uniform", default=5.628430483159852,      low=0.5,  high=50.0),
    "beta":    dict(type="log_uniform", default=40.43498719903352,      low=2.0,  high=200.0),
    "gamma":   dict(type="uniform",     default=0.21765848650443836,    low=0.01, high=0.50),
    "delta":   dict(type="uniform",     default=0.09869929008180485,    low=0.01, high=0.50),
    "epsilon": dict(type="log_uniform", default=0.0009807063473973208,  low=1e-4, high=1e-2),
    "zeta":    dict(type="log_uniform", default=0.0010498238170223528,  low=1e-4, high=1.0),

    # NEW: window-ramp reactivation (Design-24 style, simplified)
    "kappa":      dict(type="uniform",     default=0.9,                 low=0.0,  high=2.0),
    "xl_thr":     dict(type="log_uniform", default=5000.0,              low=1.0,  high=1e4),
    "xl_sharp":   dict(type="uniform",     default=2.0,                 low=0.1,  high=5.0),

    "stag_margin": dict(type="uniform",     default=0.15,                low=0.0,  high=0.25),
    "stag_sharp":  dict(type="log_uniform", default=0.01,                low=1e-3, high=0.20),

    "win_floor": dict(type="uniform",     default=0.55,                low=0.0,  high=0.90),
    "win_pow":   dict(type="log_uniform", default=0.6,                 low=0.25, high=4.0),

    "xs_star":   dict(type="uniform",     default=0.45,                low=0.05, high=0.90),

    "lr":      dict(type="log_uniform", default=1.7,                   low=1e-1, high=3.0),
}


# ------------------------------------------------------------------ #
# 3.  _grad_single
# ------------------------------------------------------------------ #

def _grad_single(vars, idx, sgn, hp):
    v  = vars["v"]
    xl = vars["xl"]
    xs = vars["xs"]

    # ===== CLAUSE EVALUATION (baseline) =====
    v_lit = v[idx] * sgn
    top_val, top_idx = torch.topk(v_lit, 2, dim=-1)
    unsat_top = 0.5 * (1.0 - top_val)
    c = unsat_top[:, 0].clamp(0.0, 1.0)

    # ===== GRADIENT TERM (baseline) =====
    G = c.unsqueeze(-1).repeat(1, 3)
    G.scatter_(1, top_idx[:, 0:1], unsat_top[:, 1:2])
    G *= (xl * xs).unsqueeze(-1)

    # ===== RIGIDITY TERM (baseline) =====
    R = torch.zeros_like(G)
    R.scatter_(1, top_idx[:, 0:1], c.unsqueeze(-1))
    R *= ((1.0 + hp.zeta * xl) * (1.0 - xs)).unsqueeze(-1)

    # ===== ACCUMULATE v GRADIENT (baseline) =====
    dv_clause = (G + R) * sgn
    dv = torch.zeros_like(v).scatter_add_(0, idx.reshape(-1), dv_clause.reshape(-1))

    # ===== AUXILIARY VARIABLE GRADIENTS =====
    dxl = hp.alpha * (c - hp.delta)

    # Baseline xs RHS + ONE ADDITION: window-ramp release near gamma from below
    gamma = torch.as_tensor(hp.gamma, device=v.device, dtype=v.dtype)

    # xl gate in log-space (only for persistently problematic clauses)
    xl_thr = torch.as_tensor(hp.xl_thr, device=v.device, dtype=v.dtype).clamp_min(1.0)
    xl_sharp = torch.as_tensor(hp.xl_sharp, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    log_xl = torch.log(xl.clamp_min(1.0))
    log_thr = torch.log(xl_thr)
    gate_xl = torch.sigmoid((log_xl - log_thr) / xl_sharp)

    # Below-gamma window: (gamma - stag_margin) < c < gamma
    stag_margin = torch.as_tensor(hp.stag_margin, device=v.device, dtype=v.dtype).clamp_min(0.0)
    stag_sharp  = torch.as_tensor(hp.stag_sharp,  device=v.device, dtype=v.dtype).clamp_min(1e-6)
    c0 = (gamma - stag_margin).clamp(0.0, 1.0)

    near_from_below = torch.sigmoid((c - c0) / stag_sharp)
    below_gamma     = torch.sigmoid((gamma - c) / stag_sharp)
    window = near_from_below * below_gamma

    # Smoothstep amplitude across the window (Design-24 style)
    denom = (gamma - c0).clamp_min(1e-6)
    t = ((c - c0) / denom).clamp(0.0, 1.0)

    win_floor = torch.as_tensor(hp.win_floor, device=v.device, dtype=v.dtype).clamp(0.0, 1.0)
    win_pow   = torch.as_tensor(hp.win_pow,   device=v.device, dtype=v.dtype).clamp_min(0.05)

    u = t.pow(win_pow)
    smooth = u * u * (3.0 - 2.0 * u)
    amp = win_floor + (1.0 - win_floor) * smooth

    xs_star = torch.as_tensor(hp.xs_star, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    push_up = torch.relu(xs_star - xs) / xs_star

    release = hp.kappa * gate_xl * window * amp * push_up  # bounded, upward-only

    rhs_xs = (xs + hp.epsilon) * (c - gamma) + release
    dxs = hp.beta * rhs_xs

    # ===== GRADIENT SCALING (baseline) =====
    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))

    dv  = -dv  * scale
    dxl = -dxl * scale
    dxs = -dxs * scale

    return {"v": dv, "xl": dxl, "xs": dxs}
