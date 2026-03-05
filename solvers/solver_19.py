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
    # Baseline
    "alpha":   dict(type="log_uniform", default=5.0,   low=0.5,   high=50.0),
    "beta":    dict(type="log_uniform", default=20.0,  low=2.0,   high=200.0),
    "gamma":   dict(type="uniform",     default=0.25,  low=0.01,  high=0.50),
    "delta":   dict(type="uniform",     default=0.05,  low=0.01,  high=0.50),
    "epsilon": dict(type="log_uniform", default=1e-3,  low=1e-4,  high=1e-2),
    "zeta":    dict(type="log_uniform", default=1e-3,  low=1e-4,  high=1.0),

    # Release (from design 44 family)
    "kappa":      dict(type="uniform",     default=0.5,   low=0.0,   high=1.5),
    "xl_thr":     dict(type="log_uniform", default=200.0, low=1.0,   high=1e4),
    "xl_sharp":   dict(type="uniform",     default=2.0,   low=0.2,   high=5.0),
    "stag_margin":dict(type="uniform",     default=0.05,  low=0.0,   high=0.15),
    "overshoot":  dict(type="uniform",     default=0.03,  low=0.0,   high=0.15),
    "band_sharp": dict(type="log_uniform", default=0.02,  low=1e-3,  high=0.10),
    "xs_star":    dict(type="uniform",     default=0.7,   low=0.05,  high=0.90),

    "lr":      dict(type="log_uniform", default=1.0,   low=1e-1,  high=3.0),
}


# ------------------------------------------------------------------ #
# 3. _grad_single
# ------------------------------------------------------------------ #

def _grad_single(vars, idx, sgn, hp):
    v  = vars["v"]
    xl = vars["xl"]
    xs = vars["xs"]

    # Literals
    v_lit = v[idx] * sgn  # (M,3)

    # Baseline hard top-2 clause cost
    top_val, top_idx = torch.topk(v_lit, 2, dim=-1)   # (M,2)
    unsat_top = 0.5 * (1.0 - top_val)                 # (M,2)
    c = unsat_top[:, 0].clamp(0.0, 1.0)               # (M,)

    # Baseline G/R construction
    G = c.unsqueeze(-1).repeat(1, 3)
    G.scatter_(1, top_idx[:, 0:1], unsat_top[:, 1:2])
    G *= (xl * xs).unsqueeze(-1)

    R = torch.zeros_like(G)
    R.scatter_(1, top_idx[:, 0:1], c.unsqueeze(-1))
    R *= ((1.0 + hp.zeta * xl) * (1.0 - xs)).unsqueeze(-1)

    dv_clause = (G + R) * sgn
    dv = torch.zeros_like(v).scatter_add_(0, idx.reshape(-1), dv_clause.reshape(-1))

    # Baseline xl update
    dxl = hp.alpha * (c - hp.delta)

    # xs update: baseline + release
    gamma = torch.as_tensor(hp.gamma, device=v.device, dtype=v.dtype)
    band_sharp = torch.as_tensor(hp.band_sharp, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    stag_margin = torch.as_tensor(hp.stag_margin, device=v.device, dtype=v.dtype).clamp_min(0.0)
    overshoot = torch.as_tensor(hp.overshoot, device=v.device, dtype=v.dtype).clamp_min(0.0)

    # xl gate in log-space
    xl_thr = torch.as_tensor(hp.xl_thr, device=v.device, dtype=v.dtype).clamp_min(1.0)
    xl_sharp = torch.as_tensor(hp.xl_sharp, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    log_xl = torch.log(xl.clamp_min(1.0))
    log_thr = torch.log(xl_thr)
    gate_xl = torch.sigmoid((log_xl - log_thr) / xl_sharp)

    # Band around gamma (kept as in design 44)
    lo = (gamma - stag_margin).clamp(0.0, 1.0)
    hi = (gamma + overshoot).clamp(0.0, 1.0)
    w_lo = torch.sigmoid((c - lo) / band_sharp)
    w_hi = torch.sigmoid((hi - c) / band_sharp)
    band = w_lo * w_hi

    # Upward-only push toward xs_star
    xs_star = torch.as_tensor(hp.xs_star, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    push_up = torch.relu(xs_star - xs) / xs_star

    # SINGLE CHANGE vs design 44 family: strict below-gamma cutoff (release == 0 for c>=gamma)
    diff = gamma - c
    below_gamma = torch.relu(diff) / (diff.abs() + 1e-6)

    release = hp.kappa * gate_xl * band * push_up * below_gamma

    rhs_xs = (xs + hp.epsilon) * (c - gamma) + release
    dxs = hp.beta * rhs_xs

    # Gradient scaling (baseline)
    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))
    dv  = -dv  * scale
    dxl = -dxl * scale
    dxs = -dxs * scale

    return {"v": dv, "xl": dxl, "xs": dxs}
