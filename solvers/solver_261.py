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
    "alpha":   dict(type="log_uniform", default=6.150410035043752,    low=0.5,  high=50.0),
    "beta":    dict(type="log_uniform", default=26.263877963908627,   low=2.0,  high=200.0),
    "gamma":   dict(type="uniform",     default=0.23600000143051147,  low=0.01, high=0.50),
    "delta":   dict(type="uniform",     default=0.061686870723080774, low=0.01, high=0.50),
    "epsilon": dict(type="log_uniform", default=0.0003905747409621244, low=1e-4, high=1e-2),
    "zeta":    dict(type="log_uniform", default=0.004146892632731432, low=1e-4, high=1.0),

    # Assist (as in 177)
    "eta":     dict(type="uniform",     default=1.6247211516180162,   low=0.0,   high=2.0),
    "c_thr":   dict(type="uniform",     default=0.5322750390119185,   low=0.50,  high=0.95),
    "w":       dict(type="uniform",     default=0.014999999664723873, low=0.005, high=0.25),
    "p_sat":   dict(type="log_uniform", default=497.9778220137977,    low=0.3,   high=1e3),

    # Tail damping threshold (as in 177)
    "p_cut":   dict(type="log_uniform", default=49999.98119564294,    low=1e3,   high=1e6),

    # NEW (single change vs 177): tune tail-damping exponent (nu=2 recovers 177)
    "nu":      dict(type="uniform",     default=2.0,                 low=1.0,   high=6.0),

    "lr":      dict(type="log_uniform", default=0.8407911214624898,   low=1e-1,  high=3.0),
}


# ------------------------------------------------------------------ #
# 3.  _grad_single
# ------------------------------------------------------------------ #

def _grad_single(vars, idx, sgn, hp):
    v  = vars["v"]   # (N,)
    xl = vars["xl"]  # (M,)
    xs = vars["xs"]  # (M,)

    # Literals
    v_lit = v[idx] * sgn  # (M,3)

    # Top-2 routing
    top_val, top_idx = torch.topk(v_lit, 2, dim=-1)  # (M,2)
    unsat_top = 0.5 * (1.0 - top_val)                # (M,2)
    c = unsat_top[:, 0]                              # (M,)

    # Baseline Top-2 gradient term
    G = c.unsqueeze(-1).repeat(1, 3)
    G.scatter_(1, top_idx[:, 0:1], unsat_top[:, 1:2])

    pressure = (xl * xs).clamp_min(0.0)              # (M,)
    G *= pressure.unsqueeze(-1)

    # Assist: smooth violation gate + peaked effective pressure + high-pressure tail damping
    c_thr = torch.as_tensor(hp.c_thr, device=v.device, dtype=v.dtype)
    w     = torch.as_tensor(hp.w,     device=v.device, dtype=v.dtype).clamp_min(1e-6)
    p_sat = torch.as_tensor(hp.p_sat, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    p_cut = torch.as_tensor(hp.p_cut, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    nu    = torch.as_tensor(hp.nu,    device=v.device, dtype=v.dtype).clamp_min(1e-3)

    z = ((c - c_thr) / w).clamp(-20.0, 20.0)
    gate = torch.sigmoid(z)  # (M,)

    denom = (pressure + p_sat).clamp_min(1e-6)
    # Peaked p_eff: ~p for small p, peaks near p~p_sat, then decays ~ p_sat^2/p
    p_eff = (p_sat * pressure / denom) * (p_sat / denom)

    # Tail damper: 1 / (1 + (pressure/p_cut)^nu). nu=2 matches design 177.
    r = (pressure / p_cut).clamp_min(0.0)
    damp = 1.0 / (1.0 + r.pow(nu))
    p_eff = p_eff * damp

    assist = torch.zeros_like(G)
    assist.scatter_(1, top_idx[:, 1:2], 1.0)  # runner-up literal
    assist *= (hp.eta * c * gate * p_eff).unsqueeze(-1)

    G = G + assist

    # Baseline rigidity term
    R = torch.zeros_like(G)
    R.scatter_(1, top_idx[:, 0:1], c.unsqueeze(-1))
    R *= ((1.0 + hp.zeta * xl) * (1.0 - xs)).unsqueeze(-1)

    # Accumulate v gradient
    dv_clause = (G + R) * sgn
    dv = torch.zeros_like(v).scatter_add_(0, idx.reshape(-1), dv_clause.reshape(-1))

    # Memory updates (baseline)
    dxl = hp.alpha * (c - hp.delta)
    dxs = hp.beta  * ((xs + hp.epsilon) * (c - hp.gamma))

    # Normalize (baseline)
    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))

    dv  = -dv  * scale
    dxl = -dxl * scale
    dxs = -dxs * scale

    return {"v": dv, "xl": dxl, "xs": dxs}
