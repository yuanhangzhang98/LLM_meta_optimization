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
    # Backbone (seeded from the strong tuned region used in recent designs)
    "alpha":   dict(type="log_uniform", default=14.504000467173922, low=0.5,  high=50.0),
    "beta":    dict(type="log_uniform", default=22.87679950018518,  low=2.0,  high=200.0),
    "gamma":   dict(type="uniform",     default=0.3713393245315345, low=0.01, high=0.50),
    "delta":   dict(type="uniform",     default=0.13541660385008228,low=0.01, high=0.50),
    "epsilon": dict(type="log_uniform", default=0.001132926021204599,low=1e-4,high=1e-2),
    "zeta":    dict(type="log_uniform", default=0.013885962445043169,low=1e-4,high=1.0),

    # SINGLE NEW KNOB: G-only pressure soft-cap
    "p_cut":   dict(type="log_uniform", default=5e4,                low=1e2,  high=1e6),

    "lr":      dict(type="log_uniform", default=2.1085465692572183, low=1e-1, high=3.0),
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

    # Top-2 backbone
    top_val, top_idx = torch.topk(v_lit, 2, dim=-1)   # (M,2)
    unsat_top = 0.5 * (1.0 - top_val)                 # (M,2)
    c = unsat_top[:, 0].clamp(0.0, 1.0)               # (M,)

    # G routing
    G = c.unsqueeze(-1).repeat(1, 3)
    G.scatter_(1, top_idx[:, 0:1], unsat_top[:, 1:2])

    # --- ONLY CHANGE vs baseline: soft-cap pressure inside G (do NOT touch R) ---
    pressure = (xl * xs).clamp_min(0.0)
    p_cut = torch.as_tensor(hp.p_cut, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    pressure_eff = p_cut * torch.tanh(pressure / p_cut)  # ~pressure when small; saturates at p_cut
    G = G * pressure_eff.unsqueeze(-1)

    # Rigidity (unchanged)
    R = torch.zeros_like(G)
    R.scatter_(1, top_idx[:, 0:1], c.unsqueeze(-1))
    R *= ((1.0 + hp.zeta * xl) * (1.0 - xs)).unsqueeze(-1)

    # Accumulate dv
    dv_clause = (G + R) * sgn
    dv = torch.zeros_like(v).scatter_add_(0, idx.reshape(-1), dv_clause.reshape(-1))

    # Memory dynamics (unchanged)
    dxl = hp.alpha * (c - hp.delta)
    dxs = hp.beta  * ((xs + hp.epsilon) * (c - hp.gamma))

    # Normalize
    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))
    dv  = -dv  * scale
    dxl = -dxl * scale
    dxs = -dxs * scale

    return {"v": dv, "xl": dxl, "xs": dxs}
