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
    "alpha":   dict(type="log_uniform", default=6.152476925471209,  low=0.5,  high=50.0),
    "beta":    dict(type="log_uniform", default=20.238004765813734, low=2.0,  high=200.0),
    "gamma":   dict(type="uniform",     default=0.235783292566568, low=0.01, high=0.50),
    "delta":   dict(type="uniform",     default=0.05198068916797638, low=0.01, high=0.50),
    "epsilon": dict(type="log_uniform", default=0.0002797593318319248, low=1e-4, high=1e-2),
    "zeta":    dict(type="log_uniform", default=0.00422732358493675, low=1e-4, high=1.0),

    # NEW: Top-2 assist, but with pressure-saturated amplitude for scale safety
    "eta":     dict(type="uniform",     default=1.634840965270996, low=0.0,  high=2.0),
    "c_thr":   dict(type="uniform",     default=0.526580006080825, low=0.50, high=0.95),
    "p_sat":   dict(type="log_uniform", default=525.4056602340823, low=0.3,  high=1e3),

    "lr":      dict(type="log_uniform", default=0.794278492030132,  low=1e-1, high=3.0),
}


# ------------------------------------------------------------------ #
# 3.  _grad_single
# ------------------------------------------------------------------ #

def _grad_single(vars, idx, sgn, hp):
    v  = vars["v"]
    xl = vars["xl"]
    xs = vars["xs"]

    # Literals
    v_lit = v[idx] * sgn  # (M,3)

    # Top-2 routing (baseline)
    top_val, top_idx = torch.topk(v_lit, 2, dim=-1)  # (M,2)
    unsat_top = 0.5 * (1.0 - top_val)                # (M,2)
    c = unsat_top[:, 0]                              # (M,)

    # ===== GRADIENT TERM (baseline) =====
    G = c.unsqueeze(-1).repeat(1, 3)
    G.scatter_(1, top_idx[:, 0:1], unsat_top[:, 1:2])
    pressure = (xl * xs).clamp_min(0.0)              # (M,)
    G *= pressure.unsqueeze(-1)

    # ===== NEW: runner-up assist with pressure saturation =====
    c_thr = torch.as_tensor(hp.c_thr, device=v.device, dtype=v.dtype)
    p_sat = torch.as_tensor(hp.p_sat, device=v.device, dtype=v.dtype).clamp_min(1e-6)

    violated = (c > c_thr).to(v.dtype)  # (M,)

    # Saturated pressure: bounded in [0, p_sat)
    p_eff = p_sat * (pressure / (pressure + p_sat))

    assist = torch.zeros_like(G)
    assist.scatter_(1, top_idx[:, 1:2], 1.0)  # runner-up literal
    assist *= (hp.eta * c * violated * p_eff).unsqueeze(-1)

    G = G + assist

    # ===== RIGIDITY TERM (baseline) =====
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
