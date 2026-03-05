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
    "alpha":   dict(type="log_uniform", default=4.874231502465941,  low=0.5,  high=50.0),
    "beta":    dict(type="log_uniform", default=20.27800223245243, low=2.0,  high=200.0),
    "gamma":   dict(type="uniform",     default=0.24966754413660636, low=0.01, high=0.50),
    "delta":   dict(type="uniform",     default=0.0468352798448109, low=0.01, high=0.50),
    "epsilon": dict(type="log_uniform", default=0.0010008993995881722, low=1e-4, high=1e-2),
    "zeta":    dict(type="log_uniform", default=0.001613480742371504, low=1e-4, high=1.0),

    # Design-13 style assist (violation + margin gated), but with pressure-saturated amplitude
    "eta":     dict(type="uniform",     default=0.2509728141702429, low=0.0,  high=2.0),
    "c_thr":   dict(type="uniform",     default=0.5977716747691739, low=0.50, high=0.95),
    "m_thr":   dict(type="uniform",     default=0.34438940085214037, low=0.0,  high=2.0),
    "p_sat":   dict(type="log_uniform", default=21.00106719691547, low=0.1,  high=1e3),

    "lr":      dict(type="log_uniform", default=1.4404676031782013,  low=1e-1, high=3.0),
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

    # Top-2 routing (baseline)
    top_val, top_idx = torch.topk(v_lit, 2, dim=-1)  # (M,2)
    unsat_top = 0.5 * (1.0 - top_val)                # (M,2)
    c = unsat_top[:, 0]                              # (M,)

    # ===== GRADIENT TERM (baseline) =====
    G = c.unsqueeze(-1).repeat(1, 3)
    G.scatter_(1, top_idx[:, 0:1], unsat_top[:, 1:2])
    G *= (xl * xs).unsqueeze(-1)

    # ===== NEW: pressure-saturated margin-gated violation-triggered assist (runner-up literal) =====
    c_thr = torch.as_tensor(hp.c_thr, device=v.device, dtype=v.dtype)
    m_thr = torch.as_tensor(hp.m_thr, device=v.device, dtype=v.dtype)
    p_sat = torch.as_tensor(hp.p_sat, device=v.device, dtype=v.dtype).clamp_min(1e-6)

    assist_c = (c - c_thr).clamp_min(0.0)  # (M,)
    margin = (top_val[:, 0] - top_val[:, 1]).clamp_min(0.0)  # (M,)
    assist_m = (m_thr - margin).clamp_min(0.0)  # (M,)

    # Clause pressure and saturation
    p = (xl * xs).clamp_min(0.0)
    p_eff = p_sat * (p / (p + p_sat))  # bounded in [0, p_sat)

    assist = torch.zeros_like(G)
    assist.scatter_(1, top_idx[:, 1:2], 1.0)
    assist *= (hp.eta * assist_c * assist_m * p_eff).unsqueeze(-1)

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
