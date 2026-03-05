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
    "alpha":   dict(type="log_uniform", default=5.0,   low=0.5,  high=50.0),
    "beta":    dict(type="log_uniform", default=20.0,  low=2.0,  high=200.0),
    "gamma":   dict(type="uniform",     default=0.25,  low=0.01, high=0.50),
    "delta":   dict(type="uniform",     default=0.05,  low=0.01, high=0.50),
    "epsilon": dict(type="log_uniform", default=1e-3,  low=1e-4, high=1e-2),
    "zeta":    dict(type="log_uniform", default=1e-3,  low=1e-4, high=1.0),

    # xs anti-lock (xl-gated weak-sat hysteresis bump)
    "kappa":    dict(type="uniform",     default=0.3,   low=0.0,  high=2.0),
    "xl_thr":   dict(type="log_uniform", default=50.0,  low=1.0,  high=1e4),
    "xl_sharp": dict(type="uniform",     default=1.0,   low=0.1,  high=5.0),

    "lr":      dict(type="log_uniform", default=1.0,   low=1e-1, high=3.0),
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

    # ===== BASELINE HARD TOP-2 BACKBONE (UNCHANGED) =====
    top_val, top_idx = torch.topk(v_lit, 2, dim=-1)          # (M,2)
    unsat_top = 0.5 * (1.0 - top_val)                        # (M,2)
    c = unsat_top[:, 0].clamp(0.0, 1.0)                      # (M,)

    G = c.unsqueeze(-1).repeat(1, 3)
    G.scatter_(1, top_idx[:, 0:1], unsat_top[:, 1:2])
    G *= (xl * xs).unsqueeze(-1)

    R = torch.zeros_like(G)
    R.scatter_(1, top_idx[:, 0:1], c.unsqueeze(-1))
    R *= ((1.0 + hp.zeta * xl) * (1.0 - xs)).unsqueeze(-1)

    dv_clause = (G + R) * sgn
    dv = torch.zeros_like(v).scatter_add_(0, idx.reshape(-1), dv_clause.reshape(-1))

    # ===== MEMORY UPDATES =====
    dxl = hp.alpha * (c - hp.delta)

    # Baseline xs + xl-gated weak-sat hysteresis bump (ONLY change)
    gamma = torch.as_tensor(hp.gamma, device=v.device, dtype=v.dtype)
    xl_thr = torch.as_tensor(hp.xl_thr, device=v.device, dtype=v.dtype).clamp_min(1.0)
    xl_sharp = torch.as_tensor(hp.xl_sharp, device=v.device, dtype=v.dtype).clamp_min(1e-6)

    log_xl = torch.log(xl.clamp_min(1.0))
    log_thr = torch.log(xl_thr)
    gate_xl = torch.sigmoid((log_xl - log_thr) / xl_sharp)   # ~1 if xl >> xl_thr

    weak_sat = torch.relu(gamma - c)                          # >0 only when c<gamma
    bump = hp.kappa * gate_xl * c * weak_sat * (1.0 - xs)     # vanishes at c=0, c>=gamma, or xs=1

    rhs_xs = (xs + hp.epsilon) * (c - gamma) + bump
    dxs = hp.beta * rhs_xs

    # ===== GRADIENT SCALING =====
    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))

    dv  = -dv  * scale
    dxl = -dxl * scale
    dxs = -dxs * scale

    return {"v": dv, "xl": dxl, "xs": dxs}
