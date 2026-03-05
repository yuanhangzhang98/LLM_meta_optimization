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
    "alpha":   dict(type="log_uniform", default=5.0,  low=0.5,  high=50.0),
    "beta":    dict(type="log_uniform", default=20.0, low=2.0,  high=200.0),
    "gamma":   dict(type="uniform",     default=0.25, low=0.01, high=0.50),
    "delta":   dict(type="uniform",     default=0.05, low=0.01, high=0.50),
    "epsilon": dict(type="log_uniform", default=1e-3, low=1e-4, high=1e-2),
    "zeta":    dict(type="log_uniform", default=1e-3, low=1e-4, high=1.0),
    "lr":      dict(type="log_uniform", default=1.0,  low=1e-1, high=3.0),

    # Design-19 tau(c): sharp near satisfaction, more diffuse when far
    "tau_floor": dict(type="log_uniform", default=1e-3, low=1e-6, high=5e-2),
    "tau_span":  dict(type="log_uniform", default=3e-1, low=1e-4, high=1.0),
    "c0":        dict(type="uniform",     default=0.08, low=0.0,  high=0.45),

    # NEW: smoothness of the near-satisfaction gate for the 2nd rigidity route
    "tau_gate":  dict(type="log_uniform", default=2e-2, low=1e-4, high=2e-1),
}


# ------------------------------------------------------------------ #
# 3.  _grad_single
# ------------------------------------------------------------------ #

def _grad_single(vars, idx, sgn, hp):
    v  = vars["v"]
    xl = vars["xl"]
    xs = vars["xs"]

    # ===== CLAUSE EVALUATION (top-2) =====
    v_lit = v[idx] * sgn                               # (M,3)
    top_val, top_idx = torch.topk(v_lit, 2, dim=-1)    # (M,2)
    unsat_top = 0.5 * (1.0 - top_val)                  # (M,2)
    c = unsat_top[:, 0]                                # (M,)

    # ===== GRADIENT TERM G (baseline/top-2) =====
    G = c.unsqueeze(-1).repeat(1, 3)
    G.scatter_(1, top_idx[:, 0:1], unsat_top[:, 1:2])
    G *= (xl * xs).unsqueeze(-1)

    # ===== RIGIDITY TERM R (Design-19 routing, but smooth gate2) =====
    denom = (1.0 - hp.c0) + 1e-6
    t = torch.relu(c - hp.c0) / denom
    tau = (hp.tau_floor + hp.tau_span * t).clamp(min=1e-6)

    w = torch.softmax(-unsat_top / tau.unsqueeze(-1), dim=-1)  # (M,2)

    # Single change vs Design 19: smooth (sigmoid) gate on the 2nd route
    tau_gate = torch.as_tensor(hp.tau_gate, device=v.device, dtype=v.dtype).clamp(min=1e-6)
    gate2 = torch.sigmoid((hp.c0 - c) / tau_gate)              # ~1 when c<<c0, ~0 when c>>c0

    w2 = w[:, 1] * gate2
    w1 = 1.0 - w2

    R = torch.zeros_like(G)
    R.scatter_(1, top_idx[:, 0:1], (c * w1).unsqueeze(-1))
    R.scatter_add_(1, top_idx[:, 1:2], (c * w2).unsqueeze(-1))
    R *= ((1.0 + hp.zeta * xl) * (1.0 - xs)).unsqueeze(-1)

    # ===== ACCUMULATE v GRADIENT =====
    dv_clause = (G + R) * sgn
    dv = torch.zeros_like(v).scatter_add_(0, idx.reshape(-1), dv_clause.reshape(-1))

    # ===== AUXILIARY VARIABLE GRADIENTS (baseline) =====
    dxl = hp.alpha * (c - hp.delta)
    dxs = hp.beta  * ((xs + hp.epsilon) * (c - hp.gamma))

    # ===== NORMALIZATION (baseline) =====
    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))

    dv  = -dv  * scale
    dxl = -dxl * scale
    dxs = -dxs * scale

    return {"v": dv, "xl": dxl, "xs": dxs}
