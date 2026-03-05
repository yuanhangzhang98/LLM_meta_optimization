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

    # Design-7 tie-only smoothing for the monitor used in dxs
    "tau_tie": dict(type="log_uniform", default=2e-2, low=1e-3, high=2e-1),
    "g0":      dict(type="log_uniform", default=2e-2, low=1e-3, high=2e-1),
    "kappa":   dict(type="log_uniform", default=1e-2, low=1e-3, high=1e-1),

    # NEW: deadband half-width around gamma for dxs to suppress chattering
    "h_band":  dict(type="log_uniform", default=1e-2, low=1e-4, high=2e-1),
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

    # --- Baseline hard top-k structure for v-updates (unchanged) ---
    top_val, top_idx = torch.topk(v_lit, 2, dim=-1)  # (M,2)
    unsat_top = 0.5 * (1.0 - top_val)                # (M,2)
    c_hard = unsat_top[:, 0]                         # (M,)

    G = c_hard.unsqueeze(-1).repeat(1, 3)
    G.scatter_(1, top_idx[:, 0:1], unsat_top[:, 1:2])
    G *= (xl * xs).unsqueeze(-1)

    R = torch.zeros_like(G)
    R.scatter_(1, top_idx[:, 0:1], c_hard.unsqueeze(-1))
    R *= ((1.0 + hp.zeta * xl) * (1.0 - xs)).unsqueeze(-1)

    dv_clause = (G + R) * sgn
    dv = torch.zeros_like(v).scatter_add_(0, idx.reshape(-1), dv_clause.reshape(-1))

    # --- Memory monitor for dxs: hard-min almost everywhere, soft only near ties ---
    u = 0.5 * (1.0 - v_lit)  # (M,3) in [0,1]
    u12, _ = torch.topk(u, 2, dim=-1, largest=False)
    u1 = u12[:, 0]
    gap = (u12[:, 1] - u12[:, 0]).clamp(min=0.0)

    w = torch.sigmoid((hp.g0 - gap) / hp.kappa).clamp(0.0, 1.0)  # tie gate

    tau = hp.tau_tie
    c_soft = (-tau * torch.logsumexp(-u / tau, dim=-1)).clamp(0.0, 1.0)
    c_mem = ((1.0 - w) * u1 + w * c_soft).clamp(0.0, 1.0)

    # Long-term memory stays driven by hard monitor
    dxl = hp.alpha * (c_hard - hp.delta)

    # NEW: deadband/hysteresis around gamma to reduce xs chattering near threshold
    d = c_mem - hp.gamma
    gate = torch.sigmoid((d.abs() - hp.h_band) / hp.kappa)  # ~0 inside band, ~1 outside
    d_eff = d * gate

    dxs = hp.beta * ((xs + hp.epsilon) * d_eff)

    # Normalize (baseline style)
    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))

    return {
        "v":  -dv  * scale,
        "xl": -dxl * scale,
        "xs": -dxs * scale,
    }
