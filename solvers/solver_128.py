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

    # Base temperature for memory softmin (Designs 5/15 style), but now made clause-adaptive.
    "tau":         dict(type="log_uniform", default=5e-3, low=1e-4, high=2e-1),

    # Controls how quickly temperature cools as the gap (u2-u1) grows.
    "gap_scale":   dict(type="log_uniform", default=1e-1, low=1e-2, high=1.0),
}


# ------------------------------------------------------------------ #
# 3.  _grad_single
# ------------------------------------------------------------------ #

def _grad_single(vars, idx, sgn, hp):
    v  = vars["v"]   # (N,)
    xl = vars["xl"]  # (M,)
    xs = vars["xs"]  # (M,)

    # ----- Baseline hard dv dynamics (UNCHANGED) -----
    v_lit = v[idx] * sgn  # (M,3)

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

    # ----- ONLY CHANGE: adaptive-temperature softmin for memory signal c_mem -----
    # per-literal unsatisfaction u in [0,1]
    u = 0.5 * (1.0 - v_lit)  # (M,3)

    # gap between best and second-best literals (small gap => tie => hotter smoothing)
    u2 = torch.topk(u, 2, dim=-1, largest=False).values  # (M,2)
    gap = (u2[:, 1] - u2[:, 0]).clamp(min=0.0, max=1.0) # (M,)

    tau0 = max(float(hp.tau), 1e-6)
    gs   = max(float(hp.gap_scale), 1e-6)

    # tau_eff in [tau0, 10*tau0], hot near ties, cool when a clear winner exists
    tie_boost = torch.exp(-gap / gs)                     # (M,) in (0,1]
    tau_eff = (tau0 * (1.0 + 9.0 * tie_boost)).clamp(min=1e-6, max=1.0)  # (M,)

    w = torch.softmax(-u / tau_eff.unsqueeze(-1), dim=-1)  # (M,3)
    c_mem = (w * u).sum(dim=-1)                            # (M,)

    dxl = hp.alpha * (c_mem - hp.delta)
    dxs = hp.beta  * ((xs + hp.epsilon) * (c_mem - hp.gamma))

    # baseline normalization
    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))

    dv  = -dv  * scale
    dxl = -dxl * scale
    dxs = -dxs * scale

    return {"v": dv, "xl": dxl, "xs": dxs}
