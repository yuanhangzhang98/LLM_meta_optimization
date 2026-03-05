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

    # NEW: temperature for smooth clause/force construction (smaller -> closer to hard top-k/min)
    "tau":     dict(type="log_uniform", default=5e-2, low=5e-3, high=5e-1),
}

# ------------------------------------------------------------------ #
# 3.  _grad_single
# ------------------------------------------------------------------ #

def _grad_single(vars, idx, sgn, hp):
    v  = vars["v"]   # (N,)
    xl = vars["xl"]  # (M,)
    xs = vars["xs"]  # (M,)

    # Literals and per-literal unsatisfaction
    v_lit = v[idx] * sgn                      # (M,3)
    u = 0.5 * (1.0 - v_lit)                   # (M,3) in [0,1]

    # Smooth min(u) (clause cost), approximating baseline c = min(u)
    tau = hp.tau
    c = -tau * torch.logsumexp(-u / tau, dim=-1)  # (M,)
    c = c.clamp(0.0, 1.0)

    # Soft-argmin weights (focus on easiest/best literal)
    w_best = torch.softmax(-u / tau, dim=-1)  # (M,3)

    # Approximate baseline "best literal gets ~second-best push":
    # compute mean unsat of non-best literals (softly) and add it mainly to best literal.
    u_other_sum = (u * (1.0 - w_best)).sum(dim=-1)          # (M,)
    other_mass = (1.0 - w_best).sum(dim=-1).clamp(min=1e-6) # (M,)
    u_other_mean = u_other_sum / other_mass                 # (M,)

    push = c.unsqueeze(-1) + w_best * u_other_mean.unsqueeze(-1)  # (M,3)

    # Gradient and rigidity terms (same outer structure as baseline)
    G = push * (xl * xs).unsqueeze(-1)
    R = (c.unsqueeze(-1) * w_best) * ((1.0 + hp.zeta * xl) * (1.0 - xs)).unsqueeze(-1)

    # Accumulate variable forces
    dv_clause = (G + R) * sgn
    dv = torch.zeros_like(v).scatter_add_(0, idx.reshape(-1), dv_clause.reshape(-1))

    # Memory dynamics (unchanged, but driven by smooth c)
    dxl = hp.alpha * (c - hp.delta)
    dxs = hp.beta  * ((xs + hp.epsilon) * (c - hp.gamma))

    # Normalize
    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))

    return {
        "v":  -dv  * scale,
        "xl": -dxl * scale,
        "xs": -dxs * scale,
    }
