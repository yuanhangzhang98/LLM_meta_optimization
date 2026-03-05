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
    "lr":      dict(type="log_uniform", default=1.0,   low=1e-1, high=3.0),

    # Smooth routing / softmin temperatures
    "tau_sel": dict(type="log_uniform", default=5e-2,  low=5e-3, high=5e-1),
    "tau_min": dict(type="log_uniform", default=2e-2,  low=1e-3, high=2e-1),

    # Bump size to turn softmin into a smooth second-min surrogate (>=1 makes c2 ~ 2nd-min at solutions)
    "bump":    dict(type="uniform",     default=2.0,   low=1.0,  high=5.0),
}


def _softmin(x, tau, dim=-1):
    # softmin(x) = -tau * logsumexp(-x/tau)
    return -tau * torch.logsumexp(-x / tau, dim=dim)


# ------------------------------------------------------------------ #
# 3.  _grad_single
# ------------------------------------------------------------------ #

def _grad_single(vars, idx, sgn, hp):
    v  = vars["v"]   # (N,)
    xl = vars["xl"]  # (M,)
    xs = vars["xs"]  # (M,)

    # Literals and per-literal unsatisfaction u in [0,1]
    v_lit = v[idx] * sgn                      # (M,3)
    u = 0.5 * (1.0 - v_lit)                   # (M,3)

    # Soft clause cost: smooth min over literals (baseline uses hard min via topk)
    c = _softmin(u, hp.tau_min, dim=-1)       # (M,)

    # Soft "best literal" selector (peaked on smallest u)
    p = torch.softmax(-u / hp.tau_sel, dim=-1)  # (M,3)

    # Smooth second-min surrogate: bump the best literal, then softmin again
    u_bumped = u + hp.bump * p
    c2 = _softmin(u_bumped, hp.tau_min, dim=-1)  # (M,)

    # Keep costs in a reasonable range (matches baseline scale expectations)
    c = c.clamp(0.0, 1.0)
    c2 = c2.clamp(0.0, 1.0)

    # Gradient term: all literals get ~c, best literal gets boosted toward ~c2
    G = c[:, None] + (c2 - c)[:, None] * p
    G = G * (xl * xs)[:, None]

    # Rigidity term: routed to best literal smoothly
    R = (c[:, None] * p) * ((1.0 + hp.zeta * xl) * (1.0 - xs))[:, None]

    # Accumulate variable gradients
    dv_clause = (G + R) * sgn
    dv = torch.zeros_like(v).scatter_add_(0, idx.reshape(-1), dv_clause.reshape(-1))

    # Auxiliary dynamics (unchanged)
    dxl = hp.alpha * (c - hp.delta)
    dxs = hp.beta  * ((xs + hp.epsilon) * (c - hp.gamma))

    # Normalize (unchanged)
    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))

    dv  = -dv  * scale
    dxl = -dxl * scale
    dxs = -dxs * scale

    return {"v": dv, "xl": dxl, "xs": dxs}
