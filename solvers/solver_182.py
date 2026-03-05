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
    "lr":      dict(type="log_uniform", default=1.0, low=1e-1, high=3.0),

    # Mild per-variable preconditioning exponent (0 = baseline, 1 = full degree norm)
    "deg_pow": dict(type="uniform", default=0.5, low=0.0, high=1.0),
}


# ------------------------------------------------------------------ #
# 3.  _grad_single
# ------------------------------------------------------------------ #

def _grad_single(vars, idx, sgn, hp):
    v  = vars["v"]   # (N,)
    xl = vars["xl"]  # (M,)
    xs = vars["xs"]  # (M,)

    # Clause literal values
    v_lit = v[idx] * sgn  # (M,3)

    # Top-2 literals per clause
    top_val, top_idx = torch.topk(v_lit, 2, dim=-1)

    # Clause cost proxy
    unsat_top = 0.5 * (1.0 - top_val)  # (M,2)
    c = unsat_top[:, 0]                # (M,)

    # Gradient term
    G = c.unsqueeze(-1).repeat(1, 3)
    G.scatter_(1, top_idx[:, 0:1], unsat_top[:, 1:2])
    G *= (xl * xs).unsqueeze(-1)

    # Rigidity term
    R = torch.zeros_like(G)
    R.scatter_(1, top_idx[:, 0:1], c.unsqueeze(-1))
    R *= ((1.0 + hp.zeta * xl) * (1.0 - xs)).unsqueeze(-1)

    # Accumulate v gradient
    dv_clause = (G + R) * sgn
    dv_raw = torch.zeros_like(v).scatter_add_(0, idx.reshape(-1), dv_clause.reshape(-1))

    # --- SINGLE MODIFICATION: hybrid global+local normalization for dv ---
    deg = torch.zeros_like(v).scatter_add_(
        0,
        idx.reshape(-1),
        torch.ones(idx.numel(), device=v.device, dtype=v.dtype),
    )
    dv = dv_raw / (deg.clamp(min=1.0) ** hp.deg_pow)

    # Auxiliary gradients (baseline)
    dxl = hp.alpha * (c - hp.delta)
    dxs = hp.beta  * ((xs + hp.epsilon) * (c - hp.gamma))

    # Baseline global scaling (kept)
    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))

    dv  = -dv  * scale
    dxl = -dxl * scale
    dxs = -dxs * scale

    return {"v": dv, "xl": dxl, "xs": dxs}
