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

    # NEW: smoothing temperature for soft responsibility (smaller -> closer to hard topk/min)
    "tau":     dict(type="log_uniform", default=0.05, low=1e-3, high=1.0),

    "lr":      dict(type="log_uniform", default=1.0, low=1e-1, high=3.0),
}


# ------------------------------------------------------------------ #
# 3.  _grad_single
# ------------------------------------------------------------------ #
def _grad_single(vars, idx, sgn, hp):
    v  = vars["v"]   # (N,)
    xl = vars["xl"]  # (M,)
    xs = vars["xs"]  # (M,)

    # Literals
    v_lit = v[idx] * sgn  # (M,3) in [-1,1]

    # Unsatisfaction per literal: u=0 means fully satisfied literal, u=1 means fully violated
    u = 0.5 * (1.0 - v_lit)  # (M,3) in [0,1]

    # Smooth responsibility for the (approx) best literal (i.e., smallest u)
    tau = torch.as_tensor(hp.tau, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    w = torch.softmax(-u / tau, dim=-1)  # (M,3)

    # Smooth clause cost (soft-min of u)
    c = (w * u).sum(dim=-1)  # (M,) in [0,1]

    # Allocate push/rigidity continuously (removes hard topk/scatter discontinuities)
    G = (xl * xs).unsqueeze(-1) * (w * c.unsqueeze(-1))
    R = ((1.0 + hp.zeta * xl) * (1.0 - xs)).unsqueeze(-1) * (w * c.unsqueeze(-1))

    dv_clause = (G + R) * sgn  # (M,3)
    dv = torch.zeros_like(v).scatter_add_(0, idx.reshape(-1), dv_clause.reshape(-1))

    dxl = hp.alpha * (c - hp.delta)
    dxs = hp.beta  * ((xs + hp.epsilon) * (c - hp.gamma))

    # Normalize for stability
    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))

    dv  = -dv  * scale
    dxl = -dxl * scale
    dxs = -dxs * scale

    return {"v": dv, "xl": dxl, "xs": dxs}
