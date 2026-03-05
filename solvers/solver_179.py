import torch

# ------------------------------------------------------------------ #
# 1.  VARIABLES_SPEC - State variable specification                  #
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
# 2.  HYPER_SPACE - Hyperparameter search space                      #
# ------------------------------------------------------------------ #

HYPER_SPACE = {
    "alpha":   dict(type="log_uniform", default=5.0,  low=0.5,  high=50.0),
    "beta":    dict(type="log_uniform", default=20.0, low=2.0,  high=200.0),
    "gamma":   dict(type="uniform",     default=0.25, low=0.01, high=0.50),
    "delta":   dict(type="uniform",     default=0.05, low=0.01, high=0.50),
    "epsilon": dict(type="log_uniform", default=1e-3, low=1e-4, high=1e-2),
    "zeta":    dict(type="log_uniform", default=1e-3, low=1e-4, high=1.0),

    # NEW: temperature for softmin/soft routing (smaller = closer to hard min/topk)
    "tau":     dict(type="log_uniform", default=5e-2, low=1e-3, high=5e-1),

    "lr":      dict(type="log_uniform", default=1.0, low=1e-1, high=3.0),
}


# ------------------------------------------------------------------ #
# 3.  _grad_single - Per-instance dynamics function                  #
# ------------------------------------------------------------------ #

def _grad_single(vars, idx, sgn, hp):
    v  = vars["v"]   # (N,)
    xl = vars["xl"]  # (M,)
    xs = vars["xs"]  # (M,)

    # Literal values in [-1, 1]
    v_lit = v[idx] * sgn  # (M,3)

    # Unsatisfaction per literal in [0,1]
    u = 0.5 * (1.0 - v_lit)  # (M,3)

    # Softmin helper (log-sum-exp)
    tau = torch.as_tensor(hp.tau, device=v.device, dtype=v.dtype).clamp(min=1e-4)

    def softmin(x, dim=-1):
        return -tau * torch.logsumexp(-x / tau, dim=dim)

    # Clause cost c ~= min(u)
    c = softmin(u, dim=-1)  # (M,)

    # For each literal i, compute min over the other two literals (softly)
    # u_masked[m,i,j] = u[m,j] but with j==i excluded via large penalty
    eye = torch.eye(3, device=v.device, dtype=v.dtype).unsqueeze(0)  # (1,3,3)
    big = torch.as_tensor(1e6, device=v.device, dtype=v.dtype)
    u_masked = u.unsqueeze(-2).expand(-1, 3, -1) + eye * big  # (M,3,3)
    min_other = softmin(u_masked, dim=-1)  # (M,3)

    # Smooth "winner" responsibilities for rigidity routing
    p_win = torch.softmax(v_lit / tau, dim=-1)  # (M,3)

    # Gradient term (drives violated clauses), now fully smooth
    G = min_other * (xl * xs).unsqueeze(-1)  # (M,3)

    # Rigidity term (stabilizes satisfied clauses), smoothly routed
    R = (p_win * c.unsqueeze(-1)) * ((1.0 + hp.zeta * xl) * (1.0 - xs)).unsqueeze(-1)  # (M,3)

    # Accumulate v gradient
    dv_clause = (G + R) * sgn  # (M,3)
    dv = torch.zeros_like(v).scatter_add_(0, idx.reshape(-1), dv_clause.reshape(-1))

    # Memory variable gradients (unchanged, but driven by smooth c)
    dxl = hp.alpha * (c - hp.delta)
    dxs = hp.beta  * ((xs + hp.epsilon) * (c - hp.gamma))

    # Normalize
    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))

    dv  = -dv  * scale
    dxl = -dxl * scale
    dxs = -dxs * scale

    return {"v": dv, "xl": dxl, "xs": dxs}
