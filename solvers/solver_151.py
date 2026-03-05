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
    # Temperature for smooth clause operators (tau -> 0 recovers hard min/top-1 routing)
    "tau":     dict(type="log_uniform", default=5e-2, low=1e-3, high=5e-1),
    "lr":      dict(type="log_uniform", default=1.0,  low=1e-1, high=3.0),
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

    # Unsatisfaction per literal u in [0,1]
    u = 0.5 * (1.0 - v_lit)  # (M,3)

    # --- Smooth operators ---
    tau = torch.as_tensor(hp.tau, device=v.device, dtype=v.dtype).clamp(min=1e-6)

    # softmin(u) = -tau * logsumexp(-u/tau)
    c = -tau * torch.logsumexp((-u) / tau, dim=-1)  # (M,)

    def _softmin2(a, b):
        ab = torch.stack(((-a) / tau, (-b) / tau), dim=-1)
        return -tau * torch.logsumexp(ab, dim=-1)

    # For each position p, approximate min over the other two literals
    u_min_other = torch.stack(
        (
            _softmin2(u[:, 1], u[:, 2]),
            _softmin2(u[:, 0], u[:, 2]),
            _softmin2(u[:, 0], u[:, 1]),
        ),
        dim=-1,
    )  # (M,3)

    # Smooth "winner" mask for rigidity term (approaches one-hot at argmax(v_lit))
    w = torch.softmax(v_lit / tau, dim=-1)  # (M,3)

    # Gradient and rigidity terms (same weighting structure as baseline)
    G = u_min_other * (xl * xs).unsqueeze(-1)  # (M,3)
    R = (w * c.unsqueeze(-1)) * ((1.0 + hp.zeta * xl) * (1.0 - xs)).unsqueeze(-1)  # (M,3)

    dv_clause = (G + R) * sgn
    dv = torch.zeros_like(v).scatter_add_(0, idx.reshape(-1), dv_clause.reshape(-1))

    dxl = hp.alpha * (c - hp.delta)
    dxs = hp.beta * ((xs + hp.epsilon) * (c - hp.gamma))

    # Normalize (as in baseline)
    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))

    dv  = -dv  * scale
    dxl = -dxl * scale
    dxs = -dxs * scale

    return {"v": dv, "xl": dxl, "xs": dxs}
