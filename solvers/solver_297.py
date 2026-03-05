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

    # G-only gate controls
    "kappa":      dict(type="log_uniform", default=2e-2, low=1e-4, high=3e-1),
    "tie_margin": dict(type="uniform",     default=0.10, low=0.00, high=0.60),
    "tie_sharp":  dict(type="log_uniform", default=2e-2, low=1e-3, high=2e-1),

    "lr":      dict(type="log_uniform", default=1.0, low=1e-1, high=3.0),
}


# ------------------------------------------------------------------ #
# 3.  _grad_single
# ------------------------------------------------------------------ #

def _grad_single(vars, idx, sgn, hp):
    v  = vars["v"]
    xl = vars["xl"]
    xs = vars["xs"]

    v_lit = v[idx] * sgn  # (M,3)

    top_val, top_idx = torch.topk(v_lit, 2, dim=-1)  # (M,2)
    unsat_top = 0.5 * (1.0 - top_val)                # (M,2)
    c = unsat_top[:, 0].clamp(0.0, 1.0)              # (M,)

    # Tie-aware + violation-aware gate on G only
    margin = (top_val[:, 0] - top_val[:, 1]).clamp_min(0.0)

    tie_margin = torch.as_tensor(hp.tie_margin, device=v.device, dtype=v.dtype)
    tie_sharp  = torch.as_tensor(hp.tie_sharp,  device=v.device, dtype=v.dtype).clamp_min(1e-6)
    tie = torch.sigmoid((tie_margin - margin) / tie_sharp)  # ~1 near tie

    gamma = torch.as_tensor(hp.gamma, device=v.device, dtype=v.dtype)
    kappa = torch.as_tensor(hp.kappa, device=v.device, dtype=v.dtype).clamp_min(1e-8)
    dpos = torch.relu(c - gamma)
    viol = dpos / (dpos + kappa)  # 0 when c<=gamma, ->1 for deep violations

    gate = 1.0 - (1.0 - viol) * tie

    # G (make non-overlapping tensor before scatter_)
    G = c.unsqueeze(-1).repeat(1, 3)  # (M,3) materialized
    G.scatter_(1, top_idx[:, 0:1], unsat_top[:, 1:2])
    G *= (xl * xs * gate).unsqueeze(-1)

    # R: hard, ungated
    R = torch.zeros_like(G)
    R.scatter_(1, top_idx[:, 0:1], c.unsqueeze(-1))
    R *= ((1.0 + hp.zeta * xl) * (1.0 - xs)).unsqueeze(-1)

    dv_clause = (G + R) * sgn
    dv = torch.zeros_like(v).scatter_add_(0, idx.reshape(-1), dv_clause.reshape(-1))

    dxl = hp.alpha * (c - torch.as_tensor(hp.delta, device=v.device, dtype=v.dtype))
    dxs = hp.beta  * ((xs + hp.epsilon) * (c - gamma))

    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))
    dv  = -dv  * scale
    dxl = -dxl * scale
    dxs = -dxs * scale

    return {"v": dv, "xl": dxl, "xs": dxs}
