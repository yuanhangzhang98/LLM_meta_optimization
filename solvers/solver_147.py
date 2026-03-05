import torch

# ------------------------------------------------------------------ #
# 1. VARIABLES_SPEC
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
# 2. HYPER_SPACE
# ------------------------------------------------------------------ #
HYPER_SPACE = {
    "alpha":   dict(type="log_uniform", default=5.0,  low=0.5,  high=50.0),
    "beta":    dict(type="log_uniform", default=20.0, low=2.0,  high=200.0),
    "gamma":   dict(type="uniform",     default=0.25, low=0.01, high=0.50),
    "delta":   dict(type="uniform",     default=0.05, low=0.01, high=0.50),
    "epsilon": dict(type="log_uniform", default=1e-3, low=1e-4, high=1e-2),
    "zeta":    dict(type="log_uniform", default=1e-3, low=1e-4, high=1.0),

    # Design-5 ingredients
    "tau":     dict(type="log_uniform", default=0.05, low=1e-3, high=1.0),
    "kappa":   dict(type="log_uniform", default=2e-2, low=1e-3, high=2e-1),

    # NEW (single change vs Design-5): make the gate sharper when xl is large
    # kappa_eff = kappa / (1 + rho*log1p(xl))
    "rho":     dict(type="log_uniform", default=0.3,  low=1e-3, high=10.0),

    "lr":      dict(type="log_uniform", default=1.0, low=1e-1, high=3.0),
}


# ------------------------------------------------------------------ #
# 3. _grad_single
# ------------------------------------------------------------------ #
def _grad_single(vars, idx, sgn, hp):
    v  = vars["v"]
    xl = vars["xl"]
    xs = vars["xs"]

    # Literals and per-literal unsatisfaction u in [0,1]
    v_lit = v[idx] * sgn              # (M,3)
    u = 0.5 * (1.0 - v_lit)           # (M,3)

    # Hard clause cost
    c, best_pos = u.min(dim=-1)       # (M,), (M,)

    # Smooth responsibility: (soft) min of the OTHER TWO u's for each literal
    tau = torch.as_tensor(hp.tau, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    e = torch.exp(-u / tau)                          # (M,3)
    e_sum = e.sum(dim=-1, keepdim=True)              # (M,1)
    eu_sum = (e * u).sum(dim=-1, keepdim=True)       # (M,1)
    denom_other = (e_sum - e).clamp_min(1e-9)        # (M,3)
    num_other = (eu_sum - e * u)                     # (M,3)
    m_other = num_other / denom_other                # (M,3)

    # Violation gate with xl-adaptive sharpness (NEW)
    kappa = torch.as_tensor(hp.kappa, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    rho = torch.as_tensor(hp.rho, device=v.device, dtype=v.dtype).clamp_min(0.0)
    kappa_eff = kappa / (1.0 + rho * torch.log1p(xl)).clamp_min(1e-6)  # (M,)
    gate = torch.sigmoid((c - hp.gamma) / kappa_eff)                   # (M,)

    # G: gated smoothed push
    G = (xl * xs * gate).unsqueeze(-1) * m_other     # (M,3)

    # R: keep sharp/selective rigidity (baseline-like)
    R = torch.zeros_like(G)
    R.scatter_(1, best_pos.unsqueeze(-1), c.unsqueeze(-1))
    R *= ((1.0 + hp.zeta * xl) * (1.0 - xs)).unsqueeze(-1)

    # Accumulate into variable gradients
    dv_clause = (G + R) * sgn
    dv = torch.zeros_like(v).scatter_add_(0, idx.reshape(-1), dv_clause.reshape(-1))

    # Memory dynamics (baseline)
    dxl = hp.alpha * (c - hp.delta)
    dxs = hp.beta  * ((xs + hp.epsilon) * (c - hp.gamma))

    # Normalize for stability
    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))

    dv  = -dv  * scale
    dxl = -dxl * scale
    dxs = -dxs * scale

    return {"v": dv, "xl": dxl, "xs": dxs}
