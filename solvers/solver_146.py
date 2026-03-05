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

    # soft responsibility temperature (smaller -> closer to hard min)
    "tau":     dict(type="log_uniform", default=0.05, low=1e-3, high=1.0),

    # violation gate sharpness
    "kappa":   dict(type="log_uniform", default=2e-2, low=1e-3, high=2e-1),

    # NEW (single principled change vs Design 5): nonzero floor for the gate
    # gate = eta + (1-eta)*sigmoid((c-gamma)/kappa)
    "eta":     dict(type="uniform",     default=0.02, low=0.0,  high=0.2),

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

    # Hard clause cost and best literal position
    c, best_pos = u.min(dim=-1)       # (M,), (M,)

    # Smooth responsibility: estimate (soft) min of the OTHER TWO u's for each literal
    tau = torch.as_tensor(hp.tau, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    e = torch.exp(-u / tau)                          # (M,3)
    e_sum = e.sum(dim=-1, keepdim=True)              # (M,1)
    eu_sum = (e * u).sum(dim=-1, keepdim=True)       # (M,1)
    denom_other = (e_sum - e).clamp_min(1e-9)        # (M,3)
    num_other = (eu_sum - e * u)                     # (M,3)
    m_other = num_other / denom_other                # (M,3)

    # Violation gate with floor (NEW)
    kappa = torch.as_tensor(hp.kappa, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    gate_raw = torch.sigmoid((c - hp.gamma) / kappa)  # (M,)
    eta = torch.as_tensor(hp.eta, device=v.device, dtype=v.dtype).clamp(0.0, 0.999)
    gate = eta + (1.0 - eta) * gate_raw               # (M,)

    # G: gated smoothed push; still multiplied by (xl*xs) to preserve satisfied-clause silence
    G = (xl * xs * gate).unsqueeze(-1) * m_other      # (M,3)

    # R: keep sharp/selective rigidity
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
