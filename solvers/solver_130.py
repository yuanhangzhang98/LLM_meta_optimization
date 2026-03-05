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
    "tau":     dict(type="log_uniform", default=0.10, low=1e-2, high=1.0),
    "lr":      dict(type="log_uniform", default=1.0,  low=1e-1, high=3.0),
}

# ------------------------------------------------------------------ #
# 3. _grad_single
# ------------------------------------------------------------------ #

def _grad_single(vars, idx, sgn, hp):
    v  = vars["v"]   # (N,)
    xl = vars["xl"]  # (M,)
    xs = vars["xs"]  # (M,)

    # Literals
    v_lit = v[idx] * sgn  # (M,3)

    # --- Keep hard clause cost (exactly zero at Boolean solutions) ---
    top_val, _ = torch.topk(v_lit, 2, dim=-1)   # (M,2)
    c = 0.5 * (1.0 - top_val[:, 0])            # (M,)

    # --- Smooth routing of forces (replace hard scatter/topk gating) ---
    tau = torch.as_tensor(hp.tau, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    w = torch.softmax(v_lit / tau, dim=-1)  # (M,3), smooth argmax weights

    u = 0.5 * (1.0 - v_lit)                 # (M,3) literal unsatisfaction

    # Smooth analogue of "use other literals' unsat on the most-satisfied literal"
    one_minus_w = 1.0 - w
    denom = one_minus_w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    u_other = (one_minus_w * u).sum(dim=-1, keepdim=True) / denom  # (M,1)

    # Baseline-like G: default c everywhere, but smoothly replace the best-literal entry with u_other
    G = c.unsqueeze(-1).expand_as(u) + w * (u_other - c.unsqueeze(-1))
    G = G * (xl * xs).unsqueeze(-1)

    # Baseline-like R: rigidity on (smoothly) most-satisfied literal(s)
    R = w * c.unsqueeze(-1)
    R = R * ((1.0 + hp.zeta * xl) * (1.0 - xs)).unsqueeze(-1)

    # Accumulate dv
    dv_clause = (G + R) * sgn
    dv = torch.zeros_like(v).scatter_add_(0, idx.reshape(-1), dv_clause.reshape(-1))

    # Memory dynamics (unchanged)
    dxl = hp.alpha * (c - hp.delta)
    dxs = hp.beta  * ((xs + hp.epsilon) * (c - hp.gamma))

    # Normalize
    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))

    dv  = -dv  * scale
    dxl = -dxl * scale
    dxs = -dxs * scale

    return {"v": dv, "xl": dxl, "xs": dxs}
