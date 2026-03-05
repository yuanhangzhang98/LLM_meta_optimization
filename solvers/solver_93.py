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

    # Top-2 smoothing (ONLY when ambiguous and far from satisfied)
    # w = softmax(-unsat_top / tau_sel)
    "tau_sel": dict(type="log_uniform", default=2e-2, low=1e-4, high=2e-1),

    # Gap gate: enable smoothing when (unsat2 - unsat1) is small
    "gap0":    dict(type="uniform",     default=5e-2, low=1e-3, high=2e-1),
    "tau_gap": dict(type="log_uniform", default=2e-2, low=1e-4, high=2e-1),

    # Near-solution hardening: disable smoothing when c is small
    "c_hard":  dict(type="uniform",     default=0.12, low=0.02, high=0.30),
    "tau_c":   dict(type="log_uniform", default=2e-2, low=1e-4, high=2e-1),
}


# ------------------------------------------------------------------ #
# 3.  _grad_single
# ------------------------------------------------------------------ #

def _grad_single(vars, idx, sgn, hp):
    v  = vars["v"]   # (N,)
    xl = vars["xl"]  # (M,)
    xs = vars["xs"]  # (M,)

    # Literals
    v_lit = v[idx] * sgn                      # (M,3)

    # Baseline hard top-2 (most satisfied)
    top_val, top_idx = torch.topk(v_lit, 2, dim=-1)   # (M,2)
    unsat_top = 0.5 * (1.0 - top_val)                 # (M,2)
    c = unsat_top[:, 0]                               # (M,)
    gap = (unsat_top[:, 1] - unsat_top[:, 0]).clamp(min=0.0)  # (M,)

    # Smoothing weights over top-2 (only used when gate opens)
    tau_sel = torch.as_tensor(hp.tau_sel, device=v.device, dtype=v.dtype).clamp(min=1e-6)
    w = torch.softmax(-unsat_top / tau_sel, dim=-1)    # (M,2)

    # Gate: smooth only when top-2 are close AND clause not near satisfied
    tau_gap = torch.as_tensor(hp.tau_gap, device=v.device, dtype=v.dtype).clamp(min=1e-6)
    tau_c   = torch.as_tensor(hp.tau_c,   device=v.device, dtype=v.dtype).clamp(min=1e-6)

    lam_gap = torch.sigmoid((hp.gap0 - gap) / tau_gap)         # ~1 when ambiguous
    lam_c   = torch.sigmoid((c - hp.c_hard) / tau_c)           # ~0 near solution
    lam = (lam_gap * lam_c).clamp(0.0, 1.0)                    # (M,)

    # -----------------------------
    # Gradient term G
    # -----------------------------
    # Baseline hard routing
    G_hard = c.unsqueeze(-1).repeat(1, 3)
    G_hard.scatter_(1, top_idx[:, 0:1], unsat_top[:, 1:2])

    # Soft-in-ambiguous-region routing: split the "extra" (gap) between top-2
    # Start from all=c, then add gap*w to the two top literals.
    G_soft = c.unsqueeze(-1).repeat(1, 3)
    extra = gap  # unsat2 - c
    G_soft.scatter_add_(1, top_idx[:, 0:1], (extra * w[:, 0]).unsqueeze(-1))
    G_soft.scatter_add_(1, top_idx[:, 1:2], (extra * w[:, 1]).unsqueeze(-1))

    # Interpolate (baseline-equivalent when lam->0)
    G = (1.0 - lam).unsqueeze(-1) * G_hard + lam.unsqueeze(-1) * G_soft
    G *= (xl * xs).unsqueeze(-1)

    # -----------------------------
    # Rigidity term R
    # -----------------------------
    # Baseline hard routing to best literal
    R_hard = torch.zeros_like(G)
    R_hard.scatter_(1, top_idx[:, 0:1], c.unsqueeze(-1))

    # Soft routing over top-2 (only when lam opens)
    R_soft = torch.zeros_like(G)
    R_soft.scatter_add_(1, top_idx[:, 0:1], (c * w[:, 0]).unsqueeze(-1))
    R_soft.scatter_add_(1, top_idx[:, 1:2], (c * w[:, 1]).unsqueeze(-1))

    R = (1.0 - lam).unsqueeze(-1) * R_hard + lam.unsqueeze(-1) * R_soft
    R *= ((1.0 + hp.zeta * xl) * (1.0 - xs)).unsqueeze(-1)

    # Accumulate dv
    dv_clause = (G + R) * sgn
    dv = torch.zeros_like(v).scatter_add_(0, idx.reshape(-1), dv_clause.reshape(-1))

    # Auxiliary dynamics (baseline; uses hard c)
    dxl = hp.alpha * (c - hp.delta)
    dxs = hp.beta  * ((xs + hp.epsilon) * (c - hp.gamma))

    # Normalize (baseline style)
    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))

    dv  = -dv  * scale
    dxl = -dxl * scale
    dxs = -dxs * scale

    return {"v": dv, "xl": dxl, "xs": dxs}
