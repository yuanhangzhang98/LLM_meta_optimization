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
    "alpha":     dict(type="log_uniform", default=5.0,   low=0.5,  high=50.0),
    "beta":      dict(type="log_uniform", default=20.0,  low=2.0,  high=200.0),
    "gamma":     dict(type="uniform",     default=0.25,  low=0.01, high=0.50),
    "delta":     dict(type="uniform",     default=0.05,  low=0.01, high=0.50),
    "epsilon":   dict(type="log_uniform", default=1e-3,  low=1e-4, high=1e-2),
    "zeta":      dict(type="log_uniform", default=1e-3,  low=1e-4, high=1.0),

    # Tie-aware smoothing ONLY for memory updates (xl/xs), not for v-forces
    "gap_thr":   dict(type="uniform",     default=0.05,  low=0.00, high=0.50),
    "gap_sharp": dict(type="log_uniform", default=0.02,  low=1e-3, high=0.20),
    "tau_max":   dict(type="log_uniform", default=0.10,  low=1e-2, high=1.00),

    "lr":        dict(type="log_uniform", default=1.0,   low=1e-1, high=3.0),
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

    # ===== BASELINE HARD TOP-2 BACKBONE (UNCHANGED FOR v) =====
    top_val, top_idx = torch.topk(v_lit, 2, dim=-1)      # (M,2)
    unsat_top = 0.5 * (1.0 - top_val)                    # (M,2)
    c_hard = unsat_top[:, 0].clamp(0.0, 1.0)             # (M,)

    G = c_hard.unsqueeze(-1).repeat(1, 3)
    G.scatter_(1, top_idx[:, 0:1], unsat_top[:, 1:2])
    G *= (xl * xs).unsqueeze(-1)

    R = torch.zeros_like(G)
    R.scatter_(1, top_idx[:, 0:1], c_hard.unsqueeze(-1))
    R *= ((1.0 + hp.zeta * xl) * (1.0 - xs)).unsqueeze(-1)

    dv_clause = (G + R) * sgn
    dv = torch.zeros_like(v).scatter_add_(0, idx.reshape(-1), dv_clause.reshape(-1))

    # ===== TIE-AWARE CLAUSE COST ONLY FOR MEMORY UPDATES =====
    # Detect near-ties between best and second-best literal
    gap = (top_val[:, 0] - top_val[:, 1]).clamp_min(0.0)  # (M,)
    gap_sharp = torch.as_tensor(hp.gap_sharp, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    gap_thr = torch.as_tensor(hp.gap_thr, device=v.device, dtype=v.dtype)
    blend = torch.sigmoid((gap_thr - gap) / gap_sharp)    # (M,) ~1 only on near ties

    # Soft max-based cost estimate (stable near ties)
    tau_min = torch.as_tensor(1e-3, device=v.device, dtype=v.dtype)
    tau_max = torch.as_tensor(hp.tau_max, device=v.device, dtype=v.dtype).clamp_min(tau_min)
    tau_eff = tau_min + blend * (tau_max - tau_min)       # (M,)

    w = torch.softmax(v_lit / tau_eff.unsqueeze(-1), dim=-1)  # (M,3)
    soft_max = (w * v_lit).sum(dim=-1)                        # (M,)
    c_soft = (0.5 * (1.0 - soft_max)).clamp(0.0, 1.0)

    c = (1.0 - blend) * c_hard + blend * c_soft

    # ===== AUXILIARY VARIABLE GRADIENTS (use tie-aware c) =====
    dxl = hp.alpha * (c - hp.delta)
    dxs = hp.beta  * ((xs + hp.epsilon) * (c - hp.gamma))

    # ===== GRADIENT SCALING =====
    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))

    dv  = -dv  * scale
    dxl = -dxl * scale
    dxs = -dxs * scale

    return {"v": dv, "xl": dxl, "xs": dxs}
