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
    # Baseline
    "alpha":   dict(type="log_uniform", default=5.0,  low=0.5,  high=50.0),
    "beta":    dict(type="log_uniform", default=20.0, low=2.0,  high=200.0),
    "gamma":   dict(type="uniform",     default=0.25, low=0.01, high=0.50),
    "delta":   dict(type="uniform",     default=0.05, low=0.01, high=0.50),
    "epsilon": dict(type="log_uniform", default=1e-3, low=1e-4, high=1e-2),
    "zeta":    dict(type="log_uniform", default=1e-3, low=1e-4, high=1.0),

    # NEW (single modification): fixed-point-safe anti-stall micro-release for xs
    # Trigger: instantaneous stagnation from pre-normalization dv_max
    # Actuator: only when rhs_base<0 (xs decaying) and scaled by c to vanish at SAT.
    "anti_rho":     dict(type="uniform",     default=0.05, low=0.0,  high=0.20),
    "anti_cap":     dict(type="uniform",     default=0.02, low=0.0,  high=0.08),
    "stall_thr":    dict(type="log_uniform", default=0.05, low=1e-3, high=1.0),
    "stall_sharp":  dict(type="log_uniform", default=0.02, low=1e-3, high=0.5),

    "lr":      dict(type="log_uniform", default=1.0, low=1e-1, high=3.0),
}


# ------------------------------------------------------------------ #
# 3. _grad_single
# ------------------------------------------------------------------ #

def _grad_single(vars, idx, sgn, hp):
    v  = vars["v"]   # (N,)
    xl = vars["xl"]  # (M,)
    xs = vars["xs"]  # (M,)

    # ===== CLAUSE EVALUATION =====
    v_lit = v[idx] * sgn  # (M,3)
    top_val, top_idx = torch.topk(v_lit, 2, dim=-1)  # (M,2)
    unsat_top = 0.5 * (1.0 - top_val)  # (M,2)
    c = unsat_top[:, 0].clamp(0.0, 1.0)  # (M,)

    # ===== GRADIENT TERM =====
    G = c.unsqueeze(-1).repeat(1, 3)
    G.scatter_(1, top_idx[:, 0:1], unsat_top[:, 1:2])
    G *= (xl * xs).unsqueeze(-1)

    # ===== RIGIDITY TERM =====
    R = torch.zeros_like(G)
    R.scatter_(1, top_idx[:, 0:1], c.unsqueeze(-1))
    R *= ((1.0 + hp.zeta * xl) * (1.0 - xs)).unsqueeze(-1)

    # ===== ACCUMULATE v GRADIENT (pre-normalization) =====
    dv_clause = (G + R) * sgn
    dv_raw = torch.zeros_like(v).scatter_add_(0, idx.reshape(-1), dv_clause.reshape(-1))

    # ===== MEMORY UPDATES (pre-normalization) =====
    dxl_raw = hp.alpha * (c - hp.delta)

    gamma = torch.as_tensor(hp.gamma, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    eps   = torch.as_tensor(hp.epsilon, device=v.device, dtype=v.dtype)
    rhs_base = (xs + eps) * (c - gamma)  # baseline rhs inside beta

    # --- NEW: stall-gated, fixed-point-safe anti-stall micro-release ---
    # stall_gate uses dv_raw magnitude (not the normalized dv).
    dv_max = dv_raw.abs().max()
    stall_thr   = torch.as_tensor(hp.stall_thr, device=v.device, dtype=v.dtype).clamp_min(1e-12)
    stall_sharp = torch.as_tensor(hp.stall_sharp, device=v.device, dtype=v.dtype).clamp_min(1e-12)
    stall_gate = torch.sigmoid((stall_thr - dv_max) / stall_sharp).clamp(0.0, 1.0)  # scalar

    anti_rho = torch.as_tensor(hp.anti_rho, device=v.device, dtype=v.dtype).clamp_min(0.0)
    anti_cap = torch.as_tensor(hp.anti_cap, device=v.device, dtype=v.dtype).clamp_min(0.0)

    # Activate only when xs is decaying (rhs_base<0), and multiply by c so anti==0 at SAT (c==0).
    anti_raw = anti_rho * stall_gate * c * (-rhs_base).clamp_min(0.0)
    anti = torch.minimum(anti_raw, anti_cap)

    dxs_raw = hp.beta * (rhs_base + anti)

    # ===== GRADIENT SCALING =====
    scale = 1.0 / (dv_raw.abs().max().clamp(min=1e-6))

    return {
        "v":  -dv_raw  * scale,
        "xl": -dxl_raw * scale,
        "xs": -dxs_raw * scale,
    }
