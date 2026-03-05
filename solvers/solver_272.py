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
    # Strong backbone defaults (kept baseline form; only dxs uses c_mem)
    "alpha":   dict(type="log_uniform", default=48.4,   low=0.5,  high=50.0),
    "beta":    dict(type="log_uniform", default=40.1,   low=2.0,  high=200.0),
    "gamma":   dict(type="uniform",     default=0.289,  low=0.01, high=0.50),
    "delta":   dict(type="uniform",     default=0.073,  low=0.01, high=0.50),
    "epsilon": dict(type="log_uniform", default=1.5e-3, low=1e-4, high=1e-2),
    "zeta":    dict(type="log_uniform", default=1.3e-3, low=1e-4, high=1.0),

    # Tie-only smoother knobs (used ONLY in dxs via c_mem)
    # g0_tie: gap size below which we start smoothing
    "g0_tie":    dict(type="log_uniform", default=3.4e-3, low=1e-4, high=5e-2),
    # kappa_tie: tie-window sharpness
    "kappa_tie": dict(type="log_uniform", default=1.0e-2, low=1e-4, high=5e-2),
    # tau_gap: how fast p(gap) moves from 0.5 (tie) toward 1 (hard-min)
    "tau_gap":   dict(type="log_uniform", default=1.2e-2, low=1e-4, high=1e-1),

    "lr": dict(type="log_uniform", default=1.65, low=1e-1, high=3.0),
}


# ------------------------------------------------------------------ #
# 3.  _grad_single
# ------------------------------------------------------------------ #

def _grad_single(vars, idx, sgn, hp):
    v  = vars["v"]
    xl = vars["xl"]
    xs = vars["xs"]

    eps = 1e-12

    # ===== CLAUSE EVALUATION (hard top-2 backbone; unchanged) =====
    v_lit = v[idx] * sgn                           # (M,3)
    top_val, top_idx = torch.topk(v_lit, 2, dim=-1)
    unsat_top = 0.5 * (1.0 - top_val)              # (M,2)
    c_hard = unsat_top[:, 0].clamp(0.0, 1.0)       # (M,)

    # ===== dv (unchanged) =====
    G = c_hard.unsqueeze(-1).repeat(1, 3)
    G.scatter_(1, top_idx[:, 0:1], unsat_top[:, 1:2])
    G *= (xl * xs).unsqueeze(-1)

    R = torch.zeros_like(G)
    R.scatter_(1, top_idx[:, 0:1], c_hard.unsqueeze(-1))
    R *= ((1.0 + hp.zeta * xl) * (1.0 - xs)).unsqueeze(-1)

    dv_clause = (G + R) * sgn
    dv = torch.zeros_like(v).scatter_add_(0, idx.reshape(-1), dv_clause.reshape(-1))

    # ===== dxl (unchanged) =====
    dxl = hp.alpha * (c_hard - hp.delta)

    # ===== dxs: ONLY CHANGE is c_mem tie-smoother (top-2 only, bounded) =====
    u = 0.5 * (1.0 - v_lit)                        # (M,3) literal-unsat in [0,1]
    u12, _ = torch.topk(u, 2, dim=-1, largest=False)
    u1 = u12[:, 0].clamp(0.0, 1.0)
    u2 = u12[:, 1].clamp(0.0, 1.0)
    gap = (u2 - u1).clamp(min=0.0)

    kappa_eff = torch.as_tensor(hp.kappa_tie, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    g0 = torch.as_tensor(hp.g0_tie, device=v.device, dtype=v.dtype).clamp_min(0.0)

    # tie window weight: ~1 when gap << g0, ~0 when gap >> g0
    w = torch.sigmoid((g0 - gap) / kappa_eff).clamp(0.0, 1.0)

    tau = torch.as_tensor(hp.tau_gap, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    # p(gap) in [0.5, 1): 0.5 at exact tie, quickly -> 1 as gap grows
    p = 0.5 + 0.5 * torch.tanh(gap / (tau + eps))

    # bounded convex combo in [u1,u2]
    c_tie = (p * u1 + (1.0 - p) * u2).clamp(0.0, 1.0)

    # revert to hard-min away from ties
    c_mem = ((1.0 - w) * u1 + w * c_tie).clamp(0.0, 1.0)

    dxs = hp.beta * ((xs + hp.epsilon) * (c_mem - hp.gamma))

    # ===== scaling (baseline style) =====
    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))

    return {
        "v":  -dv  * scale,
        "xl": -dxl * scale,
        "xs": -dxs * scale,
    }
