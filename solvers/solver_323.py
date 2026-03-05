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

    # NEW: tie-aware monitor knobs (used ONLY for xs update)
    "tau_tie": dict(type="log_uniform", default=2e-2, low=1e-4, high=2e-1),
    "a_tie":   dict(type="uniform",     default=0.4,  low=0.0,  high=1.0),

    "lr":      dict(type="log_uniform", default=1.0, low=1e-1, high=3.0),
}


# ------------------------------------------------------------------ #
# 3.  _grad_single
# ------------------------------------------------------------------ #

def _grad_single(vars, idx, sgn, hp):
    v  = vars["v"]   # (N,)
    xl = vars["xl"]  # (M,)
    xs = vars["xs"]  # (M,)

    # ===== CLAUSE EVALUATION (hard top-2 backbone) =====
    v_lit = v[idx] * sgn  # (M,3)
    top_val, top_idx = torch.topk(v_lit, 2, dim=-1)   # (M,2)
    unsat_top = 0.5 * (1.0 - top_val)                 # (M,2)
    c = unsat_top[:, 0].clamp(0.0, 1.0)               # (M,)

    # ===== GRADIENT TERM (baseline) =====
    G = c.unsqueeze(-1).repeat(1, 3)
    G.scatter_(1, top_idx[:, 0:1], unsat_top[:, 1:2])
    G *= (xl * xs).unsqueeze(-1)

    # ===== RIGIDITY TERM (baseline) =====
    R = torch.zeros_like(G)
    R.scatter_(1, top_idx[:, 0:1], c.unsqueeze(-1))
    R *= ((1.0 + hp.zeta * xl) * (1.0 - xs)).unsqueeze(-1)

    # ===== ACCUMULATE v GRADIENT (baseline) =====
    dv_clause = (G + R) * sgn
    dv = torch.zeros_like(v).scatter_add_(0, idx.reshape(-1), dv_clause.reshape(-1))

    # ===== xl update (baseline) =====
    dxl = hp.alpha * (c - hp.delta)

    # ===== xs update (SINGLE CHANGE): tie-aware monitor c_xs =====
    # Per-literal unsatisfaction in [0,1]
    u = 0.5 * (1.0 - v_lit).clamp(-1.0, 1.0)  # (M,3)
    u12, _ = torch.topk(u, 2, dim=-1, largest=False)  # (M,2), u1<=u2
    u1 = u12[:, 0]
    u2 = u12[:, 1]
    gap = (u2 - u1).clamp_min(0.0)

    tau = torch.as_tensor(hp.tau_tie, device=v.device, dtype=v.dtype).clamp_min(1e-8)
    a = torch.as_tensor(hp.a_tie, device=v.device, dtype=v.dtype).clamp(0.0, 1.0)

    # Smoothly increase the monitor toward u2 when the gap is small (near-tie).
    # Fixed exponent p=2 for sharpness without adding extra knobs.
    p = 2.0
    phi = a * torch.pow(tau / (gap + tau), p)  # in [0,a]
    c_xs = (u1 + gap * phi).clamp(0.0, 1.0)

    dxs = hp.beta * ((xs + hp.epsilon) * (c_xs - hp.gamma))

    # ===== Gradient scaling (baseline style) =====
    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))

    return {
        "v":  -dv  * scale,
        "xl": -dxl * scale,
        "xs": -dxs * scale,
    }
