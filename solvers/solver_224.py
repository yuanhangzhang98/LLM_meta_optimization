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
    # Design-73 tuned defaults (keep everything fixed except ONE principled extension)
    "alpha":   dict(type="log_uniform", default=48.3997327918925,    low=0.5,  high=50.0),
    "beta":    dict(type="log_uniform", default=40.08506794497957,   low=2.0,  high=200.0),
    "gamma":   dict(type="uniform",     default=0.28948339187872574, low=0.01, high=0.50),
    "delta":   dict(type="uniform",     default=0.07327583432197571, low=0.01, high=0.50),
    "epsilon": dict(type="log_uniform", default=0.001468934335764333, low=1e-4, high=1e-2),
    "zeta":    dict(type="log_uniform", default=0.0013418985396236333, low=1e-4, high=1.0),
    "lr":      dict(type="log_uniform", default=1.6536008035680871,  low=1e-1, high=3.0),

    # Tie-only smoothing knobs (Design-73)
    "tau_tie": dict(type="log_uniform", default=0.01712294456377018, low=1e-3, high=2e-1),
    "g0":      dict(type="log_uniform", default=0.0033876327188782085, low=1e-3, high=2e-1),
    "kappa":   dict(type="log_uniform", default=0.052906428769193214, low=1e-3, high=1e-1),

    # NEW (single principled addition): only smooth when clause monitor is near gamma
    "gam_margin": dict(type="uniform", default=0.06,  low=0.0,  high=0.25),
    "gam_sharp":  dict(type="log_uniform", default=0.02, low=1e-3, high=2e-1),
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

    # --- Hard top-2 backbone for v-updates (unchanged) ---
    top_val, top_idx = torch.topk(v_lit, 2, dim=-1)  # (M,2)
    unsat_top = 0.5 * (1.0 - top_val)                # (M,2)
    c_hard = unsat_top[:, 0]                         # (M,)

    G = c_hard.unsqueeze(-1).repeat(1, 3)
    G.scatter_(1, top_idx[:, 0:1], unsat_top[:, 1:2])
    G *= (xl * xs).unsqueeze(-1)

    R = torch.zeros_like(G)
    R.scatter_(1, top_idx[:, 0:1], c_hard.unsqueeze(-1))
    R *= ((1.0 + hp.zeta * xl) * (1.0 - xs)).unsqueeze(-1)

    dv_clause = (G + R) * sgn
    dv = torch.zeros_like(v).scatter_add_(0, idx.reshape(-1), dv_clause.reshape(-1))

    # --- Memory monitor for dxs: bounded smoothing only for near-ties AND near-gamma ---
    u = 0.5 * (1.0 - v_lit)  # (M,3) in [0,1]

    # u1<=u2 for tie detection
    u12, _ = torch.topk(u, 2, dim=-1, largest=False)  # (M,2)
    u1 = u12[:, 0]
    u2 = u12[:, 1]
    gap = (u2 - u1).clamp(min=0.0)

    # Tie gate (Design-73)
    w_tie = torch.sigmoid((hp.g0 - gap) / (hp.kappa + 1e-12)).clamp(0.0, 1.0)

    # NEW: near-gamma gate (smoothly turns off smoothing away from switching boundary)
    gamma = torch.as_tensor(hp.gamma, device=v.device, dtype=v.dtype)
    d = (u1 - gamma).abs()
    w_gam = torch.sigmoid((hp.gam_margin - d) / (hp.gam_sharp + 1e-12)).clamp(0.0, 1.0)

    w = (w_tie * w_gam).clamp(0.0, 1.0)

    # Boltzmann expectation over (u1,u2), guaranteed in [u1,u2]
    tau = torch.as_tensor(hp.tau_tie, device=v.device, dtype=v.dtype).clamp_min(1e-12)
    logits = -(u12 - u1.unsqueeze(-1)) / tau
    p = torch.softmax(logits, dim=-1)
    c_smooth = (p * u12).sum(dim=-1)  # (M,) in [u1,u2]

    c_mem = ((1.0 - w) * u1 + w * c_smooth).clamp(0.0, 1.0)

    # Long-term memory stays driven by hard monitor
    dxl = hp.alpha * (c_hard - hp.delta)
    # Short-term switching uses localized tie smoothing
    dxs = hp.beta * ((xs + hp.epsilon) * (c_mem - hp.gamma))

    # Normalize (baseline style)
    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))

    return {
        "v":  -dv  * scale,
        "xl": -dxl * scale,
        "xs": -dxs * scale,
    }
