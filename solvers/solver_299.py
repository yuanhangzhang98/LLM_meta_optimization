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
    # Backbone defaults (seeded from the strong 298/285 tuned region)
    "alpha":   dict(type="log_uniform", default=32.41489312422484,       low=0.5,  high=50.0),
    "beta":    dict(type="log_uniform", default=94.40331693104432,       low=2.0,  high=200.0),
    "gamma":   dict(type="uniform",     default=0.3052467651267156,      low=0.01, high=0.50),
    "delta":   dict(type="uniform",     default=0.09433049593097542,     low=0.01, high=0.50),
    "epsilon": dict(type="log_uniform", default=2.3866432561734922e-4,   low=1e-4, high=1e-2),
    "zeta":    dict(type="log_uniform", default=0.0032048221354713698,  low=1e-4, high=1.0),

    # --- Single principled modification vs baseline: tail-safe near-gamma xs release ---
    "kappa":       dict(type="uniform",     default=0.6,     low=0.0,   high=2.0),
    "xl_thr":      dict(type="log_uniform", default=3000.0,  low=1.0,   high=1e4),
    "xl_sharp":    dict(type="uniform",     default=2.0,     low=0.1,   high=5.0),
    # narrow band just below gamma: (gamma - band_width) < c < gamma
    "band_width":  dict(type="uniform",     default=0.06,    low=0.005, high=0.25),
    "band_sharp":  dict(type="log_uniform", default=0.03,    low=5e-3,  high=0.15),
    "xs_star":     dict(type="uniform",     default=0.45,    low=0.05,  high=0.90),

    "lr":          dict(type="log_uniform", default=1.15,    low=1e-1,  high=3.0),
}


# ------------------------------------------------------------------ #
# 3. _grad_single
# ------------------------------------------------------------------ #

def _grad_single(vars, idx, sgn, hp):
    v  = vars["v"]
    xl = vars["xl"]
    xs = vars["xs"]

    # ===== CLAUSE EVALUATION (baseline hard top-2) =====
    v_lit = v[idx] * sgn  # (M,3)
    top_val, top_idx = torch.topk(v_lit, 2, dim=-1)  # (M,2)
    unsat_top = 0.5 * (1.0 - top_val)  # (M,2)
    c = unsat_top[:, 0].clamp(0.0, 1.0)  # (M,)

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

    # ===== MEMORY UPDATES =====
    dxl = hp.alpha * (c - hp.delta)

    gamma = torch.as_tensor(hp.gamma, device=v.device, dtype=v.dtype)
    eps   = torch.as_tensor(hp.epsilon, device=v.device, dtype=v.dtype)

    # Baseline xs RHS
    rhs_base = (xs + eps) * (c - gamma)

    # ---- Tail-safe near-gamma release (single modification) ----
    # xl gate in log-space
    xl_thr   = torch.as_tensor(hp.xl_thr, device=v.device, dtype=v.dtype).clamp_min(1.0)
    xl_sharp = torch.as_tensor(hp.xl_sharp, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    log_xl   = torch.log(xl.clamp_min(1.0))
    log_thr  = torch.log(xl_thr)
    gate_xl  = torch.sigmoid((log_xl - log_thr) / xl_sharp)

    # narrow band just below gamma: (gamma - w) < c < gamma
    w = torch.as_tensor(hp.band_width, device=v.device, dtype=v.dtype).clamp(0.0, 0.99)
    s = torch.as_tensor(hp.band_sharp, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    c_lo = (gamma - w).clamp(0.0, 1.0)
    w_lo = torch.sigmoid((c - c_lo) / s)
    w_hi = torch.sigmoid((gamma - c) / s)
    near_band = (w_lo * w_hi).clamp(0.0, 1.0)

    # upward-only bounded push toward xs_star, with extra damping to shut off quickly
    xs_star = torch.as_tensor(hp.xs_star, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    push_up = torch.relu(xs_star - xs) / xs_star
    damp = (1.0 - xs).clamp(0.0, 1.0)

    kappa = torch.as_tensor(hp.kappa, device=v.device, dtype=v.dtype).clamp_min(0.0)
    release = kappa * gate_xl * near_band * damp * push_up

    dxs = hp.beta * (rhs_base + release)

    # ===== GRADIENT SCALING (baseline) =====
    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))

    return {
        "v":  -dv  * scale,
        "xl": -dxl * scale,
        "xs": -dxs * scale,
    }
