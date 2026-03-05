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
    # Backbone (seeded from best-known tuned region used in 298/302)
    "alpha":   dict(type="log_uniform", default=32.41489312422484,      low=0.5,  high=50.0),
    "beta":    dict(type="log_uniform", default=94.40331693104432,      low=2.0,  high=200.0),
    "gamma":   dict(type="uniform",     default=0.3052467651267156,     low=0.01, high=0.50),
    "delta":   dict(type="uniform",     default=0.09433049593097542,    low=0.01, high=0.50),
    "epsilon": dict(type="log_uniform", default=2.3866432561734922e-4,  low=1e-4, high=1e-2),
    "zeta":    dict(type="log_uniform", default=0.0032048221354713698, low=1e-4, high=1.0),

    # Weak-band xs release (298)
    "kappa":      dict(type="uniform",     default=0.8,     low=0.0,  high=2.0),
    "xl_thr":     dict(type="log_uniform", default=3000.0,  low=1.0,  high=1e4),
    "xl_sharp":   dict(type="uniform",     default=2.0,     low=0.1,  high=5.0),
    "eta":        dict(type="uniform",     default=0.18,    low=0.0,  high=0.49),
    "band_sharp": dict(type="log_uniform", default=0.05,    low=1e-3, high=0.20),
    "xs_star":    dict(type="uniform",     default=0.55,    low=0.05, high=0.95),

    # Tail shutoff (302)
    "xs_tail":       dict(type="uniform",     default=0.55, low=0.05, high=0.95),
    "xs_tail_sharp": dict(type="log_uniform", default=0.05, low=1e-3, high=0.20),

    # NEW (single principled change vs 302): self-limiting per-clause release at large xl
    "xl_norm": dict(type="log_uniform", default=3e4, low=1e3, high=1e6),

    "lr": dict(type="log_uniform", default=1.15, low=1e-1, high=3.0),
}


# ------------------------------------------------------------------ #
# 3. _grad_single
# ------------------------------------------------------------------ #

def _grad_single(vars, idx, sgn, hp):
    v  = vars["v"]
    xl = vars["xl"]
    xs = vars["xs"]

    # ===== CLAUSE EVALUATION (hard top-2 backbone) =====
    v_lit = v[idx] * sgn  # (M,3)
    top_val, top_idx = torch.topk(v_lit, 2, dim=-1)  # (M,2)
    unsat_top = 0.5 * (1.0 - top_val)
    c = unsat_top[:, 0].clamp(0.0, 1.0)  # (M,)

    # ===== GRADIENT TERM =====
    G = c.unsqueeze(-1).repeat(1, 3)
    G.scatter_(1, top_idx[:, 0:1], unsat_top[:, 1:2])
    G *= (xl * xs).unsqueeze(-1)

    # ===== RIGIDITY TERM =====
    R = torch.zeros_like(G)
    R.scatter_(1, top_idx[:, 0:1], c.unsqueeze(-1))
    R *= ((1.0 + hp.zeta * xl) * (1.0 - xs)).unsqueeze(-1)

    # ===== ACCUMULATE v GRADIENT =====
    dv_clause = (G + R) * sgn
    dv = torch.zeros_like(v).scatter_add_(0, idx.reshape(-1), dv_clause.reshape(-1))

    # ===== MEMORY UPDATES =====
    dxl = hp.alpha * (c - hp.delta)

    gamma = torch.as_tensor(hp.gamma, device=v.device, dtype=v.dtype)
    eps   = torch.as_tensor(hp.epsilon, device=v.device, dtype=v.dtype)
    rhs_base = (xs + eps) * (c - gamma)

    # xl gate in log-space
    xl_thr   = torch.as_tensor(hp.xl_thr, device=v.device, dtype=v.dtype).clamp_min(1.0)
    xl_sharp = torch.as_tensor(hp.xl_sharp, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    log_xl   = torch.log(xl.clamp_min(1.0))
    log_thr  = torch.log(xl_thr)
    gate_xl  = torch.sigmoid((log_xl - log_thr) / xl_sharp)

    # weak-sat band: eta < c < gamma
    eta        = torch.as_tensor(hp.eta, device=v.device, dtype=v.dtype).clamp(0.0, 0.99)
    band_sharp = torch.as_tensor(hp.band_sharp, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    w_lo = torch.sigmoid((c - eta) / band_sharp)
    w_hi = torch.sigmoid((gamma - c) / band_sharp)
    weak_band = (w_lo * w_hi).clamp(0.0, 1.0)

    # upward-only bounded push
    xs_star = torch.as_tensor(hp.xs_star, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    push_up = torch.relu(xs_star - xs) / xs_star

    # tail-safety gate
    xs_tail       = torch.as_tensor(hp.xs_tail, device=v.device, dtype=v.dtype).clamp(0.0, 1.0)
    xs_tail_sharp = torch.as_tensor(hp.xs_tail_sharp, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    gate_tail = torch.sigmoid((xs_tail - xs) / xs_tail_sharp).clamp(0.0, 1.0)

    # NEW: self-limiting amplitude when xl is huge (clause-local, smooth)
    xl_norm = torch.as_tensor(hp.xl_norm, device=v.device, dtype=v.dtype).clamp_min(1.0)
    amp_norm = (xl_norm / (xl_norm + xl.clamp_min(0.0))).clamp(0.0, 1.0)

    kappa = torch.as_tensor(hp.kappa, device=v.device, dtype=v.dtype).clamp_min(0.0)
    release = kappa * gate_xl * weak_band * push_up * gate_tail * amp_norm

    dxs = hp.beta * (rhs_base + release)

    # ===== GRADIENT SCALING =====
    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))

    return {
        "v":  -dv  * scale,
        "xl": -dxl * scale,
        "xs": -dxs * scale,
    }
