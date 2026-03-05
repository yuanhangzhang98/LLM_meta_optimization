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
    # Backbone defaults (good tuned region)
    "alpha":   dict(type="log_uniform", default=32.41489312422484,        low=0.5,  high=50.0),
    "beta":    dict(type="log_uniform", default=94.40331693104432,        low=2.0,  high=200.0),
    "gamma":   dict(type="uniform",     default=0.3052467651267156,      low=0.01, high=0.50),
    "delta":   dict(type="uniform",     default=0.09433049593097542,     low=0.01, high=0.50),
    "epsilon": dict(type="log_uniform", default=0.00023866432561734922,  low=1e-4, high=1e-2),
    "zeta":    dict(type="log_uniform", default=0.0032048221354713698,   low=1e-4, high=1.0),

    # Bounded weak-band release (Design 51/42/270 lineage)
    "kappa":      dict(type="uniform",     default=0.997187430001158,    low=0.0,  high=2.0),
    "xl_thr":     dict(type="log_uniform", default=6622.023458697489,    low=1.0,  high=1e4),
    "xl_sharp":   dict(type="uniform",     default=2.3895476605904826,   low=0.1,  high=5.0),
    "eta":        dict(type="uniform",     default=0.19022382405412103,  low=0.0,  high=0.20),
    "band_sharp": dict(type="log_uniform", default=0.12423107575665222,  low=1e-3, high=0.20),
    "xs_star":    dict(type="uniform",     default=0.4898096116751786,   low=0.05, high=0.90),

    # Below-gamma window geometry (Design 270)
    "stag_margin":    dict(type="uniform",     default=0.2061293027725612,  low=0.0,  high=0.25),
    "stag_sharp":     dict(type="log_uniform", default=0.01233619087369346, low=1e-3, high=0.20),
    "warp_sym_p":     dict(type="uniform",     default=2.0,                 low=1.0,  high=6.0),
    "stag_margin_xl": dict(type="uniform",     default=0.0,                 low=0.0,  high=1.0),

    # In-window amplitude shaping
    "win_floor": dict(type="uniform",     default=0.559803370834878,     low=0.0,  high=0.90),
    "win_pow":   dict(type="log_uniform", default=0.5176801688234721,    low=0.25, high=4.0),
    "win_tau":   dict(type="uniform",     default=0.25,                  low=0.0,  high=0.80),

    # NEW (single change vs 270): xl-conditioned absolute minimum weak-band width
    "band_sharp_abs_min": dict(type="uniform", default=0.01,             low=0.001, high=0.08),

    "lr": dict(type="log_uniform", default=1.151158655354217,            low=1e-1, high=3.0),
}


# ------------------------------------------------------------------ #
# 3. _grad_single
# ------------------------------------------------------------------ #

def _grad_single(vars, idx, sgn, hp):
    v  = vars["v"]
    xl = vars["xl"]
    xs = vars["xs"]

    # Literals
    v_lit = v[idx] * sgn  # (M,3)

    # ===== Hard top-2 backbone (unchanged) =====
    top_val, top_idx = torch.topk(v_lit, 2, dim=-1)          # (M,2)
    unsat_top = 0.5 * (1.0 - top_val)                        # (M,2)
    c = unsat_top[:, 0].clamp(0.0, 1.0)                      # (M,)

    G = c.unsqueeze(-1).repeat(1, 3)
    G.scatter_(1, top_idx[:, 0:1], unsat_top[:, 1:2])
    G *= (xl * xs).unsqueeze(-1)

    R = torch.zeros_like(G)
    R.scatter_(1, top_idx[:, 0:1], c.unsqueeze(-1))
    R *= ((1.0 + hp.zeta * xl) * (1.0 - xs)).unsqueeze(-1)

    dv_clause = (G + R) * sgn
    dv = torch.zeros_like(v).scatter_add_(0, idx.reshape(-1), dv_clause.reshape(-1))

    # ===== Memory updates =====
    dxl = hp.alpha * (c - hp.delta)

    gamma = torch.as_tensor(hp.gamma, device=v.device, dtype=v.dtype)
    eta = torch.as_tensor(hp.eta, device=v.device, dtype=v.dtype).clamp_min(0.0)

    # xl gate in log-space
    xl_thr = torch.as_tensor(hp.xl_thr, device=v.device, dtype=v.dtype).clamp_min(1.0)
    xl_sharp = torch.as_tensor(hp.xl_sharp, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    log_xl = torch.log(xl.clamp_min(1.0))
    log_thr = torch.log(xl_thr)
    gate_xl = torch.sigmoid((log_xl - log_thr) / xl_sharp)  # (0,1)

    # Weak-sat band: eta < c < gamma
    band_sharp = torch.as_tensor(hp.band_sharp, device=v.device, dtype=v.dtype).clamp_min(1e-6)

    # --- SINGLE CHANGE: xl-conditioned absolute width floor to prevent release starvation ---
    band_sharp_abs_min = torch.as_tensor(hp.band_sharp_abs_min, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    band_floor = band_sharp_abs_min * (1.0 + gate_xl)  # wider for high-xl clauses
    band_sharp_eff = torch.maximum(band_sharp, band_floor)

    w_lo = torch.sigmoid((c - eta) / band_sharp_eff)
    w_hi = torch.sigmoid((gamma - c) / band_sharp_eff)
    weak_band = w_lo * w_hi

    # Upward-only push toward xs_star
    xs_star = torch.as_tensor(hp.xs_star, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    push_up = torch.relu(xs_star - xs) / xs_star

    # Below-gamma window: (gamma-eff_margin) < c < gamma
    stag_margin = torch.as_tensor(hp.stag_margin, device=v.device, dtype=v.dtype).clamp_min(0.0)
    stag_sharp = torch.as_tensor(hp.stag_sharp, device=v.device, dtype=v.dtype).clamp_min(1e-6)

    stag_margin_xl = torch.as_tensor(hp.stag_margin_xl, device=v.device, dtype=v.dtype).clamp_min(0.0)
    mult = (1.0 + stag_margin_xl * (2.0 * gate_xl - 1.0)).clamp(0.5, 1.5)
    eff_margin = (stag_margin * mult).clamp(0.0, 0.49)

    c0 = (gamma - eff_margin).clamp(0.0, 1.0)
    near_from_below = torch.sigmoid((c - c0) / stag_sharp)
    below_gamma = torch.sigmoid((gamma - c) / stag_sharp)
    window = near_from_below * below_gamma

    # In-window amplitude shaping: smootherstep -> symmetric power-odds warp -> power
    denom = (gamma - c0).clamp_min(1e-6)
    t = ((c - c0) / denom).clamp(0.0, 1.0)

    win_tau = torch.as_tensor(hp.win_tau, device=v.device, dtype=v.dtype).clamp(0.0, 0.95)
    t2 = ((t - win_tau) / (1.0 - win_tau).clamp_min(1e-6)).clamp(0.0, 1.0)

    s = t2 * t2 * t2 * (t2 * (t2 * 6.0 - 15.0) + 10.0)  # quintic smootherstep

    warp_p = torch.as_tensor(hp.warp_sym_p, device=v.device, dtype=v.dtype).clamp_min(1.0)
    eps = torch.as_tensor(1e-12, device=v.device, dtype=v.dtype)
    a = s.clamp(0.0, 1.0).pow(warp_p)
    b = (1.0 - s).clamp(0.0, 1.0).pow(warp_p)
    sw = a / (a + b + eps)

    win_floor = torch.as_tensor(hp.win_floor, device=v.device, dtype=v.dtype).clamp(0.0, 1.0)
    win_pow = torch.as_tensor(hp.win_pow, device=v.device, dtype=v.dtype).clamp_min(0.05)
    amp = win_floor + (1.0 - win_floor) * sw.pow(win_pow)

    near_gamma = window * amp

    release = hp.kappa * gate_xl * weak_band * near_gamma * push_up

    rhs_xs = (xs + hp.epsilon) * (c - gamma) + release
    dxs = hp.beta * rhs_xs

    # ===== Gradient scaling =====
    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))

    dv  = -dv  * scale
    dxl = -dxl * scale
    dxs = -dxs * scale

    return {"v": dv, "xl": dxl, "xs": dxs}
