import math
import torch
import torch.nn.functional as F

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
    # Backbone
    "alpha":   dict(type="log_uniform", default=31.502485873316463,      low=0.5,  high=50.0),
    "beta":    dict(type="log_uniform", default=91.50944263747377,       low=2.0,  high=200.0),
    "gamma":   dict(type="uniform",     default=0.3039298104521389,      low=0.01, high=0.50),
    "delta":   dict(type="uniform",     default=0.08447270745354496,     low=0.01, high=0.50),
    "epsilon": dict(type="log_uniform", default=0.00023409728281509592, low=1e-4, high=1e-2),
    "zeta":    dict(type="log_uniform", default=0.0030059797126088168,  low=1e-4, high=1.0),

    # Bounded, gated release
    "kappa":      dict(type="uniform",     default=0.8774891722886079,   low=0.0,  high=2.0),
    "xl_thr":     dict(type="log_uniform", default=6635.795309106795,    low=1.0,  high=1e4),
    "xl_sharp":   dict(type="uniform",     default=2.39330419425236,     low=0.1,  high=5.0),
    "eta":        dict(type="uniform",     default=0.18577542191245733,  low=0.0,  high=0.20),
    "band_sharp": dict(type="log_uniform", default=0.12478389634486174,  low=1e-3, high=0.20),
    "xs_star":    dict(type="uniform",     default=0.4890735652522749,   low=0.05, high=0.90),

    # Below-gamma window geometry
    "stag_margin": dict(type="uniform",     default=0.20745491163769436,  low=0.0,  high=0.25),
    "stag_sharp":  dict(type="log_uniform", default=0.012790941546173565, low=1e-3, high=0.20),

    # Over-gamma shoulder (relative width) + absolute cap
    "over_gamma": dict(type="uniform", default=0.03,  low=0.0, high=0.15),
    "over_cap":   dict(type="uniform", default=0.02,  low=0.0, high=0.08),

    # Upper-edge decay scale (tail)
    "over_sharp": dict(type="log_uniform", default=0.012790941546173565, low=1e-3, high=0.20),

    # Hinge sharpness (turn-on for over-gamma decay)
    # Default reduced vs design 409 to reflect unshifted softplus begins decaying near γ_hi.
    "hinge_sharp": dict(type="log_uniform", default=0.003, low=1e-3, high=0.20),

    # In-window amplitude shaping
    "win_floor": dict(type="uniform",     default=0.15801796181881358, low=0.0,  high=0.90),
    "win_pow":   dict(type="log_uniform", default=0.577376239444872,   low=0.25, high=4.0),

    # Window warp strength (bounded)
    "win_edge_pow":  dict(type="log_uniform", default=1.0431200328487458, low=0.25, high=4.0),
    "win_logit_cap": dict(type="uniform",     default=5.459308552849998,  low=2.0,  high=12.0),

    # xl-adaptive temperature magnitude (tail-activated)
    "xl_temp":      dict(type="uniform",     default=0.9715800139065602, low=0.0, high=5.0),
    "xl_tail_mult": dict(type="log_uniform", default=7.958816973983766,  low=1.0, high=100.0),
    "tail_pow":     dict(type="uniform",     default=3.0315002578028385, low=1.0, high=6.0),

    "temp_cap": dict(type="uniform", default=6.765985867897925, low=0.0, high=20.0),

    "lr": dict(type="log_uniform", default=1.206498383226955, low=1e-1, high=3.0),
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

    # ===== Hard top-2 backbone =====
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
    band_sharp = torch.as_tensor(hp.band_sharp, device=v.device, dtype=v.dtype).clamp_min(1e-6)

    # xl gate in log-space
    xl_thr = torch.as_tensor(hp.xl_thr, device=v.device, dtype=v.dtype).clamp_min(1.0)
    xl_sharp = torch.as_tensor(hp.xl_sharp, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    log_xl = torch.log(xl.clamp_min(1.0))
    log_thr = torch.log(xl_thr)
    gate_xl = torch.sigmoid((log_xl - log_thr) / xl_sharp)

    # Weak-sat band: eta < c < gamma (soft)
    w_lo = torch.sigmoid((c - eta) / band_sharp)
    w_hi = torch.sigmoid((gamma - c) / band_sharp)
    weak_band = w_lo * w_hi

    # Upward-only push toward xs_star
    xs_star = torch.as_tensor(hp.xs_star, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    push_up = torch.relu(xs_star - xs) / xs_star

    # Below-gamma window lower edge
    stag_margin = torch.as_tensor(hp.stag_margin, device=v.device, dtype=v.dtype).clamp_min(0.0)
    stag_sharp = torch.as_tensor(hp.stag_sharp, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    c0 = (gamma - stag_margin).clamp(0.0, 1.0)
    near_from_below = torch.sigmoid((c - c0) / stag_sharp)

    # Capped over-gamma shoulder location
    over_gamma = torch.as_tensor(hp.over_gamma, device=v.device, dtype=v.dtype).clamp_min(0.0)
    over_cap = torch.as_tensor(hp.over_cap, device=v.device, dtype=v.dtype).clamp_min(0.0)
    rel_width = over_gamma * (1.0 - gamma)
    width = torch.minimum(rel_width, over_cap)
    gamma_hi = (gamma + width).clamp(0.0, 1.0)

    # ---- SINGLE CHANGE vs design 409 ----
    # Use an UN-SHIFTED softplus hinge for overshoot so below_gamma is inherently in (0,1]
    # (no clamp kink at c≈gamma_hi). This makes suppression turn on smoothly around gamma_hi.
    over_sharp = torch.as_tensor(hp.over_sharp, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    hinge_sharp = torch.as_tensor(hp.hinge_sharp, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    x = (c - gamma_hi) / hinge_sharp
    overshoot = F.softplus(x) * hinge_sharp
    below_gamma = torch.exp(-overshoot / over_sharp).clamp(0.0, 1.0)

    # Window edges (two-sided) with bounded logit-temperature warp
    window_raw = (near_from_below * below_gamma).clamp(0.0, 1.0)
    win_p = torch.as_tensor(hp.win_edge_pow, device=v.device, dtype=v.dtype).clamp_min(0.05)
    cap = torch.as_tensor(hp.win_logit_cap, device=v.device, dtype=v.dtype).clamp_min(0.1)

    # Tail-only xl-adaptive temperature
    xl_temp = torch.as_tensor(hp.xl_temp, device=v.device, dtype=v.dtype).clamp_min(0.0)
    xl_tail_mult = torch.as_tensor(hp.xl_tail_mult, device=v.device, dtype=v.dtype).clamp_min(1.0)
    log_thr_tail = log_thr + torch.log(xl_tail_mult)

    s_tail = (log_xl - log_thr_tail) / xl_sharp
    tail_pow = torch.as_tensor(hp.tail_pow, device=v.device, dtype=v.dtype).clamp_min(1.0)
    gate_tail = torch.sigmoid(s_tail).pow(tail_pow)

    adapt_mask = torch.sqrt((weak_band * window_raw).clamp(0.0, 1.0)) * push_up

    temp_cap = torch.as_tensor(hp.temp_cap, device=v.device, dtype=v.dtype).clamp_min(0.0)
    temp_infl_raw = (xl_temp * gate_tail * adapt_mask)
    temp_infl = (temp_infl_raw * temp_cap) / (temp_infl_raw + temp_cap + 1e-6)
    win_p_eff = win_p / (1.0 + temp_infl)

    eps = torch.as_tensor(1e-6, device=v.device, dtype=v.dtype)
    w = window_raw.clamp(eps, 1.0 - eps)
    logit_w = torch.log(w) - torch.log1p(-w)

    logit_w_warp = cap * torch.tanh((logit_w * win_p_eff) / cap)
    window = torch.sigmoid(logit_w_warp)

    # In-window amplitude (quintic smootherstep)
    denom = (gamma - c0).clamp_min(1e-6)
    t = ((c - c0) / denom).clamp(0.0, 1.0)
    s = t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    win_floor = torch.as_tensor(hp.win_floor, device=v.device, dtype=v.dtype).clamp(0.0, 1.0)
    win_pow = torch.as_tensor(hp.win_pow, device=v.device, dtype=v.dtype).clamp_min(0.05)
    amp = win_floor + (1.0 - win_floor) * s.pow(win_pow)

    near_gamma = window * amp

    release = hp.kappa * gate_xl * weak_band * near_gamma * push_up  # bounded

    rhs_xs = (xs + hp.epsilon) * (c - gamma) + release
    dxs = hp.beta * rhs_xs

    # ===== Gradient scaling =====
    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))

    dv  = -dv  * scale
    dxl = -dxl * scale
    dxs = -dxs * scale

    return {"v": dv, "xl": dxl, "xs": dxs}
