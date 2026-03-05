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
    # Backbone (seeded tuned region)
    "alpha":   dict(type="log_uniform", default=8.571243380060926,      low=0.5,  high=50.0),
    "beta":    dict(type="log_uniform", default=32.19501034954162,      low=2.0,  high=200.0),
    "gamma":   dict(type="uniform",     default=0.34914900458284026,    low=0.01, high=0.50),
    "delta":   dict(type="uniform",     default=0.0681169181592405,     low=0.01, high=0.50),
    "epsilon": dict(type="log_uniform", default=0.0006006376598709629,  low=1e-4, high=1e-2),
    "zeta":    dict(type="log_uniform", default=0.025452379409594807,   low=1e-4, high=1.0),

    # Weak-band xs release
    "kappa":      dict(type="uniform",     default=0.9729545490104339,  low=0.0,  high=2.0),
    "xl_thr":     dict(type="log_uniform", default=320.54689731763017,  low=1.0,  high=1e4),
    "xl_sharp":   dict(type="uniform",     default=1.7537985068808664,  low=0.1,  high=5.0),

    # window-safe: eta = eta_frac * gamma
    "eta_frac":   dict(type="uniform",     default=0.7836989592712212,  low=0.0,  high=0.95),
    "band_sharp": dict(type="log_uniform", default=0.05031153933412768, low=1e-3, high=0.20),
    "xs_star":    dict(type="uniform",     default=0.6793616580736854,  low=0.05, high=0.95),

    # Tail-safety
    "xs_tail":       dict(type="uniform",     default=0.49934276918573695,  low=0.05, high=0.95),
    "xs_tail_sharp": dict(type="log_uniform", default=0.016535536648730188, low=1e-3, high=0.20),
    "tail_mu":       dict(type="uniform",     default=0.4059711759309325,   low=0.0,  high=0.50),

    # Tail-floor amplitude
    "tail_floor":    dict(type="uniform", default=0.06, low=0.0,  high=0.30),

    # Tail-floor engagement (interpreted on c-gap in this design)
    "tail_rhs_thr":   dict(type="uniform",     default=0.01,  low=0.0,  high=0.20),
    "tail_rhs_sharp": dict(type="log_uniform", default=0.03,  low=1e-3, high=0.30),

    # Bounded amp_norm family (smooth-knee Hill in log-xl + floor)
    "xl_norm":   dict(type="log_uniform", default=15249.194849720376, low=1e3,  high=1e6),
    "amp_sigma": dict(type="log_uniform", default=1.156303742670224,  low=0.10, high=10.0),
    "amp_p":     dict(type="uniform",     default=2.598144403420096,  low=0.5,  high=6.0),
    "amp_floor": dict(type="uniform",     default=0.05546062416816861, low=0.0,  high=0.25),

    # In-window floor geometry
    "floor_pow": dict(type="uniform",     default=2.020842742927941,  low=1.0,  high=6.0),
    "floor_mix": dict(type="uniform",     default=0.6006904971477652, low=0.0,  high=1.0),

    # Smooth-max temperature for late-only floor engagement
    "amp_tau":   dict(type="log_uniform", default=0.01964320727951571, low=1e-3, high=0.20),

    "lr": dict(type="log_uniform", default=2.930052925520562, low=1e-1, high=3.0),
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

    gamma = torch.as_tensor(hp.gamma, device=v.device, dtype=v.dtype).clamp_min(1e-3)
    eps   = torch.as_tensor(hp.epsilon, device=v.device, dtype=v.dtype)
    rhs_base = (xs + eps) * (c - gamma)

    # xl gate in log-space
    xl_thr   = torch.as_tensor(hp.xl_thr, device=v.device, dtype=v.dtype).clamp_min(1.0)
    xl_sharp = torch.as_tensor(hp.xl_sharp, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    log_xl   = torch.log(xl.clamp_min(1.0))
    log_thr  = torch.log(xl_thr)
    gate_xl  = torch.sigmoid((log_xl - log_thr) / xl_sharp)

    # weak-sat band: eta < c < gamma (WINDOW-SAFE eta)
    eta_frac   = torch.as_tensor(hp.eta_frac, device=v.device, dtype=v.dtype).clamp(0.0, 0.95)
    eta        = (eta_frac * gamma).clamp(0.0, gamma - 1e-6)
    band_sharp = torch.as_tensor(hp.band_sharp, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    w_lo = torch.sigmoid((c - eta) / band_sharp)
    w_hi = torch.sigmoid((gamma - c) / band_sharp)
    weak_band = (w_lo * w_hi).clamp(0.0, 1.0)

    # upward-only bounded push
    xs_star = torch.as_tensor(hp.xs_star, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    push_up = torch.relu(xs_star - xs) / xs_star

    # floor shape coordinate s in weak band (smoothstep in [0,1])
    denom = (gamma - eta).clamp_min(1e-3)
    t = ((c - eta) / denom).clamp(0.0, 1.0)
    s = (t * t * (3.0 - 2.0 * t)).clamp(0.0, 1.0)

    # Tail-safety gate
    xs_tail       = torch.as_tensor(hp.xs_tail, device=v.device, dtype=v.dtype).clamp(0.0, 1.0)
    xs_tail_sharp = torch.as_tensor(hp.xs_tail_sharp, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    tail_mu       = torch.as_tensor(hp.tail_mu, device=v.device, dtype=v.dtype).clamp(0.0, 1.0)

    gate_tail_raw = torch.sigmoid((xs_tail - xs + tail_mu * gate_xl) / xs_tail_sharp).clamp(0.0, 1.0)
    blend = (s * gate_xl).clamp(0.0, 1.0)
    gate_tail = (1.0 - blend * (1.0 - gate_tail_raw)).clamp(0.0, 1.0)

    # amp_decay: smooth-knee Hill decay in log-xl
    xl_norm   = torch.as_tensor(hp.xl_norm, device=v.device, dtype=v.dtype).clamp_min(1.0)
    amp_sigma = torch.as_tensor(hp.amp_sigma, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    amp_p     = torch.as_tensor(hp.amp_p, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    amp_floor = torch.as_tensor(hp.amp_floor, device=v.device, dtype=v.dtype).clamp(0.0, 0.95)

    log_norm = torch.log(xl_norm)
    x = (log_xl - log_norm) / amp_sigma
    sp = F.softplus(x)
    amp_decay = 1.0 / (1.0 + torch.pow(sp, amp_p))

    # In-window floor geometry (mixed)
    floor_pow = torch.as_tensor(hp.floor_pow, device=v.device, dtype=v.dtype).clamp_min(1.0)
    floor_mix = torch.as_tensor(hp.floor_mix, device=v.device, dtype=v.dtype).clamp(0.0, 1.0)
    shape_388 = torch.pow(s, floor_pow)
    shape_390 = torch.pow((1.0 - s).clamp(0.0, 1.0), floor_pow)
    amp_floor_eff = (amp_floor * ((1.0 - floor_mix) * shape_388 + floor_mix * shape_390)).clamp(0.0, 0.95)

    # late-acting floor via smooth-max lower bound
    amp_tau = torch.as_tensor(hp.amp_tau, device=v.device, dtype=v.dtype).clamp_min(1e-4)
    amp_norm = (amp_decay + amp_tau * F.softplus((amp_floor_eff - amp_decay) / amp_tau)).clamp(0.0, 1.0)

    # ---- SINGLE CHANGE vs design 408 ----
    # Use c-gap (gamma - c) as decay evidence (state-consistent, not suppressed when xs is small).
    tail_rhs_thr   = torch.as_tensor(hp.tail_rhs_thr, device=v.device, dtype=v.dtype).clamp_min(0.0)
    tail_rhs_sharp = torch.as_tensor(hp.tail_rhs_sharp, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    decay_strength = (gamma - c).clamp_min(0.0)
    decay_gate = torch.sigmoid((decay_strength - tail_rhs_thr) / tail_rhs_sharp).clamp(0.0, 1.0)

    tail_floor = torch.as_tensor(hp.tail_floor, device=v.device, dtype=v.dtype).clamp(0.0, 0.95)
    floor_mask = (blend * (1.0 - gate_tail_raw) * (1.0 - amp_norm) * decay_gate).clamp(0.0, 1.0)
    gate_tail = (gate_tail + tail_floor * floor_mask * (1.0 - gate_tail)).clamp(0.0, 1.0)

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
