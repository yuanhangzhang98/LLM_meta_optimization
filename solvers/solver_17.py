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
    "alpha":   dict(type="log_uniform", default=0.5909961203695797, low=0.5,  high=50.0),
    "beta":    dict(type="log_uniform", default=21.73432909441711,  low=2.0,  high=200.0),
    "gamma":   dict(type="uniform",     default=0.10833792982038212,low=0.01, high=0.50),
    "delta":   dict(type="uniform",     default=0.01952403070016321,low=0.01, high=0.50),
    "epsilon": dict(type="log_uniform", default=0.00014854507465049844,low=1e-4, high=1e-2),
    "zeta":    dict(type="log_uniform", default=0.00024918590524877505,low=1e-4, high=1.0),

    # Bounded weak-band release (Design 24 backbone)
    "kappa":      dict(type="uniform",     default=1.3031913175393028,low=0.0,  high=2.0),
    "xl_thr":     dict(type="log_uniform", default=60.87746723931547, low=1.0,  high=1e4),
    "xl_sharp":   dict(type="uniform",     default=1.8251054877219548,low=0.1,  high=5.0),
    "eta":        dict(type="uniform",     default=0.160358811874578,low=0.0,  high=0.20),
    "band_sharp": dict(type="log_uniform", default=0.007144242253729021,low=1e-3, high=0.20),
    "xs_star":    dict(type="uniform",     default=0.5544897500387054,low=0.05, high=0.90),

    # Near-gamma stagnation focus (Design 24)
    "stag_margin": dict(type="uniform",     default=0.009161933986036014,low=0.0,  high=0.25),
    "stag_sharp":  dict(type="log_uniform", default=0.16086588966097656,low=1e-3, high=0.20),

    # NEW (single change): allow weak-band to extend slightly above gamma
    # so release can trigger for small overshoots (c just above gamma), not only just below.
    "overshoot":   dict(type="uniform",     default=0.19505064189434052,             low=0.0,  high=0.25),

    "lr":      dict(type="log_uniform", default=2.739503786025398,   low=1e-1, high=3.0),
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

    # ===== BASELINE HARD TOP-2 BACKBONE (UNCHANGED) =====
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

    # ===== MEMORY UPDATES =====
    dxl = hp.alpha * (c - hp.delta)

    gamma = torch.as_tensor(hp.gamma, device=v.device, dtype=v.dtype)
    eta = torch.as_tensor(hp.eta, device=v.device, dtype=v.dtype).clamp_min(0.0)
    band_sharp = torch.as_tensor(hp.band_sharp, device=v.device, dtype=v.dtype).clamp_min(1e-6)

    xl_thr = torch.as_tensor(hp.xl_thr, device=v.device, dtype=v.dtype).clamp_min(1.0)
    xl_sharp = torch.as_tensor(hp.xl_sharp, device=v.device, dtype=v.dtype).clamp_min(1e-6)

    log_xl = torch.log(xl.clamp_min(1.0))
    log_thr = torch.log(xl_thr)
    gate_xl = torch.sigmoid((log_xl - log_thr) / xl_sharp)   # ~1 if xl >> xl_thr

    # Weak-band (SINGLE CHANGE vs Design 24): eta < c < gamma + overshoot
    overshoot = torch.as_tensor(hp.overshoot, device=v.device, dtype=v.dtype).clamp_min(0.0)
    w_lo = torch.sigmoid((c - eta) / band_sharp)
    w_hi = torch.sigmoid(((gamma + overshoot) - c) / band_sharp)
    weak_band = w_lo * w_hi

    xs_star = torch.as_tensor(hp.xs_star, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    push_up = torch.relu(xs_star - xs) / xs_star             # in [0,1], upward-only

    # Near-gamma focus (as in Design 24): close to gamma from below
    stag_margin = torch.as_tensor(hp.stag_margin, device=v.device, dtype=v.dtype).clamp_min(0.0)
    stag_sharp = torch.as_tensor(hp.stag_sharp, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    c0 = (gamma - stag_margin).clamp(0.0, 1.0)
    near_gamma = torch.sigmoid((c - c0) / stag_sharp)

    release = hp.kappa * gate_xl * weak_band * near_gamma * push_up  # bounded in [0, kappa]

    rhs_xs = (xs + hp.epsilon) * (c - gamma) + release
    dxs = hp.beta * rhs_xs

    # ===== GRADIENT SCALING =====
    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))

    dv  = -dv  * scale
    dxl = -dxl * scale
    dxs = -dxs * scale

    return {"v": dv, "xl": dxl, "xs": dxs}
