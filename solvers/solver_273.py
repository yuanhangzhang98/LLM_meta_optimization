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
    # Backbone defaults (from best-known tuned region; as in Design 264)
    "alpha":   dict(type="log_uniform", default=14.504000467173922,       low=0.5,  high=50.0),
    "beta":    dict(type="log_uniform", default=22.87679950018518,        low=2.0,  high=200.0),
    "gamma":   dict(type="uniform",     default=0.3713393245315345,       low=0.01, high=0.50),
    "delta":   dict(type="uniform",     default=0.13541660385008228,      low=0.01, high=0.50),
    "epsilon": dict(type="log_uniform", default=0.001132926021204599,     low=1e-4, high=1e-2),
    "zeta":    dict(type="log_uniform", default=0.013885962445043169,     low=1e-4, high=1.0),

    # Bounded weak-band release (Design 48 lineage; as in Design 264)
    "kappa":      dict(type="uniform",     default=0.6608025081577263,    low=0.0,  high=2.0),
    "xl_thr":     dict(type="log_uniform", default=2562.596512060553,     low=1.0,  high=1e4),
    "xl_sharp":   dict(type="uniform",     default=3.4127970602799786,    low=0.1,  high=5.0),
    "eta":        dict(type="uniform",     default=0.0646429316096725,    low=0.0,  high=0.20),
    "band_sharp": dict(type="log_uniform", default=0.13869458180622743,   low=1e-3, high=0.20),
    "xs_star":    dict(type="uniform",     default=0.6231181608738778,    low=0.05, high=0.90),

    # Below-gamma window geometry (Design 48 lineage; as in Design 264)
    "stag_margin": dict(type="uniform",     default=0.21198765905041214,  low=0.0,  high=0.25),
    "stag_sharp":  dict(type="log_uniform", default=0.05626208189215996,  low=1e-3, high=0.20),

    # In-window amplitude shaping (Design 48 lineage; as in Design 264)
    "win_floor": dict(type="uniform",     default=0.184266876474337,     low=0.0,  high=0.90),
    "win_pow":   dict(type="log_uniform", default=0.7497234371465881,    low=0.25, high=4.0),
    "win_tau":   dict(type="uniform",     default=0.02058028573255097,   low=0.0,  high=0.80),

    # High-xl release gain (Design 48 lineage; as in Design 264)
    "xl_gain": dict(type="uniform", default=0.5849418497034736, low=0.0, high=2.0),
    "xl_mid":  dict(type="uniform", default=0.5375131244228907, low=0.2, high=0.8),

    # High-pressure damping knobs (as in Design 264)
    "p_cut": dict(type="log_uniform", default=5e4, low=1e3, high=1e6),
    "nu":    dict(type="uniform",     default=2.0, low=1.0, high=6.0),

    # NEW (single change vs Design 264): activate damping only near satisfaction
    # gate_c ~ 1 when c < c_damp_frac * gamma; ~0 otherwise.
    "c_damp_frac": dict(type="uniform",     default=0.55, low=0.10, high=1.20),
    "c_damp_sharp":dict(type="log_uniform", default=0.03, low=1e-3, high=0.20),

    "lr": dict(type="log_uniform", default=2.1085465692572183,            low=1e-1, high=3.0),
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

    # Top-2 backbone
    top_val, top_idx = torch.topk(v_lit, 2, dim=-1)          # (M,2)
    unsat_top = 0.5 * (1.0 - top_val)                        # (M,2)
    c = unsat_top[:, 0].clamp(0.0, 1.0)                      # (M,)

    # Top-2 routing
    G = c.unsqueeze(-1).repeat(1, 3)
    G.scatter_(1, top_idx[:, 0:1], unsat_top[:, 1:2])

    # Pressure p = xl*xs with near-sat-only damping (SINGLE CHANGE vs 264)
    pressure = (xl * xs).clamp_min(0.0)
    p_cut = torch.as_tensor(hp.p_cut, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    nu = torch.as_tensor(hp.nu, device=v.device, dtype=v.dtype).clamp_min(1e-3)

    gamma = torch.as_tensor(hp.gamma, device=v.device, dtype=v.dtype).clamp(0.0, 1.0)
    c_damp = (torch.as_tensor(hp.c_damp_frac, device=v.device, dtype=v.dtype).clamp_min(0.0) * gamma).clamp(0.0, 1.0)
    c_damp_sharp = torch.as_tensor(hp.c_damp_sharp, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    gate_c = torch.sigmoid((c_damp - c) / c_damp_sharp)  # ~1 when c is small (near satisfied)

    r = (pressure / p_cut).clamp_min(0.0)
    damp = 1.0 / (1.0 + gate_c * r.pow(nu))
    p_eff = pressure * damp
    G *= p_eff.unsqueeze(-1)

    # Rigidity (unchanged)
    R = torch.zeros_like(G)
    R.scatter_(1, top_idx[:, 0:1], c.unsqueeze(-1))
    R *= ((1.0 + hp.zeta * xl) * (1.0 - xs)).unsqueeze(-1)

    dv_clause = (G + R) * sgn
    dv = torch.zeros_like(v).scatter_add_(0, idx.reshape(-1), dv_clause.reshape(-1))

    # Memory updates (unchanged)
    dxl = hp.alpha * (c - hp.delta)

    eta = torch.as_tensor(hp.eta, device=v.device, dtype=v.dtype).clamp_min(0.0)
    band_sharp = torch.as_tensor(hp.band_sharp, device=v.device, dtype=v.dtype).clamp_min(1e-6)

    xl_thr = torch.as_tensor(hp.xl_thr, device=v.device, dtype=v.dtype).clamp_min(1.0)
    xl_sharp = torch.as_tensor(hp.xl_sharp, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    log_xl = torch.log(xl.clamp_min(1.0))
    log_thr = torch.log(xl_thr)
    gate_xl = torch.sigmoid((log_xl - log_thr) / xl_sharp)  # (0,1)

    # Weak-sat band: eta < c < gamma
    w_lo = torch.sigmoid((c - eta) / band_sharp)
    w_hi = torch.sigmoid((gamma - c) / band_sharp)
    weak_band = w_lo * w_hi

    xs_star = torch.as_tensor(hp.xs_star, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    push_up = torch.relu(xs_star - xs) / xs_star

    stag_margin = torch.as_tensor(hp.stag_margin, device=v.device, dtype=v.dtype).clamp_min(0.0)
    stag_sharp = torch.as_tensor(hp.stag_sharp, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    c0 = (gamma - stag_margin).clamp(0.0, 1.0)

    window = torch.sigmoid((c - c0) / stag_sharp)

    denom = (gamma - c0).clamp_min(1e-6)
    t = ((c - c0) / denom).clamp(0.0, 1.0)

    win_tau = torch.as_tensor(hp.win_tau, device=v.device, dtype=v.dtype).clamp(0.0, 0.95)
    t2 = ((t - win_tau) / (1.0 - win_tau).clamp_min(1e-6)).clamp(0.0, 1.0)

    s = t2 * t2 * t2 * (t2 * (t2 * 6.0 - 15.0) + 10.0)  # quintic smootherstep

    eps = torch.as_tensor(1e-12, device=v.device, dtype=v.dtype)
    sw = s / (s + (1.0 - s).pow(2) + eps)

    win_floor = torch.as_tensor(hp.win_floor, device=v.device, dtype=v.dtype).clamp(0.0, 1.0)
    win_pow = torch.as_tensor(hp.win_pow, device=v.device, dtype=v.dtype).clamp_min(0.05)
    amp = win_floor + (1.0 - win_floor) * sw.pow(win_pow)

    near_gamma = window * amp

    xl_gain = torch.as_tensor(hp.xl_gain, device=v.device, dtype=v.dtype).clamp_min(0.0)
    xl_mid = torch.as_tensor(hp.xl_mid, device=v.device, dtype=v.dtype).clamp(0.05, 0.95)

    hi = torch.relu(gate_xl - xl_mid) / (1.0 - xl_mid).clamp_min(1e-6)
    hi = hi.clamp(0.0, 1.0)
    release_gain = 1.0 + xl_gain * hi

    release = hp.kappa * gate_xl * release_gain * weak_band * near_gamma * push_up

    rhs_xs = (xs + hp.epsilon) * (c - gamma) + release
    dxs = hp.beta * rhs_xs

    # Scaling (unchanged)
    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))
    dv  = -dv  * scale
    dxl = -dxl * scale
    dxs = -dxs * scale

    return {"v": dv, "xl": dxl, "xs": dxs}
