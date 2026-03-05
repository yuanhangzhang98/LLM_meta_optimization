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
    # Start from the tuned stable defaults of Design 172
    "alpha":   dict(type="log_uniform", default=6.935818115471906,  low=0.5,  high=50.0),
    "beta":    dict(type="log_uniform", default=23.442527983330574, low=2.0,  high=200.0),
    "gamma":   dict(type="uniform",     default=0.2327165942916478, low=0.01, high=0.50),
    "delta":   dict(type="uniform",     default=0.06207747058084584, low=0.01, high=0.50),
    "epsilon": dict(type="log_uniform", default=0.00025726841385269225, low=1e-4, high=1e-2),
    "zeta":    dict(type="log_uniform", default=0.004342892475456097, low=1e-4, high=1.0),

    # Assist amplitude (as in 172/175)
    "eta":     dict(type="uniform",     default=1.6,   low=0.0,  high=2.0),

    # c-band parameters: narrow band-pass around the satisfaction boundary
    "c_thr":   dict(type="uniform",     default=0.53,  low=0.50, high=0.95),
    "band":    dict(type="uniform",     default=0.03,  low=0.005, high=0.10),
    "w":       dict(type="uniform",     default=0.015, low=0.005, high=0.10),

    # Pressure tail-decay (as in 175)
    "p_sat":   dict(type="log_uniform", default=500.0, low=0.3,  high=1e3),
    "nu":      dict(type="uniform",     default=2.5,   low=1.5,  high=6.0),

    # NEW: xs regime limit (turn assist off once xs is already high)
    "xs_thr":  dict(type="uniform",     default=0.7,   low=0.1,  high=0.95),
    "w_xs":    dict(type="uniform",     default=0.05,  low=0.01, high=0.25),

    "lr":      dict(type="log_uniform", default=0.75,  low=1e-1, high=3.0),
}


# ------------------------------------------------------------------ #
# 3. _grad_single
# ------------------------------------------------------------------ #

def _grad_single(vars, idx, sgn, hp):
    v  = vars["v"]   # (N,)
    xl = vars["xl"]  # (M,)
    xs = vars["xs"]  # (M,)

    # Literals
    v_lit = v[idx] * sgn  # (M,3)

    # Top-2 routing
    top_val, top_idx = torch.topk(v_lit, 2, dim=-1)  # (M,2)
    unsat_top = 0.5 * (1.0 - top_val)                # (M,2)
    c = unsat_top[:, 0]                              # (M,)

    # Baseline Top-2 gradient term
    G = c.unsqueeze(-1).repeat(1, 3)
    G.scatter_(1, top_idx[:, 0:1], unsat_top[:, 1:2])

    pressure = (xl * xs).clamp_min(0.0)              # (M,)
    G *= pressure.unsqueeze(-1)

    # --- Regime-limited super-sat assist (single principled modification) ---
    # Band-pass gate in c (only near boundary) + xs-gate (only before xs ramps),
    # plus super-saturating pressure tail-decay (Design 175 family).
    c_thr  = torch.as_tensor(hp.c_thr,  device=v.device, dtype=v.dtype)
    band   = torch.as_tensor(hp.band,   device=v.device, dtype=v.dtype).clamp_min(1e-6)
    w      = torch.as_tensor(hp.w,      device=v.device, dtype=v.dtype).clamp_min(1e-6)
    xs_thr = torch.as_tensor(hp.xs_thr, device=v.device, dtype=v.dtype)
    w_xs   = torch.as_tensor(hp.w_xs,   device=v.device, dtype=v.dtype).clamp_min(1e-6)
    p_sat  = torch.as_tensor(hp.p_sat,  device=v.device, dtype=v.dtype).clamp_min(1e-6)
    nu     = torch.as_tensor(hp.nu,     device=v.device, dtype=v.dtype).clamp_min(1.0001)

    # c band: (c_thr-band, c_thr+band)
    z_lo = ((c - (c_thr - band)) / w).clamp(-20.0, 20.0)
    z_hi = (((c_thr + band) - c) / w).clamp(-20.0, 20.0)
    gate_c = torch.sigmoid(z_lo) * torch.sigmoid(z_hi)  # (M,)

    # xs gate: on when xs < xs_thr
    z_xs = ((xs_thr - xs) / w_xs).clamp(-20.0, 20.0)
    gate_xs = torch.sigmoid(z_xs)  # (M,)

    gate = gate_c * gate_xs

    denom = (pressure + p_sat).clamp_min(1e-6)
    p_eff = pressure * (p_sat / denom).pow(nu)  # ->0 as pressure->inf

    assist = torch.zeros_like(G)
    assist.scatter_(1, top_idx[:, 1:2], 1.0)  # runner-up literal only
    assist *= (hp.eta * c * gate * p_eff).unsqueeze(-1)

    G = G + assist

    # Baseline rigidity term
    R = torch.zeros_like(G)
    R.scatter_(1, top_idx[:, 0:1], c.unsqueeze(-1))
    R *= ((1.0 + hp.zeta * xl) * (1.0 - xs)).unsqueeze(-1)

    # Accumulate v gradient
    dv_clause = (G + R) * sgn
    dv = torch.zeros_like(v).scatter_add_(0, idx.reshape(-1), dv_clause.reshape(-1))

    # Memory updates (baseline)
    dxl = hp.alpha * (c - hp.delta)
    dxs = hp.beta  * ((xs + hp.epsilon) * (c - hp.gamma))

    # Normalize (baseline)
    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))

    dv  = -dv  * scale
    dxl = -dxl * scale
    dxs = -dxs * scale

    return {"v": dv, "xl": dxl, "xs": dxs}
