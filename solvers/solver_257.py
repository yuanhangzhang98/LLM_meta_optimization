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
    "alpha":   dict(type="log_uniform", default=6.935818115471906,  low=0.5,  high=50.0),
    "beta":    dict(type="log_uniform", default=23.442527983330574, low=2.0,  high=200.0),
    "gamma":   dict(type="uniform",     default=0.2327165942916478, low=0.01, high=0.50),
    "delta":   dict(type="uniform",     default=0.06207747058084584, low=0.01, high=0.50),
    "epsilon": dict(type="log_uniform", default=0.00025726841385269225, low=1e-4, high=1e-2),
    "zeta":    dict(type="log_uniform", default=0.004342892475456097, low=1e-4, high=1.0),

    # Assist (Design-172 family): smooth c-gate + bounded/decaying effective pressure
    "eta":     dict(type="uniform",     default=1.60,   low=0.0,  high=2.0),
    "c_thr":   dict(type="uniform",     default=0.513,  low=0.50, high=0.95),
    "w":       dict(type="uniform",     default=0.015,  low=0.005, high=0.25),

    # NEW (single principled change vs Design-172): explicit-peak, tunable-tail p_eff
    # p_peak controls where the assist pressure peaks; k_tail controls high-pressure decay ~ p^{-k_tail}.
    "p_peak":  dict(type="log_uniform", default=500.0,  low=0.3,  high=1e3),
    "k_tail":  dict(type="uniform",     default=1.0,    low=0.25, high=4.0),

    "lr":      dict(type="log_uniform", default=0.754737194393524,  low=1e-1, high=3.0),
}


# ------------------------------------------------------------------ #
# 3.  _grad_single
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

    # Assist: smooth violation gate + explicit-peak, tunable-tail effective pressure
    c_thr  = torch.as_tensor(hp.c_thr,  device=v.device, dtype=v.dtype)
    w      = torch.as_tensor(hp.w,      device=v.device, dtype=v.dtype).clamp_min(1e-6)
    p_peak = torch.as_tensor(hp.p_peak, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    k_tail = torch.as_tensor(hp.k_tail, device=v.device, dtype=v.dtype).clamp_min(1e-3)

    z = ((c - c_thr) / w).clamp(-20.0, 20.0)
    gate = torch.sigmoid(z)  # (M,)

    # Peak-shaped map with explicit peak location:
    # base shape: p_eff = p / (1 + (p/p0)^(1+k))  => low-p linear, high-p ~ p^{-k}
    # choose p0 so the maximum occurs at p = p_peak.
    inv = 1.0 / (1.0 + k_tail)
    p0 = p_peak * torch.pow(k_tail, inv)  # ensures argmax at p_peak

    ratio = (pressure / p0).clamp_min(0.0).clamp_max(1e6)
    denom = 1.0 + torch.pow(ratio, 1.0 + k_tail)
    p_eff = pressure / denom

    assist = torch.zeros_like(G)
    assist.scatter_(1, top_idx[:, 1:2], 1.0)  # runner-up literal
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
