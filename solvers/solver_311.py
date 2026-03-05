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
    "alpha":   dict(type="log_uniform", default=6.15,  low=0.5,  high=50.0),
    "beta":    dict(type="log_uniform", default=23.5,  low=2.0,  high=200.0),
    "gamma":   dict(type="uniform",     default=0.236, low=0.01, high=0.50),
    "delta":   dict(type="uniform",     default=0.062, low=0.01, high=0.50),
    "epsilon": dict(type="log_uniform", default=3.9e-4, low=1e-4, high=1e-2),
    "zeta":    dict(type="log_uniform", default=4.2e-3, low=1e-4, high=1.0),

    # Assist (Design-175 family)
    "eta":     dict(type="uniform",     default=1.6,   low=0.0,  high=2.0),
    "c_thr":   dict(type="uniform",     default=0.53,  low=0.50, high=0.95),
    "w":       dict(type="uniform",     default=0.015, low=0.005, high=0.25),
    "p_sat":   dict(type="log_uniform", default=500.0, low=0.3,  high=1e3),

    # Base decay exponent (as in 175)
    "nu":      dict(type="uniform",     default=2.0,   low=1.5,  high=6.0),

    # NEW (single conceptual change vs 175): high-pressure exponent boost
    # Effective exponent: nu + mu * (p/(p+p_sat)).  mu=0 reproduces 175.
    "mu":      dict(type="uniform",     default=0.0,   low=0.0,  high=8.0),

    "lr":      dict(type="log_uniform", default=0.79,  low=1e-1, high=3.0),
}


# ------------------------------------------------------------------ #
# 3.  _grad_single
# ------------------------------------------------------------------ #

def _grad_single(vars, idx, sgn, hp):
    v  = vars["v"]
    xl = vars["xl"]
    xs = vars["xs"]

    v_lit = v[idx] * sgn  # (M,3)

    top_val, top_idx = torch.topk(v_lit, 2, dim=-1)  # (M,2)
    unsat_top = 0.5 * (1.0 - top_val)                # (M,2)
    c = unsat_top[:, 0]                              # (M,)

    # Baseline Top-2 gradient backbone
    G = c.unsqueeze(-1).repeat(1, 3)
    G.scatter_(1, top_idx[:, 0:1], unsat_top[:, 1:2])

    pressure = (xl * xs).clamp_min(0.0)
    G *= pressure.unsqueeze(-1)

    # Assist: smooth violation gate + variable-exponent super-saturation
    c_thr = torch.as_tensor(hp.c_thr, device=v.device, dtype=v.dtype)
    w     = torch.as_tensor(hp.w,     device=v.device, dtype=v.dtype).clamp_min(1e-6)
    p_sat = torch.as_tensor(hp.p_sat, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    nu    = torch.as_tensor(hp.nu,    device=v.device, dtype=v.dtype).clamp_min(1.0001)
    mu    = torch.as_tensor(hp.mu,    device=v.device, dtype=v.dtype).clamp_min(0.0)

    z = ((c - c_thr) / w).clamp(-20.0, 20.0)
    gate = torch.sigmoid(z)

    denom = (pressure + p_sat).clamp_min(1e-6)
    s = (pressure / denom).clamp(0.0, 1.0)  # smoothly approaches 1 at high pressure

    # NEW vs 175: exponent increases with pressure (no tail floor; still -> 0 as p->inf)
    expn = (nu + mu * s).clamp(1.0001, 20.0)
    base = (p_sat / denom).clamp(1e-6, 1.0)
    p_eff = pressure * torch.pow(base, expn)

    assist = torch.zeros_like(G)
    assist.scatter_(1, top_idx[:, 1:2], 1.0)  # runner-up literal
    assist *= (hp.eta * c * gate * p_eff).unsqueeze(-1)

    G = G + assist

    # Baseline rigidity term
    R = torch.zeros_like(G)
    R.scatter_(1, top_idx[:, 0:1], c.unsqueeze(-1))
    R *= ((1.0 + hp.zeta * xl) * (1.0 - xs)).unsqueeze(-1)

    dv_clause = (G + R) * sgn
    dv = torch.zeros_like(v).scatter_add_(0, idx.reshape(-1), dv_clause.reshape(-1))

    dxl = hp.alpha * (c - hp.delta)
    dxs = hp.beta  * ((xs + hp.epsilon) * (c - hp.gamma))

    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))

    return {
        "v":  -dv  * scale,
        "xl": -dxl * scale,
        "xs": -dxs * scale,
    }
