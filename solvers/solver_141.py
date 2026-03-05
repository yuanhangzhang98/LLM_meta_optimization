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
    "alpha":     dict(type="log_uniform", default=5.0,   low=0.5,  high=50.0),
    "beta":      dict(type="log_uniform", default=20.0,  low=2.0,  high=200.0),
    "gamma":     dict(type="uniform",     default=0.25,  low=0.01, high=0.50),
    "delta":     dict(type="uniform",     default=0.05,  low=0.01, high=0.50),
    "epsilon":   dict(type="log_uniform", default=1e-3,  low=1e-4, high=1e-2),
    "zeta":      dict(type="log_uniform", default=1e-3,  low=1e-4, high=1.0),

    # Coupled normalization knobs (as in Designs 24/32/41)
    "kappa":     dict(type="log_uniform", default=0.1,   low=1e-3, high=1.0),
    "scale_cap": dict(type="log_uniform", default=100.0, low=1.0,  high=1e4),

    # Robust magnitude estimator (Design 32)
    "p":         dict(type="uniform",     default=8.0,   low=2.0,  high=32.0),

    # NEW (single conceptual change vs Design 32): smooth-max coupling sharpness
    # eta -> inf recovers hard max; moderate eta avoids kink while staying near max.
    "eta":       dict(type="log_uniform", default=10.0,  low=0.5,  high=100.0),

    "lr":        dict(type="log_uniform", default=1.0,   low=1e-1, high=3.0),
}


# ------------------------------------------------------------------ #
# 3. _grad_single
# ------------------------------------------------------------------ #

def _grad_single(vars, idx, sgn, hp):
    v = vars["v"]
    xl = vars["xl"]
    xs = vars["xs"]

    # Clause evaluation
    v_lit = v[idx] * sgn  # (M,3)
    top_val, top_idx = torch.topk(v_lit, 2, dim=-1)
    unsat_top = 0.5 * (1.0 - top_val)  # (M,2)
    c = unsat_top[:, 0]                # (M,)

    # Gradient term
    G = c.unsqueeze(-1).repeat(1, 3)
    G.scatter_(1, top_idx[:, 0:1], unsat_top[:, 1:2])
    G *= (xl * xs).unsqueeze(-1)

    # Rigidity term
    R = torch.zeros_like(G)
    R.scatter_(1, top_idx[:, 0:1], c.unsqueeze(-1))
    R *= ((1.0 + hp.zeta * xl) * (1.0 - xs)).unsqueeze(-1)

    # Accumulate v gradient
    dv_clause = (G + R) * sgn
    dv = torch.zeros_like(v).scatter_add_(0, idx.reshape(-1), dv_clause.reshape(-1))

    # Memory gradients (baseline)
    dxl = hp.alpha * (c - hp.delta)
    dxs = hp.beta  * ((xs + hp.epsilon) * (c - hp.gamma))

    # ---- Robust stats + capped coupled normalization; smooth-max coupling ----
    p = torch.as_tensor(hp.p, device=v.device, dtype=v.dtype)

    def pnorm_stat(x):
        ax = x.abs().clamp(min=1e-12)
        return ax.pow(p).mean().pow(1.0 / p)

    dv_stat = pnorm_stat(dv)
    mem_stat = torch.maximum(pnorm_stat(dxl), pnorm_stat(dxs))

    eta = torch.as_tensor(hp.eta, device=v.device, dtype=v.dtype).clamp(min=1e-6)
    a = dv_stat
    b = torch.as_tensor(hp.kappa, device=v.device, dtype=v.dtype) * mem_stat

    # smoothmax(a,b) = (1/eta) * log(exp(eta a) + exp(eta b))
    denom = torch.logsumexp(torch.stack([eta * a, eta * b]), dim=0) / eta
    denom = denom.clamp(min=1e-6)

    scale = 1.0 / denom
    scale_cap = torch.as_tensor(hp.scale_cap, device=v.device, dtype=v.dtype)
    scale = torch.minimum(scale, scale_cap)

    dv = -dv * scale
    dxl = -dxl * scale
    dxs = -dxs * scale

    return {"v": dv, "xl": dxl, "xs": dxs}
