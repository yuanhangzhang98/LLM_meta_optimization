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
    "alpha":   dict(type="log_uniform", default=5.0,  low=0.5,  high=50.0),
    "beta":    dict(type="log_uniform", default=20.0, low=2.0,  high=200.0),
    "gamma":   dict(type="uniform",     default=0.25, low=0.01, high=0.50),
    "delta":   dict(type="uniform",     default=0.05, low=0.01, high=0.50),
    "epsilon": dict(type="log_uniform", default=1e-3, low=1e-4, high=1e-2),
    "zeta":    dict(type="log_uniform", default=1e-3, low=1e-4, high=1.0),

    # Smooth violation-triggered Top-2 assist (as in Design 9)
    "eta":     dict(type="uniform",     default=0.25, low=0.0,  high=2.0),
    "c_thr":   dict(type="uniform",     default=0.60, low=0.50, high=0.95),
    "tau":     dict(type="log_uniform", default=0.05, low=1e-2, high=0.3),

    # NEW (single principled addition): per-clause cap on assist magnitude
    # assist_mag <= kappa * base_mag
    "kappa":   dict(type="uniform",     default=1.0,  low=0.0,  high=3.0),

    "lr":      dict(type="log_uniform", default=1.0,  low=1e-1, high=3.0),
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

    # Top-2 routing (baseline)
    top_val, top_idx = torch.topk(v_lit, 2, dim=-1)  # (M,2), (M,2)
    unsat_top = 0.5 * (1.0 - top_val)                # (M,2)
    c = unsat_top[:, 0]                              # (M,)

    p = xl * xs  # (M,)

    # ===== Baseline gradient term =====
    G_base = c.unsqueeze(-1).repeat(1, 3)
    G_base.scatter_(1, top_idx[:, 0:1], unsat_top[:, 1:2])
    G_base = G_base * p.unsqueeze(-1)

    # ===== Smooth-gated assist (runner-up literal) with explicit magnitude cap =====
    c_thr = torch.as_tensor(hp.c_thr, device=v.device, dtype=v.dtype)
    tau   = torch.as_tensor(hp.tau,   device=v.device, dtype=v.dtype)
    gate = torch.sigmoid((c - c_thr) / tau.clamp_min(1e-6))  # (M,)

    assist_raw = torch.zeros_like(G_base)
    assist_raw.scatter_(1, top_idx[:, 1:2], 1.0)
    assist_raw = assist_raw * (hp.eta * c * gate).unsqueeze(-1) * p.unsqueeze(-1)

    # Per-clause cap: ||assist||_1 <= kappa * ||G_base||_1
    base_mag = G_base.abs().sum(dim=-1)                 # (M,)
    assist_mag = assist_raw.abs().sum(dim=-1)           # (M,)
    cap = (hp.kappa * base_mag) / (assist_mag + 1e-12)  # (M,)
    cap = cap.clamp(max=1.0)
    assist = assist_raw * cap.unsqueeze(-1)

    G = G_base + assist

    # ===== Rigidity term (baseline) =====
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
