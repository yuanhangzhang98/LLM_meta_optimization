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
    "alpha":   dict(type="log_uniform", default=4.902628216330176,  low=0.5,  high=50.0),
    "beta":    dict(type="log_uniform", default=35.97135632573113,  low=2.0,  high=200.0),
    "gamma":   dict(type="uniform",     default=0.21740654619087363, low=0.01, high=0.50),
    "delta":   dict(type="uniform",     default=0.05000000074505806, low=0.01, high=0.50),
    "epsilon": dict(type="log_uniform", default=0.0010004464195963476, low=1e-4, high=1e-2),
    "zeta":    dict(type="log_uniform", default=0.001104662591564918, low=1e-4, high=1.0),

    # From Design-21: explicit off-rate multiplier
    "eta":     dict(type="uniform",     default=0.2776329339570298,  low=0.0,  high=3.0),

    # NEW: hysteresis half-width around gamma (deadzone). kappa=0 reduces to Design-21.
    "kappa":   dict(type="uniform",     default=0.03,               low=0.0,  high=0.25),

    "lr":      dict(type="log_uniform", default=1.1226282028677665,  low=1e-1, high=3.0),
}


# ------------------------------------------------------------------ #
# 3.  _grad_single
# ------------------------------------------------------------------ #

def _grad_single(vars, idx, sgn, hp):
    v  = vars["v"]   # (N,)
    xl = vars["xl"]  # (M,)
    xs = vars["xs"]  # (M,)

    # ===== CLAUSE EVALUATION (baseline) =====
    v_lit = v[idx] * sgn                       # (M,3)
    top_val, top_idx = torch.topk(v_lit, 2, dim=-1)
    unsat_top = 0.5 * (1.0 - top_val)          # (M,2)
    c = unsat_top[:, 0]                        # (M,)

    # ===== GRADIENT TERM (baseline) =====
    G = c.unsqueeze(-1).repeat(1, 3)
    G.scatter_(1, top_idx[:, 0:1], unsat_top[:, 1:2])
    G *= (xl * xs).unsqueeze(-1)

    # ===== RIGIDITY TERM (baseline) =====
    R = torch.zeros_like(G)
    R.scatter_(1, top_idx[:, 0:1], c.unsqueeze(-1))
    R *= ((1.0 + hp.zeta * xl) * (1.0 - xs)).unsqueeze(-1)

    # ===== ACCUMULATE v GRADIENT (baseline) =====
    dv_clause = (G + R) * sgn
    dv = torch.zeros_like(v).scatter_add_(0, idx.reshape(-1), dv_clause.reshape(-1))

    # ===== AUXILIARY VARIABLE GRADIENTS =====
    dxl = hp.alpha * (c - hp.delta)

    # xs update: Design-21 asymmetric on/off + NEW hysteresis/deadzone around gamma
    # turn ON if c > gamma + kappa; turn OFF if c < gamma - kappa
    kappa = torch.clamp(torch.as_tensor(hp.kappa, device=c.device, dtype=c.dtype), min=0.0)
    gamma_on = torch.clamp(torch.as_tensor(hp.gamma, device=c.device, dtype=c.dtype) + kappa, 0.0, 1.0)
    gamma_off = torch.clamp(torch.as_tensor(hp.gamma, device=c.device, dtype=c.dtype) - kappa, 0.0, 1.0)

    pos = torch.relu(c - gamma_on)
    neg = torch.relu(gamma_off - c)

    beta_on = hp.beta
    beta_off = hp.beta * (1.0 + hp.eta)
    dxs = beta_on * (1.0 - xs) * pos - beta_off * (xs + hp.epsilon) * neg

    # ===== GRADIENT SCALING (baseline) =====
    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))

    dv  = -dv  * scale
    dxl = -dxl * scale
    dxs = -dxs * scale

    return {"v": dv, "xl": dxl, "xs": dxs}
