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
    "alpha":   dict(type="log_uniform", default=4.902628216330176,   low=0.5,  high=50.0),
    "beta":    dict(type="log_uniform", default=35.97135632573113,   low=2.0,  high=200.0),
    "gamma":   dict(type="uniform",     default=0.21740654619087363, low=0.01, high=0.50),
    "delta":   dict(type="uniform",     default=0.05000000074505806, low=0.01, high=0.50),
    "epsilon": dict(type="log_uniform", default=0.0010004464195963476, low=1e-4, high=1e-2),
    "zeta":    dict(type="log_uniform", default=0.001104662591564918, low=1e-4, high=1.0),

    # Asymmetric xs switching (Design-21)
    "eta":     dict(type="uniform",     default=0.2776329339570298,  low=0.0,  high=3.0),

    # Satisfied-only off-floor magnitude (Design-24)
    "kappa":   dict(type="uniform",     default=0.02, low=0.0, high=0.25),

    # Smooth satisfied-only gate scale (this design only changes the gate SHAPE)
    "tau_sat": dict(type="log_uniform", default=1e-2, low=1e-5, high=2e-1),

    "lr":      dict(type="log_uniform", default=1.1226282028677665, low=1e-1, high=3.0),
}


# ------------------------------------------------------------------ #
# 3. _grad_single
# ------------------------------------------------------------------ #

def _grad_single(vars, idx, sgn, hp):
    v  = vars["v"]
    xl = vars["xl"]
    xs = vars["xs"]

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

    # xs update: asymmetric switch + satisfied-only kappa floor with ZERO-LEAK smooth gate
    pos = torch.relu(c - hp.gamma)            # violated beyond gamma
    neg = torch.relu(hp.gamma - c)            # satisfied beyond gamma (>=0)

    tau = torch.as_tensor(hp.tau_sat, device=c.device, dtype=c.dtype).clamp(min=1e-12)

    # NEW (single change vs Design-30): sharper satisfied-only gate, exactly 0 at neg=0
    # gate = sqrt(neg/(neg+tau)) in [0,1), rises faster than exp gate for small neg.
    frac = neg / (neg + tau)
    gate = torch.sqrt(frac.clamp(min=0.0, max=1.0))

    off_drive = neg + hp.kappa * gate

    beta_on = hp.beta
    beta_off = hp.beta * (1.0 + hp.eta)
    dxs = beta_on * (1.0 - xs) * pos - beta_off * (xs + hp.epsilon) * off_drive

    # ===== GRADIENT SCALING (baseline) =====
    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))

    dv  = -dv  * scale
    dxl = -dxl * scale
    dxs = -dxs * scale

    return {"v": dv, "xl": dxl, "xs": dxs}
