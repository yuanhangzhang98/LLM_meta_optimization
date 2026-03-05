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
    "alpha":   dict(type="log_uniform", default=5.307646632051222,    low=0.5,  high=50.0),
    "beta":    dict(type="log_uniform", default=37.16488566644089,    low=2.0,  high=200.0),
    "gamma":   dict(type="uniform",     default=0.21761870876010456,  low=0.01, high=0.50),
    "delta":   dict(type="uniform",     default=0.09811872833778353,  low=0.01, high=0.50),
    "epsilon": dict(type="log_uniform", default=0.0010004464125052414, low=1e-4, high=1e-2),
    "zeta":    dict(type="log_uniform", default=0.0010773189704724644, low=1e-4, high=1.0),

    # Proven xs control knobs (Design-21/24)
    "eta":     dict(type="uniform",     default=0.2776329219341278,   low=0.0,  high=3.0),
    "kappa":   dict(type="uniform",     default=0.019786201968549934,                 low=0.0,  high=0.25),

    # NEW: satisfied-only, zero-leak gate family (single targeted change)
    "tau_sat": dict(type="log_uniform", default=0.009999886594658952,                 low=1e-5, high=2e-1),
    "p_sat":   dict(type="uniform",     default=1.0025796707037216,                  low=0.5,  high=3.0),

    "lr":      dict(type="log_uniform", default=1.6544256711641443,   low=1e-1, high=3.0),
}


# ------------------------------------------------------------------ #
# 3. _grad_single
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

    # xs update: asymmetric switch + satisfied-only kappa floor with ZERO-LEAK gate
    pos = torch.relu(c - hp.gamma)            # violated beyond gamma
    neg = torch.relu(hp.gamma - c)            # satisfied beyond gamma

    tau = torch.as_tensor(hp.tau_sat, device=c.device, dtype=c.dtype).clamp(min=1e-12)
    p = torch.as_tensor(hp.p_sat, device=c.device, dtype=c.dtype).clamp(min=0.5)

    # NEW (single change): complement-power gate
    # gate = 1 - (tau/(neg+tau))^p
    # Properties: gate(0)=0 exactly (zero-leak); finite, tunable boundary slope ~ p/tau; saturates to 1.
    frac = tau / (neg + tau)
    gate = 1.0 - torch.pow(frac.clamp(0.0, 1.0), p)

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
