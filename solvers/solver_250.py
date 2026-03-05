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
    # carried-over tuned defaults (Design 61/207 family)
    "alpha":   dict(type="log_uniform", default=5.628430483159852,      low=0.5,  high=50.0),
    "beta":    dict(type="log_uniform", default=40.43498719903352,      low=2.0,  high=200.0),
    "gamma":   dict(type="uniform",     default=0.21765848650443836,    low=0.01, high=0.50),
    "delta":   dict(type="uniform",     default=0.09869929008180485,    low=0.01, high=0.50),
    "epsilon": dict(type="log_uniform", default=0.0009807063473973208,  low=1e-4, high=1e-2),
    "zeta":    dict(type="log_uniform", default=0.0010498238170223528,  low=1e-4, high=1.0),

    # asymmetric xs knobs
    "eta":      dict(type="uniform",    default=0.3794440949062971,     low=0.0,  high=3.0),
    "kappa":    dict(type="uniform",    default=0.019776372159554443,   low=0.0,  high=0.25),
    "eta_hold": dict(type="uniform",    default=0.2999995351252482,     low=0.0,  high=3.0),

    # satisfied-side gate (zero-leak, saturating)
    "tau_sat": dict(type="log_uniform", default=0.010000370298734167,   low=1e-5, high=2e-1),
    "p_sat":   dict(type="uniform",     default=0.9842825158415519,     low=0.5,  high=3.0),

    # violated-side boundary reactivation strength + bump tail
    "kappa_on": dict(type="uniform",    default=0.04999601539141226,    low=0.0,  high=0.25),
    "a_on":     dict(type="log_uniform", default=1.0,                   low=0.3,  high=10.0),

    # NEW (single change vs 207): smooth tie-gate for reactivation bump
    # gate_tie = 1/(1+(gap/g_tie)^p_tie), gap = (2nd_best_u - best_u)
    "g_tie":    dict(type="log_uniform", default=0.003,                 low=1e-4, high=5e-2),
    "p_tie":    dict(type="uniform",     default=2.0,                   low=1.0,  high=6.0),

    "lr":      dict(type="log_uniform", default=1.7694676919282437,     low=1e-1, high=3.0),
}


# ------------------------------------------------------------------ #
# 3. _grad_single
# ------------------------------------------------------------------ #

def _grad_single(vars, idx, sgn, hp):
    v  = vars["v"]
    xl = vars["xl"]
    xs = vars["xs"]

    # ===== CLAUSE EVALUATION (baseline hard top-2) =====
    v_lit = v[idx] * sgn                       # (M,3)
    top_val, top_idx = torch.topk(v_lit, 2, dim=-1)
    unsat_top = 0.5 * (1.0 - top_val)          # (M,2) ; u1<=u2
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

    # xs update (Design 207 + ONE small modification: tie-gate the reactivation bump)
    pos = torch.relu(c - hp.gamma)   # violated beyond gamma
    neg = torch.relu(hp.gamma - c)   # satisfied beyond gamma

    tau = torch.as_tensor(hp.tau_sat, device=c.device, dtype=c.dtype).clamp(min=1e-12)
    p   = torch.as_tensor(hp.p_sat,   device=c.device, dtype=c.dtype).clamp(min=0.5)

    # satisfied-side zero-leak gate
    frac = tau / (neg + tau)
    gate_sat = 1.0 - torch.pow(frac.clamp(0.0, 1.0), p)  # gate_sat(0)=0 exactly

    beta_on   = hp.beta
    beta_off  = hp.beta * (1.0 + hp.eta)
    beta_hold = hp.beta * (1.0 + hp.eta_hold)

    # violated-side boundary bump: b(y)=(y/(1+y))^2 * exp(-a_on*y)
    y = pos / tau
    a_on = torch.as_tensor(hp.a_on, device=c.device, dtype=c.dtype).clamp(min=1e-6)
    bump_base = torch.square(y / (1.0 + y)) * torch.exp(-a_on * y)

    # NEW: tie-gate based on gap between best and 2nd-best literals in the clause
    gap_u = (unsat_top[:, 1] - unsat_top[:, 0]).clamp(min=0.0)
    g_tie = torch.as_tensor(hp.g_tie, device=c.device, dtype=c.dtype).clamp(min=1e-12)
    p_tie = torch.as_tensor(hp.p_tie, device=c.device, dtype=c.dtype).clamp(min=1.0)
    tie_gate = 1.0 / (1.0 + torch.pow(gap_u / g_tie, p_tie))

    bump = bump_base * tie_gate

    on_drive  = pos + hp.kappa_on * bump
    off_drive = (beta_off * neg) + (beta_hold * (hp.kappa * gate_sat))

    dxs = beta_on * (1.0 - xs) * on_drive \
        - (xs + hp.epsilon) * off_drive

    # ===== GRADIENT SCALING (baseline) =====
    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))

    return {
        "v":  -dv  * scale,
        "xl": -dxl * scale,
        "xs": -dxs * scale,
    }
