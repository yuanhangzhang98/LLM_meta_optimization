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

    # Base temperature for soft routing (only used in v-forces, NOT in memory cost c)
    "tau":     dict(type="log_uniform", default=5e-2, low=1e-3, high=5e-1),

    # Annealing strength: tau_eff = tau / (1 + kappa * log1p(xl*xs))
    "kappa":   dict(type="log_uniform", default=3e-1, low=1e-3, high=10.0),

    "lr":      dict(type="log_uniform", default=1.0,  low=1e-1, high=3.0),
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

    # --- Hard clause cost c for memory updates (baseline behavior) ---
    top_val, top_idx = torch.topk(v_lit, 2, dim=-1)
    unsat_top = 0.5 * (1.0 - top_val)  # (M,2)
    c = unsat_top[:, 0]                # (M,)

    # Unsatisfaction per literal u in [0,1]
    u = 0.5 * (1.0 - v_lit)            # (M,3)

    # --- Annealed soft routing temperature for v-forces only ---
    tau0 = torch.as_tensor(hp.tau, device=v.device, dtype=v.dtype).clamp(min=1e-6)
    kappa = torch.as_tensor(hp.kappa, device=v.device, dtype=v.dtype)

    pressure = xl * xs  # (M,) clause emphasis when violated
    tau_eff = tau0 / (1.0 + kappa * torch.log1p(pressure))
    tau_eff = tau_eff.clamp(min=1e-4)

    def _softmin2(a, b, tau_vec):
        # a,b,tau_vec: (M,)
        t = tau_vec.unsqueeze(-1)  # (M,1)
        logits = torch.stack((-a, -b), dim=-1) / t
        return -tau_vec * torch.logsumexp(logits, dim=-1)

    # Soft min over the other two literals (smooths force distribution)
    u_min_other = torch.stack(
        (
            _softmin2(u[:, 1], u[:, 2], tau_eff),
            _softmin2(u[:, 0], u[:, 2], tau_eff),
            _softmin2(u[:, 0], u[:, 1], tau_eff),
        ),
        dim=-1,
    )  # (M,3)

    # Soft "winner" routing for rigidity (approaches hard top-1 as tau_eff shrinks)
    w = torch.softmax(v_lit / tau_eff.unsqueeze(-1), dim=-1)  # (M,3)

    # Baseline weighting structure
    G = u_min_other * (xl * xs).unsqueeze(-1)
    R = (w * c.unsqueeze(-1)) * ((1.0 + hp.zeta * xl) * (1.0 - xs)).unsqueeze(-1)

    dv_clause = (G + R) * sgn
    dv = torch.zeros_like(v).scatter_add_(0, idx.reshape(-1), dv_clause.reshape(-1))

    # Memory updates still driven by HARD c
    dxl = hp.alpha * (c - hp.delta)
    dxs = hp.beta  * ((xs + hp.epsilon) * (c - hp.gamma))

    # Normalize (baseline)
    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))

    dv  = -dv  * scale
    dxl = -dxl * scale
    dxs = -dxs * scale

    return {"v": dv, "xl": dxl, "xs": dxs}
