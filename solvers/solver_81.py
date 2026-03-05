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
    # Design-11 tuned defaults (strong baseline)
    "alpha":   dict(type="log_uniform", default=48.3997327918925,      low=0.5,  high=50.0),
    "beta":    dict(type="log_uniform", default=40.08506794497957,     low=2.0,  high=200.0),
    "gamma":   dict(type="uniform",     default=0.28948339187872574,   low=0.01, high=0.50),
    "delta":   dict(type="uniform",     default=0.07327583432197571,   low=0.01, high=0.50),
    "epsilon": dict(type="log_uniform", default=0.001468934335764333,  low=1e-4, high=1e-2),
    "zeta":    dict(type="log_uniform", default=0.0013418985396236333, low=1e-4, high=1.0),
    "lr":      dict(type="log_uniform", default=1.6536008035680871,    low=1e-1, high=3.0),

    # Tie-only smoothing knobs (Design-11/75)
    "tau_tie": dict(type="log_uniform", default=0.01712294456377018,   low=1e-3, high=2e-1),
    "g0":      dict(type="log_uniform", default=0.0033876327188782085, low=1e-3, high=2e-1),
    "kappa":   dict(type="log_uniform", default=0.052906428769193214,  low=1e-3, high=1e-1),

    # NEW (single change vs Design-75): probability floor in tie softmax
    # eta_p=0 recovers Design-75. Larger eta_p => more conservative (closer to mean of u1,u2).
    "eta_p":   dict(type="uniform",     default=0.12,                  low=0.0,  high=0.5),
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

    # --- Hard top-k structure for v-updates (unchanged) ---
    top_val, top_idx = torch.topk(v_lit, 2, dim=-1)  # (M,2)
    unsat_top = 0.5 * (1.0 - top_val)                # (M,2)
    c_hard = unsat_top[:, 0]                         # (M,)

    G = c_hard.unsqueeze(-1).repeat(1, 3)
    G.scatter_(1, top_idx[:, 0:1], unsat_top[:, 1:2])
    G *= (xl * xs).unsqueeze(-1)

    R = torch.zeros_like(G)
    R.scatter_(1, top_idx[:, 0:1], c_hard.unsqueeze(-1))
    R *= ((1.0 + hp.zeta * xl) * (1.0 - xs)).unsqueeze(-1)

    dv_clause = (G + R) * sgn
    dv = torch.zeros_like(v).scatter_add_(0, idx.reshape(-1), dv_clause.reshape(-1))

    # --- Memory monitor for dxs: hard-min almost everywhere, smooth only near ties ---
    u = 0.5 * (1.0 - v_lit)  # (M,3) in [0,1]
    u12, _ = torch.topk(u, 2, dim=-1, largest=False)  # (M,2) = (u1,u2)
    u1 = u12[:, 0]
    gap = (u12[:, 1] - u12[:, 0]).clamp(min=0.0)

    w = torch.sigmoid((hp.g0 - gap) / (hp.kappa + 1e-12)).clamp(0.0, 1.0)

    # Design-75: bounded top-2 expectation; SINGLE MOD here: probability floor
    tau = hp.tau_tie + 1e-12
    p = torch.softmax(-u12 / tau, dim=-1)            # (M,2)
    eta = hp.eta_p
    p = (1.0 - eta) * p + eta * 0.5                  # (M,2), still sums to 1
    c_soft = (p * u12).sum(dim=-1)                   # in [u1,u2]

    c_mem = ((1.0 - w) * u1 + w * c_soft).clamp(0.0, 1.0)

    dxl = hp.alpha * (c_hard - hp.delta)
    dxs = hp.beta  * ((xs + hp.epsilon) * (c_mem - hp.gamma))

    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))

    return {
        "v":  -dv  * scale,
        "xl": -dxl * scale,
        "xs": -dxs * scale,
    }
