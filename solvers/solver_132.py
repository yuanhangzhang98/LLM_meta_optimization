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
    # tie-only smoothing controls
    "tau":         dict(type="log_uniform", default=0.03, low=1e-3, high=0.3),
    "tie_margin":  dict(type="uniform",     default=0.10, low=0.00, high=0.6),
    "tie_sharp":   dict(type="log_uniform", default=0.02, low=1e-3, high=0.2),
    "lr":      dict(type="log_uniform", default=1.0, low=1e-1, high=3.0),
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

    # Hard top-2 (keep baseline hard clause cost)
    top_val, top_idx = torch.topk(v_lit, 2, dim=-1)  # (M,2)
    unsat_top = 0.5 * (1.0 - top_val)                # (M,2)
    c = unsat_top[:, 0]                              # (M,)

    # --- Hybrid (hard + tie-only soft) routing over TOP-2 only ---
    # margin big => baseline hard routing; margin small => soft split over top-2
    margin = (top_val[:, 0] - top_val[:, 1]).clamp_min(0.0)  # (M,)
    tie_margin = torch.as_tensor(hp.tie_margin, device=v.device, dtype=v.dtype)
    tie_sharp  = torch.as_tensor(hp.tie_sharp,  device=v.device, dtype=v.dtype).clamp_min(1e-6)
    s = torch.sigmoid((tie_margin - margin) / tie_sharp)  # (M,) in [0,1]

    tau = torch.as_tensor(hp.tau, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    w2 = torch.softmax(top_val / tau, dim=-1)  # (M,2)

    onehot_top1 = torch.zeros_like(v_lit).scatter(1, top_idx[:, 0:1], 1.0)  # (M,3)
    w2_3 = torch.zeros_like(v_lit).scatter(1, top_idx, w2)                  # (M,3)

    # w_best sums to 1 across literals; support is either {top1} or {top1,top2}
    w_best = (1.0 - s).unsqueeze(-1) * onehot_top1 + s.unsqueeze(-1) * w2_3

    # For a candidate "best" literal among top-2, set its G-entry to the other top-2 unsat.
    # top1 gets unsat_top[:,1]; top2 gets unsat_top[:,0] (=c)
    other_vals_2 = torch.stack([unsat_top[:, 1], unsat_top[:, 0]], dim=-1)  # (M,2)
    u_other_3 = torch.zeros_like(v_lit).scatter(1, top_idx, other_vals_2)   # (M,3)

    # Baseline-like G: start from c everywhere, then (sparsely) replace best-literal entry(ies)
    G = c.unsqueeze(-1).expand_as(v_lit) + w_best * (u_other_3 - c.unsqueeze(-1))
    G = G * (xl * xs).unsqueeze(-1)

    # Baseline-like R: rigidity only on (hard or tie-split) best literal(s); never on 3rd literal
    R = (w_best * c.unsqueeze(-1))
    R = R * ((1.0 + hp.zeta * xl) * (1.0 - xs)).unsqueeze(-1)

    # Accumulate dv
    dv_clause = (G + R) * sgn
    dv = torch.zeros_like(v).scatter_add_(0, idx.reshape(-1), dv_clause.reshape(-1))

    # Memory dynamics (unchanged)
    dxl = hp.alpha * (c - hp.delta)
    dxs = hp.beta  * ((xs + hp.epsilon) * (c - hp.gamma))

    # Normalize
    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))
    dv  = -dv  * scale
    dxl = -dxl * scale
    dxs = -dxs * scale

    return {"v": dv, "xl": dxl, "xs": dxs}
