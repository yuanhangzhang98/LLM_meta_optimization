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

    # NEW (single principled change): bounded multiplicative boost for dxs
    # boost = 1 + mem_eta * sigmoid((c-gamma)/mem_kappa) * (1-xs)
    "mem_eta":   dict(type="log_uniform", default=0.5,  low=1e-2, high=10.0),
    "mem_kappa": dict(type="log_uniform", default=2e-2, low=1e-3, high=2e-1),

    "lr":      dict(type="log_uniform", default=1.0, low=1e-1, high=3.0),
}


# ------------------------------------------------------------------ #
# 3.  _grad_single
# ------------------------------------------------------------------ #

def _grad_single(vars, idx, sgn, hp):
    v  = vars["v"]
    xl = vars["xl"]
    xs = vars["xs"]

    # ===== CLAUSE EVALUATION (baseline) =====
    v_lit = v[idx] * sgn
    top_val, top_idx = torch.topk(v_lit, 2, dim=-1)
    unsat_top = 0.5 * (1.0 - top_val)
    c = unsat_top[:, 0]

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

    # ===== MEMORY DYNAMICS =====
    dxl = hp.alpha * (c - hp.delta)

    # NEW: structure-preserving, bounded multiplicative boost for xs updates
    mem_kappa = torch.as_tensor(hp.mem_kappa, device=v.device, dtype=v.dtype).clamp_min(1e-6)
    mem_eta = torch.as_tensor(hp.mem_eta, device=v.device, dtype=v.dtype).clamp_min(0.0)
    viol_gate = torch.sigmoid((c - hp.gamma) / mem_kappa)              # ~0 satisfied, ~1 violated
    boost = 1.0 + mem_eta * viol_gate * (1.0 - xs)                     # strongest when violated & xs low

    dxs = hp.beta * ((xs + hp.epsilon) * (c - hp.gamma)) * boost

    # ===== GRADIENT SCALING (baseline) =====
    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))

    dv  = -dv  * scale
    dxl = -dxl * scale
    dxs = -dxs * scale

    return {"v": dv, "xl": dxl, "xs": dxs}
