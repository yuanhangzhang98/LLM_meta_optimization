import math
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
    "alpha":   dict(type="log_uniform", default=5.0,  low=0.5,  high=50.0),
    "beta":    dict(type="log_uniform", default=20.0, low=2.0,  high=200.0),
    "gamma":   dict(type="uniform",     default=0.25, low=0.01, high=0.50),
    "delta":   dict(type="uniform",     default=0.05, low=0.01, high=0.50),
    "epsilon": dict(type="log_uniform", default=1e-3, low=1e-4, high=1e-2),
    "zeta":    dict(type="log_uniform", default=1e-3, low=1e-4, high=1.0),

    # NEW (localized smoothing for memory updates only)
    # tau_c: temperature for soft-min clause monitor (only used when clause is not near satisfied)
    "tau_c":   dict(type="log_uniform", default=5e-2, low=1e-3, high=3e-1),

    # c_cut: below this (near-satisfied), keep hard c to avoid any satisfied-clause drift
    "c_cut":   dict(type="uniform",     default=0.10, low=0.02, high=0.25),

    "lr":      dict(type="log_uniform", default=1.0, low=1e-1, high=3.0),
}


# ------------------------------------------------------------------ #
# 3. _grad_single
# ------------------------------------------------------------------ #

def _grad_single(vars, idx, sgn, hp):
    v  = vars["v"]   # (N,)
    xl = vars["xl"]  # (M,)
    xs = vars["xs"]  # (M,)

    # Literal values in [-1, 1]
    v_lit = v[idx] * sgn  # (M,3)

    # Baseline hard routing (top2)
    top_val, top_idx = torch.topk(v_lit, 2, dim=-1)          # (M,2)
    unsat_top = 0.5 * (1.0 - top_val)                        # (M,2)
    c_hard = unsat_top[:, 0]                                  # (M,)

    # ---- NEW: smooth clause monitor for memory updates only, and only when violated enough ----
    # Unsatisfaction per literal in [0,1]
    u = 0.5 * (1.0 - v_lit)                                   # (M,3)
    tau = torch.as_tensor(hp.tau_c, device=v.device, dtype=v.dtype).clamp(min=1e-4)

    # Soft-min approx to min(u) (upper-shifted so it doesn’t underestimate too much)
    # softmin(x) = -tau * logsumexp(-x/tau)
    # add tau*log(3) so output is closer to true min for 3 items
    c_soft = -tau * torch.logsumexp(-u / tau, dim=-1) + (tau * math.log(3.0))
    c_soft = c_soft.clamp(0.0, 1.0)

    c_cut = torch.as_tensor(hp.c_cut, device=v.device, dtype=v.dtype)
    c_mem = torch.where(c_hard < c_cut, c_hard, c_soft)

    # ===== GRADIENT TERM (baseline; uses hard c and hard routing) =====
    G = c_hard.unsqueeze(-1).repeat(1, 3)
    G.scatter_(1, top_idx[:, 0:1], unsat_top[:, 1:2])
    G *= (xl * xs).unsqueeze(-1)

    # ===== RIGIDITY TERM (baseline; uses hard c and hard winner) =====
    R = torch.zeros_like(G)
    R.scatter_(1, top_idx[:, 0:1], c_hard.unsqueeze(-1))
    R *= ((1.0 + hp.zeta * xl) * (1.0 - xs)).unsqueeze(-1)

    # ===== ACCUMULATE v GRADIENT =====
    dv_clause = (G + R) * sgn
    dv = torch.zeros_like(v).scatter_add_(0, idx.reshape(-1), dv_clause.reshape(-1))

    # ===== AUXILIARY VARIABLE GRADIENTS (memory uses c_mem) =====
    dxl = hp.alpha * (c_mem - hp.delta)
    dxs = hp.beta  * ((xs + hp.epsilon) * (c_mem - hp.gamma))

    # ===== NORMALIZE =====
    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))

    dv  = -dv  * scale
    dxl = -dxl * scale
    dxs = -dxs * scale

    return {"v": dv, "xl": dxl, "xs": dxs}
