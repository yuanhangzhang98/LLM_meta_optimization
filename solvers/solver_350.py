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
    # Baseline
    "alpha":   dict(type="log_uniform", default=5.0,  low=0.5,  high=50.0),
    "beta":    dict(type="log_uniform", default=20.0, low=2.0,  high=200.0),
    "gamma":   dict(type="uniform",     default=0.25, low=0.01, high=0.50),
    "delta":   dict(type="uniform",     default=0.05, low=0.01, high=0.50),
    "epsilon": dict(type="log_uniform", default=1e-3, low=1e-4, high=1e-2),
    "zeta":    dict(type="log_uniform", default=1e-3, low=1e-4, high=1.0),

    # Tie-aware monitor for dxs (localized to near-gamma-from-below)
    "tau_tie":   dict(type="log_uniform", default=0.017, low=1e-3, high=2e-1),
    "g0":        dict(type="log_uniform", default=0.08,  low=1e-3, high=1.0),
    "kappa_tie": dict(type="log_uniform", default=0.06,  low=1e-3, high=5e-1),
    "u_floor":   dict(type="log_uniform", default=0.02,  low=1e-4, high=2e-1),

    # Near-γ (from below) slice for activating tie smoothing
    "ng_dmax":   dict(type="uniform",     default=0.05,  low=0.0,  high=0.25),
    "ng_dsharp": dict(type="log_uniform", default=0.01,  low=1e-3, high=1e-1),

    "lr": dict(type="log_uniform", default=1.0, low=1e-1, high=3.0),
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

    # ===== Baseline hard top-2 backbone =====
    top_val, top_idx = torch.topk(v_lit, 2, dim=-1)          # (M,2)
    unsat_top = 0.5 * (1.0 - top_val)                        # (M,2)
    c_hard = unsat_top[:, 0].clamp(0.0, 1.0)                 # (M,)

    G = c_hard.unsqueeze(-1).repeat(1, 3)
    G.scatter_(1, top_idx[:, 0:1], unsat_top[:, 1:2])
    G *= (xl * xs).unsqueeze(-1)

    R = torch.zeros_like(G)
    R.scatter_(1, top_idx[:, 0:1], c_hard.unsqueeze(-1))
    R *= ((1.0 + hp.zeta * xl) * (1.0 - xs)).unsqueeze(-1)

    dv_clause = (G + R) * sgn
    dv = torch.zeros_like(v).scatter_add_(0, idx.reshape(-1), dv_clause.reshape(-1))

    # ===== Baseline xl update =====
    dxl = hp.alpha * (c_hard - hp.delta)

    # ===== SINGLE MODIFICATION vs baseline: dxs uses a near-γ, tie-aware monitor =====
    # u in [0,1], smaller is better
    u = (0.5 * (1.0 - v_lit)).clamp(0.0, 1.0)  # (M,3)

    # best two u's (tie detection)
    u2best, _ = torch.topk(u, 2, dim=-1, largest=False)  # (M,2)
    u1 = u2best[:, 0]
    u2 = u2best[:, 1]
    gap = (u2 - u1).clamp(min=0.0)

    eps = torch.as_tensor(1e-6, device=v.device, dtype=v.dtype)

    # gap normalized; u_floor stabilizes near u1,u2~0 (Design 232 idea)
    u_floor = torch.as_tensor(hp.u_floor, device=v.device, dtype=v.dtype)
    denom = (u1 + u2 + u_floor).clamp_min(eps)
    g_mag = (gap / denom).clamp(0.0, 1.0)

    g0 = torch.as_tensor(hp.g0, device=v.device, dtype=v.dtype)
    kappa_tie = torch.as_tensor(hp.kappa_tie, device=v.device, dtype=v.dtype).clamp_min(eps)
    w_tie = torch.sigmoid((g0 - g_mag) / kappa_tie).clamp(0.0, 1.0)  # high when near-tie

    gamma = torch.as_tensor(hp.gamma, device=v.device, dtype=v.dtype)
    ng_dmax = torch.as_tensor(hp.ng_dmax, device=v.device, dtype=v.dtype).clamp_min(0.0)
    ng_dsharp = torch.as_tensor(hp.ng_dsharp, device=v.device, dtype=v.dtype).clamp_min(eps)

    d = (gamma - c_hard)  # >0 below gamma
    below = torch.sigmoid(d / ng_dsharp)                      # ~1 only when below gamma
    d_pos = torch.relu(d)                                     # distance below gamma
    near = torch.sigmoid((ng_dmax - d_pos) / ng_dsharp)       # ~1 only if just below gamma

    smooth_w = (w_tie * below * near).clamp(0.0, 1.0)

    tau = torch.as_tensor(hp.tau_tie, device=v.device, dtype=v.dtype).clamp_min(eps)
    c_soft = (-tau * torch.logsumexp(-u / tau, dim=-1)).clamp(0.0, 1.0)

    c_mem = (c_hard + smooth_w * (c_soft - c_hard)).clamp(0.0, 1.0)

    dxs = hp.beta * ((xs + hp.epsilon) * (c_mem - gamma))

    # ===== Gradient scaling (baseline style) =====
    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))

    return {
        "v":  -dv  * scale,
        "xl": -dxl * scale,
        "xs": -dxs * scale,
    }
