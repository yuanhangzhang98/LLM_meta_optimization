import torch
import torch.nn.functional as F

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
    "lr":      dict(type="log_uniform", default=1.0,  low=1e-1, high=3.0),

    # softmin temperature used ONLY for memory signals (dxl/dxs)
    "tau":     dict(type="log_uniform", default=2e-2, low=1e-3, high=2e-1),

    # very mild smoothing for G-values only (R remains hard)
    "tau_g":   dict(type="log_uniform", default=5e-3, low=5e-4, high=5e-2),
}

# ------------------------------------------------------------------ #
# 3. _grad_single
# ------------------------------------------------------------------ #

def _grad_single(vars, idx, sgn, hp):
    v  = vars["v"]   # (N,)
    xl = vars["xl"]  # (M,)
    xs = vars["xs"]  # (M,)

    # literals and unsatisfaction
    v_lit = v[idx] * sgn                  # (M,3)
    u = 0.5 * (1.0 - v_lit)               # (M,3) in [0,1]

    # hard winner index (for rigidity and for G structure)
    top_val, top_idx = torch.topk(v_lit, 2, dim=-1)  # (M,2)
    unsat_top = 0.5 * (1.0 - top_val)                # (M,2)
    c_hard = unsat_top[:, 0]                         # (M,)
    win = top_idx[:, 0]                              # (M,)

    win_oh = F.one_hot(win, num_classes=3).to(dtype=v.dtype)  # (M,3)

    # ----------------- Hard rigidity R via masking (no scatter_) -----------------
    R = win_oh * c_hard.unsqueeze(-1)
    R = R * ((1.0 + hp.zeta * xl) * (1.0 - xs)).unsqueeze(-1)

    # ----------------- Mild smoothing in G values only -----------------
    tau_g = float(hp.tau_g)
    tau_g = 1e-4 if tau_g < 1e-4 else tau_g

    def softmin_expect(x, tau):
        w = torch.softmax(-x / tau, dim=-1)
        return (w * x).sum(dim=-1)

    # smooth estimate of min(u)
    c_g = softmin_expect(u, tau_g)  # (M,)

    # smooth estimate of second-min: exclude hard winner then softmin
    u_excl = u + 10.0 * win_oh
    second_g = softmin_expect(u_excl, tau_g)  # (M,)

    # baseline G structure: all literals get c, winner gets second-best
    G = (1.0 - win_oh) * c_g.unsqueeze(-1) + win_oh * second_g.unsqueeze(-1)
    G = G * (xl * xs).unsqueeze(-1)

    # accumulate dv
    dv_clause = (G + R) * sgn
    dv = torch.zeros_like(v).scatter_add(0, idx.reshape(-1), dv_clause.reshape(-1))

    # ----------------- Memory-only smooth clause cost (Design 5) -----------------
    tau = float(hp.tau)
    tau = 1e-4 if tau < 1e-4 else tau
    w_mem = torch.softmax(-u / tau, dim=-1)
    c_mem = (w_mem * u).sum(dim=-1)  # (M,)

    dxl = hp.alpha * (c_mem - hp.delta)
    dxs = hp.beta  * ((xs + hp.epsilon) * (c_mem - hp.gamma))

    # normalize by dv scale (baseline convention)
    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))

    dv  = -dv  * scale
    dxl = -dxl * scale
    dxs = -dxs * scale

    return {"v": dv, "xl": dxl, "xs": dxs}
