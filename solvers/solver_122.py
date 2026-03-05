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

    # Split temperatures:
    # tau_c: sharper memory cost (more like hard min)
    # tau_g: smoother drive values (reduces discontinuities) while routing stays hard
    "tau_c":   dict(type="log_uniform", default=5e-3, low=1e-4, high=5e-2),
    "tau_g":   dict(type="log_uniform", default=5e-2, low=5e-3, high=3e-1),
}

# ------------------------------------------------------------------ #
# 3. _grad_single
# ------------------------------------------------------------------ #

def _grad_single(vars, idx, sgn, hp):
    v  = vars["v"]   # (N,)
    xl = vars["xl"]  # (M,)
    xs = vars["xs"]  # (M,)

    # literals and per-literal unsatisfaction
    v_lit = v[idx] * sgn            # (M,3) in [-1,1]
    u = 0.5 * (1.0 - v_lit)         # (M,3) in [0,1]

    tau_c = max(float(hp.tau_c), 1e-4)
    tau_g = max(float(hp.tau_g), 1e-4)

    def softmin_expect(x, tau, dim=-1):
        w = torch.softmax(-x / tau, dim=dim)
        return (w * x).sum(dim=dim)

    # ----------------- Hard winner (for routing + rigidity) -----------------
    win = torch.argmin(u, dim=-1)  # (M,)
    win_oh = F.one_hot(win, num_classes=3).to(dtype=v.dtype)  # (M,3)

    # ----------------- G: baseline-like routing, smoothed values -----------------
    # c_g ~ min(u), second_g ~ min(u excluding winner)
    c_g = softmin_expect(u, tau_g, dim=-1)  # (M,)

    # exclude winner by adding a large offset (u in [0,1])
    exclude = 10.0
    u_excl = u + exclude * win_oh
    second_g = softmin_expect(u_excl, tau_g, dim=-1)  # (M,)

    # all literals get c_g; winner gets second_g instead
    G = c_g.unsqueeze(-1) + win_oh * (second_g - c_g).unsqueeze(-1)
    G = G * (xl * xs).unsqueeze(-1)

    # ----------------- R: hard rigidity (winner only) -----------------
    u_win = (u * win_oh).sum(dim=-1)  # (M,)
    R = win_oh * u_win.unsqueeze(-1)
    R = R * ((1.0 + hp.zeta * xl) * (1.0 - xs)).unsqueeze(-1)

    # accumulate dv
    dv_clause = (G + R) * sgn
    dv = torch.zeros_like(v).scatter_add(0, idx.reshape(-1), dv_clause.reshape(-1))

    # ----------------- Memory: sharper clause cost -----------------
    c_mem = softmin_expect(u, tau_c, dim=-1)  # (M,)
    dxl = hp.alpha * (c_mem - hp.delta)
    dxs = hp.beta  * ((xs + hp.epsilon) * (c_mem - hp.gamma))

    # normalize by dv scale
    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))

    dv  = -dv  * scale
    dxl = -dxl * scale
    dxs = -dxs * scale

    return {"v": dv, "xl": dxl, "xs": dxs}
