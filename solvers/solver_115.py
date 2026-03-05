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
    "lr":      dict(type="log_uniform", default=1.0,  low=1e-1, high=3.0),

    # NEW (single principled change vs Design 3): separate temperatures
    # tau_c: sharper (memory reacts to true worst literal)
    # tau_g: smoother (drive less jumpy)
    "tau_c":   dict(type="log_uniform", default=3e-2, low=3e-3, high=2e-1),
    "tau_g":   dict(type="log_uniform", default=7e-2, low=5e-3, high=3e-1),
}


# ------------------------------------------------------------------ #
# 3.  _grad_single
# ------------------------------------------------------------------ #

def _grad_single(vars, idx, sgn, hp):
    v  = vars["v"]   # (N,)
    xl = vars["xl"]  # (M,)
    xs = vars["xs"]  # (M,)

    # literals and unsatisfaction per literal
    v_lit = v[idx] * sgn           # (M,3) in [-1,1]
    u = 0.5 * (1.0 - v_lit)        # (M,3) in [0,1]

    tau_c = float(hp.tau_c)
    tau_g = float(hp.tau_g)
    tau_c = max(tau_c, 1e-4)
    tau_g = max(tau_g, 1e-4)

    def softmin_expect(x, tau, dim=-1):
        # expectation under softmax(-x/tau)
        w = torch.softmax(-x / tau, dim=dim)
        return (w * x).sum(dim=dim)

    # clause cost for memory dynamics (sharper)
    c = softmin_expect(u, tau_c, dim=-1)  # (M,)

    # smooth drive term (softer)
    g0 = softmin_expect(torch.stack([u[:, 1], u[:, 2]], dim=-1), tau_g, dim=-1)
    g1 = softmin_expect(torch.stack([u[:, 0], u[:, 2]], dim=-1), tau_g, dim=-1)
    g2 = softmin_expect(torch.stack([u[:, 0], u[:, 1]], dim=-1), tau_g, dim=-1)
    G = torch.stack([g0, g1, g2], dim=-1)                # (M,3)
    G = G * (xl * xs).unsqueeze(-1)

    # rigidity stays HARD winner-take-all (Design 3 principle)
    win = torch.argmin(u, dim=-1, keepdim=True)          # (M,1)
    u_win = u.gather(1, win)                             # (M,1)
    R = torch.zeros_like(G)
    R.scatter_(1, win, u_win)
    R = R * ((1.0 + hp.zeta * xl) * (1.0 - xs)).unsqueeze(-1)

    dv_clause = (G + R) * sgn
    dv = torch.zeros_like(v).scatter_add_(0, idx.reshape(-1), dv_clause.reshape(-1))

    dxl = hp.alpha * (c - hp.delta)
    dxs = hp.beta  * ((xs + hp.epsilon) * (c - hp.gamma))

    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))

    dv  = -dv  * scale
    dxl = -dxl * scale
    dxs = -dxs * scale

    return {"v": dv, "xl": dxl, "xs": dxs}
