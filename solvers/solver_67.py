import math
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
    # Design-11 tuned defaults for the base system
    "alpha":   dict(type="log_uniform", default=48.3997327918925,  low=0.5,  high=50.0),
    "beta":    dict(type="log_uniform", default=40.08506794497957, low=2.0,  high=200.0),
    "gamma":   dict(type="uniform",     default=0.28948339187872574, low=0.01, high=0.50),
    "delta":   dict(type="uniform",     default=0.07327583432197571, low=0.01, high=0.50),
    "epsilon": dict(type="log_uniform", default=0.001468934335764333, low=1e-4, high=1e-2),
    "zeta":    dict(type="log_uniform", default=0.0013418985396236333, low=1e-4, high=1.0),
    "lr":      dict(type="log_uniform", default=1.6536008035680871,  low=1e-1, high=3.0),

    # Tie-only smoothing knobs for dxs (memory monitor only)
    "tau_tie": dict(type="log_uniform", default=0.01712294456377018, low=1e-3, high=2e-1),

    # Gate parameters apply to the combined normalized gap in [0,1]
    "g0":      dict(type="log_uniform", default=0.08, low=1e-3, high=1.0),
    "kappa":   dict(type="log_uniform", default=0.06, low=1e-3, high=5e-1),

    # NEW: temperature controlling how close the hybrid is to max (conservative) vs mean (less conservative)
    "tau_gap": dict(type="log_uniform", default=0.03, low=1e-3, high=3e-1),
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

    # --- Memory monitor for dxs: hard-min almost everywhere, soft only near ties ---
    u = 0.5 * (1.0 - v_lit)  # (M,3) in [0,1]

    # Ordered u1<=u2<=u3
    u_sorted, _ = torch.topk(u, 3, dim=-1, largest=False)
    u1 = u_sorted[:, 0]
    u2 = u_sorted[:, 1]
    u3 = u_sorted[:, 2]

    gap = (u2 - u1).clamp(min=0.0)

    # Two normalized gap features in [0,1]
    eps = 1e-6
    g_mag = (gap / (u1 + u2).clamp(min=eps)).clamp(0.0, 1.0)
    g_spread = (gap / (u3 - u1).clamp(min=eps)).clamp(0.0, 1.0)

    # SINGLE MODIFICATION (vs Design-21): soft-max blend to tune conservativeness
    # tau_gap -> 0  => max(g_mag, g_spread)
    # larger tau_gap => ~0.5*(g_mag + g_spread)
    t = hp.tau_gap
    pair = torch.stack([g_mag / t, g_spread / t], dim=-1)
    gap_eff = (t * torch.logsumexp(pair, dim=-1) - t * math.log(2.0)).clamp(0.0, 1.0)

    w = torch.sigmoid((hp.g0 - gap_eff) / hp.kappa).clamp(0.0, 1.0)

    tau = hp.tau_tie
    c_soft = (-tau * torch.logsumexp(-u / tau, dim=-1)).clamp(0.0, 1.0)
    c_mem = ((1.0 - w) * u1 + w * c_soft).clamp(0.0, 1.0)

    # Long-term memory stays driven by hard monitor
    dxl = hp.alpha * (c_hard - hp.delta)
    # Short-term switching uses tie-only smoothing
    dxs = hp.beta * ((xs + hp.epsilon) * (c_mem - hp.gamma))

    # Normalize (baseline style)
    scale = 1.0 / (dv.abs().max().clamp(min=1e-6))

    return {
        "v":  -dv  * scale,
        "xl": -dxl * scale,
        "xs": -dxs * scale,
    }
