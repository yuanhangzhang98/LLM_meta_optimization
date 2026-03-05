import torch

# ------------------------------------------------------------------ #
# 1.  VARIABLES_SPEC - State variable specification                 #
# ------------------------------------------------------------------ #
# Each entry:  name -> dict(init=callable, shape=tuple, bounds=(lo,hi))
# The framework creates these variables and evolves them using gradients from _grad_single

VARIABLES_SPEC = {
    # v: Main boolean variables (one per SAT variable)
    # Shape (B, N) = (batch_size, num_variables)
    # Interpretation: sign(v[i]) gives boolean assignment for variable i
    "v":  dict(init=lambda s: 2 * torch.rand(s) - 1,
               shape=("B", "N"),
               bounds=(-1.0, 1.0)),

    # xl: Clause penalty weights (one per clause)
    # Shape (B, M) = (batch_size, num_clauses)
    # Used to emphasize unsatisfied clauses in the dynamics
    "xl": dict(init=lambda s: torch.ones(s),
               shape=("B", "M"),
               bounds=(1.0, 1e6)),

    # xs: Clause satisfaction indicators (one per clause)
    # Shape (B, M) = (batch_size, num_clauses)
    # Tracks which clauses are currently satisfied
    "xs": dict(init=lambda s: torch.zeros(s),
               shape=("B", "M"),
               bounds=(0.0, 1.0)),
}


# ------------------------------------------------------------------ #
# 2.  HYPER_SPACE - Hyperparameter search space                      #
# ------------------------------------------------------------------ #
# Supported types: "uniform", "log_uniform", "int", "categorical"
# HEBO optimizer tunes these parameters to find optimal performance

HYPER_SPACE = {
    # alpha: Controls xl (clause penalty weight) growth rate
    "alpha":   dict(type="log_uniform", default=5.0,  low=0.5,  high=50.0),

    # beta: Controls xs (satisfaction indicator) update rate
    "beta":    dict(type="log_uniform", default=20.0, low=2.0,  high=200.0),

    # gamma: Threshold for xs dynamics (clause satisfaction target)
    "gamma":   dict(type="uniform",     default=0.25, low=0.01, high=0.50),

    # delta: Threshold for xl dynamics (penalty activation level)
    "delta":   dict(type="uniform",     default=0.05, low=0.01, high=0.50),

    # epsilon: Regularization for xs to stalled dynamics
    "epsilon": dict(type="log_uniform", default=1e-3, low=1e-4, high=1e-2),

    # zeta: Coupling strength between xl and rigidity term
    "zeta":    dict(type="log_uniform", default=1e-3, low=1e-4, high=1.0),

    # lr: Learning rate (i.e., integration time step) for gradient descent (applied by framework)
    "lr":      dict(type="log_uniform", default=1.0, low=1e-1, high=3.0),
}


# ------------------------------------------------------------------ #
# 3.  _grad_single - Per-instance dynamics function                 #
# ------------------------------------------------------------------ #
# This is the CORE of your solver - defines how variables evolve.
#
# FRAMEWORK INTEGRATION:
# - The framework calls this function using torch.vmap to vectorize across batch
# - Input vars dict has NO batch dimension (single instance)
# - Must return gradients dict with SAME KEYS as vars (no batch dimension)
#
# INPUTS:
#   vars: dict of tensors WITHOUT batch dim
#         Example: {"v": (N,), "xl": (M,), "xs": (M,)}
#   idx:  (M, 3) - clause variable indices for this instance
#   sgn:  (M, 3) - clause literal signs (+1 or -1) for this instance
#   hp:   namespace with hyperparameters (hp.alpha, hp.beta, etc.)
#
# OUTPUT:
#   dict of gradients with same structure as vars
#   Framework applies: vars[k] = vars[k] - hp.lr * gradients[k]
#
# LIMITATIONS:
# - No data-dependent control flow (torch.vmap limitation). Use masking if needed.

def _grad_single(vars, idx, sgn, hp):
    """Compute per-instance gradients. Return dict with same keys as vars."""
    # Extract current state variables (no batch dimension)
    v  = vars["v"]   # (N,) - boolean variable values
    xl = vars["xl"]  # (M,) - clause penalty weights
    xs = vars["xs"]  # (M,) - clause satisfaction indicators

    # ===== CLAUSE EVALUATION =====
    # Compute literal values: v[idx] selects variables, multiply by sign
    v_lit   = v[idx] * sgn                       # (M,3) - literal values for each clause

    # Find top 2 literals in each clause (most satisfied)
    top_val, top_idx = torch.topk(v_lit, 2, dim=-1)

    # Measure "unsatisfaction" (0 = satisfied, 1 = violated)
    unsat_top = 0.5 * (1.0 - top_val)            # (M,2)
    c = unsat_top[:, 0]                          # (M,) - clause cost (0 if satisfied)

    # ===== GRADIENT TERM (drives toward satisfaction) =====
    # Push variables to satisfy clauses, weighted by xl*xs
    G = c.unsqueeze(-1).repeat(1, 3)
    G.scatter_(1, top_idx[:, 0:1], unsat_top[:, 1:2])
    G *= (xl * xs).unsqueeze(-1)                 # Weight by clause penalties

    # ===== RIGIDITY TERM (prevents oscillation) =====
    # Stabilizes solution once found, weighted by (1-xs)
    R = torch.zeros_like(G)
    R.scatter_(1, top_idx[:, 0:1], c.unsqueeze(-1))
    R *= ((1.0 + hp.zeta * xl) * (1.0 - xs)).unsqueeze(-1)

    # ===== ACCUMULATE v GRADIENT =====
    # Sum contributions from all clauses containing each variable
    dv_clause = (G + R) * sgn                    # Apply literal signs
    dv = torch.zeros_like(v).scatter_add_(0, idx.reshape(-1),
                                          dv_clause.reshape(-1))

    # ===== AUXILIARY VARIABLE GRADIENTS =====
    dxl = hp.alpha * (c - hp.delta)
    dxs = hp.beta  * ((xs + hp.epsilon) * (c - hp.gamma))

    # ===== GRADIENT SCALING =====
    # Normalize to prevent numerical instability
    scale = 1 / (dv.abs().max().clamp(min=1e-6))

    # Negate for gradient descent (framework does vars -= lr * grad)
    dv  = -dv  * scale
    dxl = -dxl * scale
    dxs = -dxs * scale

    # Return gradients for all variables
    return {"v": dv, "xl": dxl, "xs": dxs}
