# base_sat_solver.py
# ------------------------------------------------------------
# Generic, vmap-based Digital MemComputing solver that works
# ------------------------------------------------------------
import os, json
import importlib.util
import sys
import types, torch, torch.nn as nn
from torch.func import vmap

# Handle both module import and direct execution
try:
    from .dataset import SAT_dataset
except ImportError:
    from dataset import SAT_dataset

# Dynamic import of solver components based on SOLVER_ID environment variable
_solver_id = os.environ.get('SOLVER_ID', None)
if _solver_id is not None:
    # Import from experiment-specific solver using direct file loading
    # This supports workspace-specific solver directories
    import workspace_config
    _solvers_dir = workspace_config.get_solvers_dir()
    _solver_path = _solvers_dir / f"solver_{_solver_id}.py"
    _spec = importlib.util.spec_from_file_location(f"_solver_{_solver_id}", _solver_path)
    _solver_module = importlib.util.module_from_spec(_spec)
    sys.modules[f"_solver_{_solver_id}"] = _solver_module
    _spec.loader.exec_module(_solver_module)
    VARIABLES_SPEC = _solver_module.VARIABLES_SPEC
    HYPER_SPACE = _solver_module.HYPER_SPACE
    _grad_single = _solver_module._grad_single
else:
    # Import from baseline solver
    try:
        from .solver_baseline import VARIABLES_SPEC, HYPER_SPACE, _grad_single
    except ImportError:
        from solver_baseline import VARIABLES_SPEC, HYPER_SPACE, _grad_single

torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)


def _solved_single(v, idx, sgn):
    """Boolean: all clauses satisfied for ONE instance."""
    v_lit = v[idx] * sgn
    C     = 0.5 * (1.0 - v_lit.max(dim=-1).values)
    return (C < 0.5).all()


# ============================================================#
# 4.  Generic SATSolver                                       #
# ============================================================#
class SATSolver(nn.Module):
    """
    Digital MemComputing machine for batched 3-SAT, parameterised by

    * `variables_spec` – dict(name -> {init, shape})
    * `bounds_spec`    – dict(name -> (low, high))
    * `hyper_params`   – **plain dict** of scalars
    """

    def __init__(self,
                 sat_instance: dict,
                 variables_spec: dict,
                 bounds_spec: dict,
                 hyper_params: dict):
        super().__init__()

        # ---------- problem tensors (no grad) -------------------------------
        self.idx = sat_instance["clause_idx"]          # (B,M,3)
        self.sgn = sat_instance["clause_sign"]         # (B,M,3)
        self.batch, self.M, _ = self.idx.shape
        self.N = variables_spec["v"]["shape"][-1]      # last dim of v

        # ---------- learnable variables -------------------------------------
        self.vars = nn.ParameterDict()
        for name, cfg in variables_spec.items():
            tensor = cfg["init"](cfg["shape"])
            self.vars[name] = nn.Parameter(tensor)

        self.bounds = bounds_spec
        self.var_names = tuple(variables_spec.keys())

        # ---------- hyper-params as namespace -------------------------------
        hp = types.SimpleNamespace(**hyper_params)
        self.hp = hp

        # ---------- build vmapped wrappers AUTOMATICALLY --------------------
        # We vectorize over the leading batch dimension for:
        #   - self.vars: a dict of (B, ...) tensors  (use in_dims dict of zeros)
        #   - self.idx:  (B, M, 3)   -> in_dims = 0
        #   - self.sgn:  (B, M, 3)   -> in_dims = 0
        in_dims_vars = {k: 0 for k in self.var_names}   # pytree spec for dict
        self._grad_batched = vmap(
            lambda vardict, idx, sgn: _grad_single(vardict, idx, sgn, hp),
            in_dims=(in_dims_vars, 0, 0),
            randomness="different"
        )

        self._check_batched = vmap(
            _solved_single, in_dims=(0, 0, 0)
        )

        # ---------- optimiser ----------------------------------------------
        lr = hyper_params.get("lr", 1e-2)
        self.opt = torch.optim.SGD(self.parameters(), lr=lr)

    def _vars_pytree(self):
        # Convert ParameterDict -> plain dict[TENSOR] for functorch
        return {k: self.vars[k] for k in self.var_names}

    # ---------------------------------------------------------------- grad --
    # compute grads via vmapped _grad_single → dict of grads.
    @torch.no_grad()
    def _calc_grads(self):
        grads = self._grad_batched(self._vars_pytree(), self.idx, self.sgn)  # dict of (B, ...) grads
        for name in self.var_names:
            self.vars[name].grad = grads[name]

    # ----------------------------------------------------------- is_solved --
    @torch.no_grad()
    def _is_solved(self):
        v = self.vars["v"]
        return self._check_batched(v, self.idx, self.sgn)

    # ---------------------------------------------------------------- solve --
    @torch.no_grad()
    def solve(self, max_steps=1000, run_till_median=False, verbose=False, log_interval=100):
        """
        Euler integrate for max_steps.
        Returns the fraction of unsolved instances (lower is better).
        """
        median_step = max_steps
        median_flag = False
        for step in range(1, max_steps + 1):
            self.opt.zero_grad()
            self._calc_grads()
            self.opt.step()

            for name, (lo, hi) in self.bounds.items():
                self.vars[name].data.clamp_(lo, hi)

            solved = self._is_solved().sum().item()
            if not median_flag and solved > self.batch // 2:
                median_step = step
                median_flag = True
                if run_till_median:
                    if verbose:
                        print(f"N={self.N}  Step {step}  Solved {solved}/{self.batch} (median reached)")
                    return 1 - solved / self.batch, median_step
            if verbose and (step % log_interval == 0):
                print(f"N={self.N}  Step {step}  Solved {solved}/{self.batch}")
        
        solved = self._is_solved().sum().item()
        if not median_flag:
            # Project median step using current solved count
            median_step = int(max_steps * (self.batch / 2 + 0.5) / (solved + 0.5))
            
        return 1 - solved / self.batch, median_step


# ============================================================#
# 5.  Simple factory for the tuner                            #
# ============================================================#
def make_solver(sat_instance, variables_spec, bounds_spec, hyper_params):
    return SATSolver(
        sat_instance   = sat_instance,
        variables_spec = variables_spec,
        bounds_spec    = bounds_spec,
        hyper_params   = hyper_params
    )


def _materialise_shapes(spec: dict, B, N, M):
    """Return copy with 'shape' resolved to actual tuples."""
    out = {}
    lookup = {"B": B, "N": N, "M": M}
    for name, cfg in spec.items():
        shape = tuple(lookup.get(s, s) for s in cfg["shape"])
        out[name] = dict(cfg)       # shallow copy
        out[name]["shape"] = shape
    return out


def run_solver(batch, N, ratio, run_till_median=False, max_steps=1000, verbose=False, log_interval=100, hp=None):
    M = int(ratio * N)
    var_spec = _materialise_shapes(VARIABLES_SPEC, batch, N, M)
    if hp is None:
        if os.path.exists("best_params.json"):
            hp = json.load(open("best_params.json"))["params"]
            print(f"Using hyper-parameters from 'best_params.json': {hp}")
        else:
            hp = {n: c["default"] for n, c in HYPER_SPACE.items()}
            print(f"Using default hyper-parameters: {hp}")
    bounds = {k: v["bounds"] for k,v in var_spec.items()}
    dataset = SAT_dataset(batch, N, ratio)
    CLAUSE_IDX, CLAUSE_SGN = dataset.generate_instances()
    sat_instance = {"clause_idx": CLAUSE_IDX, "clause_sign": CLAUSE_SGN}

    solver = make_solver(
        sat_instance   = sat_instance,
        variables_spec = var_spec,
        bounds_spec    = bounds,
        hyper_params   = hp
    )

    unsolved_fraction, median_steps = solver.solve(max_steps=max_steps,
                                      run_till_median=run_till_median,
                                      verbose=verbose,
                                      log_interval=log_interval)
    return unsolved_fraction, median_steps


if __name__ == "__main__":
    BATCH, N, RATIO = 100, 50, 4.3
    try:
        HYPER_PARAMS = json.load(open("best_params.json"))["params"]
    except FileNotFoundError:
        HYPER_PARAMS = {n: c["default"] for n, c in HYPER_SPACE.items()}
    unsolved_fraction, _ = run_solver(BATCH, N, RATIO,
                                      run_till_median=False, max_steps=1000, verbose=True, log_interval=100)
