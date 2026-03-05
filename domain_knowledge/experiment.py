import os, sys

def experiment(**kwargs):
    design_id = kwargs.get('design_id', None)
    batch = kwargs.get('batch', 100)
    N = kwargs.get('N', 10)
    ratio = kwargs.get('ratio', 4.3)
    run_till_median = kwargs.get('run_till_median', True)
    max_steps = kwargs.get('max_steps', 50)
    
    if design_id is not None:
        os.environ['SOLVER_ID'] = str(design_id)

        # Force reload of sat_solver to pick up new solver components
        # Remove cached solver modules (both old package-style and new direct-loaded)
        modules_to_remove = [k for k in sys.modules.keys() if 'sat_solver' in k or k.startswith('solvers.solver_') or k.startswith('_solver_') or '_loaded_solvers.' in k]
        for mod in modules_to_remove:
            del sys.modules[mod]

    # Import after setting environment variable so sat_solver picks up correct components
    from domain_knowledge.sat_solver import run_solver

    unsolved_fraction, median_step = run_solver(
        batch=batch,
        N=N,
        ratio=ratio,
        run_till_median=run_till_median,
        max_steps=max_steps,
        verbose=True,
        log_interval=max(100, max_steps // 10),
        hp=None
    )

    return {'unsolved_fraction': unsolved_fraction, 'median_step': median_step}

