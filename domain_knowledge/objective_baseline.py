import numpy as np
from scipy.stats import linregress


def objective(experiment_results):
    """
    Research objective: Efficiently and reliably solve large, hard 3-SAT instances.
    Proxy objective here: Estimate log10(median steps) at N=2000.

    This is a fast proxy that extrapolates from limited experiments. 
    It may not accurately predict performance at N=2000.

    Args:
        experiment_results: List of experiment dicts with 'N' and 'median_step' keys

    Returns:
        float: Estimated log10(median steps) at N=2000 (lower is better)
    """
    # Edge case: No experiments available
    if not experiment_results:
        return 1000.0  # Arbitrary high value

    # Edge case: Only 1 experiment - insufficient for regression
    if len(experiment_results) < 2:
        return 500.0  # Arbitrary high value
    
    Ns = np.array([result['N'] for result in experiment_results])
    medians = np.array([result['median_step'] for result in experiment_results])
    N_target = 2000

    # Fit power law: log(median) ~ a + b*log(N)
    poly_fit = linregress(np.log(Ns), np.log(medians))
    poly_pred = np.exp(poly_fit.intercept + poly_fit.slope * np.log(N_target))
    r2_poly = poly_fit.rvalue ** 2

    # Fit exponential: log(median) ~ a + b*N
    exp_fit = linregress(Ns, np.log(medians))
    exp_pred = np.exp(exp_fit.intercept + exp_fit.slope * N_target)
    r2_exp = exp_fit.rvalue ** 2

    # Edge case: Clip r2 to avoid division by zero when r2=1
    epsilon = 1e-3
    r2_poly = min(r2_poly, 1 - epsilon)
    r2_exp = min(r2_exp, 1 - epsilon)

    # Softmax weighting with temperature (lower temp = more winner-take-all)
    temperature = 10.0
    weights = np.array([1 / (1 - r2_poly), 1 / (1 - r2_exp)])
    weights = np.exp((weights - np.max(weights)) / temperature)
    weights = weights / np.sum(weights)

    # Weighted prediction
    objective_value = np.log10(weights[0] * poly_pred + weights[1] * exp_pred)

    return objective_value