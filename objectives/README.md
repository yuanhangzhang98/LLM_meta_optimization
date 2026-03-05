# Objective Functions

This directory contains objective functions that define what to optimize.

## Objective File Format

Each objective file `objective_N.py` (where N is the objective ID) must define:

```python
def objective(experiment_results: List[Dict[str, Any]]) -> float:
    """Compute optimization metric from experiment results.

    Args:
        experiment_results: List of experiment dicts with parameters and outcomes

    Returns:
        float: Metric value (lower is better for minimization)
    """
    ...
```

## The True Final Objective

**TRUE OBJECTIVE**: Defined in `problem_config.py` and `domain_knowledge/`

This is the ultimate goal, but it's **NOT PRACTICAL** to evaluate directly because:
- Poor designs may require excessive computational resources at target fidelity
- Evaluating each design at this fidelity is computationally prohibitive
- Need fast proxies to guide the search efficiently

## Objective Evolution Strategy

The system uses **curriculum learning** with evolving LOW-FIDELITY PROXY objectives that estimate the true objective:

### Proxy Quality Hierarchy

All objectives are **PROXIES** trying to estimate the true objective. Better proxies should:
- Correlate more strongly with true target performance
- Reduce estimation error compared to previous proxies
- Balance computational cost with prediction accuracy

### Evolution Strategy

1. **Baseline (objective_0)**: Fast but potentially inaccurate
   - Uses limited fidelity/evaluation budget from problem config
   - May underestimate poor designs due to evaluation limits
   - Scaling fits may not persist to true target

2. **Improved proxies (objective_1, 2, ...)**: Better estimates
   - May use higher fidelity (larger scale, more evaluation budget)
   - May use better extrapolation methods
   - May incorporate domain knowledge
   - Goal: Replace previous proxies with more accurate predictions

3. **Final evaluation**: Only for top candidates
   - Actual evaluation at target fidelity (may require substantial resources)
   - Done manually or in final verification stage
   - Too expensive for routine optimization

## Available Objectives

- **objective_0.py**: Baseline low-fidelity proxy (parameters defined in implementation)

## Objective Agent

The objective agent runs periodically (every 5-10 iterations) to:
- Analyze whether current proxy is discriminating well among designs
- Generate new proxy objectives that better estimate true target performance
- Re-evaluate top designs with new proxies to validate improvement
- Progress from fast/rough proxies → accurate/expensive proxies

**Key Insight**: ALL objectives in this directory are PROXIES. The true objective (defined in problem config) is too expensive to use for optimization. The goal is to create better and better proxies that help us find the truly best design without exhaustive evaluation.
