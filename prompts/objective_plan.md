## Research Context
{main_research_context}

## Meta-Agent Guidance
{meta_agent_directions}

Consider this guidance when designing your objective function. The meta-agent has analyzed research progress and identified areas that need attention.

## Task
Generate a proxy objective function and multi-fidelity experiment schedules that better estimate the research goal:
1. **Objective**: Estimate the true research goal from experiment results
2. **Schedules**: Low/medium/high fidelity evaluations that efficiently rule out poor designs while accurately assessing top performers

## Discovery Philosophy
The baseline objective is deliberately simple. You have freedom to design objectives that:
- Target any experiment or combination of all experiments
- Use any combination of available metrics
- Apply any scaling model or none at all
- Incorporate uncertainty, robustness, or other advanced concepts

Learn from existing results and think about what truly matters for identifying the best algorithms.

## Baseline Code
**Objective:**
```python
{baseline_objective_code}
```

**Schedules:**
```python
{baseline_schedule_code}
```

## Existing Objectives
{existing_objective_summary}

## Existing Schedules
{existing_schedule_summary}

## Recent Experiment Results
{recent_experiments_objectives_summary}

## Required Functions

**Objective function:**
```python
def objective(experiment_results):
    """Estimate the research goal from experiment results.

    Args:
        experiment_results: List of dicts with experiment details
    Returns:
        Float to be MINIMIZED
    """
    return objective_value
```

**Schedule functions:**
```python
def schedule_low(design_id):
    """Quick evaluation to filter poor designs."""
    return experiment_results

def schedule_medium(design_id):
    """Moderate evaluation to distinguish promising designs."""
    return experiment_results

def schedule_high(design_id):
    """Thorough evaluation of top designs."""
    return experiment_results
```

## Experiment Interface
```python
{experiment_code}
```

## Guidelines
- `experiment_results` is a list of dicts containing all experiment kwargs and outputs
- Objective should be smooth and friendly to Bayesian optimization (avoid large penalties for failures)
- Objective should adapt to different schedules and remain backward compatible when possible
- Available imports: `math`, `numpy`, `scipy`, `torch`, and standard libraries

## Output Format

Return a JSON object with:
```json
{
  "objective_description": "What this objective measures (one line, comprehensive but use extremely concise language)",
  "objective_code": "Complete Python code with imports and objective() function",
  "schedule_description": "How schedules differ across fidelities (one line, comprehensive but use extremely concise language)",
  "schedule_code": "Complete Python code with imports and schedule_low/medium/high() functions"
}
```
