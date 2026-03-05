## Research Context
{main_research_context}

## Planner Context
{planner_context}

## Architecture
- `domain_knowledge/{framework_module}.py` - Main framework
- `domain_knowledge/{baseline_filename}` - Baseline components
- `solvers/solver_N.py` - Your experiment components
- Objective: Consensus of all generated objectives (to MINIMIZE)
- `schedules/schedule_{current_schedule_id}.py` - Current experiment schedule

{framework_module}.py dynamically imports your {num_components} components from solver_N.py.

## Baseline Components
```python
{base_solver_code}
```

## Reference Experiments
{reference_experiments}

## Current Objective
{objective_description}

## Current Schedule
{schedule_description}

## Task
Design a new experiment by modifying the {num_components} core components:
1. Make ONE small, principled modification to baseline
2. Build on proven ideas from reference experiments
3. Follow the Planner's direction and rationale
4. Explain how changes should improve the objective

## Required Components
{component_descriptions}

Available imports: `math`, `numpy`, `scipy`, `torch`, and standard libraries

## Output Format
Return a JSON object with:
```json
{{
  "explanation": "Rationale for modification, referencing evidence and strategy",
  "solver_code": "Complete Python code with imports and components: {component_names}"
}}
```
