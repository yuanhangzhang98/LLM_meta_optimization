## Research Context
{main_research_context}

## Task
Select {DESIGNER_AGENT_COUNT} promising research directions and identify experiments to review in detail.

## Database Overview
- Total experiments: {total_experiments}
- Best performance: {best_performance}
- Recent experiments: {recent_summary}

## Experiment Summaries
Up to {MAX_EXPERIMENT_SUMMARY} experiments ranked by upper confidence bound (UCB) from Monte Carlo graph search (MCGS):
{experiment_summaries}

## Guidelines
- Request details for up to {MAX_EXPERIMENT_DETAIL} experiments
- Prioritize high-UCB experiments and complementary ideas
- Avoid redundant or near-duplicate directions
- Hyperparameter tuning is automatic via HEBO optimizer - don't assign this to Designer Agents
- Your goal: MINIMIZE the objective function

## Output Format
```json
{{
  "lookup_experiment_ids": [/* int IDs of experiments to review */],
  "context_rationale": "Concise reasoning behind chosen directions and their relevance"
}}
```