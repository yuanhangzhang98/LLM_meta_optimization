## Experiment Details
{design_details}

## Task
Summarize progress and assign {DESIGNER_AGENT_COUNT} designer agents distinct research directions with up to {MAX_DESIGNER_REFERENCE} reference experiments each.

## Output Format
```json
{{
  "current_phase": "early_exploration|systematic_search|exploitation|stagnation|breakthrough_needed",
  "key_insights": ["Top takeaways explaining what works"],
  "success_patterns": ["Shared traits of strong experiments"],
  "failure_patterns": ["Shared traits of weak experiments"],

  "research_directions": ["Direction for d1", "Direction for d2"],
  "strategy_rationales": ["Rationale for d1", "Rationale for d2"],
  "focus_areas": ["themes for d1", "themes for d2"],
  "avoid_areas": ["pitfalls for d1", "pitfalls for d2"],
  "reference_design_ids": [[int IDs for d1], [int IDs for d2]]
}}
```

## Guidelines
- All arrays must have length {DESIGNER_AGENT_COUNT} and align by index
- Directions must be non-overlapping, complementary, and distinct
