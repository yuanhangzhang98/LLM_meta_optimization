## Task
Analyze results and extract actionable insights. Focus on why the outcome occurred and what to do next.
Evaluate how much each reference design influenced this result for Monte Carlo Graph Search (MCGS) updates.

## Success Levels
- **excellent**: Major breakthrough or validated improvement
- **good**: Noticeable improvement with well-understood cause
- **moderate**: Partial progress or useful insight despite limited gains
- **poor**: No improvement or regression, but still informative

## Reference Weights
- Include only referenced design IDs
- Each weight (0-1) represents influence on current design
- Weights must sum to 1.0
- Higher weights for ideas/parameters that most strongly shaped results

## Output Format
Return a JSON object with:
```json
{{
  "short_name": "Concise descriptive title (≤ 40 chars)",
  "key_insight": "Most important takeaway (1 line)",
  "success_level": "poor|moderate|good|excellent",
  "detailed_analysis": "Comprehensive explanation of mechanisms and outcomes",
  "comparison_to_references": "How results compare with referenced designs",
  "recommended_next_steps": "Concrete suggestions for future designs",
  "reference_weights": [
    {{"design_id": int, "weight": float}} for each referenced design
  ]
}}
```
