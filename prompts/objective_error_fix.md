## Error
```
{error_message}
{full_traceback}
```

## Task
Fix the error in your objective/schedule code.

## Output Format

Return a JSON object with:
```json
{
  "error_summary": "What went wrong and how you fixed it",
  "objective_code": "Corrected Python code with imports and objective() function (blank if not needed)",
  "schedule_code": "Corrected Python code with imports and schedule_low/medium/high() functions (blank if not needed)"
}
```
