# Evaluation Suites

This directory stores regression-oriented evaluation skeletons for prompt and workflow changes.
Each suite uses JSONL so it can be consumed by simple scripts or future benchmark runners.

Field conventions:
- `case_id`: stable identifier.
- `route`: expected workflow route.
- `prompt_id`: prompt family under evaluation.
- `input`: user request.
- `expected_checks`: assertions for lightweight regression checks.
