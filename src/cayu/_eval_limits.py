"""Shared suite-policy bounds, independent of Runtime and Evals imports."""

# Per-run trial concurrency. Aggregate active trials are bounded separately by
# EvalExecutionCapacity; this only stops an obviously mistyped suite setting.
EVAL_SUITE_MAX_CONCURRENCY = 1_000
