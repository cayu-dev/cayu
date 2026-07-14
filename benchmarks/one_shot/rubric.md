# One-shot rubric

Rubric version: **1**.

A trial passes only when all hard gates pass and the case-specific behavior is
present in the first submission.

## Hard gates

- The project imports and its declared factory constructs through public APIs.
- `cayu inspect --json` and `cayu check --json` succeed with no qualifying findings.
- Deterministic no-key runtime tests and trajectory evals pass.
- The implementation uses only public Cayu interfaces.
- Existing Cayu session, approval, budget, artifact, recovery, and verification
  behavior is composed rather than reimplemented locally.
- Every tool has a closed schema and accurate `ToolEffect`.
- External effects have an enforcing policy/approval boundary.
- Evidence labels static, hermetic, process, and optional live proof honestly.
- No framework-specific human correction was given after the initial prompt.

## Framework leverage

The implementation should use the smallest sufficient Cayu shape. Deducting
unneeded framework machinery is part of success: a non-workflow case should not
gain workflows/tasks/approvals/servers/memory/multi-agent components without a
behavioral reason.

## Evidence quality

Record command, exit status, stdout/stderr artifact, and layer for every claimed
check. A passing test that mocks private methods, replaces the runtime, or never
crosses the public application boundary is not acceptance evidence.
