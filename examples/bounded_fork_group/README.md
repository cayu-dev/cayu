# Bounded Fork Group

Part of Cayu's [advanced runtime example suite](../ADVANCED_RUNTIME_EXAMPLES.md).

This example freezes one completed source session, creates two caller-named
sibling transcript forks, and runs them with bounded parallelism and one shared
causal budget. An application-owned deterministic gate rejects one seed
attempt, Cayu preserves its healthy sibling, and an application-owned planner
supplies one bounded replacement without choosing the replacement attempt or
session identity. Only the eligible attempts reach the tool-free evaluator.

The example invokes the public `CayuApp.run_fork_group(...)` API directly. It
then repeats the identical request and proves that the terminal result is
reconstructed without another provider call. Its result assertions cover the
exact frozen checkpoint/profile relationship, one causal budget, gate and
planner identities, immutable replacement lineage, retention of the rejected
seed, structural removal of a deliberately registered evaluator tool, complete
eligible dispositions with one winner, and token-usage evidence for every
session. The application still owns mutation and promotion policy; the example
does not add population scheduling, distributed dispatch, or workspace
promotion to Cayu.

```bash
uv run python -m examples.bounded_fork_group.app

OPENAI_API_KEY=... uv run python -m examples.bounded_fork_group.app \
  --mode live --provider openai --trials 1
```
