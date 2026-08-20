# Bounded Fork Group

Part of Cayu's [advanced runtime example suite](../ADVANCED_RUNTIME_EXAMPLES.md).

This example freezes one completed source session, creates two caller-named
sibling transcript forks, runs them with bounded parallelism and one shared
causal budget, applies an application-owned deterministic gate to each result,
and sends only bounded structured evidence to a tool-free evaluator. The
evaluator must cover both successful branches exactly once and select one.

The example invokes the public `CayuApp.run_fork_group(...)` API directly. It
then repeats the identical request and proves that the terminal result is
reconstructed without another provider call. Its result assertions cover the
exact frozen checkpoint/profile relationship, one causal budget, gate identity,
structural removal of a deliberately registered evaluator tool, complete
dispositions with one winner, and token-usage evidence for every session. It
does not demonstrate viable sibling evaluation after a failed branch,
replacement candidates, distributed dispatch, or workspace promotion.

```bash
uv run python -m examples.bounded_fork_group.app

OPENAI_API_KEY=... uv run python -m examples.bounded_fork_group.app \
  --mode live --provider openai --trials 1
```
