# Counterfactual Approval

Part of Cayu's [advanced runtime example suite](../ADVANCED_RUNTIME_EXAMPLES.md).
See [Advanced runtime strategies](../../docs/advanced-runtime-examples.md) for
measured observations and proof boundaries.

This example pauses on a real Cayu tool approval, then spends the waiting time on
three authority-free child sessions: assume approved, assume denied, and explain
the decision. The approved continuation revalidates external state, performs one
protected mutation, and launches a read-only verifier against the actual result.

```bash
uv run python -m examples.counterfactual_approval.app
# Gemini
GEMINI_API_KEY=... uv run python -m examples.counterfactual_approval.app --mode live --provider gemini
# OpenAI
OPENAI_API_KEY=... uv run python -m examples.counterfactual_approval.app --mode live --provider openai
# Claude
ANTHROPIC_API_KEY=... uv run python -m examples.counterfactual_approval.app --mode live --provider anthropic
```

The speculative sessions have no mutation tools. Their output is advisory and
versioned. A second paused Cayu approval proves stale versions are rejected at
tool execution. The primary path loses acknowledgement after the downstream
mutation and receipt commit, rebuilds the application and downstream client,
and uses `inspect_tool_effect` followed by `reconcile_tool_effect` to continue
without executing the protected action twice. The registered application
reconciler checks the exact call, stable key, arguments and downstream receipt;
an operator-supplied result alone cannot settle the call. `resolve_tool_approval`
authorizes the first execution, not its speculative child sessions.

`DeployServiceTool` deliberately declares `ToolEffect.EXTERNAL`: an ambiguous
execution must be reconciled, not automatically retried. The example-owned
downstream SQLite database atomically stores its service mutation and receipt,
and supports stable-key lookup and conflicting-key rejection. Its invocation counter
is separate from the mutation counter, so downstream idempotency cannot hide
an accidental Cayu redispatch. Approval policy remains independent of this
recovery contract.

The deterministic scenario asserts one original tool invocation, one mutation,
one receipt and unchanged history on identical reconciliation replay. The real
process-loss tests in `tests/recovery/test_tool_effect_sigkill.py` use this same
adapter with a separate durable Cayu SQLite store. They interrupt both ordinary
and approval execution before invocation, during execution, after the external
commit, during receipt persistence, and after selection before continuation.
A fresh worker first recovers the dead invocation, then explicitly reconciles.
If ordinary recovery already consumed the selected tool result, receipt replay
is read-only and normal session resume owns the remaining model continuation.
An absent or pending downstream receipt remains unresolved without tool retry.

These are fixture-backed guarantees, not universal exactly-once delivery.
Another external system must supply its own authoritative receipt/lookup or
idempotency contract and validator; an LLM statement is not such evidence.
