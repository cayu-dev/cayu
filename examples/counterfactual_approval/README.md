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
tool execution. The primary path injects a durable-write failure after the
external mutation, rebuilds the application around the same session store, and
uses `recover_tool_approval` plus the external receipt to continue without
executing the protected action twice. Only `resolve_tool_approval` can authorize
the first execution.
