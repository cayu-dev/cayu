# Historical Anthropic prompt-cache observation

Date: 2026-06-23

This is a historical measurement from the prototype that preceded the current
`PromptCacheCompactor` verification. It is retained as an observation, not as a
correctness test or current pricing claim.

The isolated-prefix run reported 185,046 cache-read tokens for a near-expiry
compaction and 0 cache-read tokens for an uncompacted control after its original
entry expired. Under the prices recorded on that date, the prototype calculated
91.78% lower post-initial marginal cost for the compacted branch.

That experiment had important limits:

- it used a no-tool session, so it did not prove that tool definitions remained
  byte-stable across the cache request;
- it exercised only the first compaction cycle;
- its control was a growing full-context session, not an alternative bounded
  `ModelCompactor` strategy; and
- its hard-coded prices and one provider/model/prompt do not support a universal
  savings claim.

The executable verification now lives in
[`examples/prompt_cache_compaction/`](../examples/prompt_cache_compaction/).
The current paired live benchmark is documented separately in
[Live Anthropic Haiku cost-savings results](anthropic-haiku-cost-savings-results.md).
Its deterministic and Anthropic-live modes use a tool, preserve thinking and
tool request shape on the first cache-aware compaction, perform a second bounded
incremental compaction, and verify durable usage accounting. The live path
reports provider token counters without embedding prices:

```bash
ANTHROPIC_API_KEY=... uv run python -m examples.prompt_cache_compaction.app \
  --mode live --provider anthropic --trials 1
```

The current executable example also runs a bounded `ModelCompactor` control
from the same captured compactable source and reports that comparison-only
attempt separately from candidate-session usage. Dollar evidence is emitted
only when a provenance-bearing caller price book is supplied. This historical run
does not cover the attachment path. That boundary is verified hermetically in
`tests/core/test_prompt_cache_compactor.py`, including resolved attachment bytes
on the exact cached request prefix and bounded fallback when provider/model
identity or tool-based structured-output shape makes reuse unsafe.
