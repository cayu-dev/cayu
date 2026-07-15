# Cache-Aware Research Council

Part of Cayu's [advanced runtime example suite](../ADVANCED_RUNTIME_EXAMPLES.md).
See [Advanced runtime strategies](../../docs/advanced-runtime-examples.md) for
measured observations and proof boundaries.

This example prepares one research session, forks three distinct strategies from
the same checkpoint and causal budget, evaluates them from a sibling fork, and
repairs a substantive weakness found by the evaluator. It records the
application-controlled decision to compact before the next provider cache
boundary. The expiry timestamp is explicitly labeled as an injected observation;
Gemini does not report a cache-expiration deadline through this API.
When the injected observation enters the safety margin, the example triggers
Cayu's checkpoint-backed compaction, rebuilds the application around the same
session store, and only then creates the research forks.
From the same post-decision source checkpoint, it runs the same three branch
prompts with the full transcript and with compacted context. The evidence
envelope records total provider-reported input usage, including retries, and
requires the compacted candidate to use fewer tokens than that paired baseline;
it does not infer savings from character counts.

```bash
uv run python -m examples.cache_aware_research_council.app
# Gemini
GEMINI_API_KEY=... uv run python -m examples.cache_aware_research_council.app --mode live --provider gemini
# OpenAI
OPENAI_API_KEY=... uv run python -m examples.cache_aware_research_council.app --mode live --provider openai
# Claude
ANTHROPIC_API_KEY=... uv run python -m examples.cache_aware_research_council.app --mode live --provider anthropic
```

Assertions cover lineage, shared causal budget, the paired token delta, strategy
diversity, evaluator criticism, repair, and the cache-window decision. They do
not assert exact prose.
Input-token savings use total provider-reported usage, including structured-output
retries. First-attempt context size and model-step counts are recorded separately
to show whether compaction reduced the prepared request. Set
`CAYU_RESEARCH_COUNCIL_PRICE_BOOK` to a provenance-bearing `PriceBook` JSON file
to add a fail-closed dollar comparison across the paired branch sessions. Without
a matching price the result remains explicitly unpriced.

See [the live Anthropic Haiku benchmark](../../docs/anthropic-haiku-cost-savings-results.md)
for a three-trial result, exact denominators, and pricing provenance.
