# Prompt-cache compaction

This example verifies the complete `PromptCacheCompactor` lifecycle and a
paired bounded-control comparison around a real tool-using Cayu session:

1. the first compaction extends the exact provider request prefix, including
   tool definitions and thinking options;
2. its provider-reported cache-read usage is written to durable session usage;
3. after the candidate session completes, a bounded `ModelCompactor` receives
   the same captured compactable source as a comparison-only attempt;
4. candidate-session usage and comparison-only usage are reported separately;
5. both strategies preserve a mandatory retention token, and the candidate
   recovers it after the second compaction as a task-specific quality floor;
6. the next session compaction summarizes only the previous checkpoint plus the
   newly compactable delta; and
7. the checkpoint records the transition from `PromptCacheCompactor` to the
   bounded `ModelCompactor` path.

The deterministic mode exercises seven candidate-session model requests plus
one explicitly separated control request without network access. Its extra
pre-compaction turn writes the large tool-result conversation prefix before the
cache-aware call, so a cache read cannot be attributed only to stable system or
tool-definition tiers. Deterministic dollar output uses a fixture catalog that
is labeled as simulated provider pricing:

```bash
uv run python -m examples.prompt_cache_compaction.app
```

The Anthropic mode requires an API key and checks the provider's raw cache
counters. Cache observability is available only when the raw provider payload
contains cache fields; normalized zero defaults do not count as provider-reported
zeroes. Without a caller catalog it reports token categories and an explicit
`unpriced` result:

```bash
ANTHROPIC_API_KEY=... uv run python -m examples.prompt_cache_compaction.app \
  --mode live --provider anthropic --trials 1
```

`CAYU_ANTHROPIC_MODEL` selects the model and
`CAYU_ANTHROPIC_CACHE_LINES` controls the stable tool-result size. The defaults
are `claude-haiku-4-5-20251001` and 400 lines. Haiku uses budgeted extended
thinking because it does not support adaptive thinking; models such as Sonnet
4.6 retain the adaptive low-effort configuration. Set
`CAYU_ANTHROPIC_THINKING_MODE=budgeted|adaptive` when selecting another model
whose capability differs from those defaults. The bounded control receives the
same neutral thinking configuration as the cache-aware candidate.

To estimate the two first-compaction attempts in dollars, point
`CAYU_PROMPT_CACHE_MODEL_CATALOG` at an application-owned `ModelCatalog` JSON
file. The result includes catalog version/generation time and the matched
model's source, URL, and as-of date. A missing or non-matching entry remains
unpriced rather than being treated as free:

```bash
ANTHROPIC_API_KEY=... \
CAYU_PROMPT_CACHE_MODEL_CATALOG=/path/to/model-catalog.json \
uv run python -m examples.prompt_cache_compaction.app \
  --mode live --provider anthropic --trials 1
```

Every successful run writes structured evidence under
`.cayu-example-results/prompt-cache-compaction/`. Live success means all request
shape, paired-source, two-cycle, usage-accounting, and cache-read assertions
passed. It does not establish a universal cost-savings percentage or cache
lifetime. See [Cost optimization and governance](../../docs/cost-optimization.md)
for the evidence standard and failure cases.

See [the live Anthropic Haiku benchmark](../../docs/anthropic-haiku-cost-savings-results.md)
for the current three-trial savings result, exact paired costs, and pricing
provenance.
