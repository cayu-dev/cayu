# Cost Optimization and Governance with Cayu

Cayu separates two questions that are often blurred together:

1. **Optimization:** did a strategy reduce provider work for this workload?
2. **Governance:** was every attempt attributable, priced from an auditable
   source, bounded before execution, and visible at session and workflow scope?

Optimization is empirical. A cache, compactor, smaller model, or shared prefix
can save money for one workload and lose money after cache writes, retries,
repairs, or quality failures for another. Governance is the durable runtime
contract that lets an application measure and control those outcomes.

## Strategy map

| Strategy | Best-fit workload | Cayu primitives | Evidence to require | Quality check | Common crossover or failure |
| --- | --- | --- | --- | --- | --- |
| Refresh a warm prompt cache before compaction | Long, stable provider prefix with enough reuse to amortize cache-write and refresh cost | `PromptCacheCompactor`, `CheckpointCompactionContextPolicy`, normalized cache usage | Compare with bounded `ModelCompactor` from the same compactable source; report read, write, uncached, output, and retry-inclusive usage | Summary preserves concrete constraints and later turns still complete correctly | Cold or expired cache, provider/model/request-shape drift, cache-write premium, or too little future reuse |
| Compact shared context, then fork | Several independent branches need the same expensive source context | durable checkpoints, `fork_session`, causal budget lineage, `ModelCompactor` | Paired compacted and full-context branches from one resolved source; report first attempts and whole-workflow totals | Evaluator finds material weaknesses and a repair addresses them | Compaction loses decisive evidence; evaluator or repair retries consume the input saving |
| Use a bounded or cheaper compaction model | Repeated sessions where full-prefix refresh is unavailable or uneconomical | `ModelCompactor(model=...)`, bounded input, compaction usage events | Compare the same compactable source at the same output cap; include escalation and repair attempts | Task-specific continuation/eval floor, not summary prose equality | Cheap model causes omissions, more retries, or an expensive repair |
| Retrieve a small working set | Large corpora where only a small subset is relevant per step | knowledge injection, visibility/taint metadata, context policies | Compare identical tasks with and without retrieval; include embedding/search/model calls | Recall and answer-quality gates on known relevant evidence | Retrieval misses, stale indexes, hostile stored content, or retrieval overhead dominates |
| Escalate only hard cases | High-volume workflows with a reliable cheap-model acceptance test | model override, loop/eval policies, durable attempt events, causal budgets | Compare cheap-first plus all escalation attempts with an always-expensive control | Deterministic or evaluator-backed acceptance threshold | Weak gate accepts bad output or escalates so often that savings disappear |

The first two strategies have executable advanced cost examples today. The
other rows are supported building blocks or application patterns, not Cayu
benchmark claims.

## Current executable evidence

The current live Haiku benchmark measured **66.17% aggregate savings** for
prompt-cache compaction and **48.56% aggregate savings** across paired compacted
research branch calls, with all six final trials passing their runtime and quality
gates. See [Live Anthropic Haiku cost-savings results](anthropic-haiku-cost-savings-results.md)
for the exact paired denominators, range, pricing provenance, and run IDs.

Cayu currently has six advanced runtime examples. Two are deliberately
cost-optimization examples:

- [Prompt-cache compaction](../examples/prompt_cache_compaction/) runs a
  cache-aware candidate and a bounded `ModelCompactor` control from the same
  compactable source. Candidate session usage and the comparison-only attempt
  are reported separately. Deterministic mode uses clearly labeled fixture
  prices; live mode reports provider counters and emits dollar evidence only
  when the caller supplies a provenance-bearing model catalog.
- [Cache-aware research council](../examples/cache_aware_research_council/)
  compares compacted and uncompacted branches, then runs evaluation and repair
  as separately attributed workflow sessions. A caller-supplied model catalog
  adds a provenance-gated, retry-inclusive dollar comparison across the paired
  branch sessions. It does not provide an explicit same-checkpoint, pre-expiry
  compaction lifecycle.

The other four advanced examples focus on bounded population evaluation,
authority during approval, verified repository repair, and taint-preserving
incident response. They may affect operational cost, but Cayu does not present
them as savings examples.

## Typed paired reports and claim strength

`compare_paired_cost_quality(...)` is Cayu's canonical accounting path for a
cost-optimization claim. Applications provide a bounded
`PairedCostQualityComparisonRequest`: stable pair identity, baseline and
candidate attempts, provider-reported `CostLineItem` evidence, an explicit
output budget, opaque generation-settings revision, pricing-catalog identity,
and application-owned quality results.
Cayu recomputes operation, session, branch, side, pair, and aggregate totals and
returns a `PairedCostQualityComparisonReport` with `schema_version=2`. Version 2
links each cost line item to its governing execution profile when that durable
authority is available.

The status is the strongest wording supported by the evidence:

| Status | What the application may say | What it must not imply |
| --- | --- | --- |
| `verified` | “For these eligible pairs, under quality contract X vY at threshold Z and price book P, the candidate's estimated cost was N% lower/higher.” | A universal saving, a provider invoice, or quality beyond the declared gate. |
| `measured_unmatched` | “Provider usage and estimated cost differed by X, but the declared quality evidence was missing, asymmetric, unavailable, or failed.” | “Savings” or equivalence. The raw delta is an observation only. |
| `unpriced` | “Comparable provider usage was retained, but at least one required attempt had no matching price.” | A dollar delta, zero cost, or a savings percentage. |
| `unavailable` | “The requested comparison could not be made; inspect the typed findings.” | A measured or priced comparison. |

Overall report status is fail-closed, but aggregate arithmetic includes only
`verified` pairs from one declared workload and quality cohort under one price
book snapshot, and lists every excluded pair and reason. A
`measured_unmatched` pair can retain its raw cost delta while remaining
ineligible for the verified aggregate. `unpriced` retains usage and pricing
gaps; missing usage is `unavailable`, never zero.

When projecting durable runtime events, retain every dispatched provider
attempt. A `model.started`, `model.retry`, or `model.attempt_discarded` record
without matching completion counters becomes a missing-usage attempt; it must
not disappear merely because a later retry completed successfully.

The application owns the quality policy. Both sides must use the same contract
name and version, threshold, role, task/source identity, output budget, and
`ComparableGenerationSettings` revision. That revision is an opaque `sha256:`
commitment to canonical, non-secret output-affecting settings; a mismatch fails
closed. Quality evidence references are likewise opaque `sha256:` identifiers
whose private artifact mapping stays with the application—raw prompts,
messages, model output, credentials, and arbitrary metadata are not fields in
the report. The request is capped at 4,096 attempts and 8 MiB of canonical
JSON, with durable integer ceilings and 64-digit, 64-decimal-place costs on
each cost projection. Different models or legitimate pricing tiers may be part
of one strategy comparison; every attempt still retains its configured/effective
identity and resolved price provenance. Mixed currencies, different catalog
identities, absent provenance, contradictory usage, missing sides, and duplicate
identities fail closed.

Pair and aggregate arithmetic uses `Decimal`. Savings are
`baseline_cost - candidate_cost`; percentage is that value divided by baseline
cost and rounded to two decimal places with round-half-even. A zero baseline
has `savings_percentage_state="zero_baseline"` and no percentage. A candidate
that costs more retains a negative saving and
`cost_direction="increased_cost"`.

Run `uv run python examples/cost_quality_comparison.py` for a deterministic
report containing all four statuses. The advanced prompt-cache and research
council examples emit the same public v2 shape while keeping their
workload-specific quality checks in scenario code.

## Cost governance map

| Governance need | Runtime contract | What it prevents |
| --- | --- | --- |
| Causal dollar budgets | `causal_budget_id`, causal `BudgetLimit`, `BudgetPolicy`, and causal usage/cost summaries group a parent, forks, subagents, evaluators, and repairs into one work item | Showing a cheap branch while omitting source, judge, repair, or child-session spend |
| Reservations | `BudgetReservation` plus a shared budget ledger reserves a caller-supplied worst-case model step before the provider call and reconciles normalized actual usage afterward | Concurrent workers individually passing a stale budget check and overspending the shared cap |
| Fail-closed unknown pricing | Interrupt budgets reject missing provider/model prices unless `allow_unpriced=True` is explicit | Treating an unknown model step as free or silently safe |
| Retry and repair attribution | attempt/retry events, compaction `purpose`, session lineage, and causal summaries keep extra work visible | Advertising first-attempt savings that disappear after retries or repairs |
| Cache read/write accounting | `UsageMetrics.cache` separates read, write, cached, and uncached input counters; cost line items can price each category independently | Treating every input token as full-price or ignoring cache creation premiums |
| Session and whole-workflow reports | `get_session_usage/cost`, `get_causal_budget_usage/cost`, and session/causal summary endpoints derive reports from durable events | Reconstructing spend from logs, one process, or only the terminal model response |

Pricing policy remains application owned. `PriceBook` records version,
generation time, price-specific provenance, effective dates, tiered prices, and
cache rates; `default_price_book()` supplies Cayu's dated public snapshot.
`ModelCatalog` is a separate metadata-only routing and capability resource and is
never accepted by cost or budget APIs. Applications should choose the appropriate
price book for their commercial agreement and fail closed for unknown or expired
prices when enforcing dollar limits.

Usage and estimated cost are not invoices. A provider may charge a failed or
aborted attempt without returning usable token counters; negotiated rates,
gateways, regional pricing, rounding, and catalog age can also differ from the
runtime estimate. Cayu preserves error/retry evidence and unpriced gaps rather
than manufacturing zero-cost line items. A `verified` paired report verifies
the declared comparison contract; it does not turn an estimate into an invoice.

## Evidence protocol for a savings claim

A credible comparison should record all of the following:

1. **Matched source:** both variants start from the same durable source or the
   same captured compactable request, with identical task prompts and model
   configuration except for the strategy under test.
2. **Provider counters:** report uncached input, cache read, cache write,
   output, reasoning when available, attempts, and missing usage.
3. **Two denominators:** show the first candidate attempt and the
   retry-inclusive whole workflow. Include source preparation, compaction,
   evaluation, repair, and child sessions in the latter.
4. **Pricing provenance:** identify price-book version, generated-at time,
   provider source, URL, as-of date, currency, match rule, and pricing tier.
   Without that evidence, report tokens rather than dollars.
5. **Quality floor:** use task outcomes, deterministic checks, or evaluator
   criteria that can reject a cheaper but unusable result.
6. **Failure distribution:** run enough trials to report median/range and show
   cache misses, retries, repairs, and cases where the control wins.

The key metric is not “tokens removed from a prompt.” It is acceptable outcomes
per retry-inclusive workflow dollar under a pricing snapshot the operator can
audit.

## Near-expiry compaction strategy

The intended application strategy is:

1. record a provider-, application-, or simulation-sourced cache observation;
2. evaluate a pure deadline policy with an injected clock and safety margin;
3. separately evaluate whether refreshing is economical under current pricing,
   expected future reuse, bounded miss exposure, and a quality floor;
4. schedule the explicit operation in the application or task system;
5. compact idempotently against an expected source version; and
6. fork full and compacted views from that same resolved checkpoint for paired
   evidence.

The cache-aware compactor and paired bounded-control example intentionally do
not add a hidden timer or claim to provide normalized cache observations, the
deadline decision, an explicit `compact_session` operation,
stale-version/idempotency protection, app-owned scheduling, or same-checkpoint
fork views. An unknown deadline or unknown economic result must remain
explicit; neither should be silently treated as permission to spend.

## Start here

- Run the six deterministic product stories with
  `uv run pytest -q tests/advanced_examples`.
- Run the prompt-cache pair with
  `uv run python -m examples.prompt_cache_compaction.app`.
- Inspect durable session and causal reports with
  `uv run python examples/usage_cost_summary.py`.
- Inspect all paired proof statuses with
  `uv run python examples/cost_quality_comparison.py`.
- Review [Advanced runtime strategies](advanced-runtime-examples.md) for dated
  live observations and proof boundaries.
- Review [Live Anthropic Haiku cost-savings results](anthropic-haiku-cost-savings-results.md)
  for the current paired benchmark and reproduction commands.
- Review [Runtime contracts](runtime-contracts.md) for the complete accounting,
  pricing, budget, reservation, and recovery semantics.
