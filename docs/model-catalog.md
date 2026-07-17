# Model catalog and price book

Cayu ships two independent, dated package resources:

- `default_model_catalog()` returns objective model identity, aliases, capabilities,
  limits, lifecycle, and model-fact provenance.
- `default_price_book()` returns billing identities, matching rules, effective-dated
  price schedules, tier/cache rates, currency, and pricing provenance.

Both loaders are offline, validate the bundled JSON on every call, and return fresh
objects. Runtime code never fetches or updates either resource.

```python
from decimal import Decimal

from cayu import BudgetLimit, default_model_catalog, default_price_book

models = default_model_catalog()
prices = default_price_book()

selected = models.route(needs_tools=True, min_context=200_000)
limit = BudgetLimit(max_estimated_cost=Decimal("5.00"), pricing=prices)
```

## Why they are separate

Model facts and commercial prices have different owners and change on different
schedules. A model can remain technically identical while its public price changes;
custom or negotiated rates should not require an application to restate capabilities;
and a gateway billing identity can be priced without becoming a routable model.

`ModelInfo` therefore contains no rates, and `ModelCatalog` cannot be passed to cost or
budget APIs. `ModelPrice` contains no routing capabilities. Cost, budget, server,
dashboard, and evaluation paths accept one complete `PriceBook` through `pricing=`.
There is no implicit model-catalog projection, overlay, or precedence merge.

## Coverage and provider keys

The bundled resources cover a reviewed subset of current tool-capable offerings under
these provider keys:

- `openai` — OpenAI API
- `anthropic` — Anthropic API
- `google` — Gemini API / AI Studio
- `vertex` — Google Cloud Vertex AI
- `azure` — Azure OpenAI
- `bedrock` — Amazon Bedrock Runtime

The Bedrock subset is deliberately narrow: Global Claude Sonnet 4.6 at the Standard
(`default` on the wire) service tier, from the source regions documented by its AWS model
card. Geographic profiles, direct regional invocation, Reserved/Priority/Flex tiers,
provisioned throughput, negotiated rates, and other models require a complete
application-owned `PriceBook`.

Provider names are part of both model and price identity. Model aliases never alias a
provider name. Account discounts, regional premiums, negotiated rates, taxes, invoice
rounding, and reconciliation remain application/provider concerns.

## Matching

Bundled records use exact canonical IDs, exact aliases, and explicit narrow
`match_prefixes` for documented snapshot forms. Matching is case-insensitive for the
provider key, deterministic, and selects an exact match before the longest prefix.
Unknown identities stay visibly unpriced.

Application-owned `ModelPrice.fixed(...)` values default to prefix matching for concise
one-off books. Choose a narrow stable prefix or set `match="exact"` for one literal ID:

```python
from decimal import Decimal

from cayu import ModelPrice, PriceBook

prices = PriceBook(
    prices=(
        ModelPrice.fixed(
            provider_name="gateway",
            model="frontier-v2",
            match="exact",
            input_per_million=Decimal("2.00"),
            output_per_million=Decimal("8.00"),
        ),
    )
)
```

`fixed()` creates one application-owned schedule without a start or end date. It is
appropriate only when the application deliberately owns an indefinite rate.

## Amazon Bedrock billing identity

Bedrock pricing resolves from the exact `modelId` sent to `ConverseStream`, the source
region of the Boto client, and the requested/effective service tier. Omitting
`serviceTier` means Standard and is represented as `default`. When Bedrock reports an
effective tier, reconciliation uses it; a Reserved request without a reported effective
tier is unpriced because AWS may overflow it to Standard.

Contextual Bedrock prices must use `match="exact"` and a `pricing_context` whose
`dimensions` contain exactly `source_region` and `service_tier`; aliases and prefixes
are rejected. Global and geographic inference profiles therefore cannot cross-match,
nor can two source regions or tiers. An unresolved region or unsupported tier stays
visibly unpriced and strict budgets fail closed before dispatch.

Existing application price books must migrate every legacy
`provider_name="bedrock"` row before upgrading: add a `pricing_context` with the exact
source-region and service-tier sets, splitting a row when regions or tiers have
different rates.
Cayu rejects provider/model-only Bedrock rows during `PriceBook` validation instead of
silently leaving them unmatched. Use `service_tier=["default"]` for Standard on-demand
pricing; add a separate `reserved` selector only when the application has an explicit
Reserved rate.

Application inference-profile ARNs are opaque. Cayu never calls `GetInferenceProfile`
or guesses the underlying model. An application must either price the exact ARN or add a
`PricingResourceMapping(provider_name="bedrock", resource_id=...,
pricing_model=...)` to its price book. Provisioned-throughput and negotiated prices
likewise remain explicit application data; Cayu does not invent per-token amortization
for fixed commitments.

Contextual providers that distinguish cache-write TTLs can declare
`requires_cache_write_ttls=True` in their `ContextualPricingRequirement`. A price may
also opt into that behavior directly by listing `cache_write_ttls`. Otherwise a
provider-neutral billing identity continues to use the ordinary
`cache_write_input_per_million` rate.

Bedrock cache-write usage retains the reported 5-minute and 1-hour token breakdown.
Unknown-TTL writes are priced only when the contextual price's `cache_write_ttls`
declares exactly one supported TTL (or all applicable TTL rates are equal); otherwise
the whole step is unpriced rather than applying a guessed rate.

## Effective dates and reconstruction

Provider prices use explicit `PriceSchedule` values with inclusive UTC calendar-date
boundaries. Schedules for one `ModelPrice` must be ordered and non-overlapping.

- Post-hoc session and causal cost select a schedule from each durable
  `model.completed.timestamp`.
- Budget preflight and reservation select a schedule from the runtime's injected clock.
- Reconciliation prices the durable completion event, preserving the historical
  schedule and provenance in the resulting line item.
- No applicable schedule yields an explicit unpriced line item. An expired schedule is
  distinguished from an unknown identity, and strict budgets fail closed.

For example, the bundled Claude Sonnet 5 prices contain the known promotional schedule
through August 31, 2026 and the published standard schedule beginning September 1,
2026. Catalog freshness says when a fact was checked; schedule validity says when a
commercial rate applies.

The bundled price book is not an exhaustive historical archive. If an application must
reconstruct older events, it must retain every schedule needed for that history in its
own complete price book.

## Provenance and freshness

Every bundled model record and price schedule carries an official HTTPS source and an
`as_of` verification date. Structural CI validates both resources, provider coverage,
matching rules, positive bounded rates, schedule structure, source allowlists, package
contents, and clean-wheel loading.

Freshness gates apply to model facts and the price schedule active on the check date.
Historical schedules retain their original evidence so reconstruction stays auditable;
future schedules retain the evidence for the published transition and become
freshness-gated when they take effect. A gap or expired current price triggers
verification rather than falling back to another schedule.

Ordinary PR CI does not make an unchanged branch fail solely because wall time passed.
The weekly maintenance workflow performs the freshness audit, and release-tag
verification enforces it for published artifacts. Installed releases remain offline and
deterministic; applications decide when to upgrade or replace either snapshot.

## Repository maintenance

Repository-only automation lives under `maintenance/model_catalog`. A browser-backed
verification returns a metadata candidate and the corresponding price candidate. The
refresh validates and diffs both complete resources, then writes them in the same
reviewed change. A bundled price-only identity must be explicitly documented in
`BUNDLED_PRICE_ONLY_IDENTITIES`; application-owned books remain independent and can use
any gateway or negotiated billing identity. Every bundled routable model keeps a
canonical price so the maintenance verifier can re-check it; that repository invariant
does not couple application-owned `ModelCatalog` and `PriceBook` values.

Model-fact freshness and price freshness are evaluated independently. Pricing evidence
updates only schedule provenance; capability or lifecycle facts require their own
official model-page URL and quote. When an official page publishes a future rate and
start date, the verifier preserves the current schedule and creates the future schedule
at that UTC date instead of treating the discovery date as the transition.

The maintenance agent uses official-host allowlists, bounded browser evidence, strict
per-session budgets, and a fixed verifier model. Inconclusive or suspicious changes keep
the committed records and produce review flags. Builds and runtime loading never execute
the maintenance agent or contact provider sites.

## Pre-v0.1 migration

The pre-v0.1 interface intentionally removed the overlapping pricing paths:

- `PricingCatalog` → `PriceBook`
- `ModelPricing` → `ModelPrice`
- `default_model_catalog()` for model facts; `default_price_book()` for rates
- `estimate_session_cost(..., pricing=prices)` with no `catalog=` parameter
- `BudgetLimit(pricing=prices)` accepts `PriceBook` only
- dashboard config uses `priceBook`, not `pricingCatalog`

No permanent compatibility aliases are provided. This keeps one pricing contract before
the public interface stabilizes.
