# Model catalog

Cayu ships a small, dated model-catalog snapshot for cost estimation, budget
enforcement, routing metadata, dashboards, and audit views. Applications opt into it
explicitly:

```python
from cayu import BudgetLimit, default_model_catalog

catalog = default_model_catalog()

limit = BudgetLimit(
    max_estimated_cost="5.00",
    pricing=catalog,
)
```

`default_model_catalog()` reads and validates data bundled in the installed Cayu wheel.
It performs no network request, has no cache, and returns a newly parsed `ModelCatalog`
on each call. A running app therefore keeps the same prices until its owner upgrades Cayu
or deliberately supplies another catalog.

## Coverage and provider keys

The default is an advertised subset, not a claim to cover every model or commercial
arrangement. It contains reviewed, current, tool-capable models for these provider keys:

- `openai` — OpenAI API
- `anthropic` — Anthropic API
- `google` — Gemini API / AI Studio
- `vertex` — models priced on Google Cloud Vertex AI
- `azure` — Azure OpenAI

Bedrock is intentionally outside the initial default because regional inference profiles
and pricing dimensions need separate policy. Apps can provide Bedrock or any other prices
through the existing `PricingCatalog` API.

Provider names are part of the match. For example, configure a Gemini-compatible
`ChatCompletionsProvider` with `name="google"` if it should use bundled Google prices.
Applications created from older examples that registered this provider as `gemini` must rename
it to `google` or supply an application-owned catalog under the custom provider key. Model
`aliases` create exact model pricing keys; they never alias provider names and never prefix-match
nearby model families. For example, OpenAI's `gpt-5.6` alias resolves exactly to
`gpt-5.6-sol`, while `gpt-5.6-mini` remains unknown.
Account discounts, negotiated rates, regional prices, taxes, and provider invoice rounding
are not represented. Cost remains an estimate, not billing authority.

Provider pricing is verified independently rather than inferred from another provider's
offering of the same model. The bundled Anthropic Claude models include their full 1M context
window at the standard rate. The bundled Vertex Claude records represent Google's global
endpoint rates, whose current <=200K and >200K columns are equal for these models; Vertex
multi-region and regional endpoint premiums are outside this default. OpenAI's GPT-5.6 Sol,
Terra, and Luna records use their independently verified short- and long-context rates above
the 272K boundary, including the tier-local cache-write rates. Azure announces those model
variants but does not yet publish corresponding token prices through its official pricing page
or retail-prices feed, so Cayu does not copy OpenAI's rates into Azure or guess them.

## Matching, gaps, and overrides

Bundled catalog records use exact canonical IDs, exact aliases, and explicit narrow
`match_prefixes` only where the provider publishes snapshot forms. OpenAI's generic
`gpt-5.6` alias resolves exactly to `gpt-5.6-sol`; it does not price a nearby name such as
`gpt-5.6-mini`. Azure and Anthropic dated forms use narrow `-20` prefixes; Vertex also supports
its `@20` snapshot form; Google numeric revisions use `-00`. The longest matching prefix wins
after exact canonical and alias matches.
One-off `ModelPricing` values default to raw prefix matching, so custom catalogs must choose
narrow stable prefixes or set `match="exact"` for one literal ID.
This is a deliberate pre-v0.1.0 compatibility correction; exact matching remains available.

Unknown usage is surfaced as an unpriced line item. Budget limits fail closed on unknown
prices unless an application explicitly sets `allow_unpriced=True`. Cayu does not guess a
nearby model or silently assign zero cost.

For app-specific prices, construct a `PricingCatalog` directly. Cost and budget APIs
accept either that flat compatibility contract or the richer `ModelCatalog`; public app APIs
normally receive one complete source. The low-level estimators additionally support an explicit
flat overlay beside a catalog: same-currency flat entries override matching catalog records,
while the catalog continues to price models absent from the overlay.

When cost is estimated directly with `estimate_session_cost(catalog=catalog, ...)`, each
priced line item records the matched catalog provenance. `BudgetLimit`,
`CayuApp.get_session_cost`, dashboard summaries, and server cost endpoints also accept the
`ModelCatalog` directly and select the tier from each completed step's input-token count.
`pricing_catalog()` returns the compatibility `PricingCatalog` shape while retaining
optional context-tier and provenance metadata on projected rows. Cost, budget, server, and
eval interfaces therefore produce the same result from a `ModelCatalog` and its projection.
Hand-authored legacy rows without that optional metadata keep their flat-price behavior and
serialized shape.

When a provider reports a concrete resolved model, that identity is authoritative for
pricing. Cayu consults the requested model only when the provider reports no model identity;
an unknown resolved model remains visibly unpriced and strict budgets fail closed.

## Provenance and freshness

Every bundled record includes an official HTTPS source URL and an `as_of` verification
date. Structural CI requires USD pricing, positive bounded rates, tool support, the advertised
provider set, official provider-owned hosts, and no deprecated records. Ordinary PR CI does
not compare those dates to the wall clock, so an unchanged commit cannot begin failing merely
because time passed. The scheduled refresh has a separate 30-day freshness audit; release
artifact validation enforces the same check when a `v*` release tag is pushed.

This is intentionally stricter than runtime loading: old installed releases remain usable
and deterministic, while maintenance and release automation surface an already stale snapshot.
`default_model_catalog()` deliberately emits no runtime age warning; applications choose when
to upgrade Cayu or replace the snapshot, avoiding time-dependent warnings in otherwise
deterministic application startup.

## Maintenance and releases

The repository-only maintenance agent lives under `maintenance/model_catalog`. It uses Cayu,
a real browser, structured output, strict per-session cost limits, and official-page evidence to
re-verify records. It also audits each provider's fixed official recommendation page and flags
visibly when a recommended model is not represented by a canonical record or exact alias; adding
the record and its verified prices remains a reviewed code/data change. The default weekly GitHub
Actions workflow runs its pinned
`agent-browser`/Chromium process directly on the Actions host with a scrubbed subprocess
environment; callers can instead supply a runner-backed Cayu environment to isolate browser
commands, while `search_web` remains host-backed. Browser tools receive a provider-specific
official-host allowlist, reject non-HTTPS/userinfo/nonstandard-port URLs, use one isolated browser
session, and validate the effective URL after redirects and clicks. The workflow refreshes records older than
14 days and opens a pull request containing a field-level semantic diff. Its browser job has
read-only repository permissions and passes only the candidate JSON and an escaped, bounded
report to a separate PR-creation job. Inconclusive verification keeps the committed record and
flags it for human review; it never deletes a model as an inference. Automated updates with zero,
implausibly large, or greater-than-4x price changes are also preserved and flagged for manual
correction.

The maintenance verifier itself is the canonical `openai/gpt-5.6-luna` record. Candidate
validation requires that record to remain present, active, tool-capable, and priced. An automated
deprecation verdict preserves it and flags manual replacement, preventing maintenance from
deleting the model needed to run its next refresh.

Reviewers approve catalog changes like code changes. Pull requests opened with the repository's
`GITHUB_TOKEN` may require a maintainer to approve their CI run under GitHub's workflow policy.
The committed JSON is the only release input. CI validates its schema, coverage, source policy,
wheel/sdist inclusion, and loading from a clean wheel; the scheduled and release checks enforce
age. Builds never live-fetch provider data, so the same source tree produces the same artifacts.

The former `cayu-models` repository can be archived after this maintenance path has run
successfully and a Cayu release containing the catalog is published. Its obsolete flat
`{"prices": ...}` pipeline is not part of this architecture; Cayu has one canonical
`ModelCatalog` schema and projects it into the existing `PricingCatalog` where required.
