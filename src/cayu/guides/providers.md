# Cayu providers

Cayu focuses on OpenAI, Anthropic, OpenRouter, Google, AWS, and Vertex AI
endpoints. Other services that expose OpenAI Chat Completions work through
`ChatCompletionsProvider`.

Provider selection is explicit. `CAYU_PROVIDER` is only a scaffold convenience
for `openai`, `anthropic`, `openrouter`, `cayu-gateway`, and `openai-subscription`; it is not the
complete Cayu provider surface. Credentials authenticate but never select one.

## Primary integrations

| Service | Cayu provider | Setup |
| --- | --- | --- |
| OpenAI Platform | `OpenAIProvider()` | `OPENAI_API_KEY`; use an OpenAI model ID |
| Anthropic API | `AnthropicProvider()` | `ANTHROPIC_API_KEY`; use an Anthropic model ID |
| OpenRouter | `ChatCompletionsProvider(name="openrouter", api_key_env="OPENROUTER_API_KEY", base_url="https://openrouter.ai/api/v1")` | `OPENROUTER_API_KEY`; require an explicit `vendor/model` slug |
| Cayu Gateway | `GatewayProvider(base_url="https://YOUR_GATEWAY/v1")` | `CAYU_GATEWAY_API_KEY`; explicit endpoint and model |
| Google AI Studio | `ChatCompletionsProvider(name="google", api_key_env="GEMINI_API_KEY", base_url="https://generativelanguage.googleapis.com/v1beta/openai")` | Use a Gemini API model ID |
| Amazon Bedrock | `BedrockProvider(region_name=...)` | Install `cayu[aws]`; use AWS credentials and a Bedrock model or inference-profile ID |
| Anthropic on Vertex AI | `VertexProvider(project_id=..., region=...)` | Install `cayu[vertex]`; use Google credentials and a Vertex Claude model ID |
| OpenAI subscription | `OpenAISubscriptionProvider()` | Run `cayu auth openai login`; local development and evaluation only |

Google AI Studio automatically uses Gemini usage accounting. For Gemini through
another OpenAI-compatible Vertex or gateway endpoint, pass
`usage_dialect=UsageDialect.GEMINI` explicitly.

## OpenAI subscription

`OpenAISubscriptionProvider` experimentally runs agents against the Codex backend
with the developer's own ChatGPT sign-in: `cayu auth openai login`, or add
`--headless` for the device-code flow. It is for the subscription holder's local
development and evaluation only, not production, customer-facing or multi-user
services, credential sharing, resale, or bypassing plan limits. Use the OpenAI
Platform API in production. Model availability follows the subscription;
generated projects select `gpt-6-luna`, and `CAYU_MODEL` overrides it. Sign-in
credentials stay in the trusted Cayu process and never reach runners or sandboxes.

## Explicit fallback targets

Use `RunRequest.failover=ModelFailoverPolicy(...)` with explicit provider/model pairs
in `fallbacks`. Register all candidates; `max_total_attempts` bounds each model step.
Local retries run first. Only typed retryable service rejection before accepted
output/effects permits fallback; cancellation, deadlines and ambiguity do not.
All candidates must support required tools, files, output and thinking features.
Resume retains selection; changing the chain requires profile adoption. Budgets,
deadlines and root authority do not reset; actual target pricing/usage still applies.
Tool auxiliary inference stays separate; custom stores must attest atomic selection.
See `examples/provider_failover.py` for events and a network-free native-store example.

## Thinking effort compatibility

Use `ThinkingConfig` for typed reasoning settings. Supported values, model/transport
restrictions, precedence, and backend acceptance boundaries are in
`cayu guide thinking` ([compatibility matrix](thinking.md)). `xlow` is not a verified
native value; Runtime does not equate it with `minimal` or `low`.

## Recoverable OpenAI background responses

Long OpenAI Responses calls can opt into provider-owned background execution:

```python
from cayu import AgentSpec, CayuApp
from cayu.providers import OpenAIProvider

app = CayuApp()
app.register_provider(OpenAIProvider(background=True), default=True)
app.register_agent(AgentSpec(name="assistant", model="gpt-5.6"))
```

This is a provider-registration choice, not a per-request OpenAI option.
Cayu sends `background: true`, `stream: true`, and `store: true`, records the
OpenAI response ID before accepting later output, and can retrieve or resume
that same response after worker loss. `ModelRequest.options["openai"]` cannot
override these fields. The default `OpenAIProvider()` remains synchronous.

Enable this only after reviewing the deployment tradeoffs:

- OpenAI background responses have higher time to first token than synchronous
  responses.
- A non-ZDR project stores the response at OpenAI so it can be retrieved and
  resumed. OpenAI documents a 30-day Responses application-state retention
  period and says data for `store=true` responses is retained for at least 30
  days. OpenAI forces `store=false` under Zero Data Retention, but background
  mode still stores response data on disk for roughly ten minutes for polling.
  Confirm that behavior satisfies the application's retention policy before
  enabling it.
- Cayu currently enables this mode only for the global `api.openai.com` base
  URL and rejects region-specific OpenAI domains. OpenAI separately documents
  that `background=true` is unavailable on its EU regional route. Validate the
  account, model, project policy, processing location, and storage location for
  the intended production deployment.
- OpenAI does not document exact recovery of a lost create acknowledgement from
  Cayu's idempotency key. If the response ID was never made durable, Cayu reports
  `ambiguous_submission` and does not submit the request again automatically.

This capability reconnects one provider operation. It is distinct from
server-side conversation chaining (`previous_response_id`) and from Cayu's
durable transcript. `OpenAISubscriptionProvider`, `ChatCompletionsProvider`,
Anthropic, Bedrock, Vertex, and custom adapters do not gain this capability.
See OpenAI's [data controls](https://developers.openai.com/api/docs/guides/your-data)
and [background mode](https://developers.openai.com/api/docs/guides/background)
guides for the current provider policy.

## OpenRouter

`cayu new APP --provider openrouter` generates the first-class preset. Set
`OPENROUTER_API_KEY` and an explicit `CAYU_MODEL=vendor/model`; Cayu deliberately
has no mutable router, free, or paid default. `CAYU_PROVIDER=openrouter` selects
the same preset in a neutral scaffold. Optional `OPENROUTER_HTTP_REFERER` and
`OPENROUTER_APP_TITLE` add attribution, while
`OPENROUTER_ROUTER_METADATA=enabled` retains only bounded routing evidence.
The upstream provider is reported by name (`upstream_provider`) when it is a
short plain name without a registered secret; other values become a digest. A
short secret that was never registered can still pass as a name; see
`docs/provider-error-diagnostics.md`.

Put routing controls in `AgentSpec.provider_options["openrouter"]`. Streamed
`reasoning_details` are concatenated in order and privately replayed unchanged,
value-for-value, across tool continuations; malformed state fails before tools execute.
Raw OpenRouter `usage.cost` evidence stays separate from Cayu PriceBook estimates.
Native structured-output support remains model/upstream dependent.

## Cayu Gateway

`cayu new APP --provider cayu-gateway` selects the Gateway adapter. Configure
`CAYU_GATEWAY_API_KEY`, `CAYU_GATEWAY_BASE_URL` (HTTPS, including `/v1`), and
`CAYU_MODEL`. A neutral scaffold can select it with `CAYU_PROVIDER=cayu-gateway`.
The SDK takes the endpoint explicitly:

```python
from cayu import AgentSpec, CayuApp, GatewayProvider

provider = GatewayProvider(base_url="https://YOUR_GATEWAY/v1")
app = CayuApp()
app.register_provider(provider, default=True)
app.register_agent(AgentSpec(
    name="assistant",
    model="YOUR_MODEL",
    provider_options={"cayu_gateway": {"max_completion_tokens": 2048}},
))
```

The provider uses ordinary Chat Completions streaming, tools, reasoning, and
native JSON schema output when supported by the selected Gateway model. Options
belong under `provider_options["cayu_gateway"]`. `await provider.get_models()`
reads the models visible to the key; it does not change an agent's model or limits.

Completed model events retain the response `id` and raw `usage`, including token
counters and Gateway-reported `cost`, `cost_currency`, and `cost_status`. Cost is
a decimal USD string when reported. A pending or unavailable cost is `null`,
which is distinct from a reported zero. For an available response ID,
`await provider.get_generation(request_id)` reads current status and usage using
the same key (which needs receipt-read scope). Lookup does not replay output,
resubmit inference, reserve funds, or settle a charge. Close the provider with
`await provider.aclose()` when finished with it.

The usage dashboard and `/api/usage/rollup` show up to 100 latest completion-time
reported-cost observations within the requested session filters and time window.
They are separate from PriceBook estimates, are not summed into a bill, and do
not represent current wallet balances. Truncation is explicit; narrow the filters
to inspect a smaller window. Memory, SQLite, and PostgreSQL support this projection;
custom stores may report it as unavailable. Generation lookup can return a later
reported amount but does not rewrite historical completion events or automatically
refresh these observations.

Gateway owns prices, balances, reservations, spending caps, and settlement.
Runtime's PriceBook estimates and `max_estimated_cost` remain local execution
safeguards; they are separate from the reported charge and do not promise a strict
financial ceiling. Interrupting or deleting a Runtime session does not release a
Gateway hold. A lost connection can leave remote execution and cost unknown;
response IDs are retained in completed model events. The normal provider privacy
boundary omits untrusted error IDs, so an interrupted response may leave no ID
available for lookup. The adapter disables automatic retries
of failed inference requests because a new POST can create a new charge.

### Optional catalog-priced local budgets

Load a price snapshot at application startup, before accepting work:

```python
from decimal import Decimal
from cayu import BudgetLimit, BudgetPolicy, BudgetReservation

prices = await provider.price_book()
app.budget_policy = BudgetPolicy(limits=(BudgetLimit(
    scope="app",
    max_estimated_cost=Decimal("1.00"),
    pricing=prices,
    reservation=BudgetReservation(max_input_tokens=8192, max_output_tokens=2048),
),))
```

This uses ordinary Runtime reservations and estimate-based settlement. The helper
is optional: inference does not fetch prices automatically. It makes one authenticated
`/v1/models` lookup and returns an independently usable `PriceBook`; applications
may compose its `prices` with other providers' entries in their own price book.
It never reads generation receipts, sends a charge ceiling, or changes Cloud limits.

Prices match exact model IDs under `cayu_gateway`. Nano-USD per unit is converted
to USD per million tokens as `nano_usd / per_units / 1000`, using decimal arithmetic
and rounding repeating fractions upward. Input, output, cache-read and cache-write
components map to the corresponding `PriceTier` fields. Reasoning tokens already
count as output in Runtime, so output uses the higher of the output and reasoning
rates, without adding a second reasoning charge. Each schedule's `Provenance.source`
records the exact Gateway `price_id`; its URL identifies the catalog and its `as_of`
is `unspecified`, as is the book's `generated_at`: the catalog supplies no
authoritative timestamp. Unchanged catalog pricing retains the same book and budget
identity across refresh and restart. Applications may log fetch time separately;
it is not a promised price-validity window or part of pricing identity.

Models with nonzero request/tool-call charges, unknown dimensions, unsupported
currency/version, malformed or duplicate components, or missing input/output rates
are omitted. Absent cache rates remain absent. Explicit zero token rates remain valid
zero prices. Duplicate/invalid model identities reject the catalog; an empty catalog
or one with no usable prices raises `ValueError`. No zero-price placeholder or default
catalog fallback is installed. An omitted model fails ordinary reserving-budget
admission before dispatch. Keep those budgets configured; omitting a budget is not a
substitute for handling a failed price lookup.

To refresh, explicitly call `await provider.price_book()` again, then install a
complete replacement `BudgetPolicy` as above (preserving your other limits).
Fetching does not mutate the old book, existing reservations, or application policy.
If fetching/validation fails, no replacement occurs; decide whether to retain the
previous estimate or suspend new admissions. Existing sessions retain their recorded
execution profile; changing their pricing follows normal explicit profile-adoption
rules, not automatic migration. Catalog prices can change after loading: local
estimates and reservations are not a guaranteed supplier-charge ceiling.

## Compatible Chat Completions

Fireworks, Baseten Model APIs, OpenCode Go, and other compatible endpoints work
through Cayu's generic adapter. Register it and route the agent to its name:

```python
from cayu import AgentSpec, CayuApp, ChatCompletionsProvider

provider = ChatCompletionsProvider(
    name="fireworks",
    api_key_env="FIREWORKS_API_KEY",
    base_url="https://api.fireworks.ai/inference/v1",
)
app = CayuApp()
app.register_provider(provider, default=True)
app.register_agent(
    AgentSpec(
        name="assistant",
        model="accounts/fireworks/models/YOUR_MODEL_ID",
        provider_name="fireworks",
        system_prompt="Help the user.",
    )
)
```

Change the provider name, base URL, API-key environment variable, and model ID
together:

| Service | Base URL and credential | Model ID |
| --- | --- | --- |
| Fireworks | `https://api.fireworks.ai/inference/v1`; `FIREWORKS_API_KEY` | `accounts/fireworks/models/...` |
| Baseten Model APIs | `https://inference.baseten.co/v1`; `BASETEN_API_KEY` | Baseten catalog model ID |
| OpenCode Go | `https://opencode.ai/zen/go/v1`; `OPENCODE_API_KEY` | Raw API ID such as `grok-4.5`, never `opencode-go/...` |
| Together AI | `https://api.together.ai/v1`; `TOGETHER_API_KEY` | Together catalog model ID |
| Mistral AI | `https://api.mistral.ai/v1`; `MISTRAL_API_KEY` | Mistral catalog model ID |
| Ollama | Commonly `http://127.0.0.1:11434/v1`; placeholder key | Pulled model name |
| vLLM | Commonly `http://127.0.0.1:8000/v1`; server key or placeholder | Served model name |

For local HTTP endpoints such as Ollama or vLLM, pass `allow_http=True`;
`OpenAIProvider` and `AnthropicProvider` accept the same opt-in for local
Responses or Messages servers. OpenCode
Go models span multiple protocols: use `ChatCompletionsProvider` for its Chat
Completions models and the matching Cayu protocol adapter for its other models.
Always use the raw API model ID.

Set `AgentSpec.provider_name` when routing explicitly, or register one provider
with `default=True`. Keep credentials out of `AgentSpec`, model IDs, and source
files. `cayu check` reports missing or ambiguous provider routes without calling
a provider.
