# WebBridge: explicit web execution profiles

`WebBridge` is the low-friction construction boundary over Cayu's ordinary web
tools. The application selects one profile at setup; the model still sees only
the stable `web_search`, `web_fetch`, and `screenshot_page` schemas supported by
that profile.

| Profile | Tools | Executes in | Credentials | Isolation and cost |
| --- | --- | --- | --- | --- |
| `WebBridge.trusted_local()` | `web_fetch` | trusted Cayu host process | none | DNS/URL admission, not process or network isolation; ordinary host network cost |
| `WebBridge.hosted(adapter=...)` | adapter-supported `web_search` and/or `web_fetch` | trusted hosted-adapter path | declared `SecretRef` values resolved by the invocation credential proxy | provider request, usage, and cost semantics; no runner/browser authority |
| `WebBridge.sandboxed_browser(...)` | `web_fetch`, `screenshot_page` | exact admitted runner | none in browser state | brokered egress, confirmed cancellation/cleanup, pinned worker handshake, runner and artifact cost |

Construction fails if the adapter exposes no supported capability, a hosted
adapter does not declare reference-only credential authority, the sandboxed
environment lacks a configured artifact store, or the pinned browser image and
worker do not match the runner-owned workload authority. A static environment
must already expose admitted runner evidence. An `EnvironmentFactory` instead
exposes side-effect-free pre-create evidence plus the same workload and artifact
authority, so a `VirtualEgressEnvironmentFactory` can be assembled during agent
registration before its per-session runner exists. The sandboxed tools bind the
configured runner candidate, exact environment/egress authority, workload, and
artifact-store identity and revalidate all four after the factory materializes the session runner. Browser
or provider failure, missing capability, and egress denial never cause fallback
to host-process fetching or another provider.

```python
from cayu import AgentSpec, WebBridge

bridge = WebBridge.trusted_local()
app.register_agent(
    AgentSpec(name="researcher", model="your-model"),
    tools=bridge.tools,
    execution_requirements=bridge.execution_requirements,
)
```

Changing `bridge` is application configuration; prompts do not choose the
provider or security profile. See the complete
[`browse -> extract -> verify`](../examples/webbridge/research.py) and
[`external cron -> Task -> worker -> agent`](../examples/webbridge/daily_check.py)
recipes. The [recipe notes](../examples/webbridge/README.md) explain canonical
source binding, per-page failure isolation, and why recurrence remains owned by
an external scheduler today.

Hosted adapters implement the runtime-checkable
`WebBridgeCredentialAuthorityProvider` contract. Its
`webbridge_credential_authority()` method returns only the provider origin and
owned `SecretRef` values; it must not resolve credentials or perform I/O.
`bridge.register_agent(...)` verifies that the selected concrete application
environment exposes a `CredentialProxy` whose side-effect-free
`supports_webbridge_credential_authority(...)` declaration accepts that exact
origin/reference authority. The built-in passthrough and allowlist proxies
implement this declaration; custom proxies fail closed until they do. For opaque hosted
adapter code that must resume after process restart, pass an application-owned
`ExecutionProfileBehaviorIdentity` to `WebBridge.hosted(...)`; that identity is
frozen into every hosted tool profile. The selected `EnvironmentSpec` also
needs its own stable execution-profile identity for the complete app profile to
reconstruct after restart. Hosted tools repeat the compatibility
check against the active invocation proxy before adapter dispatch, so selecting
a different environment on `RunRequest` cannot bypass registration.

## Trusted local fetch

`WebFetchTool` is Cayu's low-friction way for an agent running locally to read a
public HTTPS page. It uses direct HTTP from the Cayu application process. It
does not require a browser executable, Playwright, or a hosted search provider.

```python
from cayu import AgentSpec, WebFetchTool

app.register_agent(
    AgentSpec(name="researcher", model="your-model"),
    tools=[WebFetchTool()],
)
```

The model sees one closed `web_fetch` argument:

```json
{"url": "https://example.com/reference"}
```

Provider selection, credentials, headers, HTTP methods, ports, timeouts, and
security-policy choices are application configuration, not model arguments.
The result includes the requested and final canonical URLs, page title when
available, `text` representation, extracted content, redirect evidence, a
truncation flag, and `truncation_reasons` identifying bounded title or content.
The final URL, page title, and extracted text are wrapped together as untrusted
reference data in the model-facing result; embedded closing markers are
neutralized.

## Network boundary

The default trusted-process adapter accepts only credentialless HTTPS on port
443. It rejects URL user information and IP-literal destinations. Before every
initial or redirected request, it resolves the hostname, requires every DNS
answer to be a global address, and connects to one admitted address while
preserving the original hostname for the HTTP `Host` header and TLS certificate
validation. Redirects are followed manually and pass through the same
validation again.

The transport requests identity content encoding and rejects compressed,
missing-media-type, or unsupported-media-type successful responses before
reading their bodies. Redirect and non-success bodies are left unread and are
classified from status and headers. Response bytes, extracted text, title text,
redirect count, accepted content types, and total elapsed time are bounded. The
stable operational error codes are `invalid_url`, `destination_denied`,
`dns_failure`, `redirect_denied`, `timeout`, `unsupported_content`, and
`oversized_response`.
Error results do not include response bodies or exception text. Other
connection failures use `fetch_failed`; a completed non-success response uses
`http_status` with its numeric status code and still omits the body.

These controls prevent the tool's URL input from becoming an unrestricted
private-network fetch. They do **not** make the tool a process or network
isolation boundary. `WebFetchTool.run()` executes as trusted Python code in the
Cayu application process and the application process owns its outbound network
access. Select the runner-backed browser adapter described below when
JavaScript page execution or a sandbox network boundary is required; do not
describe the default adapter as running inside a configured sandbox merely
because the agent also has a sandbox-backed environment.

The tool has no authenticated-browsing surface. Do not place credentials in a
URL. Authenticated APIs should use explicit Cayu credential and egress
boundaries instead.

## Runner-backed browser adapter

`WebFetchTool(adapter=BrowserWebFetchAdapter())` preserves the same tool name,
closed model input, output envelope, taint behavior, and limits while dispatching
a versioned Playwright/Chromium worker through the current session runner. It
never falls back to direct host HTTP. The selected environment must provide the
compatible browser image plus current admission evidence for brokered,
deny-by-default egress and deterministic command cancellation and cleanup.

Prefer `WebBridge.sandboxed_browser(environment=..., browser_image=...)` for
application setup. It validates those runner claims and artifact storage before
agent registration, binds their identities plus the exact environment/factory
egress authority into the constructed tools, and
requires the pinned `cayu-browser-fetch:5-playwright-1.62.0` image declaration.
The versioned worker handshake still verifies protocol, worker, and Playwright
versions on every dispatch. Browser inspection needs no mutable workspace; the
profile records `workspace_requirement="none"` instead of silently depending
on an ambient host directory.

For stateful interaction, select the same image through
`WebBridge.sandboxed_browser(..., interactive=True)`. That profile exposes one
closed `browser_session` tool rather than the one-shot fetch/screenshot pair.
It preserves a bounded live page across model turns, uses revision-bound opaque
refs, and publishes screenshots/downloads through ArtifactStore. It never
falls back to host browsing. See [Stateful browser sessions](browser-session.md).

The browser worker returns compact readable `text` for ordinary pages. It
deterministically selects `accessibility` when links, tables, forms, navigation
landmarks, or interactive labels would otherwise lose page relationships,
including controls in open shadow roots. Trusted metadata before the untrusted
page envelope tells the model which representation was selected and whether it
was truncated; the structured result retains the same fields for application
consumers. Both forms share the configured content-byte limit; accessibility
inspection also has hard accessibility-depth and aggregate composed-DOM-node
ceilings. Applications can lower the node ceiling with
`BrowserWebFetchAdapter(max_dom_nodes=...)`; the model-facing input remains only
`url`. After the render-settle period, the worker freezes page-authored
JavaScript and inspects the stable document from a browser-owned isolated
world. Page-defined getters or prototype overrides therefore cannot mutate the
document between node accounting and accessibility capture. Up to 32 admitted
main/child frame documents are inspected in stable tree order. Their text or
accessibility sections carry explicit frame URLs, share page-wide node and
content budgets, and retain the existing depth, request, response-byte, and
elapsed-time ceilings; frame attachment, detachment, navigation, unsupported
media, or excess frame count fails closed. Hidden, inert, and `aria-hidden`
frame subtrees still consume those safety limits but cannot select the
representation or contribute model-facing content.

Browser destinations are still application configuration. Register every
document and subresource host as an `ApprovedEgressDestination` under a
`BrowserEgressPolicy`; redirects, frames, scripts, stylesheets, images, and
fonts receive no transitive authority from the page. See
[`virtual-egress.md`](virtual-egress.md#javascript-rendered-web_fetch-in-an-admitted-runner)
and the [versioned image example](../examples/browser_fetch/README.md) for the
complete environment and image configuration.

## Sandboxed screenshot artifacts

`ScreenshotPageTool` uses the same versioned browser worker, admitted runner,
virtual-egress policy, redirect checks, request/response ceilings, ephemeral
context, and bounded cleanup path. It has no trusted-process fallback. The
model-facing `screenshot_page` schema is closed:

```json
{"url": "https://example.com/reference", "full_page": false}
```

`full_page` is the only model-selected presentation option. PNG format,
viewport, maximum page width and height, maximum pixels, screenshot bytes,
redirects, requests, response bytes, and elapsed time remain application-owned
constructor limits. The worker checks layout bounds before a full-page capture;
the host then validates the bounded base64 transport as a complete PNG,
including dimensions, chunk order, and checksums.

The tool requires both an admitted browser runner and an `ArtifactStore` on the
active environment. It stores the PNG as a session-scoped artifact and returns
one provider-neutral `image` file attachment. Its text and structured results
contain only canonical page evidence, capture dimensions, and the artifact
reference—never PNG bytes or base64. Provider translation resolves the artifact
only at the subsequent model-request boundary. Missing artifact storage,
unsupported capture, oversized layout, oversized PNG, timeout, browser crash,
and artifact publication failure remain distinct bounded errors without worker
stderr or exception text.

```python
from cayu import BrowserWebFetchAdapter, ScreenshotPageTool, WebFetchTool

tools = [
    WebFetchTool(adapter=BrowserWebFetchAdapter()),
    ScreenshotPageTool(),
]
```

Artifact creation is a durable mutation, so `screenshot_page` declares
`ToolEffect.EXTERNAL`. Runtime tool calls receive a stable idempotency identity;
the built-in hashes it into an opaque deterministic artifact ID and reconciles
an acknowledgement-lost write against exact bytes and metadata. Direct calls
without that runtime identity remain conservatively external. Add
`screenshot_page` to the same web taint-source policy as `web_fetch` when later
tools can publish or act on page-derived evidence.

## Taint policy

Fetched text is untrusted input. Mark `web_fetch` as a taint source when later
tools can publish data or perform sensitive actions:

```python
from cayu import TaintAwareToolPolicy, ToolPolicyDecision

web_policy = TaintAwareToolPolicy(
    taint_sources={"web_fetch": ["web"]},
    protected_tools={"send_email": ["web"], "execute_sql": ["web"]},
    decision=ToolPolicyDecision.REQUIRE_APPROVAL,
)

app.register_agent(
    AgentSpec(name="researcher", model="your-model"),
    tools=[WebFetchTool(), send_email, execute_sql],
    tool_policy=web_policy,
)
```

Taint is origin-based runtime policy, not a prompt-injection detector. The tool
still participates in Cayu's normal policy, approval, hook, event, transcript,
budget, cancellation, and recovery paths, and declares `ToolEffect.NONE`.

Applications can lower the trusted defaults with `max_response_bytes`,
`max_content_bytes`, `timeout_seconds`, and `max_redirects`. Constructor limits
have finite hard ceilings; none of these controls are exposed to the model.

## OpenAI-hosted Responses search

OpenAI's Responses API can execute `web_search` inside a model turn. Enable it
as typed registration authority, separately from Cayu-executed tools:

```python
from cayu import AgentSpec, OpenAIWebSearch

app.register_agent(
    AgentSpec(name="researcher", model="gpt-5.6-luna"),
    hosted_tools=[
        OpenAIWebSearch(
            search_context_size="medium",
            external_web_access=True,
            allowed_domains=("python.org",),
            return_token_budget="default",
            include_sources=True,
        )
    ],
)
```

This is provider-owned execution. OpenAI owns the network request, results,
retry behavior, quota, and billing; Cayu's local tool policy, approvals, runner,
DNS policy, and egress proxy do not mediate it. User or model prose cannot
enable search or widen its registration-time filters. Search results, source
records, and citations are retained as untrusted external evidence.

Cayu sends the native `{"type":"web_search"}` tool through `OpenAIProvider`
and the experimental `OpenAISubscriptionProvider`. It preserves bounded
lifecycle events, complete returned source metadata, inline URL citations,
completed OpenAI replay state, exact provider token usage, and a distinct count
for every terminal search call. A stream lost after search starts records an
unknown outcome; a provider retry may search and bill again. This is not an
exactly-once boundary.

Lifecycle and citation events include a runtime-owned provider-operation ID in
addition to model-step, attempt, provider, model, and hosted-call identity.
Citation offsets are normalized over the final assembled assistant text even
when OpenAI returns multiple `output_text` parts. Terminal lifecycle events are
the aggregate usage source; the response-completion copy exists only to attach
the same calls to per-attempt pricing, so the two durable views do not double
count successful searches.

`external_web_access=False` is sent exactly and means cache-only provider
access. Domain filters use bare domain names. `include_sources=True` retains the
complete source list independently of the subset cited inline. Strict cost
budgets reject this capability because the Responses API provides no hard
per-response search-call ceiling. Non-strict accounting reports completed calls
at the configured price-book rate, or explicitly unpriced/unknown evidence.
Preflight currently admits the reviewed `gpt-5.6` aliases and `chat-latest`;
other model names fail closed until their native Responses web-search support
is established. `return_token_budget="unlimited"` remains restricted to GPT-5.

See [`examples/openai_hosted_web_search.py`](../examples/openai_hosted_web_search.py)
for a live API-key example that prints durable citations, complete sources, and
session usage. The subscription backend remains an experimental local-development
path, not a documented OpenAI Platform API or API-credit substitute.

## Provider-neutral Cayu-executed search and fetch

`WebSearchTool` adds the provider-neutral `web_search` contract. Its closed
model input contains a required `query` and optional `num_results`; the
application fixes the default and maximum result counts, per-snippet bytes,
aggregate snippet bytes, and deadline. Results retain provider order as
one-based `rank`, canonical HTTPS source URLs, bounded titles and snippets, and
optional normalized publication dates or timestamps. A nullable provider title
falls back to the bounded canonical URL rather than invalidating the complete
result set. Provider scores remain under namespaced provider metadata and never
replace the portable rank.

Application-owned restrictions can be fixed with `WebSearchRestrictions`.
They are carried only to the adapter and never added to the model schema. An
adapter must enforce every configured restriction or return
`unsupported_semantics`; it cannot silently broaden a domain-, date-, country-,
locale-, or content-type-restricted search.

The opt-in `ExaWebAdapter` implements both `WebSearchAdapter` and
`WebFetchAdapter` without an Exa SDK dependency. Selecting it changes
application wiring, not the model-facing `web_search` or `web_fetch` names and
arguments:

```python
from cayu import (
    AgentSpec,
    AllowlistProxy,
    Environment,
    EnvironmentSpec,
    ExaWebAdapter,
    LocalEnvVault,
    SecretRef,
    WebFetchTool,
    WebSearchTool,
)

vault = LocalEnvVault({"exa_api_key": "EXA_API_KEY"})
proxy = AllowlistProxy(vault, allowed_destinations=["api.exa.ai"])
exa = ExaWebAdapter(
    api_key_ref=SecretRef(name="exa_api_key"),
    search_type="auto",
    search_max_age_hours=24,
    fetch_max_age_hours=24,
)

app.register_environment(
    Environment(EnvironmentSpec(name="research"), vault=vault, proxy=proxy),
    default=True,
)
app.register_agent(
    AgentSpec(name="researcher", model="your-model"),
    tools=[
        WebSearchTool(
            adapter=exa,
            default_results=5,
            max_results=10,
            max_snippet_bytes=2_048,
            max_total_snippet_bytes=8_192,
        ),
        WebFetchTool(adapter=exa),
    ],
)
```

`LocalEnvVault` resolves the reference above from `EXA_API_KEY`; applications
may use any `Vault` instead. The raw key
is authorized and resolved through the active invocation credential proxy. It
does not enter tool schemas, arguments, runner state, or result metadata. The
adapter also rejects a request before dispatch if any non-header payload value
collides with a currently protected secret.

Exa mode, provider origin, authentication header, content-cache age, moderation,
provider-response ceiling, and the two tool budgets are constructor settings.
The adapter sends at most one provider request per tool call and performs no
hidden retry. Rate limits retain a bounded `Retry-After` hint; successful calls
retain bounded request IDs, warnings, and typed estimated-cost metadata under
`provider_metadata.exa`. These provider values and all result content remain
untrusted evidence.

Hosted `web_fetch` preserves the established fetch result fields. Exa does not
publish a redirect chain, so a response that changes the canonical URL fails
with `unsupported_semantics` rather than inventing redirect evidence. Provider
and per-URL failures have stable error codes and never include raw response
bodies. Oversized, malformed, denied-credential, timeout, cancellation, and
rate-limit paths remain distinct; cancellation is never converted into a tool
error. Exa currently accepts at most 10,000 requested content characters; when
returned text reaches that request ceiling, `provider_content_limit` makes the
narrower provider boundary explicit in `truncation_reasons`.

### Parallel Search and Extract

`ParallelAIWebAdapter` implements the same two adapter protocols through
Parallel's synchronous Search and Extract APIs, also without an additional SDK
dependency. Replacing Exa requires only application wiring; the agent still
calls the same tools with the same arguments and receives the same portable
result fields:

```python
from datetime import date

from cayu import (
    ParallelAIWebAdapter,
    SecretRef,
    WebFetchTool,
    WebSearchRestrictions,
    WebSearchTool,
)

parallel = ParallelAIWebAdapter(
    api_key_ref=SecretRef(name="parallel_api_key"),
    search_mode="advanced",
    search_location="us",
    search_objective="Prefer authoritative primary sources.",
    fetch_objective="Extract the main factual content.",
    search_fetch_max_age_seconds=3_600,
    fetch_max_age_seconds=3_600,
)

tools = [
    WebSearchTool(
        adapter=parallel,
        restrictions=WebSearchRestrictions(
            include_domains=("example.com", ".gov"),
            published_on_or_after=date(2025, 1, 1),
        ),
    ),
    WebFetchTool(adapter=parallel),
]
```

Configure the environment's vault to map `parallel_api_key` to
`PARALLEL_API_KEY`, and admit `api.parallel.ai` through its proxy as in the Exa
example. Search mode, objectives, cache freshness, cache fallback, and response
ceilings remain application settings. `search_location` maps to Parallel's
non-binding geo-targeting hint; it is deliberately separate from strict search
restrictions. Parallel domain and publication-date restrictions are mapped to
its source policy. Country, locale, and content-type restrictions fail before
dispatch because Parallel's current Search contract cannot prove them.
Configured objectives are limited to Parallel's 5,000-character provider
boundary, and each search query is limited to the provider's separate
200-character ceiling. A query or fixed-objective/query composition that
exceeds its boundary fails before credential access or dispatch.

Parallel's ordered excerpts become locally byte-bounded snippets or fetch
content. Full-content mode is deliberately disabled: an unexpected
`full_content` response cannot replace the bounded excerpts. Request and session
IDs, typed SKU usage, and warnings are bounded and retained only under
`provider_metadata.parallel`. Extract's per-URL error body is never exposed;
the portable result contains a stable `fetch_failed` code, an optional HTTP
status, and bounded provider metadata. As with Exa, there is one provider
request per tool call, no hidden retry, credential values stay behind the
active proxy, and cancellation remains authoritative.

When search results can flow into sensitive tools, configure `web_search` as a
taint source alongside `web_fetch`; neither tool silently changes an
application's policy.
