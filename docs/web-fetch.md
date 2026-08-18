# Local web fetch

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
