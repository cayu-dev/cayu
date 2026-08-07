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

The built-in adapter accepts only credentialless HTTPS on port 443. It rejects
URL user information and IP-literal destinations. Before every initial or
redirected request, it resolves the hostname, requires every DNS answer to be a
global address, and connects to one admitted address while preserving the
original hostname for the HTTP `Host` header and TLS certificate validation.
Redirects are followed manually and pass through the same validation again.

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
access. Use a separately admitted sandbox/browser tool when page execution or
workload isolation is required; do not describe this local adapter as running
inside a configured sandbox merely because the agent also has a sandbox-backed
environment.

The tool has no authenticated-browsing surface. Do not place credentials in a
URL. Authenticated APIs should use explicit Cayu credential and egress
boundaries instead.

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
