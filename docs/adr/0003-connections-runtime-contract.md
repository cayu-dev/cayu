# ADR 0003: Connections in the Runtime

- Status: proposed.
- Date: 2026-11-07.
- Context: Cayu Cloud's Connections design (`plans/connections-design.md` in cayu-cloud): an
  Organization's model credentials and app connectors, used by Agents through Cloud brokers
  without the credential ever entering the Agent.

## Built so far

`cayu.connections` has `ConnectionClient`, `CloudConnectionBroker` and `LocalConnections`, and
`cayu cloud connectors|connections` manages connections. `cayu.connectors.slack` and
`cayu.connectors.snowflake` are the first connector toolkits: ready tools on a slot, an approval
policy for their write tools and an argument-review preset ([connector toolkits](../connector-toolkits.md)).
They take their client as an argument until `ToolContext.connections` exists. Slot declarations,
the manifest document, `ToolContext.connections`, MCP through a slot and model slots are still
proposals.

## Decision summary

An application declares the connections and models it needs as named **slots**. The manifest
publishes them beside managed settings. Tools reach a bound connection through a
`ConnectionClient` on `ToolContext.connections`. A host decides how a slot is served: Cloud's
broker, local credentials, or anything else that implements the protocol. Runtime code never
names an Organization's connection and never holds its credential when Cloud serves it.

## What exists and what it can't do

Checked at `110b2ab40`:

- `McpServerSpec.connection_id` (`mcp/base.py:41`) is an identity label. It is hashed into
  `cayu.mcp.connection_identity.v1` for manifest history and tool authorization, and never
  resolves a credential.
- `McpServerSpec.secret_headers` are resolved inside the Agent process when the client connects
  (`mcp/http.py:372-395`). A broker that resolved them would put the token in the Agent, so
  credential-free MCP can't be built on them.
- `ToolContext.vault` and `.proxy` come from the Environment (`environments/base.py:208`), not
  from `CayuApp`.
- `CayuApp` registers providers explicitly (`applications.py:6064`). It never selects
  `GatewayProvider` from `CAYU_GATEWAY_BASE_URL`; only the scaffold reads that variable.
- `ToolExecutionRequirement` alternatives are `executable` and `runner_capability` only.
- The attention-handoff token is sent outbound to Cloud only. The inbox routes accept the server's
  `open`, `basic` and `external` access modes, never that token.

## Declarations

```python
app.require_connection("mailbox", connector="gmail", scopes=["gmail.send", "gmail.readonly"])
app.require_connection("warehouse", connector="snowflake", optional=True)
app.require_model("primary", default="claude-sonnet-5-5")
```

- `ConnectionRequirement(slot, connector, scopes, optional)` and
  `ModelRequirement(slot, default)` are frozen models. Slot names are 1-63 lowercase letters,
  digits or interior hyphens, unique across both kinds.
- The manifest carries `manifest["connections"]`, a sorted `cayu.connection-requirements/v1`
  document with its own digest, assembled like `managed_settings_schema_document`, so Cloud
  reads it the way it reads `settings`. It holds no credential and no Organization name.
- The first slice declares slots at the application level only. A tool-level
  `ToolExecutionRequirement` alternative of kind `connection` (a tool offered only while its slot
  is bound) follows once a host reports slot state.

## ConnectionClient

Built (`cayu.connections`):

```python
class ConnectionClient(Protocol):
    async def request(
        self, slot: str, method: str, path: str, *,
        query: Mapping[str, str | Sequence[str]] | None = None,
        headers: Mapping[str, str] | None = None,
        body: bytes = b"",
    ) -> ConnectionResponse: ...                  # status, headers, body
```

Proposed beside it: `credential(slot) -> ShortLivedCredential` for opt-in connectors, which
Cloud's broker doesn't issue yet, and `slots()` in the protocol (only `CloudConnectionBroker`
has it today).

- **Exposure (proposed).** `ToolContext.connections: ConnectionClient | None`, set by the
  application runtime, not the Environment: a slot belongs to the application's identity, which
  every environment it runs in shares. This is a new wiring path beside `vault` and `proxy`,
  and it's the main structural choice to review. Until then, tools construct a client
  themselves.
- **Errors.** `ConnectionCallError`, raised only when the request wasn't sent, with the
  broker's `code`, HTTP `status` and `retry_after` (seconds, from the broker's `Retry-After`):
  `agent_credential_invalid`, `connection_unbound`, `connection_unavailable`,
  `connection_not_ready`, `connector_not_brokered`, `credential_invalid`, `scope_not_allowed`,
  `path_not_allowed`, `path_invalid`, `method_not_allowed`, `headers_invalid`, `body_too_large`,
  `request_too_large`, `too_many_calls`, `upstream_failed`, `upstream_response_too_large`. The
  client refuses `path_invalid`, `headers_invalid` and `body_too_large` itself, before sending.
  The connector's own status is the answer's, whatever it is. A bound connection that fails
  never falls back to another credential.
- **`CloudConnectionBroker`.** Configured from `CAYU_CONNECTIONS_URL`, the broker's base URL
  (`<api>/v1/agent-connections`), and `CAYU_CONNECTIONS_TOKEN`, a credential Cloud issues each
  Agent for the broker alone, so a leaked attention token can't use the Agent's connections and
  either can be replaced by itself; the URL without the token raises. Requests go to
  `POST {url}/{slot}/request` as JSON (`method`, `path`, `query`, `headers`, `body` base64), and
  `GET {url}` lists the Agent's slots. Bodies are limited to 5 MiB each way, each call to 25
  seconds at the broker, and nothing is streamed.
- **`LocalConnections`.** For laptops and tests: each slot is a `LocalSlot(base_url, headers)`
  the application gives, so the same tool code runs locally with the developer's own
  credential. It refuses the paths the broker refuses and any path that would leave the slot's
  host.

## MCP through a slot

`McpServerSpec.connection_slot` (new) routes an HTTP MCP server's requests through
`ConnectionClient.request`. The broker attaches the credential, and `secret_headers` stay
unresolved. `connection_id` keeps its identity meaning. `stdio` servers can't use a slot.

## Models

- When the Agent has `CAYU_GATEWAY_BASE_URL` and `CAYU_GATEWAY_API_KEY`, `CayuApp` registers a
  `GatewayProvider` itself and resolves every `require_model` slot through it. Each request names
  its slot (header `X-Cayu-Model-Slot`), and the Gateway applies the slot's binding. The request's
  `model` is the slot's default, which the Gateway may replace.
- Where Cloud delivers a slot directly (the SaaS path before SaaS moves onto the Gateway), it sets
  `CAYU_MODEL_<SLOT>_PROVIDER`, `_MODEL` and `_API_KEY`, and `CayuApp` builds that provider.
- Without either, the slot's default and the application's own provider configuration apply.

## Notification content

Cloud's design reads an inbox item's content with the Agent's attention credential at delivery
time. That would be the inbox's first inbound use of that token. The alternative is for the
handoff to carry the content Cloud may deliver, already filtered by the Agent. Cloud forwards it
without storing it, and the inbox gains no new auth path. This ADR proposes the alternative;
Cloud's owners decide.

## Open questions

1. Is the application runtime the right owner of `ToolContext.connections`, or should the
   Environment carry it as it does `vault`?
2. Is a slot-naming header on Gateway requests acceptable, or should the Gateway resolve the
   Agent's only model slot without one?
3. Which connector toolkits come after Slack and Snowflake? Snowflake goes through its SQL REST
   API, which keeps to the broker; a driver would need egress the broker doesn't give.
