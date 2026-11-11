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
`cayu.connectors.sap` reads SAP S/4HANA's OData V2 services, with no write tools.
They take their client as an argument until `ToolContext.connections` exists. Model slots are
built ([Models](#models)). Connection slot declarations, the manifest document,
`ToolContext.connections` and MCP through a slot are still proposals.

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
```

- `ConnectionRequirement(slot, connector, scopes, optional)` is a frozen model, beside the built
  `ModelSlot` ([Models](#models)). Slot names are 1-63 lowercase letters, digits or interior
  hyphens, unique across connection and model slots.
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

Decided for installs with the Gateway; built.

```python
app.require_model("primary", default="bedrock/claude-sonnet-4-5", label="Primary model")
app.require_model("fast", default="bedrock/claude-haiku-4-5")
app.register_agent(support_spec, model_slot="primary")
app.register_agent(triage_spec, model_slot="fast")
```

- **Declaration.** `CayuApp.require_model(slot, *, default, label=None)` declares a model slot
  (`cayu.connections.ModelSlot`); `default` is a model id (1-128 letters, digits or `_.:/-`), or
  None when the slot needs a binding. `register_agent(..., model_slot=...)` names the slot an
  agent uses. Slot names follow the connection slot rule and are declared once; an application
  declares at most 64. An agent naming a slot the application doesn't declare is refused when
  the application is described (`describe`, `cayu check`, registration) and when a session
  starts.
- **Publication.** The agent registry registration carries `model_slots` (`[{"name", "default",
  "label"?}]`), left out when the Release declares none. A 422 `model_slots_invalid` (or a
  `settings_schema_invalid` whose `detail.also` names it) keeps the rest of the registration,
  and the process registers without its slots from then on. The app manifest (schema 18) lists
  `model_slots` and each agent's `model_slot`.
- **Binding.** The Organization binds a slot, per Agent, to one of its approved models in Cayu
  Cloud (`cayu cloud connections bind SLOT --model MODEL --application AGENT`; `cayu cloud
  models list` shows them). Developers bind the slots of Agents they can change; admins keep
  the allowed models, default, lock and caps in the Gateway policy.
- **Reading.** `ModelSlots` reads the broker's `GET {CAYU_CONNECTIONS_URL}?release=<id>`
  (`models`, `approved_models`; `release` is the process's Release, left out when it isn't
  known, so the slots listed are the ones that Release declares) in the background from when a
  server, worker, `cayu run` entrypoint or `cayu inbox` decision starts, then every 25 seconds,
  each read bounded by 10 seconds; startup waits at most 2 seconds for the first read. It keeps
  the last good snapshot: an unreachable broker, a malformed answer or a refused Release
  (`release_unknown`) changes nothing, and it logs once per outage. A slot whose `source` this
  Runtime doesn't know binds nothing. Starting refuses an agent on an undeclared slot;
  registration then leaves the slots out rather than fail.
- **Resolution at session start.** A new session's model is, in order: a queued durable
  child's prepared model, `RunRequest.target`, the deprecated Gateway policy selection, the
  agent's slot (its binding, else its declared default), then `AgentSpec.model`. The session
  pins that model; a resumed session keeps the one it started with, so a binding changes new
  sessions only. `session.started` records `model_slot: {"slot", "source"}`, where `source` is
  `bound`, `default`, or `unread` (the default, because bindings were never read since the
  process started).
- **Refusal, never substitution.** A slot whose model Cloud says isn't approved refuses new
  sessions with `ModelSlotUnavailable` (`model_not_approved`); a slot with neither binding nor
  default refuses them with `model_slot_unbound`. Nothing falls back to another model, and the
  Gateway never substitutes one: it admits or refuses the exact model. A Gateway refusal during
  a session is final for that session; failover stays explicit (`RunRequest.failover`).
- **Attribution.** A session whose model came from its slot records it when it is created
  (runtime-owned session metadata `cayu:model_slot`, `{"slot", "source", "model"}`, which
  callers can't set; a queued durable child is created with it when prepared). Its requests
  carry the slot name (`ModelRequest.model_slot`) only while they use that model: not a
  failover attempt or a resumed target on another model. A fork keeps the record only while it
  keeps the model. Sessions whose model came from anywhere else never name a slot, and a later
  Release doesn't relabel a running session.
  `GatewayProvider` sends it as `cayu.model_slot` in the request body (with `schema_version`
  1), and leaves the member out when there is none. The Gateway records it for attribution
  only.
- **Retiring the old ways.** `model_setting()` is deprecated: an agent registered with it uses
  a slot named after the agent (`_` as `-`, as Cloud moves its stored value there), declared
  with the agent's own model as its default; a bound slot wins over the setting's value, and
  `cayu check` reports `MODEL_SETTING_DEPRECATED`. Declaring a slot with `require_model` that is
  such an agent's slot name is refused. `ModelPolicyController`,
  `CayuApp(model_policy=...)` and `CAYU_MODEL_POLICY_CONFIG` are deprecated and removed in the
  next release. `CAYU_MODEL` keeps working until the scaffolds no longer need it; `cayu new`
  declares a `primary` slot whose default is the agent's model.
- Without a broker (outside Cloud) a slot uses its declared default, recorded as `default`.

## Notification content

Cloud's design reads an inbox item's content with the Agent's attention credential at delivery
time. That would be the inbox's first inbound use of that token. The alternative is for the
handoff to carry the content Cloud may deliver, already filtered by the Agent. Cloud forwards it
without storing it, and the inbox gains no new auth path. This ADR proposes the alternative;
Cloud's owners decide.

## Open questions

1. Is the application runtime the right owner of `ToolContext.connections`, or should the
   Environment carry it as it does `vault`?
2. Which connector toolkits come after Slack and Snowflake? Snowflake goes through its SQL REST
   API, which keeps to the broker; a driver would need egress the broker doesn't give.
