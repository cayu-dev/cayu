# Cayu offline capability references

Cayu is a Python agent framework with an integrated durable runtime. These
capabilities let you assemble the domain-specific harness around your model;
sessions, authority, budgets and recovery govern its execution.

These compact references are shipped in the Cayu package. They describe the
public seam to start from and the proof boundary to preserve. Use the smallest
section that matches the requested behavior. For an end-to-end operational
change, start with `cayu guide durable-operations`; it composes the sessions,
approvals, tool effects, pending actions, verification, and recovery seams below.

## domain-tool

A native Python tool subclasses `Tool`, declares one immutable `ToolSpec`, and
implements `async def run(ctx: ToolContext, args: dict) -> ToolResult`.

For a tool wrapping an application-owned service, use
`cayu guide durable-service-tools` for stable component identities and explicit
knowledge injection or environment binding across process reconstruction.

`ToolSpec` is the registered declaration:

- `name`: required durable tool identity.
- `description`: model-facing purpose; defaults to an empty string.
- `input_schema`: JSON Schema for `args`; defaults to unconstrained `{}`. Declare
  properties, required fields, and `additionalProperties` deliberately.
- `parallel_safe`: whether Cayu may execute the tool alongside siblings; defaults
  to `True`.
- `effect`: replay/mutation classification; defaults to `ToolEffect.EXTERNAL`, but
  application tools should choose it explicitly with `cayu guide tool-effects`.

The public `Tool.schema` property is authoritative at registration. Override it
only when the runtime schema is derived rather than the declared
`ToolSpec.input_schema`. Most subclasses declare `spec` on the class as shown
below; a dynamically configured subclass may instead pass a `ToolSpec` to the
inherited constructor. Tool instances expose that declaration through `spec`,
derive `name` and `description` from it, and expose the effective input schema
through `schema`.

`ToolContext` carries invocation identity and admitted runtime resources. Cayu
constructs `ToolContext` for normal runtime execution and supplies the session
identity, applicable agent/environment/budget identities, JSON `metadata`, and
the resource handles admitted for that invocation. `session_id` is the only
required `ToolContext` constructor field. Optional identity fields are
`agent_name`, `environment_name`, `causal_budget_id`, `workspace_id`,
`artifact_store_id`, and `idempotency_key`. Depending on the registered
environment, Cayu may also provide `workspace`, `artifact_store`, `runner`,
`vault`, `proxy`, and `knowledge_store`. These resource handles may be absent,
so a tool that requires one must fail clearly when it is `None`. `mcp_servers`
is always a tuple, and absence is represented by `()`; check whether it is
empty rather than comparing it with `None`. Do not set the runtime-owned
secret-capture hooks; they are Cayu wiring rather than tool-author
configuration.

`ToolResult` has four fields:

- `content`: model-facing durable text; defaults to `""`.
- `structured`: optional JSON object for workflows and operator surfaces.
- `artifacts`: JSON object descriptors for durable outputs; defaults to `[]`.
- `is_error`: whether the tool reports a failed result; defaults to `False`.

`structured` and `artifacts` are recursively read-only after construction. Use
`result.model_dump(mode="json")` when a serializer or transformation needs
ordinary JSON dictionaries and lists, and construct a replacement `ToolResult`
when changing a result.

Construct a bare unit-test context with
`ToolContext(session_id="test-session")`. This direct implementation unit test
needs no application graph:

```python
import asyncio

from cayu import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec


class LookupTool(Tool):
    spec = ToolSpec(
        name="lookup",
        description="Look up one reviewed record.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        effect=ToolEffect.NONE,
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(
            content=f"Found {args['query']}",
            structured={"session_id": ctx.session_id},
        )


tool = LookupTool()
context = ToolContext(session_id="test-session")
result = asyncio.run(tool.run(context, {"query": "Cayu"}))
assert result.content == "Found Cayu"
assert result.structured == {"session_id": "test-session"}
```

Calling `run` directly proves the implementation for caller-supplied arguments.
It does not prove schema validation, policy authorization, runtime resource
admission, event emission, or model behavior. For a fresh starter run
`cayu generate tool NAME --agent AGENT --effect EFFECT` to create a schema,
runtime test, eval, and tracer-bullet check loop through Cayu.

## approvals

Run `cayu guide order-support` for the installed restartable support agent and
external-receipt-to-native-approval adapter.

The runnable proposal-to-verification recipe is `cayu guide durable-operations`.
Tool effects describe replay risk; tool policies authorize execution. Use
`AlwaysRequireApprovalToolPolicy` for a named external-effect tool that needs a
human decision. Handle `SessionStatus.INTERRUPTED`, persist the pending action,
and resume through the public approval request APIs. Approval prompts alone are
not enforcement.

For application validation plus maintained authorization, use a narrowing
composition:

```python
from cayu.tools import GuardedToolPolicy, RequiredArguments, AlwaysRequireApprovalToolPolicy

policy = GuardedToolPolicy(
    guards=(RequiredArguments({"ask_user": ("question",)}), application_guard),
    then=AlwaysRequireApprovalToolPolicy(tools=("remember_knowledge",)),
)
```

A custom `application_guard` implements async `check(request)` and returns `None`
or a `ToolPolicyResult(decision=ToolPolicyDecision.DENY, ...)`. Any other result or
ordinary exception denies. Guards cannot authorize a call or change the request
passed to downstream enforcement. Cayu checks coverage without running guards;
unknown downstream policies and overriding policy subclasses remain unknown.
Declare a guard's `execution_profile_identity` for portable recovery and bump it
when behavior changes. Undeclared custom guards remain process-local.


## environments

An `Environment` groups a workspace, runner, artifact store, vault, credential
proxy, knowledge store, and MCP servers. Register a static environment or an
`EnvironmentFactory`; choose a default explicitly. Local workspaces and runners
are development conveniences, not isolation boundaries. Start with Docker for
local development and Lambda MicroVM for cloud sandboxing. Docker shares the
host kernel and does not satisfy `ExecutionRequirements.untrusted()`. See the
[execution path and host table](https://github.com/cayu-dev/cayu/blob/main/docs/execution-capabilities.md#choose-a-sandbox-and-restrict-network-access)
for Docker runners/coding environments, cloud sandboxes and network restrictions.

## artifacts

Use an `ArtifactStore` for stable uploads and generated outputs. Bridge a
workspace file into an artifact with the public copy helpers, retain artifact
identity in durable state, and test with a temporary local artifact store.

## secrets-egress

Resolve secrets through a `Vault`/`SecretRef`, and put outbound authority in
credential proxies and `HttpEgressPolicy`. Do not place secret values in source,
prompts, manifests, diagnostics, tool results, or generated plans. Virtual
credentials keep provider credentials out of the tool process. Install
`cayu[egress]` for Docker or `cayu[aws,egress]` for Lambda MicroVM, and select
an enforcing adapter to
restrict network access, including credentialless package-index destinations.
The [virtual-egress guide](https://github.com/cayu-dev/cayu/blob/main/docs/virtual-egress.md)
shows `VirtualEgressEnvironmentFactory` and deny-by-default policy. The Docker
coding environment disables networking when no such access is needed.

## mcp

Use `McpServerSpec`, an MCP client, and `McpToolset`/`McpToolAdapter` to expose
remote tools. Register `tools=toolset.tools` for a static snapshot, or explicitly
register `mcp_toolsets=(toolset,)` when the application owns the complete source
and needs `await app.refresh_mcp_toolset(toolset)`. Refreshable sources require a
stable `connection_id`; wrappers sharing one live session also share one source
owner and generation fence, and additions never widen existing session ceilings.
Targeted, discovered, and provider-native references keep using Cayu's canonical
catalogue-bound authority: a changed live source generation rejects an older
reference before policy or target work, while an exactly unchanged refresh keeps
the existing generation. Approval recovery, resume, and forks continue through
the ordinary execution-profile and durable capability-ceiling checks rather than
an MCP-specific recovery path.
Apply `McpManifestPolicy`, and test discovery limits, quarantine, stale snapshots,
timeouts, naming collisions, and policy rejection without trusting server-provided
descriptions as authority. When an explicitly refresh-owned built-in session declares
exact `tools.listChanged: true`, Cayu fences the dirty source and coalesces
`notifications/tools/list_changed` into the same governed refresh path. Stdio uses
its existing reader; Streamable HTTP also owns an optional GET/SSE listener, with
HTTP 405 falling back to explicit refresh and never polling. See
`examples/mcp_list_changed_refresh.py` for a deterministic credential-free server.
MCP 2026-07-28 subscription and cache-hint capabilities remain deferred.

## sessions

Run `cayu guide order-support` for the installed restartable support agent and
external-receipt-to-native-approval adapter.

`SessionStore` owns durable session identity, transcript, events, status,
checkpoints, and pending actions. Use `RunRequest.session_id` when identity must
be stable and the resume/interrupt/fork APIs for lifecycle changes. Inspect a
configured store read-only with `cayu session`.

## context

Context policies select model-facing history; context counting and pressure
estimation decide when to trim or compact. Compaction produces a checkpointed
summary but does not erase the durable transcript. Default-on privacy-safe
`RequestFootprint` events describe the final prepared request without retaining
its content or making a provider call; optional keyed HMAC identities make
request and cache-prefix equality comparable within one key version. Official
provider token counting remains separately opt-in. Overflow recovery must be
bounded and provider-neutral.

## knowledge

Knowledge stores hold reviewed/retrievable entries, separate from transcript
history and working files. Use explicit namespace, visibility, status, and
actor fields. Prove local remember/search behavior with an in-memory or SQLite
store before adding embeddings or remote infrastructure.

## background-work

`TaskStore` owns durable work; a dispatcher claims it and a worker executes it.
Event watchers react to persisted events with delivery identity and retries.
Keep enqueue, claim, execution, result recording, and recovery distinct so a
process crash does not silently duplicate work.
Pass `run_task_worker(..., execution_timeout_s=...)` so a hung run ends with
`limit_reached` and fails its task instead of holding the claim; without
`run_task_worker`, wrap the code that starts runs in
`with default_run_deadline(ExecutionDeadline.after(seconds)):`
(both from `cayu.deadlines`). See `cayu guide long-running#time-limit-for-worker-runs`.

## workflows-hooks

Workflow helpers compose deterministic steps around agent runs. Runtime hooks
observe or gate documented phases. Keep orchestration state explicit and use
`workflow_tool_names` when instructions depend on exact registered tool names;
`cayu check` validates that structural contract.

## subagents

`SubagentTool` delegates model work to a child session with explicit context
and execution policy. Parent and child retain separate durable identities.
Test child completion/interruption assertions and define cancellation or
background behavior rather than treating delegation as an in-process function.

## external-agents

To run a vendor's whole agent, such as Claude Code or Codex, as a Cayu session
with recorded tokens and cost, a deadline, a USD cap and secret scrubbing,
install `cayu[claude-agent-sdk]` or `cayu[codex]` and run
`cayu guide external-agents`.

## evals

An `EvalPlan` combines a `CayuApp` and `EvalSuite`. Use
`ScriptedModelProvider`, in-memory stores, runtime-native trajectory assertions,
and bounded probes for hermetic regression proof. Scripted calls prove handling
of predetermined behavior, not live prompt comprehension.

## cost-control

Usage events feed session summaries, pricing, budgets, and run limits. Estimated
cost is evidence with provenance, not a bill. Read it with
`app.get_session_cost(session_id, pricing)`; unpriced usage is counted, not zero.

A per-session dollar cap is one line:

```python
from cayu.budgets import BudgetLimit

RunRequest(agent_name="support", messages=[...], budget_limits=(BudgetLimit.usd(5),))
```

`BudgetLimit.usd(5)` is `scope="session"`, priced by `default_price_book()`: the
dated public prices shipped with your Cayu release, each with a source URL and
`as_of` date. To change one price, pass
`pricing=default_price_book().with_prices(ModelPrice.fixed(...))`, or pass your
own `PriceBook`. A model the book does not price is refused before dispatch
unless `allow_unpriced=True`. Other scopes: `"run"`, `"agent"` (key = agent name), `"causal"` (key = causal budget id, which
defaults to the root session id and is shared by its subagents) and `"app"`;
the last three can also go in `CayuApp(budget_policy=BudgetPolicy(limits=...))`.
`window=BudgetWindow.rolling(seconds=3600)` or
`BudgetWindow.calendar(period="day")` limits the spend counted; the default is
all time. `action="notify"` only emits `budget.limit_reached`.

At the cap: spend is checked before every model call. Once it reaches the cap
the session emits `session.limit_reached` (`limit="estimated_cost"`), records
that step's tool calls as failed with `reason="limit_reached"` without running
them, and ends `interrupted` with `interruption_type="limit_reached"`. A session
cap does not reserve, so the call admitted just under it can end above it. For a
hard ceiling, reserve against the session's causal budget:

```python
BudgetLimit.usd(5, scope="causal", key=session_id,
                reservation=BudgetReservation(max_input_tokens=200_000, max_output_tokens=32_000))
```

A call whose reservation would pass the cap is refused before dispatch with
`budget.reservation_failed`. Size the reservation for the largest call; a call
that uses more than it reserved is still charged in full.

Resume: resuming with the same cap stops again without a provider call. The cap
and its prices are part of the session's execution profile, so a raised cap
needs an `ExecutionProfilePolicy` that returns `ADOPT` with
`authority_decision=AUTHORIZED` when only `INVOCATION_BUDGET_POLICY`,
`APPLICATION_BUDGET_POLICY` or `RUNTIME` changed, plus
`ResumeRequest(budget_limits=(BudgetLimit.usd(10),), profile_adoption=...)`.
No tool call from before the stop runs again, including those skipped at the
cap. Bundled prices can change on
upgrade; sessions that must resume across upgrades should load a saved book
(`dump_price_book`/`load_price_book`) instead.

Deployment-wide: `scope="app"` with a reservation, on a `PostgresBudgetLedger`
or a single-host `SQLiteBudgetLedger` that every process shares, is a
deployment-wide cap; reservations are atomic in the ledger. Every process must
configure the identical limit, price book included, because the ledger counts
spend per limit definition. Without a reservation, or with the default
in-memory ledger, an app-scope limit is not a hard cap across processes.

## observability

`cayu inspect` and `cayu check` are structural and credential-free. The console
and dashboard inspect durable state; logging and OpenTelemetry sinks observe
events. Read `tool.exposure.recorded` for content-minimized model-step capability
projection and `request.footprint.recorded` for final-request shape and typed
proof availability. Both may include the application-selected public non-secret
tool-exposure profile label; do not treat local estimates or fingerprints as
billing, exact provider-wire evidence, or proof about hidden provider prompts.
Do not confuse successful export with live provider, network, sandbox, or
deployment verification.

## server

The server extra provides an HTTP control plane over the same application and
durable stores. A generated project's `dev` extra installs it; for trusted local
inspection run `uv run cayu serve --dev` and open
`http://127.0.0.1:8000/cayu/`. Without `--dev`, `cayu serve` requires the
`[tool.cayu.serve].auth` target; generated agent and coding projects set it to
`cayu.server.environment_auth:OPERATOR_BASIC_AUTH`, which reads
`CAYU_OPERATOR_USERNAME` and `CAYU_OPERATOR_PASSWORD` at startup and refuses to
start without them (on Cayu Cloud, `cayu cloud service credentials --application
APP` shows them). The target may be any request-to-`AuthContext` dependency, such
as `BasicAuth.from_environment(...)` with other variables or, with
`cayu[oidc]`, `OidcBearerAuth.from_environment()` for bearer tokens from your
identity provider; it guards operator access to the control plane,
not end-user product login, which belongs in the service preset's
`AuthenticatedProductAccess`. Use `mount_cayu(..., path="/cayu")` when an
existing FastAPI product owns the host server; that mount requires
`AuthenticatedAccess(...)` on any public listener. Do not substitute client-IP
or forwarded-header checks for authentication. Put authentication/authorization
at the application boundary, run schema migration explicitly, and separate API
processes from task workers. Test the app factory and auth boundary before
deployment.
`OidcBearerAuth` from `cayu[oidc]` verifies JWT bearer tokens from an OpenID
Connect provider (Cognito, Auth0, Okta, Entra ID, Google, Workday) for either
operator access or, through `product_dependency()`, product access. It does not
provide browser sign-in.

## advanced-runtime

Advanced authority, isolation, caching, speculation, and recovery strategies
compose only when their evidence boundaries are explicit. Preserve stable
operation identity, fail closed on unknown policy coverage, and verify live
capabilities separately from a structural manifest.

## Human-attention notifications

Use `cayu guide human-attention` for the canonical EventSink, pending-action,
durable destination, and restart-reconciliation integration.
