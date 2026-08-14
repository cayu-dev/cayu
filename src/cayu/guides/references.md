# Cayu offline capability references

These compact references are shipped in the Cayu package. They describe the
public seam to start from and the proof boundary to preserve. Use the smallest
section that matches the requested behavior. For an end-to-end operational
change, start with `cayu guide durable-operations`; it composes the sessions,
approvals, tool effects, pending actions, verification, and recovery seams below.

## domain-tool

A native Python tool subclasses `Tool`, declares one immutable `ToolSpec`, and
implements `async def run(ctx: ToolContext, args: dict) -> ToolResult`.

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

The runnable proposal-to-verification recipe is `cayu guide durable-operations`.
Tool effects describe replay risk; tool policies authorize execution. Use
`AlwaysRequireApprovalToolPolicy` for a named external-effect tool that needs a
human decision. Handle `SessionStatus.INTERRUPTED`, persist the pending action,
and resume through the public approval request APIs. Approval prompts alone are
not enforcement.

## environments

An `Environment` groups a workspace, runner, artifact store, vault, credential
proxy, knowledge store, and MCP servers. Register a static environment or an
`EnvironmentFactory`; choose a default explicitly. Local workspaces and runners
are development conveniences, not isolation boundaries.

## artifacts

Use an `ArtifactStore` for stable uploads and generated outputs. Bridge a
workspace file into an artifact with the public copy helpers, retain artifact
identity in durable state, and test with a temporary local artifact store.

## secrets-egress

Resolve secrets through a `Vault`/`SecretRef`, and put outbound authority in
credential proxies and `HttpEgressPolicy`. Do not place secret values in source,
prompts, manifests, diagnostics, tool results, or generated plans. Virtual
credentials keep provider credentials out of the tool process.

## mcp

Use `McpServerSpec`, an MCP client, and `McpToolset`/`McpToolAdapter` to expose
remote tools. Apply `McpManifestPolicy` to the discovered manifest before tools
are registered. Test discovery limits, timeouts, naming collisions, and policy
rejection without trusting server-provided descriptions as authority.

## sessions

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

## evals

An `EvalPlan` combines a `CayuApp` and `EvalSuite`. Use
`ScriptedModelProvider`, in-memory stores, runtime-native trajectory assertions,
and bounded probes for hermetic regression proof. Scripted calls prove handling
of predetermined behavior, not live prompt comprehension.

## cost-control

Usage events feed session summaries, pricing, budgets, and run limits. Configure
the model catalog and price book explicitly, distinguish unpriced usage, and
test the stop boundary. Estimated cost is evidence with provenance, not a bill.

## observability

`cayu inspect` and `cayu check` are structural and credential-free. The console
and dashboard inspect durable state; logging and OpenTelemetry sinks observe
events. Read `request.footprint.recorded` for content-free final-request shape
and typed proof availability; do not treat local estimates or fingerprints as
billing, exact provider-wire evidence, or proof about hidden provider prompts.
Do not confuse successful export with live provider, network, sandbox, or
deployment verification.

## server

The server extra provides an HTTP control plane over the same application and
durable stores. A generated project's `dev` extra installs it; for trusted local
inspection run `uv run cayu serve --dev` and open
`http://127.0.0.1:8000/cayu/`. Use `mount_cayu(..., path="/cayu")` when an
existing FastAPI product owns the host server; that mount requires
`AuthenticatedAccess(...)` on any public listener. Do not substitute client-IP
or forwarded-header checks for authentication. Put authentication/authorization
at the application boundary, run schema migration explicitly, and separate API
processes from task workers. Test the app factory and auth boundary before
deployment.

## advanced-runtime

Advanced authority, isolation, caching, speculation, and recovery strategies
compose only when their evidence boundaries are explicit. Preserve stable
operation identity, fail closed on unknown policy coverage, and verify live
capabilities separately from a structural manifest.
