# Cayu offline capability references

These compact references are shipped in the Cayu package. They describe the
public seam to start from and the proof boundary to preserve. Use the smallest
section that matches the requested behavior.

## domain-tool

Subclass `Tool`, declare `ToolSpec(input_schema=...)`, and implement
`async def run(...) -> ToolResult`. The schema should describe properties,
required fields, and `additionalProperties`; `{}` is valid but unconstrained.
The public `Tool.schema` property is authoritative at registration. For a fresh
starter run `cayu generate tool NAME --agent AGENT --effect EFFECT` to create a
schema, runtime test, eval, and tracer-bullet check loop.

## approvals

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
summary but does not erase the durable transcript. Overflow recovery must be
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
events. Do not confuse successful export with live provider, network, sandbox,
or deployment verification.

## server

The server extra provides an HTTP control plane over the same application and
durable stores. Put authentication/authorization at the application boundary,
run schema migration explicitly, and separate API processes from task workers.
Test the app factory and auth boundary before deployment.

## advanced-runtime

Advanced authority, isolation, caching, speculation, and recovery strategies
compose only when their evidence boundaries are explicit. Preserve stable
operation identity, fail closed on unknown policy coverage, and verify live
capabilities separately from a structural manifest.
