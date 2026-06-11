# Runtime Contracts

This is a design/maintainer document for the current framework foundation. It names the first contracts that must stabilize before the framework grows higher-level features.

## Boundary Data

Framework boundary data should be portable across local processes, remote runners, hosted runtimes, event stores, dashboards, and replay tools.

Payloads, metadata, tool arguments, tool results, model options, checkpoints, task data, and event data are JSON data. They must contain JSON-compatible values: objects, arrays, strings, integers, finite floats, booleans, and null. Tuples, arbitrary Python objects, non-string object keys, circular references, NaN, and Infinity are not valid boundary data. Task input, result, error, and metadata fields are top-level JSON objects with JSON-compatible nested values.

Runtime APIs copy framework objects at boundaries. User code should not mutate registered specs, request objects, message parts, event payloads, tool results, or provider events and expect those mutations to change already-registered or already-emitted runtime state.

## ContextPolicy

Builds the model-facing message list immediately before each provider request.

The durable transcript is the source record of what happened in the session. A context policy is a projection of that transcript for one model call. It may trim old messages, replace large tool results, inject retrieved context, or implement app-specific conversation routing. It must not destructively rewrite the stored transcript.

`DefaultContextPolicy` returns the current runtime transcript as model-facing context while stripping old native file attachment references by default. Custom policies implement `build(ContextRequest) -> list[Message]`. The runtime passes copied session, agent, message, environment, step, and metadata values into the policy, validates the returned messages, and then sends those messages to the provider. Invalid policy output fails the session before a provider request is made.

Context output must preserve complete tool rounds: assistant tool calls must be followed by matching tool results, and tool results cannot appear without their preceding assistant tool calls. Policies that trim recent history should use `trim_context_turns(...)` for user-turn based history or `trim_context_messages(...)` for message-count based history instead of slicing blindly. Both helpers preserve leading system messages by default.

Built-in policies include `RecentTurnsContextPolicy`, `MessageWindowContextPolicy`, and `CheckpointCompactionContextPolicy`. Recent-turn and message-window policies are pure projections over the current transcript. Built-in policies keep only the latest file-attachment tool result provider-resolvable by default; older attachment references are replaced with text/structured summaries using `strip_old_file_attachments(...)` so providers do not receive the same file bytes on every later request. Checkpoint-backed compaction is runtime-managed: it summarizes older messages through a `ContextCompactor`, stores summary state in the session checkpoint under `context_compaction`, emits `context.compaction.started`, `context.compaction.completed` or `context.compaction.failed`, emits `session.checkpointed` after successful checkpoint writes, and sends leading system messages, compacted user-context summary, and recent complete turns to the provider. It does not delete or rewrite transcript messages.

Compaction checkpoints store the summary and `compacted_transcript_cursor`, the provider-neutral transcript position covered by that summary. The model-facing summary is injected as synthetic user context, not as a system instruction, and is not appended to the durable transcript. Compaction events include cursor, compactor, count, error, and provider metadata needed for audit/debugging, but they do not include the summary text.

`TranscriptDigestCompactor` is the deterministic fallback. It converts older messages into a clipped text digest and does not perform semantic summarization. `ModelCompactor` is the provider-backed implementation for production semantic summaries: it sends a text-only compaction request with no tools to a configured `ModelProvider`, rejects tool calls from the compaction model, and stores the returned text as the checkpoint summary. Model compaction bounds the serialized compaction input with `max_input_chars` by default so very large transcripts cannot create unbounded provider requests; the default prompt preserves compaction instructions and existing summary while clipping only the newly compacted transcript digest. Callers can tune or disable that bound explicitly. Callers can provide `system_prompt` to change compaction-model behavior and `prompt_builder` to replace the user prompt body.

## Agent, Environment, Session

Cayu separates agent definition, execution environment, and session state:

- `AgentSpec`: model, system prompt, tool declarations, and metadata.
- `Environment`: workspace, artifact store, runner, vault, MCP servers, and execution metadata.
- `RunRequest` / `ResumeRequest` / `Session`: one run of an agent, optionally in a named environment, with messages, status, events, and checkpoints.

This mirrors the useful Managed Agents separation of brain, hands, and durable run history without copying any one provider API. A run may omit an environment for simple provider/tool tests, but concrete file, command, sandbox, vault, or MCP-backed tools should hang off an environment.

Runner and workspace implementations should share the same execution boundary.
For a sandbox-backed environment, `exec_command` and file tools must both talk to
the sandbox, not split command execution into the sandbox and file access into
the trusted host. `MicrosandboxRunner` pairs with `MicrosandboxWorkspace`.
`E2BRunner` pairs with `E2BWorkspace`. E2B's Python SDK command API executes
strings through Bash, so Cayu maps process-form commands with shell quoting
before sending them to E2B; this keeps Cayu's public command contract stable
while making the E2B-specific execution semantics explicit.

## ToolPolicy

Authorizes registered tool calls immediately before execution.

Tool policy is Cayu's first scoped-authority primitive. It is separate from provider formatting and runner isolation:

- providers decide what the model requested
- tool policy decides whether a registered tool call may execute
- tools and runners perform the work only after authorization

`AllowAllToolPolicy` is the default so existing simple agents continue to run without extra configuration. `StaticToolPolicy` provides a small allow/deny scope for common cases. Deny rules win over allow rules. Custom policies implement `authorize(ToolPolicyRequest) -> ToolPolicyResult`.

Denied tool calls are recoverable by default. The runtime emits `tool.call.started`, then `tool.call.blocked`, does not run the tool implementation, appends an error `tool_result` to the provider-neutral transcript, and lets the model continue. Tool policy implementation errors are not recoverable tool failures; they fail the session because the authority layer itself is broken.

Policies may also return `ToolPolicyDecision.REQUIRE_APPROVAL`. This is a durable interrupt, not an in-memory UI callback. The runtime authorizes the model's whole tool-call round before execution; if any call requires approval, it persists a `pending_tool_approval` checkpoint for the round, emits `session.checkpointed`, emits `tool.call.approval_requested`, marks the session `interrupted`, and emits `session.interrupted`. No tool implementation in that round runs before approval.

Callers resolve pending approvals with `CayuApp.resolve_tool_approval(ToolApprovalRequest(...))`. Approval emits `session.resumed`, emits `tool.call.approved` for approval-gated calls, runs executable calls in the stored round, appends the grouped tool-result message, clears the pending checkpoint, and continues the model loop. Denial emits `session.resumed`, does not execute the interrupted round, appends error results for the round, clears the pending checkpoint, and continues the model loop. The grouped tool-result message and cleared checkpoint are persisted through one atomic `SessionStore` update; if that update fails, the session returns to `interrupted` with the pending approval still present so the approval can be retried instead of creating an invalid provider history. Approval retry uses stored terminal tool events for the same approval as the execution ledger: completed, failed, blocked, or approval-denied tool results are reused and never re-executed. If a tool has `tool.call.started` for that approval without a terminal event, Cayu leaves the session `interrupted` with `manual_recovery_required` instead of re-running a side-effecting tool whose outcome is unknown. The caller can then use `CayuApp.recover_tool_approval(ToolApprovalRecoveryRequest(...))` to mark the externally verified outcome as `completed` or `failed` and provide the exact message the model should see. Recovery first claims the interrupted session, then persists the caller-supplied message as the terminal tool result, reuses it as the approval outcome, clears the checkpoint, and continues the model loop without executing the tool again. Cayu does not infer domain facts for recovery. Normal `ResumeRequest` rejects sessions with pending tool approvals because provider histories require the assistant tool-call round to be followed by matching tool results before any later conversation.

Policy requests receive copied session, agent, tool-call, argument, environment, workspace, and metadata values. Mutating a policy request does not mutate the tool arguments that may later reach the tool.

## Event

Append-only event emitted by the runtime for sessions, model steps, tools, workflows, memory, runners, and storage-visible lifecycle changes.
Framework event types are enumerated. Extension events must use the `custom.` namespace so typos do not silently become durable event names.

Events power:

- terminal output
- dashboard
- webhooks
- session replay
- hosted platform adapters
- tests and debugging

## SessionStore

Creates sessions, stores events, stores provider-neutral transcripts, and checkpoints runtime state.

`RunRequest.session_id` is an optional caller-provided id for a new session. It must be unique. `RunRequest.task_id` optionally links a session run to an existing task. Reusing `RunRequest.session_id` never resumes an existing session.
`RunRequest.environment_name` optionally selects a registered environment. If omitted, the runtime may use the default registered environment; if no environment is registered, simple runs can still execute without one.
Events emitted for an environment-backed run carry `environment_name` as a top-level event identity field, not as payload data. Runtime code owns this field and converts provider stream events before emitting runtime events.

`ResumeRequest` explicitly continues an existing session. It loads the stored provider-neutral transcript, appends the new request messages to that same transcript, emits `session.resumed`, and runs the same model/tool loop as a new session. Resume uses the session's stored agent, provider, model, runtime, and environment identity instead of current application defaults. This prevents an application default change from silently continuing an old session on a different provider or model. `ResumeRequest.model` may update the session's active model within the stored provider; that update is durable and future resumes use the new model. Provider switching is intentionally not part of resume yet. A session can be resumed only from `completed`, `failed`, or `interrupted`. `pending`, `running`, and `cancelled` sessions are rejected so concurrent workers do not continue the same session at the same time and cancelled work does not restart accidentally. Interrupted sessions with a pending tool approval must be continued through `resolve_tool_approval(...)`, not `resume(...)`. Session stores expose an atomic status transition for this boundary.

`CancelSessionRequest` marks a `pending`, `running`, or `interrupted` session as `cancelled` and emits `session.cancelled`. Cancellation is durable and cooperative: active runtime loops check session status before model requests, after provider responses, before tool-policy planning, and between tool calls. The runtime does not claim to kill every in-flight provider request immediately; provider and runner hard cancellation are adapter-specific capabilities. If a linked task has been started and the runtime observes cancellation before terminal completion, the task is marked `cancelled` and `task.cancelled` is emitted. The optional server exposes this as `POST /api/sessions/{session_id}/cancel`.

`ForkSessionRequest` creates a new session branch from an existing `completed`, `failed`, or `interrupted` session without mutating the source. Fork keeps the source provider fixed, copies the source transcript, optionally applies a same-provider model override, and persists `parent_session_id` on the child session. Full-session forks copy checkpoint state by default and emit `session.forked` on the child session. Partial transcript forks use `transcript_cursor` to copy messages through a 1-based provider-neutral transcript cursor and must set `copy_checkpoint=False`; checkpoint state is not safe to copy when it may refer to transcript messages omitted from the fork. Interrupted sessions cannot be forked without checkpoint state. When a pending tool approval checkpoint is forked, Cayu copies the approval state but clears the source task id so resolving the fork cannot update a task owned by the source session.

`DispatchRequest` asks a `Dispatcher` to submit work for an existing session and return a `DispatchHandle`. Dispatch is separate from fork: fork decides what state a branch starts from, while dispatch decides how a session run is placed. The default `InlineDispatcher` runs immediately in the current process by resuming the target session through the normal runtime loop, then returns a completed, failed, interrupted, or cancelled handle based on the terminal session event. It is useful for tests, local execution, and proving orchestration logic, but it is not durable background execution. Production apps can provide another `Dispatcher` that submits work to an external queue or hosted runtime and returns a queued/submitted handle while events are observed through the session store. `CayuApp.dispatch_inline(...)` is the explicit local streaming API for callers that want to consume ordinary `session.resumed`, model, tool, task, interrupt, cancellation, and terminal session events directly. `DispatchRequest.task_id` optionally links dispatched work to a task; using it with inline execution requires `CayuApp(task_store=...)`.

`RuntimeHook` provides lifecycle automation around durable runtime boundaries. Terminal session phases are `after_session_completed`, `after_session_failed`, `after_session_interrupted`, and `after_session_cancelled`. These hooks run only after the terminal session status and terminal event have already been persisted. A hook failure does not rewrite the terminal session status; Cayu records `hook.failed` and continues to later hooks. Successful hooks emit `hook.started` and `hook.completed`, including the hook scope, terminal event id/type, and JSON-safe action summaries. `RuntimeHookContext` exposes copied session/event data plus controlled helpers for `fork_session`, `create_task`, `dispatch`, `dispatch_inline`, and custom event emission.

`after_tool_call` runs after Cayu has persisted a terminal tool result event from the model/tool loop, such as `tool.call.completed`, `tool.call.failed`, or `tool.call.blocked`. It receives `ToolCallHookContext`, which exposes copied session data, the persisted tool event, tool name/id, copied arguments, copied `ToolResult`, optional task id, and the same controlled helpers as terminal hooks. Mutating context arguments or results does not mutate the durable transcript or the tool result sent back to the model. This phase is for policy telemetry, audit trails, memory extraction, and follow-up work; it is not a tool-result rewrite hook.

Hook helper side effects are persisted and sent to event sinks; the parent run stream yields the hook telemetry events. Hook-emitted custom events must use the `custom.` namespace. `CayuApp(runtime_hooks=[...])` registers app-level hooks for global middleware, while `register_agent(..., runtime_hooks=[...])` registers hooks that run only for that agent. App-level hooks run before agent-level hooks. Hooks that fork and dispatch follow-up sessions should still guard on session metadata, task type, or another app-owned marker when they can process their own follow-up sessions. This lets apps implement follow-up work such as “fork this completed builder session and dispatch a knowledge-extraction task” without making fork, task, or dispatch mean the same thing.

`SQLiteSessionStore` is the durable local implementation. It stores sessions, append-only events, provider-neutral transcript messages, and the latest checkpoint in SQLite, while keeping session identity and event identity fields queryable as columns. Session identity includes agent, provider, active model, runtime, environment, and parent session. Event identity includes event type, agent, environment, workflow, and tool. `InMemorySessionStore` remains for tests and small examples. Hosted use can later provide a different `SessionStore`, such as Postgres, without changing runtime behavior.

JSONL can be added later as an export/debug format. It should not be the primary Cayu session store because dashboards, replay, task orchestration, retries, and hosted runtimes need indexed structured queries and transactional state updates.

Session stores expose two read surfaces:

- `load_events(session_id)` returns the full event list for one session.
- `query_events(EventQuery(...))` returns `EventRecord` values with durable sequence numbers for filtered timeline/dashboard reads.

Session stores also expose `list_sessions(SessionQuery(...))` for dashboard and replay views, and `load_transcript(session_id)` for the provider-neutral model conversation used by resume and compaction APIs. Runtime code can write one event with `append_event(...)` or write a durable batch with `append_events(...)`. Runtime code appends transcript messages as it builds the model conversation: initial messages, resumed request messages, assistant model messages, and tool-result messages. Batched event appends must be atomic: if one event in the batch is invalid or duplicated, none of the batch should be persisted. Terminal tool events must be durable before approval retry can safely skip execution. `append_transcript_messages_and_checkpoint(...)` must also be atomic: it is the boundary used when closing an interrupted tool-approval round, where the transcript cannot be updated independently from the checkpoint.

## TaskStore

Creates and updates optional durable units of work.

A task is not a PM-specific object. It is a generic work item that can represent a webhook job, background agent run, workflow step, orchestrator assignment, coding task, invoice-processing job, report generation job, or external automation. Simple one-off agent calls do not need tasks; they can use only sessions and events.

`Task` values have type, status, optional session/parent-task/assigned-agent identity, JSON-object input, optional JSON-object result/error, JSON-object metadata, and lifecycle timestamps. `TaskStore` exposes:

- `create_task(TaskCreate(...))`
- `load_task(task_id)`
- `list_tasks(TaskQuery(...))`
- `start_task(task_id, session_id=...)`
- `complete_task(task_id, result)`
- `fail_task(task_id, error)`
- `cancel_task(task_id, error=...)`

Valid task lifecycle is intentionally small for the foundation:

```text
pending -> running
pending -> completed | failed | cancelled
running -> completed | failed | cancelled
terminal statuses do not transition
```

`InMemoryTaskStore` exists for tests and examples. `SQLiteTaskStore` is the durable local implementation.

`CayuApp(task_store=...)` can link an agent run to an existing task through `RunRequest.task_id`. The runtime starts that task with the created session id, emits `task.started`, and then marks the task completed, failed, or cancelled before emitting the terminal session event. This is a task/session bridge, not a queue worker, retry engine, workflow engine, or agent communication table.

## EventSink

Receives events and forwards them somewhere:

- Python logging
- dashboard websocket
- webhook
- log file
- database
- hosted platform adapter

`CayuApp` registers `LoggingEventSink` by default. It emits concise summaries to
`logging.getLogger("cayu")` and does not configure process-wide handlers, levels,
or formatters. It must not log full prompts, raw file contents, or raw tool
arguments by default. Logged values are escaped onto a single line to avoid log
injection. Error summaries can include lower-level exception text, so
applications that resolve secrets should configure `LoggingEventSink(redactor=...)`
to redact known secret values. Applications can disable the default sink with
`CayuApp(enable_logging=False)`, pass additional sinks through
`CayuApp(event_sinks=[...])`.

## Agent

Turns messages into event streams using:

- model providers
- tools
- memory
- workflows
- runtime services

The initial `CayuApp` runtime registers agent specs, model providers, and tools, then emits and persists events for one session run. A run may make multiple model requests: model output can request tools, the runtime executes those tools, appends assistant `tool_call` messages and matching `tool_result` messages, and calls the model again until the model completes without tool calls or `RunRequest.max_steps` is exceeded. Multiple tool calls from one model step are grouped into one assistant message and one tool-result message in Cayu's internal transcript. Provider adapters must emit a `completed` stream event for each model step; a stream that ends silently is treated as a failed runtime contract.

`CayuApp.run()` and `CayuApp.resume()` are event-stream APIs. Runtime failures are represented as terminal `session.failed` events rather than re-raised exceptions from the iterator. A stricter programmatic API can be added later on top of the same runtime path.

## Provider

Model providers translate model-specific APIs into Cayu runtime contracts.

Provider adapters must:

- receive a copied `ModelRequest`
- yield `ModelStreamEvent` values
- emit a `completed` stream event for each model step
- stop emitting after `completed`
- keep provider-specific API payload formatting isolated inside the adapter

The runtime owns conversion from `ModelStreamEvent` to durable runtime `Event` records, including `session_id`, `agent_name`, and `environment_name`. Provider errors should be yielded as model error stream events; the runtime records the model error event and fails the session. Tool calls should be emitted as structured tool-call stream events so the runtime can execute tools and feed structured results back into the next model step.

Providers that require opaque response items for stateless continuation may return transcript-only `provider_state` in completed stream-event payloads. The runtime stores that state as assistant `ProviderStatePart` content so future provider requests can replay it, but strips it from `model.completed` event payloads and compaction metadata. Provider state is not user-facing text and should not be treated as dashboard telemetry.

`AnthropicProvider` adapts the Anthropic Messages API to Cayu's provider-neutral transcript. It keeps Cayu `system` messages as Anthropic's top-level `system` field, maps assistant tool calls to Anthropic `tool_use` blocks, and maps Cayu tool-result messages back to Anthropic user `tool_result` blocks.

`OpenAIProvider` adapts the OpenAI Responses API to the same Cayu transcript. It keeps Cayu `system` messages as OpenAI `instructions`, maps assistant tool calls to Responses `function_call` items, maps Cayu tool-result messages to `function_call_output` items, and sets `store: false` by default so Cayu remains the durable session source of truth. It uses OpenAI Responses server-sent-event streaming by default, normalizes typed text/function-call/completed events into Cayu provider stream events, and enforces a provider-event idle timeout so a stalled stream fails the model step instead of leaving the session running indefinitely. Callers can override OpenAI request options through `ModelRequest.options["openai"]` except for fields owned by the provider contract.

Configure OpenAI transport timeouts on the provider. `timeout_s` controls ordinary HTTP transport timeouts; `stream_idle_timeout_s` controls how long a streaming response may go without a parsed provider event before Cayu treats the model step as stalled:

```python
OpenAIProvider(timeout_s=600, stream_idle_timeout_s=300)
```

`AnthropicProvider` currently uses complete API responses and yields normalized Cayu stream events from the returned model response. Anthropic server-sent-event streaming can be added behind the same provider contract later.

## Tool

Runs a capability and returns `ToolResult`.

Tool declarations are captured when an agent is registered with `CayuApp`.
The registered name, description, and input schema are the public contract shown to the model for that agent.
Changing `tool.spec` after registration does not update the registered agent or the model-facing tool declaration.
To change a tool's public contract, create/register a new agent configuration or re-register the tool through an explicit runtime API once one exists.

Tool results must support:

- model-facing text
- structured output
- artifacts
- error state

String-only tool results are not enough for the final framework.

Tool failures are recoverable by default. They are recorded as `tool.call.failed` events and returned to the model as structured `tool_result` message parts with `is_error=true`. Tool policy denials are recorded separately as `tool.call.blocked`, do not execute the tool, and are also returned to the model as structured error `tool_result` message parts. The session itself should fail for provider errors, runtime contract violations, max-step exhaustion, storage failures, or unrecoverable infrastructure problems.

Framework-native tools receive runtime services through `ToolContext`: workspace, artifact store, runner, vault, and MCP server specs. These references are intentionally runtime-only. They are excluded from `ToolContext.model_dump()` so context metadata can cross storage, event, dashboard, and replay boundaries without serializing live service objects. Serializable service identity fields such as `workspace_id` and `artifact_store_id` may be present when the active environment exposes them.

The first built-in tools are:

- `read_file`: read text from the active workspace by `path`, capture workspace image/PDF files as artifact snapshots when an artifact store is configured, read text artifacts by `artifact_id`, or return provider-neutral image/PDF attachment references for capable providers
- `write_file`: write UTF-8 text to the active workspace, capped by `max_bytes`
- `list_files`: list files in the active workspace, capped by `limit`
- `list_artifacts`: list session- or environment-scoped artifact metadata, capped by `limit`
- `exec_command`: execute an explicit process argv or shell script with the active runner, capped by `timeout_s` and `max_output_bytes`

These tools are ordinary `Tool` implementations. They prove the environment-service contract but do not make file or command access mandatory for all agents.

Default built-in tool caps are intentionally large enough for normal coding work but small enough to protect model context and runtime memory:

- `read_file`: 256 KB by default, 4 MB maximum per call
- `read_file` native file attachments: 8 MB by default, 8 MB maximum per call for the built-in tool instance. Applications may raise or lower that tool-facing cap with `ReadFileTool(default_attachment_limit_bytes=..., max_attachment_limit_bytes=...)`.
- Runtime file attachment resolution: 8 MB maximum per attachment, 32 MB maximum total per provider request, and 20 attachments maximum per provider request by default. Applications may override those runtime caps with `CayuApp(max_file_attachment_bytes=..., max_total_file_attachment_bytes=..., max_file_attachments_per_request=...)`.
- `write_file`: 256 KB by default, 4 MB maximum per call
- `list_files`: 500 paths by default, 10,000 maximum per call
- `list_artifacts`: 500 artifacts by default, 10,000 maximum per call
- `exec_command`: 60 seconds by default, 600 seconds maximum per call; 50,000 bytes stdout and 50,000 bytes stderr by default, 200,000 bytes maximum per stream per call

## Workflow

Coordinates deterministic or agent-assisted multi-step execution.

Workflows need durable step state, retries, pause/resume, failure modes, and event emission.

## Runner

Executes commands/code and returns stdout, stderr, exit code, timeout/cancel flags, and artifacts.

Runner commands use `ExecCommand`:

- `process`: explicit argv list for normal command execution
- `shell`: explicit shell script for bash-like behavior

The framework should not pass a single ambiguous command string to runners. Use process mode unless shell parsing, expansion, and quoting are intentional.
Runner output capture is bounded by `output_limit_bytes` and returns `stdout_truncated` / `stderr_truncated` flags when output is capped. Direct runner calls default to 1 MiB per stream; the model-facing `exec_command` tool passes its smaller 50,000-byte default into the runner. This limit belongs in the runner, not only in tool post-processing, so commands cannot exhaust runtime memory before the model-facing result is built.

Remote runners may talk to a runner service inside EC2/ECS/Daytona/etc.
`LocalRunner` is available for development and trusted local execution. It is not a sandbox. By default it inherits the parent process environment and overlays any explicit `env` values; set `inherit_env=False` when commands should only receive the explicit environment passed to the runner.

`MicrosandboxRunner` is available as an optional microVM-backed runner:

```bash
pip install "cayu[microsandbox]"
```

```python
from cayu import Environment, EnvironmentSpec, MicrosandboxRunner
from microsandbox import Network

async with await MicrosandboxRunner.create(
    "agent-session-123",
    image="python:3.13",
    replace=True,
    network=Network.none(),  # microsandbox SDK object; app-owned policy
) as runner:
    environment = Environment(
        EnvironmentSpec(name="sandboxed"),
        runner=runner,
    )
```

`MicrosandboxRunner.create(...)` passes extra keyword arguments through to
`microsandbox.Sandbox.create(...)`, so applications can configure images,
volumes, network policies, resource limits, labels, patches, and Microsandbox
secret placeholders without Cayu inventing a lossy abstraction over those
backend-specific controls.

Lifecycle is explicit:

- `close_action="remove"`: stop and remove a sandbox created for a session or test.
- `close_action="stop"`: stop the sandbox but leave the persisted record.
- `close_action="detach"`: detach and let the sandbox outlive the Python process.
- `close_action="none"`: attach/use only; no lifecycle action on close.

Use `MicrosandboxRunner.from_existing(...)` when a separate control plane owns
creation and lifecycle.

The runner executes all commands under an absolute guest root, `/workspace` by
default. Per-command `cwd` values must be relative to that root. `env` values
are explicit overlays only; host process environment variables are not inherited.
Vault integrations should resolve only the specific secrets needed at the
execution boundary and pass them through the runner or Microsandbox's own secret
placeholder mechanism. A microVM boundary prevents ordinary workspace escape,
but it does not make broad host mounts, host env inheritance, or unscoped secret
injection safe.

## Workspace

Filesystem boundary. For coding agents this is often a target repo. For document/data agents this may be a working directory where tools create intermediate outputs.
`LocalWorkspace` is available for local filesystem-backed work. It resolves paths under one root and rejects path traversal outside that root.
Workspace reads and listings are bounded at the workspace contract through `max_bytes` and `limit`, returning result objects with `truncated` metadata. Tools should rely on these bounded APIs instead of reading full files or full directory listings and truncating afterward.
The built-in `read_file(path=...)` treats byte-level binary evidence as stronger than filename/MIME hints, so binary bytes are not decoded into model context just because a path has a text-like extension. Text-looking source files remain readable even when platform MIME tables classify an extension incorrectly.

`MicrosandboxWorkspace` exposes a Microsandbox filesystem root through the same
workspace contract:

```python
from cayu import Environment, EnvironmentSpec, MicrosandboxRunner, MicrosandboxWorkspace

runner = await MicrosandboxRunner.create("session-123")
environment = Environment(
    EnvironmentSpec(name="sandbox"),
    runner=runner,
    workspace=MicrosandboxWorkspace(runner, workspace_id="sandbox-workspace"),
)
```

Use `MicrosandboxWorkspace` when file tools must operate inside the same
Microsandbox boundary as `exec_command`. It uses Microsandbox's native
filesystem API, so it does not require Python inside the sandbox image for file
operations. Its `root` defaults to `/workspace`, matching `MicrosandboxRunner`.

`RunnerWorkspace` is the generic fallback for runners that do not have a native
filesystem adapter. It uses small Python helper programs executed through the
runner for read/write/list operations. This keeps the workspace contract
portable across custom runners, but the runner image must provide Python 3, or
`RunnerWorkspace(..., python_executable=...)` must point to an equivalent Python
executable available inside the runner.

Workspace result objects enforce consistent metadata:

- `WorkspaceReadResult`: `truncated` must equal `len(content) < total_bytes`
- `WorkspaceListResult` complete list: `truncated=false` and `total_count == len(paths)`
- `WorkspaceListResult` truncated list: `truncated=true` and `total_count is None or total_count >= len(paths)`

## ArtifactStore

Uploaded/generated file reference boundary. Artifacts are not the active project filesystem. They are durable file blobs with metadata, content type, size, creation time, and explicit scope.

`LocalArtifactStore` is available for local filesystem-backed artifact storage. It stores each artifact as content plus JSON metadata under one root. Session-scoped artifacts require `session_id`; environment-scoped artifacts require `environment_name`. `read_file(artifact_id=...)` enforces that the artifact belongs to the current session or current environment before exposing content to the model.

Configure an artifact store on the environment when the agent should inspect uploaded/generated artifacts or workspace PDFs/images:

```python
Environment(
    EnvironmentSpec(name="local"),
    workspace=LocalWorkspace("./workspace", workspace_id="local"),
    artifact_store=LocalArtifactStore("./.cayu/artifacts", store_id="local-artifacts"),
)
```

Artifact reads and listings are bounded through `max_bytes`, `max_attachment_bytes`, and `limit`. Text artifacts are decoded as UTF-8 with replacement for invalid bytes. Workspace image/PDF path reads are first captured into session-scoped artifact snapshots so the inspected bytes are durable across replay, resume, fork, and provider projection. Image and PDF artifacts return a small model-facing note plus a persisted `cayu.file_attachment.v1` reference in the tool result only after the built-in reader validates that the bytes are parseable. The persisted transcript/event stores the reference, not base64 bytes.

### Workspace/artifact bridge

The artifact store is not the agent's mutable filesystem. The workspace is not the durable upload/output store. Move files between them explicitly:

```python
from cayu import copy_artifact_to_workspace, copy_workspace_file_to_artifact

await copy_artifact_to_workspace(
    artifact_store,
    workspace,
    artifact_id,
    "inputs/invoice.pdf",
)

# Agent tools or app-owned scripts can now work on /workspace/inputs/invoice.pdf.

output = await copy_workspace_file_to_artifact(
    workspace,
    artifact_store,
    "results/invoice-summary.json",
    session_id=session_id,
    agent_name="invoice-agent",
    environment_name="local",
    metadata={"source_artifact_id": artifact_id},
)
```

By default these helpers refuse to write partial copies when a file exceeds `max_bytes`. Increase `max_bytes` for large files, or pass `allow_truncated=True` only when a partial copy is intentional and safe for the application.

Common patterns:

- Coding agent: clone the repo into the workspace, work there, and commit/push through explicit policy. No artifact copy is needed for normal source files.
- Document/invoice agent: keep the original upload as an artifact, copy it into the workspace only when a path-based tool/script must edit or process it, and store the final output as a new artifact.
- Workspace PDF/image inspection: `read_file(path=...)` captures a stable artifact snapshot before provider-native inspection, because the workspace file can change after the tool result is written.

Immediately before a provider request, the runtime scans model-facing tool results for `cayu.file_attachment.v1` references, resolves the referenced bytes from the active `ArtifactStore`, verifies session/environment scope again, and passes a temporary `cayu_file_attachments` map in `ModelRequest.options`. Provider adapters translate that temporary map into native provider content:

- Anthropic: `image` and `document` content blocks inside `tool_result` content.
- OpenAI Responses: the text `function_call_output` plus a following user input item containing `input_image` or `input_file`.

This keeps Cayu's durable transcript provider-neutral and avoids dumping base64 into event/session storage. The built-in `read_file` supports text artifacts, provider-native image attachments, and provider-native PDF attachments. Image/PDF inspection requires the optional file dependencies installed with `cayu[files]`; with them, oversized images can be resized and PDF page ranges can be extracted. Without them, the tool returns a clear error instead of emitting an unvalidated native attachment.

`ReadFileTool` is extensible through artifact readers. Workspace text reads remain built in. Workspace image/PDF path reads are captured as artifacts and then use the same artifact-reader chain. The common extension path is additive: pass `extra_artifact_readers` to run app-owned readers before Cayu's defaults, while keeping the built-in text, image, and PDF readers available as fallbacks.

```python
from cayu import ArtifactReadRequest, ReadFileTool, ToolResult


class InvoiceOcrReader:
    def can_read(self, artifact):
        return artifact.content_type in {"application/pdf", "image/png", "image/jpeg"}

    async def read(self, request: ArtifactReadRequest):
        # App-owned OCR/parser logic here.
        return ToolResult(content="Extracted invoice fields ...")


read_file = ReadFileTool(extra_artifact_readers=[InvoiceOcrReader()])
```

Applications that only need to add one format do not need to reimplement workspace reading, artifact lookup, scope checks, or provider-neutral attachment creation.

Applications that need strict control can pass `artifact_readers=[...]` instead. That replaces the full artifact-reader chain and intentionally disables Cayu's default artifact readers.

Provider-native file upload APIs can be added later as provider-specific optimizations behind the same artifact reference boundary. Remote stores such as S3 can be added as `ArtifactStore` implementations without changing the model-facing tool contract.

Artifact result objects enforce consistent metadata:

- `ArtifactReadResult`: `truncated` must equal `len(content) < total_bytes`
- `ArtifactListResult` complete list: `truncated=false` and `total_count == len(artifacts)`
- `ArtifactListResult` truncated list: `truncated=true` and `total_count >= len(artifacts)`

## Vault

Secrets abstraction. `SecretRef` is the serializable boundary value; raw secret
values are resolved only by trusted application/runtime code and should not be
placed in model prompts, tool schemas, transcripts, durable events, or ordinary
logs.

An `Environment` can attach a vault resolver:

```python
Environment(
    EnvironmentSpec(name="local"),
    vault=LocalEnvVault({"github_token": "GITHUB_TOKEN"}),
)
```

The built-in local implementations are:

- `LocalEnvVault`: maps secret names to environment variables in the trusted app process.
- `StaticVault`: stores in-memory secrets for tests and trusted local development.

`SecretEnv` represents a deliberate environment variable injection:

```python
SecretEnv(name="GITHUB_TOKEN", ref=SecretRef(name="github_token"))
```

Runner/MCP integrations should accept secret refs and resolve them through the
active environment vault at the execution boundary. The model should not receive
a general-purpose secret-reading tool. Application-owned tools can use secrets
internally by resolving refs in trusted code and returning safe results.

MCP config separates plain and secret values:

- `env` / `headers`: non-secret strings
- `secret_env` / `secret_headers`: `SecretRef` values resolved by runtime

The framework should not guess whether a key name is sensitive. Secrets already
present inside a workspace, such as `.env`, `.npmrc`, or cloud credential files,
are a workspace/sandbox policy problem rather than a vault-resolution problem.
Production runners should execute untrusted shell/code in an isolated sandbox
with minimal environment and narrow mounts. `SecretRedactor` is available for
defense-in-depth redaction of known resolved secret values before persistence or
display, but redaction is not the security boundary.

MCP is an interoperability layer, not the required custom tool model. Application-owned
Python tools should use Cayu's native `Tool` contract. External or separately packaged
tool servers can be connected through MCP.

The first MCP implementation supports stdio servers:

- `StdioMcpClient` launches an explicit argv command and speaks newline-delimited
  JSON-RPC over stdin/stdout.
- stderr is treated as server logging, not protocol output.
- The client rejects servers that negotiate an unsupported MCP protocol version.
- Stdio writes are timeout-bounded separately from server response waits, so a
  broken or backpressured MCP subprocess cannot hang Cayu before the request
  timeout starts. A write timeout closes the stdio session because the server may
  already have received part or all of the JSON-RPC message.
- Timed-out or caller-cancelled in-flight requests send MCP
  `notifications/cancelled` when the request has already been written, except for
  `initialize`, which MCP clients must not cancel. The notification write is
  best-effort and timeout-bounded; if it is interrupted, the stdio session is
  closed instead of being reused.
- Session shutdown closes the child process stdin first, waits for graceful exit,
  then escalates to terminate/kill if the server does not exit.
- `connect_mcp_toolset(...)` initializes the server, lists tools, and returns
  one `McpToolset` that owns the live MCP session and its `McpToolAdapter`
  instances.
- Callers must close the toolset when the application or environment shuts down.
  Tool adapters intentionally reuse that initialized session instead of launching
  a fresh MCP process for every tool call.
- The initialize result is available as `McpToolset.initialize_result`, including
  protocol version, server info, capabilities, and server instructions.
- `McpToolAdapter` exposes one MCP tool as a normal Cayu `Tool`, so tool policies,
  approvals, events, transcript persistence, and provider adapters work through the
  same path as framework-native tools.
- Cayu tool names are prefixed with the MCP server namespace, such as
  `mcp__local-mcp__echo`, to make provenance visible and avoid collisions.

This first stdio client does not resolve `secret_env` itself. Secret resolution belongs
at the environment/vault boundary before the subprocess is started. Streamable HTTP MCP,
OAuth, MCP prompts, sampling, elicitation, and automatic resource injection are future
layers. MCP resources should remain explicit and policy-controlled instead of being
dumped into model context automatically.

## KnowledgeStore

Searchable memory/knowledge interface. Default local implementation should eventually support file indexing plus SQLite FTS.
