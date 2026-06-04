# Runtime Contracts

This is a design/maintainer document for the current framework foundation. It names the first contracts that must stabilize before the framework grows higher-level features.

## Boundary Data

Framework boundary data should be portable across local processes, remote runners, hosted runtimes, event stores, dashboards, and replay tools.

Payloads, metadata, tool arguments, tool results, model options, checkpoints, task data, and event data are JSON data. They must contain JSON-compatible values: objects, arrays, strings, integers, finite floats, booleans, and null. Tuples, arbitrary Python objects, non-string object keys, circular references, NaN, and Infinity are not valid boundary data. Task input, result, error, and metadata fields are top-level JSON objects with JSON-compatible nested values.

Runtime APIs copy framework objects at boundaries. User code should not mutate registered specs, request objects, message parts, event payloads, tool results, or provider events and expect those mutations to change already-registered or already-emitted runtime state.

## ContextPolicy

Builds the model-facing message list immediately before each provider request.

The durable transcript is the source record of what happened in the session. A context policy is a projection of that transcript for one model call. It may trim old messages, replace large tool results, inject retrieved context, or implement app-specific conversation routing. It must not destructively rewrite the stored transcript.

`DefaultContextPolicy` returns the current runtime transcript unchanged. Custom policies implement `build(ContextRequest) -> list[Message]`. The runtime passes copied session, agent, message, environment, step, and metadata values into the policy, validates the returned messages, and then sends those messages to the provider. Invalid policy output fails the session before a provider request is made.

Context output must preserve complete tool rounds: assistant tool calls must be followed by matching tool results, and tool results cannot appear without their preceding assistant tool calls. Policies that trim recent history should use `trim_context_turns(...)` for user-turn based history or `trim_context_messages(...)` for message-count based history instead of slicing blindly. Both helpers preserve leading system messages by default.

Built-in policies include `RecentTurnsContextPolicy`, `MessageWindowContextPolicy`, and `CheckpointCompactionContextPolicy`. Recent-turn and message-window policies are pure projections over the current transcript. Checkpoint-backed compaction is runtime-managed: it summarizes older messages through a `ContextCompactor`, stores summary state in the session checkpoint under `context_compaction`, emits `context.compaction.started`, `context.compaction.completed` or `context.compaction.failed`, emits `session.checkpointed` after successful checkpoint writes, and sends leading system messages, compacted user-context summary, and recent complete turns to the provider. It does not delete or rewrite transcript messages.

Compaction checkpoints store the summary and `compacted_transcript_cursor`, the provider-neutral transcript position covered by that summary. The model-facing summary is injected as synthetic user context, not as a system instruction, and is not appended to the durable transcript. Compaction events include cursor, compactor, count, error, and provider metadata needed for audit/debugging, but they do not include the summary text.

`TranscriptDigestCompactor` is the deterministic fallback. It converts older messages into a clipped text digest and does not perform semantic summarization. `ModelCompactor` is the provider-backed implementation for production semantic summaries: it sends a text-only compaction request with no tools to a configured `ModelProvider`, rejects tool calls from the compaction model, and stores the returned text as the checkpoint summary. Model compaction bounds the serialized compaction input with `max_input_chars` by default so very large transcripts cannot create unbounded provider requests; the default prompt preserves compaction instructions and existing summary while clipping only the newly compacted transcript digest. Callers can tune or disable that bound explicitly. Callers can provide `system_prompt` to change compaction-model behavior and `prompt_builder` to replace the user prompt body.

## Agent, Environment, Session

Cayu separates agent definition, execution environment, and session state:

- `AgentSpec`: model, system prompt, tool declarations, and metadata.
- `Environment`: workspace, runner, vault, MCP servers, and execution metadata.
- `RunRequest` / `ResumeRequest` / `Session`: one run of an agent, optionally in a named environment, with messages, status, events, and checkpoints.

This mirrors the useful Managed Agents separation of brain, hands, and durable run history without copying any one provider API. A run may omit an environment for simple provider/tool tests, but concrete file, command, sandbox, vault, or MCP-backed tools should hang off an environment.

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

`ResumeRequest` explicitly continues an existing session. It loads the stored provider-neutral transcript, appends the new request messages to that same transcript, emits `session.resumed`, and runs the same model/tool loop as a new session. Resume uses the session's stored agent, provider, model, runtime, and environment identity instead of current application defaults. This prevents an application default change from silently continuing an old session on a different provider or model. `ResumeRequest.model` may update the session's active model within the stored provider; that update is durable and future resumes use the new model. Provider switching is intentionally not part of resume yet. A session can be resumed only from `completed`, `failed`, or `interrupted`. `pending` and `running` sessions are rejected so concurrent workers do not continue the same session at the same time. Interrupted sessions with a pending tool approval must be continued through `resolve_tool_approval(...)`, not `resume(...)`. Session stores expose an atomic status transition for this boundary.

`ForkSessionRequest` creates a new session branch from an existing `completed`, `failed`, or `interrupted` session without mutating the source. Fork keeps the source provider fixed, copies the source transcript, optionally applies a same-provider model override, and persists `parent_session_id` on the child session. Full-session forks copy checkpoint state by default and emit `session.forked` on the child session. Partial transcript forks use `transcript_cursor` to copy messages through a 1-based provider-neutral transcript cursor and must set `copy_checkpoint=False`; checkpoint state is not safe to copy when it may refer to transcript messages omitted from the fork. Interrupted sessions cannot be forked without checkpoint state. When a pending tool approval checkpoint is forked, Cayu copies the approval state but clears the source task id so resolving the fork cannot update a task owned by the source session.

`DispatchRequest` asks a `Dispatcher` to submit work for an existing session and return a `DispatchHandle`. Dispatch is separate from fork: fork decides what state a branch starts from, while dispatch decides how a session run is placed. The default `InlineDispatcher` runs immediately in the current process by resuming the target session through the normal runtime loop, then returns a completed, failed, or interrupted handle based on the terminal session event. It is useful for tests, local execution, and proving orchestration logic, but it is not durable background execution. Production apps can provide another `Dispatcher` that submits work to an external queue or hosted runtime and returns a queued/submitted handle while events are observed through the session store. `CayuApp.dispatch_inline(...)` is the explicit local streaming API for callers that want to consume ordinary `session.resumed`, model, tool, task, interrupt, and terminal session events directly. `DispatchRequest.task_id` optionally links dispatched work to a task; using it with inline execution requires `CayuApp(task_store=...)`.

`RuntimeHook` provides lifecycle automation around durable session state. The first supported phases are `after_session_completed`, `after_session_failed`, and `after_session_interrupted`. These hooks run only after the terminal session status and terminal event have already been persisted. A hook failure does not rewrite the terminal session status; Cayu records `hook.failed` and continues to later hooks. Successful hooks emit `hook.started` and `hook.completed`, including the hook scope, terminal event id/type, and JSON-safe action summaries. `RuntimeHookContext` exposes copied session/event data plus controlled helpers for `fork_session`, `create_task`, `dispatch`, `dispatch_inline`, and custom event emission. Hook helper side effects are persisted and sent to event sinks; the parent run stream yields the hook telemetry events. Hook-emitted custom events must use the `custom.` namespace. `CayuApp(runtime_hooks=[...])` registers app-level hooks for global middleware, while `register_agent(..., runtime_hooks=[...])` registers hooks that run only for that agent. App-level hooks run before agent-level hooks. Hooks that fork and dispatch follow-up sessions should still guard on session metadata, task type, or another app-owned marker when they can process their own follow-up sessions. This lets apps implement follow-up work such as “fork this completed builder session and dispatch a knowledge-extraction task” without making fork, task, or dispatch mean the same thing.

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

`CayuApp(task_store=...)` can link an agent run to an existing task through `RunRequest.task_id`. The runtime starts that task with the created session id, emits `task.started`, and then marks the task completed or failed before emitting the terminal session event. This is a task/session bridge, not a queue worker, retry engine, workflow engine, or agent communication table.

## EventSink

Receives events and forwards them somewhere:

- stdout
- dashboard websocket
- webhook
- log file
- database
- hosted platform adapter

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

`OpenAIProvider` adapts the OpenAI Responses API to the same Cayu transcript. It keeps Cayu `system` messages as OpenAI `instructions`, maps assistant tool calls to Responses `function_call` items, maps Cayu tool-result messages to `function_call_output` items, and sets `store: false` by default so Cayu remains the durable session source of truth. Callers can override OpenAI request options through `ModelRequest.options["openai"]` except for fields owned by the provider contract.

The first provider implementations use complete API responses and yield normalized Cayu stream events from the returned model response. Server-sent-event streaming can be added behind the same provider contract later.

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

Framework-native tools receive runtime services through `ToolContext`: workspace, runner, vault, and MCP server specs. These references are intentionally runtime-only. They are excluded from `ToolContext.model_dump()` so context metadata can cross storage, event, dashboard, and replay boundaries without serializing live service objects.

The first built-in tools are:

- `read_file`: read UTF-8 text from the active workspace, capped by `max_bytes`
- `write_file`: write UTF-8 text to the active workspace, capped by `max_bytes`
- `list_files`: list files in the active workspace, capped by `limit`
- `exec_command`: execute an explicit process argv or shell script with the active runner, capped by `timeout_s` and `max_output_bytes`

These tools are ordinary `Tool` implementations. They prove the environment-service contract but do not make file or command access mandatory for all agents.

Default built-in tool caps are intentionally large enough for normal coding work but small enough to protect model context and runtime memory:

- `read_file`: 256 KB by default, 4 MB maximum per call
- `write_file`: 256 KB by default, 4 MB maximum per call
- `list_files`: 500 paths by default, 10,000 maximum per call
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

## Workspace

Filesystem/artifact boundary. For coding agents this is often a target repo. For document/data agents this may be uploaded files and generated outputs.
`LocalWorkspace` is available for local filesystem-backed work. It resolves paths under one root and rejects path traversal outside that root.
Workspace reads and listings are bounded at the workspace contract through `max_bytes` and `limit`, returning result objects with `truncated` metadata. Tools should rely on these bounded APIs instead of reading full files or full directory listings and truncating afterward.

Workspace result objects enforce consistent metadata:

- `WorkspaceReadResult`: `truncated` must equal `len(content) < total_bytes`
- `WorkspaceListResult` complete list: `truncated=false` and `total_count == len(paths)`
- `WorkspaceListResult` truncated list: `truncated=true` and `total_count is None or total_count >= len(paths)`

## Vault

Secrets abstraction. Raw secret values should be injected into tools/runners by runtime and should not be placed in model prompts.

MCP config separates plain and secret values:

- `env` / `headers`: non-secret strings
- `secret_env` / `secret_headers`: `SecretRef` values resolved by runtime

The framework should not guess whether a key name is sensitive.

## KnowledgeStore

Searchable memory/knowledge interface. Default local implementation should eventually support file indexing plus SQLite FTS.
