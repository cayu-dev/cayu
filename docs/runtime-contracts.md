# Runtime Contracts

This is a design/maintainer document for Cayu's runtime framework. It names contracts that must stay stable as the framework grows higher-level features.

## Boundary Data

Framework boundary data should be portable across local processes, remote runners, hosted runtimes, event stores, dashboards, and replay tools.

Payloads, metadata, tool arguments, tool results, model options, checkpoints, task data, and event data are JSON data. They must contain JSON-compatible values: objects, arrays, strings, integers, finite floats, booleans, and null. Tuples, arbitrary Python objects, non-string object keys, circular references, NaN, and Infinity are not valid boundary data. Task input, result, error, and metadata fields are top-level JSON objects with JSON-compatible nested values.

Runtime APIs copy framework objects at boundaries. User code should not mutate registered specs, request objects, message parts, event payloads, tool results, or provider events and expect those mutations to change already-registered or already-emitted runtime state.

## ContextPolicy

Builds the model-facing message list immediately before each provider request.

The durable transcript is the source record of what happened in the session. A context policy is a projection of that transcript for one model call. It may trim old messages, replace large tool results, inject retrieved context, or implement app-specific conversation routing. It must not destructively rewrite the stored transcript.

`DefaultContextPolicy` returns the current runtime transcript as model-facing context while stripping old native file attachment references by default. Custom policies implement `build(ContextRequest) -> list[Message]`. The runtime passes copied session, agent, message, environment, step, metadata, and `context_usage` values into the policy, validates the returned messages, and then sends those messages to the provider. Invalid policy output fails the session before a provider request is made.

`ContextRequest.context_usage` is derived from durable `model.completed` events already stored for the session. It exposes actual normalized usage from the previous completed model call, including input/output/total token counts and provider/model identity when available. It is intentionally post-call state: a policy can use it to compact, roll, truncate, or route the next request after prior actual usage crossed an application threshold. When the latest completed event includes a transcript cursor, `context_usage.input_pressure` adds a local transcript-pressure estimate: previous actual input tokens plus an estimate for transcript messages appended after that cursor. This is `method="observed_plus_estimated_delta"` and `confidence="estimated"`; it is useful for policy inspection but is not provider-authoritative token counting, billing data, or an exact final-provider-request context-window guarantee.

`UsageTriggeredContextPolicy` packages actual and estimated pressure triggers. Below threshold it delegates to a base policy. Once the previous call's input/total tokens meet an exact threshold, or once `trigger_estimated_context_tokens` is met, later context builds delegate to a triggered policy such as a smaller `RecentTurnsContextPolicy` or `MessageWindowContextPolicy`. The estimated trigger is computed after the base policy has produced model-facing context, so it includes knowledge injected by that base policy, trimming performed by that base policy, known tool schemas, structured-output tool/schema wiring, provider-visible request options, tool-call arguments, tool-result text, thinking/provider-state parts, and conservative file-attachment size estimates from attachment references. Provider adapters expose `ModelContextPressureProfile` hints for local calibration of image attachment floors, document/PDF attachment floors, document byte density, and tool-schema payload density; runtime estimators consume those hints without branching on provider names. Tool-result structured data and artifact reference metadata are not counted as prompt text unless a provider-facing adapter actually sends them. The trigger compares `estimated_context_window_tokens`, which is `estimated_context_input_tokens + reserved_output_tokens`, so applications can reserve generation/reasoning headroom before the hard provider context limit. This estimate uses `method="observed_plus_estimated_delta_with_overhead"` when it can anchor on previous actual provider input usage and `method="local_full_request_estimate"` when it must estimate the whole model-facing request locally. Anchored estimates do not add stable request overhead twice: `model.completed` records component-only overhead counts, and the next estimate adds only the current overhead delta if tools, structured output, or provider-visible options changed. Both remain local and conservative; providers can still count differently. The trigger is sticky by default and stored under the `usage_triggered_context` session checkpoint key so a lower-usage compact/windowed call does not immediately return the session to the base policy. Set `sticky=False` only for explicit last-call-only routing.

Estimated triggers speculatively evaluate `base_policy` before deciding whether to switch to `triggered_policy`. For that reason, `base_policy` must be side-effect-free in estimated-trigger mode. Runtime-managed base policies are rejected because they can perform compaction, emit checkpoint payloads, or call a compactor model during context construction. Put compaction or other managed behavior in `triggered_policy`, or use exact post-call thresholds such as `min_input_tokens` when the base policy must be runtime-managed.

Estimated triggers can optionally verify with the active provider's `count_input_tokens(...)` implementation by setting `verify_estimate_with_provider_count=True`. This remains explicit because a remote provider counter can add latency, rate-limit pressure, and provider-specific billing uncertainty. Verification runs only when the local estimate is near the threshold according to `provider_count_threshold_ratio`, or when `provider_count_min_delta_tokens` marks the newly estimated delta as large. If the provider count succeeds, the decision uses `provider_count_input_tokens + reserved_output_tokens`; if the provider does not support counting or the count fails, the policy falls back to the local estimate.

Context output must preserve complete tool rounds: assistant tool calls must be followed by matching tool results, and tool results cannot appear without their preceding assistant tool calls. Policies that trim recent history should use `trim_context_turns(...)` for user-turn based history or `trim_context_messages(...)` for message-count based history instead of slicing blindly. Both helpers preserve leading system messages by default.

Built-in policies include `RecentTurnsContextPolicy`, `MessageWindowContextPolicy`, `UsageTriggeredContextPolicy`, `CheckpointCompactionContextPolicy`, and `KnowledgeInjectionPolicy`. Recent-turn and message-window policies are pure projections over the current transcript. Built-in policies keep only the latest file-attachment tool result provider-resolvable by default; older attachment references are replaced with text/structured summaries using `strip_old_file_attachments(...)` so providers do not receive the same file bytes on every later request. Checkpoint-backed compaction is runtime-managed: it summarizes older messages through a `ContextCompactor`, stores summary state in the session checkpoint under `context_compaction`, emits `context.compaction.started`, `context.compaction.completed` or `context.compaction.failed`, emits `session.checkpointed` after successful checkpoint writes, and sends leading system messages, compacted user-context summary, and recent complete turns to the provider. It does not delete or rewrite transcript messages.

Compaction checkpoints store the summary and `compacted_transcript_cursor`, the provider-neutral transcript position covered by that summary. The model-facing summary is injected as synthetic user context, not as a system instruction, and is not appended to the durable transcript. Compaction events include cursor, compactor, count, error, and provider metadata needed for audit/debugging, but they do not include the summary text.

`TranscriptDigestCompactor` is the deterministic fallback. It converts older messages into a clipped text digest and does not perform semantic summarization. `ModelCompactor` is the provider-backed implementation for production semantic summaries: it sends a text-only compaction request with no tools to a configured `ModelProvider`, rejects tool calls from the compaction model, and stores the returned text as the checkpoint summary. Model compaction bounds the serialized compaction input with `max_input_chars` by default so very large transcripts cannot create unbounded provider requests; the default prompt preserves compaction instructions and existing summary while clipping only the newly compacted transcript digest. Callers can tune or disable that bound explicitly. Callers can provide `system_prompt` to change compaction-model behavior and `prompt_builder` to replace the user prompt body.

## Context Counting

Context counting observes the final provider request after context policy output, knowledge injection, structured-output tool wiring, and file attachment resolution. It is disabled by default. `CayuApp(context_counting=ContextCountingConfig(mode="observe"))` asks the active `ModelProvider` to run `count_input_tokens(ModelRequest)` before each provider attempt.

`count_input_tokens(...)` is optional provider behavior. Providers that support an official remote counter should return `InputTokenCountResult(method="official", confidence="high")`. Built-in OpenAI Responses and Anthropic Messages providers call their official input-token count endpoints. OpenAI-compatible Chat Completions providers report unavailable by default because there is no single portable count endpoint across those vendors. Local tokenizers and heuristics may return lower-confidence results for observability, but they are not hard provider-limit guarantees. The runtime passes a copied `ModelRequest` into the counter so provider counting code cannot mutate the actual request sent to `stream(...)`. Providers that do not implement counting return `None`; the runtime records that as unavailable. If a counter raises, Cayu emits `context.count.failed` and still performs the model call because observe mode is not an enforcement boundary.

Official remote counters are not the default context-overflow mechanism. They submit the request payload to the provider before the model call, so applications must treat them as extra provider API calls. Anthropic documents Messages token counting as free with separate RPM limits. OpenAI documents Responses input-token counting, but does not document whether that counting request is free or billed. Cayu records that provider-specific status in the count result metadata and does not fold count requests into generation usage or cost totals. Remote counters are appropriate for debugging, calibration, live verification, sampled observability, or explicit near-limit confirmation where the caller accepts the extra provider request. Production context policy should first use local budgeting, compaction, rolling windows, truncation, or fail-fast thresholds, then handle provider context-limit errors as a recovery path.

In observe mode, every model attempt emits `context.pressure.estimated` before `model.started` with a local final-request estimate and no prompt text, tool schemas, or attachment contents. When the model later emits `model.completed` with normalized `usage_metrics.input_tokens`, Cayu emits `context.pressure.reconciled` with the same observation id, the pre-call local estimate, actual input tokens, delta, and relative error. The reconciliation compares actual input only against `estimated_context_input_tokens`; any `reserved_output_tokens` remain a separate context-window headroom decision.

Successful provider-count observation emits `context.counted` before `model.started`. The event payload includes model/provider identity, step/attempt, message count and roles, tool count, provider-option keys, an opaque observation id, and the provider's count result. It does not include prompt text, tool schemas, attachment contents, or a deterministic request-derived fingerprint. When the model later emits `model.completed` with normalized `usage_metrics.input_tokens`, Cayu emits `context.count.reconciled` with the same observation id, pre-call count, actual input tokens, delta, and relative error. Reconciliation is telemetry only; it does not rewrite budgets, transcripts, checkpoints, or context.

This contract is the first slice for future context-budget enforcement. Later reducers or hard budget policies should consume the same final request boundary and count result, but enforcement must be explicit. Official remote counters should remain opt-in for cases where exact pre-call accounting is worth the extra provider request; they should not be required for the default budget path.

## Context Overflow Classification

Providers should raise `ModelContextOverflowError` when a request clearly exceeds a model context window or documented request-size boundary. Built-in adapters expose provider-specific subclasses (`OpenAIContextOverflowError`, `AnthropicContextOverflowError`, and `ChatCompletionsContextOverflowError`) that also inherit from their provider API error class. Runtime code can catch `ModelContextOverflowError`; provider-specific code can catch the corresponding API error. Built-in adapters classify only conservative signals: OpenAI `context_length_exceeded` responses or equivalent context-length messages, Anthropic `413 request_too_large` plus explicit prompt/context-too-long invalid requests, and OpenAI-compatible Chat Completions responses with explicit context-too-long messages such as Gemini's "input context is too long." Generic quota/rate errors such as Gemini `RESOURCE_EXHAUSTED` are not context overflow.

Context overflow recovery is opt-in per agent through `register_agent(..., context_overflow_policy=...)`. When the initial provider request for a model step raises `ModelContextOverflowError`, Cayu emits `context.overflow.detected`, rebuilds model-facing messages with the configured overflow policy, emits `context.overflow.recovering`, and runs the rebuilt request through the normal model-step retry policy. Cayu performs at most one overflow rebuild for a model step; if the rebuilt request also raises `ModelContextOverflowError`, Cayu emits `context.overflow.failed` and fails the session. The durable transcript is not deleted or rewritten by overflow recovery. A checkpoint-backed overflow policy may write a compaction checkpoint; projection-only policies such as recent-turn or message-window policies only change the retry request sent to the provider.

## Agent, Environment, Session

Cayu separates agent definition, execution environment, and session state:

- `AgentSpec`: model, system prompt, tool declarations, and metadata.
- `Environment`: workspace, artifact store, runner, vault, credential proxy, MCP servers, and execution metadata.
- `RunRequest` / `ResumeRequest` / `Session`: one run of an agent, optionally in a named environment, with messages, status, events, and checkpoints.

This mirrors the useful Managed Agents separation of brain, hands, and durable run history without copying any one provider API. A run may omit an environment for simple provider/tool tests, but concrete file, command, sandbox, vault, or MCP-backed tools should hang off an environment.

Environment-scoped `workspace_instructions` are optional operating guidance for the active workspace. They are rendered into the initial system/instruction message for a new session as a labeled workspace section, not as a user message. Apps can pass explicit instruction text or a `WorkspaceInstructionsConfig` that loads bounded UTF-8 files from the workspace root, such as `AGENTS.md`, `.cayu/AGENTS.md`, or app-configured paths like `CLAUDE.md`. Missing files are ignored; absolute paths and parent-directory escapes are rejected. Workspace instructions are stored in the session's initial transcript system message for reproducible resume/fork behavior.

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

`AllowAllToolPolicy` is the default so existing simple agents continue to run without extra configuration. `StaticToolPolicy` provides a small allow/deny scope for common cases. Deny rules win over allow rules. `ParameterConstrainedToolPolicy` validates selected tool arguments with per-tool rules before the tool implementation runs. Built-in rules include required fields, string allowlists, and denied regex patterns over dotted JSON argument paths such as `request.url`. Violations return either `DENY` or `REQUIRE_APPROVAL`; they never silently rewrite tool arguments. `TaintAwareToolPolicy` protects sensitive tools after configured untrusted source tools have produced output in the same session. It is origin-based, not a prompt-injection scanner: apps label source tools such as `read_email`, `fetch_url`, or `read_pdf` with taint labels, then protect outbound tools such as `send_email`, `make_payment`, or `execute_sql` from those labels. Cayu derives prior taint from durable terminal tool events and also applies taint within one model tool-call round before any tool implementation runs. Custom policies implement `authorize(ToolPolicyRequest) -> ToolPolicyResult`.

Denied tool calls are recoverable by default. The runtime emits `tool.call.started`, then `tool.call.blocked`, does not run the tool implementation, appends an error `tool_result` to the provider-neutral transcript, and lets the model continue. Tool policy implementation errors are not recoverable tool failures; they fail the session because the authority layer itself is broken.

Policies may also return `ToolPolicyDecision.REQUIRE_APPROVAL`. This is a durable interrupt, not an in-memory UI callback. The runtime authorizes the model's whole tool-call round before execution; if any call requires approval, it persists a `pending_tool_approval` checkpoint for the round, emits `session.checkpointed`, emits `tool.call.approval_requested`, marks the session `interrupted`, and emits `session.interrupted` with `interruption_type="tool_approval_required"`. No tool implementation in that round runs before approval.

Callers resolve pending approvals with `CayuApp.resolve_tool_approval(ToolApprovalRequest(...))`. Approval emits `session.resumed`, emits `tool.call.approved` for approval-gated calls, runs executable calls in the stored round, appends the grouped tool-result message, clears the pending checkpoint, and continues the model loop. Denial emits `session.resumed`, does not execute the interrupted round, appends error results for the round, clears the pending checkpoint, and continues the model loop. The grouped tool-result message and cleared checkpoint are persisted through one atomic `SessionStore` update; if that update fails, the session returns to `interrupted` with the pending approval still present so the approval can be retried instead of creating an invalid provider history. Approval retry uses stored terminal tool events for the same approval as the execution ledger: completed, failed, blocked, or approval-denied tool results are reused and never re-executed. If a tool has `tool.call.started` for that approval without a terminal event, Cayu leaves the session `interrupted` with `manual_recovery_required` instead of re-running a side-effecting tool whose outcome is unknown. The caller can then use `CayuApp.recover_tool_approval(ToolApprovalRecoveryRequest(...))` to mark the externally verified outcome as `completed` or `failed` and provide the exact message the model should see. Recovery first claims the interrupted session, then persists the caller-supplied message as the terminal tool result, reuses it as the approval outcome, clears the checkpoint, and continues the model loop without executing the tool again. Cayu does not infer domain facts for recovery. Normal `ResumeRequest` rejects sessions with pending tool approvals because provider histories require the assistant tool-call round to be followed by matching tool results before any later conversation.

Ordinary tool rounds are also crash-recoverable. When a model step produces normal executable tool calls, Cayu atomically appends the assistant tool-call message and a private `pending_tool_round` checkpoint before executing tools. Tool events for that round include a `tool_round_id`; terminal `tool.call.completed`, `tool.call.failed`, and `tool.call.blocked` events with the matching `tool_round_id` and `tool_call_id` are the execution ledger. When the grouped tool-result transcript message is appended, Cayu clears `pending_tool_round` in the same atomic `append_transcript_messages_and_checkpoint(...)` update. If the process dies before that close, the next run or resume repairs the transcript before adding new messages or building provider context: recorded terminal outcomes are converted into matching `tool_result` parts; tool calls with no recorded start event become explicit not-executed error results; tool calls that started but have no terminal event become explicit unknown-outcome error results. Cayu never re-executes an ordinary tool call during crash recovery.

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

An app can register either a concrete `Environment` or an `EnvironmentFactory` under an `EnvironmentSpec` name. A factory receives durable session context (`session_id`, `agent_name`, `environment_name`, parent session id, causal budget id, labels, metadata, and previous reconnect metadata for that session/environment) and returns a concrete `Environment` for that session. The returned environment must keep the registered environment name so resume/fork/dispatch do not silently switch identity. Factory-backed environments are resolved before workspace binding, MCP setup, tool execution, and, for new sessions, workspace-instruction loading. The runtime emits `environment.factory.started`, `environment.factory.completed`, and `environment.factory.failed` with JSON-safe diagnostics and never serializes live workspace, runner, or vault objects. `EnvironmentFactoryResult.reconnect_metadata` is non-secret durable state such as a sandbox id, region, image, or attach handle; Cayu stores it in the session checkpoint under the registered environment name before emitting `environment.factory.completed`, and passes it back on later resume, approval continuation, or recovery. Forks that copy checkpoint state also copy reconnect metadata, so factories that need child-session isolation should use `parent_session_id` to ignore the inherited metadata and allocate a fresh environment. Factory failure fails a new session before `session.started`; for pending approval continuation or manual approval recovery, factory failure is emitted before `session.resumed` and the session returns to `interrupted` with the approval still recoverable. Factory-created resources that need runtime cleanup should be represented through the returned environment's `WorkspaceBinding`, runner, or another app-owned lifecycle handle; the factory contract itself is create/attach only and is not a hidden cleanup hook. Static environments still validate workspace instructions before creating a session, preserving the existing early-failure behavior for ordinary local runs.

When an environment has a `WorkspaceBinding`, the runtime emits durable binding lifecycle events around the session execution boundary. `environment.binding.started` and `environment.binding.completed` are emitted before `session.started` or before approved tool-continuation work resumes. `environment.binding.failed` is emitted before the terminal failure when binding cannot be created. Terminal session events are preceded by `environment.binding.finalize_started` and either `environment.binding.finalize_completed` or `environment.binding.finalize_failed`. Finalize failures are also copied onto the terminal event payload as `binding_finalize_error` so session outcome and cleanup outcome can be inspected together. Binding event payloads contain JSON-safe identity and diagnostic fields such as binding type, configured workspace id, source workspace id, bound workspace id, bound path, outcome, and error details; they never serialize live workspace or runner objects. `BoundWorkspace.source_workspace` is the durable workspace originally provided by the environment. `BoundWorkspace.workspace` is the workspace visible to tools during the run; for native bindings they can be the same object, while copy/mount bindings may expose a different sandbox workspace.

`SyncBinding` is the built-in copy binding. It copies files from the durable source workspace into a target workspace during `bind(...)`, exposes the target workspace to tools, and can copy creates/updates/deletes back to the source during `finalize(...)`. The target workspace is usually a session-scoped sandbox workspace such as `E2BWorkspace`, `MicrosandboxWorkspace`, or `RunnerWorkspace`; it can also be a local scratch workspace for tests and local development. A `target_workspace_factory` receives `SyncBindingContext` with source workspace, runner, session id, agent name, environment name, and request metadata so apps can create or attach a per-session target. Source and target must be different workspaces. By default, `SyncBinding` cleans the target workspace before copying in, syncs back for every terminal outcome, and propagates deletions. Apps can configure `pattern`, `max_files`, `max_file_bytes`, `clean_target`, `sync_back`, and `delete_missing` to match their lifecycle policy. The target workspace should be dedicated to that binding/session when `clean_target="always"` is used. When `clean_target="never"` is used, Cayu records the target baseline at bind time and does not sync pre-existing target-only files back to the source; it only syncs source-origin files plus files created during the run. Built-in `SyncBinding` keeps baseline path state in process and keeps durable binding event metadata to counts/policies; custom bindings should persist their own non-secret sync token when finalization must survive process restart. `SyncBinding` owns `bound.metadata["sync_binding"]` and final snapshot metadata keys `target_workspace_id`, `outcome`, `copied_files`, `copied_bytes`, and `deleted_files`; app-provided binding metadata must not use those keys. `max_file_bytes=None` means `SyncBinding` does not add its own per-file cap, but the workspace being read may still enforce its default read limit. Configure source and target workspace read limits, or `max_file_bytes`, explicitly for large files.

`GitRepositoryBinding` is the built-in Git checkout binding. It requires a `LocalWorkspace` or a `RunnerWorkspace` because cloning and fetching require running the `git` executable. For E2B, Microsandbox, Docker, and SBX command runners, wrap the runner with `RunnerWorkspace` when the repo should be cloned inside the sandbox. On bind, Cayu clones into an empty workspace or fetches an existing Git work tree, verifies the configured remote URL by default, refuses dirty existing repositories by default, checks out the requested ref, fast-forwards to the fetched remote branch when possible, and records repo/ref/branch/commit/dirty metadata in `bound.metadata["git_repository"]` and the bound snapshot. It does not merge, rebase, or rewrite divergent clean branches; Git failures in those cases are surfaced so the app or agent can decide how to proceed. On finalize, Cayu records the final commit and dirty state in a final Git `WorkspaceSnapshot`. The binding never commits, pushes, creates branches, or creates pull requests; those are explicit agent/tool or trusted app workflows. Because `repo_url` is durable metadata, HTTP(S) URLs with embedded credentials are rejected; private repository access should use trusted app setup, SSH agent configuration, or a credential helper outside the persisted URL. For untrusted sandbox runners, do not expose long-lived Git credentials through generic shell access; use public repositories, trusted host-side credentials, or a dedicated brokered Git tool. App-provided binding metadata must not use the reserved `git_repository` key.

`Workspace.delete(path)` is part of the workspace contract. It deletes a file if it exists and is used by bindings to propagate removed files. Built-in local, runner-backed, E2B, and Microsandbox workspaces implement it. Workspaces should keep delete path validation as strict as read/write path validation: relative paths only, no root escape, and no deleting directories through the file delete API.

`BoundWorkspace.snapshot` and a `WorkspaceBinding.finalize(...)` return value can carry `WorkspaceSnapshot` records. A snapshot is non-secret durable identity for a concrete workspace version: for example a git commit, S3 object generation, EFS snapshot id, sandbox filesystem generation, or app-owned sync token. `environment.binding.completed` includes `bound_snapshot` when the binding knows the version made visible to the runner. `environment.binding.finalize_completed` includes `final_snapshot` when finalization syncs or persists a new version after the session. Snapshots do not implement storage, copying, mounting, or cleanup by themselves; those behaviors remain in the binding/workspace/runner implementation. Apps should treat snapshot metadata as observability and replay/debug context, not as a place for secrets.

`ResumeRequest` explicitly continues an existing session. It loads the stored provider-neutral transcript, appends the new request messages to that same transcript, emits `session.resumed`, and runs the same model/tool loop as a new session. Resume uses the session's stored agent, provider, model, runtime, and environment identity instead of current application defaults. This prevents an application default change from silently continuing an old session on a different provider or model. `ResumeRequest.model` may update the session's active model within the stored provider; that update is durable and future resumes use the new model. Provider switching is intentionally not part of resume yet. A session can be resumed only from `completed`, `failed`, or `interrupted`. `pending`, `running`, and `interrupting` sessions are rejected so concurrent workers do not continue the same session at the same time. Interrupted sessions with a pending tool approval must be continued through `resolve_tool_approval(...)`, not `resume(...)`. Session stores expose an atomic status transition for this boundary.

`InterruptSessionRequest` moves a `pending` or `running` session to `interrupting`, signals registered active work for that session in the current `CayuApp` process, and only then finalizes the session as `interrupted` with a durable `session.interrupted` event. `interrupting` is not resumable; it means provider/tool/runner work is still being stopped or the final transcript repair has not completed. `interrupted` is resumable. If the current process does not own the active run, or the active run is still stopping and repairing its transcript, the interrupt request reports that interruption is still finalizing and leaves the session `interrupting`; callers should retry, poll the session/event store, or subscribe until `session.interrupted` appears. Runtime loops check session status before model requests, after provider responses, before tool-policy planning, and between tool calls. The durable session status remains the cross-worker source of truth. Every `session.interrupted` event includes a normalized `interruption_type`: `operator_requested` for explicit operator/API interruption, `tool_approval_required` for approval pauses, and `runtime_interrupted` for runtime/status-driven interruption repair. `CayuApp.recover_incomplete_session(...)` and `CayuApp.recover_incomplete_sessions(...)` are worker-startup recovery helpers, not model continuation APIs. They do not call the model and do not execute tools. They repair pending ordinary tool rounds from durable terminal tool events when possible, preserve pending tool approvals for `resolve_tool_approval(...)`, finalize stale `interrupting` sessions, and mark explicitly selected abandoned `pending`/`running` sessions as `interrupted` so a user or app can resume deliberately later. Batch recovery requires explicit `statuses`; supported values are `interrupting`, `running`, and `pending`. Apps should include `running` or `pending` only when an external deployment/worker boundary proves those sessions are abandoned. Runner adapters are responsible for cleaning up underlying subprocesses or remote execution when their `exec(...)` coroutine is cancelled. `E2BRunner` and `MicrosandboxRunner` default to `cancellation_cleanup="command"` for user/runtime interruption and `timeout_cleanup="command"` for command timeout, so interactive coding sandboxes keep their workspace after an operator interrupt or ordinary command timeout. Cleanup waits are bounded by `cancel_timeout_s`. If E2B has not yet returned a command handle when interruption or timeout arrives, Cayu first tries to stop the start attempt and resolve the handle within the cleanup window. With `"command"` cleanup, Cayu reports deferred cleanup and keeps waiting in the background for a bounded adapter-owned window. If that delayed command start never resolves, Cayu preserves the sandbox but closes that runner's exec path so later commands do not overlap with an unknown command state. With `"sandbox"` cleanup, Cayu kills the sandbox immediately because the sandbox is the configured cleanup boundary. Cayu preserves `cayu.runner_cleanup.v1` diagnostics on synthetic interrupted tool results or timed-out `ExecResult.artifacts`. Apps that prefer a stronger cleanup boundary can set `cancellation_cleanup="sandbox"` or `timeout_cleanup="sandbox"` explicitly. Interrupting a session does not automatically cancel a linked task; task state remains application/workflow owned. The optional server exposes this as `POST /api/sessions/{session_id}/interrupt`.

`ForkSessionRequest` creates a new session branch from an existing `completed`, `failed`, or `interrupted` session without mutating the source. Fork keeps the source provider fixed, copies the source transcript, optionally applies a same-provider model override, and persists `parent_session_id` on the child session. Full-session forks copy checkpoint state by default and emit `session.forked` on the child session. Partial transcript forks use `transcript_cursor` to copy messages through a 1-based provider-neutral transcript cursor and must set `copy_checkpoint=False`; checkpoint state is not safe to copy when it may refer to transcript messages omitted from the fork. Interrupted sessions cannot be forked without checkpoint state. When a pending tool approval checkpoint is forked, Cayu copies the approval state but clears the source task id so resolving the fork cannot update a task owned by the source session.

`DispatchRequest` asks a `Dispatcher` to submit work for an existing session and return a `DispatchHandle`. Dispatch is separate from fork: fork decides what state a branch starts from, while dispatch decides how a session run is placed. The default `InlineDispatcher` runs immediately in the current process by resuming the target session through the normal runtime loop, then returns a completed, failed, or interrupted handle based on the terminal session event. It is useful for tests, local execution, and proving orchestration logic, but it is not durable background execution. Production apps can provide another `Dispatcher` that submits work to an external queue or hosted runtime and returns a queued/submitted handle while events are observed through the session store. `CayuApp.dispatch_inline(...)` is the explicit local streaming API for callers that want to consume ordinary `session.resumed`, model, tool, task, interrupt, and terminal session events directly. `DispatchRequest.task_id` optionally links dispatched work to a task; using it with inline execution requires `CayuApp(task_store=...)`.

`SubagentTool` is model-facing delegation over the same session substrate, not a
separate runtime. It creates a new child `RunRequest` with `parent_session_id`
set to the calling session and `causal_budget_id` inherited from the caller,
then runs the configured child agent through the normal Cayu loop. The child
agent has its own `AgentSpec`, tools, policies, model, context policy, and
durable events. Foreground subagents wait for the child terminal event; the
parent receives only a bounded `ToolResult` containing the child session id,
status, and model-facing result. `SubagentSpec.result_max_chars` caps the child
text copied into the parent transcript. Background subagents return after the
child emits its first runtime event, so the parent receives the child session id
without waiting for completion. The active runtime process must keep running for
in-process background child work to finish; external queue placement remains a
dispatcher responsibility.
Consumers that need child progress should observe the event sink or query
sessions by `parent_session_id`; child events are not rewritten as parent
events. If the parent model needs a background child result later, register
`SubagentResultTool`. It is parent-scoped: the model can fetch one returned
`child_session_id` or `all=true` for background subagents created by the current
parent session. Interrupting an active parent session also interrupts running
background subagent children.
The initial subagent context mode is `task_only`: the child receives the
delegated task as a user message and does not copy the parent's transcript.
Transcript-copying remains the job of `ForkSessionRequest`; future subagent
context modes can compose with fork when a child truly needs inherited
conversation state.

`RuntimeHook` provides lifecycle automation around durable runtime boundaries. Terminal session phases are `after_session_completed`, `after_session_failed`, and `after_session_interrupted`. These hooks run only after the terminal session status and terminal event have already been persisted. A hook failure does not rewrite the terminal session status; Cayu records `hook.failed` and continues to later hooks. Successful hooks emit `hook.started` and `hook.completed`, including the hook scope, terminal event id/type, and JSON-safe action summaries. `RuntimeHookContext` exposes copied session/event data plus controlled helpers for `fork_session`, `create_task`, `dispatch`, `dispatch_inline`, and custom event emission.

`before_tool_call` runs after `ToolPolicy` authorizes a call and before the tool executes — policy is the security gate, this phase is the transform layer. It receives `BeforeToolCallHookContext` (copied session data, tool name/id, copied arguments, optional task id, and the same controlled helpers as terminal hooks) and may return a `BeforeToolCallDecision`: `proceed` (or `None`) runs the tool unchanged, `proceed_modified` runs it with replaced `modified_arguments`, `short_circuit` skips the tool and uses a `synthetic_result`, and `block` skips the tool and returns an error result carrying `block_reason` as `tool.call.blocked`. Hooks compose in app-then-agent registration order, each seeing the prior hook's modified arguments; the first `short_circuit`/`block` stops the chain. When a hook replaces the arguments (`proceed_modified`), the effective arguments are **re-authorized by `ToolPolicy`** before the tool runs, so the gate always vets what actually executes — a hook cannot slip modified arguments past policy. A re-authorization that returns `deny` blocks the call; `require_approval` on hook-modified arguments also blocks (fail-safe) rather than re-entering approval, which is unsupported in v1. The executed (effective) arguments are what `after_tool_call`, `ToolCallOutcome`, and the result event's `effective_arguments` field carry; the `tool.call.started` event keeps the model's originally requested arguments. A hook-modified call is therefore authorized twice — once on the original arguments, once on the effective arguments; the re-authorization request carries `metadata[TOOL_POLICY_REAUTHORIZATION_METADATA_KEY] = True`, so a stateful policy (rate limiter, counter, audit sink) can re-verify the effective arguments while incrementing/logging only once.

`after_tool_call` runs after the tool executes and **before** the tool result event is persisted, so it can rewrite the result the transcript keeps and the model sees. It receives `ToolCallHookContext` (copied session data, the tool event, tool name/id, copied arguments, copied `ToolResult`, optional task id, and the controlled helpers) and may return an `AfterToolCallDecision`: `pass_through` (or `None`) leaves the result unchanged, `modify` replaces it with `modified_result`. `modify` is honored only for **real tool outcomes** — the executed tool's `tool.call.completed`/`failed` result and a before-hook `short_circuit` result. On authority/infrastructure results — policy `tool.call.blocked`, an unregistered-tool error, a before-hook `block`, and operator approval denials — `after_tool_call` still runs but is observe-only, so a hook cannot rewrite a gate decision into a success. Secret redaction runs before `after_tool_call`, over both the result **and** the context arguments, so an after-hook never observes raw secrets; a modified result is re-redacted. (`before_tool_call` receives raw arguments by design — it is the input-transform layer, so a redact-an-argument hook must see the real value.) The `ToolCallHookContext.tool_event` handed to the hook is not yet persisted during this phase (it persists afterward, with the same id) — use `context.result`, do not reload it. Mutating the context's copied arguments or result is still a no-op — modification happens only through the returned decision. Because after-hooks run before persistence, the tool result event (`tool.call.completed`/`failed`/`blocked`) is emitted after that phase's `hook.started`/`hook.completed` events. Both tool phases apply to subagent tool calls through app-level hooks, since a subagent reuses the parent `CayuApp`. A hook that raises or returns an invalid decision emits `hook.failed` and the call proceeds unmodified (before) or passes through (after); the tool round is never aborted by a hook error. The manual approval-recovery path (`resolve_tool_approval` supplying a recovered result) persists before hooks run, so `after_tool_call` is observe-only there.

Hook helper side effects are persisted and sent to event sinks; the parent run stream yields the hook telemetry events. Hook-emitted custom events must use the `custom.` namespace. `CayuApp(runtime_hooks=[...])` registers app-level hooks for global middleware, while `register_agent(..., runtime_hooks=[...])` registers hooks that run only for that agent. App-level hooks run before agent-level hooks. Hooks that fork and dispatch follow-up sessions should still guard on session metadata, task type, or another app-owned marker when they can process their own follow-up sessions. This lets apps implement follow-up work such as “fork this completed builder session and dispatch a knowledge-extraction task” without making fork, task, or dispatch mean the same thing.

`SQLiteSessionStore` is the durable local implementation. It stores sessions, append-only events, provider-neutral transcript messages, and the latest checkpoint in SQLite, while keeping session identity and event identity fields queryable as columns. Session identity includes agent, provider, active model, runtime, environment, and parent session. Event identity includes event type, agent, environment, workflow, and tool. `InMemorySessionStore` remains for tests and small examples. Hosted use can later provide a different `SessionStore`, such as Postgres, without changing runtime behavior.

JSONL can be added later as an export/debug format. It should not be the primary Cayu session store because dashboards, replay, task orchestration, retries, and hosted runtimes need indexed structured queries and transactional state updates.

Session stores expose two read surfaces:

- `load_events(session_id)` returns the full event list for one session.
- `query_events(EventQuery(...))` returns `EventRecord` values with durable sequence numbers for filtered timeline/dashboard reads.

The optional FastAPI server exposes the same event query surface at
`GET /api/sessions/{session_id}/events`. The endpoint validates the session,
accepts `after_sequence`, `limit`, `event_type`, `tool_name`, `agent_name`,
`environment_name`, and `workflow_name` filters, and returns durable event
records with `sequence`, `has_more`, and `next_sequence`. Clients should use
this endpoint for timelines, logs, replay panes, and polling instead of fetching
the full session when they only need events.

The optional server also exposes `GET /api/sessions/{session_id}/transcript`
for paginated transcript inspection. It accepts `offset`, `limit`, and `role`
filters and returns provider-neutral messages with their zero-based transcript
`index`. Transcript pagination is intentionally offset-based because the current
transcript store is append-only per session and compaction/resume already reason
about transcript positions as message counts. Events remain the source of truth
for execution chronology; transcript messages are the provider-neutral
conversation state used by resume, context policy, and compaction.

The optional server exposes `GET /api/sessions/{session_id}/summary` for compact
session health views. It combines session identity/status, storage-backed event
totals and counts, the latest event, transcript message count, normalized usage,
and a derived outcome. The outcome is computed from durable events rather than
stored as separate state. It reports the current status reason, terminal event,
latest retry event in the latest session invocation, and compact event details
such as limit values, error type, or interruption type when available. It does
not estimate cost because Cayu does not own provider pricing; callers pass their own pricing to
`POST /api/sessions/{session_id}/cost` when they need cost estimates.

Session stores also expose `list_sessions(SessionQuery(...))` for dashboard and replay views, and `load_transcript(session_id)` for the provider-neutral model conversation used by resume and compaction APIs. Runtime code can write one event with `append_event(...)` or write a durable batch with `append_events(...)`. Runtime code appends transcript messages as it builds the model conversation: initial messages, resumed request messages, assistant model messages, and tool-result messages. Batched event appends must be atomic: if one event in the batch is invalid or duplicated, none of the batch should be persisted. Terminal tool events must be durable before approval retry and ordinary tool-round recovery can safely skip execution. `append_transcript_messages_and_checkpoint(...)` must also be atomic: it is the boundary used when opening or closing recoverable tool rounds, where the transcript cannot be updated independently from the checkpoint.

## TaskStore

Creates and updates optional durable units of work.

A task is not a PM-specific object. It is a generic work item that can represent a webhook job, background agent run, workflow step, orchestrator assignment, coding task, invoice-processing job, report generation job, or external automation. Simple one-off agent calls do not need tasks; they can use only sessions and events.

`Task` values have type, status, optional session/parent-task/assigned-agent identity, JSON-object input, optional JSON-object result/error, JSON-object metadata, and lifecycle timestamps. `TaskStore` exposes:

- `create_task(TaskCreate(...))`
- `load_task(task_id)`
- `list_tasks(TaskQuery(...))`
- `start_task(task_id, session_id=...)`
- `attach_task(task_id, session_id=..., worker_id=...)`
- `pause_task(task_id, reason=..., payload=...)`
- `block_task(task_id, reason=..., payload=...)`
- `mark_task_needs_attention(task_id, reason=..., payload=...)`
- `resume_task(task_id)`
- `claim_task(worker_id, TaskQuery(...), lease_seconds=...)`
- `heartbeat(task_id, worker_id, extend_seconds=...)`
- `release_task(task_id, worker_id)`
- `reclaim_expired(query=..., max_reclaims=...)`
- `complete_task(task_id, result)`
- `fail_task(task_id, error)`
- `cancel_task(task_id, error=...)`

Valid task lifecycle is intentionally small for the foundation:

```text
pending -> running
pending -> claimed
pending -> paused | blocked | needs_attention
claimed -> running
claimed -> pending
claimed -> paused | blocked | needs_attention
running (unattached) -> paused | blocked | needs_attention
paused | blocked | needs_attention -> paused | blocked | needs_attention
paused | blocked | needs_attention -> pending
pending -> completed | failed | cancelled
claimed -> completed | failed | cancelled
running -> completed | failed | cancelled
paused | blocked | needs_attention -> completed | failed | cancelled
terminal statuses do not transition
```

`paused`, `blocked`, and `needs_attention` are task-level orchestration states. They stop queue workers from claiming the task until trusted app code calls `resume_task(...)`. Held tasks can move directly among held states, so an app can escalate `blocked` to `needs_attention` without briefly returning the task to the claimable queue. They are not session pause states and they do not interrupt a live model loop. A running task that is already attached to a session cannot be moved into one of these held states through `TaskStore`; use session interruption/recovery for active model work instead. `status_reason` and `status_payload` describe the current held state and are cleared when the task resumes or becomes terminal.

`InMemoryTaskStore` exists for tests and examples. `SQLiteTaskStore` is the durable local implementation.

There are two supported task execution modes:

1. **Direct task/session link.** `CayuApp(task_store=...)` can link an agent run to an existing pending task through `RunRequest.task_id`. The runtime starts that task with the created session id, emits `task.started`, and marks the task completed or failed when the run reaches those terminal states. Use this when app code already decided exactly which task and session should run.
2. **Worker-claimed queue task.** App-owned worker code can atomically claim one unattached pending task with `claim_task(worker_id, query)`. The claim marks the task `claimed`, records `worker_id`, and sets `lease_expires_at`; it does not attach a session or mark the task started. The worker must pass both `task_id` and `task_worker_id` to `RunRequest`; Cayu then calls `attach_task(...)` to move the live owned claim to `running` with the created session id. The worker should call `heartbeat(...)` while it is doing pre-run work or while the agent run is active. It can `release_task(...)` before session attachment if it decides not to process the task. Another worker can call `reclaim_expired(...)` to return abandoned unattached claims to `pending`.

Claim queries intentionally do not support `session_id`, `limit`, or `offset`. Queue claims always pick one unattached pending task; tasks already linked to a session are no longer free queue work. `reclaim_expired(...)` also ignores attached tasks, even if their lease timestamp has passed, because the associated session may still be running or recoverable through session recovery. Once a claimed task is attached to a session, runtime completion/failure owns the task terminal update; the app should observe the session/task events instead of releasing that task back to the queue.

Held tasks are not reclaimed by lease cleanup. If a worker claims a task and discovers a dependency or human-review requirement before attaching a session, it should call `block_task(...)`, `pause_task(...)`, or `mark_task_needs_attention(...)`; these clear worker ownership and lease state. Later, app/operator code can call `resume_task(...)` to return the task to the pending queue.

The server exposes the same lifecycle for operator/backend integrations through `POST /api/tasks/{task_id}/pause`, `POST /api/tasks/{task_id}/block`, `POST /api/tasks/{task_id}/needs-attention`, and `POST /api/tasks/{task_id}/resume`. Hold endpoints accept optional `reason` and `payload` fields. `GET /api/tasks` stays a compact list view and does not include task input/result/error/metadata; lifecycle mutation responses return the full task detail for the task that was changed.

This is a durable ownership primitive, not a project-management system, retry scheduler, DAG engine, or agent messaging table. Apps own assignment policy, priorities, dependency graphs, retry timing, human workflows, and worker deployment. `examples/task_worker_loop.py` shows the queue-worker pattern with claim, heartbeat, run, failure, and reclaim paths.

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

## Event Watchers

Event watchers are durable app-side processors for events that were already
persisted by the runtime. They are for side effects that should survive worker
restarts, such as sending a budget alert email after `budget.limit_reached`,
posting a webhook when a task completes, or dispatching follow-up work after a
session finishes.

They are deliberately separate from runtime hooks and event sinks:

- hooks run inside the model/tool loop and can affect runtime decisions;
- event sinks fan out live events for logs, CLIs, dashboards, or webhooks;
- event watchers pull durable events later and record delivery state.

Watchers are trusted application code. The model cannot install arbitrary
watchers or scripts. A watcher has a stable name, an `EventQuery` filter, and a
handler. `CayuApp.run_event_watchers([...])` processes matching events with
ordered at-least-once delivery. The watcher cursor advances only after the
handler succeeds or the event reaches `max_attempts` and is dead-lettered. If a
handler fails below the attempt ceiling, the cursor stays on that event and the
next watcher run retries it before later matching events.
Watcher throughput is controlled by `EventWatcher.batch_size` and the
`run_event_watchers(..., limit=...)` call; `EventQuery.limit` is ignored for
delivery because the watcher owns the cursor.

`InMemoryEventWatcherStore` is useful for tests and single-process examples.
`SQLiteEventWatcherStore` persists watcher cursors, leases, attempts, last
errors, and dead-letter counts for durable local apps.
`PostgresEventWatcherStore` provides the same contract for hosted multi-worker
apps and uses transactional row locks to serialize claims for the same watcher.
A live lease prevents two workers with the same watcher name from handling the
same event at the same time; an expired lease can be claimed again, so handlers
must be idempotent. Use a stable idempotency key such as
`(watcher_name, event.id)` when calling external systems.

Changing a watcher filter while reusing the same watcher name changes the
meaning of its cursor. Use a new watcher name when the event selection changes
semantically.

## Agent

Turns messages into event streams using:

- model providers
- tools
- memory
- workflows
- runtime services

The initial `CayuApp` runtime registers agent specs, model providers, and tools, then emits and persists events for one session run. A run may make multiple model requests: model output can request tools, the runtime executes those tools, appends assistant `tool_call` messages and matching `tool_result` messages, and calls the model again until the model completes without tool calls or `RunRequest.max_steps` is exceeded. Multiple tool calls from one model step are grouped into one assistant message and one tool-result message in Cayu's internal transcript. Provider adapters must emit a `completed` stream event for each model step; a stream that ends silently is treated as a failed runtime contract.

`CayuApp.run()` and `CayuApp.resume()` are event-stream APIs. Runtime failures are represented as terminal `session.failed` events rather than re-raised exceptions from the iterator. A stricter programmatic API can be added later on top of the same runtime path.

## Model Step Outcomes

Provider `completed` stream events keep their raw provider payload in the
durable `model.completed` event and also include normalized completion metadata:

```json
{
  "completion": {
    "finish_reason": "stop",
    "raw_finish_reason": "end_turn",
    "status": "completed"
  }
}
```

`completion.finish_reason` is Cayu's provider-neutral finish reason. Current
values are `stop`, `tool_calls`, `length`, `content_filter`, `error`, and
`unknown`. `raw_finish_reason` and `status` preserve the provider-facing values
when they exist.

The runtime also classifies the assembled assistant step and stores that in
`model.completed.step_classification`. The classifier is based on the assistant
message, tool calls, provider state, and normalized completion metadata. Current
types are:

- `continue`: the assistant requested tool calls.
- `final`: the assistant produced user-visible content and no tool calls.
- `length`: the provider stopped because an output limit was reached.
- `filtered`: the provider stopped because content was filtered.
- `failed`: the provider reported a failed model step.
- `think_only`: the provider returned continuation state but no visible text or
  tool calls.
- `invalid`: the step had no visible content, tool calls, or provider state.

This classification is observability and policy input. The default runtime loop
still continues on tool calls and stops when there are no tool calls. Repair
policies, structured-output retry, length continuation, and goal/task gates
should build on this typed step outcome instead of parsing raw provider payloads
or guessing from transcript shape.

## Before-Stop Loop Policies

`LoopPolicy.before_stop(...)` is the runtime seam immediately before Cayu marks a
no-tool-call assistant step as complete. It receives a `BeforeStopContext` with
the current `Session`, `AssistantStepResult`, `StepClassification`, step number,
max steps, and request metadata. Policies are ordinary Python objects and can be
registered on `CayuApp`, on an agent, or on an individual `RunRequest`,
`ResumeRequest`, `DispatchRequest`, or tool-approval continuation request.
The generic before-stop seam runs for ordinary final assistant steps. When a
`StructuredOutputSpec` is active, structured-output validation owns the final
retry/completion path so the generic before-stop policy does not compete with
the runtime final-output contract.

Policies run in deterministic order: app policies, then agent policies, then
request policies. The first non-`complete` decision wins. Supported decisions:

- `complete`: allow normal session completion.
- `continue`: append the returned user `Message` to the durable transcript and
  call the model again, as long as another model step remains.
- `interrupt`: mark the session `interrupted` with a durable
  `session.interrupted` event so the caller can resume later.
- `fail`: fail the session through the normal `session.failed` path.

Cayu emits durable `custom.loop.before_stop.started`,
`custom.loop.before_stop.completed`, `custom.loop.before_stop.selected`, and
`custom.loop.before_stop.failed` events for configured policies. A policy
exception fails the session; this is intentional because before-stop policies
control whether the runtime is allowed to complete. Side-effect-only behavior
belongs in runtime hooks instead.

## Structured Output

`StructuredOutputSpec` is the contract for final JSON output. It can be
attached to `RunRequest`, `ResumeRequest`, `DispatchRequest`,
`ToolApprovalRequest`, and `ToolApprovalRecoveryRequest`. The spec contains a
`json_schema` object, an optional name, a bounded `max_retries`, an optional
repair prompt, and a `strategy`. `max_retries` defaults to `2`, meaning the
initial model attempt may be followed by two repair attempts. The default
strategy is `tool`, which is the provider-neutral portable path.

When `strategy="tool"` is present, Cayu adds a runtime-owned
`__cayu_submit_structured_output` tool to provider requests and injects
provider-facing system guidance. The tool takes a single `output` argument. Cayu
validates that value against the spec's schema. The tool is internal runtime
plumbing: it is not registered by the app, does not execute user code, does not
go through tool approval, and does not count against user tool-call limits.

If the model calls the final-output tool by itself with a valid value, Cayu
appends a tool result to the durable transcript, emits
`structured_output.validated` with parsed JSON output, and completes the
session. If the submitted value is invalid, Cayu appends an error tool result,
emits `structured_output.failed`, emits `structured_output.retry` when retries
and model steps remain, and calls the model again with the provider-valid tool
result in context. Cayu writes those tool results before completing, retrying,
or failing so provider tool-call/tool-result history remains valid.

If the model calls the final-output tool in the same round as other tools, Cayu
rejects the entire round with error tool results and does not execute side
effects. The model can retry the needed work and submit final structured output
in a later round.

If the model ignores the final-output tool and returns plain final text, Cayu
treats that as a structured-output failure even when the text happens to be
valid JSON. If retries remain, it appends a synthetic user repair message to the
durable provider-neutral transcript, emits `structured_output.retry`, and calls
the model again. Cayu writes that repair message only when another model step is
available. If retries are exhausted, or no model step remains for repair, the
session fails with `session.failed`.

When `strategy="native"` is present, Cayu requires a provider that explicitly
supports native structured output. OpenAI maps the spec to the Responses API
`text.format` JSON-schema request shape. In native mode, Cayu does not inject the
runtime final-output tool; it validates the final assistant text as JSON against
the same schema before emitting `structured_output.validated`. Runtime
validation remains the correctness boundary, even when the provider also
enforces the schema.

## Usage Metrics

Provider `completed` stream events may include the provider's raw `usage` payload.
The runtime keeps that raw payload in the durable `model.completed` event and,
when token counters are available, adds provider-neutral `usage_metrics` beside
it.

Normalized usage includes:

- `input_tokens`
- `output_tokens`
- `total_tokens`
- `reasoning_output_tokens`
- `cache.read_tokens`
- `cache.write_tokens`
- `cache.cached_input_tokens`
- `cache.uncached_input_tokens`

OpenAI cached input token counters and Anthropic cache read/write token counters
are normalized into this shape. The provider-specific raw `usage` value remains
available for callers that need exact provider fields. If Cayu cannot normalize
a provider's usage payload, the durable `model.completed` event still keeps raw
`usage`; it simply omits `usage_metrics`.

Prompt caching behavior is not a universal runtime contract. Providers differ:
some apply prompt caching automatically, some require or benefit from explicit
cache controls, and some expose TTLs or routing hints. Cayu's runtime contract is
to preserve raw provider usage and normalize cache observability where possible.
Provider-specific cache controls should remain provider options rather than a
single Cayu-level abstraction that hides incompatible semantics.

Normal agent runs accept stable provider options on `AgentSpec`:

```python
AgentSpec(
    name="assistant",
    model="gpt-5.5",
    provider_options={
        "openai": {
            "prompt_cache_key": "tenant-a-agent",
            "prompt_cache_retention": "24h",
        },
        "anthropic": {
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        },
    },
)
```

The runtime copies these options into every provider request for that agent, then
adds framework-owned request metadata such as `agent_metadata`, environment
metadata, step number, and resolved file attachments. Provider-backed compaction
uses `ModelCompactor(options=...)` because compaction is its own direct model
request. OpenAI prompt caching is mostly automatic and can use
`prompt_cache_key` / `prompt_cache_retention` as provider options. Anthropic
automatic prompt caching uses top-level `cache_control`; explicit block-level
cache breakpoints are intentionally not modeled in Cayu's provider-neutral
message contract yet.

`CayuApp.get_session_usage(session_id)` derives totals from durable session
events. The optional FastAPI server exposes the same value at
`GET /api/sessions/{session_id}/usage`. Usage summaries are observability
records; retry, budget, and stop policies should consume them instead of parsing
provider-specific payloads directly.

`CayuApp.get_causal_budget_usage(causal_budget_id)` derives the same normalized
usage totals across every session whose stored `causal_budget_id` matches. The
summary includes `session_ids`, `session_count`, and per-session
`session_summaries` so callers can see which parent/fork/task-linked sessions
are included. The optional server exposes this grouped view at
`GET /api/causal-budgets/{causal_budget_id}/usage`.

`PricingCatalog` and `ModelPricing` estimate session cost from durable
`model.completed` events. Pricing is caller supplied; Cayu does not hardcode
provider prices. `CayuApp.get_session_cost(session_id, pricing)` walks each
model step, matches the provider-returned model to exact or prefix pricing
entries, then falls back to the requested model recorded on the step when the
provider returned a resolved/snapshot name. It returns a `SessionCostSummary`
with per-step `CostLineItem` records. Missing usage or missing pricing is
reported as unpriced line items so dashboards and operators can see estimation
gaps instead of silently treating them as free. If cache read/write prices are
omitted, the estimator falls back to the configured input-token price for those
counters.

The optional FastAPI server exposes the same estimator at
`POST /api/sessions/{session_id}/cost`. The request body supplies a
`PricingCatalog` and optional `currency`; the response is the JSON form of
`SessionCostSummary`, with decimal cost values serialized as strings for stable
API output.

`CayuApp.get_causal_budget_cost(causal_budget_id, pricing)` uses the same
pricing contract and line-item estimator across all sessions in that causal
budget. The optional server exposes this at
`POST /api/causal-budgets/{causal_budget_id}/cost`; the response includes
`causal_budget_id`, `session_ids`, `session_count`, and the same estimated-cost
fields as per-session cost summaries, plus `session_costs` for per-session
breakdown.

The optional server also exposes
`POST /api/causal-budgets/{causal_budget_id}/summary` for one-call work-item
observability. It accepts the same pricing body as the causal cost endpoint and
returns the included sessions, each session's derived outcome and event counts,
the grouped usage summary, and the grouped cost summary. This endpoint is a
composition of durable session/event data; it does not add another accounting or
budget-enforcement path.

Sessions also carry app-owned `labels`: stable key/value dimensions such as
`organization`, `project`, `owner`, `workflow`, `customer`, or `environment`.
Labels are for application grouping and filtering, not for runtime control flow.
Use labels when the dimension belongs to the app domain and more dimensions may
appear over time. Use `causal_budget_id` when the dimension is the execution
accounting group for one work item and its forked/subagent sessions. A common
shape is:

```python
RunRequest(
    agent_name="assistant",
    session_id="invoice_1042_root",
    causal_budget_id="invoice_1042",
    labels={
        "organization": "org_123",
        "project": "ap_q2",
        "workflow": "invoice-review",
    },
    messages=[Message.text("user", "Review invoice 1042.")],
)
```

`SessionQuery.labels` performs exact key/value matching. `label_selectors`
support existence and set-style matching through `LabelSelectorRequirement`:

```python
SessionQuery(
    labels={"organization": "org_123"},
    label_selectors=[
        {"key": "project", "operator": "in", "values": ["ap_q2", "research"]},
        {"key": "archived", "operator": "not_exists"},
    ],
)
```

The optional server exposes the same filtering on `GET /api/sessions` through
repeated `label=key=value` and `label_selector=...` query parameters. Supported
selector forms are `workflow`, `!archived`, `project=ap_q2`,
`project!=legacy`, `project in (ap_q2,research)`, and
`project notin (legacy,archived)`.

For many-session health views, use `POST /api/sessions/summary` with the same
typed filters, exact labels, and label selectors as `GET /api/sessions`. It
returns the matched sessions, per-session outcome and event counts, aggregate
normalized usage, and optional aggregate/per-session cost when the request body
includes a `PricingCatalog`. This is the right endpoint for app dashboards like
"usage and cost for org 123's AP Q2 invoice sessions" where there may not be one
shared causal budget id.

```bash
curl -X POST \
  "http://localhost:8000/api/sessions/summary?label=organization=org_123&label_selector=project%20in%20(ap_q2,research)" \
  -H "Content-Type: application/json" \
  -d '{"pricing":{"prices":[{"provider_name":"openai","model":"gpt-5.5","match":"prefix","input_per_million":"2.00","output_per_million":"8.00","cache_read_input_per_million":"0.50"}]}}'
```

For compact health views, use the server's
`GET /api/sessions/{session_id}/summary`. The summary endpoint includes outcome
data derived through `SessionStore.summarize_outcome(session_id)`: current
status reason, compact terminal details, and the latest retry event for the
latest session invocation. It complements, but does not replace, event replay:
outcome answers "why is this session in its current state?", usage answers "how
many tokens/cache counters were observed?", and cost answers "what does my
pricing table estimate for those events?"

Cost estimation is observability, not billing authority. Provider invoices,
rounding, regional pricing, provider-side discounts, or account-specific terms
can differ from a caller's pricing table. Cost stop policies should build on
this primitive only after the app has supplied pricing appropriate for its
deployment.

`RunLimits` provides hard token/tool/time stop controls for runtime calls.
`BudgetLimit` provides estimated-cost stop controls backed by a caller-supplied
`PricingCatalog`. Request-scoped `BudgetLimit` entries can be attached through
`budget_limits` on `RunRequest`, `ResumeRequest`, `DispatchRequest`,
`ToolApprovalRequest`, and `ToolApprovalRecoveryRequest`. `scope="session"` is
the default: token, tool-call, and cost limits are session-cumulative because
they are evaluated from durable `model.completed` and `tool.call.started`
events. `scope="run"` evaluates token, tool-call, and cost limits against the
delta since the current `run(...)`, `resume(...)`, dispatch, or
approval-continuation invocation started. `max_elapsed_seconds` is always scoped
to the current runtime invocation and resets for each call.

Budget limits are estimates, not billing records. They use normalized usage
metrics and the app's pricing table. By default, a request-scoped interrupt
budget fails closed when a newly observed model step has no matching pricing
entry; Cayu interrupts instead of silently treating unknown usage as free. A
request-scoped notify budget emits `budget.limit_reached` for the same
unverifiable usage and continues. Apps that intentionally allow missing prices
can set `allow_unpriced=True` for that request.

Request `budget_limits` may use `scope="session"` or `scope="run"` for direct
request budgets. They may also use `scope="agent"` or `scope="causal"` for
dynamic work-item budgets passed by the caller; those keys must match the
current agent name or current `causal_budget_id`. `scope="app"` is accepted for
intentional per-request global checks, but durable app-wide policy normally
belongs on `CayuApp(budget_policy=...)`. Request budget limits must not include
reservations; strict concurrent reservations are app-policy/ledger behavior.
`BudgetLimit.action` defaults to `"interrupt"`. With `action="notify"`, Cayu
emits a durable `budget.limit_reached` event when the estimated-cost threshold
is reached, but it does not emit `session.limit_reached`, does not interrupt the
session, and does not close pending tool rounds.

App-level budgets are configured separately on `CayuApp` through
`BudgetPolicy`. A policy contains app-wide, agent-scoped, and causal
`BudgetLimit` entries. Budget windows default to `BudgetWindow.all_time()`.
`BudgetWindow.rolling(seconds=...)` evaluates durable model events whose UTC
event timestamp is inside the trailing moving window at the time Cayu checks the
limit. `BudgetWindow.calendar(period="day" | "week" | "month",
timezone="...")` evaluates durable model events inside the current local
calendar period for the configured IANA timezone. Calendar days reset at local
midnight, calendar weeks start on Monday, and calendar months start on the first
day of the month. Rolling and calendar windows can be used together as separate
`BudgetLimit` entries when an app needs both spend-velocity protection and
daily/monthly accounting. Budget limits are estimated from the same normalized
usage and caller-supplied `PricingCatalog` used by request-scoped
`BudgetLimit` entries.

`scope="app"` applies to all sessions and must not set `key`. `scope="agent"`
applies when `key` matches the agent name. `scope="causal"` applies when `key`
matches `RunRequest.causal_budget_id`. If omitted, a root session's
`causal_budget_id` defaults to `task_id` when the run is linked to a task,
otherwise to its session id. Forked sessions inherit the source session's causal
budget id, so parent and child sessions can share a single work-item budget.

The runtime checks matching app-policy budget limits before every model step and
again after each completed model step. A pre-model check also verifies that the
current requested provider/model has matching pricing unless `allow_unpriced=True`. Each
app-policy check emits `budget.checked`. If an interrupt budget is reached, the
runtime emits `budget.limit_reached` and then follows the controlled stop path:
`session.limit_reached`, `session.interrupted`, and a resumable interrupted
session. If a notify budget is reached, the runtime emits `budget.limit_reached`
with `action="notify"` and continues. Apps can route those durable notify events
to email, Slack, webhooks, or dashboards through trusted app/event-sink code.
A resume under the same exhausted matching interrupt budget stops again until
the app changes policy, raises the limit, fixes missing pricing, or opts into
unpriced usage with `allow_unpriced=True`.

`SessionBudgetStore` is the default budget store. It reads from the same durable
event stream already configured for sessions, including timestamp filters for
rolling and calendar windows, so `SQLiteSessionStore` can back budget
accounting across process restarts and multiple workers that share the same
database. Enforcement is cooperative: Cayu checks before model calls and again
after model completions.

Strict concurrent hard caps use `BudgetLimit.reservation` plus a
`BudgetLedger`. A reservation declares the maximum input, output, cache-read,
and cache-write tokens the application is willing to fund for one provider step.
Before the provider call, Cayu prices that worst-case step with the same
`PricingCatalog` and atomically reserves it in the ledger. Accepted reservations
emit `budget.reserved`; failed reservations emit `budget.reservation_failed`,
then `budget.limit_reached`, and stop before the provider request. After
`model.completed`, Cayu reconciles the reservation to actual normalized usage
and emits `budget.reconciled`. If the model step fails before completion, Cayu
releases the reservation and emits `budget.reservation_released`.
With rolling or calendar budget windows, unresolved active reservations continue
to consume capacity until they are reconciled or released; reconciled spend ages
out by the reconciliation/model-completion timestamp.

`InMemoryBudgetLedger` is the default and is only strict inside one process.
Multi-worker apps that need hard shared caps should pass `SQLiteBudgetLedger`
or their own `BudgetLedger` implementation. Reservation amounts are
application-provided upper bounds; Cayu does not infer how large a future model
step will be. Reservation limits require matching pricing and cannot use
`allow_unpriced=True`. Reservation limits also require `action="interrupt"`
because reservations are hard-cap accounting, not observe-only alerts.
`InMemoryBudgetStore` only supports simple app/agent event filtering. Causal
budgets require `SessionBudgetStore` or another session-aware `BudgetStore`
because they depend on persisted session identity.

When an interrupt limit is reached, the runtime emits `session.limit_reached`,
updates the session to `interrupted`, and emits `session.interrupted` with
`interruption_type="limit_reached"`. This is a controlled pause, not a runtime
failure. The session can be resumed later with a higher cumulative budget, a
different instruction, no cumulative budget, or a run-scoped limit. If a resume
call repeats the same already-exhausted session-scoped token, tool-call, or cost
limit, Cayu interrupts again before doing more work; this prevents a lifetime
budget from being bypassed by repeatedly continuing the same session. If
`scope="run"` is used, token/tool-call/cost counters start from the invocation
baseline but the `usage_summary` in `session.limit_reached` remains the
cumulative session summary. The event's `actual` field is the value evaluated
for the selected scope. Cost-limit events also include `cost_summary`; decimal
cost values are serialized as strings for JSON stability. `cost_summary` is
cumulative, matching `usage_summary`; use `actual` for the scoped value that
triggered the stop. If a model step has already produced tool calls when an
interrupt limit is reached, Cayu does not execute those tools. It appends
skipped `tool_result` messages and emits `tool.call.failed` events before the
terminal interruption event so the provider-neutral transcript remains valid for
resume. Notify budgets do not skip tools or change session status. App-policy
notify events are edge-triggered once per matching threshold/window; subsequent
`budget.checked` events continue to expose the above-limit state.

## Retry Policy

`RetryPolicy` controls retry attempts for one provider model step. It can be
configured as a `CayuApp(retry_policy=...)` default or attached to
`RunRequest`, `ResumeRequest`, `DispatchRequest`, `ToolApprovalRequest`, and
`ToolApprovalRecoveryRequest`. Request-level policy overrides the app default.
The default policy has `max_attempts=1`, which means retries are disabled.

Retries are deliberately scoped to the model provider request. The runtime emits
`model.started` for each attempt. If a retryable provider error happens, Cayu
emits `model.error`, emits durable `model.retry` with attempt, next attempt,
reason, status code, delay, provider, and model fields, waits for the configured
backoff delay, and starts a new provider attempt. Retried failed attempts do not
append assistant messages to the provider-neutral transcript.
When retries are enabled, provider-derived `model.text.delta`, `model.error`,
and `model.completed` events include `step`, `attempt`, and `max_attempts` so
SSE consumers, dashboards, and replay tools can distinguish failed-attempt
output from the successful attempt.

Cayu does not retry tool execution. If a provider attempt emits tool calls and
then fails before the model step completes, those tool calls have not executed
yet and the provider step can still be retried. Once a provider step completes,
the assistant tool-call message may be appended and tool side effects may start;
later failures are handled by tool failure, approval recovery, interruption, or
session failure paths instead of provider-step retry.

Built-in retry classification covers HTTP 429/500/502/503/504/529 status text,
timeouts, connection/network failures, and rate-limit messages. Permanent
quota/billing failures are not retried even when a provider reports them with
HTTP 429. Provider adapters should keep enough error detail in
`ModelStreamEvent.error(...)` for classification and debugging while still
sanitizing secrets.

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

`AnthropicProvider` adapts the Anthropic Messages API to Cayu's provider-neutral transcript. It keeps Cayu `system` messages as Anthropic's top-level `system` field, maps assistant tool calls to Anthropic `tool_use` blocks, and maps Cayu tool-result messages back to Anthropic user `tool_result` blocks. Callers can override Anthropic request options through `ModelRequest.options["anthropic"]` except for fields owned by the provider contract.

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

Framework-native tools receive runtime services through `ToolContext`: workspace, artifact store, runner, vault, credential proxy, knowledge store, and MCP server specs. These references are intentionally runtime-only. They are excluded from `ToolContext.model_dump()` so context metadata can cross storage, event, dashboard, and replay boundaries without serializing live service objects. Serializable service identity fields such as `workspace_id` and `artifact_store_id` may be present when the active environment exposes them.

The first built-in tools are:

- `read_file`: read text from the active workspace by `path`, capture workspace image/PDF files as artifact snapshots when an artifact store is configured, read text artifacts by `artifact_id`, or return provider-neutral image/PDF attachment references for capable providers
- `write_file`: write UTF-8 text to the active workspace, capped by `max_bytes`
- `list_files`: list files in the active workspace, capped by `limit`
- `list_artifacts`: list session- or environment-scoped artifact metadata, capped by `limit`
- `exec_command`: execute an explicit process argv or shell script with the active runner, capped by `timeout_s` and `max_output_bytes`
- `subagent`: delegate a bounded task to a configured child Cayu agent; foreground mode returns the child result, while background mode returns the child session id after startup
- `subagent_result`: fetch one background subagent result by `child_session_id`, or wait for all background subagents started by the current parent session
- `list_knowledge`: discover active knowledge entries and facets without requiring a lexical search term
- `search_knowledge`: search the active knowledge store with bounded previews and optional filters
- `read_knowledge`: expand bounded chunks from a returned knowledge entry
- `remember_knowledge`: propose a new knowledge entry through the active knowledge store, subject to application write policy

These tools are ordinary `Tool` implementations. They prove the environment-service contract but do not make file or command access mandatory for all agents.

Default built-in tool caps are intentionally large enough for normal coding work but small enough to protect model context and runtime memory:

- `read_file`: 256 KB by default, 4 MB maximum per call
- `read_file` native file attachments: 8 MB by default, 8 MB maximum per call for the built-in tool instance. Applications may raise or lower that tool-facing cap with `ReadFileTool(default_attachment_limit_bytes=..., max_attachment_limit_bytes=...)`.
- Runtime file attachment resolution: 8 MB maximum per attachment, 32 MB maximum total per provider request, and 20 attachments maximum per provider request by default. Applications may override those runtime caps with `CayuApp(max_file_attachment_bytes=..., max_total_file_attachment_bytes=..., max_file_attachments_per_request=...)`.
- `write_file`: 256 KB by default, 4 MB maximum per call
- `list_files`: 500 paths by default, 10,000 maximum per call
- `list_artifacts`: 500 artifacts by default, 10,000 maximum per call
- `exec_command`: 60 seconds by default, 600 seconds maximum per call; 50,000 bytes stdout and 50,000 bytes stderr by default, 200,000 bytes maximum per stream per call
- `list_knowledge`: 10 entries or facets per group by default, 25 maximum per call/group; 240 bytes of preview text per entry by default, 4 KB maximum per entry, and 20 KB total preview text by default, 128 KB maximum per call
- `search_knowledge`: 10 hits by default, 25 maximum per call; 320 bytes of preview text per hit by default, 4 KB maximum per hit, and 20 KB total preview text by default, 128 KB maximum per call
- `read_knowledge`: 5 chunks by default, 50 maximum per call; 20 KB chunk text by default, 128 KB maximum per call
- `remember_knowledge`: 64 KB app-configured accepted text by default, 512 KB maximum for a registered tool instance; internal indexing defaults to 4 KB target chunks and 100 chunks maximum for the built-in tool instance; writes that would exceed the configured text or chunk capacity are rejected before persistence

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

Remote runner command cleanup is bounded. `E2BRunner` and `MicrosandboxRunner` expose `cancel_timeout_s`, defaulting to 5 seconds, `cancellation_cleanup`, defaulting to `"command"`, and `timeout_cleanup`, defaulting to `"command"`. Caller cancellation raises `RunnerCancelledError`, an `asyncio.CancelledError` subclass that carries cleanup diagnostics so the runtime can interrupt the session while preserving what happened. E2B delayed-start cleanup may continue briefly in the background after the foreground interruption or timeout cleanup wait; if it cannot resolve the command start, the runner exec path is closed while the sandbox is preserved. Command timeouts return `ExecResult(timed_out=True)`.

Both cleanup fields accept the same three modes:

- `"sandbox"`: call the sandbox provider's kill/terminate operation and close the runner. This is the strongest generic cleanup boundary for shells, child processes, and background work.
- `"command"`: call the provider command handle's `kill()` method. The runner stays reusable when ordinary command cleanup completes, fails, or times out after a command handle exists. If E2B never returns a delayed command handle after interruption or timeout, Cayu preserves the sandbox but closes that runner's exec path because later commands would overlap with unknown command state. Cayu records cleanup diagnostics so the app can surface the uncertainty without destroying workspace state.
- `"none"`: do not try to stop the command or sandbox. Cayu records a skipped cleanup diagnostic and leaves the runner reusable for ordinary cancellations where command state is already known. In the unresolved E2B delayed-start case, the runner exec path remains closed because Cayu cannot prove no late command is still running. This is for callers that own cleanup outside Cayu.

If cleanup fails or times out, Cayu includes a structured `cayu.runner_cleanup.v1` artifact with adapter, action, status, timeout, and error details when available.

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

The built-in composition vaults combine other vaults behind one `Vault` (e.g. static API keys +
per-tenant dynamic tokens in one environment):

- `ChainVault(*vaults)`: tries each vault in order; the first that resolves the secret wins.
  A vault raising `SecretNotFound` is skipped; any other error propagates.
- `RoutedVault(routes, *, fallback=None)`: routes each secret name to a specific vault, with an
  optional fallback for unrouted names.

`SecretEnv` represents a deliberate environment variable injection:

```python
SecretEnv(name="GITHUB_TOKEN", ref=SecretRef(name="github_token"))
```

Runner/MCP integrations should accept secret refs and resolve them through the
active environment vault at the execution boundary. The model should not receive
a general-purpose secret-reading tool. Application-owned tools can use secrets
internally by resolving refs in trusted code and returning safe results.

## Credential Proxy

`Vault` resolves `SecretRef` values. `CredentialProxy` is the runtime boundary
for trusted tools that need a controlled way to use those credentials. An
environment can expose both:

```python
Environment(
    EnvironmentSpec(name="trusted-tools"),
    vault=vault,
    proxy=PassthroughProxy(vault),
)
```

Tools then receive `ctx.proxy`. `PassthroughProxy` is a local/trusted
implementation that resolves through the active vault; production apps can
provide stricter proxies that authorize only specific destinations, scopes, or
actions.

The proxy does not replace the vault. The vault is still the source of secret
values; the proxy is the controlled use boundary exposed to trusted tools.
Generic model-driven command execution should not receive vault secrets by
default, because a command can print its own environment.

When a tool resolves a `ResolvedSecret` through `ctx.proxy`, Cayu records that
resolved value for the current tool result and extends the active redactor before
emitting durable events, storing transcripts, or building the next provider
request. Redaction matches resolved secret values, not `SecretRef` names or
metadata keys. A name such as `sendgrid_api_key` can remain visible while the raw
secret value is replaced with `[REDACTED_SECRET]`.

Trusted tools can also call `ctx.proxy.authorize_request(...)` before using a
credential for an outbound action. Cayu emits a durable
`credential.proxy.checked` event for each check with the destination, credential
reference name, action, metadata, allow/deny result, and proxy result metadata.
The event is redacted with any secrets resolved during the same tool call.

Authorization checks are an audit/enforcement hook for trusted proxy-aware
tools. Cayu records the decision; the proxy and tool implementation remain
responsible for denying or avoiding the outbound action when `allowed=False`.
`examples/credential_proxy_tool.py` shows a complete trusted-tool flow with a
strict proxy wrapper, `ctx.proxy.authorize_request(...)`,
`ctx.proxy.resolve(...)`, durable `credential.proxy.checked` events, and
redaction of a proxy-resolved value before event/transcript persistence.

This redaction is defense in depth. It does not intercept sandbox network calls,
make arbitrary shell commands safe for secrets, or guarantee safety for secrets
already present inside a workspace.

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
- `inherit_env=False` does not pass an empty environment. It copies only a small
  launcher safelist (`PATH`, `HOME`, locale/temp/profile variables) so common
  stdio launchers such as `npx` and `uvx` can start while host secrets remain
  excluded by default. Use explicit `env` for non-secret configuration and
  `secret_env` for vault-resolved child secrets.
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
- The discovered MCP tool manifest is fingerprinted as `McpToolset.manifest_hash`.
  The hash covers the server name, initialize metadata that affects tool
  descriptions, and every advertised tool name, generated Cayu tool name,
  description, input schema, and annotations. `McpToolAdapter` exposes the same
  value as `mcp_manifest_hash` and includes it in structured tool results so
  events/transcripts can show which MCP contract produced a result.
- Before a run or resume enters the model loop, Cayu emits a durable
  `mcp.manifest.checked` event for each MCP toolset exposed through the agent's
  registered tool adapters. The event compares the current manifest hash with
  prior durable events for the same server/environment and marks it as
  `first_seen`, `unchanged`, or `changed`. The payload stores the manifest hash,
  compact per-tool hashes, and added/removed/changed Cayu tool names; it does
  not store full schemas or server instructions. The comparison key is a stable
  `manifest_identity` built from the server name and generated Cayu tool names,
  so distinct same-name MCP toolsets are audited independently. This is an audit
  signal for dashboards, watchers, and deployment checks. It becomes an
  authorization decision only when `CayuApp(mcp_manifest_policy=...)` is
  configured.
- `McpManifestPolicy` maps manifest statuses and diffs to `allow`, `alert`, or
  `block`. `alert` keeps the run moving and records the policy decision on
  `mcp.manifest.checked`; `block` emits `mcp.manifest.blocked` and fails the
  session before the provider receives the changed MCP tool definitions. Specific
  diff actions such as `on_tools_added`, `on_tools_removed`, `on_tools_changed`,
  and `on_server_changed` override the generic `on_changed` action when set. If
  several changed aspects are present, Cayu applies the strictest resulting
  action. Without a configured policy, manifest checks remain audit-only.
- `McpToolAdapter` exposes one MCP tool as a normal Cayu `Tool`, so tool policies,
  approvals, events, transcript persistence, and provider adapters work through the
  same path as framework-native tools.
- Cayu tool names are prefixed with the MCP server namespace, such as
  `mcp__local-mcp__echo`, to make provenance visible and avoid collisions.

This first stdio client does not resolve `secret_env` itself. Secret resolution belongs
at the environment/vault boundary before the subprocess is started. Streamable HTTP MCP,
OAuth, MCP prompts, sampling, elicitation, and automatic resource injection are future
layers. MCP resources should remain explicit and policy-controlled instead of being
dumped into model context automatically. MCP manifest hashes are audit/debug
fingerprints, not signatures; tool policy still controls whether an allowed MCP
tool call may execute.

## KnowledgeStore

`KnowledgeStore` is Cayu's contract for durable reusable context: facts,
preferences, procedures, instructions, skills, documents, examples, warnings,
decisions, events, summaries, and app-defined custom kinds.

The first framework slices define the data contract, an `InMemoryKnowledgeStore`
for tests, demos, and single-process local apps, a `SQLiteKnowledgeStore` for
durable local knowledge, and a `PostgresKnowledgeStore` for production Postgres
deployments. There is no legacy memory alias in the public surface; use the
`Knowledge*` names and entry/chunk methods directly.

`InMemoryKnowledgeStore` is a simple keyword backend. `SQLiteKnowledgeStore` uses
SQLite FTS5/BM25 for durable keyword search. `PostgresKnowledgeStore` uses native
Postgres full-text search with entry/chunk filters. These stores support `auto`
and `keyword` query modes and reject semantic, hybrid, and external modes so apps
do not mistake keyword stores for embedding or external retrieval backends.
`InMemoryEmbeddingKnowledgeStore` is an opt-in in-memory semantic backend for
tests, demos, and small single-process apps. It uses a configured
`TextEmbeddingProvider`, keeps vectors outside chunk metadata, supports
`semantic` mode with cosine similarity, reports `score_normalized` as `(cosine +
1) / 2`, compares `semantic_min_score` against that normalized score, and uses a
bounded keyword boost for `hybrid` and `auto` mode. It does not persist vectors.
`PostgresEmbeddingKnowledgeStore` is the durable Postgres semantic backend. It
requires the pgvector `vector` extension, a configured `TextEmbeddingProvider`,
and explicit `embedding_dimensions` because the pgvector column type includes the
dimension. It keeps embeddings in a derived `cayu_knowledge_embeddings` table,
updates vectors on entry/chunk writes, reuses persisted vectors across process
restarts, and leaves plain `PostgresKnowledgeStore` usable without pgvector.
Semantic search uses chunks that already have vectors, with pgvector HNSW
indexing when `embedding_dimensions <= 2000`; larger dimensions use exact
pgvector search. Bulk backfill for existing knowledge is explicit via
`backfill_embeddings(..., limit=N)`, which embeds a bounded batch of matching
chunks. By default backfill only embeds missing or stale vectors; pass
`refresh_existing=True` to re-embed chunks whose current vector already matches
the configured model and dimensions.

Postgres embedding operational contract:

- pgvector must be installed in the target database. `schema_mode=CREATE` or
  `MIGRATE` runs `CREATE EXTENSION IF NOT EXISTS vector`; otherwise create the
  extension before starting the store.
- `embedding_dimensions` is required because pgvector stores the dimension in the
  column type (`vector(N)`). A store configured with a different dimension than
  the existing embedding table fails at startup with a dimension mismatch instead
  of silently mixing vector sizes. Changing dimensions requires rebuilding the
  derived embedding table before starting a store with the new dimensions.
- The embedding table is derived data. Rebuilding it is safe when source
  knowledge remains in `cayu_knowledge_entries` and `cayu_knowledge_chunks`.
- Existing keyword knowledge is not embedded implicitly at search time. Use
  bounded `backfill_embeddings(KnowledgeListQuery(...), limit=N)` jobs to index
  it deliberately. Repeated default backfills advance through missing/stale
  chunks; use `refresh_existing=True` only when intentionally re-embedding
  current rows for the configured model and dimensions. Switching to another
  embedding model with the same dimensions is a bounded re-indexing job; switching
  dimensions is a schema rebuild.
- Search embeds the query text, then searches persisted chunk vectors matching
  the configured model, dimensions, and current chunk content hash. Updating an
  entry or replacing chunks updates the affected vectors.
- HNSW is created for `vector(N)` when `N <= 2000`, matching pgvector's HNSW
  limit for the `vector` type. Larger dimensions are valid for exact pgvector
  search, but do not get the HNSW index in this store.
- Embedding calls are provider calls. Apps should account for provider latency,
  rate limits, retention, and billing when writing entries or running backfill.

- `KnowledgeEntry`: one reusable knowledge record with `namespace`, `labels`,
  extensible `kind`, visibility, status, source refs, audit timestamps,
  importance/confidence hints, aspects, impact targets, and metadata.
- `KnowledgeChunk`: bounded readable chunks for long entries. Stores may keep
  one default chunk for short entries, or replace the complete chunk set after
  indexing a larger source.
- `KnowledgeQuery`: scoped retrieval request with simple query text, structured
  keyword fields (`any_terms`, `all_terms`, `none_terms`, `phrases`), namespace,
  labels, kinds, status/visibility filters, aspects, impact targets, source
  filters, mode, result limit, and preview byte cap.
- `KnowledgeSearchResult`: result envelope containing copied hits, truncation
  metadata, configured limits, and `total_hits_known` when the backend can count
  candidates.
- `KnowledgeListQuery`: filter-only discovery request that can list entries or
  group matching entries by kind, label, aspect, impact target, visibility,
  source type, or namespace.
- `KnowledgeListResult`: copied entry previews plus optional facets/counts for
  agent navigation before targeted search.

`KnowledgeQuery.max_bytes` bounds returned `text_preview` payloads from
`search(...)`. It does not rewrite the copied `KnowledgeEntry` or
`KnowledgeChunk` objects inside each `KnowledgeHit`; model-facing knowledge tools
and automatic injection layers must build their own bounded provider context
from previews or explicitly read bounded chunks.

The store contract is intentionally entry/chunk oriented:

```python
await store.put_entry(entry)
await store.get_entry(entry_id)
await store.update_entry_status(entry_id, KnowledgeStatus.ARCHIVED)
await store.transition_entry_status(
    entry_id,
    from_status=KnowledgeStatus.PENDING,
    to_status=KnowledgeStatus.ACTIVE,
)
await store.delete_entry(entry_id, hard=False)
await store.replace_chunks(entry_id, chunks)
await store.put_entry_with_chunks(entry, chunks)
await store.read_chunks(entry_id, chunk_index=3, around=1)
result = await store.search(query)
listing = await store.list_entries(list_query)
```

Embeddings use a separate provider contract so model providers and vector stores
do not need to share implementation details:

```python
embedding_result = await embedding_provider.embed_texts(
    TextEmbeddingRequest(
        model="text-embedding-3-small",
        texts=["GitHub pushes should use the credential proxy."],
        dimensions=512,
    )
)
```

`OpenAIProvider` implements `TextEmbeddingProvider` with the OpenAI embeddings
endpoint. The returned `TextEmbeddingResult` contains copied float vectors and
provider-reported usage when the provider returns it. Embedding calls are normal
provider API calls: apps should treat their latency, rate limits, retention, and
billing as provider-specific behavior.

`KnowledgeIndexer` is the deterministic local indexing helper for app-owned text.
It is not an agent and does not connect to remote memory systems. It converts one
text body into a `KnowledgeEntry` plus bounded `KnowledgeChunk` records, computes
source/chunk hashes, preserves heading context for Markdown-like text, and can
write atomically through any `KnowledgeStore`.

```python
indexer = KnowledgeIndexer(store)
result = await indexer.index_text(
    KnowledgeIndexRequest(
        text=markdown,
        entry_id="repo-rules",
        title="Repo rules",
        kind="procedure",
        labels={"project": "cayu"},
        source_type="file",
        source_uri="workspace://AGENTS.md",
    )
)
```

Small overlap is enabled by default so important text is less likely to be split
across chunk boundaries. `max_chunks` and byte limits keep indexing bounded.
`skip_unchanged=True` avoids rewriting only when the stored source hash, derived
entry metadata, and derived chunks all match the newly indexed output.

Apps can expose explicit agent recall by attaching a knowledge store to the
environment and registering the built-in tools:

```python
store = SQLiteKnowledgeStore("knowledge.sqlite")
environment = Environment(
    EnvironmentSpec(name="local"),
    knowledge_store=store,
)

app.register_environment(environment, default=True)
app.register_agent(
    AgentSpec(name="assistant", model="gpt-5.5"),
    tools=[ListKnowledgeTool(), SearchKnowledgeTool(), ReadKnowledgeTool()],
)
```

`list_knowledge` builds a bounded active-only `KnowledgeListQuery` from filters
and optional `group_by`, returning compact entry previews and facets/counts so
the agent can discover what kinds of knowledge exist before guessing search
terms. The model-facing tool schema uses a list of facet fields, for example
`group_by=["kind"]` or `group_by=["kind", "aspect", "label"]`; direct runtime
calls may also pass a single facet field string. For large stores, agents should
usually call it with `group_by` first instead of listing many entries.
When `group_by` is set, the model-facing tool omits entry previews by default;
pass `include_entries=true` only when a small entry sample is useful. `limit`
also caps returned facets per group; `facets_truncated=true` means more facet
buckets matched than were returned.
`search_knowledge` builds a bounded active-only `KnowledgeQuery` from
model-provided query text, structured keyword fields (`any`/`all`/`none`/
`phrases`), and filters. It returns ranked compact previews plus entry/chunk ids;
`preview_bytes` controls per-hit snippet size, while `max_bytes` caps the total
tool payload.
`read_knowledge` then expands one returned entry by `entry_id`, optional
`chunk_index`, neighboring `around` window, chunk cap, and byte cap. These tools
fail as normal tool errors when the active environment has no `knowledge_store`,
so apps explicitly choose which agents can recall durable knowledge.

Apps may also register `RememberKnowledgeTool` when an agent should be allowed
to propose new durable knowledge. The tool is create-only: it does not accept an
`entry_id` and cannot edit, archive, or delete existing entries. It writes
through `KnowledgeIndexer` / `put_entry_with_chunks` so entry text, chunks, and
source hashes stay consistent across stores. `RememberKnowledgePolicy` controls
the actual stored status, namespace, visibility, required labels, and allowed
kinds. When `allowed_kinds` is configured, the registered tool instance exposes
those values as the model-facing `kind` enum while still enforcing them at
runtime. Model-facing inputs are deliberately limited to `text`, optional
`title`, optional `kind`, and optional topical `aspects`; namespace, labels,
status, visibility, impact targets, importance, and confidence are app-owned for
this tool. The default policy stores model-authored entries as `pending`; normal
search/list queries exclude pending entries, while reviewer/app code can query
pending entries through the store API. `KnowledgeReviewWorkflow` is the built-in
app-side helper for this path: it lists pending entries inside a configured
namespace/label scope, approves pending entries by moving them to `active`, and
rejects pending entries by moving them to `archived`. It is not a model-facing
tool. The packaged server/dashboard uses the same workflow when `CayuApp` is
constructed with `knowledge_store=...`; `knowledge_review_namespace` and
`knowledge_review_labels` limit which pending entries the dashboard can list or
approve/reject. The server exposes `GET /api/knowledge/pending` plus
`GET /api/knowledge/pending/{entry_id}` for scoped detail inspection and
`POST /api/knowledge/{entry_id}/approve` and `/reject` for that dashboard flow.
Active writes require `allow_active_writes=True`. The accepted text size is
configured when the app registers the tool and is not exposed as a
model-controlled argument. If
persistence fails after the tool has generated an entry id but the complete
durable entry and chunks match the intended write, including an embedding hook
failure in an embedding-backed store, the tool preserves the knowledge and
returns success with a structured `post_write_error` warning for app/event
consumers; the model-facing tool content still reports only that the knowledge
was stored. If the durable entry/chunks are not present or do not match the
intended write, the tool returns an error and attempts to hard-delete the
generated entry id.

Filters are retrieval hints, not an authorization boundary. Production apps
should attach a store already scoped to the active tenant/user/project, or
provide a wrapper store/tool that enforces those constraints before calling the
underlying store. The model does not need to know every namespace, label, kind,
aspect, impact target, or source value to use `search_knowledge`; it can search
with only `query`, then use returned hit metadata to refine later searches. Apps
that rely on non-default namespaces or strict project/user labels should make
that scope part of the registered agent/tool configuration or instructions.

Apps can also use `KnowledgeInjectionPolicy` when knowledge should be recalled
automatically before a model call instead of only through explicit tools. The
policy wraps another context policy, searches the active environment's
`knowledge_store` with the latest user message and configured filters, then
injects bounded snippets as model-facing synthetic user context before the latest
real user message. It does not append those snippets to the durable transcript.
It emits `knowledge.search.started`, `knowledge.search.completed`,
`knowledge.search.failed`, and `knowledge.injected` events for audit/debugging.
Search failures are fail-open by default; configure `fail_open=False` when a
missing knowledge lookup should fail the session before the provider request.

This slice does not add graph retrieval, remote source connectors, background
remembering workers, or agent-led mutation of existing knowledge. Those layers
should build on the same `KnowledgeEntry` / `KnowledgeChunk` / `KnowledgeQuery`
/ `KnowledgeIndexer` / `TextEmbeddingProvider` contracts rather than
introducing separate memory, skill, or document-store APIs.
