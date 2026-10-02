# Cayu Architecture

This is a design/maintainer document for Cayu's production agent runtime. It records architecture decisions and intended direction; it is not a complete end-user guide.

Cayu is a production agent runtime for building long-horizon agents, multi-agent workflows, and sandboxed tool runtimes.

The runtime should run locally, on a VPS, in Docker, in ECS, or in any other standard execution environment. Hosted deployments should be adapters around the runtime, not a requirement for using it.

Cayu executes application-owned agents directly. `SessionEngine`,
`ModelStepExecutor`, and `ToolRoundExecutor` own the agent loop; provider
adapters translate model APIs into Cayu's provider-neutral contracts; durable
events record execution; and the optional OpenTelemetry sink projects those
events into traces. Claude Code and other coding agents may author a Cayu
application. The Claude Agent SDK and other coding-agent runtimes are not
runtime dependencies or sources of execution authority.

## Core Decisions

- Repo/package/CLI name: `cayu`
- Language: Python for v1
- Cayu repository structure: horizontal by subsystem
- Generated user projects: Rails-like default layout, vertical domain modules allowed
- CLI: developer/admin utility, not the primary product interface
- Dashboard: optional viewer over runtime events and session storage
- MCP: interoperability layer, not the required custom tool model
- Runtime model: separate agent, environment, and session concerns

MCP tools should enter the runtime as normal Cayu tools through adapters. That keeps
external servers under the same policy, approval, event, and transcript model as native
Python tools while preserving MCP as an interoperability boundary.

## Dependency Direction

```text
core
  providers -> core
  artifacts -> core
  runners -> core
  workspaces -> core
  storage -> core
  vaults -> core
  proxies -> vaults + core
  egress -> proxies + vaults + core   (Docker adapter also uses runners)
  mcp -> core
  environments -> artifacts + workspaces + runners + vaults + proxies + mcp
  runtime -> core + providers + artifacts + runners + workspaces + storage + vaults + proxies + mcp
  workflows -> core + runtime
  cli -> runtime + project scaffolding
  dashboard -> runtime API / event store
```

`core` should stay small and stable. It defines events, messages, agents, tools, the abstract workflow contract, and shared value objects.
`workflows` contains orchestration-as-code helper primitives layered above the runtime; it may depend on runtime session/event contracts, but runtime should not depend on workflow helpers.

## HTTP route ownership

`server/routes.py` composes the HTTP routers and owns the durable eval worker
lifespan. `server/_captured_evaluation_routes.py` owns target catalog and captured
evaluation preview, save, export and launch. Its separate launch registrar keeps
durable launch conditional on an eval runtime and preserves route order. Launch
reuses the current-candidate validator shared with save/export and composes the
existing promotion, scoring and run-admission functions. Reviewed captured
evidence is published before a fresh run is admitted; accepted retries reuse
the stored run and result record before renewed execution-profile checks.

`server/_memory_report_routes.py` owns memory experiment report
readback, exact stored-result and execution-profile validation, construction,
and JSON/HTML responses. Its registrar receives the bounded report router,
eval store, runtime target registry, and shared authentication dependency.

The specialized report route class retains authentication before parsing,
private request byte limits, redacted errors, cache headers, and request-schema
publication. Both composition and report handlers use `server/_http_json.py`
for the same parsed-body identity and JSON validation/rendering helpers. For
each published result, the owner reads the run, result, and run again before
accepting evidence, so a changed durable run cannot produce a report from
inconsistent readback.

`server/_judge_calibration_routes.py` owns fixed-evidence calibration preview,
judge execution, durable report publication and retrieval. It receives the
bounded eval router, runtime store and target registry, and shared auth
dependencies. Each registration creates its own striped run-ID locks; a
calibration checks for a stored run, validates current judge authority, executes
trials and saves the report while holding its lock. Repeated requests reuse
the stored report, and conflicting definitions retain their existing rejection.
The eval worker lifespan remains in router composition.

`server/_suite_authoring_routes.py` owns authored suite preview, save, catalog,
detail and download. Its registrar receives the bounded eval router, runtime
store, target registry and shared auth dependencies. Authoring diagnostics check
current judge authority, public material and exact scenario references; repeated
scenario revisions share one read, with at most 16 reads in flight. Authoring
reads and run launch use the same suite revision loader to enforce persistence
support, private storage errors and current target visibility.

`server/_suite_launch_routes.py` owns authored-suite launch preview and start,
with explicit store, registry and shared authentication inputs. It composes
the existing selection, scenario preflight, execution-profile, exposure and
admission functions. Preview prepares at most 16 scenario cases concurrently.
Fresh launches prepare every part and publish all derived corpora before
ordered admission. Complete retries reuse the admitted runs; partial retries
retain the same request identity and require current launch readiness before
continuing. Cost-budget narrowing remains with this workflow. Other launch
families and the eval worker lifespan retain their current owners.

`server/_scenario_routes.py` owns scenario preview, save, catalog,
detail, download, artifact fixture preparation and launch. Authoring and launch share
the scenario revision loader and preflight boundary with explicit store and
target-registry inputs. Fixture preparation checks the reviewed revision,
copies the selected retained bytes into an environment fixture, clears that
temporary artifact selection and checks readiness against the updated scenario.
It returns the new scenario revision without saving it. Separate authoring and
launch registrars preserve route order and compose existing library functions.
Launch preserves the reviewed binding, exact execution profile and execution
bounds, publishes the derived corpus before admission, and replays accepted
requests before renewed readiness checks. Router composition retains the shared
bounded/auth boundary and the durable eval worker lifespan.

`server/_corpus_routes.py` owns corpus import, catalog, detail, download,
suite/case browsing and launch. Its registrars receive the bounded router, store,
target registry and shared auth dependencies, and compose the existing corpus
validation, serialization and storage operations. Catalog reads and run creation
share one revision loader that enforces current target visibility and private
storage errors. Separate management and launch registrars preserve route order.
Launch uses the shared JSON, invocation, request binding, retry, execution-profile
and admission functions. Accepted retries return before renewed readiness checks;
fresh runs validate profiles and execution bounds before admission.
Router composition retains the shared bounded/auth boundary and worker lifespan.

`server/_evaluation_run_routes.py` owns durable run listing, detail, scenario
approval, cancellation, results, report downloads and comparisons. It receives
the bounded router, store, target registry, optional result catalog and shared
auth dependencies. Run visibility is checked before result loading, and the run
is reloaded after publication becomes visible. Reporting and comparison compose
the existing library functions. Router composition registers the evaluation
route families and retains the worker lifespan.

`server/_eval_run_admission.py` owns the shared HTTP invocation, request identity,
retry lookup, execution-profile preparation and durable admission functions.
Corpus, authored-suite, scenario and captured-session launch handlers compose
these functions with explicit store or registry inputs where required. An
accepted retry is resolved before preparing new work. Fresh work retains the
published and effective profile checks, execution bounds and target redaction
before persistence. Each step remains separately callable; family-specific
preflight, publication and budget narrowing stay with the launch handlers.

## Runtime Shape

```text
RunRequest
  -> SessionStore creates session
  -> Environment provides execution context
  -> Agent runtime streams provider/tool events
  -> EventSink emits to terminal/dashboard/webhook
  -> SessionStore persists append-only event log
```

Every important action should produce an event. Events are the shared contract for debugging, dashboards, hosted integrations, replay, and tests.
Event identity fields such as agent, environment, workflow, and tool should be top-level event fields so event stores and dashboards can index them without parsing payload JSON.

Runtime inputs are copied at runtime boundaries. Runtime code should depend on explicit registration and validated contract objects, not on later mutation of user-owned Python objects.

JSON-like contract fields should remain portable across local, hosted, and remote execution. They should contain JSON-compatible values only, without Python-specific object identity, circular references, or special numeric values such as NaN and Infinity.

### Runtime ownership

`CayuApp` is the stable public façade and composition root. It validates public
requests, owns registrations and configuration, constructs the runtime
collaborators, and delegates execution. Deep runtime modules do not import or
accept the complete application object.

```text
CayuApp
  -> RuntimeEventWriter
  -> SessionControl
  -> RunLimitController
  -> EnvironmentLifecycle
  -> ModelStepExecutor
  -> ToolRoundExecutor
  -> DurableSubagentCoordinator
  -> RecoveryCoordinator
  -> SessionEngine
```

`SessionEngine` owns run, resume, fork, explicit compaction, queued-message
delivery, task linkage, model/tool loop decisions, interruption terminalization,
turn completion, and terminal hooks. `RecoveryCoordinator` owns durable paused
continuations, manual outcome reconciliation, incomplete-session repair, and
abandoned-run finalization. Model, tool, environment, limit, control, and event
modules own their complete lower-level behavior slices.

`TerminalEvidenceFinalization` owns terminal-evidence inspection and crash repair,
as well as live claim transfer, exact renewal, heartbeat-monitored preparation and
streamed completion. Recovery and the engine share one instance. It borrows
recovery's existing claim supervisor, cleanup supervisor and worker registry, so
closing an observer cannot release a claim while its work is still active. Repair
retains exact event identity, redaction and marker cleanup. Approval/round checks
come from `_approval_support.py`; user-input authority and recovery claim
acquisition remain with recovery. Shared claim records live in `_recovery_claims.py`.

Request and interrupted-run lifetimes in `_terminal_finalization_lifetime.py` own
preparation, claim handoff, heartbeat shutdown and settlement. They reuse
SessionControl's task-bound handoffs and select execution under the existing
supervisor. Borrowed work runs inside its original recovery worker and cannot
release that worker's claim. The engine supplies interruption policy and terminal
publication; the lifetime supplies the session authenticated by its final renewal.
Evidence inspection and claimed crash repair remain independently usable.

`ToolRoundExecutor` delegates ordinary tool-round publication to
`DurableToolRound`. The owner reserves capacity for private terminal stages before
dispatch, retains the round's lifecycle evidence, and publishes staged results in
model order after secret scopes are sealed. It then commits the transcript and
checkpoint together, retaining the exact request for acknowledgement-loss replay.
The execution caller supplies policy decisions, tool execution and result hooks.
It does not prepare the final publication request or manage individual stage
leases. Closing the ordinary publication stream closes its active terminal hook
stream before returning.

`SessionEngine` also delegates ordinary round closure after a run limit to this
owner. It retains completed effects, publishes skipped results for unstarted
calls, and commits the round using the same publication operation. A repeated
closure reads the existing receipt and transcript. Cancellation remains observable
after publication and deferred input materialization. Only live execution creates
dispatch state.

`RecoveryCoordinator` delegates interruption snapshot validation, missing-result
selection and recovered round publication to the same owner. Interruption captures
the transcript cursor before settlement; publication retains that cursor as its
concurrency fence. The owner completes the assistant's secret projection, restores
staged capacity, publishes safe terminals, and commits through the shared
exact-replay operation.
Recovered terminal events reach the caller after commit and deferred input
materialization. An error while consuming a terminal hook closes that hook stream
before returning. Missing-result selection skips recorded and staged calls,
retains blocked results for unexposed calls, and synthesizes unknown outcomes
only after reconciliation permits closure. The recovery coordinator supplies
one per-call resolver for native operations, external-effect journals and child
sessions. It also coordinates workspace settlement and isolated dispatch evidence.
An uncertain external effect retains its reconciliation fence. Selection and
publication use the same round owner without authorizing a second tool execution.

Execution, continuation and recovery share `load_pending_tool_round` in
`_tool_round_recovery.py` for fresh loads followed immediately by round parsing.
It returns the exact
checkpoint snapshot and its detached validated round together, so publication
uses the input it checked. Each call performs one store read with the caller's
current redactor, rejection-consumption policy and runtime-session provenance;
it retains no cache. Checkpoint transforms, reads with deadline handling, and
callers already holding a shared or copied snapshot keep using the synchronous
parser at that same boundary.

Live and recovered structured-output tool rounds also publish through
`DurableToolRound`. Live execution validates the original provider arguments and
checks the result against the durable validation snapshot; recovery uses that
recorded snapshot. The owner binds validation and retry events to the same atomic
transcript/checkpoint publication. It retains the distinct live and recovery event
order and returns live cancellation to session control before auxiliary events
reach the caller.
The session engine owns model-step limits, retry scheduling and session completion.

Approval and user-input continuations use the owner's private
`ToolRoundContinuation` phase. It restores reserved capacity, retains per-call
hook modes, seals secret snapshots, fences stages left by earlier attempts and
publishes terminals in model order. The phase preserves the distinct paused-round
staging rules. Closing its publication stream closes the active result stream
before returning. Unpublished durable stages retain their capacity for recovery.
The recovery coordinator supplies approval decisions and answers, dispatches
authorized calls, and owns the exact pending-action closure and subsequent
session continuation. These closures retain their approval/input receipts and
authority checks.

The owner reads fresh checkpoint state for each observation and uses the same
source snapshot when preparing final publication. It does not cache validation
across checkpoint writes. Shared staging, projection and receipt algorithms keep
their existing authority and cancellation checks. The recovery coordinator no
longer constructs the private staging coordinator. Session-level interruption,
manual outcome reconciliation and approval resolution retain their existing owners.

`DurableSubagentCoordinator` owns the staged parent seed, child-session, queue-task,
receipt, and restart-reconciliation handoff for task-backed subagents. The application
facade and task dispatcher reach it only through narrow preparation, settlement, and
acknowledgement operations.

Some collaborators need to call session orchestration but are constructed before
`SessionEngine`. `CayuApp` supplies those edges as narrow typed callables; the
internal modules never type against or depend on the complete façade interface.
This keeps dependency direction explicit while allowing `CayuApp` to remain the
single composition root.

### Session checkpoint evidence

`sessions/checkpoints.py` owns root checkpoint decoding and schema migrations.
The adjacent private modules `_model_completion_publication`, `_terminal_evidence`,
`_invocation_terminal_decision`, and `_provider_operation_cancellation_claim` own
the shared records, validation, and terminal-event classification used by sessions,
storage, and runtime. These components can be imported without loading runtime.
Runtime retains execution, recovery, and publication orchestration; its former
evidence module paths forward to the session owners for compatibility.

`sessions/_browser_control_checkpoint.py` owns browser-control checkpoint
visibility, exact mutation and close authority, protected-root projection, and
receipt validation. Runtime publishers and store guards share its read and
mutation scopes. The former runtime path forwards to the same definitions and
scope state. Operator authentication, browser I/O, invocation admission and
publication orchestration remain with their existing owners; the checkpoint
rules can operate independently of runtime.

## Multi-Agent Shape

Cayu must support systems where multiple agents collaborate through shared state.

```text
Agent A
  -> writes record/task/event to SharedState
  -> trigger starts Agent B
  -> Agent B claims task and writes result
  -> orchestrator reviews, retries, escalates, or delegates
```

This requires both deterministic orchestration and LLM orchestrator agents.

## Workspace, Runner, Sandbox

Cayu follows an agent/environment/session separation:

- `Agent`: an `AgentSpec` — model, system prompt, and metadata. Tools, explicit complete-source MCP registrations, model-visible tool exposure, optional provider-neutral catalogue discovery, and tool authorization are attached separately at `register_agent(spec, tools=..., mcp_toolsets=..., tool_exposure_policy=..., tool_discovery_mode=..., tool_policy=...)`.
- `Environment`: workspace, artifact store, runner, vault, credential proxy, MCP servers, and execution metadata.
- `Session`: one run of an agent in an environment, with messages, status, events, and checkpoints.

The `*Spec` types (`AgentSpec`, `EnvironmentSpec`, …) are the portable, serializable core of a declaration; live objects — tools, workspaces, runners, providers — are attached at construction or registration, not stored on the spec.

- `Workspace`: active filesystem an agent can work with, such as a target repo or working directory.
- `ArtifactStore`: uploaded/generated durable file references scoped to a session or environment.
- `Runner`: executes explicit `ExecCommand` values in a workspace or sandbox.
- `ProcessIsolatedTool`: reconstructs one explicitly declared trusted adapter in
  a disposable POSIX process session when hard wall-clock liveness is required.
  This is a process lifecycle boundary, not a filesystem, network, credential,
  privilege, or hostile-code sandbox.
- `Sandbox`: isolated workspace plus runner plus lifecycle and limits.

Workspaces and artifacts are separate on purpose:

- Use the workspace for mutable work: cloned repositories, temporary files, generated outputs, editable documents, test results, and command-line processing.
- Use the artifact store for durable file objects: original uploads, stable snapshots, final outputs, evidence, attachments, and files that must survive replay/fork/resume independently of the current workspace state.
- Use Git inside the workspace for code repositories when the Git remote is the source of truth. A coding agent should usually clone into the sandbox workspace, edit there, and only commit/push through explicit user policy.
- Copy artifacts into the workspace only when a tool or script needs a path-backed mutable file. Store workspace files back as artifacts only when a generated/edited result should become durable output.

There is no implicit bidirectional sync between artifacts and the workspace. Copies are explicit one-way operations.

Model-facing file reads should persist artifact references, not provider-specific file payloads. The runtime resolves those references from the active artifact store immediately before provider calls, and provider adapters translate them into Anthropic/OpenAI/etc. native file/image/document content. Built-in context policies strip older native attachment references from provider-facing history while keeping transcript summaries, so file-heavy sessions do not resend the same bytes indefinitely. This keeps transcripts portable while still allowing multimodal providers to inspect images and PDFs.

`LocalRunner` is not a sandbox. It is only a development or already-disposable-environment execution backend.

Process commands should use argv form. Shell execution should be an explicit mode, because hosted runners need to enforce quoting, limits, logging, and security consistently.

## Storage and Memory

Files are good source-of-truth for prompts, instructions, workflows, manuals, skills, and human-reviewed memories.

Databases/indexes are better for sessions, event logs, high-volume memories, permissions, embeddings, search, and hosted multi-user state.

Default local strategy:

```text
files for human-readable source
SQLite for sessions, append-only events, transcripts, checkpoints, and indexes
SQLite FTS/BM25 for default keyword retrieval
provider-neutral embeddings plus in-memory semantic retrieval for demos/tests
backend-specific durable vector indexes later
```

The local durable session store is `SQLiteSessionStore`. New projects conventionally share `data/cayu.db` across Cayu's SQLite-backed runtime stores; applications may select another path explicitly. It keeps the event log append-only, but stores indexed identity columns beside the JSON event payload so dashboards and replay tools do not have to scan transcript files. Session records also persist provider, active model, runtime, agent, environment, and a redacted typed execution-profile identity so resume cannot silently adopt changed runtime, provider target or adapter, durable instructions, context selection, knowledge injection, compaction, provider request controls, application or invocation budgets, structured output, finalization, direct-tool schema, application-declared implementation behavior, registered or invocation policy order, hook order, environment/runner semantics, grant baseline, or effect authority. New sessions start from the agent's default model and freeze that profile in the creation transaction. Resume compares the candidate again inside its status/checkpoint transaction; mismatch evidence contains fingerprints and changed component classes, never raw prompts, schemas, code, policy bodies, or credentials. Model-attempt, usage, cost, structured-result, compaction, budget, tool, approval, runner, hook, environment, workspace, credential-proxy, and virtual-egress evidence references the immutable invocation-profile fingerprint that governed it. An explicit clean-boundary `ModelTarget` adoption atomically updates provider/model identity and the expected profile with its run epoch, checkpoint, interaction admission, portable-projection marker, and `session.model.switched` event; generic retry never changes targets silently. The store keeps the immutable transcript used by resume and checkpoint-backed context compaction, while the projection marker excludes pre-switch provider and thinking state only from later model-facing requests. Storage APIs support filtered session listing, filtered event queries with durable sequence cursors, transcript loading, atomic status/checkpoint/target transitions, and atomic batched event appends. JSONL is better treated as an export/debug format than as Cayu's primary runtime store.

Context policies are runtime projections over transcript messages, not storage. They let applications customize the model-facing conversation history by trimming, compacting, replacing bulky tool results, or injecting retrieved context while preserving the raw durable transcript for audit, debugging, resume, and future compaction.

Tasks are optional durable work items, not a required execution model. A simple agent can run with only sessions and events. A background job, orchestrated multi-agent app, webhook processor, or dashboard-visible queue can use `TaskStore` to track work status, inputs, outputs, errors, ownership, and parent/child relationships. Worker-owned completion and failure use an atomic terminal receipt so acknowledgement loss can be reconciled without applying a second task transition; that receipt covers Cayu task state, not exactly-once external effects. Interrupted-session handoff uses a separate exact receipt so transient ownership-release failure preserves the running task/session link for bounded retry or expired-lease recovery. Application-owned verified-work contracts add an independent completion-authority boundary: workers propose completion, while durable verifier claims and decisions gate the terminal transition. Every contract also freezes the exact application-owned result-resolver identity used to read the accepted result from durable application state. `CayuApp.resolve_completion_result(...)` invokes that side-effect-free resolver and feeds the validated content into the existing `apply_completion_decision(...)` receipt boundary. Receipt-first replay does not require the process-local resolver after application has committed. `InMemoryTaskStore`, `SQLiteTaskStore`, and `PostgresTaskStore` expose the same verified-work lifecycle; SQLite is the local durable implementation and PostgreSQL supplies cross-process row-lock authority.

Verified-work persistence policy and immutable-authority validation live in
`tasks/_verified_work_policy.py` and `tasks/_verified_work_authority.py`.
Durable verifier-profile records, fingerprints and profile-policy contracts
live in `tasks/completion_verifier_profiles.py`. Task stores and scheduling
code import these owners directly. The former runtime module paths forward
existing imports, including persisted pickle class names. Profile contracts
still use the execution-profile value types in `runtime/execution_identity.py`
and `runtime/execution_profiles.py`; the completion coordinators remain separate
orchestration owners.

`verification/_verified_completion.py` owns completion composition for one
application lifetime. Public completion methods and verified workers share
its verifier, decision-application and result-resolver instances. Worker
settlement calls those phases directly; admission and invocation-release
evidence remain explicit session-execution dependencies. Prepared authority
and exact verifier settlement handles belong to each operation, rather than
being cached on the shared owner.

`cayu.verification` contains the worker, verifier/resolver adapter contracts,
completion phases and shared leased-adapter machinery. Its lazy public API
exposes the individual adapters and worker; applications can still call verify,
resolve and apply independently. The existing root/runtime exports and three
public runtime submodules forward to the same canonical definitions. Stored
pickle globals using those older public paths remain readable.

Application, HTTP/CLI and coding-product composition may depend on verification.
Runtime, storage, sessions and tasks do not import its implementation; package
contract tests enforce that boundary, including imports through public aliases.
The runtime forwarding modules and root/runtime export declarations are explicit
compatibility exceptions. Verification may use existing runtime execution
services; task contracts remain below both layers. The existing coding-product
composition continues through supported compatibility imports.

Completion verifier and result-resolver coordinators use one private
`LeasedAdapterRunner` for process-local execution ownership. It holds
single-flight lock lifetimes, admission capacity, captured callback tasks,
heartbeat tasks, and exact retained drain identities. Single-flight covers
durable admission through publication. Capacity remains reserved through claim
or publication settlement, including callbacks that outlive cancellation or a
timeout. A stale drain acknowledgement cannot consume a successor's drain.
Verified worker callbacks use the same owner to settle work and its heartbeat
before returning. The coordinators retain durable authority, registration,
lease renewal policy, result validation, and credential-safe diagnostics.

Retained drains share one settlement record. The runner creates or adopts its
exact cleanup task once, observes its returned failure once, and automatically
clears only a successful matching drain. Failed cleanup stays available until
its owner acknowledges it. Lease policies retain their capacity-release boundaries.

Claim and publication heartbeats share a private renewal driver. It owns
the renewal clock, exact renewal task, and first ownership-loss notification,
retaining an in-flight renewal through timeout or cancellation settlement.
Lease adapters retain authority validation, acknowledgement checks, and
diagnostic policy. Verification shutdown cancels its renewal; publication
shutdown waits for its mutation. Publication waits also recheck a deadline
extended by a concurrent foreground renewal.
