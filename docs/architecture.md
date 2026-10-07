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

`_application_registration.py` owns validation of agent/environment declarations
and provider model patterns, registered-tool validation and copying, and tool
descriptor construction. These functions take explicit inputs and can be used
without loading `CayuApp`. Agent registration, MCP refresh and public inspection
compose the same functions through the registry described below. Environment
and provider registration also reuse these validators.

`ApplicationAgentRegistry` owns agent declarations, their thinking-source
metadata, MCP source claims and the lock used to publish refreshed catalogues.
It also releases static and refreshable MCP sources after in-flight work drains,
and tracks notification refreshes still settling after release.
Registration claims every source before publishing the agent; refresh validates
every affected agent before replacing the shared catalogue. It composes the
registration validators above. `CayuApp` captures the public registration site,
supplies configuration defaults and keeps application admission around refresh.
The registry shares the application admission gate so shutdown prevents new MCP
claims and waits for admitted work before releasing existing claims. Notification
refreshes enter through that same admission boundary. Manifest and
isolated replay readers use the registry's current publication; replay installs
its isolated declarations without claiming live MCP sources. The registry can
also be composed directly without importing the application or execution
controllers by supplying an admission gate and its other explicit dependencies.

`ApplicationProviderRegistry` owns validated provider declarations, the default
provider and model-pattern matching. Registration snapshots routing patterns,
usage dialect and secret-free execution identity while retaining the live
provider for execution. `CayuApp` supplies the public registration provenance
and delegates registration and lookup. Manifests and scaffolding read the
registry; manifests use its matching rules when describing ambiguous routing.
Isolated replay installs copied declaration maps with recorded providers while
preserving admitted identities and the default. The registry composes directly
with a secret redactor, without importing application or execution controllers.

`ApplicationEnvironmentRegistry` owns environment and factory declarations,
explicit default selection, artifact-store identities and the session-closure
inventory built from those declarations. Both registration paths validate the
prospective closure coordinator before publishing any environment, artifact
store or default. Shared artifact stores remain deduplicated by identity;
qualified closure adapter identities and knowledge-before-artifact ordering
are preserved. Metadata inspection copies declarations without materializing
factories. The component composes directly with a session store, redactor,
clock, optional knowledge store and closure adapters. `CayuApp` supplies those
dependencies and public registration provenance. Runtime materialization,
idle-resource release and shutdown keep their existing owners.

`_application_task_creation.py` owns task creation and work-contract publication
and lookup. Its operations take explicit task-store, session-store and redactor
dependencies, without importing the application or execution controllers. They
validate requests, resolve invocation provenance and authenticate store results,
including exact contract identity and scheduled creation replay. Ordinary tasks
remain usable without work contracts. Contract-bound mutations reuse the existing
cancellation-quiescent store boundary; native transactions and scheduling keep
their existing owners. `CayuApp` retains the public signatures and lifecycle
tracking and releases request references before awaiting the operation so
rejected sensitive input is not retained in its traceback.

`_application_context_views.py` owns completed-view publication, ownership
transitions, selection and authenticated readback. Its functions receive the
session store, participant coordinator and session-identity resolver explicitly.
Publication composes the existing historical projection and resource owners;
views without resources need no environment or agent resolvers. Native stores
retain snapshot capture, exact replay and transaction checks. `CayuApp` supplies
registrations and projection hooks and keeps lifecycle tracking around each
complete operation. The component can be used without constructing an application
or loading execution controllers.

`_application_accounting.py` owns session and causal-budget usage/cost readback
with explicit session-store, identity-resolution and identifier-projection
dependencies. It composes native accounting snapshots, session pagination and
the existing pricing aggregation. Stores retain incremental reads and sequence
boundaries; pricing rules retain their existing owner. `CayuApp` keeps the public
reporting methods and lifecycle tracking. HTTP usage responses compose the same
snapshot and projection functions, retaining generation/sequence evidence for
ETags without calling private application reporting helpers.

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
  -> WorkAttemptCoordinator
  -> QueuedDispatchCoordinator
```

`SessionEngine` owns run, resume, fork, explicit compaction, queued-message
delivery, task linkage, model/tool loop decisions, interruption terminalization,
turn completion, and terminal hooks. `RecoveryCoordinator` owns durable paused
continuations, manual outcome reconciliation, incomplete-session repair, and
abandoned-run finalization. Model, tool, environment, limit, control, and event
modules own their complete lower-level behavior slices.

`WorkAttemptCoordinator` owns application work-attempt admission, execution
claim checks and renewal, recovery, and proposal publication. It owns one
process-local execution identity and refreshes it after a process fork. TaskStore
retains durable claim and receipt authority; the existing engine retains session
preparation, execution and settlement behind the coordinator's typed interface.
`CayuApp` supplies current-store access, run defaults, public-session resolution
and the checkpoint guard, and keeps admission/tracking and request detachment at
its public entrances. The verified worker can still acquire an acknowledged
recovery claim, maintain its heartbeat, and resume recovery separately. The
coordinator can be composed without importing the application or concrete engine;
its collaborators provide the same execution and checkpoint contracts.

`QueuedDispatchCoordinator` owns the session side of durable queued dispatch:
request preparation, frozen profile and session-instance validation, execution
or exact terminal replay, settlement classification, and receipt acknowledgement.
`TaskStoreDispatcher` retains queue leases and task terminalization; session stores
retain atomic checkpoint mutations. `SessionEngine` provides profile resolution
and execution, and `DurableSubagentCoordinator` retains prepared-child authority.
The coordinator takes explicit collaborators and can reconcile persisted queue
receipts without importing the application or concrete engine. `CayuApp` composes
these parts and keeps dispatch admission and stream cleanup at its entrance.

`ProviderOperationCancellationOwner` owns durable provider cancellation claims,
lease renewal, cancellation evidence, accounting handoff and exact claim release.
Live execution and recovered interruption share this owner through
`ModelStepExecutor`. It composes the existing `ProviderOperationCancellationLifecycle`,
which retains process-local task admission, deduplication and shutdown ownership.
The durable owner receives explicit store, event writer, budget controller and
recovery-context reader dependencies. The reader validates the stage on each use;
the owner does not cache authority or change backend transaction boundaries.

`ProviderOperationRecoveryOwner` owns exact start recovery, reconnect/retrieval,
saved stream-progress reconciliation and recovered completion publication. Live
execution shares its progress publication boundary and the same cancellation
owner. Completion contracts, completion delivery, stream validation/event
projection and hosted tool-discovery preparation live in independent runtime
modules used by both execution and recovery. Consumers import these parts from
their owning modules; the executor composes the operations and retains the
imports it uses. The executor wires the recovery owner's store, event writer,
run-limit controller, redactor, clock and cancellation owner. Live retry decisions and
automatic compaction remain with the executor.

`ProviderOperationStartOwner` owns background-provider dispatch, exact start
identity publication, bounded cancellation settlement and late acknowledgement
reconciliation. It shares the existing cancellation lifecycle and retains its
late reconciliation tasks. Run-local admission, notification consumption and
context-exposure callbacks preserve the caller's authority; the start owner
orders these boundaries around dispatch and publication. Its per-attempt state
records observed effects even when startup raises or closes, so the executor's
stream cleanup retains the exact operation, stream and durable-identity status.

`LiveModelAttempt` owns one live model attempt from provider-stream consumption
through response construction, cancellation cleanup and completion publication.
It composes the existing startup, cancellation and recovery owners with explicit
store, event writer, session control, redactor and clock dependencies. Run-local
authority callbacks preserve their dispatch and publication order. The model-step
executor delegates the complete attempt and retains request preparation,
observation and retry/failover scheduling. `ModelStepRun` retains context recovery
and automatic compaction coordination.

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

`sessions/_checkpoint_preservation.py` owns callback-visible checkpoint copies,
protected-root preservation and the shared lifecycle/workspace authority scopes.
The runtime adapter and native stores use this same owner, which composes the
existing browser, producer, continuation and collaboration-export rules and the
schema decoder in `sessions/checkpoints.py`. Callback state stays detached and
restored private roots count toward the full document limit. Runtime admission
checks and native memory locks, SQLite writer transactions and PostgreSQL
transactions remain with their existing owners.

`sessions/_checkpoint_transforms.py` composes schema decoding, detached callback
input, output validation, protected-root preservation and version stamping. It
supports ordinary, store-time and operation-publication callbacks using the
existing session contracts. `runtime/_checkpoint_store.py` applies those shared
transforms while retaining runtime dispatch and store capability checks.

`sessions/_checkpoint_publication.py` owns publication request stamping and
checkpoint decode, encode and writer-schema mutation projection. It reuses the
canonical publication records and decoder; the runtime adapter selects this
composition through the existing task-local codec scope. Native stores retain
the atomic publication, validation and persistence boundary.

`providers/retry_policy.py` owns immutable retry configuration, its default status
codes and the policy validation helper. Saved tool rounds, approvals and runtime
share the same policy class. Supported root and runtime imports resolve to that
class; retry classification, suppression, backoff and execution remain with their
existing runtime owners.

`execution_units.py` owns shared model-step, model-attempt, tool-round and
budget-limit identities, including ID generation, validation, copying and removal
of caller-supplied authority fields. Checkpoint readers, approvals, budgets, native
stores and runtime use the same records without importing execution owners for
these contracts. Supported root and runtime imports resolve to those same objects;
runtime dispatch and native transactional validation retain their current owners.

`_event_schema.py` owns event payload policies, the shared schema registry and
private linkage reads. It can inspect event identity without importing execution
or store owners. Pending-action evidence and runtime projection share its registry.
Retry-decision data lives in `providers/_retry_decision.py`; workspace path records
and their schema fields live in `workspaces/_revision_records.py`. Supported public
imports resolve to those same classes. Tool-result schema metadata lives beside
its attestation code in `tools/_shared_artifact_result_schema.py` and
`tools/_web_access_result_schema.py`. Runtime retains event redaction, publication,
delivery and retry execution; tool-result attestation retains its authority checks.

`sessions/_completion_finalization.py` owns the saved cleanup marker key, reader
and validation, including detached results and the encoded byte limit. Session
queue publication and runtime recovery share this component without importing
environment lifecycle execution to read the marker. Reading validates evidence;
native stores retain transaction and authority checks, and runtime retains
environment allocation, cleanup and recovery decisions.

`sessions/_tool_call_evidence.py` owns the shared event scan used by pending-action
queries and runtime recovery. It matches pending calls, classifies starts and
terminal evidence, and detects conflicting or manually reconciled history.
Callers supply scope predicates and terminal validation; runtime retains outcome
reconstruction and recovery decisions. `tools/_argument_publication.py` owns the
argument quarantine and projection rules shared by this scan, approval records,
runtime publication and evaluation replay. Both components work without session
stores or execution owners.

`sessions/_assistant_tool_round_publication.py` owns saved assistant publication
state, staged tool-terminal records, and their identity, timing and exposure
validation. Approval and pending-round checkpoints share these records with
runtime staging and recovery. `tools/_policy_evidence.py` owns their tool-policy
evidence classification. Both components work without execution or store owners;
runtime retains publication, hook execution and secret-scope resolution.

`budgets/run_limits.py` owns lightweight run-limit configuration and copy/presence
checks. `budgets/_run_limit_accounting.py` owns durable usage/time origins,
run-budget authority records, and the shared pause/resume/rebase validation rules.
Checkpoint records and runtime use the same types and bounded accounting data.
These rules work without runtime clock adapters, stop-policy evaluation, execution
owners or session stores. `runtime/_run_limit_accounting.py` captures and restores
the live monotonic clock origin; `runtime/stop_policy.py` retains stop decisions
and auxiliary admission checks. Public stop-policy imports retain their identity.

`sessions/_pending_tool_round.py` owns the saved pending-round record, checkpoint
key, identity helper and shared owned-JSON validation marker. Its validators work
without runtime execution or session stores. `sessions/_pending_tool_round_reader.py`
owns checkpoint loading and parsing, using the same marker after taking ownership
of a complete durable JSON snapshot. It returns fresh checkpoints and detached
rounds without a second nested copy. Checkpoint secret validation lives in
`sessions/_checkpoint_secret_validation.py`; web-access and shared-artifact result
controls live in `tools/_web_access_results.py` and `tools/_shared_artifact_results.py`.
Their exact attestation and persisted-control checks are shared by checkpoint
validation and runtime result handling. Checkpoint publication, secret resolution
and recovery execution retain their runtime owners.

`sessions/_pending_approval_reader.py` owns saved approval parsing, paired-round
scope checks and the pure projection/evidence helpers used by pending actions.
Session inspection and runtime recovery use the same reader and canonical approval
models. Approval resolution, checkpoint writes, live policy evaluation and event
publication remain with their existing owners.

`sessions/_staged_tool_terminal_reader.py` owns saved terminal-result reads and
recovery-safe projections. Pending actions and runtime recovery share its ordering,
detachment and incomplete-secret-scope handling. Checkpoint reads validate one
ordinary or user-input round owner from a fresh snapshot. The shared validator in
`tools/_terminal_controls.py` preserves typed terminal controls and redacts failure
evidence. These components work without runtime recovery or result-processing
owners; checkpoint writes, hooks and publication keep their existing owners.

`sessions/_foreground_child_checkpoint.py` owns saved child waits, terminal
selections and continuation records, plus their readers and close projections.
It shares the canonical effect identity in `sessions/_tool_effect_intent.py` and
the saved approval-resolution intent in `sessions/_pending_approval_reader.py`.
These components work without the runtime wait, effect-state or approval owners.
Runtime still authenticates live parent/child authority, resolves actions and
executes continuations; storage retains atomic publication and fencing.

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

`sessions/_producer_checkpoint.py` owns producer attachment/index records,
checkpoint visibility and projection, reserved-operation guards, and the shared
publication scope. `_producer_cleanup_contract.py` owns the immutable cleanup
receipt used by the index and runtime readback. Runtime publishers and store
guards use the same definitions and scope; the former runtime imports remain
compatible. Admission, output publication, cleanup and recovery execution retain
their existing owners and native transaction boundaries.

`execution_profiles.py` owns shared profile identities, decision records,
validation, comparisons and pure identity projections. Shared operator identities
and redacted audit payloads live in `approvals/actors.py`. The session component
`sessions/_execution_profile_checkpoint.py` owns persisted profile records and
metadata/checkpoint readers and writers. Both work without runtime or store
implementations. Runtime composes these components for profile construction and
admission, and retains policy execution, diagnostics and decision attestation.
Public exports and the former runtime imports resolve to the same definitions;
native stores retain their existing atomic publication and fencing checks.

`sessions/_invocation_lifecycle.py` owns invocation command and result values,
receipt ledgers, exact replay validation, and checks against transaction-owned
session state. `sessions/authority.py`, `_durable_operation_ownership.py`, and
`invocation_release.py` own the shared fencing values, operation-ownership rules,
and portable release evidence. Runtime retains live invocation context, cleanup
admission, command dispatch, and receipt publication that composes continuation
state. Both layers share the same release-authority tokens and validated command
copies. Existing imports remain compatible, and native stores retain their
transaction boundaries. The lifecycle contracts still use session models.

`sessions/_session_continuation.py` owns durable continuation records, limits,
identity and digest helpers, and deterministic validation.
`sessions/_temporary_continuation.py` owns temporary-service records and their
selection, transition and capacity checks. Record construction works without
loading runtime or store implementations; admission-command validation composes
the shared invocation contracts. Existing runtime imports and public exports
resolve to the same definitions. Runtime retains admission and dispatch, and
checkpoint publication; native stores retain their transaction boundaries.

`sessions/_session_continuation_scope.py` and `_temporary_continuation_scope.py`
own the shared authority contexts and store-facing checks. Runtime producers and
transactional validators use the same context objects through both canonical and
legacy imports. Runtime retains live invocation authentication and preparation,
parking, retirement and temporary-admission scope creation. Preparation validation
retains the authenticated invocation by identity and repeats its authority and
current-writer checks inside the native publication boundary. The shared module
keeps a type-only reference to that runtime context; it does not construct one.
Nested scopes, task context and cancellation preserve their existing lifetimes.

Continuation persistence rules live alongside those contracts in
`sessions/_session_continuation_store.py`, `_temporary_continuation_store.py`,
`_temporary_service_target.py` and `_side_service_preparation.py`. They own the
bounded checkpoint index, record/history comparisons, native receipt projection,
and preparation of source/receiving-session updates. Each component remains
independently importable from its session module.
The continuation store component also validates reserved operation-record keys
against their namespace, continuation, temporary-service or side-session target
records. It requires the shared authenticated publication scope and exact record
identity; validation does not grant read or execution authority.
Native stores invoke these rules inside their existing lock or transaction.
Joint preparation validates both sessions before either update is applied;
runtime dispatch and collaboration permit registration retain their own owners.

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

SQLite knowledge-schema checks live in `storage/_sqlite_knowledge_schema.py`.
Work-context, recall-delivery and recall-subscription checks live together in
`storage/_sqlite_work_context_schema.py`. Both owners use the shared catalog
readers in `storage/_sqlite_catalog.py`. Evaluation result, case, scenario,
authored-suite, calibration and run checks live in `storage/_sqlite_eval_schema.py`.
Verified-work contracts, completion verification, attempt admission and lifecycle
receipt checks live together in `storage/_sqlite_verified_work_schema.py`.
Task invocation metadata, terminal receipts, retries and interrupted handoffs
are checked by `storage/_sqlite_task_schema.py`.
Session identity, grants, deferred inputs, queued messages and child-lifecycle
checks live in `storage/_sqlite_session_schema.py`.
These owners inspect existing tables, indexes, views and constraints.
Schema reconciliation retains revision
gates and validation order in `storage/_sqlite_support.py`, alongside migration
history and execution. These domain checks can run on a read-only connection
without importing migration history or store adapters.

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

Knowledge records and access scopes live in `knowledge/records.py` and
`knowledge/scopes.py`. `knowledge/relations.py` owns exact-revision relation and
lineage contracts, detached copies, page validation and relation publication
preparation/replay validation. `knowledge/maintenance_contracts.py` owns reviewed
maintenance proposals, decisions, receipts and their deterministic preparation,
consistency and replay checks. `knowledge/activation_contracts.py` owns governance
configuration, activation requests, decisions, authority, receipts and their
deterministic preparation and validation, including bounded activation-retirement
records. Shared exact entry material and revision helpers live with records;
access-scope fingerprints and retained access snapshots live with scopes.
`knowledge/changes.py` owns change records, bounded pages, consumer claims and
progress, including detached copies, claim fingerprints and deterministic
validation and initialization. `knowledge/indexing.py` owns embedding identities
and projections, index readiness and coverage, bounded indexing outcomes, and
their deterministic construction, copying, fingerprints and transition checks.
`knowledge/search.py` owns search/list queries, hits, results and facets, with
detached copies, validation and shared search-term normalization. Ranking,
access checks and search execution remain with the storage implementations.
`knowledge/publication_contracts.py` owns revision-publication preparation,
receipts, request fingerprints and replay validation. It composes record, scope
and activation contracts without reading ambient authority or writing storage.
Retained asynchronous publication tasks remain owned by `knowledge/_publication.py`.
These contracts, application activation policies and the maintenance
router/planner can be used without loading a storage implementation.
`knowledge/base.py` owns the `KnowledgeStore` interface, default scope handling
and optional-operation refusals. Custom stores can implement it without loading
a built-in backend. Resource constraints still intersect through `knowledge/access.py`;
backend access checks and atomic persistence operations remain with each store.
`storage/knowledge_memory.py` owns the ordinary in-memory knowledge backend,
including its list facets and update timestamps. The optional embedding subclass,
stored vectors and similarity helpers live in `storage/knowledge_embedding_memory.py`.
It inherits the ordinary backend and composes the same knowledge contracts and rules.
`storage/memory.py` retains the existing import surface. Shared change-time and
operation-identity validation live in `knowledge/changes.py` and `knowledge/records.py`;
the governance metadata key belongs to `knowledge/maintenance_contracts.py`.
SQL backends and knowledge services import those owners directly. The ordinary
store retains synchronous preparation and mutation without added awaits or locks.
`knowledge/_access_rules.py` owns shared authorization snapshots, change audiences
and access decisions for entries, relations, maintenance and activation history.
Memory, SQLite and PostgreSQL call these rules inside their existing storage
operations; each backend retains its transaction and mutation boundaries.
`knowledge/_revision_rules.py` owns shared revision preparation: successor
invariants, chunk construction and evidence identity and chunk remapping.
The stores compose these rules inside the same operations, including ambient
resource relabel checks, while retaining their clocks and persistence owners.
`knowledge/_activation_rules.py` owns shared approval authority and review-scope
checks, activation receipt matching, and reviewed-approval receipt preparation
and replay. The stores and review adapter compose these rules directly; clocks,
access checks and atomic writes remain inside the existing store operations.
`knowledge/_maintenance_rules.py` validates reviewed source and replacement
revisions, publication boundaries and exact source evidence, and prepares the
replacement and archived source revisions. Memory, SQLite and PostgreSQL compose
these rules inside their existing access, lock and transaction boundaries.
`knowledge/_relation_queries.py` owns relation and lineage projections, query
and access-scope fingerprints, cursor encoding and validation, and bounded result
pages. Memory, SQLite and PostgreSQL supply authorized candidates from their
existing snapshots and retain native filtering, ordering and transaction owners.
`knowledge/_embedding_backfill.py` owns backfill query and access-scope binding,
cursor encoding and validation, and the memory page-ordering key. Memory and
PostgreSQL share the cursor rules while retaining candidate selection, native
ordering, embedding-provider calls and transaction ownership.
`knowledge/_query_rules.py` owns shared entry filters and expiry checks,
semantic-query text preparation and paired knowledge/index frontier validation.
Memory composes these rules for search, listing and backfill; SQLite and PostgreSQL
share frontier validation, and PostgreSQL also shares semantic-query text.
Native filtering, access checks, per-entry clock behavior and transactions retain
their existing boundaries.
`knowledge/_search_scoring.py` owns shared keyword matching and entry/title/chunk
scoring, including phrase field boundaries, exclusions and best-match selection.
Memory and PostgreSQL compose these rules with the existing query tokenization;
authorized candidate selection and native ranking stay in the stores.
`knowledge/_retrieval_results.py` builds bounded search hits, chunk previews and
evidence from authorized candidates. Memory, SQLite and PostgreSQL share these
rules while retaining their existing read operations, native result construction
and validation boundaries. The shared rules preserve byte and item limits, rank,
score metadata, detached records and completeness flags.
The existing `cayu`, `cayu.storage` and `cayu.storage.memory` imports resolve to
the same canonical types, including persisted legacy pickle class paths.

Tasks are optional durable work items, not a required execution model. A simple agent can run with only sessions and events. A background job, orchestrated multi-agent app, webhook processor, or dashboard-visible queue can use `TaskStore` to track work status, inputs, outputs, errors, ownership, and parent/child relationships. Worker-owned completion and failure use an atomic terminal receipt so acknowledgement loss can be reconciled without applying a second task transition; that receipt covers Cayu task state, not exactly-once external effects. Interrupted-session handoff uses a separate exact receipt so transient ownership-release failure preserves the running task/session link for bounded retry or expired-lease recovery. Application-owned verified-work contracts add an independent completion-authority boundary: workers propose completion, while durable verifier claims and decisions gate the terminal transition. Every contract also freezes the exact application-owned result-resolver identity used to read the accepted result from durable application state. `CayuApp.resolve_completion_result(...)` invokes that side-effect-free resolver and feeds the validated content into the existing `apply_completion_decision(...)` receipt boundary. Receipt-first replay does not require the process-local resolver after application has committed. `InMemoryTaskStore`, `SQLiteTaskStore`, and `PostgresTaskStore` expose the same verified-work lifecycle; SQLite is the local durable implementation and PostgreSQL supplies cross-process row-lock authority.

Verified-work persistence policy and immutable-authority validation live in
`tasks/_verified_work_policy.py` and `tasks/_verified_work_authority.py`.
Durable verifier-profile records, fingerprints and profile-policy contracts
live in `tasks/completion_verifier_profiles.py`. Task stores and scheduling
code import these owners directly. The former runtime module paths forward
existing imports, including persisted pickle class names. Profile contracts
still use the execution-profile value types in `runtime/execution_identity.py`
and `execution_profiles.py`; the completion coordinators remain separate
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
