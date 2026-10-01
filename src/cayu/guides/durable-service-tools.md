# Durable service-backed tools

For a complete installed-package support agent with restartable clarification and
external approval receipts, run `cayu guide order-support`.

Use this recipe when a custom tool wraps an application-owned service and must
resume a saved session in a replacement process. For the complete proposal,
approval, action, verification, and recovery lifecycle, keep using
`cayu guide durable-operations`. This guide supplies the component identity and
knowledge bindings needed when reconstructing that application.

## Two separate bindings

| Configuration | What it establishes | What it does not establish |
| --- | --- | --- |
| `SQLiteSessionStore` or another persistent session store | Durable conversation, checkpoints, and pending actions | Equivalent tool, policy, or environment behavior after rebuilding the app |
| `ExecutionProfileBehaviorIdentity` | An explicit versioned declaration of component behavior | Persistence, authorization, or downstream idempotency |
| `CayuApp(knowledge_store=store)` | Application-level knowledge configuration | An ordinary tool's `ctx.knowledge_store` |
| Injected bound store, or selected environment's store and scope | The knowledge service the tool can actually query | Authority outside the supplied scope |

A direct `tool.run()` test misses runtime registration and admission. Test a
real `CayuApp`, persist a session, and reconstruct the app in a second OS
process. An undeclared opaque service object may make the reconstructed
execution profile incompatible even when its Python class and tool name match.

## Declare component identity

```python
from cayu import ExecutionProfileBehaviorIdentity

reader_identity = ExecutionProfileBehaviorIdentity(
    name="handbook-reader",
    behavior_version="1",
    implementation_version="1",
)
```

Place this declaration on `ToolSpec.execution_profile_identity`. A custom
policy returns its own declaration from the
`ToolPolicy.execution_profile_identity` property. A registered custom
environment declares its own `EnvironmentSpec.execution_profile_identity`.
Declare each behavior-bearing component, not just the tool:

```python
from cayu import EnvironmentSpec, ToolPolicy

class HandbookPolicy(ToolPolicy):
    @property
    def execution_profile_identity(self):
        return ExecutionProfileBehaviorIdentity(
            name="handbook-policy", behavior_version="1", implementation_version="1"
        )

    # Implement authorize(request) for your application's actual authority rules.

environment_spec = EnvironmentSpec(
    name="handbook",
    execution_profile_identity=ExecutionProfileBehaviorIdentity(
        name="handbook-environment", behavior_version="1", implementation_version="1"
    ),
)
```

Keep the logical name stable. Change `behavior_version` when externally
observable semantics or relevant non-secret configuration changes; change
`implementation_version` for every implementation deployment, even when its
public contract remains the same. Encode configuration revisions in these
versions; the identity has no arbitrary configuration field. Do not put secrets,
credentials, object addresses, random UUIDs, process IDs, or request IDs here.
Ordinary service data can evolve under an unchanged retrieval contract.

This declaration asserts equivalent behavior across reconstruction. It does not
prove equivalence or make incompatible changes safe. On a legitimate profile
change, start a new session or follow the explicit execution-profile adoption
contract in `cayu guide anatomy`; never automatically adopt on mismatch, disable
checks, or reuse an old version to conceal changed behavior. Operation
idempotency keys and durable approval/round/call IDs serve different purposes;
see `cayu guide tool-effects` and `cayu guide durable-operations`.

## Inject the supplied knowledge store

Constructor injection is the smallest ordinary-tool pattern. Validate a
required dependency before registering the tool or issuing a provider request:

```python
import json
from cayu import KnowledgeQuery, KnowledgeStore, Tool, ToolEffect, ToolResult, ToolSpec

class HandbookReader(Tool):
    spec = ToolSpec(
        name="lookup_handbook",
        description="Search the authorized handbook.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        effect=ToolEffect.NONE,
        execution_profile_identity=reader_identity,
    )

    def __init__(self, store: KnowledgeStore):
        if not isinstance(store, KnowledgeStore) or store.bound_access_scope() is None:
            raise ValueError("HandbookReader requires the supplied, bound knowledge store")
        super().__init__()
        self.store = store

    async def run(self, ctx, args):
        found = await self.store.search(KnowledgeQuery(text="shipping", namespace="handbook"))
        payload = found.model_dump(mode="json")
        return ToolResult(content=json.dumps(payload), structured=payload)

```

Register `HandbookReader(the_supplied_scoped_store)` on your agent.

Use the actual supplied service. Do not construct an empty replacement store or
silently return empty evidence when a required resource is missing. A namespace
filter narrows retrieval; it is not authorization. The hosting application
derives the bound scope from trusted principal/session ownership. Keep tenant
or organization restrictions, labels, visibility, and other scope constraints;
never accept model-selected tenant identity or broaden the scope to fix a miss.
Inject a store bound to that authority at the appropriate application lifetime.

## Bind through an environment

When tools should consume environment resources, explicitly bind and select the
environment instead. Given the same supplied bound `store`:

```python
from cayu import Environment

environment = Environment(
    environment_spec,
    knowledge_store=store,
    knowledge_access_scope=store.bound_access_scope(),
)
```

Call `app.register_environment(environment)` and select
`environment_name="handbook"` in `RunRequest`.

Inside the tool, require `ctx.knowledge_store` and `ctx.knowledge_access_scope`,
then call `ctx.knowledge_store.search(query, access_scope=ctx.knowledge_access_scope)`.
An explicit scope must match the store's bound scope. Application-level
knowledge configuration alone does not populate these context fields; without
a selected environment binding they may be `None`. Built-in knowledge tools
use this environment path too; see `cayu guide references#knowledge`.

## Prove reconstruction

The repository example `examples/durable_service_tools/app.py` is credential-free
and uses public APIs with SQLite sessions and a scripted provider. Run `start`
and `resume` against one fresh directory in separate OS processes. It verifies
that the supplied store is searched, current scoped evidence reaches the
provider, forbidden records stay excluded, and the earlier conversation survives.
Use `--wiring environment` on both commands to exercise the environment path.

The example also supplies a versioned custom policy and a small SQLite receipt
tool. Its `pause`, `approve` or `deny`, and `repeat` phases exercise durable
approval reconstruction and completed-resolution behavior. This local fixture
proves only its own atomic insert/idempotency contract, not generic exactly-once
delivery to external services. Product handlers still authenticate and authorize
the resolver against exact review evidence as shown in `cayu guide durable-operations`.

Negative checks matter: a changed tool, policy, or environment version must fail
before a protected effect or provider request; an opaque undeclared tool must
remain incompatible across reconstruction. For a real service, also test
unknown external outcomes and reconciliation using `cayu guide tool-effects`.

## Reading admission errors

`app.run` creates a new session. For an ordinary next conversational turn:

```python
from cayu import CayuApp, Message, ResumeRequest

async def next_turn(app: CayuApp, existing_id: str) -> None:
    async for event in app.resume(ResumeRequest(
        session_id=existing_id,
        messages=[Message.text("user", "Next turn")],
    )):
        print(event.type)
```

An existing session may instead be running, awaiting approval/input, or require
recovery. Inspect its status and pending actions before choosing the corresponding
resolution/recovery API (`cayu guide references#sessions`). Ordinary resume does not resolve
an approval.

`ExecutionProfileMismatchError.differences` reports bounded class-level categories:
`opaque_identity` means at least one compared class has process-local identity;
`other_or_unknown` means the available evidence cannot diagnose the change.
An opaque category is evidence about identity strength, not proof that reconstruction
caused the mismatch. Declare stable behavior and implementation identities from the
first run. Adding one later does not repair an existing opaque baseline.

The persisted profile format retains aggregate digests and identity strength, not
individual member identifiers or declared versions. No durable evidence extension is
introduced here. Consequently errors cannot distinguish a changed declared version
from a member addition/removal, nor name the member responsibly. Inspect the persisted
execution-profile decision, its changed classes, and the application's declarations.
At a clean resume boundary, real behavior changes require a new session or explicit
profile adoption. Pending model or tool recovery is not an adoption boundary. Never
reuse an old version declaration to conceal a change. Adoption rejection and migration
requirements remain distinct outcomes.

### Retained sessions blocked by changed registrations

Startup isolates each incompatible interruption-cascade root, keeps its checkpoint
and unknown outcomes intact, and continues with later roots and the interrupted pass.
The stale interrupting root is planned before mutation. A hard blocker is reported
without attempting repair; a typed execution-profile rejection or a manual
model-completion recovery requirement after planning is isolated too. Store failures,
invalid cursors and unclassified errors still fail startup.
Missing registrations and invalid cascade markers are reported with fixed blocker codes.

`await app.get_startup_recovery_status()` returns the latest process-local sweep. It
keeps the integer return value of `resume_pending_interruption_cascades` and separately
reports completion, scheduled roots, total blocked roots, and at most 100 blocked session
IDs with blocker codes. No prompts, arguments, exception text, checkpoints, or component
configuration appear in the result. Truncation is explicit; inspect an individual session
with the ordinary recovery planner. Repeated startups plan blocked roots read-only and
do not append duplicate rejection evidence.

The same result is available at `GET /api/recovery/startup` for `create_server`, or
`GET /cayu/api/recovery/startup` with the default mount. Configure authenticated server
access and restrict it to operators; authentication alone does not provide tenant
isolation. This noninteractive command prints JSON from the running process:

```bash
curl --fail-with-body --silent --show-error \
  --header "Authorization: Bearer ${CAYU_OPERATOR_TOKEN}" \
  "${CAYU_OPERATOR_BASE_URL}/api/recovery/startup"
```

A supported recovery path is to restore the complete compatible application registration
in a separate operator process against the same store, with the original declared provider,
tool and policy implementations. Quiesce other owners first. Do this only when those
implementations remain available and correct; a historical opaque process identity cannot
be recreated by adding a stable declaration later. Use the restored factory to produce
and review a fresh exact plan, then execute its allowed decisions:

```bash
cayu recovery plan previous_agent:build_app --session SESSION_ID \
  --inactive-for-seconds 0 --output retained-plan.json
cayu recovery execute retained-plan.json --target previous_agent:build_app \
  --execution-id restored-registration-1
```

The recovery receipt attributes the action to the explicit execution ID. It does not
adopt the changed profile or dispatch unknown external effects automatically. If the
restored plan requires an explicit model/tool outcome decision, inspect its evidence and
supply a decisions JSON file with `--decisions`; do not edit stored events or fabricate
success. If a compatible registration cannot be restored, leave the retained evidence
intact and operate healthy sessions under the new registration.

## Human pauses and elapsed limits

`RunLimits(max_elapsed_seconds=900, scope="run")` allows 900 seconds of active
run time. Durable approval and user-input waits do not consume that allowance.
The runtime retains the pause interval and its first resolution timestamp in
existing checkpoint writes; a continuation or restart preserves token, call and
cost accounting. Work after the first decision still consumes the elapsed allowance.
Use `scope="session"` when the elapsed bound should include wall-clock human waiting,
and use approval expiry when a particular grant should expire.

Histories written by older releases retain their original elapsed accounting.
A recorded `limit_reached` approval skip can be reconciled through normal
`CayuApp.resume(ResumeRequest(session_id=..., messages=[Message.text("user", "Continue")], limits=...))` without
editing stored events. Supply the session's compatible registration and the intended
limits. A valid skip remains not executed; resume never fabricates a start or
re-dispatches the skipped effect. Conflicting started or terminal evidence remains
an explicit recovery error. See the runtime contract's run elapsed-time semantics
for the exact interval and provenance rules.

## Attributing runtime time around a tool

`CayuApp` keeps a bounded, process-local timing view by default. It contains
durations, counts, byte sizes, registered tool names and public identifiers.
It never includes arguments, results, prompts, SQL, or exception messages.
Read it after consuming the run; the newest records come first:

```python
async def print_timing(app, session_id):
    rounds = await app.inspect_recent_tool_round_timing(session_id, limit=20)
    preparation = await app.inspect_recent_model_step_preparation_timing(session_id, limit=20)
    for record in (*rounds, *preparation):
        print(record.model_dump_json())
```

The private session id or a public session reference already in the local
buffer can select these observations. This read does not access the database
or grant execution/recovery authority. The buffer is shared across sessions
and defaults to 64 records, with at most 128 calls per round. Configure bounds
with `RuntimeTimingConfig(recent_capacity=..., max_calls_per_round=...)` passed
as `CayuApp(runtime_timing=...)`; `calls_truncated` reports omitted calls.
`RuntimeTimingConfig(enabled=False)` disables collection and delivery.

A `ToolRoundTiming` has aggregate `phases` plus `calls`, each with the same
twelve phase names. Shared round work, such as planning the complete approval
set and committing the transcript, belongs to the round rather than being
charged repeatedly to each call. A zero per-call value therefore does not
mean that the shared operation did not run.

| Phase | Measured work |
| --- | --- |
| `authorization` | Tool policy, approval planning, exposure checks and final dispatch reauthorization. |
| `admission` | Pending-round/policy checkpointing and capacity reservation, including reservation waits. |
| `started_persistence` | Appending `tool.call.started` and delivering its durable side effects. |
| `effect_state` | Durable tool-effect intent, transition and reconciliation records, including the intent written before dispatch and recovery's per-call reconciliation. |
| `execution` | The application's tool invocation through effect completion; process isolation includes dispatch transport. |
| `result_processing` | Terminal hooks, result validation/redaction and terminal event preparation. |
| `staging` | Preparing, validating and durably staging a terminal checkpoint, sealing its secret snapshot, refreshing staged outcomes and recording hook completion on the stage. |
| `sibling_wait` | From this call's stage until the last sibling call stages its terminal. Zero for a single-call round and for the last call to finish. |
| `publication_queue_wait` | From the end of `sibling_wait` until this call's publication starts: the rest of dispatch, sealing the round, the pre-publication read and publication of earlier calls in model order. |
| `publication` | Pinning publication timing, verifying that checkpoint, appending the terminal event and delivering its durable side effects. |
| `round_commit` | The round's closing write: preparing and atomically publishing the transcript, checkpoint and round receipt, including exact replay after a lost acknowledgement and delivery of its events. For a paused attempt, the approval or user-input pause write; for an approved or answered continuation, its approval/input close. |
| `unattributed` | Remaining round orchestration outside the named phases, such as interruption checks and pending-round reads. |

Durations use a monotonic clock and exclude nested phase intervals. Generator
consumer backpressure is excluded from the active round phases. The two waits
are measured between runtime marks instead, so they can include time the
consumer takes with the round's yielded events. Each phase also reports
`first_started_at` and `last_completed_at`, the wall-clock bounds of its observed
entries; a phase entered more than once can have a window longer than its
duration. Parallel call durations and the two waits overlap other calls' work:
do not sum them to infer elapsed round wall time. Use `duration_seconds` for
observed round wall time. For a live round with serial dispatch, the active
phases (all except the two waits) add up to it, apart from consumer time
between yielded events. An approved or answered continuation does not charge
its remaining dispatch orchestration to `unattributed`, and some continuation
dispatch paths record only their staging, so its phases can add up to less
than its duration. Each local round execution or recovery attempt emits one
record: interrupted, cancelled and approval-paused attempts are `incomplete`,
an approved or answered continuation emits its own record for the same round,
and publication after process loss is `recovered`. Closing a round that was
interrupted before its live runner started is neither. Missing phases are
zero; Cayu does not reconstruct execution time from old receipts.

A foreground subagent tool's `execution` includes the child session's elapsed
time. The child's rounds and model steps report their own records under the
child session, and their phases are not subtracted from the parent call.

`ModelStepPreparationTiming` covers the gap from the last locally observed
round commit to the next `model.started` in the same run epoch. `handoff` is
the part before the model step starts preparing; it includes consumer
backpressure on the round's final events and session-loop scheduling.
`context_policy`, automatic `recall` and `counting` are then measured
separately, and `preparation` is the remaining time inside the model step
before the provider request starts. A commit is used once: a later step with
no round in between, such as a structured-output repair, starts at its own
preparation, as does the first model step. Database costs are measured only
while an observed phase is active.

Each phase reports `store_transaction_count`, `store_lock_wait_seconds`,
`store_execution_seconds`, `store_commit_seconds` and `store_bytes_written`.
These describe the maintained native store connection paths, including
session, task, budget and knowledge work and context propagated into SQLite
worker threads. Memory/custom stores have no
native database cost measurements. SQLite counts explicit/implicit driver
transactions and autocommit reads; PostgreSQL counts actual transaction
starts, including reads. Commit time excludes no-op commits. Bytes are bound
write payload sizes, including serialized JSON, rather than physical disk
pages, WAL bytes or replication traffic. No SQL or parameter values leave
the accumulator.

Lock wait includes local lock/read-pool or PostgreSQL pool admission. SQLite
`BEGIN IMMEDIATE` and PostgreSQL `FOR UPDATE` acquisition latency is also
reported there and in SQL execution time. That database acquisition latency
includes query/network work: it is not an exact database-engine lock-wait
counter. Lock/SQL/commit counters describe components of elapsed phase time;
they are not additional durations to add to it.

To export observations, override `EventSink.emit_timing(record)` or supply
`CayuApp(timing_sinks=[...])` with an async `emit_timing` method. Event sinks
that do not override `emit_timing` receive nothing and start no delivery
worker. `LoggingEventSink` emits the content-free JSON at DEBUG only when
constructed with `log_runtime_timing=True`; the default application logger
does not. Delivery is non-durable, best effort, outside invocation authority,
and bounded by a 128-record queue plus a one-second timeout per sink by
default. Slow/full queues drop observations; sink errors increment
`app.runtime_timing_status().failed_deliveries`. They never fail the workload,
append failure events, or retry durable sink receipts.
`await app.flush_runtime_timing()` waits for queued observations, and
`await app.close_runtime_timing()` stops delivery at shutdown after one flush
bounded by the sink timeout; records it could not deliver count as dropped.
The mounted and standalone Cayu servers close timing delivery on shutdown. An
application can be used from successive `asyncio.run()` calls: queued records
move to the running loop. The records and recent view do not survive restart.

`OpenTelemetryEventSink` exports round and preparation spans with their
observed start and end, and one `cayu.tool.phases` child per call under that
call's `execute_tool` span. The per-call span starts at the call's first
observed phase and ends at its last, normally the end of its publication; each
phase event is stamped at the phase's first entry and carries its duration and
store costs. Because the call's span includes authorization before
`tool.call.started` and staging/publication after the effect, it can extend
beyond its parent. The `execute_tool` span ends at the attested
effect-completion timestamp and carries the effect-completed, terminal-staged
and publication-started timestamps as events, so the latter two fall after
its end. If a timing record arrives before the sink has seen the session or a
call's tool span, the sink holds it (at most 1,000 records) until they appear
or the session ends; `close_runtime_timing()` exports any still held under the
best known parent. A call whose tool span never arrives stays under the round
span rather than becoming a trace root.

For example, suppose `execution` takes 50 ms, staging reports three commits
and 300 ms commit time, publication reports two commits and 200 ms, and
round commit reports one commit and 100 ms. Those 600 ms are runtime
persistence cost. If the first of two serial calls also shows 2 s of
`sibling_wait`, that is the second call running; it does not mean the first
tool or its result append took 2 s. A large `publication_queue_wait` on a
later call is the earlier calls' publication and the round's sealing, which
their own `publication` and `unattributed` phases account for. Compare
transaction counts and commit time before optimizing the application tool
or changing the storage medium.

Collection adds no durable event, checkpoint or receipt write per phase.
When timing is disabled, or outside a measured phase, the SQLite connection
returns the driver's own cursors, so rows are not iterated in Python. The
provider-free benchmark can be rerun noninteractively. Pass another checkout's
`src` directory, such as the merge base, as the no-timing baseline:

```console
uv run --no-sync python maintenance/benchmark_runtime_phase_timing.py --backend sqlite --iterations 20
uv run --no-sync python maintenance/benchmark_runtime_phase_timing.py --backend sqlite --iterations 20 --rounds 5 --baseline-src ../cayu-main/src
uv run --no-sync python maintenance/benchmark_runtime_phase_timing.py --backend memory --iterations 20 --rounds 5 --baseline-src ../cayu-main/src
```

The benchmark uses no timing/event/logging sink and a no-op tool; production
provider/tool latency will change the relative overhead.

One local macOS run on a shared, loaded machine compared 100 samples per
configuration (5 fresh worker processes of 20 runs each) against the merge
base on `main`, which has no timing collection. SQLite medians were 0.848 s on
`main`, 0.838 s with timing disabled and 0.840 s enabled. Memory-store medians
were 0.722 s, 0.724 s and 0.720 s. Every difference is within about 1.2% in
either direction, below the run-to-run noise of this measurement, so it shows
no measurable overhead rather than a speedup. These small local point
estimates do not predict overhead on a different machine or storage mount.
