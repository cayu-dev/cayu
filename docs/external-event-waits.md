# Durable external-event waits

External waits compose a native `SessionStore` election owner, an authenticated
application policy, and optional session and task adapters. Constructing these
parts starts no worker. Event-only waits require neither a `TaskStore` nor a
`CollaborationStore`.

Run the credential-free, fresh-process example:

```sh
python examples/external_event_wait.py demo /tmp/my-new-external-wait-demo
```

It retains its SQLite databases and simulated job records, and covers ordinary
event delivery, event-before-registration, timeout, process restart and repeated
servicing without another provider dispatch. Its provider is deterministic, not
a paid service. Use a new directory; the demo does not reset existing evidence.

The supported session boundary is a **completed whole model turn, including its
complete tool round**, in an ordinary root session. This does not persist an
arbitrary Python stack, a live provider stream, a workflow activity, or an OS
process. Participant-owned, child, and task-owned session entrances are not
supported by `SessionExternalWaitAdapter`.

Successful native structured output, structured-output tool submission, and
final-tool completion park at the same boundary as an ordinary text answer.
Service retains the exact source model stage before admitting continuation and
restores its structured-output, final-tool, thinking, retry, and budget controls;
it does not replace them with request defaults after restart. This stage reference
is internal handoff evidence, not caller-supplied execution authority.

Request-local loop policies remain executable application components, not stored
Python objects. Supply them as `SessionExternalWaitAdapter(...,
request_loop_policies=(policy, ...))` for recovery and event servicing, as well as
in the original run/resume request. After restart, reconstruct policies with the
same declared execution-profile identities. Missing or changed policies fail the
existing profile check; the adapter never drops a policy to make recovery work.
Application-wide policies remain configured on `CayuApp` as usual.

## Assemble and reserve before submission

```python
from cayu import (
    ExternalCorrelationRequest, ExternalEventWaits, ExternalWaitAccessPolicy,
    ExternalWaitContext, ExternalWaitScope, ExternalWaitRegistration,
    SessionExternalWaitAdapter, ExternalWaitHost,
)

class JobPolicy(ExternalWaitAccessPolicy):
    def authorize(self, context, *, scope, source, action):
        return (
            context.principal == "trusted-job-host"
            and scope.application_scope == "video-service"
            and source == "renderer"
        )

context = ExternalWaitContext(principal="trusted-job-host")
waits = ExternalEventWaits(store=app.session_store, access_policy=JobPolicy())
adapter = SessionExternalWaitAdapter(app, waits)
host = ExternalWaitHost(adapter, context=context)
scope = ExternalWaitScope(application_scope="video-service", generation=1)
correlation = await waits.reserve_correlation(
    ExternalCorrelationRequest(
        scope=scope, source="renderer", correlation_key="video-42",
        deadline=None,  # Or an aware absolute datetime, retained unchanged on retry.
    ),
    context=context,
)
registration = ExternalWaitRegistration(
    correlation=correlation, operation_key="wait-video-42",
    projector_id="json", projector_version=1,
)
await waits.register(registration, context=context)
```

The host authenticates the webhook or local caller **before** constructing its
context. Do not copy a principal from event JSON. The registered policy must
grant the exact scope, source, and action; only `True` grants permission.

Reserve correlation durably before submitting the domain job. The application
still owns submission and uses the external system's stable idempotency key.
If submission succeeds but acknowledgement is lost, reconcile that key with the
external system. A wait receipt cannot prove whether the external job ran.
Explicit wait cancellation does not cancel that job.

## Park, deliver, and continue

```python
from cayu import ExternalEventDelivery, Message, RunRequest

parked = await adapter.run_to_wait(
    RunRequest(
        agent_name="video-agent", session_id="video-session-42",
        messages=[Message.text("user", "The renderer job was submitted.")],
    ),
    registration,
    context=context,
)

# In an authenticated webhook process, possibly before register/run_to_wait:
delivery = await waits.deliver(
    ExternalEventDelivery(
        correlation=correlation, delivery_id="renderer-result-42",
        payload_json='{"video_url":"https://example.invalid/video-42"}',
    ),
    context=context,
)

# Reconstruct app, waits and adapter after restart, then explicitly service:
page = await host.service_once(scope=scope, source="renderer")
snapshot = await waits.inspect(correlation, context=context)
```

`resume_to_wait(ResumeRequest(...), registration, context=...)` applies the same
whole-turn boundary to a new turn of an existing ordinary root session. Exact
completed retries return the retained result; changed input conflicts. A pending
execution is not permission to create a replacement session or invoke the model
again. Ordinary `app.resume()` cannot bypass a retained external-wait boundary.

`service_wait(registration, context=...)` is the one-wait adapter. `ExternalWaitHost`
performs bounded discovery and servicing; callers may instead explicitly run
`host.run(scope=..., source=..., stop=asyncio.Event())`. Stop that loop before
closing the app. These components reuse native continuation admission, claims,
receipts, execution profiles, budgets and writer fences. They do not create a
parallel continuation engine or promise exactly-once external side effects.

`service_once()` reports per-row conflicts, fenced writers, capacity failures,
and orphaned preparations in `page.failures`, using only the correlation key and
a bounded failure kind. A row that changed after discovery is a conflict, not an
authorization failure. Those rows remain pending while later independent rows are
serviced. A failure is not permission to exclude or replay execution. The
explicit `run()` loop logs each failing row once per failure kind until it
recovers, drains all available pages, then waits `interval_s` between sweeps.
Authorization failures and invalid discovery requests propagate.

`recover_to_wait(registration, context=..., inactive_for_seconds=60)` uses the
native recovery lease to reconstruct a committed assistant result interrupted
before parking. It checks the original profile and current wait authorization,
retains the original ticket, and reruns the normal post-model/before-stop gates
without repeating that provider request. A before-stop interruption does not
become a parked wait. A before-stop continuation retains the original interaction
and runs its additional step under native recovery authority before parking.
If a turn pauses for tool approval or user input, `run_to_wait()` reports
`ExternalWaitUnavailable` until the whole-turn boundary is reached. Resolve the
pause with the normal `resolve_tool_approval()` or `resolve_user_input()` API,
then recover the external wait with `recover_to_wait()` (or let the host do so).
Native pause continuation retains the completed turn's execution controls for
external servicing across restart, including its request-level budgets.
While either native pause is unresolved, host polling leaves its session and
checkpoint unchanged. Direct `recover_to_wait()` reports `ExternalWaitUnavailable`
before claiming a recovery writer. Cancellation and explicit retirement still use
the native cleanup path.
Completed native structured output and completed structured/final-tool rounds
reuse their durable validation and tool receipts rather than rerunning tools.
If parking committed before the process died but writer release did not, recovery
uses the same native claim and writer succession to finish release. It does not
rerun the accepted turn's stop policies or provider request. A parked ticket alone
is not release evidence: a live execution owner prevents takeover, and exact
post-release recovery replay does not acquire another epoch.
For cancellation or an explicit retirement request, the host uses ordinary
incomplete-session interruption recovery before retiring the exact wait. With
a store that keeps session execution leases, a live lease is what keeps a
running writer from being taken over; other stores wait out the default
60-second inactivity grace first. This
also covers interruption before parking: cleanup does not re-enter the model
loop or run stop policies. Native tool-effect and environment cleanup guards
still apply; a control request is not evidence that an external effect stopped.
The elected outcome does not change. A still-live writer continues to fence
cleanup; failure of the release write leaves responsibility pending for
authenticated reconstruction.
Before a ticket exists, native admission records a bounded cleanup origin bound
to the exact external preparation and registration. Recovery carries it through
authenticated writer transfers and release, so lifecycle-ledger compaction does
not erase pre-ticket cleanup evidence. This proof grants no execution authority;
exclusion still requires the current writer's durable release.
Exact completed recovery retries return the retained receipt. The host attempts
this recovery for pending bound waits; a live native
execution lease still excludes takeover.

An interrupted partial tool round retains each completed tool result. Unknown
external tool effects must be reconciled through the application's existing
`inspect_tool_effect()` / `reconcile_tool_effect()` receiver before recovering the
whole-turn wait. An external event is not proof of a tool's result, and the host
does not repeat an uncertain tool call.

If admission committed before any native wait ticket was created, use ordinary
`recover_incomplete_session()` to settle the interrupted native writer, then
`exclude_prepared_execution()` to discharge its exact unbound preparation.
Exclusion requires durable original-admission, same-invocation rebind lineage,
and writer-release evidence; it does not infer release from cancellation or
process death. The selected external outcome is preserved, but exclusion does not deliver it to
the session. Automatic continuation is unavailable across this pre-ticket gap:
the preparation retains request commitments, not a replayable original request.
The host reports such a row as `orphaned` once its session execution owner is
lost, or once the preparation is 60 seconds old without a live writer; it stays
pending until you recover and exclude it.
Native model/tool uncertainty still requires its existing reconciliation before
cleanup can finish.
Do not interpret a pending inspection or `ExternalWaitUnavailable` as permission
to repeat the original execution or allocate a replacement job.

## Election, early input, and exact retries

Memory and SQLite use the configured ownership clock; PostgreSQL samples its
server clock after obtaining native ownership. An event wins only if it is
accepted strictly before the deadline. At equality or afterward, timeout wins.
Election uses integer epoch milliseconds: sub-millisecond precision is discarded
from both the owner time and the absolute deadline before comparing them. The
original deadline remains part of the exact registration identity.
The first durably selected outcome cannot be changed by a delayed timer, another
event, or cancellation. An event's sender timestamp is not election authority.

Unknown correlations are rejected. A reserved correlation accepts bounded early
input before registration. That acknowledgement means retained input, **not**
session execution or permanent retention. The early retention interval defaults
to 24 hours, at most 30 days. If unregistered retention expires, observation
selects `unavailable`, and later registration cannot revive it. Registered waits
use their retained deadline (or remain event-only), not an early inbox expiry.

Each delivery ID commits its canonical original content before redaction. An
identical retry replays its receipt; altered content conflicts, even if different
secrets would redact to the same marker. Additional matching events do not
replace the winner. Delivered-after-settlement receipts report their disposition
without continuing a second time. Keep stable correlation, registration,
delivery, cancellation and native execution identities after timeout or lost
acknowledgement; inspect/reconcile them rather than allocating replacement keys.

Outcomes are `event`, `timeout`, `cancelled`, or `unavailable`. Only event/timeout
can produce continuation input. Cancellation/unavailability discharge native
responsibility only with positive exclusion or writer-release evidence.

If initial/resumed execution stops before creating a native continuation ticket,
the application may explicitly call
`exclude_prepared_execution(registration, context=...)`. The store requires
either the exact still-unconsumed admission frontier or positive writer-release
evidence; it atomically fences delayed creation/admission. An elected event or
timeout is retained unchanged, not relabeled as cancellation. The host never
automatically abandons a winning event this way. Already-bound continuations and
unreleased writers require their own native recovery/retirement evidence, not
this exclusion. Exact exclusion remains replayable after restart and deletion.

For an already-bound wait with a terminal outcome, the application can instead
call `retire_execution(registration, operation_key="stable-cleanup-key",
context=...)`. This records an exact, store-timestamped retirement request without
changing the elected event or timeout. Inspection exposes `retirement_requested`,
not the private native control. Keep the same operation key through interruption
or acknowledgement loss; a changed key conflicts.

The request alone does not mean that execution has stopped. Native retirement
still requires positive writer release and settled service evidence, and races
against native continuation admission. A live/unresolved writer remains fenced.
An in-flight service may win; its exact admission is reconciled rather than
replaced or relabeled as excluded. Check the returned `handoff`: `excluded` means
native retirement completed, while `settled` means continuation admission won.
The host discovers pending retirement after restart and completes the native
retirement/acknowledgement handoff without recovering the original model turn.
No automatic retirement is inferred merely because an accepted event is old.

## Projection and bounds

The default registered `JsonExternalWaitProjector` forwards sanitized event JSON;
other outcomes have a small kind object. Register an `ExternalWaitProjector` by
explicit `(ID, version)` for a deterministic domain projection. Projectors must
have no external effects. Projection commits once before native admission;
reconstruction uses that exact projection rather than running a changed callback.

`ExternalWaitLimits` narrows native finite ceilings: 4,096 correlations per scope
generation, 32 MiB reserved payload/projection capacity, 32 delivery identities per
correlation, and 64 KiB each for payload and projection. Encoded durable records
also have a 512 KiB ceiling. Page size defaults to 32 and cannot exceed 256.
Identifiers are bounded and reject controls/known secrets. Inspection exposes
safe identity, deadline, outcome, projection and pending-handoff status, not the
native continuation permit or invocation authority. Raw event secrets must also
be registered with the configured redactor by the trusted host.

## Optional scheduled hints and bounded cleanup

`TaskStoreWaitScheduler(waits=..., task_store=..., scheduler_id=...)` composes the
existing managed one-shot task machinery. `schedule(registration, context=...)`
retains an exact task intent before task creation and acknowledges its publication
afterward. `reconcile(...)` discovers interrupted publication. Its
`worker_handler(context=...)` plugs into the existing task worker; it observes the
wait-store outcome and does not dispatch a model. It starts no worker itself.
Early or delayed hints cannot change election policy; explicit host polling also
observes due deadlines. Event-only registrations produce no scheduled task.
After scope retirement, an exact retained timer task completes as an obsolete
hint, including after wait-record pruning. The scheduler verifies the durable
retirement tombstone and task identity; missing records alone are not proof.
`notify()` returns `None` for this case, without publishing another outcome.

This shares the native continuation ticket/checkpoint owner and managed
[task scheduler](task-scheduling.md) with the one-shot self-wake design.
Self-wake owns model-facing timer waits. External-event waits add correlation, early input
and event/timeout arbitration, not a second scheduling system.

`observe_page(scope=..., source=..., context=..., after=..., limit=...)` advances
due outcomes in bounded pages using native time. Each item commits separately;
interrupted pages may be repeated. It neither cancels jobs nor discards handoffs.

Administrators can call `retire_scope(scope, operation_key=..., context=...)`
only after every record is terminal and all runtime/timer handoffs are settled.
The policy must explicitly grant `source="*", action="retire"` for the entire
generation; ordinary source access is insufficient. An immutable tombstone and
content commitment fence all later mutations. Exact retirement retries return
the original receipt, while changed operations conflict.

`prune_retired_scope(receipt, limit=..., context=...)` requires the exact receipt
and `source="*", action="cleanup"`. It removes only a bounded terminal page,
never the tombstone. Retrying a lost pruning acknowledgement can remove the next
page; its result is maintenance progress, not an exact per-record deletion receipt.
Old keys cannot be reopened. An explicitly new generation is a distinct scope.

All native backends must qualify these ownership operations. Unsupported custom
stores/wrappers fail before admission. Closing the app seals adapter entrances
and tracks in-flight native mutations; it does not delete durable waits or cancel
domain jobs. The standalone owner is independently usable and has its own
`aclose()` observation drain.

Storage revision 115 upgrades existing version-2 continuation indexes inside the
migration transaction. It obtains each added purpose from the exact native record
committed by that index, preserving continuation and lifecycle receipt hashes.
Missing or conflicting receiving evidence rolls back the revision; migration
does not synthesize completion, release, or exclusion. Run the normal explicit
storage migration before deploying the new binary against an existing database.
