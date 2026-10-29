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

## Run them from `cayu serve` and `cayu worker`

`app.host_external_waits(...)` builds the parts below once from the application's
own session and task stores, in the factory that both commands load. After that the
task worker services the waits and `cayu serve` takes their webhooks, so the
application writes no servicing loop, deadline handler or delivery route:

```python
import json

from cayu.external_wait_hosting import ExternalEventSource, ReceivedExternalEvent
from cayu.external_waits import ExternalWaitContext
from cayu.sessions.external_waits import ExternalWaitScope

SCOPE = ExternalWaitScope(application_scope="video-service", generation=1)


class RendererWebhook(ExternalEventSource):
    source = "renderer"

    async def receive(self, request):
        if not valid_signature(request):  # Your sender's own verification.
            return None  # The route answers 401.
        body = await request.json()
        return ReceivedExternalEvent(
            context=ExternalWaitContext(principal="renderer-webhook"),
            # The generation you issued the key in, from the signed body or URL.
            scope=ExternalWaitScope(
                application_scope="video-service", generation=body["generation"]
            ),
            correlation_key=body["job_id"],
            delivery_id=body["delivery_id"],
            payload_json=json.dumps(body["result"]),
        )


def build_app():
    app = CayuApp(session_store=..., task_store=...)
    app.host_external_waits(
        access_policy=JobPolicy(),  # As in the next section.
        context=ExternalWaitContext(principal="cayu-worker"),
        scopes={SCOPE: ("renderer",)},
        sources=(RendererWebhook(),),
        delivery_path="/webhooks/waits",
    )
    return app
```

- `context` is the principal the worker services and schedules deadlines as; the
  policy must grant it `read` and `service` for every configured scope and source,
  or construction fails. An application caller needs `reserve` and `register` to
  start a wait, and `service` to park a session on it. A webhook's principal, from
  its source, needs `deliver`.
- `sources` and `delivery_path` are optional. Without them nothing is served, and the
  application delivers through `hosting.deliver(...)` itself. A path segment holds
  letters, digits, `-`, `.`, `_` or `~`.
- `max_request_bytes` bounds a webhook body; it defaults to the payload limit plus
  16 KiB (80 KiB) and is at most 1 MiB.
- The task store must schedule tasks: the worker continues sessions from tasks.

Register, park and deliver through the returned hosting instead of the bare parts.
`hosting.register(...)` registers the wait as the caller and schedules its deadline
task as the hosting. `hosting.run_to_wait(...)` and `hosting.resume_to_wait(...)`
park the session as the adapter does. `hosting.deliver(...)` delivers an event. The
parts stay available as `hosting.waits`, `hosting.adapter`, `hosting.host` and
`hosting.scheduler`.

What runs where:

- **Web process.** `cayu serve` mounts `POST {delivery_path}/{source}`, next to the
  project's own routes, which may share its prefix but not sit under the path
  itself. The route is outside `cayu serve --auth` and the Developer dashboard's
  access policy: each `ExternalEventSource` authenticates its own callers in
  `receive(request)`, and a principal never comes from the event JSON. The source
  also returns the exact scope generation the correlation was issued in, read from
  the signed request, so put the generation in the callback URL or the correlation
  key you give the sender. A key is never reopened within a generation, so scope and
  key name exactly one wait: a webhook replayed after you move to a new generation
  that reuses the key is refused, not delivered to the new wait. The body is read
  up to `max_request_bytes` before the source sees it. The route answers:
  - 202 with the delivery receipt;
  - 401 when the source refuses the request, and for a source that isn't
    configured, so a caller can't tell which sources exist;
  - 403 when the access policy refuses the sender, before the correlation is
    looked up;
  - 404 for an unknown correlation, or a scope not configured for the source;
  - 409 for a conflicting retry;
  - 413 for a body over `max_request_bytes`, without calling the source;
  - 422 when the source raises `ValueError`, `KeyError` or `TypeError` reading the
    body, or the payload is malformed or over the payload limit;
  - 429 with `Retry-After` when the wait owner is at capacity, and 503 when the
    acknowledgement is unavailable.

  There is no rate limit per sender; 429 reflects only the owner's capacity, so put
  any per-sender limit in front of the route. `hosting.delivery_router()` returns
  the same route for any other ASGI host, and `hosting.serves_delivery(app)` says
  whether an ASGI app includes it. `create_agent_service(...)` mounts it in a
  maintained service when the app hosts waits first, refusing a delivery path
  inside the service's own paths; `cayu serve` refuses a `service_factory` whose
  service doesn't serve it. The web process never continues a session.
- **Worker.** `run_task_worker`, and so `cayu worker` and `cayu worker --until-idle`,
  runs two task types itself, beside the application's handler, as it does for
  `sleep_until` wakes. Every continuation runs inside one of these tasks, under its
  lease and the worker's time limits, and counts as a handled task.
  - `cayu.external-wait-deadline.v1`, the deadline task, observes the timeout and
    continues the session in the same task. A deadline task created by a scheduler
    the application wired by hand, with another `scheduler_id`, goes to the
    application's handler.
  - `cayu.external-wait-service.v1` continues one parked wait whose outcome was
    elected. Delivery, parking, the deadline task and the sweep share one attempt
    chain per outcome, so they converge on the same tasks: an accepted delivery to
    a parked session creates the first attempt, due at once; a retried webhook and
    a later event that doesn't win create none; an event that arrives before the
    session starts parking is scheduled when it parks.
  - A wait whose execution was prepared but whose binding wasn't published (the
    process failed between the two) is scheduled too, and its attempt repairs the
    binding. The binding is published before the turn runs, so such a session has
    no completed turn to continue: its attempts keep retrying, without dispatching
    the model, until you retire it with `hosting.adapter.retire_execution(...)`.
  - An attempt that fails, or ends while its parked session still can't continue
    (behind a live writer, or before the wait store elected a timeout), gets a
    successor with backoff from 30 seconds to an hour. Each successor is timed from
    when its predecessor ran, and the 24-hour retry window starts when the first
    attempt ran, so attempts that only run long after the event still get their
    retries. A deadline task that elects the timeout but can't finish continues
    in that chain, so `--until-idle` reports when to come back.
  - On its expired-lease reclaim cadence, starting with its first pass and then
    every `reclaim_every_s` (60 seconds by default), the worker's maintenance slot
    sweeps every configured scope and source. It finishes interrupted deadline
    scheduling, and gives a parked wait whose outcome is elected a live attempt:
    the first one after a delivery whose task creation failed, or a successor when
    the last attempt failed or ended without servicing it. It only reads a wait
    that needs nothing, and never continues a session itself.
  - A worker whose `query` leaves these task types out, or narrows by session,
    agent, parent, status or text, doesn't claim them and doesn't sweep. Run at
    least one worker that claims them, with `reclaim` on.

On Cayu Cloud the web process sleeps and the worker runs. List the delivery path
under `[webhooks] paths` in `cayu-cloud.toml`, so a webhook sent while the Agent sleeps
is kept and delivered once it is ready. A deadline is a delayed task and a delivered
event a task due now, so a host that starts `cayu worker --until-idle` when work is
due needs nothing specific to external waits. Hosted waits keep the adapter's
boundary: ordinary root sessions only.

## Assemble and reserve before submission

```python
from cayu.sessions.external_waits import ExternalCorrelationRequest, ExternalWaitScope, ExternalWaitRegistration
from cayu.external_waits import ExternalEventWaits, ExternalWaitAccessPolicy, ExternalWaitContext
from cayu.session_external_waits import SessionExternalWaitAdapter
from cayu.external_wait_host import ExternalWaitHost

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
from cayu.sessions.external_waits import ExternalEventDelivery
from cayu import Message, RunRequest

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
wait-store outcome and does not dispatch a model. It starts no worker itself;
[hosted waits](#run-them-from-cayu-serve-and-cayu-worker) run it, and continue the
session, without one.
Early or delayed hints cannot change election policy; explicit host polling also
observes due deadlines. Event-only registrations produce no scheduled task.
After scope retirement, an exact retained timer task completes as an obsolete
hint, including after wait-record pruning. The scheduler verifies the durable
retirement tombstone and task identity; missing records alone are not proof.
`notify()` returns `None` for this case, without publishing another outcome.

Model-facing timer waits belong to the opt-in `sleep_until` tool (see
[task scheduling](task-scheduling.md#let-an-agent-sleep-and-continue-later)). It
shares the managed [task scheduler](task-scheduling.md) with these hints and, for a
task-owned session, the worker's parked handoff. External-event waits add
correlation, early input and event/timeout arbitration, not a second scheduling
system.

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
