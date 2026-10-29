# Durable one-shot task scheduling

A scheduled task exists durably before it becomes runnable. Workers may stop,
restart, or miss a notification without losing that task. The task store checks
eligibility atomically with claiming; a process timer is only a polling hint.

## Create a follow-up

Supply an explicit task identity, an aware timestamp, and a schedule policy:

```python
from datetime import UTC, datetime
from cayu.tasks import TaskCreate, TaskSchedulePolicy

request = TaskCreate(
    task_id="customer-42-followup-1",
    type="customer-followup",
    available_at=datetime(2027, 1, 15, 12, tzinfo=UTC),
    schedule_policy=TaskSchedulePolicy(),
    input={"customer_id": "42"},
)
task = await app.create_task(request)
```

Retry the same creation request after acknowledgement loss. Reusing its task ID
with different creation content conflicts; retrying identical content converges
on the existing occurrence. Preserve the originally chosen timestamp rather than
recalculating a relative delay on every retry. Creation replay does not reset a
later edit or terminal outcome.

Without `schedule_policy`, existing `available_at` behavior is unchanged. Managed
schedules require queue admission; they cannot be directly started while still
unadmitted. Ordinary immediate tasks and trusted direct-start semantics for
unmanaged tasks remain separate.

## Edit or cancel with a revision

```python
from cayu.tasks import TaskRescheduleRequest, TaskScheduleCancelRequest

receipt = await app.reschedule_task(TaskRescheduleRequest(
    task_id=task.id,
    operation_id="move-followup-1",
    expected_revision=task.schedule.revision,
    available_at=datetime(2027, 1, 16, 12, tzinfo=UTC),
    policy=TaskSchedulePolicy(),
))
cancelled = await app.cancel_scheduled_task(TaskScheduleCancelRequest(
    task_id=task.id,
    operation_id="cancel-followup-1",
    expected_revision=receipt.schedule.revision,
))
```

The revision rejects stale controllers. An operation ID binds the complete edit
or cancellation request. Exact retries return the original receipt, even after
another accepted transition; they never overwrite newer state. Changed content
under the same operation ID conflicts. Reload the task to observe its current
state rather than interpreting an old receipt as the latest snapshot.

Rescheduling replaces the full policy and is allowed only before first admission,
while pending or held and without a live execution owner. It cannot move an
already-admitted occurrence. Cancellation after execution starts follows existing
owned cancellation settlement: a cancellation request is not evidence that a
remote call or handler has stopped. Do not retry external work based on that
request alone.

## Expiry and late observation

All timestamps normalize to UTC. Eligibility opens at `available_at`, including
equality. PostgreSQL production decisions use the database transaction clock;
Memory and SQLite use their store clock. Synchronize hosts that share a SQLite
database. A worker's local timer cannot override the store's decision.

`expires_at`, when supplied, must be later than `available_at`. It is an exclusive
latest admission time: observation exactly at expiry is expired. It is not a
running-task deadline and is unrelated to `lease_expires_at`.

The default `fire_once` policy permits one late admission. `skip` refuses admission
when lateness is strictly greater than `misfire_grace_seconds` (default 60).
Equality remains eligible. Expiry takes precedence over either late policy.
Expired/skipped tasks reach an explicit non-execution terminal outcome with
scheduling history. A worker must observe the queue for these transitions to be
materialized; wall-clock passage alone does not write a terminal record.

First admission ends schedule-expiry evaluation for that occurrence. Reclaiming
its lease continues the same occurrence rather than applying expiry again.
Retry-series budgets and elapsed deadlines are independent authority; moving an
unstarted first attempt does not renew its cumulative allowance or deadline.
For pending tasks, an exhausted retry-series elapsed deadline takes precedence
when schedule expiry or skip would also refuse admission: the task fails with
`elapsed_exhausted` and scheduling failure evidence. Held tasks retain their
existing hold semantics; schedule maintenance may still expire or skip them.

## Workers and inspection

Use the existing `run_task_worker` entry point. The shared worker loop combines
store next-due observations with bounded polling and admission notifications.
Missing a notification can delay discovery, but cannot lose or prematurely admit
work. Work remains FIFO by creation order among eligible candidates, not by
scheduled timestamp.

To learn when scheduled work next needs a worker without claiming it, call
`TaskStore.next_task_schedule_wakeup(query=None)`. It returns a
`TaskScheduleWakeup` (`from cayu.tasks import TaskScheduleWakeup`):

| Field | Meaning |
| --- | --- |
| `as_of` | The store clock when it observed (PostgreSQL's transaction clock; the Memory and SQLite store clock). |
| `next_available_at` | When the earliest matching pending task whose time has not come becomes claimable, or `None`. |
| `next_expiry_at` | When the earliest matching unadmitted scheduled task's `expires_at` passes, or `None`. |
| `maintenance_required` | Whether a matching unadmitted scheduled task already expired or is past its `skip` grace, so a worker's sweep must settle it now. |

`query` applies the worker's claim filters, such as a task type; `None` observes
every unattached task. Times are UTC. Measure them from `as_of` rather than from
the caller's clock. `run_task_worker` reads this observation between polls, and
Cayu Cloud's [schedule route](runtime-contracts.md#schedule) serves it for every
task to a host that wakes a sleeping application. The result is a hint and
grants no claim authority: a task it names may be claimed, rescheduled or
cancelled by the time the caller acts, and the store still decides eligibility
atomically when a worker claims. Stores that do not support scheduling raise
`NotImplementedError`.

A host that doesn't keep a worker running calls `run_task_worker_until_idle`
(or `cayu worker NAME --until-idle`) instead. It handles what is claimable,
returns once nothing is, and reports the store's `next_available_at` and
`next_expiry_at` for its query so the host knows when to start it again; see
[run due tasks and exit](project-workers.md#run-due-tasks-and-exit).

`app.list_task_schedule_events(task_id, after_sequence=0, limit=100)` returns
task-owned evidence, including before a session is attached. Pages are ordered by
sequence; pass the last sequence to read the next page. The limit is 1–1,000.
History includes scheduled/rescheduled, eligible/misfired, expired/skipped,
claimed, cancellation, execution, hold/resume, and terminal transitions as they
occur. A successful journal write is part of the native task mutation, not an
independent session-event write.

A scheduled graph member that becomes `dependency_skipped` records
`task.schedule_dependency_skipped`, including transitive dependency skips. This
is distinct from the late-policy `task.schedule_skipped` and is not a misfire or
an admission. Native stores commit the task outcome, graph evidence, and schedule
history atomically. Exact graph replay and persistent-store reopening preserve
that history without adding duplicate terminal events.

Protected HTTP routes expose the same controls:

- `POST /api/tasks/schedule/reschedule` — the typed reschedule request above.
- `POST /api/tasks/schedule/cancel` — the typed cancellation request above.
- `GET /api/tasks/{task_id}/schedule/events` — sequence-paginated history.

Task list/detail responses include nullable `schedule`. Public scheduling
responses omit private intent hashes. Mutation requests are limited to 8 KiB;
invalid bodies receive fixed diagnostics rather than echoed input. Authentication
uses the server's configured access policy; scheduling grants no additional tenant,
budget, tool, approval, or provider authority.

Custom stores must explicitly support `supports_task_scheduling` and implement
atomic creation replay, revision-fenced mutations, receipts, scheduling history,
next-due observation, and scheduling-aware lifecycle transitions. Merely persisting
`available_at` does not establish this capability. Application-facing mutations
also require the existing positive cancellation-quiescence store contract.

Storage revision 90 is a breaking writer boundary. Stop all workers and other
database writers before migrating, then restart them with binaries supporting
revision 90. Older writers do not preserve schedule revisions or admission
policy; do not mix them with scheduling-aware writers. Migration adds nullable
schedule state to existing tasks without turning those tasks into managed
schedules.

## Process reconstruction and recurrence

[`durable_followup.py`](../examples/durable_followup.py) separates the producer,
worker, and inspection commands. It records a local follow-up decision without
contacting a customer. Stop the worker before the due time and start it again
against the same database; the task remains pending until the store admits it.

For a next occurrence, the application chooses a durable completion boundary and
creates a new one-shot with a deterministic occurrence ID and explicit timestamp.
It owns retrying that publication after crashes, bounded catch-up, approval, and
spend decisions. There is no automatic cron loop or recurring-series definition.
This primitive alone does not keep a session asleep or guarantee exactly-once
external side effects; `sleep_until`, below, builds the session wake on it.

## Let an agent sleep and continue later

Three kinds of waiting look alike but differ in what survives a crash:

| Wait | How | What stays alive | After a crash |
| --- | --- | --- | --- |
| Live blocking wait | `exec_command` running `sleep 600`, at most 600 seconds | the worker, runner, command and environment | lost |
| Durable one-shot self-wake | the `sleep_until` tool | nothing | the same session continues on time |
| Recurring schedule | the application creates each next one-shot task, or a host schedule such as Cloud `[[schedules]]` | nothing | the application's own publication retry |

`SleepUntilTool` is opt-in. Register it on the agents that may schedule their own
continuation; registration refuses it unless the application has a task store that
supports scheduling:

```python
from datetime import timedelta
from cayu.tools import SleepUntilTool

app.register_agent(agent, tools=[SleepUntilTool(max_sleep=timedelta(days=7))])
```

The model passes exactly one of `duration_seconds` or `until`, an ISO 8601
timestamp with an offset. The application clock turns it into one UTC deadline
once. A deadline further away than `max_sleep` (30 days by default), a missing
offset, or a non-finite or non-positive duration is refused as an error result. A
deadline that has already passed, including exactly now, returns at once.

For a future deadline the runtime parks the call before the tool runs, as its own
`pending_wake` pause opened under the `wake-open:{wake_id}` receipt. It then creates
task `cayu-session-wake-<wake id>` of type `cayu.session-wake.v1`, due at the
deadline, and interrupts the session with `interruption_type="wake_scheduled"`.
Creating that task is idempotent: a retry or a racing reconciliation converges on
the same task. No worker slot, lease, process or provider stream is held while
the session sleeps.

`run_task_worker` runs wake tasks itself, beside the application's handler, so no
handler needs to know about them; a worker whose `query` leaves out their type does
not claim them. When it starts, and then on its expired-lease reclaim cadence, it
runs `reconcile_session_wakes(app)`, which gives every open wake a live attempt; a
damaged session is logged and skipped. When the store makes the task due, the
runtime closes the wake once under `wake-close:{wake_id}`, resolves the original
`sleep_until` call with a fixed timer result (`status`, `wake_id`, `requested`,
`deadline`, `deadline_resolved_at`) and continues the same session to its next
pause or end. The continuation's `session.resumed` event carries
`interruption_type="wake_scheduled"`, so supervision and Cloud never count it as a
person's answer. No user message is added, and the call and its result stay in
provider order. Other tools from the same round run then, after the deadline,
against the limits and budget projection made when the round paused.

"Continues once" holds across failures: the continuation is claimed and
published exactly once, and duplicate, late or competing deliveries settle as
`already_delivered` without calling the model again. A delivery that fails (for
example a missing provider, a changed execution profile, or application and
store clocks more than five minutes apart) fails only that attempt task, whose
error stays on the task, and schedules the next attempt after 30 seconds,
doubling to at most an hour. Attempts continue for `SESSION_WAKE_RETRY_WINDOW`
(24 hours), counted from the deadline or, if no worker ran the first attempt on
time, from when one did; an attempt whose worker lost its lease or was fenced is
replaced too. After the window, each reconciliation pass logs one warning naming
the spent wakes, and you continue the session directly. A crash during the woken
continuation itself, after the call received its result, leaves the session
`running` and its attempt fenced, which Cloud task health shows. After
`app.recover_abandoned_sessions()` the session is `interrupted` with no pending
action, and an ordinary `resume` continues it. The sleeping call keeps its one
timer result, and the wake never continues the session a second time.

If the process dies while another tool from the woken round runs, that tool has
started with no result and cannot be run again safely. The next attempt settles
as `manual_recovery_required`, no further attempt is created, and the session's
pending action is a `manual_recovery` naming that tool's `tool_call_id`. Settle
it with `app.recover_user_input(UserInputRecoveryRequest(...))`, giving the
sleeping call's accepted timer answer (`answer` and the wake's `resolved_by`, as the
delivery sent them) and the tool's verified outcome. A subagent from the woken
round that pauses for a person is an ordinary delegated action: the wake has
already delivered, and the round continues when the subagent ends. A wake's id is the
`wake_id` of its pending action, from
`app.session_store.query_pending_actions(PendingActionQuery(kind=PendingActionKind.WAKE))`
(both in `cayu.sessions`), and the `wake_id` in its `session.interrupted` event's
`wake` payload. These helpers are public in `cayu.tasks`:

```python
from cayu.tasks import deliver_session_wake, session_wake_attempts

for attempt in await session_wake_attempts(app, wake_id):
    print(attempt.input["attempt"], attempt.status, attempt.error)
await deliver_session_wake(app, session_id=session_id, wake_id=wake_id)
```

A wake never resumes before its deadline.

Once an interruption of a sleeping session is admitted, it cancels the wake's
pending attempts, whether or not the caller reads its events, and closes the
sleep in the same atomic step that stops the session, even while the sleep is
still opening. Deleting the session cancels the attempts too. A wake attempt that
runs anyway finds the pause closed, settles as `closed`, and leaves the session
alone. A delivery whose continuation stops before the call gets its result (for
example, its environment cannot reconnect) fails that attempt, and the next
attempt retries it.
`resume`, `fork` and `resolve_user_input` refuse a sleeping session.

A sleep is not a request for anyone's input. Inside the runtime its pending action
has kind `wake`, no `attention_id`, and `wake_at`; the inbox, human-attention
notifications, user-input queries and the server's `/api/pending-actions` never
list it, and the session's status stays `interrupted`. Cayu Cloud activity reports
it as `waiting_for.kind = "wake"` with `wake_at` and no content. Sleep time is left
out of run-scoped `RunLimits` and budgets, and of a run-default execution deadline,
as a human wait is. Session-scoped limits (`RunLimits(scope="session")`) stay
wall-clock, so a long sleep spends them. The session keeps its original execution
profile; reconstructed applications need stable provider and tool identities, as
for any resumed pause.

`sleep_until` declares `ToolEffect.NONE`: the runtime schedules the wake before the
tool runs, and `run` itself only reports a refusal or an already-passed deadline,
so replaying it after a crash is safe. A tool policy that should gate sleeping
names `sleep_until` rather than relying on its effect.

This version writes checkpoint schema 12 for every session it touches, sleeping
or not. An older runtime refuses those sessions as
`checkpoint_schema_version_too_new` instead of misreading them, so rolling back
across this version is not supported (see
[checkpoint versions](runtime-contracts.md)).

This first slice sleeps only top-level sessions that no task owns, in rounds where
no tool needed approval. In a child session, a task-owned session, or a round
that paused for approval (including approval of `sleep_until` itself),
`sleep_until` returns an error result instead of sleeping. The environment
follows its ordinary lifecycle while the session sleeps; in-memory process state
is not kept.
