# Durable one-shot task scheduling

A scheduled task exists durably before it becomes runnable. Workers may stop,
restart, or miss a notification without losing that task. The task store checks
eligibility atomically with claiming; a process timer is only a polling hint.

## Create a follow-up

Supply an explicit task identity, an aware timestamp, and a schedule policy:

```python
from datetime import UTC, datetime
from cayu import TaskCreate, TaskSchedulePolicy

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
from cayu import TaskRescheduleRequest, TaskScheduleCancelRequest

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

`app.list_task_schedule_events(task_id, after_sequence=0, limit=100)` returns
task-owned evidence, including before a session is attached. Pages are ordered by
sequence; pass the last sequence to read the next page. The limit is 1–1,000.
History includes scheduled/rescheduled, eligible/misfired, expired/skipped,
claimed, cancellation, execution, hold/resume, and terminal transitions as they
occur. A successful journal write is part of the native task mutation, not an
independent session-event write.

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
This primitive does not keep a session asleep, resolve a pending timer tool call,
or guarantee exactly-once external side effects.
