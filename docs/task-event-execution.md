# Event identity and execution budgets

Task events share an immutable attempt budget and execution deadline across
resubmissions and native retry successors. Built-in stores enforce admission
atomically; workers enforce execution and retain ownership through cleanup.

## Contract

An event is a logical operation, while a task is one admitted attempt. A retry
key identifies the event. Its first admission fixes the event's start time and
budget. Resubmitting, restarting a worker, and creating a retry-series successor
must preserve that identity and may not reset its deadline.

Equal payloads do not establish identity. `TaskCreate.event_id` explicitly names
an event within its task type and authenticated actor (trust, subject, tenant). The conventional string `input.event_id` supplies
the same identity when a caller has already included it. An explicit `retry_key`
overrides this derivation. Independent operations use different event IDs even
when their payloads match. An ordinary root with no event identity gets its own
identity; the runtime cannot identify an undisclosed relationship between jobs.

The existing retry-budget defaults remain configurable. There is no universal
ten-minute runtime limit. A workload may explicitly use
`TaskRetryBudget.unlimited()`. Managed schedules, session-owned continuation and
verified-work contracts retain their own ownership contracts; they must not be
silently reclassified as unrelated root events.

## Enforcement and visibility

The event deadline bounds the admitted operation, not just later admissions.
Workers consult store-authoritative time during dispatch and renewal, request
cancellation at the deadline, drain owned execution, and retain cancellation
ownership until cleanup is proved. Human waits do not reset the event deadline.
A stopped worker cannot certify cleanup; recovery retains that distinction.

When a task resumes an existing session or settles its human wait, the runtime
stores the event deadline on that session before continuing. A later process
enforces the retained bound while that event has unfinished work, including
when an explicit replacement deadline is requested. After a session turn ends without a pending approval or user-input request,
a different event may continue it under its own budget, or an explicit deadline
may start a new turn outside a task. This restores the session deadline that
preceded the settled event; it never resets the same event or extends an
unrelated session deadline. Stale continuation requests do not change the stored deadline.

Deadline exhaustion is persisted independently of a later create request.
Idle-worker reconciliation covers pending or paused work; active owners observe
the same terminal event state. Attempt exhaustion is recorded after the final
admitted attempt finishes unsuccessfully, rather than on a hypothetical next
submission. No running attempt is falsely marked stopped because the admission
counter has reached its limit.

The event record exposes the fixed policy, deadline, counts and terminal reason.
Task inspection exposes its event association. In-memory, SQLite and PostgreSQL
implement the same atomic admission and terminal-state contract. A storage
revision fences older workers that would ignore event execution deadlines.

## Proof

`tests/core/test_task_event_execution.py` covers identity, concurrent admission,
shared successor capacity, restart, active hangs, cancellation cleanup, paused
work, scheduled expiry, unlimited execution, and durable terminal reasons across
the three built-in stores. `tests/core/test_continuation_deadlines.py` verifies
that approval and user-input renewal cannot extend an event deadline.

Revision 133 fences older workers. Tasks admitted before this revision retain
their existing contracts; migration does not assign retroactive event deadlines.
Historical admission counts are retained separately as `legacy_attempts`, because
those tasks lack terminal event accounting. New attempts inherit the existing
policy; their final failure can close admission without claiming that historical
work has stopped. Omitted budgets inherit the first admission policy; an explicit
different budget is rejected.
Event records survive task retention, preserving deduplication and terminal
exhaustion evidence.
