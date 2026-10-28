# Long-running jobs

Long-running jobs need approval before side effects, waits for outside replies,
reminders and deadlines, a spend cap, and proof that all of it survives a
restart. Use Cayu's part for each instead of approval tables, timestamp polling
or throwaway restart scripts. `cayu new NAME --preset long-running` generates a
small invoice-payment project with all five wired and tested.

## Approval before a side effect

```python
from cayu import Tool, ToolEffect, ToolResult, ToolSpec
from cayu.runtime import ExecutionProfileBehaviorIdentity
from cayu.tools import AlwaysRequireApprovalToolPolicy

PAID = {}  # Stands in for the payment provider, keyed by idempotency key.

class PayInvoice(Tool):
    spec = ToolSpec(
        name="pay_invoice", description="Pay one invoice.", effect=ToolEffect.IDEMPOTENT,
        input_schema={"type": "object", "properties": {"invoice_id": {"type": "string", "maxLength": 256}}},
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="pay_invoice", behavior_version="1", implementation_version="1"),
        max_terminal_payload_bytes=64 * 1024)

    async def run(self, ctx, args):
        if len(args["invoice_id"]) > 256:
            return ToolResult(content="Input exceeds the 256-character limit.", is_error=True)
        PAID.setdefault(ctx.idempotency_key, args["invoice_id"])  # A repeat pays nothing.
        return ToolResult(content=f"Paid {args['invoice_id']}.")

approval_policy = AlwaysRequireApprovalToolPolicy(
    tools=["pay_invoice"], expires_in_seconds=72 * 3600)
```

Register the tool with `tool_policy=approval_policy`. Each call pauses the
session durably until `app.resolve_tool_approval(...)` or the Developer dashboard's
Pending page resolves it; resolving after the expiry denies the call.
`ctx.idempotency_key` is stable for the call, including Cayu's one replay after
a crash, so pass it to your provider. Deep guides: `cayu guide durable-operations`
and `cayu guide tool-effects`.

## Approval without a model turn

```python
async def propose_payment(app, invoice_id):
    """Workflow code that already knows the exact call asks a person to approve it."""
    return await app.propose_tool_call(  # .status is "pending_approval" until decided.
        "invoices", "pay_invoice", {"invoice_id": invoice_id}, session_id=f"pay-{invoice_id}")
```

No model is called, so the text can't drift. The agent's tool policy decides
as for a model-raised call (guards apply, including to an approver's edits),
and the pause is an ordinary approval: the Pending page or
`app.resolve_tool_approval(...)` settles it, a restarted worker resumes it, and
`ToolCallProposalOutcome.from_events(...)` reads where it stands. Pass
`task_id`, `task_worker_id` and `task_lease_expires_at` to attach a task, or
run `cayu.approvals.tool_call_proposal_request(...)` yourself. The arguments
are stored with the session, so keep secrets out of them.

## React to approval outcomes

```python
from cayu.events import EventType
from cayu.observability import EventWatcher, serve_event_watchers
from cayu.sessions import EventQuery

OUTCOMES = {}  # Stands in for your records.
decided = (EventType.TOOL_CALL_APPROVED, EventType.TOOL_CALL_APPROVAL_DENIED)


def record_outcome(context):
    event = context.record.event
    OUTCOMES[event.session_id] = event.type.value  # Keyed, so a redelivery is harmless.


approval_outcomes = EventWatcher(
    name="approval-outcomes", query=EventQuery(event_types=decided), handler=record_outcome)


async def run_watchers(app, stop):  # A [tool.cayu.workers] entrypoint.
    await serve_event_watchers(app, stop)
```

Call `app.register_event_watcher(approval_outcomes)` in the factory. `cayu
serve` runs registered watchers while it serves, so a denial on the Pending
page reaches the handler there; `run_watchers` runs them in a worker. Delivery
is ordered, at least once, retried up to `max_attempts` and leased by watcher
name. An app built on `open_application_stores(...)`'s session store keeps its
watcher cursors in the same SQLite file or PostgreSQL database
(`stores.event_watcher_store`), so a restart does not deliver old events again.
Otherwise pass `CayuApp(event_watcher_store=SQLiteEventWatcherStore(path))` or
`PostgresEventWatcherStore` (from `cayu.storage`). Without either the store is in
memory, so a restart delivers every old matching event again; the example keeps
it only because its test app is too.

## Waiting for an outside reply

```python
from cayu.approvals import UserInputResponse
from cayu.sessions import PendingActionKind, PendingActionQuery
from cayu.tools import UserInputTool

ask_user = UserInputTool()  # When the model calls ask_user, the session pauses.


async def record_reply(app, session_id, reply_text):
    """Call this when the email, document or receipt arrives, days later."""
    page = await app.session_store.query_pending_actions(
        PendingActionQuery(session_id=session_id, kind=PendingActionKind.USER_INPUT)
    )
    [question] = page.actions
    reply = UserInputResponse(session_id=session_id, input_id=question.input_id, answer=reply_text)
    async for _ in app.resolve_user_input(reply):
        pass
```

The question is stored with the session, so any process can answer it and the
run continues from there. Questions never expire; give each a deadline timer.
If the answering process dies mid-turn, the question is closed but the session
stays `running`. `app.recover_abandoned_sessions(...)` settles sessions with no
live owner, and a `ResumeRequest` with the same `budget_limits` continues them;
`expected_run_epoch` keeps two workers from both continuing one session. The
preset's timer worker does both. Deep guide: `cayu guide human-attention`.

## Timers

```python
from datetime import timedelta

from cayu.tasks import TaskCreate, TaskSchedulePolicy


def reminder_task(action, remind_after=timedelta(hours=24), deadline=timedelta(hours=72)):
    paused_at = action.event.event.timestamp  # Fixed, so a retry builds the same task.
    return TaskCreate(
        task_id=f"reminder-{action.attention_id}",
        type="pending-action-timer",
        input={"session_id": action.session.id, "attention_id": action.attention_id},
        available_at=paused_at + remind_after,
        schedule_policy=TaskSchedulePolicy(expires_at=paused_at + deadline),
    )
```

Create it with `await app.create_task(...)`; creating it again returns the same
task. Run due tasks with `run_task_worker(app, app.task_store, handler, ...)`.
The task store, not a process timer, decides when a task is due, so timers
survive restarts, and `expires_at` drops a reminder that would land after the
deadline. The handler checks the action and the session's status before acting;
the preset's `operations/timers.py` runs as `cayu worker timers`.
Deep guide: [task scheduling](https://github.com/cayu-dev/cayu/blob/main/docs/task-scheduling.md).

## Event retry policy

Choose the event's completion contract before setting timeouts. A service that
must eventually deliver a result may intentionally retry for hours or days.
A service with a deadline must instead expose a terminal failure when it gives
up. Ordinary root tasks receive a bounded event automatically: three attempts
and 1,800 seconds by default. An explicit `retry_budget` selects the workload's
limits; otherwise `retry_policy` supplies them when present. There is no universal
ten-minute cap. Two identical jobs can be independent work.

`TaskCreate.event_id` identifies a logical event within its task type and access
scope. A nonblank string `input["event_id"]` supplies that identity when the explicit
field is absent; `retry_key` overrides it. With neither identity, each root is an
independent event. Resubmission must preserve identity: Cayu does not infer it
from equal payloads. Managed schedules, session continuations, child work and
verified-work contracts retain their own contracts; explicit retry keys can
associate scheduled or child queue work with an event.

Prefer submitting an event once and letting `TaskRetryPolicy` own its failed
attempts. Its elapsed budget spans the retry series and the worker enforces it
on the active handler. Pick both limits from the application's requirements:

```python
from cayu.tasks import (
    TaskCreate, TaskQuery, TaskRetryPolicy, TaskRetryAttemptReport,
    TaskRetryAttemptDisposition, run_task_worker,
)


async def submit_event(app, event_id, request, *, max_attempts, deadline_seconds):
    return await app.create_task(TaskCreate(
        task_id=f"delivery-event-{event_id}", type="event-delivery",
        input={"request": request, "event_id": event_id},
        retry_policy=TaskRetryPolicy(
            max_attempts=max_attempts, max_elapsed_seconds=float(deadline_seconds),
        ),
    ))


async def run_event_jobs(app, deliver, *, max_tasks=None):
    async def handle_event(app, task, worker_id):
        result = await deliver(task.input["request"], event_id=task.input["event_id"])
        return TaskRetryAttemptReport(
            idempotency_key=f"delivered-{task.id}",
            disposition=TaskRetryAttemptDisposition.SUCCEEDED,
            result=result,
        )

    await run_task_worker(
        app, app.task_store, handle_event, worker_id="deliveries",
        query=TaskQuery(type="event-delivery"), max_tasks=max_tasks,
    )
```

Supply an async `deliver(request, *, event_id)` that returns a JSON object and
uses the stable event ID to make external effects idempotent. This retry-series
worker returns `TaskRetryAttemptReport`; it must not call `complete_managed_task`
like the ordinary invoice worker below. Classify expected retryable failures
with a retryable report; unexpected exceptions follow the worker's classification.

Retain the event-to-task identity durably and inspect that task's retry series;
do not have a sweeper create a new root after the series has given up. Handlers
must report failed attempts through the task retry APIs (or let the worker
handle their exceptions), not complete failed work as successful.

If the application needs to own resubmission, pass a stable `retry_key` and an
explicit `TaskRetryBudget` on every submission, or use `event_id`. The first
admission fixes policy and deadline. Native successors and resubmissions consume
the same attempt budget; restarting a worker cannot reset the clock. Workers
cancel active handlers when the shared deadline expires, including time spent
waiting for human input. Pending and paused tasks expire during worker maintenance.
Keep a worker running for maintenance; a stopped process cannot enforce a deadline.

Attempt exhaustion becomes durable when the final admitted attempt fails.
Using all slots blocks new submissions without cancelling admitted attempts.
Success closes the event and refuses duplicate submissions with
`TaskEventAlreadyCompleted`. Inspect `task.invocation.event_retry_key` with
`cayu inspect --retry-key KEY` for the policy, deadline, counts and terminal reason.

`TaskRetryBudget.unlimited()` explicitly opts out of attempt and elapsed limits;
success deduplication still applies. Cancellation requests do not prove cleanup:
a handler that resists cancellation retains its owner until it finishes. Handler
cleanup must await its owned operations; returning while detached work runs does
not make that work safe. External effects still require idempotency.

`cayu check` warns with `TASK_EVENT_RETRY_POLICY_UNDECLARED` when source declares
a bounded worker and unkeyed, policy-free task creation for the same literal
task type. It is an advisory source check, not proof that the application retries;
dynamic declarations may not be visible. It changes no runtime policy.

## Time limit for worker runs

```python
import asyncio

from cayu import Message, RunRequest, run_to_completion
from cayu.deadlines import ExecutionDeadline
from cayu.sessions import SessionStatus
from cayu.tasks import TaskCreate, TaskQuery, TaskRetryBudgetExhausted
from cayu.tasks import complete_managed_task, run_task_worker

JOB_TIMEOUT_S = 300  # The longest one handler call may take: retries and grace included.
JOB_GRACE_S = 30  # Its last 30 seconds, kept for finishing a delivery.
DELIVERED = {}  # Stands in for your outbox and the system you deliver to.


async def handle_job(app, task, worker_id):
    while True:  # Retry until one run works; the time limit ends the loop.
        outcome = await run_to_completion(app, RunRequest(
            agent_name="invoices", messages=[Message.text("user", task.input["request"])],
            execution_deadline=ExecutionDeadline.after(120),  # One attempt.
        ))
        if outcome.status is SessionStatus.INTERRUPTED:
            await complete_managed_task(app.task_store, task, worker_id,
                                        {"paused_session": outcome.session_id})
            return  # Continue this same session when its pending action is resolved.
        if outcome.ok:
            break
        await asyncio.sleep(1)
    DELIVERED.setdefault(task.id, outcome.session_id)  # Keyed, so a repeat is harmless.
    await complete_managed_task(app.task_store, task, worker_id, {"session": outcome.session_id})


async def run_jobs(app, stop=None):
    await run_task_worker(app, app.task_store, handle_job, worker_id="jobs-1",
                          query=TaskQuery(type="invoice-job"), stop=stop,
                          execution_timeout_s=JOB_TIMEOUT_S, execution_grace_s=JOB_GRACE_S)


async def retry_job(app, event_id, attempt, request):  # Your sweeper, after a failed task.
    try:  # Choose TaskRetryBudget(...) on the first admission and keep it on every retry.
        await app.create_task(TaskCreate(task_id=f"invoice-{event_id}-{attempt}",
            type="invoice-job", input={"request": request}, retry_key=f"invoice-{event_id}"))
    except TaskRetryBudgetExhausted as exhausted:
        return exhausted.state  # Stop submitting; already-admitted attempts may still be active.
```

Runs are unattached (`RunRequest.task_id` is absent). An interruption completes
this launch task with a durable session reference; resolve that session's pending
action, if any, and deliver the result in its continuation owner. Never retry a
pause as a new payment session. Successful runs deliver before completing the task.
Without `execution_timeout_s`, the event deadline still bounds a hung call. Which limit ends what:

- `execution_timeout_s` is the whole call, grace included: at 270 of 300 seconds
  new runs are refused and `current_handler_time_limit().expired` is true; at 300
  the handler is cancelled and a running session ends `session.interrupted`. An
  unsettled task fails with `ExecutionDeadlineExceeded`, or its `TaskRetryPolicy`
  decides. Grace defaults to 30 s or half the limit; a larger one is refused.
- A run's `execution_deadline` ends that session, here one attempt. Runs
  without one end when new runs are refused.
- `retry_key` ends the event: tasks with one key (the event's ID, not the
  task's) share a `TaskRetryBudget`, by default 3 tasks within 30 minutes. The
  next is refused with `TaskRetryBudgetExhausted`; the key gives up after the
  last admitted failure or when the shared deadline expires. Choose a different
  budget with `retry_budget=` on the first admission. Inspect the durable reason
  with `cayu inspect --retry-key KEY`.
- `RetryPolicy` ends a model call that keeps hitting rate limits or outages:
  by default, at most 5 attempts. Rate limits and overloads have a 300-second
  retry budget (`max_retry_elapsed_s`), counted from the first attempt, with
  each delay capped at 60 seconds (`max_delay_s`). Other retryable failures use
  7.5 to 9.5 seconds of ordinary backoff in total; `Retry-After` can extend it.
  These are configurable retry limits, not a deadline that cancels a model call.
  Choose them for the provider and workload before retrying whole sessions;
  see `cayu guide providers#provider-retries`.

A delivery still running at the limit is cancelled and may or may not have
happened, so make it idempotent. Don't catch `asyncio.CancelledError`: a
handler that keeps running keeps its task claimed, with a `task_handler_timeout`
cancellation request, until it returns. A run that pauses for a person returns;
settling gives a run without its own `execution_deadline` back the time it had
left, within its original event deadline. An explicit session deadline can be
replaced, but replacement cannot extend the enclosing event deadline. Without
`run_task_worker`, wrap the handler in `async with HandlerTimeLimit(300, grace_s=30):`
and fail the task on `ExecutionDeadlineExceeded`. The generated timer worker sets
no limit, so the sessions it continues inline follow their own deadlines.

## Spend cap

```python
from decimal import Decimal

from cayu import Message, RunRequest
from cayu.budgets import BudgetLimit, default_price_book

spend_cap = BudgetLimit(
    scope="session", max_estimated_cost=Decimal("5"), pricing=default_price_book()
)
request = RunRequest(
    agent_name="invoices",
    messages=[Message.text("user", "Pay invoice INV-7.")],
    budget_limits=(spend_cap,),
)
```

Cayu checks the cap around each model call with its bundled, dated price book.
At the cap the session stops `interrupted` with `session.limit_reached`, and
that step's tool calls never run. Spend from every process that ran the
session counts. Unpriced models are refused unless the limit sets
`allow_unpriced=True`. Deep guide: `cayu guide references#cost-control`.

## Restart tests

```python
import asyncio

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu.providers import ModelStreamEvent
from cayu.testing import RestartHarness


def build_app(*, provider, session_store, task_store):
    app = CayuApp(session_store=session_store, task_store=task_store)
    app.register_provider(provider, default=True)
    agent = AgentSpec(name="invoices", model="scripted-model")
    app.register_agent(agent, tools=[PayInvoice(), ask_user], tool_policy=approval_policy)
    return app


def test_payment_survives_restart(tmp_path):
    pay = ModelStreamEvent.tool_call(name="pay_invoice", arguments={"invoice_id": "INV-7"})
    script = [[pay, ModelStreamEvent.completed()], [ModelStreamEvent.completed()]]
    say = [Message.text("user", "Pay INV-7.")]
    run = RunRequest(agent_name="invoices", session_id="inv-7", messages=say)

    async def scenario():
        async with RestartHarness(build_app, store_dir=tmp_path, model_script=script) as h:
            paused = await h.run(run)
            await h.restart()
            assert (await h.approve(paused.pending_approvals[0], actor="ops")).completed
            await h.assert_tool_ran_once("inv-7", "pay_invoice")

    asyncio.run(scenario())
```

Write one test per pause point: an approval, a question answered with
`h.answer_user_input(...)`, and a timer. With no clock control, a timer test
moves the task to now with `app.reschedule_task(...)` and runs the worker once.
The preset kills child processes to test recovery: `cayu guide restart-tests`.
