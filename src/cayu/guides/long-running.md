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
name. Give the app a durable cursor with
`CayuApp(event_watcher_store=SQLiteEventWatcherStore(path))` (from `cayu.storage`)
on the same SQLite file as its other stores, or `PostgresEventWatcherStore` in a
deployment. The default store is in memory, so a restart delivers every old
matching event again; the example keeps it only because its test app is too.

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
up. Cayu does not infer event identity from equal inputs or impose a universal
event deadline. Two identical jobs can be independent work.

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
explicit `TaskRetryBudget` on every submission. That budget only decides whether
another task may be created: its elapsed limit does not cancel an already-running
handler. Carry the event's remaining execution deadline into attempts as well.
`TaskRetryBudget.unlimited()` explicitly opts out of admission limits for an
event that must keep retrying. A handler timeout alone does not bound resubmission.

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
    try:  # Same retry_key for every task of this event; TaskRetryBudget(...) changes the cap.
        await app.create_task(TaskCreate(task_id=f"invoice-{event_id}-{attempt}",
            type="invoice-job", input={"request": request}, retry_key=f"invoice-{event_id}"))
    except TaskRetryBudgetExhausted as exhausted:
        return exhausted.state  # Given up for good: record it and stop retrying.
```

Runs are unattached (`RunRequest.task_id` is absent). An interruption completes
this launch task with a durable session reference; resolve that session's pending
action, if any, and deliver the result in its continuation owner. Never retry a
pause as a new payment session. Successful runs deliver before completing the task.
Without `execution_timeout_s`, a hung call stays claimed. Which limit ends what:

- `execution_timeout_s` is the whole call, grace included: at 270 of 300 seconds
  new runs are refused and `current_handler_time_limit().expired` is true; at 300
  the handler is cancelled and a running session ends `session.interrupted`. An
  unsettled task fails with `ExecutionDeadlineExceeded`, or its `TaskRetryPolicy`
  decides. Grace defaults to 30 s or half the limit; a larger one is refused.
- A run's `execution_deadline` ends that session, here one attempt. Runs
  without one end when new runs are refused.
- `retry_key` ends the event: tasks with one key (the event's ID, not the
  task's) share a `TaskRetryBudget`, by default 3 tasks within 30 minutes. The
  next is refused with `TaskRetryBudgetExhausted` and the key stays given up, as
  `cayu inspect --retry-key KEY` shows. Raise it with `retry_budget=`.
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
left, while an explicit deadline keeps running (settle with a new one). Without
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
