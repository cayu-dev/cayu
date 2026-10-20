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
        input_schema={"type": "object", "properties": {"invoice_id": {"type": "string"}}},
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="pay_invoice", behavior_version="1", implementation_version="1"),
    )

    async def run(self, ctx, args):
        PAID.setdefault(ctx.idempotency_key, args["invoice_id"])  # A repeat pays nothing.
        return ToolResult(content=f"Paid {args['invoice_id']}.")


approval_policy = AlwaysRequireApprovalToolPolicy(
    tools=["pay_invoice"], expires_in_seconds=72 * 3600)
```

Register the tool with `tool_policy=approval_policy`. Each call pauses the
session durably until `app.resolve_tool_approval(...)` or the control plane's
Pending page resolves it; resolving after the expiry denies the call.
`ctx.idempotency_key` is stable for the call, including Cayu's one replay after
a crash, so pass it to your provider. Deep guides: `cayu guide durable-operations`
and `cayu guide tool-effects`.

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

## Time limit for worker runs

```python
from cayu import Message, RunRequest, SessionStatus, run_to_completion
from cayu.tasks import TaskHandlerOutcome, TaskQuery, run_task_worker

RUN_TIMEOUT_S = 300  # The longest one job may take, from handler start.


async def handle_job(app, task, worker_id):
    outcome = await run_to_completion(app, RunRequest(
        agent_name="invoices", messages=[Message.text("user", task.input["request"])],
        task_id=task.id, task_worker_id=worker_id, task_lease_expires_at=task.lease_expires_at,
    ))
    if outcome.status is SessionStatus.INTERRUPTED:
        return TaskHandlerOutcome.SESSION_INTERRUPTED
    return None


async def run_jobs(app, stop=None):
    await run_task_worker(app, app.task_store, handle_job, worker_id="jobs-1",
                          query=TaskQuery(type="invoice-job"),
                          execution_timeout_s=RUN_TIMEOUT_S, stop=stop)
```

Without `execution_timeout_s`, a hung tool or model stream keeps the task
`claimed` forever. With it, every `RunRequest` the handler builds gets that
deadline unless it sets its own. At expiry the run ends `session.interrupted`
with `limit_reached`, the runtime fails the attached task with
`ExecutionDeadlineExceeded`, and a retry in the same handler call is refused.
Don't retry past it. The deadline also counts time paused for a person, so a
job that waits for approval needs a limit that covers the wait.

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
