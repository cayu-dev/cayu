"""Files for the ``long-running`` preset of ``cayu new``.

The preset starts from the agent preset's files and adds one small working
example of each long-running pattern: a side effect behind approval with an
idempotency key, a reply the agent waits for, reminder and deadline timers, a
per-session spend cap, and a restart test for each pause point. Every piece is
an ordinary library part wired in its owning module, so a project can keep,
replace, or delete each one on its own.
"""

from __future__ import annotations

from collections.abc import Callable

from cayu.cli.scaffold_plan import ApplicationPlan

_PAYMENTS_INTEGRATION_PY = '''"""Client for the payment service that the pay_invoice tool calls.

Replace LocalPaymentService with your payment provider's client and pass the
tool's idempotency key through as the provider's idempotency key, so a repeated
call returns the first payment instead of paying twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class PaymentReceipt:
    payment_id: str
    invoice_id: str
    amount_cents: int


class PaymentService(Protocol):
    async def pay(
        self, *, invoice_id: str, amount_cents: int, idempotency_key: str
    ) -> PaymentReceipt: ...


class LocalPaymentService:
    """In-memory stand-in for local runs and tests. It is not durable.

    It treats the idempotency key the way a payment provider does: the same key
    returns the first receipt and never pays again.
    """

    def __init__(self) -> None:
        self.payments: dict[str, PaymentReceipt] = {}

    async def pay(
        self, *, invoice_id: str, amount_cents: int, idempotency_key: str
    ) -> PaymentReceipt:
        existing = self.payments.get(idempotency_key)
        if existing is not None:
            return existing
        receipt = PaymentReceipt(
            payment_id=f"payment-{len(self.payments) + 1}",
            invoice_id=invoice_id,
            amount_cents=amount_cents,
        )
        self.payments[idempotency_key] = receipt
        return receipt
'''

_PAYMENTS_TOOL_PY = '''"""pay_invoice: a side effect that runs only after approval, once per call."""

from typing import Any

from cayu import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.runtime import ExecutionProfileBehaviorIdentity

from integrations.payments import PaymentService


class PayInvoiceTool(Tool):
    spec = ToolSpec(
        name="pay_invoice",
        description="Pay one vendor invoice. A person approves each payment first.",
        input_schema={
            "type": "object",
            "properties": {
                "invoice_id": {"type": "string"},
                "amount_cents": {"type": "integer", "minimum": 1},
            },
            "required": ["invoice_id", "amount_cents"],
            "additionalProperties": False,
        },
        # The payment service collapses repeats on the idempotency key below.
        effect=ToolEffect.IDEMPOTENT,
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="__PROJECT_NAME__.pay_invoice",
            behavior_version="1",
            implementation_version="1",
        ),
    )

    def __init__(self, payments: PaymentService) -> None:
        super().__init__()
        self.payments = payments

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        if ctx.idempotency_key is None:
            return ToolResult(
                content="No idempotency key; refusing to pay.", is_error=True
            )
        receipt = await self.payments.pay(
            invoice_id=args["invoice_id"],
            amount_cents=args["amount_cents"],
            # Cayu keeps this key for the call, including a replay after a crash.
            idempotency_key=ctx.idempotency_key,
        )
        return ToolResult(
            content=f"Paid invoice {receipt.invoice_id}: payment {receipt.payment_id}.",
            structured={"payment_id": receipt.payment_id},
        )
'''

_TOOLS_REGISTRATION_PY = '''"""Native model-callable tools selected for the starter agent."""

from cayu.tools import UserInputTool
from cayu import Tool

from integrations.payments import PaymentService
from tools.payments import PayInvoiceTool


def build_agent_tools(*, payments: PaymentService) -> tuple[Tool, ...]:
    """ask_user waits for an outside reply; pay_invoice is the side effect."""

    return (UserInputTool(), PayInvoiceTool(payments))


def external_effect_tool_names() -> tuple[str, ...]:
    """Return tools whose calls pause for approval under policies/tools.py.

    To make a tool ask a person before it acts, add its name here (or generate it
    with ``cayu generate tool NAME --effect external``, which does this for you).
    """

    return ("pay_invoice",)
'''

_TOOL_POLICY_PY = '''"""Tool exposure and authorization policy for registered specialists."""

from datetime import timedelta

from cayu.tools import (
    AlwaysRequireApprovalToolPolicy,
    GuardedToolPolicy,
    RequiredArguments,
    StaticToolExposurePolicy,
    ToolExposurePolicy,
    ToolPolicy,
)

# An approval nobody answers expires. Resolving it later denies the call.
APPROVAL_EXPIRES_AFTER = timedelta(hours=72)


def build_tool_policy(external_tool_names: tuple[str, ...]) -> ToolPolicy:
    """Deny malformed calls, then pause before each listed external effect."""

    return GuardedToolPolicy(
        # Approval cannot repair a call with missing arguments.
        guards=(RequiredArguments({"ask_user": ("question",)}),),
        then=AlwaysRequireApprovalToolPolicy(
            tools=external_tool_names,
            expires_in_seconds=APPROVAL_EXPIRES_AFTER.total_seconds(),
        ),
    )


def build_tool_exposure_policy(tools: tuple[str, ...]) -> ToolExposurePolicy:
    """Expose one explicit list; registration alone never grants model visibility."""

    return StaticToolExposurePolicy(profile_id="standard-local-v1", tools=tools)
'''

_BUDGETS_PY = '''"""Per-session spend cap.

Cayu checks the cap around each model call. A session that reaches it stops
`interrupted` with a `session.limit_reached` event, and the tool calls of the
step that reached it never run. The cap counts spend from every process that
ran the session.
"""

from decimal import Decimal

from cayu.budgets import BudgetLimit, ModelPrice, PriceBook, default_price_book

from configuration import configured_model

SESSION_SPEND_CAP_USD = "5.00"


def session_spend_cap(*, pricing: PriceBook | None = None) -> BudgetLimit:
    """Cap one session's estimated model spend, priced by project_price_book().

    Models missing from the price book are refused rather than run unpriced.
    A resumed session must restate the same cap, so use one cap per session.
    """

    return BudgetLimit(
        scope="session",
        max_estimated_cost=Decimal(SESSION_SPEND_CAP_USD),
        pricing=project_price_book() if pricing is None else pricing,
    )


def project_price_book() -> PriceBook:
    """Cayu's bundled, dated prices plus this project's own entries.

    Add entries here for other models or negotiated rates. The scripted entry
    prices the ScriptedModelProvider that tests and evals use; no real provider
    is named "scripted".
    """

    bundled = default_price_book()
    return PriceBook(
        price_book_version=f"{bundled.price_book_version}+project",
        generated_at=bundled.generated_at,
        prices=(*bundled.prices, *scripted_price_book().prices),
        contextual_pricing_requirements=bundled.contextual_pricing_requirements,
        resource_mappings=bundled.resource_mappings,
    )


def scripted_price_book(usd_per_million_tokens: str = "1") -> PriceBook:
    """Price the ScriptedModelProvider that hermetic tests and evals use."""

    return PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="scripted",
                model=configured_model(),
                match="exact",
                input_per_million=Decimal(usd_per_million_tokens),
                output_per_million=Decimal(usd_per_million_tokens),
            ),
        )
    )
'''

_WORKFLOW_PY = '''"""Start and continue invoice-payment sessions with their spend cap."""

from cayu import Message, ResumeRequest, RunRequest
from cayu.budgets import BudgetLimit

from agents.agent import AGENT
from policies.budgets import session_spend_cap


def invoice_request(
    text: str, *, session_id: str, spend_cap: BudgetLimit | None = None
) -> RunRequest:
    """Build the request that starts one invoice job.

    The spend cap stays with the session when it pauses and resumes in
    another process.
    """

    return RunRequest(
        agent_name=AGENT.name,
        session_id=session_id,
        messages=[Message.text("user", text)],
        budget_limits=(spend_cap if spend_cap is not None else session_spend_cap(),),
    )


CONTINUE_MESSAGE = "Continue the job from where it stopped."


def continue_request(
    session_id: str, *, expected_run_epoch: int | None = None
) -> ResumeRequest:
    """Continue a session whose process died and that Cayu has since recovered.

    Recovery settles the lost model call without running it again, so the
    session needs a new turn. The request restates the session's spend cap.
    Pass the run epoch you inspected so the resume is refused with
    SessionRunFenced if another process continued the session first.
    """

    return ResumeRequest(
        session_id=session_id,
        messages=[Message.text("user", CONTINUE_MESSAGE)],
        budget_limits=(session_spend_cap(),),
        expected_run_epoch=expected_run_epoch,
    )
'''

_PROMPT_PY = '''"""System and developer prompt material for the starter agent."""

SYSTEM_PROMPT_PARTS: tuple[str, ...] = (
    "You pay vendor invoices.",
    "If the invoice ID or amount is missing, ask the requester with ask_user "
    "and wait for the reply.",
    "Call pay_invoice once per invoice. A person approves each payment before it runs.",
)
'''

_TIMERS_PY = '''"""Reminder and deadline timers for pending approvals and questions.

Each pending approval or question gets two one-shot tasks in the task store: a
reminder and a deadline. The store, not this process, decides when a task is
due, so timers survive restarts and missed polls. The same worker also
recovers sessions whose process died mid-run and continues them. Run it with
`uv run --no-sync cayu worker timers`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
from contextlib import aclosing, suppress
from datetime import timedelta

from cayu import CayuApp
from cayu.approvals import (
    ResolutionActor,
    ResolutionActorSource,
    ToolApprovalDecision,
    ToolApprovalRequest,
    UserInputResponse,
)
from cayu.sessions import (
    PendingActionKind,
    PendingActionQuery,
    PendingActionRecord,
    SessionQuery,
    SessionRunFenced,
    SessionStatus,
)
from cayu.tasks import (
    Task,
    TaskCreate,
    TaskQuery,
    TaskSchedulePolicy,
    complete_managed_task,
    run_task_worker,
)

from agents.agent import AGENT
from policies.tools import APPROVAL_EXPIRES_AFTER
from workflows.invoices import continue_request

TIMER_TASK_TYPE = "pending-action-timer"
REMIND_AFTER = timedelta(hours=24)
# Cayu expires approvals itself (policies/tools.py). A question waits until
# someone answers it, so this deadline answers it for them.
REPLY_DEADLINE = timedelta(days=7)
NO_REPLY_ANSWER = "No reply arrived before the deadline."
# A running session with no live owner and no activity for this long is
# treated as abandoned: its process died. Live sessions are never taken over.
ABANDONED_AFTER = timedelta(minutes=15)
SWEEP_INTERVAL_S = 60.0
# Timer handlers consume continuations inline. Leave their handler unbounded
# so user sessions keep their own deadlines through approval and input recovery.

logger = logging.getLogger(__name__)


async def schedule_timers(app: CayuApp, session_id: str | None = None) -> list[Task]:
    """Create the reminder and deadline for every open approval or question.

    Safe to repeat: task IDs and due times come from the pending action itself,
    so a retry returns the tasks already created.
    """

    tasks: list[Task] = []
    cursor: str | None = None
    while True:
        page = await app.session_store.query_pending_actions(
            PendingActionQuery(session_id=session_id, cursor=cursor, limit=200)
        )
        for action in page.actions:
            for request in _timer_requests(action):
                tasks.append(await app.create_task(request))
        if not page.has_more or page.next_cursor is None:
            return tasks
        cursor = page.next_cursor


def _timer_requests(action: PendingActionRecord) -> tuple[TaskCreate, ...]:
    if action.attention_id is None:
        return ()
    if action.kind is PendingActionKind.TOOL_APPROVAL:
        wait = APPROVAL_EXPIRES_AFTER
    elif action.kind is PendingActionKind.USER_INPUT:
        wait = REPLY_DEADLINE
    else:
        return ()
    paused_at = action.event.event.timestamp
    deadline = paused_at + wait
    target = {"session_id": action.session.id, "attention_id": action.attention_id}
    return (
        TaskCreate(
            task_id=f"reminder-{action.attention_id}",
            type=TIMER_TASK_TYPE,
            title=f"Remind: {action.title}",
            input={**target, "timer": "reminder"},
            available_at=paused_at + REMIND_AFTER,
            # A reminder still waiting at the deadline is dropped, not sent late.
            schedule_policy=TaskSchedulePolicy(expires_at=deadline),
        ),
        TaskCreate(
            task_id=f"deadline-{action.attention_id}",
            type=TIMER_TASK_TYPE,
            title=f"Deadline: {action.title}",
            input={**target, "timer": "deadline"},
            available_at=deadline,
            schedule_policy=TaskSchedulePolicy(),
        ),
    )


async def send_reminder(action: PendingActionRecord) -> None:
    """Deliver a reminder. Replace the log line with your email or chat sender."""

    logger.info("Reminder: %s (session %s)", action.title, action.session.id)


async def handle_timer(app: CayuApp, task: Task, worker_id: str) -> None:
    """Run one due timer, then complete its task."""

    session_id = task.input["session_id"]
    action = await _pending_action(app, session_id, task.input["attention_id"])
    if action is None:
        # Someone already answered. If the process that answered died before
        # the session finished its turn, recover and continue it here.
        continued = await continue_if_abandoned(app, session_id)
        outcome = "continued" if continued else "already_resolved"
    elif task.input["timer"] == "reminder":
        await send_reminder(action)
        outcome = "reminded"
    else:
        await _close_at_deadline(app, action)
        outcome = "closed"
    # The session may have paused again; give the new pause its own timers.
    await schedule_timers(app, session_id)
    session = await app.session_store.load(session_id)
    # "running" here means a live process owns the session, or it has not been
    # idle for ABANDONED_AFTER yet; the worker's sweep continues it if it stalls.
    status = None if session is None else session.status.value
    assert app.task_store is not None
    await complete_managed_task(
        app.task_store, task, worker_id, {"outcome": outcome, "session_status": status}
    )


async def continue_abandoned_sessions(
    app: CayuApp, *, inactive_for: timedelta = ABANDONED_AFTER
) -> list[str]:
    """Recover sessions whose process died, then continue this agent's ones.

    `recover_abandoned_sessions` skips sessions with a live execution owner and
    sessions active within `inactive_for`, so a running worker keeps its work.
    It settles the lost model call without running it again and leaves the
    session interrupted. The second pass continues every interrupted session
    of this agent that recovery stopped, so a session whose recovering process
    died before continuing it is picked up by a later sweep.
    """

    await app.recover_abandoned_sessions(
        inactive_for_seconds=int(inactive_for.total_seconds())
    )
    continued: list[str] = []
    cursor: str | None = None
    while True:
        page = await app.session_store.list_sessions(
            SessionQuery(
                status=SessionStatus.INTERRUPTED,
                agent_name=AGENT.name,
                cursor=cursor,
                limit=100,
            )
        )
        for session in page.sessions:
            try:
                if await _continue_recovered(app, session.id):
                    continued.append(session.id)
            except Exception:
                # For example a session started without this project's spend
                # cap: resuming it with a different cap is refused. It stays
                # interrupted and is retried on the next sweep.
                logger.exception("Could not continue recovered session %s.", session.id)
        if page.next_cursor is None:
            return continued
        cursor = page.next_cursor


async def continue_if_abandoned(
    app: CayuApp, session_id: str, *, inactive_for: timedelta | None = None
) -> bool:
    """Recover and continue one session if its process died; return whether it ran."""

    session = await app.session_store.load(session_id)
    if session is None:
        return False
    if session.status in (SessionStatus.RUNNING, SessionStatus.INTERRUPTING):
        window = ABANDONED_AFTER if inactive_for is None else inactive_for
        return session_id in await continue_abandoned_sessions(app, inactive_for=window)
    # A recovery that finished before its continuation started.
    return await _continue_recovered(app, session_id)


async def _continue_recovered(app: CayuApp, session_id: str) -> bool:
    session = await app.session_store.load(session_id)
    if (
        session is None
        or session.agent_name != AGENT.name
        or session.status is not SessionStatus.INTERRUPTED
    ):
        return False
    outcome = await app.session_store.summarize_outcome(session_id)
    stopped = outcome.terminal_event
    # Continue only interruptions that recovery made. An operator's interrupt,
    # a reached spend cap, or a tool call left for a person must stay stopped.
    if (
        stopped is None
        or stopped.event.payload.get("recovered") is not True
        or outcome.details.get("manual_recovery_required") is True
    ):
        return False
    pending = await app.session_store.query_pending_actions(
        PendingActionQuery(session_id=session_id, limit=1)
    )
    if pending.actions:
        return False  # Paused for a person, not stalled.
    # Bind the resume to the run epoch inspected above. If another process
    # continued the session since, Cayu refuses this resume instead of
    # starting a second turn.
    request = continue_request(session_id, expected_run_epoch=session.run_epoch)
    try:
        async with aclosing(app.resume(request)) as events:
            async for _ in events:
                pass
    except SessionRunFenced:
        return False
    return True


async def _pending_action(
    app: CayuApp, session_id: str, attention_id: str
) -> PendingActionRecord | None:
    page = await app.session_store.query_pending_actions(
        PendingActionQuery(session_id=session_id, limit=200)
    )
    return next(
        (item for item in page.actions if item.attention_id == attention_id), None
    )


async def _close_at_deadline(app: CayuApp, action: PendingActionRecord) -> None:
    if action.kind is PendingActionKind.TOOL_APPROVAL:
        assert action.approval_id and action.round_id and action.tool_call_id
        # Past its expiry Cayu records this as approval_expired and denies the call.
        stream = app.resolve_tool_approval(
            ToolApprovalRequest(
                session_id=action.session.id,
                approval_id=action.approval_id,
                tool_round_id=action.round_id,
                tool_call_id=action.tool_call_id,
                decision=ToolApprovalDecision.DENY,
                reason="No decision before the deadline.",
                resolved_by=_timer_actor(),
            )
        )
    else:
        assert action.input_id
        stream = app.resolve_user_input(
            UserInputResponse(
                session_id=action.session.id,
                input_id=action.input_id,
                answer=NO_REPLY_ANSWER,
                resolved_by=_timer_actor(),
            )
        )
    async with aclosing(stream) as events:
        async for _ in events:
            pass


def _timer_actor() -> ResolutionActor:
    return ResolutionActor(subject="timers", source=ResolutionActorSource.SYSTEM)


async def run_timer_worker(app: CayuApp, stop: asyncio.Event) -> None:
    """`cayu worker timers`: schedule timers, run due ones, continue stalled sessions."""

    async def sweep() -> None:
        # Pauses happen in any process (the server, run.py, a worker), so find
        # them here instead of relying on every caller to schedule timers.
        while not stop.is_set():
            try:
                await continue_abandoned_sessions(app)
            except Exception:
                logger.exception(
                    "Continuing abandoned sessions failed; retrying next sweep."
                )
            await schedule_timers(app)
            with suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), SWEEP_INTERVAL_S)

    assert app.task_store is not None
    async with asyncio.TaskGroup() as group:
        group.create_task(sweep())
        group.create_task(
            run_task_worker(
                app,
                app.task_store,
                handle_timer,
                worker_id=f"timers-{socket.gethostname()}-{os.getpid()}",
                query=TaskQuery(type=TIMER_TASK_TYPE),
                recover_interrupted_handoffs=False,
                # Continuations follow the session deadline, without a timer-task cap.
                execution_timeout_s=None,
                stop=stop,
            )
        )
'''

_TEST_RESTART_PY = '''"""Each pause point survives a process restart.

RestartHarness builds the app from build_app on durable SQLite stores, stops
it, and builds a new app on the same stores, as a server restart or redeploy
would. See `uv run --no-sync cayu guide restart-tests`.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path

import operations.timers
from cayu import CayuApp
from cayu.providers import ModelStreamEvent
from cayu.sessions import SessionStatus
from cayu.tasks import (
    Task,
    TaskQuery,
    TaskRescheduleRequest,
    TaskSchedulePolicy,
    run_task_worker,
)
from cayu.testing import RestartHarness

from app import build_app
from integrations.payments import LocalPaymentService
from operations.timers import (
    TIMER_TASK_TYPE,
    continue_abandoned_sessions,
    continue_if_abandoned,
    handle_timer,
    schedule_timers,
)
from workflows.invoices import invoice_request

ROOT = Path(__file__).resolve().parents[1]
USAGE = {"usage": {"input_tokens": 100, "output_tokens": 20}}
PAY = [
    ModelStreamEvent.tool_call(
        name="pay_invoice", arguments={"invoice_id": "INV-7", "amount_cents": 12500}
    ),
    ModelStreamEvent.completed({"finish_reason": "tool_calls", **USAGE}),
]
ASK = [
    ModelStreamEvent.tool_call(
        name="ask_user", arguments={"question": "What is the amount?"}
    ),
    ModelStreamEvent.completed({"finish_reason": "tool_calls", **USAGE}),
]


def final(text: str) -> list[ModelStreamEvent]:
    return [ModelStreamEvent.text_delta(text), ModelStreamEvent.completed(USAGE)]


def request(text: str):
    return invoice_request(text, session_id="invoice-7")


async def run_deadline_now(app: CayuApp, timers: list[Task]) -> Task:
    """Move the deadline to now and run it once.

    The harness has no clock control, so this uses the same public call an
    operator uses to expire a request early.
    """

    assert app.task_store is not None
    deadline = next(task for task in timers if task.input["timer"] == "deadline")
    assert deadline.schedule is not None
    await app.reschedule_task(
        TaskRescheduleRequest(
            task_id=deadline.id,
            operation_id="test-expire-now",
            expected_revision=deadline.schedule.revision,
            available_at=datetime.now(UTC),
            policy=TaskSchedulePolicy(),
        )
    )
    handled = await run_task_worker(
        app,
        app.task_store,
        handle_timer,
        worker_id="test-timers",
        query=TaskQuery(type=TIMER_TASK_TYPE),
        max_tasks=1,
        poll_interval_s=0.05,
        reclaim=False,
        recover_interrupted_handoffs=False,
    )
    assert handled == 1
    done = await app.task_store.load_task(deadline.id)
    assert done is not None
    return done


def test_payment_approved_after_restart_runs_once(tmp_path) -> None:
    payments = LocalPaymentService()  # The payment provider outlives every process.

    async def scenario() -> None:
        async with RestartHarness(
            partial(build_app, payments=payments),
            store_dir=tmp_path,
            model_script=[PAY, final("Paid INV-7.")],
        ) as harness:
            paused = await harness.run(request("Pay INV-7 for $125."))
            [approval] = paused.pending_approvals
            assert payments.payments == {}

            await harness.restart()
            done = await harness.approve(approval, actor="finance@example.test")

            assert done.completed
            assert done.final_text == "Paid INV-7."
            await harness.assert_tool_ran_once("invoice-7", "pay_invoice")
            assert [receipt.invoice_id for receipt in payments.payments.values()] == [
                "INV-7"
            ]

    asyncio.run(scenario())


def test_reply_answered_after_restart_continues_the_job(tmp_path) -> None:
    payments = LocalPaymentService()

    async def scenario() -> None:
        async with RestartHarness(
            partial(build_app, payments=payments),
            store_dir=tmp_path,
            model_script=[ASK, PAY, final("Paid INV-7.")],
        ) as harness:
            paused = await harness.run(request("Pay invoice INV-7."))
            [question] = paused.pending_user_inputs

            await harness.restart()
            waiting = await harness.answer_user_input(question, "$125.00")
            [approval] = waiting.pending_approvals

            await harness.restart()
            done = await harness.approve(approval, actor="finance@example.test")

            assert done.completed
            await harness.assert_tool_ran_once("invoice-7", "pay_invoice")
            assert len(payments.payments) == 1

    asyncio.run(scenario())


def test_deadline_timer_closes_the_approval_after_restart(tmp_path) -> None:
    payments = LocalPaymentService()

    async def scenario() -> None:
        async with RestartHarness(
            partial(build_app, payments=payments),
            store_dir=tmp_path,
            model_script=[PAY, final("The payment was not approved in time.")],
        ) as harness:
            await harness.run(request("Pay INV-7 for $125."))
            timers = await schedule_timers(harness.app, "invoice-7")
            assert sorted(task.input["timer"] for task in timers) == [
                "deadline",
                "reminder",
            ]

            await harness.restart()
            app = harness.app
            assert app.task_store is not None
            # The task store kept both timers, and neither is due for hours.
            probe = await app.task_store.claim_task(
                "probe", TaskQuery(type=TIMER_TASK_TYPE)
            )
            assert probe is None
            done = await run_deadline_now(app, timers)

            assert done.result == {"outcome": "closed", "session_status": "completed"}
            assert await harness.pending_actions("invoice-7") == ()
            assert payments.payments == {}

    asyncio.run(scenario())


# Runs in a child process: save the answer, then die during the next model call.
ANSWER_THEN_DIE = """
import asyncio, os, sys

from cayu.approvals import UserInputResponse
from cayu.evals import ScriptedModelProvider
from cayu.storage import open_application_stores

from app import build_app


async def die(self, request):
    os._exit(17)
    yield


async def main(database, input_id):
    ScriptedModelProvider.stream = die
    stores = open_application_stores(None, sqlite_path=database)
    app = build_app(
        provider=ScriptedModelProvider([]),
        session_store=stores.session_store,
        task_store=stores.task_store,
    )
    answer = UserInputResponse(session_id="invoice-7", input_id=input_id, answer="$125.00")
    async for _ in app.resolve_user_input(answer):
        pass


asyncio.run(main(sys.argv[1], sys.argv[2]))
"""

# Runs in a child process: recover the abandoned session, then die before the
# continuation starts.
RECOVER_THEN_DIE = """
import asyncio, os, sys
from datetime import timedelta

from cayu import CayuApp
from cayu.evals import ScriptedModelProvider
from cayu.storage import open_application_stores

from app import build_app
from operations.timers import continue_abandoned_sessions


async def die(self, request, **kwargs):
    os._exit(18)
    yield


async def main(database):
    CayuApp.resume = die
    stores = open_application_stores(None, sqlite_path=database)
    app = build_app(
        provider=ScriptedModelProvider([]),
        session_store=stores.session_store,
        task_store=stores.task_store,
    )
    await continue_abandoned_sessions(app, inactive_for=timedelta(0))


asyncio.run(main(sys.argv[1]))
"""


def run_and_die(script: str, *args: str, exit_code: int) -> None:
    """Run a script in another process on the same database; it must die."""

    path = os.pathsep.join(filter(None, (str(ROOT), os.environ.get("PYTHONPATH"))))
    died = subprocess.run(
        [sys.executable, "-c", script, *args],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": path},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert died.returncode == exit_code, died.stdout + died.stderr


async def answer_in_a_process_that_dies(
    harness: RestartHarness, database: Path
) -> None:
    """Pause on the question, then answer it in a process that dies mid-turn."""

    paused = await harness.run(request("Pay invoice INV-7."))
    [question] = paused.pending_user_inputs
    assert question.input_id is not None
    run_and_die(ANSWER_THEN_DIE, str(database), question.input_id, exit_code=17)


def test_reply_saved_before_a_crash_is_continued_after_restart(
    tmp_path, monkeypatch
) -> None:
    payments = LocalPaymentService()
    database = tmp_path / "cayu.db"

    async def scenario() -> None:
        async with RestartHarness(
            partial(build_app, payments=payments),
            database_url=f"sqlite:///{database}",
            model_script=[ASK, PAY, final("Paid INV-7.")],
        ) as harness:
            paused = await harness.run(request("Pay invoice INV-7."))
            [question] = paused.pending_user_inputs
            assert question.input_id is not None
            timers = await schedule_timers(harness.app, "invoice-7")

            # Another process saves the answer, then dies mid-turn.
            run_and_die(ANSWER_THEN_DIE, str(database), question.input_id, exit_code=17)

            await harness.restart()
            app = harness.app
            stuck = await app.session_store.load("invoice-7")
            assert stuck is not None and stuck.status is SessionStatus.RUNNING
            assert await harness.pending_actions("invoice-7") == ()

            # The question's deadline fires. Its process is gone, so treat the
            # session as abandoned now rather than after ABANDONED_AFTER.
            monkeypatch.setattr(operations.timers, "ABANDONED_AFTER", timedelta(0))
            done = await run_deadline_now(app, timers)
            assert done.result == {
                "outcome": "continued",
                "session_status": "interrupted",
            }

            [approval] = await harness.pending_actions("invoice-7")
            finished = await harness.approve(approval, actor="finance@example.test")
            assert finished.completed
            await harness.assert_tool_ran_once("invoice-7", "pay_invoice")
            assert len(payments.payments) == 1

    asyncio.run(scenario())


def test_recovery_that_dies_before_continuing_is_picked_up_later(tmp_path) -> None:
    payments = LocalPaymentService()
    database = tmp_path / "cayu.db"

    async def scenario() -> None:
        async with RestartHarness(
            partial(build_app, payments=payments),
            database_url=f"sqlite:///{database}",
            model_script=[ASK, PAY, final("Paid INV-7.")],
        ) as harness:
            await answer_in_a_process_that_dies(harness, database)
            # A timer worker recovers the session, then dies before continuing it.
            run_and_die(RECOVER_THEN_DIE, str(database), exit_code=18)

            await harness.restart()
            app = harness.app
            stranded = await app.session_store.load("invoice-7")
            assert stranded is not None
            assert stranded.status is SessionStatus.INTERRUPTED

            # Nothing is running anymore, so a later sweep has nothing to
            # recover. It still finds the interrupted session and continues it.
            assert await continue_abandoned_sessions(
                app, inactive_for=timedelta(0)
            ) == ["invoice-7"]

            [approval] = await harness.pending_actions("invoice-7")
            finished = await harness.approve(approval, actor="finance@example.test")
            assert finished.completed
            await harness.assert_tool_ran_once("invoice-7", "pay_invoice")
            assert len(payments.payments) == 1

    asyncio.run(scenario())


def test_a_stale_continuation_does_not_start_another_turn(
    tmp_path, monkeypatch
) -> None:
    database = tmp_path / "cayu.db"

    async def scenario() -> None:
        async with RestartHarness(
            partial(build_app, payments=LocalPaymentService()),
            database_url=f"sqlite:///{database}",
            # The last turn is there only so a duplicate turn would be counted.
            model_script=[ASK, final("INV-7 is already paid."), final("Again.")],
        ) as harness:
            await answer_in_a_process_that_dies(harness, database)
            await harness.restart()
            app = harness.app
            await app.recover_abandoned_sessions(inactive_for_seconds=0)

            # Two workers decide to continue the session. The second finishes
            # the continuation after the first checked the session but before
            # the first one's resume starts.
            resume = app.resume
            raced = False

            async def resume_after_another_worker(req, **kwargs):
                nonlocal raced
                if not raced:
                    raced = True
                    assert await continue_if_abandoned(app, "invoice-7")
                async for event in resume(req, **kwargs):
                    yield event

            monkeypatch.setattr(app, "resume", resume_after_another_worker)
            assert await continue_if_abandoned(app, "invoice-7") is False

            session = await app.session_store.load("invoice-7")
            assert session is not None and session.status is SessionStatus.COMPLETED
            # The question, then one continuation. The stale worker added none.
            assert len(harness.model_requests) == 2

    asyncio.run(scenario())
'''

_TEST_SPEND_CAP_PY = '''"""The per-session spend cap stops a session before it spends more."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from functools import partial

from cayu import EventType, run_to_completion
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.sessions import InMemorySessionStore, SessionStatus
from cayu.tasks import InMemoryTaskStore
from cayu.testing import RestartHarness

from app import build_app
from integrations.payments import LocalPaymentService
from policies.budgets import (
    SESSION_SPEND_CAP_USD,
    scripted_price_book,
    session_spend_cap,
)
from workflows.invoices import invoice_request

PAY = [
    ModelStreamEvent.tool_call(
        name="pay_invoice", arguments={"invoice_id": "INV-7", "amount_cents": 12500}
    ),
    ModelStreamEvent.completed(
        {
            "finish_reason": "tool_calls",
            "usage": {"input_tokens": 100, "output_tokens": 20},
        }
    ),
]


def test_session_stops_at_the_spend_cap_before_paying() -> None:
    payments = LocalPaymentService()
    app = build_app(
        provider=ScriptedModelProvider([PAY]),
        session_store=InMemorySessionStore(),
        task_store=InMemoryTaskStore(),
        payments=payments,
    )
    # At $100,000 per million tokens this one turn costs $12, over the cap.
    cap = session_spend_cap(pricing=scripted_price_book("100000"))

    outcome = asyncio.run(
        run_to_completion(
            app, invoice_request("Pay INV-7.", session_id="invoice-7", spend_cap=cap)
        )
    )

    assert outcome.status is SessionStatus.INTERRUPTED
    [reached] = [e for e in outcome.events if e.type == EventType.SESSION_LIMIT_REACHED]
    assert Decimal(str(reached.payload["maximum"])) == Decimal(SESSION_SPEND_CAP_USD)
    # The model asked to pay, but the cap stopped the step before the tool ran.
    assert payments.payments == {}


def test_spend_cap_counts_spend_from_every_process(tmp_path) -> None:
    payments = LocalPaymentService()
    three_dollars = {"usage": {"input_tokens": 100_000, "output_tokens": 50_000}}
    ask = [
        ModelStreamEvent.tool_call(
            name="ask_user", arguments={"question": "What is the amount?"}
        ),
        ModelStreamEvent.completed({"finish_reason": "tool_calls", **three_dollars}),
    ]
    pay = [
        PAY[0],
        ModelStreamEvent.completed({"finish_reason": "tool_calls", **three_dollars}),
    ]

    async def scenario() -> None:
        async with RestartHarness(
            partial(build_app, payments=payments),
            store_dir=tmp_path,
            model_script=[ask, pay],
        ) as harness:
            cap = session_spend_cap(pricing=scripted_price_book("20"))
            paused = await harness.run(
                invoice_request(
                    "Pay invoice INV-7.", session_id="invoice-7", spend_cap=cap
                )
            )
            await harness.restart()
            # $3 before the restart and $3 after it passes the $5 cap.
            stopped = await harness.answer_user_input(
                paused.pending_user_inputs[0], "$125.00"
            )

            assert stopped.status is SessionStatus.INTERRUPTED
            assert any(
                e.type == EventType.SESSION_LIMIT_REACHED for e in stopped.events
            )
            assert payments.payments == {}

    asyncio.run(scenario())
'''

_EVAL_PY = """from cayu import EventType
from cayu.evals import (
    EvalCase,
    EvalPlan,
    EvalSuite,
    EventOccurred,
    ScriptedModelProvider,
    SessionInterrupted,
)
from cayu.providers import ModelStreamEvent
from cayu.sessions import InMemorySessionStore
from cayu.tasks import InMemoryTaskStore

from app import build_app
from workflows.invoices import invoice_request


def build_eval() -> EvalPlan:
    provider = ScriptedModelProvider(
        [
            ModelStreamEvent.tool_call(
                name="pay_invoice",
                arguments={"invoice_id": "INV-7", "amount_cents": 12500},
            ),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "tool_calls",
                    "usage": {"input_tokens": 100, "output_tokens": 20},
                }
            ),
        ]
    )
    app = build_app(
        provider=provider,
        session_store=InMemorySessionStore(),
        task_store=InMemoryTaskStore(),
    )
    suite = EvalSuite(
        id="invoice-payment",
        cases=[
            EvalCase(
                id="payment-waits-for-approval",
                request=invoice_request(
                    "Pay INV-7 for $125.", session_id="eval-invoice-7"
                ),
                assertions=[
                    SessionInterrupted(),
                    EventOccurred(EventType.TOOL_CALL_APPROVAL_REQUESTED),
                ],
            )
        ],
    )
    return EvalPlan(app=app, suite=suite)
"""

_WORKERS_TOML = """
# `uv run --no-sync cayu worker timers` runs reminders and deadlines.
[tool.cayu.workers]
timers = "operations.timers:run_timer_worker"
"""

_GUIDANCE = """
## Long-running patterns

This project is a small invoice-payment agent that shows one working example of
each pattern a long-running job needs. Each part is an ordinary Cayu part in its
own module; keep, replace, or delete them separately. The one-page reference is
`uv run --no-sync cayu guide long-running`.

| Pattern | Where | Cayu part | Test |
| --- | --- | --- | --- |
| Approval before a side effect | `policies/tools.py`, `tools/payments.py` | `AlwaysRequireApprovalToolPolicy`, `ToolEffect.IDEMPOTENT`, `ToolContext.idempotency_key` | `tests/test_restart.py` |
| Waiting for an outside reply | `tools/registration.py` | `UserInputTool` (`ask_user`), answered with `resolve_user_input` | `tests/test_restart.py` |
| Reminders and deadlines | `operations/timers.py` | `TaskCreate` with `available_at` and `TaskSchedulePolicy`, `run_task_worker` | `tests/test_restart.py` |
| Per-session spend cap | `policies/budgets.py`, `workflows/invoices.py` | `BudgetLimit(scope="session")`, `default_price_book()` | `tests/test_spend_cap.py` |
| Restart proof | `tests/test_restart.py` | `cayu.testing.RestartHarness` | the tests themselves |

- Replace `LocalPaymentService` in `integrations/payments.py` with your payment
  provider's client. It is an in-memory stand-in, not durable.
- Start jobs with `workflows.invoices.invoice_request(...)` so every session
  carries the spend cap. Sessions started elsewhere (`run.py`, the control
  plane) need the same `budget_limits` on their request, and a resumed session
  must restate the cap it started with.
- Run reminders and deadlines with `uv run --no-sync cayu worker timers`. It
  finds open approvals and questions itself, whichever process paused them, and
  uses `recover_abandoned_sessions` to continue sessions whose process died
  mid-turn (`ABANDONED_AFTER` in `operations/timers.py`).
"""


def long_running_project_files(
    *,
    files: dict[str, str],
    render: Callable[[str], str],
    plan: ApplicationPlan,
) -> dict[str, str]:
    """Return the files that turn the agent layout into the long-running example."""

    if plan.preset != "long-running":
        raise ValueError("long_running_project_files requires the long-running preset")
    updated = {
        "integrations/payments.py": _PAYMENTS_INTEGRATION_PY,
        "tools/payments.py": render(_PAYMENTS_TOOL_PY),
        "tools/registration.py": _TOOLS_REGISTRATION_PY,
        "policies/tools.py": _TOOL_POLICY_PY,
        "policies/budgets.py": _BUDGETS_PY,
        "workflows/invoices.py": _WORKFLOW_PY,
        "prompts/agent.py": _PROMPT_PY,
        "operations/timers.py": _TIMERS_PY,
        "tests/test_restart.py": _TEST_RESTART_PY,
        "tests/test_spend_cap.py": _TEST_SPEND_CAP_PY,
        "app.py": _replace_each(
            files["app.py"],
            (
                (
                    "from environments.local import build_local_environment\n",
                    "from environments.local import build_local_environment\n"
                    "from integrations.payments import LocalPaymentService, PaymentService\n",
                ),
                (
                    "    artifact_store: ArtifactStore | None = None,\n) -> CayuApp:",
                    "    artifact_store: ArtifactStore | None = None,\n"
                    "    payments: PaymentService | None = None,\n) -> CayuApp:",
                ),
                (
                    "        context_policy=build_context_policy(),\n    )",
                    "        context_policy=build_context_policy(),\n"
                    "        payments=payments if payments is not None else LocalPaymentService(),\n"
                    "    )",
                ),
            ),
        ),
        "agents/registration.py": _replace_each(
            files["agents/registration.py"],
            (
                (
                    "from agents.agent import AGENT\n",
                    "from agents.agent import AGENT\nfrom integrations.payments import PaymentService\n",
                ),
                (
                    "    context_policy: ContextPolicy | None = None,\n) -> None:",
                    "    context_policy: ContextPolicy | None = None,\n"
                    "    payments: PaymentService,\n) -> None:",
                ),
                (
                    "starter_tools = list(build_agent_tools())",
                    "starter_tools = list(build_agent_tools(payments=payments))",
                ),
            ),
        ),
        "pyproject.toml": _replace_each(
            files["pyproject.toml"],
            (("\n[tool.cayu.session_store]\n", _WORKERS_TOML + "\n[tool.cayu.session_store]\n"),),
        ),
        "README.md": files["README.md"] + _GUIDANCE,
        "AGENTS.md": files["AGENTS.md"] + _GUIDANCE,
    }
    if "evals/agent.py" in files:
        updated["evals/agent.py"] = _EVAL_PY
    return updated


def _replace_each(source: str, replacements: tuple[tuple[str, str], ...]) -> str:
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError(f"long-running preset could not place {old.strip()!r}")
        source = source.replace(old, new)
    return source
