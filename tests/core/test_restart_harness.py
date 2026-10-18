"""RestartHarness drives one application across simulated process restarts."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    Message,
    RunRequest,
    SessionStatus,
    Tool,
    ToolContext,
    ToolEffect,
    ToolResult,
    ToolSpec,
)
from cayu.providers import ModelStreamEvent
from cayu.runtime import ExecutionProfileBehaviorIdentity, ExecutionProfileMismatchError
from cayu.sessions import InMemorySessionStore
from cayu.testing import RestartHarness, RestartStep
from cayu.tools import (
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
    UserInputTool,
)


class RefundLedger:
    """Stands in for the payment service: it outlives every app generation."""

    def __init__(self) -> None:
        self.refunds: list[dict[str, Any]] = []


class IssueRefund(Tool):
    def __init__(self, ledger: RefundLedger, version: str = "1") -> None:
        self.ledger = ledger
        self.spec = ToolSpec(
            name="issue_refund",
            description="Refund an order.",
            input_schema={
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
                "additionalProperties": False,
            },
            effect=ToolEffect.IDEMPOTENT,
            parallel_safe=False,
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name="issue_refund", behavior_version=version, implementation_version="1"
            ),
        )
        super().__init__()

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        self.ledger.refunds.append({"session_id": ctx.session_id, **args})
        return ToolResult(content=f"Refunded {args['order_id']}.")


class RefundsNeedApproval(ToolPolicy):
    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="refunds-need-approval", behavior_version="1", implementation_version="1"
        )

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        if request.tool_name == "issue_refund":
            return ToolPolicyResult(decision=ToolPolicyDecision.REQUIRE_APPROVAL)
        return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)


def build_app(
    *,
    provider,
    session_store,
    task_store,
    ledger: RefundLedger,
    refund_version: str = "1",
) -> CayuApp:
    app = CayuApp(session_store=session_store, task_store=task_store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="refunds", model="scripted-model", system_prompt="Handle refunds."),
        tools=[IssueRefund(ledger, refund_version), UserInputTool()],
        tool_policy=RefundsNeedApproval(),
    )
    return app


def call(name: str, arguments: dict[str, Any], call_id: str) -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.tool_call(id=call_id, name=name, arguments=arguments),
        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
    ]


def final(text: str) -> list[ModelStreamEvent]:
    return [ModelStreamEvent.text_delta(text), ModelStreamEvent.completed()]


def refund_request(session_id: str = "refund-1") -> RunRequest:
    return RunRequest(
        agent_name="refunds",
        session_id=session_id,
        messages=[Message.text("user", "Order A-1 arrived broken. Please refund it.")],
    )


REFUND_SCRIPT = [
    call("issue_refund", {"order_id": "A-1"}, "refund-call"),
    final("Refunded order A-1."),
]


@pytest.fixture(params=["sqlite", "postgres"])
def store_location(request, tmp_path: Path) -> dict[str, Any]:
    if request.param == "sqlite":
        return {"store_dir": tmp_path}
    return {"database_url": request.getfixturevalue("migrated_postgres_url")}


def _harness(ledger: RefundLedger, script, **location: Any) -> RestartHarness:
    return RestartHarness(
        lambda **parts: build_app(ledger=ledger, **parts),
        model_script=script,
        **location,
    )


def test_approval_after_a_clean_restart_runs_the_tool_once(
    store_location: dict[str, Any],
) -> None:
    ledger = RefundLedger()
    session_id = f"refund-{uuid4().hex}"

    async def scenario() -> None:
        async with _harness(ledger, REFUND_SCRIPT, **store_location) as harness:
            paused = await harness.run(refund_request(session_id))
            assert paused.status is SessionStatus.INTERRUPTED
            [approval] = paused.pending_approvals
            assert approval.tool_name == "issue_refund"
            assert ledger.refunds == []
            first_app = harness.app

            await harness.restart()
            assert harness.app is not first_app
            assert harness.generation == 2
            assert [action.id for action in await harness.pending_actions(session_id)] == [
                approval.id
            ]

            done = await harness.approve(approval, actor="ops@example.test")
            assert done.completed
            assert done.final_text == "Refunded order A-1."
            assert done.pending_actions == ()
            assert harness.remaining_model_turns == 0
            assert len(harness.model_requests) == 2

            completed = await harness.assert_tool_ran_once(session_id, "issue_refund")
            assert completed.payload["tool_call_id"] == "refund-call"
            assert ledger.refunds == [{"session_id": session_id, "order_id": "A-1"}]

            # A further restart neither repeats the call nor loses the history.
            await harness.restart()
            await harness.assert_tool_ran_once(session_id, "issue_refund")
            assert len(ledger.refunds) == 1

    asyncio.run(scenario())


def test_denial_after_restart_never_runs_the_tool(tmp_path: Path) -> None:
    ledger = RefundLedger()
    script = [
        call("issue_refund", {"order_id": "A-1"}, "refund-call"),
        final("I could not refund the order."),
    ]

    async def scenario() -> None:
        async with _harness(ledger, script, store_dir=tmp_path) as harness:
            await harness.run(refund_request())
            await harness.restart()
            done = await harness.deny("refund-1", actor="ops@example.test", reason="Too late.")
            assert done.completed
            assert done.final_text == "I could not refund the order."
            with pytest.raises(AssertionError, match="started 0 times"):
                await harness.assert_tool_ran_once("refund-1", "issue_refund")
        assert ledger.refunds == []

    asyncio.run(scenario())


def test_user_input_answered_by_a_later_process(tmp_path: Path) -> None:
    ledger = RefundLedger()
    script = [
        call("ask_user", {"question": "Which order?"}, "ask-call"),
        call("issue_refund", {"order_id": "A-7"}, "refund-call"),
        final("Refunded order A-7."),
    ]

    async def scenario() -> None:
        async with _harness(ledger, script, store_dir=tmp_path) as harness:
            asked = await harness.run(refund_request())
            [question] = asked.pending_user_inputs
            assert question.question == "Which order?"
            assert asked.pending_approvals == ()

            await harness.restart()
            waiting = await harness.answer_user_input(
                "refund-1", "Order A-7.", actor="customer@example.test"
            )
            assert [action.tool_name for action in waiting.pending_approvals] == ["issue_refund"]

            await harness.restart()
            done = await harness.approve("refund-1", actor="ops@example.test")
            assert done.completed
            await harness.assert_tool_ran_once("refund-1", "issue_refund")
            assert harness.generation == 3
        assert [refund["order_id"] for refund in ledger.refunds] == ["A-7"]

    asyncio.run(scenario())


def test_resume_continues_a_completed_conversation_after_restart(tmp_path: Path) -> None:
    ledger = RefundLedger()
    script = [final("Which order is it?"), final("Thanks, noted.")]

    async def scenario() -> None:
        async with _harness(ledger, script, store_dir=tmp_path) as harness:
            first = await harness.run(refund_request())
            assert first.completed
            await harness.restart()
            second = await harness.resume("refund-1", messages=[Message.text("user", "A-1.")])
            assert second.completed
            assert second.final_text == "Thanks, noted."
            # The second process saw the first process's transcript.
            texts = [str(message.content) for message in harness.model_requests[1].messages]
            assert any("Which order is it?" in text for text in texts)

    asyncio.run(scenario())


def test_changing_a_tool_identity_between_restarts_fails_like_production(
    tmp_path: Path,
) -> None:
    ledger = RefundLedger()
    versions = iter(["1", "2"])

    def factory(**parts: Any) -> CayuApp:
        return build_app(ledger=ledger, refund_version=next(versions), **parts)

    async def scenario() -> None:
        async with RestartHarness(factory, model_script=REFUND_SCRIPT, store_dir=tmp_path) as h:
            await h.run(refund_request())
            await h.restart()
            with pytest.raises(ExecutionProfileMismatchError):
                await h.approve("refund-1", actor="ops@example.test")
        assert ledger.refunds == []

    asyncio.run(scenario())


def test_in_memory_stores_are_refused(tmp_path: Path) -> None:
    ledger = RefundLedger()

    def ignores_stores(*, provider, session_store, task_store) -> CayuApp:
        del session_store, task_store
        return build_app(
            ledger=ledger,
            provider=provider,
            session_store=InMemorySessionStore(),
            task_store=None,
        )

    async def scenario() -> None:
        harness = RestartHarness(ignores_stores, store_dir=tmp_path)
        with pytest.raises(ValueError, match="in-memory stores lose every session"):
            await harness.start()
        await harness.aclose()

    asyncio.run(scenario())
    with pytest.raises(ValueError, match="in-memory stores"):
        RestartHarness(ignores_stores, database_url="sqlite:///:memory:")
    with pytest.raises(ValueError, match="exactly one of store_dir or database_url"):
        RestartHarness(ignores_stores)
    with pytest.raises(ValueError, match="exactly one of store_dir or database_url"):
        RestartHarness(ignores_stores, store_dir=tmp_path, database_url="sqlite:////x.db")


def test_an_exhausted_model_script_fails_the_step_with_its_error(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with _harness(RefundLedger(), [], store_dir=tmp_path) as harness:
            step = await harness.run(refund_request())
            assert step.status is SessionStatus.FAILED
            assert step.error is not None
            assert harness.remaining_model_turns == 0

    asyncio.run(scenario())


class _StoreCloseProbe:
    """Wraps the session store's close to fail or wait once, then close for real."""

    def __init__(self, *, fail_first: bool = False) -> None:
        self.fail_first = fail_first
        self.release = asyncio.Event()
        self.started = asyncio.Event()
        self.attempts = 0
        self.closed = False
        self.wait_first = not fail_first

    def factory(self, ledger: RefundLedger):
        def build(*, provider, session_store, task_store) -> CayuApp:
            real_close = session_store.close

            async def close() -> None:
                self.attempts += 1
                self.started.set()
                if self.attempts == 1 and self.fail_first:
                    raise OSError("transient close failure")
                if self.attempts == 1 and self.wait_first:
                    await self.release.wait()
                await real_close()
                self.closed = True

            session_store.close = close
            return build_app(
                ledger=ledger,
                provider=provider,
                session_store=session_store,
                task_store=task_store,
            )

        return build


def test_a_failed_store_close_is_retried_by_the_next_aclose(tmp_path: Path) -> None:
    probe = _StoreCloseProbe(fail_first=True)

    async def scenario() -> None:
        harness = RestartHarness(probe.factory(RefundLedger()), store_dir=tmp_path)
        await harness.start()
        with pytest.raises(OSError, match="transient close failure"):
            await harness.aclose()
        assert not probe.closed

        await harness.aclose()
        assert probe.attempts == 2
        assert probe.closed
        with pytest.raises(RuntimeError, match="closed"):
            await harness.start()

    asyncio.run(scenario())


def test_start_after_a_failed_restart_closes_the_kept_stores_first(tmp_path: Path) -> None:
    probe = _StoreCloseProbe(fail_first=True)

    async def scenario() -> None:
        harness = RestartHarness(probe.factory(RefundLedger()), store_dir=tmp_path)
        await harness.start()
        with pytest.raises(OSError, match="transient close failure"):
            await harness.restart()
        assert not probe.closed

        await harness.start()
        assert probe.attempts == 2
        assert probe.closed
        assert harness.generation == 2
        await harness.aclose()

    asyncio.run(scenario())


def _failing_factory(close):
    def build(*, provider, session_store, task_store) -> CayuApp:
        del provider, task_store
        real_close = session_store.close

        async def wrapped() -> None:
            await close(real_close)

        session_store.close = wrapped
        raise ValueError("factory construction failed")

    return build


def test_aclose_retries_a_failed_close_after_a_factory_error(tmp_path: Path) -> None:
    attempts = 0
    closed = False

    async def close(real_close) -> None:
        nonlocal attempts, closed
        attempts += 1
        if attempts == 1:
            raise OSError("transient close failure")
        await real_close()
        closed = True

    async def scenario() -> None:
        harness = RestartHarness(_failing_factory(close), store_dir=tmp_path)
        with pytest.raises(OSError, match="transient close failure"):
            await harness.start()
        await harness.aclose()
        assert attempts == 2
        assert closed

    asyncio.run(scenario())


def test_aclose_waits_for_a_close_cancelled_after_a_factory_error(tmp_path: Path) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    closed = False

    async def close(real_close) -> None:
        nonlocal closed
        started.set()
        await release.wait()
        await real_close()
        closed = True

    async def scenario() -> None:
        harness = RestartHarness(_failing_factory(close), store_dir=tmp_path)
        starter = asyncio.create_task(harness.start())
        await started.wait()
        starter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await starter

        closing = asyncio.create_task(harness.aclose())
        await asyncio.sleep(0.05)
        assert not closing.done(), "aclose returned before the stores closed"
        release.set()
        await closing
        assert closed

    asyncio.run(scenario())


def test_a_cancelled_aclose_is_finished_by_the_next_one(tmp_path: Path) -> None:
    probe = _StoreCloseProbe()

    async def scenario() -> None:
        harness = RestartHarness(probe.factory(RefundLedger()), store_dir=tmp_path)
        await harness.start()
        first = asyncio.create_task(harness.aclose())
        await probe.started.wait()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

        second = asyncio.create_task(harness.aclose())
        await asyncio.sleep(0.05)
        assert not second.done(), "aclose returned before the stores closed"
        probe.release.set()
        await second
        assert probe.closed
        assert probe.attempts == 1

    asyncio.run(scenario())


def test_restart_step_is_exported_with_the_harness() -> None:
    import cayu.testing as testing
    from cayu.testing import restart

    assert testing.RestartHarness is restart.RestartHarness
    assert testing.RestartStep is restart.RestartStep is RestartStep
