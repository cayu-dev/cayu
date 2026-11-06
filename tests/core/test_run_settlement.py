"""Every exit of an admitted invocation settles exactly once, in a fixed order."""

from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace

import pytest
from tests.core._run_exit_matrix import BodyFailure, exit_cases, run_exit_case

from cayu.runtime import _run_settlement
from cayu.runtime._recovery_ownership import _run_recovery_cleanup_steps
from cayu.runtime._run_exit import RunExit, RunExitKind
from cayu.runtime._run_settlement import (
    OutcomeStage,
    RunLedger,
    RunSettlement,
    drain_event_stream,
)
from cayu.runtime._session_finalization import _mark_interaction_transition_run_fence
from cayu.sessions.authority import SessionRunFenced
from cayu.sessions.records import SessionStatus

_POINTS = 3
_PROFILE = object()


class _World:
    """Durable authority of one admitted invocation, and every write against it."""

    def __init__(self, *, status: SessionStatus = SessionStatus.RUNNING) -> None:
        self.session = SimpleNamespace(instance_id="instance", run_epoch=4, status=status)
        self.checkpoint: dict[str, object] = {"active": _PROFILE}
        self.calls: list[str] = []
        self.reads = 0
        self.decide_gate: asyncio.Event | None = None
        self.decide_failure: BaseException | None = None

    async def load(self, session_id):
        self.reads += 1
        return self.session

    async def load_checkpoint(self, session_id):
        return self.checkpoint

    async def finalize_abandoned_session_by_id(self, session_id, **kwargs):
        self.calls.append("decide")
        if self.decide_gate is not None:
            await self.decide_gate.wait()
        if self.decide_failure is not None:
            raise self.decide_failure
        self.session.status = SessionStatus.INTERRUPTED

    def forget_refused_interaction_publications(self, session_id):
        return None

    async def abort_environment_setup(self, **kwargs):
        self.calls.append("environment")

    async def release_run_fence_after_environment_cleanup(self, **kwargs):
        self.calls.append("fence")
        self.session.run_epoch += 1

    async def run_cleanup_steps(self, *, authoritative_failure, steps, cancellation_baseline=0):
        return await _run_recovery_cleanup_steps(
            authoritative_failure=authoritative_failure,
            steps=steps,
            cancellation_baseline=cancellation_baseline,
        )


@pytest.fixture(autouse=True)
def _checkpoint_readers(monkeypatch):
    monkeypatch.setattr(
        _run_settlement,
        "active_invocation_execution_profile_from_checkpoint",
        lambda checkpoint: checkpoint.get("active"),
    )
    monkeypatch.setattr(
        _run_settlement,
        "invocation_terminal_decision_from_checkpoint",
        lambda checkpoint: checkpoint.get("decision"),
    )
    monkeypatch.setattr(
        _run_settlement,
        "settled_invocation_terminal_decision_from_checkpoint",
        lambda checkpoint: checkpoint.get("settled"),
    )
    monkeypatch.setattr(
        _run_settlement,
        "pending_completion_finalization_from_checkpoint",
        lambda checkpoint: checkpoint.get("pending"),
    )


def _settlement(world: _World) -> RunSettlement:
    return RunSettlement(
        session_store=world,
        session_finalization=world,
        environment_lifecycle=world,
        recovery_ownership=world,
    )


async def _settled(settlement: RunSettlement, ledger: RunLedger, body):
    """Forward ``body`` as the only stream the consumer sees."""

    async with settlement.settling(ledger), contextlib.aclosing(body) as owned:
        async for item in owned:
            yield item


def _ledger(world: _World) -> RunLedger:
    context = SimpleNamespace(
        binding=SimpleNamespace(
            session_id="session",
            session_instance_id="instance",
            run_epoch=4,
            interaction_id="interaction",
        ),
        active_profile=_PROFILE,
        profile=None,
        registered_agent=None,
        registered_environment=None,
    )
    ledger = RunLedger(
        session_id="session",
        run_epoch=4,
        invocation_context=context,
    )
    ledger.local_release.append(lambda: world.calls.append("local"))
    return ledger


@pytest.mark.parametrize("case", exit_cases(_POINTS), ids=lambda case: case.id)
def test_every_exit_settles_once_in_order(case):
    async def scenario():
        world = _World()
        settlement = _settlement(world)
        ledger = _ledger(world)
        outcome = await run_exit_case(
            case,
            points=_POINTS,
            stream_for=lambda body: _settled(settlement, ledger, body),
        )
        return world, ledger, outcome

    world, ledger, outcome = asyncio.run(scenario())
    if case.kind == "return":
        # The body owns its own happy path; nothing durable is re-read.
        assert world.calls == ["local"]
        assert world.reads == 0
        assert outcome.escaped is None
        assert outcome.received == list(range(_POINTS))
        return
    assert world.calls == ["decide", "environment", "fence", "local"]
    assert ledger.stage is OutcomeStage.FENCE_RELEASED
    assert world.session.status is SessionStatus.INTERRUPTED
    if case.kind == "close":
        assert outcome.escaped is None
        assert outcome.received == list(range(case.point + 1))
    elif case.kind == "cancel":
        assert type(outcome.escaped) is asyncio.CancelledError
    elif case.kind == "grouped-cancel":
        assert type(outcome.escaped) is BaseExceptionGroup
        assert [type(leaf) for leaf in outcome.escaped.exceptions] == [
            asyncio.CancelledError,
            OSError,
        ]
    else:
        assert type(outcome.escaped) is BodyFailure


def test_repeated_cancellation_does_not_interrupt_settlement():
    async def scenario():
        world = _World()
        world.decide_gate = asyncio.Event()
        settlement = _settlement(world)
        ledger = _ledger(world)
        started = asyncio.Event()

        async def body():
            started.set()
            await asyncio.Event().wait()
            yield 0

        async def consume():
            async for _ in _settled(settlement, ledger, body()):
                pass

        task = asyncio.create_task(consume())
        await started.wait()
        task.cancel()
        for _ in range(5):
            await asyncio.sleep(0)
        assert world.calls == ["decide"]
        task.cancel()
        await asyncio.sleep(0)
        world.decide_gate.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        return world

    world = asyncio.run(scenario())
    assert world.calls == ["decide", "environment", "fence", "local"]


def test_decided_remainder_runs_before_environment_and_fence():
    async def scenario():
        world = _World(status=SessionStatus.COMPLETED)
        settlement = _settlement(world)
        ledger = _ledger(world)

        async def remainder():
            world.calls.append("remainder")

        async def body():
            ledger.decided(remainder)
            yield "interaction.completed"
            yield "session.completed"

        stream = _settled(settlement, ledger, body())
        assert await anext(stream) == "interaction.completed"
        await stream.aclose()
        return world, ledger

    world, ledger = asyncio.run(scenario())
    assert world.calls == ["remainder", "environment", "fence", "local"]
    assert ledger.stage is OutcomeStage.FENCE_RELEASED
    assert ledger.remainder is None


def test_completed_body_clears_its_remainder():
    async def scenario():
        world = _World()
        settlement = _settlement(world)
        ledger = _ledger(world)

        async def body():
            ledger.decided(lambda: pytest.fail("remainder already finished"))
            yield 0
            ledger.advance(OutcomeStage.TERMINAL_PUBLISHED)

        return world, [item async for item in _settled(settlement, ledger, body())]

    world, received = asyncio.run(scenario())
    assert received == [0]
    assert world.calls == ["local"]


def test_fence_loss_performs_no_durable_work():
    fenced = SessionRunFenced("stale")
    _mark_interaction_transition_run_fence(fenced)

    async def scenario():
        world = _World()
        settlement = _settlement(world)
        ledger = _ledger(world)

        async def body():
            yield 0
            raise fenced

        with pytest.raises(SessionRunFenced):
            async for _ in _settled(settlement, ledger, body()):
                pass
        return world

    world = asyncio.run(scenario())
    assert world.calls == ["local"]
    assert world.reads == 0


@pytest.mark.parametrize("marker", ["decision", "pending"])
def test_committed_decision_is_left_to_its_recovery(marker):
    async def scenario():
        world = _World()
        world.checkpoint[marker] = {"outcome": "failed"}
        settlement = _settlement(world)
        ledger = _ledger(world)
        await settlement.settle(ledger, RunExit(RunExitKind.ABANDONED, GeneratorExit()))
        return world, ledger

    world, ledger = asyncio.run(scenario())
    assert world.calls == []
    assert ledger.stage is OutcomeStage.ADMITTED


def test_an_outcome_the_body_committed_is_left_to_the_body():
    async def scenario():
        world = _World(status=SessionStatus.FAILED)
        settlement = _settlement(world)
        await settlement.settle(
            _ledger(world), RunExit(RunExitKind.CANCELLED, asyncio.CancelledError())
        )
        return world

    world = asyncio.run(scenario())
    # The body decided FAILED and owns its own release; a failed release is
    # not retried here (it stays visible exactly as the body reported it).
    assert world.calls == []


def test_settled_decision_of_this_epoch_is_left_to_its_replay():
    async def scenario():
        world = _World(status=SessionStatus.INTERRUPTING)
        world.checkpoint["settled"] = SimpleNamespace(run_epoch=4)
        settlement = _settlement(world)
        await settlement.settle(_ledger(world), RunExit(RunExitKind.FAILED, RuntimeError()))
        return world

    world = asyncio.run(scenario())
    assert world.calls == []


def test_decided_outcome_releases_once():
    async def scenario():
        world = _World(status=SessionStatus.COMPLETED)
        settlement = _settlement(world)
        ledger = _ledger(world)
        ledger.decided()
        exit = RunExit(RunExitKind.CANCELLED, asyncio.CancelledError())
        await settlement.settle(ledger, exit)
        await settlement.settle(ledger, exit)
        return world, ledger

    world, ledger = asyncio.run(scenario())
    assert world.calls == ["environment", "fence"]
    assert ledger.stage is OutcomeStage.FENCE_RELEASED


def test_a_released_or_refenced_invocation_owes_nothing():
    async def scenario():
        world = _World()
        world.session.run_epoch = 5
        settlement = _settlement(world)
        ledger = _ledger(world)
        await settlement.settle(ledger, RunExit(RunExitKind.ABANDONED, GeneratorExit()))
        return world, ledger

    world, ledger = asyncio.run(scenario())
    assert world.calls == []
    assert ledger.stage is OutcomeStage.FENCE_RELEASED


def test_rebind_moves_settlement_to_the_successor():
    async def scenario():
        world = _World()
        settlement = _settlement(world)
        ledger = _ledger(world)
        world.session.run_epoch = 5
        successor = SimpleNamespace(
            binding=SimpleNamespace(
                session_id="session",
                session_instance_id="instance",
                run_epoch=5,
                interaction_id="interaction",
            ),
            active_profile=_PROFILE,
            profile=None,
            registered_agent=None,
            registered_environment=None,
        )
        ledger.rebind(successor)
        await settlement.settle(ledger, RunExit(RunExitKind.CANCELLED, asyncio.CancelledError()))
        return world

    world = asyncio.run(scenario())
    assert world.calls == ["decide", "environment", "fence"]


def test_process_level_stop_leaves_the_invocation_to_recovery():
    async def scenario():
        world = _World()
        settlement = _settlement(world)
        ledger = _ledger(world)

        async def body():
            yield 0
            raise SystemExit(3)

        with pytest.raises(SystemExit):
            async for _ in _settled(settlement, ledger, body()):
                pass
        return world

    world = asyncio.run(scenario())
    assert world.calls == ["local"]
    assert world.reads == 0


def test_failing_remainder_still_releases_environment_and_fence():
    async def scenario():
        world = _World(status=SessionStatus.COMPLETED)
        settlement = _settlement(world)
        ledger = _ledger(world)

        async def remainder():
            world.calls.append("remainder")
            raise OSError("task completion store unavailable")

        ledger.decided(remainder)
        with pytest.raises(OSError):
            await settlement.settle(ledger, RunExit(RunExitKind.ABANDONED, GeneratorExit()))
        return world, ledger

    world, ledger = asyncio.run(scenario())
    assert world.calls == ["remainder", "environment", "fence"]
    assert ledger.stage is OutcomeStage.DECIDED


def test_a_spent_publication_is_not_recorded_as_published():
    async def scenario():
        async def publication():
            yield "task.completed"
            raise asyncio.CancelledError()

        stream = publication()
        assert await anext(stream) == "task.completed"
        with pytest.raises(asyncio.CancelledError):
            await anext(stream)
        return await drain_event_stream(stream)

    assert asyncio.run(scenario()) is False


@pytest.mark.parametrize("failing", ["decision", "deferred-input"])
def test_failing_decision_still_releases_environment_and_fence(failing):
    async def scenario():
        world = _World()
        settlement = _settlement(world)
        ledger = _ledger(world)
        failure = OSError("session store unavailable")
        if failing == "decision":
            world.decide_failure = failure
        else:

            async def publish_deferred_input():
                world.calls.append("deferred-input")
                raise failure

            ledger.before_abandonment = publish_deferred_input
        with pytest.raises(OSError):
            await settlement.settle(
                ledger, RunExit(RunExitKind.CANCELLED, asyncio.CancelledError())
            )
        return world, ledger

    world, ledger = asyncio.run(scenario())
    first = "decide" if failing == "decision" else "deferred-input"
    assert world.calls == [first, "environment", "fence"]
    assert ledger.stage is OutcomeStage.ADMITTED
