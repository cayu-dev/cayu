from __future__ import annotations

import asyncio

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    EnqueueSessionMessageRequest,
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    EnvironmentSpec,
    EventType,
    Message,
    ModelProvider,
    RunRequest,
    SessionMessageDeliveryMode,
)
from cayu.providers import ModelRequest, ModelStreamEvent


class _QueuedProvider(ModelProvider):
    name = "queued-exposure"

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest):
        self.requests.append(request)
        if len(self.requests) == 1:
            self.started.set()
            await self.release.wait()
        yield ModelStreamEvent.text_delta("Done.")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _EmptyFactory(EnvironmentFactory):
    def __init__(self) -> None:
        self.calls = 0

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        self.calls += 1
        return EnvironmentFactoryResult(
            environment=Environment(EnvironmentSpec(name=request.environment_name))
        )


@pytest.mark.parametrize("factory_backed", [False, True])
@pytest.mark.parametrize("enqueue", [False, True])
def test_public_on_idle_preserves_environment_exposure(factory_backed: bool, enqueue: bool) -> None:
    async def run() -> None:
        provider = _QueuedProvider()
        factory = _EmptyFactory()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider)
        spec = EnvironmentSpec(name="local")
        if factory_backed:
            app.register_environment_factory(spec, factory, default=True)
        else:
            app.register_environment(Environment(spec), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="test-model", provider_name=provider.name)
        )

        async def consume():
            return [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="queued-exposure",
                        messages=[Message.text("user", "Start.")],
                    )
                )
            ]

        task = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(provider.started.wait(), 5)
            if enqueue:
                request = EnqueueSessionMessageRequest(
                    session_id="queued-exposure",
                    idempotency_key="one",
                    content="Continue.",
                    delivery_mode=SessionMessageDeliveryMode.ON_IDLE,
                )
                queued = await app.enqueue_session_message(request)
                duplicate = await app.enqueue_session_message(request)
                assert queued.replayed is False
                assert duplicate.replayed is True
                assert duplicate.message.queue_id == queued.message.queue_id
            provider.release.set()
            events = await asyncio.wait_for(task, 10)
            assert events[-1].type is EventType.SESSION_COMPLETED
            assert not any(event.type is EventType.SESSION_FAILED for event in events)
            assert len(provider.requests) == (2 if enqueue else 1)
            deliveries = [
                event for event in events if event.type is EventType.SESSION_MESSAGE_DELIVERED
            ]
            assert len(deliveries) == int(enqueue)
            if enqueue:
                transcript = await app.session_store.load_transcript("queued-exposure")
                assert (
                    sum(
                        message.content == Message.text("user", "Continue.").content
                        for message in transcript
                    )
                    == 1
                )
            assert factory.calls == int(factory_backed)
        finally:
            provider.release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await app.drain_environment_cleanups()

    asyncio.run(run())


def test_queued_exposure_transfer_revokes_predecessor(monkeypatch: pytest.MonkeyPatch) -> None:
    from cayu.runtime import _environment_lifecycle
    from cayu.runtime._environment_exposure import require_environment_exposed

    transfer = _environment_lifecycle.transfer_queued_environment_exposure
    transfers = []

    def checked_transfer(*, session, predecessor, successor):
        environment = predecessor.registered_environment
        exposure = environment.environment_exposure
        admission = exposure.admission
        decision = admission.decision
        lock = admission.renewal_lock
        settlement = admission.settlement_task

        def require(context):
            require_environment_exposed(
                environment,
                session=session,
                invocation_context=context,
                registered_agent=context.registered_agent,
                execution_profile=context.profile,
            )

        require(predecessor)
        with pytest.raises(RuntimeError, match="exact runtime-admitted exposure authority"):
            require(successor)
        transfer(session=session, predecessor=predecessor, successor=successor)
        require(successor)
        with pytest.raises(RuntimeError, match="exact runtime-admitted exposure authority"):
            require(predecessor)
        with pytest.raises(RuntimeError, match="exact runtime-admitted exposure authority"):
            transfer(session=session, predecessor=predecessor, successor=successor)
        assert environment.environment_exposure is exposure
        assert exposure.admission is admission
        assert admission.decision is decision
        assert admission.renewal_lock is lock
        assert admission.settlement_task is settlement
        transfers.append(successor)

    monkeypatch.setattr(
        _environment_lifecycle, "transfer_queued_environment_exposure", checked_transfer
    )
    test_public_on_idle_preserves_environment_exposure(factory_backed=False, enqueue=True)
    assert len(transfers) == 1


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("proof", ["verified", "expired", "changed_image"])
@pytest.mark.parametrize("delivery_path", ["queued", "stop_resume"])
def test_queued_search_tool_preserves_exact_admission(
    monkeypatch, tmp_path, backend, proof, delivery_path
):
    from datetime import UTC, datetime, timedelta

    from cayu import (
        ExecutionAdmissionCandidate,
        ExecutionCapabilityEvidence,
        InMemorySessionStore,
        ResumeRequest,
        SearchTextTool,
        SQLiteSessionStore,
        StopAfterCurrentToolRoundRequest,
        ToolExecutableRequirement,
    )
    from cayu.environments.admission import (
        ExecutionExecutableEvidence,
        ExecutionToolRequirementEvidence,
    )
    from cayu.runners import Runner
    from cayu.runtime import _environment_lifecycle
    from cayu.runtime._environment_exposure import require_environment_exposed
    from cayu.runtime.execution_profiles import active_invocation_execution_profile_from_checkpoint

    class EvidenceRunner(Runner):
        def __init__(self):
            self.observed_at = datetime.now(UTC)
            self.image = "sha256:" + "1" * 64
            self.snapshots = 0
            self.observers = []

        def execution_admission_observer(self, requirements):
            observer = super().execution_admission_observer(requirements)
            self.observers.append(observer)
            return observer

        def execution_admission_candidate(self):
            self.snapshots += 1
            fingerprint = "sha256:" + "2" * 64
            return ExecutionAdmissionCandidate(
                candidate="local",
                evidence=ExecutionCapabilityEvidence(
                    subject="local",
                    environment_fingerprint=fingerprint,
                    image_fingerprint=self.image,
                    unclaimed_reason_code="security_unclaimed",
                    tool_requirements=ExecutionToolRequirementEvidence(
                        environment_fingerprint=fingerprint,
                        image_fingerprint=self.image,
                        executables=(
                            ExecutionExecutableEvidence(
                                executable="rg",
                                state="live_verified",
                                observed_at=self.observed_at,
                                valid_until=self.observed_at
                                + timedelta(seconds=2 if proof == "expired" else 60),
                                requirement_fingerprint=ToolExecutableRequirement(
                                    executable="rg"
                                ).fingerprint,
                            ),
                        ),
                    ),
                ),
            )

        async def exec(self, command, **kwargs):
            raise AssertionError("The scripted provider must not dispatch a tool.")

    async def run():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "queue.db")
        )
        runner = EvidenceRunner()
        provider = _QueuedProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), runner=runner), default=True
        )
        app.register_agent(
            AgentSpec(name="assistant", model="test-model", provider_name=provider.name),
            tools=[SearchTextTool()],
        )
        transfer = _environment_lifecycle.transfer_queued_environment_exposure
        transfers = []

        def checked_transfer(*, session, predecessor, successor):
            exposure = predecessor.registered_environment.environment_exposure
            observer = exposure.observer
            decision = exposure.admission.decision
            settlement = exposure.admission.settlement_task
            transfer(session=session, predecessor=predecessor, successor=successor)
            assert successor.registered_environment.environment_exposure is exposure
            assert exposure.observer is observer
            assert exposure.admission.decision is decision
            assert exposure.admission.settlement_task is settlement
            assert observer.requirements.tool_requirements[0].tool_name == "search_text"
            with pytest.raises(RuntimeError, match="exact runtime-admitted exposure authority"):
                require_environment_exposed(
                    predecessor.registered_environment,
                    session=session,
                    invocation_context=predecessor,
                    registered_agent=predecessor.registered_agent,
                    execution_profile=predecessor.profile,
                )
            transfers.append(runner.snapshots)

        monkeypatch.setattr(
            _environment_lifecycle, "transfer_queued_environment_exposure", checked_transfer
        )

        async def consume():
            return [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="queued-search",
                        messages=[Message.text("user", "Start.")],
                    )
                )
            ]

        task = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(provider.started.wait(), 10)
            await app.enqueue_session_message(
                EnqueueSessionMessageRequest(
                    session_id="queued-search",
                    idempotency_key="next",
                    content="Continue.",
                    delivery_mode=(
                        SessionMessageDeliveryMode.NEXT_TURN
                        if delivery_path == "stop_resume"
                        else SessionMessageDeliveryMode.ON_IDLE
                    ),
                )
            )
            if delivery_path == "stop_resume":
                session = await store.load("queued-search")
                profile = active_invocation_execution_profile_from_checkpoint(
                    await store.load_checkpoint("queued-search")
                )
                assert session is not None and profile is not None
                await app.stop_after_current_tool_round(
                    StopAfterCurrentToolRoundRequest(
                        session_id=session.id,
                        session_instance_id=session.instance_id,
                        interaction_id=profile.interaction_id,
                        expected_run_epoch=session.run_epoch,
                        idempotency_key="stop-before-continuation",
                    )
                )
                assert task.cancelling() == 0
            if proof == "expired":
                await asyncio.sleep(2.1)
            elif proof == "changed_image":
                runner.image = "sha256:" + "3" * 64
            provider.release.set()
            events = await asyncio.wait_for(task, 15)
            if delivery_path == "stop_resume":
                assert events[-1].type is EventType.SESSION_INTERRUPTED
                assert len(provider.requests) == 1
                assert not task.cancelled() and task.cancelling() == 0
                assert transfers == []
                initial_observer = runner.observers[-1]
                events = [
                    event
                    async for event in app.resume(
                        ResumeRequest(
                            session_id="queued-search",
                            messages=[Message.text("user", "Resume.")],
                        )
                    )
                ]
                if proof == "expired":
                    # Preflight refuses before constructing final observation.
                    assert runner.observers[-1] is initial_observer
                else:
                    assert runner.observers[-1] is not initial_observer
                assert runner.observers[-1].requirements == initial_observer.requirements
            else:
                assert len(transfers) == 1
                assert runner.snapshots > transfers[0]
            admitted = proof == "verified" or (
                delivery_path == "stop_resume" and proof == "changed_image"
            )
            assert len(provider.requests) == (2 if admitted else 1)
            if admitted:
                assert (
                    sum(
                        message.content == Message.text("user", "Continue.").content
                        for message in provider.requests[1].messages
                    )
                    == 1
                )
            assert events[-1].type is (
                EventType.SESSION_COMPLETED if admitted else EventType.SESSION_FAILED
            )
            if not admitted:
                assert events[-1].payload["error_type"] == "ExecutionAdmissionError"
            durable = await store.load_events("queued-search")
            assert sum(event.type is EventType.SESSION_MESSAGE_DELIVERED for event in durable) == (
                0 if delivery_path == "stop_resume" and not admitted else 1
            )
        finally:
            provider.release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await app.drain_environment_cleanups()
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(run())
