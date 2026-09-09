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
    from cayu.runtime import _session_engine
    from cayu.runtime._environment_exposure import require_environment_exposed

    transfer = _session_engine.transfer_queued_environment_exposure
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

    monkeypatch.setattr(_session_engine, "transfer_queued_environment_exposure", checked_transfer)
    test_public_on_idle_preserves_environment_exposure(factory_backed=False, enqueue=True)
    assert len(transfers) == 1
