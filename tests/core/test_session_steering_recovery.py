from __future__ import annotations

import asyncio

import pytest
from tests.core.test_provider_operation_offline_recovery import (
    _OfflineOperationAdapter,
    _OfflineOperationProvider,
    _prepare_explicit_fallback_resolution,
)
from tests.core.test_queued_session_messages import BlockingTool, ToolRoundProvider
from tests.core.test_structured_output_tool_round_recovery import _answer_spec, _RecordingProvider

from cayu import (
    AgentSpec,
    CayuApp,
    EnqueueSessionMessageRequest,
    EventType,
    IncompleteSessionRecoveryRequest,
    InMemorySessionStore,
    Message,
    ResumeRequest,
    RunRequest,
    SessionStatus,
    SQLiteSessionStore,
    StopAfterCurrentToolRoundRequest,
)
from cayu._exception_groups import exception_cause
from cayu.providers import ModelStreamEvent, ProviderOperationConnection, ProviderOperationStatus
from cayu.runtime.execution_profiles import active_invocation_execution_profile_from_checkpoint
from cayu.runtime.provider_operations import (
    inspect_provider_operation,
    load_pending_provider_operation_disposition,
)
from cayu.runtime.structured_output import STRUCTURED_OUTPUT_TOOL_NAME


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_safe_stop_before_explicit_fallback_dispatch_retires_pending_disposition(
    tmp_path, backend
) -> None:
    async def scenario() -> None:
        class PausedFallbackProvider(_OfflineOperationProvider):
            name = "steered-fallback"

            def __init__(self):
                super().__init__(ProviderOperationStatus.UNAVAILABLE)
                self.entered = asyncio.Event()
                self.release = asyncio.Event()

            async def billing_identity_for_request(self, request):
                self.entered.set()
                await self.release.wait()
                return None

        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "fallback.db")
        )
        provider = PausedFallbackProvider()
        app, resolution = await _prepare_explicit_fallback_resolution(
            store,
            session_id="steered-fallback",
            provider=provider,
            recovery_context={"max_steps": 2},
        )
        controller = CayuApp(session_store=store, enable_logging=False)

        async def resolve():
            return [event async for event in app.resolve_provider_operation(resolution)]

        owner = asyncio.create_task(resolve())
        try:
            await asyncio.wait_for(provider.entered.wait(), timeout=15)
            session = await store.load(resolution.session_id)
            assert session is not None
            profile = active_invocation_execution_profile_from_checkpoint(
                await store.load_checkpoint(session.id)
            )
            assert profile is not None
            await controller.stop_after_current_tool_round(
                StopAfterCurrentToolRoundRequest(
                    session_id=session.id,
                    session_instance_id=session.instance_id,
                    interaction_id=profile.interaction_id,
                    expected_run_epoch=session.run_epoch,
                    idempotency_key="stop-before-fallback",
                )
            )
            assert not owner.done() and owner.cancelling() == 0
            provider.release.set()
            events = await asyncio.wait_for(owner, timeout=20)
            assert any(event.type is EventType.SESSION_INTERRUPTED for event in events)
            assert await load_pending_provider_operation_disposition(store, session.id) is None
            assert provider.adapter.start_calls == 0
            stopped = await store.load(session.id)
            assert stopped is not None and stopped.status is SessionStatus.INTERRUPTED
        finally:
            provider.release.set()
            if not owner.done():
                owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize(
    "lost_ack", ["exception", "cancel", "cancel_cleanup_failure", "cancel_read_failure"]
)
def test_steering_promotion_ack_loss_preserves_terminal_and_cancellation(
    tmp_path, backend, lost_ack
) -> None:
    async def scenario() -> None:
        base = InMemorySessionStore if backend == "memory" else SQLiteSessionStore
        cleanup_failure = OSError("cooperative terminal store unavailable")
        reject_terminal = lost_ack == "cancel_cleanup_failure"
        delivered_cancellation: asyncio.CancelledError | None = None
        classification_failure = OSError("safe-stop classification read failed")
        fail_next_checkpoint = False

        class PromotionStore(base):
            invocation_lifecycle_command_version = 1
            terminal_interaction_publication_version = 1

            async def transition_status_and_checkpoint(self, *args, **kwargs):
                nonlocal delivered_cancellation, fail_next_checkpoint
                result = await super().transition_status_and_checkpoint(*args, **kwargs)
                if kwargs.get("to_status") is SessionStatus.INTERRUPTING and not committed.is_set():
                    committed.set()
                    if lost_ack != "exception":
                        try:
                            await asyncio.Event().wait()
                        except asyncio.CancelledError as cancellation:
                            delivered_cancellation = cancellation
                            fail_next_checkpoint = lost_ack == "cancel_read_failure"
                            raise
                    raise OSError("promotion acknowledgement lost")
                return result

            async def load_checkpoint(self, *args, **kwargs):
                nonlocal fail_next_checkpoint
                if fail_next_checkpoint:
                    fail_next_checkpoint = False
                    raise classification_failure
                return await super().load_checkpoint(*args, **kwargs)

            async def publish_interaction_transition(self, *args, **kwargs):
                if reject_terminal and kwargs.get("to_status") is SessionStatus.INTERRUPTED:
                    raise cleanup_failure
                return await super().publish_interaction_transition(*args, **kwargs)

        committed = asyncio.Event()
        store = (
            PromotionStore() if backend == "memory" else PromotionStore(tmp_path / "promotion.db")
        )
        provider = ToolRoundProvider()
        tool = BlockingTool()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])
        controller = CayuApp(session_store=store, enable_logging=False)

        async def run():
            return [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="promotion-loss",
                        messages=[Message.text("user", "investigate")],
                    )
                )
            ]

        owner = asyncio.create_task(run())
        try:
            await asyncio.wait_for(tool.started.wait(), timeout=15)
            session = await store.load("promotion-loss")
            assert session is not None
            profile = active_invocation_execution_profile_from_checkpoint(
                await store.load_checkpoint(session.id)
            )
            assert profile is not None
            await controller.enqueue_session_message(
                EnqueueSessionMessageRequest(
                    session_id=session.id,
                    content="focus on Y",
                    delivery_mode="next_turn",
                    idempotency_key="correction",
                )
            )
            await controller.stop_after_current_tool_round(
                StopAfterCurrentToolRoundRequest(
                    session_id=session.id,
                    session_instance_id=session.instance_id,
                    interaction_id=profile.interaction_id,
                    expected_run_epoch=session.run_epoch,
                    idempotency_key="stop-before-lost-promotion-ack",
                )
            )
            tool.release.set()
            await asyncio.wait_for(committed.wait(), timeout=15)
            if lost_ack != "exception":
                owner.cancel()
                with pytest.raises(asyncio.CancelledError) as caught:
                    await asyncio.wait_for(owner, timeout=20)
                assert owner.cancelled() and owner.cancelling() == 1
                assert caught.value is delivered_cancellation
                if lost_ack == "cancel_read_failure":
                    assert exception_cause(caught.value) is classification_failure
                    await app.recover_incomplete_session(
                        IncompleteSessionRecoveryRequest(
                            session_id=session.id, inactive_for_seconds=0
                        )
                    )
                if reject_terminal:
                    retained = exception_cause(caught.value)
                    assert retained is not None

                    def leaves(error):
                        if isinstance(error, BaseExceptionGroup):
                            return [leaf for child in error.exceptions for leaf in leaves(child)]
                        return [error]

                    assert sum(error is cleanup_failure for error in leaves(retained)) == 1
                    assert len(provider.requests) == 1
                    assert not any(
                        e.type is EventType.SESSION_MESSAGE_DELIVERED
                        for e in await store.load_events(session.id)
                    )
                    reject_terminal = False
                    await app.recover_incomplete_session(
                        IncompleteSessionRecoveryRequest(
                            session_id=session.id, inactive_for_seconds=0
                        )
                    )
            else:
                await asyncio.wait_for(owner, timeout=20)
                assert not owner.cancelled() and owner.cancelling() == 0
            stopped = await store.load(session.id)
            assert stopped is not None and stopped.status is SessionStatus.INTERRUPTED
            events = await store.load_events(session.id)
            assert sum(e.type is EventType.SESSION_INTERRUPTED for e in events) == 1
            assert sum(e.type is EventType.TOOL_CALL_COMPLETED for e in events) == 1
            assert not any(e.type is EventType.SESSION_MESSAGE_DELIVERED for e in events)
            assert len(provider.requests) == 1
            # A terminal event alone is insufficient: fence release must permit
            # the next public invocation and preserve exactly-once delivery.
            async for _ in app.resume(
                ResumeRequest(session_id=session.id, messages=[Message.text("user", "continue")])
            ):
                pass
            events = await store.load_events(session.id)
            assert sum(e.type is EventType.SESSION_MESSAGE_DELIVERED for e in events) == 1
            assert len(provider.requests) == 2
        finally:
            tool.release.set()
            if not owner.done():
                owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("valid", [True, False])
def test_steering_publishes_complete_synthetic_round_before_terminal(
    tmp_path, backend, valid
) -> None:
    async def scenario() -> None:
        class PausedProvider(_RecordingProvider):
            def __init__(self) -> None:
                super().__init__(
                    [
                        [
                            ModelStreamEvent.tool_call(
                                id="final-answer",
                                name=STRUCTURED_OUTPUT_TOOL_NAME,
                                arguments={
                                    "output": {"answer": "done"} if valid else {"wrong": "value"}
                                },
                            ),
                            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                        ]
                    ]
                )
                self.started = asyncio.Event()
                self.release = asyncio.Event()

            async def stream(self, request):
                self.started.set()
                await self.release.wait()
                async for event in super().stream(request):
                    yield event

        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "synthetic-round.db")
        )
        provider = PausedProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        controller = CayuApp(session_store=store, enable_logging=False)

        async def run():
            return [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="steered-synthetic",
                        messages=[Message.text("user", "answer structurally")],
                        structured_output=_answer_spec(),
                    )
                )
            ]

        owner = asyncio.create_task(run())
        try:
            await asyncio.wait_for(provider.started.wait(), timeout=15)
            session = await store.load("steered-synthetic")
            assert session is not None
            profile = active_invocation_execution_profile_from_checkpoint(
                await store.load_checkpoint(session.id)
            )
            assert profile is not None
            await controller.stop_after_current_tool_round(
                StopAfterCurrentToolRoundRequest(
                    session_id=session.id,
                    session_instance_id=session.instance_id,
                    interaction_id=profile.interaction_id,
                    expected_run_epoch=session.run_epoch,
                    idempotency_key="stop-after-synthetic",
                )
            )
            provider.release.set()
            events = await asyncio.wait_for(owner, timeout=20)
            assert len(provider.requests) == 1
            assert events[-1].type is EventType.SESSION_INTERRUPTED
            durable = await store.load_events(session.id)
            terminal_type = EventType.TOOL_CALL_COMPLETED if valid else EventType.TOOL_CALL_FAILED
            assert sum(e.type is terminal_type for e in durable) == 1
            assert sum(e.type is terminal_type for e in events) == 1
            validation_type = (
                EventType.STRUCTURED_OUTPUT_VALIDATED
                if valid
                else EventType.STRUCTURED_OUTPUT_FAILED
            )
            assert sum(e.type is validation_type for e in durable) == 1
            # Public projection aliases private event IDs. Compare the ordered
            # semantic round evidence, not raw versus projected identities.
            round_types = {
                terminal_type,
                EventType.STRUCTURED_OUTPUT_VALIDATING,
                validation_type,
                EventType.STRUCTURED_OUTPUT_RETRY,
            }
            assert [e.type for e in events if e.type in round_types] == [
                e.type for e in durable if e.type in round_types
            ]
            assert await store.load_active_model_completion_stage(session.id) is None
        finally:
            provider.release.set()
            if not owner.done():
                owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_accepted_stop_does_not_replace_unknown_provider_outcome(tmp_path, backend) -> None:
    async def scenario() -> None:
        class HangingAdapter(_OfflineOperationAdapter):
            def __init__(self) -> None:
                super().__init__(ProviderOperationStatus.UNAVAILABLE)
                self.started = asyncio.Event()
                self.cancel_calls = 0

            async def start(self, request):
                self.start_calls += 1
                self.start_requests.append(request)

                async def events():
                    self.started.set()
                    await asyncio.Event().wait()
                    yield  # pragma: no cover

                return ProviderOperationConnection(
                    state=self.state,
                    status=ProviderOperationStatus.IN_PROGRESS,
                    events=events(),
                )

            async def cancel(self, state):
                self.cancel_calls += 1
                return await super().cancel(state)

        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "unknown-provider.db")
        )
        provider = _OfflineOperationProvider(ProviderOperationStatus.UNAVAILABLE)
        provider.adapter = HangingAdapter()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        controller = CayuApp(session_store=store, enable_logging=False)

        async def run():
            return [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="steered-unknown-provider",
                        messages=[Message.text("user", "perform the remote operation")],
                    )
                )
            ]

        owner = asyncio.create_task(run())
        try:
            await asyncio.wait_for(provider.adapter.started.wait(), timeout=15)
            session = await store.load("steered-unknown-provider")
            assert session is not None
            profile = active_invocation_execution_profile_from_checkpoint(
                await store.load_checkpoint(session.id)
            )
            assert profile is not None
            await controller.enqueue_session_message(
                EnqueueSessionMessageRequest(
                    session_id=session.id,
                    content="focus on Y",
                    delivery_mode="next_turn",
                    idempotency_key="correction",
                )
            )
            await controller.stop_after_current_tool_round(
                StopAfterCurrentToolRoundRequest(
                    session_id=session.id,
                    session_instance_id=session.instance_id,
                    interaction_id=profile.interaction_id,
                    expected_run_epoch=session.run_epoch,
                    idempotency_key="stop-after-provider",
                )
            )
            stage = await store.load_active_model_completion_stage(session.id)
            assert stage is not None
            assert not owner.done() and owner.cancelling() == 0
            assert provider.adapter.cancel_calls == 0
            # Abandon the waiter through real cancellation. This does not prove
            # that the remote background operation stopped.
            owner.cancel()
            with pytest.raises(asyncio.CancelledError):
                await owner
            assert owner.cancelled() and owner.cancelling() == 1
            assert await store.load_active_model_completion_stage(session.id) is not None

            recovered_provider = _OfflineOperationProvider(ProviderOperationStatus.UNAVAILABLE)
            restarted = CayuApp(session_store=store, enable_logging=False)
            restarted.register_provider(recovered_provider, default=True)
            restarted.register_agent(AgentSpec(name="assistant", model="fake-model"))
            events = [
                event
                async for event in restarted.resume(
                    ResumeRequest(
                        session_id=session.id, messages=[Message.text("user", "continue")]
                    )
                )
            ]
            assert recovered_provider.adapter.start_calls == 0
            assert provider.adapter.start_calls == 1
            assert provider.adapter.cancel_calls == 0
            retained = await store.load_active_model_completion_stage(session.id)
            assert retained is not None and retained.stage.stage_id == stage.stage.stage_id
            assert any(e.type is EventType.PROVIDER_OPERATION_RECOVERY_REQUIRED for e in events)
            inspection = await inspect_provider_operation(store, session.id)
            assert inspection.allowed_resolutions == ("fallback_retry", "fail")
            assert not any(
                e.type is EventType.SESSION_MESSAGE_DELIVERED
                for e in await store.load_events(session.id)
            )
        finally:
            if not owner.done():
                owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(scenario())
