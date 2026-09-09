from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError
from tests.core.test_explicit_session_compaction import UsageCompactionProvider
from tests.core.test_queued_session_messages import (
    BlockingApprovalProvider,
    BlockingTool,
    ToolRoundProvider,
)
from tests.core.test_session_store_shared_conformance import (
    conformance_postgres_dsn as conformance_postgres_dsn,
)
from tests.core.test_user_input import _ScriptedProvider

from cayu import (
    AgentSpec,
    AlwaysRequireApprovalToolPolicy,
    CayuApp,
    CheckpointCompactionContextPolicy,
    EnqueueSessionMessageRequest,
    EventType,
    InMemorySessionStore,
    Message,
    ModelCompactor,
    ModelStreamEvent,
    PostgresSessionStore,
    ResumeRequest,
    RunRequest,
    SessionIdentity,
    SessionStatus,
    SQLiteSessionStore,
    ToolApprovalDecision,
    ToolApprovalRequest,
    UserInputResponse,
)
from cayu.runtime.execution_profiles import active_invocation_execution_profile_from_checkpoint
from cayu.runtime.session_steering import (
    SessionSteeringConflict,
    SessionSteeringReceipt,
    StopAfterCurrentToolRoundRequest,
    copy_stop_after_current_tool_round_request,
)
from cayu.tools.user_input import UserInputTool
from cayu.vaults.redaction import SecretRedactor


def _request(**changes) -> StopAfterCurrentToolRoundRequest:
    return StopAfterCurrentToolRoundRequest(
        **{
            "session_id": "session",
            "session_instance_id": "instance",
            "interaction_id": "interaction",
            "expected_run_epoch": 1,
            "idempotency_key": "stop-1",
            **changes,
        }
    )


@pytest.mark.parametrize("epoch", [True, False, 0, -1, 2**63, "1", 1.0])
def test_stop_request_requires_exact_durable_epoch(epoch) -> None:
    with pytest.raises(ValidationError):
        _request(expected_run_epoch=epoch)


@pytest.mark.parametrize(
    "field", ["session_id", "session_instance_id", "interaction_id", "idempotency_key"]
)
@pytest.mark.parametrize("value", ["", " padded ", "bad\0name", "bad\ud800name", "x" * 513, 1])
def test_stop_request_rejects_invalid_identity(field, value) -> None:
    with pytest.raises(ValidationError):
        _request(**{field: value})


def test_stop_request_copy_revalidates_mutated_fields_without_serialization(recwarn) -> None:
    class Canary:
        def __repr__(self) -> str:
            return "PRIVATE-CANARY"

    request = _request()
    object.__setattr__(request, "idempotency_key", Canary())
    with pytest.raises(ValidationError) as caught:
        copy_stop_after_current_tool_round_request(request)
    assert "PRIVATE-CANARY" not in str(caught.value)
    assert "PRIVATE-CANARY" not in repr(caught.value)
    assert not recwarn


def test_stop_request_copy_detaches_valid_request() -> None:
    request = _request()
    copied = copy_stop_after_current_tool_round_request(request)
    assert copied == request
    assert copied is not request
    assert str(SessionSteeringConflict()) == (
        "Session steering authority or accepted request conflicts."
    )


@pytest.mark.parametrize(
    "field", ["session_id", "session_instance_id", "interaction_id", "idempotency_key"]
)
@pytest.mark.parametrize("malformed", [False, True])
def test_public_stop_rejects_secret_or_mutated_identity_before_store_access(
    field, malformed, recwarn, caplog, capsys
) -> None:
    canary = "PRIVATE-STEERING-CREDENTIAL"

    class Canary:
        def __repr__(self) -> str:
            return canary

    class ReadCanaryStore(InMemorySessionStore):
        reads = 0

        async def load_session_operation(self, *args, **kwargs):
            self.reads += 1
            raise AssertionError("Rejected request reached the store.")

    async def scenario() -> None:
        store = ReadCanaryStore()
        app = CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor(canary),
            enable_logging=False,
        )
        request = _request(**{field: canary})
        if malformed:
            # Keep the valid secret-bearing sibling to detect whole-model
            # diagnostics as well as wrong-type serializer warnings.
            object.__setattr__(request, "expected_run_epoch", Canary())
        with pytest.raises(ValidationError if malformed else ValueError) as caught:
            await app.stop_after_current_tool_round(request)
        assert canary not in str(caught.value)
        assert canary not in repr(caught.value)
        assert store.reads == 0
        assert not recwarn
        output = capsys.readouterr()
        assert canary not in output.out + output.err + caplog.text

    asyncio.run(scenario())


@pytest.mark.parametrize("version", [True, False, 0, 2, "1", 1.0])
def test_steering_receipt_requires_exact_schema_version(version) -> None:
    with pytest.raises(ValidationError):
        SessionSteeringReceipt(
            schema_version=version, request=_request(), execution_profile_fingerprint="a" * 64
        )


@pytest.mark.parametrize("version", [None, False, True, 0, 2, "1", 1.0])
def test_unsupported_store_rejects_before_steering_publication(version) -> None:
    class UnsupportedStore(InMemorySessionStore):
        session_steering_version = version
        publications = 0

        async def publish_session_operation(self, *args, **kwargs):
            self.publications += 1
            raise AssertionError("Unsupported store received a mutation.")

    async def scenario() -> None:
        store = UnsupportedStore()
        app = CayuApp(session_store=store, enable_logging=False)
        before = await store.create(
            RunRequest(agent_name="assistant", session_id="session", messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        with pytest.raises(RuntimeError, match="does not support atomic safe steering"):
            await app.stop_after_current_tool_round(_request())
        assert store.publications == 0
        assert await store.load("session") == before

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_stop_during_compaction_settles_summary_before_stopping(tmp_path, backend) -> None:
    async def scenario() -> None:
        class PausedCompactor(UsageCompactionProvider):
            def __init__(self) -> None:
                super().__init__()
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
            else SQLiteSessionStore(tmp_path / "steered-compaction.db")
        )
        app = CayuApp(session_store=store, enable_logging=False)
        controller = CayuApp(session_store=store, enable_logging=False)
        provider = UsageCompactionProvider(summary="done")
        compactor = PausedCompactor()
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=compactor, model="summary-model"),
                max_user_turns=1,
                compact_after_messages=2,
            ),
        )

        async def run():
            return [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="steered-compaction",
                        messages=[
                            Message.text("user", "old"),
                            Message.text("assistant", "old answer"),
                            Message.text("user", "current"),
                        ],
                    )
                )
            ]

        owner = asyncio.create_task(run())
        try:
            await asyncio.wait_for(compactor.started.wait(), timeout=15)
            session = await store.load("steered-compaction")
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
                    idempotency_key="stop-after-summary",
                )
            )
            assert not owner.done() and owner.cancelling() == 0
            compactor.release.set()
            events = await asyncio.wait_for(owner, timeout=20)
            assert compactor.calls == 1
            assert provider.calls == 0
            assert events[-1].type is EventType.SESSION_INTERRUPTED
            durable = await store.load_events(session.id)
            assert sum(e.type is EventType.CONTEXT_COMPACTION_COMPLETED for e in durable) == 1
            assert not any(e.type is EventType.SESSION_FAILED for e in durable)
            assert await store.load_active_model_completion_stage(session.id) is None
            resumed = [
                event
                async for event in app.resume(
                    ResumeRequest(
                        session_id=session.id, messages=[Message.text("user", "continue")]
                    )
                )
            ]
            assert resumed[-1].type is EventType.SESSION_COMPLETED
            assert provider.calls == 1
        finally:
            compactor.release.set()
            if not owner.done():
                owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "method",
    [
        "publish_session_operation",
        "_publish_session_operation",
        "prepare_model_completion_stage",
        "_prepare_model_completion_stage_atomic",
        "deliver_queued_session_messages",
    ],
)
def test_custom_override_must_explicitly_own_steering_protocol(method) -> None:
    async def unverified_override(self, *args, **kwargs):
        raise AssertionError("Unverified override was invoked.")

    custom_type = type("CustomStore", (InMemorySessionStore,), {method: unverified_override})

    async def scenario() -> None:
        store = custom_type()
        await store.create(
            RunRequest(agent_name="assistant", session_id="session", messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        app = CayuApp(session_store=store, enable_logging=False)
        with pytest.raises(RuntimeError, match="does not support atomic safe steering"):
            await app.stop_after_current_tool_round(_request())

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("max_steps", [1, 8])
@pytest.mark.parametrize("lost_ack", [False, True, "cancel"])
def test_steering_finishes_tool_then_continuation_delivers_queue_once(
    tmp_path, backend, max_steps, lost_ack, monkeypatch
) -> None:
    async def scenario() -> None:
        class MultiToolRoundProvider(ToolRoundProvider):
            async def stream(self, request):
                if self.requests:
                    async for event in super().stream(request):
                        yield event
                    return
                self.requests.append(request)
                for call_id in ("first-tool", "second-tool"):
                    yield ModelStreamEvent.tool_call(id=call_id, name="blocking_tool", arguments={})
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})

        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "sessions.db")
        )
        app = CayuApp(session_store=store, enable_logging=False)
        controller = CayuApp(session_store=store, enable_logging=False)
        provider = MultiToolRoundProvider()
        tool = BlockingTool()
        app.register_provider(provider)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])

        async def execute() -> None:
            async for _ in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="steered",
                    messages=[Message.text("user", "use the tool")],
                    max_steps=max_steps,
                )
            ):
                pass

        owner = asyncio.create_task(execute())
        try:
            await asyncio.wait_for(tool.started.wait(), timeout=15)
            queued = await controller.enqueue_session_message(
                EnqueueSessionMessageRequest(
                    session_id="steered",
                    idempotency_key="correction",
                    content="focus on Y",
                    delivery_mode="next_turn",
                )
            )
            session = await store.load("steered")
            assert session is not None
            profile = active_invocation_execution_profile_from_checkpoint(
                await store.load_checkpoint(session.id)
            )
            assert profile is not None
            request = StopAfterCurrentToolRoundRequest(
                session_id=session.id,
                session_instance_id=session.instance_id,
                interaction_id=profile.interaction_id,
                expected_run_epoch=session.run_epoch,
                idempotency_key="stop-1",
            )
            for field, value in (
                ("session_instance_id", "another-instance"),
                ("interaction_id", "another-interaction"),
                ("expected_run_epoch", request.expected_run_epoch + 1),
            ):
                with pytest.raises(SessionSteeringConflict):
                    await controller.stop_after_current_tool_round(
                        request.model_copy(update={field: value})
                    )
            if lost_ack:
                original = store.publish_session_operation
                committed = asyncio.Event()

                async def lose_ack(*args, **kwargs):
                    await original(*args, **kwargs)
                    committed.set()
                    if lost_ack == "cancel":
                        await asyncio.Event().wait()
                    raise OSError("lost steering acknowledgement")

                with monkeypatch.context() as patch:
                    patch.setattr(store, "publish_session_operation", lose_ack)
                    if lost_ack == "cancel":
                        accepting = asyncio.create_task(
                            controller.stop_after_current_tool_round(request)
                        )
                        try:
                            await asyncio.wait_for(committed.wait(), timeout=15)
                            accepting.cancel()
                            with pytest.raises(asyncio.CancelledError):
                                await accepting
                            assert accepting.cancelled()
                            assert accepting.cancelling() == 1
                        finally:
                            if not accepting.done():
                                accepting.cancel()
                            await asyncio.gather(accepting, return_exceptions=True)
                    else:
                        with pytest.raises(OSError, match="lost steering acknowledgement"):
                            await controller.stop_after_current_tool_round(request)
            receipt = await controller.stop_after_current_tool_round(request)
            for field, value in (
                ("expected_run_epoch", request.expected_run_epoch + 1),
                ("idempotency_key", "different-request"),
            ):
                with pytest.raises(SessionSteeringConflict):
                    await controller.stop_after_current_tool_round(
                        request.model_copy(update={field: value})
                    )
            assert await controller.stop_after_current_tool_round(request) == receipt
            during = await store.load(session.id)
            assert during is not None and during.status is SessionStatus.RUNNING
            assert not owner.done()
            assert owner.cancelling() == 0
            assert len(provider.requests) == 1
            tool.release.set()
            await asyncio.wait_for(owner, timeout=20)
            stopped = await store.load(session.id)
            assert stopped is not None and stopped.status is SessionStatus.INTERRUPTED
            assert len(provider.requests) == 1
            events = await store.load_events(session.id)
            assert sum(event.type is EventType.SESSION_INTERRUPTED for event in events) == 1
            completed = [
                i for i, event in enumerate(events) if event.type is EventType.TOOL_CALL_COMPLETED
            ]
            interrupted = next(
                i for i, event in enumerate(events) if event.type is EventType.SESSION_INTERRUPTED
            )
            assert len(completed) == 2
            assert max(completed) < interrupted
            assert not any(event.type is EventType.SESSION_MESSAGE_DELIVERED for event in events)
            assert await controller.stop_after_current_tool_round(request) == receipt
            async for _ in app.resume(
                ResumeRequest(
                    session_id=session.id,
                    messages=[Message.text("user", "continue")],
                    max_steps=max_steps,
                )
            ):
                pass
            deliveries = [
                event
                for event in await store.load_events(session.id)
                if event.type is EventType.SESSION_MESSAGE_DELIVERED
            ]
            assert len(deliveries) == 1
            assert deliveries[0].payload["queue_id"] == queued.message.queue_id
            assert len(provider.requests) == 2
            assert provider.requests[1].messages[-1].content[0].text == "focus on Y"
        finally:
            tool.release.set()
            if not owner.done():
                owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_accepted_stop_waits_for_approval_and_settled_tool(tmp_path, backend) -> None:
    async def scenario() -> None:
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "approval.db")
        )
        app = CayuApp(session_store=store, enable_logging=False)
        controller = CayuApp(session_store=store, enable_logging=False)
        provider = BlockingApprovalProvider()
        tool = BlockingTool()
        app.register_provider(provider)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[tool],
            tool_policy=AlwaysRequireApprovalToolPolicy(),
        )

        async def run():
            return [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="approval-steered",
                        messages=[Message.text("user", "use the protected tool")],
                    )
                )
            ]

        owner = asyncio.create_task(run())
        try:
            await asyncio.wait_for(provider.first_started.wait(), timeout=15)
            session = await store.load("approval-steered")
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
                    idempotency_key="stop-with-approval",
                )
            )
            provider.release_first.set()
            events = await asyncio.wait_for(owner, timeout=20)
            approval = next(
                event for event in events if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
            )
            assert not tool.started.is_set()
            assert len(provider.requests) == 1
            tool.release.set()
            resolved = [
                event
                async for event in app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id=session.id,
                        approval_id=approval.payload["approval"]["approval_id"],
                        tool_round_id=approval.payload["tool_round_id"],
                        tool_call_id=approval.payload["tool_call_id"],
                        decision=ToolApprovalDecision.APPROVE,
                    )
                )
            ]
            assert tool.started.is_set()
            assert len(provider.requests) == 1
            assert any(event.type is EventType.SESSION_INTERRUPTED for event in resolved)
            assert not any(event.type is EventType.SESSION_FAILED for event in resolved)
            final = await store.load(session.id)
            assert final is not None and final.status is SessionStatus.INTERRUPTED
        finally:
            provider.release_first.set()
            tool.release.set()
            if not owner.done():
                owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_accepted_stop_preserves_user_input_pause_and_answer(tmp_path, backend) -> None:
    async def scenario() -> None:
        class PausingProvider(_ScriptedProvider):
            def __init__(self) -> None:
                super().__init__([("ask-1", "ask_user", {"question": "Which environment?"})])
                self.started = asyncio.Event()
                self.release = asyncio.Event()

            async def stream(self, request):
                if not self.requests:
                    self.started.set()
                    await self.release.wait()
                async for event in super().stream(request):
                    yield event

        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "user-input.db")
        )
        app = CayuApp(session_store=store, enable_logging=False)
        controller = CayuApp(session_store=store, enable_logging=False)
        provider = PausingProvider()
        app.register_provider(provider)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[UserInputTool()])

        async def run():
            return [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="user-input-steered",
                        messages=[Message.text("user", "ask before continuing")],
                    )
                )
            ]

        owner = asyncio.create_task(run())
        try:
            await asyncio.wait_for(provider.started.wait(), timeout=15)
            session = await store.load("user-input-steered")
            assert session is not None
            profile = active_invocation_execution_profile_from_checkpoint(
                await store.load_checkpoint(session.id)
            )
            assert profile is not None
            queued = await controller.enqueue_session_message(
                EnqueueSessionMessageRequest(
                    session_id=session.id,
                    content="focus on Y",
                    idempotency_key="queued-input-correction",
                    delivery_mode="next_turn",
                )
            )
            await controller.stop_after_current_tool_round(
                StopAfterCurrentToolRoundRequest(
                    session_id=session.id,
                    session_instance_id=session.instance_id,
                    interaction_id=profile.interaction_id,
                    expected_run_epoch=session.run_epoch,
                    idempotency_key="stop-with-input",
                )
            )
            provider.release.set()
            paused = await asyncio.wait_for(owner, timeout=20)
            awaiting = next(
                event for event in paused if event.type is EventType.SESSION_AWAITING_USER_INPUT
            )
            pause_terminal = [
                event for event in paused if event.type is EventType.SESSION_INTERRUPTED
            ]
            assert len(pause_terminal) == 1
            assert pause_terminal[0].payload["interruption_type"] == "user_input_required"
            answered = [
                event
                async for event in app.resolve_user_input(
                    UserInputResponse(
                        session_id=session.id,
                        input_id=awaiting.payload["input_id"],
                        answer="staging",
                    )
                )
            ]
            assert len(provider.requests) == 1
            assert any(event.type is EventType.SESSION_INTERRUPTED for event in answered)
            assert any(event.type is EventType.TOOL_CALL_COMPLETED for event in answered)
            events = await store.load_events(session.id)
            assert not any(event.type is EventType.SESSION_MESSAGE_DELIVERED for event in events)
            resumed = [
                event
                async for event in app.resume(
                    ResumeRequest(
                        session_id=session.id, messages=[Message.text("user", "continue")]
                    )
                )
            ]
            assert resumed[-1].type is EventType.SESSION_COMPLETED
            assert len(provider.requests) == 2
            deliveries = [
                event
                for event in await store.load_events(session.id)
                if event.type is EventType.SESSION_MESSAGE_DELIVERED
            ]
            assert len(deliveries) == 1
            assert deliveries[0].payload["queue_id"] == queued.message.queue_id
            assert provider.requests[-1].messages[-1].content[0].text == "focus on Y"
        finally:
            provider.release.set()
            if not owner.done():
                owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_stop_accepted_after_round_poll_prevents_next_model_dispatch(
    tmp_path, backend, request
) -> None:
    """Acceptance must compose with dispatch admission, not just a loop-local poll."""

    dsn = request.getfixturevalue("conformance_postgres_dsn") if backend == "postgres" else None

    async def scenario() -> None:
        base = {
            "memory": InMemorySessionStore,
            "sqlite": SQLiteSessionStore,
            "postgres": PostgresSessionStore,
        }[backend]
        preparing_next = asyncio.Event()
        release_preparation = asyncio.Event()

        class PausedPreparationStore(base):
            invocation_lifecycle_command_version = 1
            terminal_interaction_publication_version = 1
            session_steering_version = 1
            preparation_count = 0

            async def prepare_model_completion_stage(self, *args, **kwargs):
                self.preparation_count += 1
                if self.preparation_count == 2:
                    preparing_next.set()
                    await release_preparation.wait()
                return await super().prepare_model_completion_stage(*args, **kwargs)

        if backend == "memory":
            store = PausedPreparationStore()
        elif backend == "sqlite":
            store = PausedPreparationStore(tmp_path / "dispatch-race.db")
        else:
            store = PausedPreparationStore(dsn, min_size=1, max_size=2)
        app = CayuApp(session_store=store, enable_logging=False)
        controller = CayuApp(session_store=store, enable_logging=False)
        provider = ToolRoundProvider()
        tool = BlockingTool()
        tool.release.set()
        app.register_provider(provider)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])

        async def run():
            return [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="steering-dispatch-race",
                        messages=[Message.text("user", "use the tool")],
                    )
                )
            ]

        owner = asyncio.create_task(run())
        try:
            await asyncio.wait_for(preparing_next.wait(), timeout=15)
            session = await store.load("steering-dispatch-race")
            assert session is not None
            profile = active_invocation_execution_profile_from_checkpoint(
                await store.load_checkpoint(session.id)
            )
            assert profile is not None
            assert await store.load_active_model_completion_stage(session.id) is None
            await controller.stop_after_current_tool_round(
                StopAfterCurrentToolRoundRequest(
                    session_id=session.id,
                    session_instance_id=session.instance_id,
                    interaction_id=profile.interaction_id,
                    expected_run_epoch=session.run_epoch,
                    idempotency_key="stop-before-dispatch",
                )
            )
            release_preparation.set()
            events = await asyncio.wait_for(owner, timeout=20)
            assert len(provider.requests) == 1
            assert events[-1].type is EventType.SESSION_INTERRUPTED
        finally:
            release_preparation.set()
            if not owner.done():
                owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)
            if isinstance(store, (SQLiteSessionStore, PostgresSessionStore)):
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_stop_accepted_before_queued_interaction_handoff_preserves_queue(
    tmp_path, backend, request
) -> None:
    dsn = request.getfixturevalue("conformance_postgres_dsn") if backend == "postgres" else None

    async def scenario() -> None:
        if backend == "memory":
            store = InMemorySessionStore()
        elif backend == "sqlite":
            store = SQLiteSessionStore(tmp_path / "queue-handoff.db")
        else:
            store = PostgresSessionStore(dsn, min_size=1, max_size=2)
        app = CayuApp(session_store=store, enable_logging=False)
        controller = CayuApp(session_store=store, enable_logging=False)
        provider = ToolRoundProvider()
        tool = BlockingTool()
        completed_interaction = asyncio.Event()
        release_handoff = asyncio.Event()
        app.register_provider(provider)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])

        async def run():
            events = []
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="steered-queue-handoff",
                    messages=[Message.text("user", "use the tool")],
                )
            ):
                events.append(event)
                if event.type is EventType.INTERACTION_COMPLETED:
                    completed_interaction.set()
                    await release_handoff.wait()
            return events

        owner = asyncio.create_task(run())
        try:
            await asyncio.wait_for(tool.started.wait(), timeout=15)
            await controller.enqueue_session_message(
                EnqueueSessionMessageRequest(
                    session_id="steered-queue-handoff",
                    idempotency_key="correction",
                    content="focus on Y",
                    delivery_mode="next_turn",
                )
            )
            tool.release.set()
            await asyncio.wait_for(completed_interaction.wait(), timeout=15)
            session = await store.load("steered-queue-handoff")
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
                    idempotency_key="stop-before-handoff",
                )
            )
            release_handoff.set()
            events = await asyncio.wait_for(owner, timeout=20)
            assert len(provider.requests) == 2
            assert events[-1].type is EventType.SESSION_INTERRUPTED
            assert not any(e.type is EventType.SESSION_MESSAGE_DELIVERED for e in events)
            resumed = [
                event
                async for event in app.resume(
                    ResumeRequest(
                        session_id=session.id, messages=[Message.text("user", "continue")]
                    )
                )
            ]
            assert resumed[-1].type is EventType.SESSION_COMPLETED
            assert len(provider.requests) == 3
            durable = await store.load_events(session.id)
            assert sum(e.type is EventType.SESSION_MESSAGE_DELIVERED for e in durable) == 1
            assert provider.requests[-1].messages[-1].content[0].text == "focus on Y"
        finally:
            tool.release.set()
            release_handoff.set()
            if not owner.done():
                owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)
            if isinstance(store, (SQLiteSessionStore, PostgresSessionStore)):
                await store.close()

    asyncio.run(scenario())
