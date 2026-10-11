"""Shared session-store provider recovery contract cases."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import AsyncIterator
from hashlib import sha256
from typing import Any

import pytest
from tests.core._execution_profile_fixtures import (
    profiled_session_identity,
)
from tests.core._workload_secret_support import FakeProvider
from tests.core.session_store_conformance.support import (
    _approval_recovery_environment_spec,
    _assistant_model_completion_publication,
    _automatic_recall_recovery_policy,
    _AutomaticRecallRecoveryKnowledgeStore,
    _close_store,
    _collect_events,
    _identity,
    _open_store,
    _reopen_store,
    _SimulatedProcessLoss,
    _UnusedForkProvider,
)
from tests.provider_cleanup_assertions import without_redacted_cleanup_context

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.budgets.billing import BillingIdentity
from cayu.build_provenance import (
    RuntimeBuildArtifactKind,
    RuntimeBuildProvenance,
    RuntimeBuildProvenanceOrigin,
)
from cayu.context.footprints import RequestFootprintConfig
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.evals.testing import ScriptedModelProvider
from cayu.events import (
    Event,
    EventType,
    event_with_runtime_payload_authority,
)
from cayu.execution_units import ModelAttemptIdentity
from cayu.memory.evidence import (
    ContextExposureState,
    RecallEvidenceQuery,
)
from cayu.messages import (
    Message,
    MessageRole,
    ProviderStatePart,
)
from cayu.providers.base import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.providers.operations import ProviderOperationState, ProviderOperationStatus
from cayu.runtime._event_projection import (
    PRIVATE_EVENT_AUTHORITY,
)
from cayu.runtime.authority import SessionRunFenced
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.provider_operations import (
    ProviderOperationInspectionStatus,
    ProviderOperationResolutionAction,
    ProviderOperationResolutionRequest,
    commit_provider_operation_progress,
    inspect_provider_operation,
    load_pending_provider_operation_disposition,
    load_provider_operation_resolution,
    load_recoverable_provider_operation,
    provider_operation_progress_event_id,
    provider_operation_progress_payload,
    resolve_provider_operation_stage,
)
from cayu.sessions._invocation_scope import _deactivate_session_run_fence
from cayu.sessions._model_stage import (
    ModelCompletionStageRequest,
    SessionModelCompletionStageConflict,
)
from cayu.sessions._terminal_evidence import (
    _checkpoint_with_session_run_operation,
)
from cayu.sessions.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
)
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.records import Session, SessionIdentity, SessionStatus, SessionStatusConflict
from cayu.sessions.recovery import IncompleteSessionRecoveryAction, IncompleteSessionRecoveryRequest
from cayu.sessions.requests import (
    InterruptSessionRequest,
    ResumeRequest,
    RunRequest,
)
from cayu.sessions.terminal_evidence import (
    TerminalSessionEvidenceError,
    TerminalSessionEvidenceErrorCode,
)
from cayu.storage.memory import KnowledgeEntry
from cayu.tools.base import (
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
)
from cayu.tools.exposure import (
    ToolCapabilityCeiling,
)
from cayu.vaults.base import SecretRef
from cayu.vaults.redaction import SecretRedactor
from cayu.vaults.static import StaticVault

pytestmark = pytest.mark.postgres


def test_session_store_conformance_persists_provider_cancellation_diagnostics(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)

        class BillingCollisionProvider(FakeProvider):
            def __init__(self) -> None:
                super().__init__([])
                self.hook_started = asyncio.Event()
                self.cleanup_failed = asyncio.Event()
                self.requests = 0

            async def billing_identity_for_request(
                self,
                request: ModelRequest,
            ) -> BillingIdentity | None:
                del request

                async def child() -> None:
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        self.cleanup_failed.set()
                        raise RuntimeError("private billing cleanup diagnostic") from None

                async with asyncio.TaskGroup() as group:
                    group.create_task(child())
                    self.hook_started.set()
                    await asyncio.Event().wait()
                return None

            async def stream(
                self,
                request: ModelRequest,
            ) -> AsyncIterator[ModelStreamEvent]:
                del request
                self.requests += 1
                yield ModelStreamEvent.completed({"finish_reason": "stop"})

        try:
            provider = BillingCollisionProvider()
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="assistant", model="fake-model"))
            session_id = f"provider-cancellation-{session_store_case[0]}"

            async def consume() -> None:
                async for _ in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "hello")],
                    )
                ):
                    pass

            task = asyncio.create_task(consume())
            await asyncio.wait_for(provider.hook_started.wait(), timeout=10)
            task.cancel("caller cancellation")
            with pytest.raises(asyncio.CancelledError):
                await task
            assert task.cancelling() == 1
            assert task.cancelled()
            assert provider.cleanup_failed.is_set()
            assert provider.requests == 0

            store = await _reopen_store(session_store_case, store)
            session = await store.load(session_id)
            assert session is not None
            assert session.status == SessionStatus.INTERRUPTED
            records = await store.query_events(EventQuery(session_id=session_id, limit=100))
            events = [record.event for record in records]
            assert EventType.MODEL_ERROR not in {event.type for event in events}
            assert EventType.SESSION_FAILED not in {event.type for event in events}
            interrupted_events = [
                event for event in events if event.type == EventType.SESSION_INTERRUPTED
            ]
            assert len(interrupted_events) == 1
            interrupted = interrupted_events[0]
            assert interrupted.payload["provider_cancellation_failures"] == [
                {
                    "phase": "billing_identity_for_request",
                    "error": (
                        "Provider billing identity cleanup failed during caller cancellation."
                    ),
                    "error_type": "BillingIdentityCleanupError",
                }
            ]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_repairs_provider_cancellation_terminal_publication(
    session_store_case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)

        class BillingCollisionProvider(FakeProvider):
            def __init__(self) -> None:
                super().__init__([])
                self.hook_started = asyncio.Event()

            @property
            def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
                return ExecutionProfileBehaviorIdentity(
                    name="tests:provider-cancellation-repair",
                    behavior_version="1",
                    implementation_version="1",
                )

            async def billing_identity_for_request(
                self,
                request: ModelRequest,
            ) -> BillingIdentity | None:
                del request

                async def child() -> None:
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        raise RuntimeError("private billing cleanup diagnostic") from None

                async with asyncio.TaskGroup() as group:
                    group.create_task(child())
                    self.hook_started.set()
                    await asyncio.Event().wait()
                return None

        try:
            provider = BillingCollisionProvider()
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="assistant", model="fake-model"))
            session_id = f"provider-cancellation-repair-{session_store_case[0]}"
            original_terminal_stream = app._terminal_event_publication.publish_recovered

            async def fail_interrupted_publication(request):  # type: ignore[no-untyped-def]
                if request.event.type == EventType.SESSION_INTERRUPTED:
                    raise RuntimeError("private terminal publication failure")
                async for event in original_terminal_stream(request):
                    yield event

            monkeypatch.setattr(
                app._terminal_event_publication,
                "publish_recovered",
                fail_interrupted_publication,
            )

            async def consume() -> None:
                async for _ in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "hello")],
                    )
                ):
                    pass

            task = asyncio.create_task(consume())
            await asyncio.wait_for(provider.hook_started.wait(), timeout=10)
            task.cancel("caller cancellation")
            with pytest.raises(asyncio.CancelledError) as raised:
                await task
            assert type(raised.value.__cause__) is RuntimeError

            store = await _reopen_store(session_store_case, store)
            checkpoint = await store.load_checkpoint(session_id)
            assert checkpoint is not None
            assert checkpoint["pending_session_interrupt"]["provider_cancellation_failures"] == [
                {
                    "phase": "billing_identity_for_request",
                    "error": (
                        "Provider billing identity cleanup failed during caller cancellation."
                    ),
                    "error_type": "BillingIdentityCleanupError",
                }
            ]

            recovered_app = CayuApp(session_store=store, enable_logging=False)
            recovered_app.register_provider(BillingCollisionProvider(), default=True)
            recovered_app.register_agent(AgentSpec(name="assistant", model="fake-model"))
            recovered = await recovered_app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=session_id,
                    inactive_for_seconds=0,
                )
            )
            assert any(event.type == EventType.SESSION_INTERRUPTED for event in recovered.events)
            repaired_checkpoint = await store.load_checkpoint(session_id)
            assert repaired_checkpoint is not None
            assert "pending_session_interrupt" not in repaired_checkpoint
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_recovers_provider_cancellation_after_marker_only_crash(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)

        class BillingCollisionProvider(FakeProvider):
            def __init__(self) -> None:
                super().__init__([])
                self.hook_started = asyncio.Event()

            @property
            def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
                return ExecutionProfileBehaviorIdentity(
                    name="tests:provider-cancellation-marker-crash",
                    behavior_version="1",
                    implementation_version="1",
                )

            async def billing_identity_for_request(
                self,
                request: ModelRequest,
            ) -> BillingIdentity | None:
                del request

                async def child() -> None:
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        raise RuntimeError("private billing cleanup diagnostic") from None

                async with asyncio.TaskGroup() as group:
                    group.create_task(child())
                    self.hook_started.set()
                    await asyncio.Event().wait()
                return None

        try:
            provider = BillingCollisionProvider()
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="assistant", model="fake-model"))
            session_id = f"provider-cancellation-marker-crash-{session_store_case[0]}"
            original_publish_checkpoint_and_events = store.publish_checkpoint_and_events
            marker_committed = False

            async def lose_process_after_marker(  # type: ignore[no-untyped-def]
                session_id_arg,
                **kwargs,
            ):
                nonlocal marker_committed
                result = await original_publish_checkpoint_and_events(
                    session_id_arg,
                    **kwargs,
                )
                checkpoint = await store.load_checkpoint(session_id_arg)
                marker = None if checkpoint is None else checkpoint.get("pending_session_interrupt")
                if (
                    not marker_committed
                    and type(marker) is dict
                    and "provider_cancellation_failures" in marker
                ):
                    marker_committed = True
                    raise _SimulatedProcessLoss("after provider interruption marker")
                return result

            async def consume() -> None:
                async for _ in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "hello")],
                    )
                ):
                    pass

            task = asyncio.create_task(consume())
            await asyncio.wait_for(provider.hook_started.wait(), timeout=10)
            store.publish_checkpoint_and_events = lose_process_after_marker  # type: ignore[method-assign]
            try:
                task.cancel("caller cancellation")
                with pytest.raises(
                    _SimulatedProcessLoss,
                    match="after provider interruption marker",
                ):
                    await task
            finally:
                delattr(store, "publish_checkpoint_and_events")

            store = await _reopen_store(session_store_case, store)
            before = await store.load(session_id)
            assert before is not None
            assert before.status == SessionStatus.RUNNING
            checkpoint = await store.load_checkpoint(session_id)
            assert checkpoint is not None
            marker = checkpoint["pending_session_interrupt"]
            assert marker["provider_cancellation_failures"] == [
                {
                    "phase": "billing_identity_for_request",
                    "error": (
                        "Provider billing identity cleanup failed during caller cancellation."
                    ),
                    "error_type": "BillingIdentityCleanupError",
                }
            ]

            def corrupt_provider_diagnostic(
                _session: Session,
                current: dict[str, Any] | None,
            ) -> dict[str, Any]:
                assert current is not None
                copied = copy.deepcopy(current)
                copied["pending_session_interrupt"]["provider_cancellation_failures"][0][
                    "error"
                ] = "untrusted reconstructed diagnostic"
                return copied

            good_checkpoint = copy.deepcopy(checkpoint)
            await store.transform_checkpoint(session_id, corrupt_provider_diagnostic)
            malformed_checkpoint = await store.load_checkpoint(session_id)
            recovered_app = CayuApp(session_store=store, enable_logging=False)
            recovered_app.register_provider(BillingCollisionProvider(), default=True)
            recovered_app.register_agent(AgentSpec(name="assistant", model="fake-model"))
            with pytest.raises(ValueError, match="failure error is invalid"):
                await recovered_app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(
                        session_id=session_id,
                        inactive_for_seconds=0,
                    )
                )
            assert await store.load_checkpoint(session_id) == malformed_checkpoint

            def restore_provider_diagnostic(
                _session: Session,
                _current: dict[str, Any] | None,
            ) -> dict[str, Any]:
                return copy.deepcopy(good_checkpoint)

            await store.transform_checkpoint(session_id, restore_provider_diagnostic)

            def corrupt_provider_interruption_type(
                _session: Session,
                current: dict[str, Any] | None,
            ) -> dict[str, Any]:
                assert current is not None
                copied = copy.deepcopy(current)
                copied["pending_session_interrupt"]["interruption_type"] = "future-untrusted"
                return copied

            await store.transform_checkpoint(session_id, corrupt_provider_interruption_type)
            malformed_checkpoint = await store.load_checkpoint(session_id)
            with pytest.raises(ValueError, match="interruption type is invalid"):
                await recovered_app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(
                        session_id=session_id,
                        inactive_for_seconds=0,
                    )
                )
            assert await store.load_checkpoint(session_id) == malformed_checkpoint
            await store.transform_checkpoint(session_id, restore_provider_diagnostic)

            recovered = await recovered_app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id)
            )
            assert recovered.status == SessionStatus.INTERRUPTED
            interrupted = [
                event for event in recovered.events if event.type == EventType.SESSION_INTERRUPTED
            ]
            assert len(interrupted) == 1
            assert (
                interrupted[0].payload["provider_cancellation_failures"]
                == marker["provider_cancellation_failures"]
            )
            repaired_checkpoint = await store.load_checkpoint(session_id)
            assert repaired_checkpoint is not None
            assert "pending_session_interrupt" not in repaired_checkpoint
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_persists_provider_stream_cancellation_diagnostics(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)

        class StreamCollisionIterator:
            def __init__(self) -> None:
                self.reads = 0
                self.close_started = asyncio.Event()

            def __aiter__(self) -> StreamCollisionIterator:
                return self

            async def __anext__(self) -> ModelStreamEvent:
                self.reads += 1
                if self.reads == 1:
                    return ModelStreamEvent.text_delta("partial")
                raise RuntimeError("private provider stream diagnostic")

            async def aclose(self) -> None:
                self.close_started.set()
                await asyncio.Event().wait()

        class StreamCollisionProvider(FakeProvider):
            def __init__(self) -> None:
                super().__init__([])
                self.iterator = StreamCollisionIterator()
                self.requests = 0

            def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
                del request
                self.requests += 1
                return self.iterator

        try:
            provider = StreamCollisionProvider()
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="assistant", model="fake-model"))
            session_id = f"provider-stream-cancellation-{session_store_case[0]}"

            async def consume() -> None:
                async for _ in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "hello")],
                    )
                ):
                    pass

            task = asyncio.create_task(consume())
            await asyncio.wait_for(provider.iterator.close_started.wait(), timeout=10)
            task.cancel("caller cancellation")
            with pytest.raises(asyncio.CancelledError):
                await task
            assert task.cancelling() == 1
            assert task.cancelled()
            assert provider.requests == 1

            store = await _reopen_store(session_store_case, store)
            session = await store.load(session_id)
            assert session is not None
            assert session.status == SessionStatus.INTERRUPTED
            records = await store.query_events(EventQuery(session_id=session_id, limit=100))
            events = [record.event for record in records]
            assert EventType.MODEL_ERROR not in {event.type for event in events}
            assert EventType.SESSION_FAILED not in {event.type for event in events}
            interrupted_events = [
                event for event in events if event.type == EventType.SESSION_INTERRUPTED
            ]
            assert len(interrupted_events) == 1
            interrupted = interrupted_events[0]
            assert [
                without_redacted_cleanup_context(failure)
                for failure in interrupted.payload["provider_cancellation_failures"]
            ] == [
                {
                    "phase": "model_stream",
                    "error": "Model provider stream failed before cancellation.",
                    "error_type": "ModelProviderStreamError",
                },
                {
                    "phase": "provider_stream_cleanup",
                    "error": "Provider stream cleanup did not complete normally.",
                    "error_type": "ProviderStreamCleanupError",
                    "cleanup_diagnostic_version": 1,
                    "cleanup_action": "stream_close",
                    "cleanup_reason": "cleanup_cancelled",
                    "cleanup_exception_type": "CancelledError",
                    "cleanup_failure_phase": "close",
                    "cleanup_close_exception_type": "CancelledError",
                    "cleanup_close_exception_message": "redacted",
                    "cancellation_requested": True,
                    "stream_close_state": "not_confirmed",
                    "remote_cancellation_state": "unknown",
                    "remote_settlement_state": "unknown",
                    **{
                        key: next(
                            record.event.payload[key]
                            for record in records
                            if record.event.type is EventType.MODEL_STARTED
                        )
                        for key in ("model_step_id", "model_attempt_id")
                    },
                },
            ]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_preserves_openrouter_reasoning_details(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = f"openrouter-reasoning-details-{session_store_case[0]}"
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=_identity(),
            )
            details = [
                {
                    "type": "reasoning.encrypted",
                    "data": "opaque-provider-state",
                    "signature": "signed-state",
                    "format": "future-format-v9",
                    "unknown": {"ordered": [3, 1, 2]},
                }
            ]
            await store.append_transcript_messages(
                session_id,
                [
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=[
                            ProviderStatePart(
                                provider="chat_completions",
                                state={
                                    "type": "reasoning_details",
                                    "version": 1,
                                    "details": details,
                                },
                            )
                        ],
                    )
                ],
            )
            store = await _reopen_store(session_store_case, store)

            transcript = await store.load_transcript(session_id)

            assert len(transcript) == 1
            part = transcript[0].content[0]
            assert type(part) is ProviderStatePart
            assert part.state["details"] == details
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_preserves_runtime_build_provenance_across_reopen(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        provenance = RuntimeBuildProvenance.from_artifact_digest(
            origin=RuntimeBuildProvenanceOrigin.EXPLICIT_MANIFEST,
            artifact_kind=RuntimeBuildArtifactKind.OTHER,
            artifact_digest=sha256(b"session-build-a").hexdigest(),
            source_revision="revision-a",
        )
        session_id = f"runtime-build-provenance-{session_store_case[0]}"
        try:
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=SessionIdentity(
                    provider_name="fake",
                    model="fake-model",
                    runtime_version="0.4.0",
                    runtime_build_provenance=provenance,
                ),
            )
            store = await _reopen_store(session_store_case, store)

            session = await store.load(session_id)
            inspection = await store.inspect_identity(session_id)

            assert session is not None
            assert session.runtime_build_provenance == provenance
            assert session.runtime_build_fingerprint == provenance.fingerprint
            assert session.runtime_source_revision == "revision-a"
            assert inspection.runtime_build_provenance == provenance
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_repairs_terminal_evidence_durably(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            app = CayuApp(session_store=store, enable_logging=False)

            class CompleteThenBlockProvider(ModelProvider):
                name = "fake"

                def __init__(self) -> None:
                    self.calls = 0
                    self.second_started = asyncio.Event()

                @property
                def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
                    return ExecutionProfileBehaviorIdentity(
                        name="tests:terminal-evidence-provider",
                        behavior_version="1",
                        implementation_version="1",
                    )

                async def stream(self, _request: Any):
                    self.calls += 1
                    if self.calls == 2:
                        self.second_started.set()
                        await asyncio.Event().wait()
                        raise AssertionError("unreachable")
                    yield ModelStreamEvent.completed({"finish_reason": "stop"})

            provider = CompleteThenBlockProvider()
            expected_types = {
                SessionStatus.COMPLETED: EventType.SESSION_COMPLETED,
                SessionStatus.FAILED: EventType.SESSION_FAILED,
                SessionStatus.INTERRUPTED: EventType.SESSION_INTERRUPTED,
            }
            original_epochs: dict[str, int] = {}
            for status in expected_types:
                session_id = f"terminal-evidence-{status.value}"
                await store.create(
                    RunRequest(
                        agent_name="removed_agent",
                        session_id=session_id,
                        messages=[Message.text("user", "finish")],
                        tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
                    ),
                    identity=profiled_session_identity(
                        provider_name="fake",
                        model="fake-model",
                        provider=provider,
                    ),
                )
                terminal = await store.update_status(session_id, status)
                original_epochs[session_id] = terminal.run_epoch
                if status == SessionStatus.INTERRUPTED:
                    await store.checkpoint(
                        session_id,
                        {
                            "pending_session_interrupt": {
                                "reason": "worker restart",
                                "metadata": {"worker": "old"},
                                "interruption_type": "operator_requested",
                                "interruption_request_id": "interrupt-before-crash",
                            }
                        },
                    )

                repaired = await app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(session_id=session_id)
                )
                assert repaired.actions == (
                    IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE,
                )

            store = await _reopen_store(session_store_case, store)
            for status, expected_type in expected_types.items():
                session_id = f"terminal-evidence-{status.value}"
                session = await store.load(session_id)
                records = await store.query_events(
                    EventQuery(
                        session_id=session_id,
                        event_types=tuple(expected_types.values()),
                    )
                )
                assert session is not None
                assert session.status == status
                assert session.run_epoch == original_epochs[session_id] + 2
                assert [record.event.type for record in records] == [expected_type]
                assert await store.load_checkpoint(session_id) == {
                    CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION
                }

            resumed_app = CayuApp(session_store=store, enable_logging=False)
            resumed_app.register_provider(provider, default=True)
            resumed_app.register_agent(AgentSpec(name="removed_agent", model="fake-model"))
            resumed_events = [
                event
                async for event in resumed_app.resume(
                    ResumeRequest(
                        session_id="terminal-evidence-interrupted",
                        messages=[Message.text("user", "continue")],
                    )
                )
            ]
            assert resumed_events[-1].type == EventType.SESSION_COMPLETED

            async def collect_second_resume() -> list[Event]:
                return [
                    event
                    async for event in resumed_app.resume(
                        ResumeRequest(
                            session_id="terminal-evidence-interrupted",
                            messages=[Message.text("user", "continue again")],
                        )
                    )
                ]

            second_resume = asyncio.create_task(collect_second_resume())
            await asyncio.wait_for(provider.second_started.wait(), timeout=10)
            independent_interrupt = [
                event
                async for event in resumed_app.interrupt_session(
                    InterruptSessionRequest(
                        session_id="terminal-evidence-interrupted",
                        reason="new operator request",
                        metadata={"worker": "new"},
                    )
                )
            ]
            second_resume_events = await asyncio.wait_for(second_resume, timeout=5)
            assert independent_interrupt[-1].type == EventType.SESSION_INTERRUPTED
            assert second_resume_events[-1].id == independent_interrupt[-1].id

            durable_interruptions = await store.query_events(
                EventQuery(
                    session_id="terminal-evidence-interrupted",
                    event_type=EventType.SESSION_INTERRUPTED,
                )
            )
            assert len(durable_interruptions) == 2
            repaired_interruption, later_interruption = (
                record.event for record in durable_interruptions
            )
            assert repaired_interruption.payload["reason"] == "worker restart"
            assert repaired_interruption.payload["metadata"] == {"worker": "old"}
            assert (
                repaired_interruption.payload["interruption_request_id"] == "interrupt-before-crash"
            )
            assert later_interruption.payload["reason"] == "new operator request"
            assert later_interruption.payload["metadata"] == {"worker": "new"}
            assert (
                later_interruption.payload["interruption_request_id"]
                != repaired_interruption.payload["interruption_request_id"]
            )

            concurrent_session_id = "terminal-evidence-concurrent"
            await store.create(
                RunRequest(
                    agent_name="removed_agent",
                    session_id=concurrent_session_id,
                    messages=[Message.text("user", "finish")],
                ),
                identity=_identity(),
            )
            await store.update_status(concurrent_session_id, SessionStatus.COMPLETED)
            request = IncompleteSessionRecoveryRequest(session_id=concurrent_session_id)
            concurrent_results = await asyncio.gather(
                CayuApp(session_store=store, enable_logging=False).recover_incomplete_session(
                    request
                ),
                CayuApp(session_store=store, enable_logging=False).recover_incomplete_session(
                    request
                ),
            )
            assert (
                sum(
                    result.actions == (IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE,)
                    for result in concurrent_results
                )
                == 1
            )
            assert all(
                result.actions
                in {
                    (IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE,),
                    (IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,),
                    (IncompleteSessionRecoveryAction.SKIPPED_TERMINAL,),
                }
                for result in concurrent_results
            )
            concurrent_events = await store.query_events(
                EventQuery(
                    session_id=concurrent_session_id,
                    event_type=EventType.SESSION_COMPLETED,
                )
            )
            assert len(concurrent_events) == 1
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_quarantines_late_secret_tool_arguments(
    session_store_case,
) -> None:
    class ResolveLateSecretTool(Tool):
        spec = ToolSpec(
            name="resolve_late_secret",
            description="Resolve one late invocation secret.",
            input_schema={"type": "object", "additionalProperties": True},
        )

        async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
            assert args["provided"] == secret
            assert ctx.vault is not None
            await ctx.vault.resolve(SecretRef(name="api_key"))
            return ToolResult(content="done")

    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            provider = FakeProvider(
                [
                    [
                        ModelStreamEvent.tool_call(
                            id="call_late_secret",
                            name="resolve_late_secret",
                            arguments={"provided": secret, secret: {"nested": secret}},
                        ),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ],
                    [
                        ModelStreamEvent.text_delta("done"),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ],
                ]
            )
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_environment(
                Environment(
                    EnvironmentSpec(name="local"),
                    vault=StaticVault({"api_key": secret}),
                ),
                default=True,
            )
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[ResolveLateSecretTool()],
            )
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="late-secret-store-conformance",
                        messages=[Message.text("user", "run")],
                    )
                )
            ]

            durable_events = await store.load_events("late-secret-store-conformance")
            started = next(
                event for event in durable_events if event.type is EventType.TOOL_CALL_STARTED
            )
            assert started.payload["arguments_state"] == "quarantined"
            assert "arguments" not in started.payload
            assert secret not in repr(events)
            assert secret not in repr(await store.load_transcript("late-secret-store-conformance"))
            assert secret not in repr(provider.requests[1].messages)
        finally:
            await _close_store(store)

    secret = "late-store-tool-start-secret-canary"
    asyncio.run(run())


def test_session_store_conformance_deletes_reconciled_consecutive_interruptions(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = "terminal-evidence-consecutive-interrupt-delete"
        first_payload = {
            "interruption_type": "tool_approval_required",
            "approval": {"approval_id": "approval-before-operator"},
        }
        second_payload = {
            "reason": "new operator request",
            "metadata": {"worker": "new"},
            "interruption_type": "operator_requested",
            "interruption_request_id": "independent-interrupt",
        }
        try:
            await store.create(
                RunRequest(
                    agent_name="removed_agent",
                    session_id=session_id,
                    messages=[Message.text("user", "finish")],
                ),
                identity=_identity(),
            )
            await store.update_status(session_id, SessionStatus.INTERRUPTED)
            await store.append_event(
                session_id,
                Event(
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=session_id,
                    payload=first_payload,
                ),
            )

            await store.transition_status_and_checkpoint(
                session_id,
                from_statuses={SessionStatus.INTERRUPTED},
                to_status=SessionStatus.INTERRUPTING,
                checkpoint_transform=lambda _session, _checkpoint: {
                    "pending_session_interrupt": second_payload
                },
            )
            await store.update_status(session_id, SessionStatus.INTERRUPTED)
            app = CayuApp(session_store=store, enable_logging=False)
            repaired = await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id)
            )
            assert repaired.actions == (IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE,)
            assert repaired.events[0].payload == {
                **second_payload,
                "interruption_request_id": PRIVATE_EVENT_AUTHORITY,
            }

            durable_interruptions = await store.query_events(
                EventQuery(
                    session_id=session_id,
                    event_type=EventType.SESSION_INTERRUPTED,
                )
            )
            assert durable_interruptions[-1].event.payload == second_payload

            store = await _reopen_store(session_store_case, store)
            settled = await CayuApp(
                session_store=store,
                enable_logging=False,
            ).recover_incomplete_session(IncompleteSessionRecoveryRequest(session_id=session_id))
            assert settled.actions == (IncompleteSessionRecoveryAction.SKIPPED_TERMINAL,)
            await store.delete_session(session_id)
            assert await store.load(session_id) is None
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_repairs_pre_boundary_resume_failure(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = "terminal-evidence-pre-boundary-resume"
        operation_id = "93f3184b-976c-4fdc-8889-c2918890581b"
        try:
            await store.create(
                RunRequest(
                    agent_name="removed_agent",
                    session_id=session_id,
                    messages=[Message.text("user", "finish")],
                ),
                identity=_identity(),
            )
            await store.update_status(session_id, SessionStatus.COMPLETED)
            await store.append_event(
                session_id,
                Event(
                    type=EventType.SESSION_COMPLETED,
                    session_id=session_id,
                ),
            )
            await store.transition_status_and_checkpoint(
                session_id,
                from_statuses={SessionStatus.COMPLETED},
                to_status=SessionStatus.RUNNING,
                checkpoint_transform=lambda current_session, checkpoint: (
                    _checkpoint_with_session_run_operation(
                        checkpoint=checkpoint,
                        current_session=current_session,
                        operation_id=operation_id,
                    )
                ),
            )
            await store.update_status(session_id, SessionStatus.FAILED)
            await store.release_run_fence(session_id)

            store = await _reopen_store(session_store_case, store)
            repaired = await CayuApp(
                session_store=store,
                enable_logging=False,
            ).recover_incomplete_session(IncompleteSessionRecoveryRequest(session_id=session_id))
            records = await store.query_events(
                EventQuery(
                    session_id=session_id,
                    event_types=(
                        EventType.SESSION_COMPLETED,
                        EventType.SESSION_FAILED,
                    ),
                )
            )
            assert repaired.actions == (IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE,)
            assert [record.event.type for record in records] == [
                EventType.SESSION_COMPLETED,
                EventType.SESSION_FAILED,
            ]
            assert records[-1].event.payload["session_run_operation_id"] == operation_id
            assert await store.load_checkpoint(session_id) == {
                CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION
            }
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize(
    "lifecycle_event_type",
    (
        EventType.SESSION_STARTED,
        EventType.SESSION_RESUMED,
        EventType.SESSION_FORKED,
    ),
)
def test_session_store_conformance_repairs_terminal_older_than_current_lifecycle(
    session_store_case,
    lifecycle_event_type: EventType,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        lifecycle_label = lifecycle_event_type.value.replace(".", "-")
        session_id = f"terminal-evidence-pre-boundary-{lifecycle_label}"
        operation_id = "93f3184b-976c-4fdc-8889-c2918890581b"
        try:
            await store.create(
                RunRequest(
                    agent_name="removed_agent",
                    session_id=session_id,
                    messages=[Message.text("user", "finish")],
                ),
                identity=_identity(),
            )
            await store.update_status(session_id, SessionStatus.COMPLETED)
            await store.append_events(
                session_id,
                [
                    Event(
                        id=f"terminal-evidence-{lifecycle_label}-older-other-operation",
                        type=EventType.SESSION_COMPLETED,
                        session_id=session_id,
                        payload={"session_run_operation_id": "older-operation"},
                    ),
                    Event(
                        id=f"terminal-evidence-{lifecycle_label}-older-matching-operation",
                        type=EventType.SESSION_COMPLETED,
                        session_id=session_id,
                        payload={"session_run_operation_id": operation_id},
                    ),
                ],
            )
            await store.transition_status_and_checkpoint(
                session_id,
                from_statuses={SessionStatus.COMPLETED},
                to_status=SessionStatus.RUNNING,
                checkpoint_transform=lambda current_session, checkpoint: (
                    _checkpoint_with_session_run_operation(
                        checkpoint=checkpoint,
                        current_session=current_session,
                        operation_id=operation_id,
                    )
                ),
            )
            await store.append_event(
                session_id,
                Event(
                    id=f"terminal-evidence-newest-{lifecycle_label}-boundary",
                    type=lifecycle_event_type,
                    session_id=session_id,
                ),
            )
            await store.update_status(session_id, SessionStatus.COMPLETED)

            store = await _reopen_store(session_store_case, store)
            with pytest.raises(TerminalSessionEvidenceError) as missing:
                await store.load_terminal_session_evidence(session_id)
            assert missing.value.code is TerminalSessionEvidenceErrorCode.TERMINAL_EVENT_MISSING
            await store.release_run_fence(session_id)

            repaired = await CayuApp(
                session_store=store,
                enable_logging=False,
            ).recover_incomplete_session(IncompleteSessionRecoveryRequest(session_id=session_id))
            records = await store.query_events(
                EventQuery(
                    session_id=session_id,
                    event_type=EventType.SESSION_COMPLETED,
                )
            )
            assert repaired.actions == (IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE,)
            assert [record.event.type for record in records] == [
                EventType.SESSION_COMPLETED,
                EventType.SESSION_COMPLETED,
                EventType.SESSION_COMPLETED,
            ]
            assert [record.event.id for record in records[:2]] == [
                f"terminal-evidence-{lifecycle_label}-older-other-operation",
                f"terminal-evidence-{lifecycle_label}-older-matching-operation",
            ]
            assert records[-1].event.payload["terminal_evidence_repaired"] is True
            assert records[-1].event.payload["session_run_operation_id"] == operation_id
            assert await store.load_checkpoint(session_id) == {
                CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION
            }

            store = await _reopen_store(session_store_case, store)
            settled = await CayuApp(
                session_store=store,
                enable_logging=False,
            ).recover_incomplete_session(IncompleteSessionRecoveryRequest(session_id=session_id))
            settled_records = await store.query_events(
                EventQuery(
                    session_id=session_id,
                    event_type=EventType.SESSION_COMPLETED,
                )
            )
            assert settled.actions == (IncompleteSessionRecoveryAction.SKIPPED_TERMINAL,)
            assert settled_records == records
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_recovers_abandoned_resumed_run(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = "terminal-evidence-abandoned-resumed-run"
        operation_id = "terminal-evidence-abandoned-resumed-operation"
        try:
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "finish")],
                ),
                identity=_identity(),
            )
            await store.update_status(session_id, SessionStatus.COMPLETED)
            await store.append_event(
                session_id,
                Event(
                    type=EventType.SESSION_COMPLETED,
                    session_id=session_id,
                ),
            )
            await store.transition_status_and_checkpoint(
                session_id,
                from_statuses={SessionStatus.COMPLETED},
                to_status=SessionStatus.RUNNING,
                checkpoint_transform=lambda current_session, checkpoint: (
                    _checkpoint_with_session_run_operation(
                        checkpoint=checkpoint,
                        current_session=current_session,
                        operation_id=operation_id,
                    )
                ),
            )
            _deactivate_session_run_fence(session_id)

            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(_UnusedForkProvider(), default=True)
            app.register_agent(AgentSpec(name="assistant", model="fake-model"))
            recovered = await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id)
            )

            store = await _reopen_store(session_store_case, store)
            session = await store.load(session_id)
            checkpoint = await store.load_checkpoint(session_id)
            interrupted = await store.query_events(
                EventQuery(
                    session_id=session_id,
                    event_type=EventType.SESSION_INTERRUPTED,
                )
            )
            assert recovered.actions == (IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED,)
            assert session is not None
            assert session.status == SessionStatus.INTERRUPTED
            assert len(interrupted) == 1
            assert interrupted[0].event.payload["session_run_operation_id"] == operation_id
            assert checkpoint is not None
            assert "session_run_operation" not in checkpoint
            assert "pending_session_interrupt" not in checkpoint
            assert "incomplete_session_recovery_claim" not in checkpoint
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_provider_progress_event_and_cursor_are_atomic(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = "sess_provider_progress_conformance"
        interaction_id = "interaction-provider-progress-conformance"
        try:
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[],
                ),
                identity=_identity(),
                interaction_started_event=Event(
                    id="provider-progress-conformance-interaction-started",
                    type=EventType.INTERACTION_STARTED,
                    session_id=session_id,
                    interaction_id=interaction_id,
                ),
                interaction_source_messages=[],
            )
            await store.checkpoint(created.id, {})
            identity = ModelAttemptIdentity(
                model_step_id="mstep_" + "a" * 32,
                model_attempt_id="matt_" + "b" * 32,
            )
            stage_result = await store.prepare_model_completion_stage(
                created.id,
                request=ModelCompletionStageRequest(
                    stage_id=f"{identity.model_step_id}:dispatch:0",
                    logical_step_id=identity.model_step_id,
                    dispatch_ordinal=0,
                    intent={
                        **identity.payload(),
                        "provider_name": "conformance-provider",
                        "requested_model": "conformance-model",
                    },
                ),
                expected_statuses={created.status},
                expected_run_epoch=created.run_epoch,
                expected_transcript_cursor=0,
            )
            state = ProviderOperationState(
                operation_id="provider-operation-conformance",
                stream_protocol="conformance-v1",
                recovery_metadata={"cursor": 0, "opaque": {"page": "start"}},
            )
            await store.append_events(
                created.id,
                [
                    Event(
                        type=EventType.MODEL_STARTED,
                        session_id=created.id,
                        interaction_id=interaction_id,
                        agent_name="assistant",
                        payload={
                            "provider": "conformance-provider",
                            "model": "conformance-model",
                            "step": 1,
                            "attempt": 1,
                            "max_attempts": 1,
                            **identity.payload(),
                        },
                    ),
                    Event(
                        type=EventType.PROVIDER_OPERATION_STARTED,
                        session_id=created.id,
                        interaction_id=interaction_id,
                        agent_name="assistant",
                        payload={
                            "provider": "conformance-provider",
                            "model": "conformance-model",
                            "step": 1,
                            "attempt": 1,
                            "max_attempts": 1,
                            **identity.payload(),
                            "source_run_epoch": created.run_epoch,
                            "start_id": f"provider-operation:{identity.model_attempt_id}",
                            "state_version": state.version,
                            "operation_id": state.operation_id,
                            "stream_protocol": state.stream_protocol,
                            "status": ProviderOperationStatus.IN_PROGRESS.value,
                            "recovery_metadata": state.recovery_metadata.model_dump(mode="json"),
                        },
                    ),
                ],
            )

            async def commit(delta: str, cursor: int, current: ProviderOperationState):
                stream_event = ModelStreamEvent.text_delta(
                    delta,
                    recovery_metadata={
                        "cursor": cursor,
                        "opaque": {"page": f"after-{cursor}"},
                    },
                )
                event = Event(
                    id=provider_operation_progress_event_id(
                        stage_result.stage.stage_id,
                        cursor,
                    ),
                    type=EventType.MODEL_TEXT_DELTA,
                    session_id=created.id,
                    interaction_id=interaction_id,
                    agent_name="assistant",
                    payload={
                        "delta": delta,
                        "step": 1,
                        "attempt": 1,
                        "max_attempts": 1,
                        **identity.payload(),
                        "provider_operation_progress": provider_operation_progress_payload(
                            current,
                            stream_event,
                        ),
                    },
                )
                return await commit_provider_operation_progress(
                    store,
                    stage=stage_result.stage,
                    model_attempt_identity=identity,
                    current_state=current,
                    stream_event=stream_event,
                    event=event,
                    expected_run_epoch=created.run_epoch,
                )

            first = await commit("hel", 1, state)
            replay = await commit("hel", 1, state)
            second = await commit("lo", 2, first.state)

            assert first.replayed is False
            assert replay.replayed is True
            assert second.replayed is False
            assert second.state.recovery_metadata.cursor == 2
            text_events = [
                event
                for event in await store.load_events(created.id)
                if event.type == EventType.MODEL_TEXT_DELTA
            ]
            assert [event.payload["delta"] for event in text_events] == ["hel", "lo"]
            recovered = await load_recoverable_provider_operation(
                store,
                stage_result.stage,
            )
            assert recovered is not None
            assert recovered.state.recovery_metadata.cursor == 2
            assert recovered.state.recovery_metadata.opaque == {"page": "after-2"}
            assert [event.delta for event in recovered.accepted_stream_events] == [
                "hel",
                "lo",
            ]
        finally:
            await store.release_run_fence(session_id)
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_reconstructs_provider_resolution_after_restart(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = "sess_provider_resolution_conformance"
        interaction_id = "interaction-provider-resolution-conformance"
        try:
            assert store._supports_atomic_model_completion_stage_release_protocol()
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[],
                ),
                identity=_identity(),
                interaction_started_event=Event(
                    id="provider-resolution-conformance-interaction-started",
                    type=EventType.INTERACTION_STARTED,
                    session_id=session_id,
                    interaction_id=interaction_id,
                ),
                interaction_source_messages=[],
            )
            await store.checkpoint(created.id, {})
            identity = ModelAttemptIdentity(
                model_step_id="mstep_" + "c" * 32,
                model_attempt_id="matt_" + "d" * 32,
            )
            stage = (
                await store.prepare_model_completion_stage(
                    created.id,
                    request=ModelCompletionStageRequest(
                        stage_id=f"{identity.model_step_id}:dispatch:0",
                        logical_step_id=identity.model_step_id,
                        dispatch_ordinal=0,
                        intent={
                            **identity.payload(),
                            "provider_name": "conformance-provider",
                            "requested_model": "conformance-model",
                            "recovery_context": {
                                "execution_profile_fingerprint": "e" * 64,
                            },
                        },
                    ),
                    expected_statuses={created.status},
                    expected_run_epoch=created.run_epoch,
                    expected_transcript_cursor=0,
                )
            ).stage
            state = ProviderOperationState(
                operation_id="provider-resolution-operation",
                stream_protocol="conformance-v1",
            )
            common_payload = {
                "provider": "conformance-provider",
                "model": "conformance-model",
                "step": 1,
                "attempt": 1,
                "max_attempts": 1,
                **identity.payload(),
            }
            await store.append_events(
                created.id,
                [
                    Event(
                        type=EventType.MODEL_STARTED,
                        session_id=created.id,
                        interaction_id=interaction_id,
                        agent_name="assistant",
                        payload=common_payload,
                    ),
                    Event(
                        type=EventType.PROVIDER_OPERATION_STARTING,
                        session_id=created.id,
                        interaction_id=interaction_id,
                        agent_name="assistant",
                        payload={
                            **common_payload,
                            "source_run_epoch": created.run_epoch,
                            "start_id": f"provider-operation:{identity.model_attempt_id}",
                            "start_idempotency_support": "unsupported",
                        },
                    ),
                    Event(
                        type=EventType.PROVIDER_OPERATION_STARTED,
                        session_id=created.id,
                        interaction_id=interaction_id,
                        agent_name="assistant",
                        payload={
                            **common_payload,
                            "source_run_epoch": created.run_epoch,
                            "start_id": f"provider-operation:{identity.model_attempt_id}",
                            "state_version": state.version,
                            "operation_id": state.operation_id,
                            "stream_protocol": state.stream_protocol,
                            "status": ProviderOperationStatus.UNAVAILABLE.value,
                            "recovery_metadata": {},
                        },
                    ),
                    event_with_runtime_payload_authority(
                        Event(
                            type=EventType.PROVIDER_OPERATION_RECOVERY_REQUIRED,
                            session_id=created.id,
                            interaction_id=interaction_id,
                            agent_name="assistant",
                            payload={
                                **common_payload,
                                "source_run_epoch": created.run_epoch,
                                "run_epoch": created.run_epoch,
                                "operation_id": state.operation_id,
                                "stream_protocol": state.stream_protocol,
                                "status": "unavailable",
                                "recovery_reason": "unavailable",
                            },
                        ),
                        "model_attempt_id",
                        "model_step_id",
                        "operation_id",
                        "stream_protocol",
                    ),
                ],
            )
            interrupted = await store.transition_status(
                created.id,
                from_statuses={created.status},
                to_status=SessionStatus.INTERRUPTED,
            )
            request = ProviderOperationResolutionRequest(
                session_id=created.id,
                stage_id=stage.stage_id,
                expected_run_epoch=interrupted.run_epoch,
                action=ProviderOperationResolutionAction.FAIL,
                reason="conformance operator failure",
            )
            resolved = await resolve_provider_operation_stage(
                store,
                request,
                redactor=SecretRedactor(),
            )
            assert resolved.replayed is False
            assert await store.load_active_model_completion_stage(created.id) is None
            pending = await load_pending_provider_operation_disposition(store, created.id)
            assert pending is not None
            assert pending[0].resolution_id == resolved.record.resolution_id

            store = await _reopen_store(session_store_case, store)
            reconstructed = await load_provider_operation_resolution(
                store,
                created.id,
                stage.stage_id,
            )
            assert reconstructed is not None
            assert reconstructed.replayed is True
            assert reconstructed.record.action is ProviderOperationResolutionAction.FAIL
            reconstructed_pending = await load_pending_provider_operation_disposition(
                store,
                created.id,
            )
            assert reconstructed_pending is not None
            assert reconstructed_pending[0] == pending[0]
            inspection = await inspect_provider_operation(store, created.id)
            assert (
                inspection.status
                is ProviderOperationInspectionStatus.PROVIDER_OPERATION_UNAVAILABLE
            )
            assert inspection.resolution_action is ProviderOperationResolutionAction.FAIL
            replay = await resolve_provider_operation_stage(
                store,
                request,
                redactor=SecretRedactor(),
            )
            assert replay.replayed is True
        finally:
            await store.release_run_fence(session_id)
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_provider_resolution_loses_to_terminal_completion(
    session_store_case,
    monkeypatch,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        session_id = "sess_provider_resolution_terminal_race"
        interaction_id = "interaction-provider-resolution-terminal-race"
        release_resolution = asyncio.Event()
        resolution_entered = asyncio.Event()
        resolution_task: asyncio.Task | None = None
        try:
            created = await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[],
                ),
                identity=_identity(),
                interaction_started_event=Event(
                    id="provider-resolution-terminal-race-interaction-started",
                    type=EventType.INTERACTION_STARTED,
                    session_id=session_id,
                    interaction_id=interaction_id,
                ),
                interaction_source_messages=[],
            )
            await store.checkpoint(created.id, {})
            identity = ModelAttemptIdentity(
                model_step_id="mstep_" + "4" * 32,
                model_attempt_id="matt_" + "5" * 32,
            )
            intent = {
                **identity.payload(),
                "provider_name": "conformance-provider",
                "requested_model": "conformance-model",
                "recovery_context": {
                    "execution_profile_fingerprint": "6" * 64,
                },
            }
            stage = (
                await store.prepare_model_completion_stage(
                    created.id,
                    request=ModelCompletionStageRequest(
                        stage_id=f"{identity.model_step_id}:dispatch:0",
                        logical_step_id=identity.model_step_id,
                        dispatch_ordinal=0,
                        intent=intent,
                    ),
                    expected_statuses={created.status},
                    expected_run_epoch=created.run_epoch,
                    expected_transcript_cursor=0,
                )
            ).stage
            common_payload = {
                "provider": "conformance-provider",
                "model": "conformance-model",
                "step": 1,
                "attempt": 1,
                "max_attempts": 1,
                **identity.payload(),
            }
            await store.append_events(
                created.id,
                [
                    Event(
                        type=EventType.MODEL_STARTED,
                        session_id=created.id,
                        interaction_id=interaction_id,
                        agent_name="assistant",
                        payload=common_payload,
                    ),
                    Event(
                        type=EventType.PROVIDER_OPERATION_STARTED,
                        session_id=created.id,
                        interaction_id=interaction_id,
                        agent_name="assistant",
                        payload={
                            **common_payload,
                            "source_run_epoch": created.run_epoch,
                            "start_id": f"provider-operation:{identity.model_attempt_id}",
                            "state_version": 1,
                            "operation_id": "provider-resolution-terminal-race-operation",
                            "stream_protocol": "conformance-v1",
                            "status": ProviderOperationStatus.UNAVAILABLE.value,
                            "recovery_metadata": {},
                        },
                    ),
                    event_with_runtime_payload_authority(
                        Event(
                            type=EventType.PROVIDER_OPERATION_RECOVERY_REQUIRED,
                            session_id=created.id,
                            interaction_id=interaction_id,
                            agent_name="assistant",
                            payload={
                                **common_payload,
                                "source_run_epoch": created.run_epoch,
                                "run_epoch": created.run_epoch,
                                "operation_id": "provider-resolution-terminal-race-operation",
                                "stream_protocol": "conformance-v1",
                                "status": "unavailable",
                                "recovery_reason": "unavailable",
                            },
                        ),
                        "model_attempt_id",
                        "model_step_id",
                        "operation_id",
                        "stream_protocol",
                    ),
                ],
            )
            interrupted = await store.transition_status(
                created.id,
                from_statuses={created.status},
                to_status=SessionStatus.INTERRUPTED,
            )
            request = ProviderOperationResolutionRequest(
                session_id=created.id,
                stage_id=stage.stage_id,
                expected_run_epoch=interrupted.run_epoch,
                action=ProviderOperationResolutionAction.FAIL,
                reason="terminal completion must win the resolution race",
            )
            original_publish_session_operation = store.publish_session_operation

            async def publish_after_terminal_completion(*args, **kwargs):
                resolution_entered.set()
                await release_resolution.wait()
                return await original_publish_session_operation(*args, **kwargs)

            monkeypatch.setattr(
                store,
                "publish_session_operation",
                publish_after_terminal_completion,
            )
            resolution_task = asyncio.create_task(
                resolve_provider_operation_stage(
                    store,
                    request,
                    redactor=SecretRedactor(),
                )
            )
            await asyncio.wait_for(resolution_entered.wait(), timeout=10)

            completion = await store.complete_model_completion_stage(
                created.id,
                stage_id=stage.stage_id,
                publication=_assistant_model_completion_publication(
                    session_id=created.id,
                    stage_id=stage.stage_id,
                    logical_step_id=stage.logical_step_id,
                    intent=intent,
                    completion_event_id="provider-resolution-terminal-race-completed",
                    source_transcript_cursor=0,
                    assistant_message=Message.text("assistant", "terminal completion won"),
                    event_payload=identity.payload(),
                ),
            )
            assert completion.stage.state == "completed"
            release_resolution.set()

            with pytest.raises(
                SessionModelCompletionStageConflict,
                match="became terminal before external disposition",
            ):
                await resolution_task
            assert (
                await load_provider_operation_resolution(
                    store,
                    created.id,
                    stage.stage_id,
                )
                is None
            )
            assert (
                await load_pending_provider_operation_disposition(
                    store,
                    created.id,
                )
                is None
            )
            active = await store.load_active_model_completion_stage(created.id)
            assert active is not None
            assert active.stage.state == "completed"
            assert not any(
                event.type is EventType.PROVIDER_OPERATION_RESOLVED
                for event in await store.load_events(created.id)
            )
        finally:
            release_resolution.set()
            if resolution_task is not None and not resolution_task.done():
                resolution_task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await resolution_task
            await store.release_run_fence(session_id)
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("signal_order", ["status-first", "cancel-first", "caller-cancel"])
def test_session_store_conformance_interrupt_does_not_cancel_consumer(
    session_store_case,
    monkeypatch,
    signal_order,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(ScriptedModelProvider([]), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "interrupt-consumer-order"
        control = app._session_control
        boundary = asyncio.Event()
        signal_sent = asyncio.Event()
        original_check = control.raise_if_interrupted
        original_cancel = control.cancel_active_runs
        original_handle = app._session_finalization.handle_session_interrupted
        observed = []
        consumer = None

        async def hold_status_check(sid):
            boundary.set()
            await signal_sent.wait()
            await original_check(sid)

        def signal(sid):
            assert control._interrupt_targets(sid)
            if signal_order == "cancel-first":
                original_cancel(sid)
            signal_sent.set()
            return True

        async def publish(**kwargs):
            assert consumer is not None
            if signal_order != "cancel-first":
                assert control.is_emitting_interrupted(session_id, task=consumer)
                assert consumer.cancelling() == 0
                assert original_cancel(session_id)
                if signal_order == "caller-cancel":
                    consumer.cancel("caller cancellation")
            async for event in original_handle(**kwargs):
                yield event

        monkeypatch.setattr(control, "raise_if_interrupted", hold_status_check)
        monkeypatch.setattr(control, "cancel_active_runs", signal)
        monkeypatch.setattr(app._session_finalization, "handle_session_interrupted", publish)

        async def collect():
            async for event in app.run(
                RunRequest(
                    session_id=session_id,
                    agent_name="assistant",
                    messages=[Message.text("user", "hello")],
                )
            ):
                observed.append(event)

        consumer = asyncio.create_task(collect())
        try:
            await asyncio.wait_for(boundary.wait(), 10)
            interrupt_events = await asyncio.wait_for(
                _collect_events(
                    app.interrupt_session(
                        InterruptSessionRequest(session_id=session_id, reason="pause")
                    )
                ),
                10,
            )
            if signal_order == "caller-cancel":
                with pytest.raises(asyncio.CancelledError, match="caller cancellation"):
                    await consumer
                assert consumer.cancelling() == 1
            else:
                await consumer
                assert consumer.cancelling() == 0
            interrupted = [e for e in observed if e.type == EventType.SESSION_INTERRUPTED]
            assert len(interrupted) == 1
            assert [e.id for e in interrupt_events if e.type == EventType.SESSION_INTERRUPTED] == [
                interrupted[0].id
            ]
            assert (await store.load(session_id)).status is SessionStatus.INTERRUPTED
            assert not any(e.type == EventType.SESSION_FAILED for e in observed)
        finally:
            signal_sent.set()
            if not consumer.done():
                consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)
            await app.aclose()
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize(
    "signal_order",
    [
        "before-refusal",
        "after-refusal",
        "during-probe-load",
        "during-probe-checkpoint",
        "during-cleanup",
    ],
)
@pytest.mark.parametrize("with_recall", [False, True])
def test_session_store_conformance_interrupt_wins_model_preparation(
    session_store_case,
    monkeypatch,
    signal_order,
    with_recall,
) -> None:
    import cayu.runtime._model_step_executor as model_step_module

    if signal_order == "during-cleanup" and not with_recall:
        pytest.skip("Context-exposure cleanup requires automatic recall")

    async def run() -> None:
        store = await _open_store(session_store_case)
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            request_footprint=RequestFootprintConfig(
                fingerprint_key_id="automatic-recall-recovery",
                fingerprint_key="automatic-recall-recovery-key-material",
            ),
        )
        provider = ScriptedModelProvider([])
        app.register_provider(provider, default=True)
        if with_recall:
            knowledge = _AutomaticRecallRecoveryKnowledgeStore()
            await knowledge.create_entry(
                KnowledgeEntry(
                    id="automatic-recall-recovery",
                    text="Atlas recovery evidence says Friday.",
                )
            )
            app.register_environment(
                Environment(
                    _approval_recovery_environment_spec(),
                    knowledge_store=knowledge,
                ),
                default=True,
            )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=_automatic_recall_recovery_policy() if with_recall else None,
        )
        session_id = "interrupt-model-preparation"
        ready = asyncio.Event()
        signal_sent = asyncio.Event()
        original_prepare = store.prepare_model_completion_stage
        original_cancel = app._session_control.cancel_active_runs
        original_load_exposure = store.load_context_exposure
        runtime_store = app._model_step_executor._session_store
        original_load = runtime_store.load
        original_checkpoint = runtime_store.load_checkpoint
        original_handle = app._session_finalization.handle_session_interrupted
        refused = False
        cancellation_injected = False
        probe_task = None
        original_probe = model_step_module._model_preparation_lost_to_interrupt

        async def probe(*args, **kwargs):
            nonlocal probe_task
            probe_task = asyncio.current_task()
            try:
                return await original_probe(*args, **kwargs)
            finally:
                probe_task = None

        async def prepare(*args, **kwargs):
            nonlocal refused
            ready.set()
            await signal_sent.wait()
            assert (await store.load(session_id)).status is SessionStatus.INTERRUPTING
            try:
                return await original_prepare(*args, **kwargs)
            except (SessionStatusConflict, SessionRunFenced):
                refused = True
                raise

        def signal(sid):
            if signal_order == "before-refusal":
                original_cancel(sid)
            signal_sent.set()
            return True

        async def load(*args, **kwargs):
            nonlocal cancellation_injected
            if (
                asyncio.current_task() is probe_task
                and signal_order == "during-probe-load"
                and not cancellation_injected
            ):
                cancellation_injected = True
                assert original_cancel(session_id)
                await asyncio.sleep(0)
            return await original_load(*args, **kwargs)

        async def load_checkpoint(*args, **kwargs):
            nonlocal cancellation_injected
            if (
                asyncio.current_task() is probe_task
                and signal_order == "during-probe-checkpoint"
                and not cancellation_injected
            ):
                cancellation_injected = True
                assert original_cancel(session_id)
                await asyncio.sleep(0)
            return await original_checkpoint(*args, **kwargs)

        async def load_exposure(*args, **kwargs):
            nonlocal cancellation_injected
            if refused and signal_order == "during-cleanup" and not cancellation_injected:
                cancellation_injected = True
                assert original_cancel(session_id)
                await asyncio.sleep(0)
            return await original_load_exposure(*args, **kwargs)

        async def publish(**kwargs):
            if signal_order == "after-refusal":
                assert refused
                assert original_cancel(session_id)
            async for event in original_handle(**kwargs):
                yield event

        monkeypatch.setattr(model_step_module, "_model_preparation_lost_to_interrupt", probe)
        monkeypatch.setattr(store, "prepare_model_completion_stage", prepare)
        monkeypatch.setattr(store, "load_context_exposure", load_exposure)
        monkeypatch.setattr(runtime_store, "load", load)
        monkeypatch.setattr(runtime_store, "load_checkpoint", load_checkpoint)
        monkeypatch.setattr(app._session_control, "cancel_active_runs", signal)
        monkeypatch.setattr(app._session_finalization, "handle_session_interrupted", publish)
        consumer = asyncio.create_task(
            _collect_events(
                app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "When is Atlas released?")],
                    )
                )
            )
        )
        interrupter = None
        try:
            try:
                await asyncio.wait_for(ready.wait(), 10)
            except TimeoutError:
                if consumer.done():
                    await consumer
                raise
            interrupter = asyncio.create_task(
                _collect_events(
                    app.interrupt_session(
                        InterruptSessionRequest(session_id=session_id, reason="pause")
                    )
                )
            )
            observed = await asyncio.wait_for(asyncio.shield(consumer), 10)
            interrupted_events = await asyncio.wait_for(interrupter, 10)
            terminal = [e for e in observed if e.type == EventType.SESSION_INTERRUPTED]
            assert len(terminal) == 1
            assert [
                e.id for e in interrupted_events if e.type == EventType.SESSION_INTERRUPTED
            ] == [terminal[0].id]
            assert (await store.load(session_id)).status is SessionStatus.INTERRUPTED
            assert consumer.cancelling() == 0
            assert not provider.requests
            records = await store.query_events(EventQuery(session_id=session_id))
            assert not any(
                r.event.type in {EventType.SESSION_FAILED, EventType.INTERACTION_FAILED}
                for r in records
            )
            if signal_order.startswith("during-"):
                assert cancellation_injected
            if with_recall:
                exposures = (
                    await store.list_context_exposures(RecallEvidenceQuery(session_id=session_id))
                ).items
                assert len(exposures) == 1
                assert exposures[0].state is ContextExposureState.CANCELLED
        finally:
            signal_sent.set()
            tasks = [consumer] + ([] if interrupter is None else [interrupter])
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await app.aclose()
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("termination_fails", [False, True])
def test_session_store_conformance_cancel_during_post_preparation_cleanup(
    session_store_case,
    monkeypatch,
    termination_fails,
) -> None:
    import cayu.runtime._model_step_executor as model_step_module
    from cayu.runtime._run_limits import (
        BudgetReservationLeaseLost,
        BudgetReservationLeaseLostBeforeModelDispatch,
    )

    async def run() -> None:
        store = await _open_store(session_store_case)
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            request_footprint=RequestFootprintConfig(
                fingerprint_key_id="automatic-recall-recovery",
                fingerprint_key="automatic-recall-recovery-key-material",
            ),
        )
        provider = ScriptedModelProvider([])
        app.register_provider(provider, default=True)
        knowledge = _AutomaticRecallRecoveryKnowledgeStore()
        await knowledge.create_entry(
            KnowledgeEntry(
                id="automatic-recall-recovery",
                text="Atlas recovery evidence says Friday.",
            )
        )
        app.register_environment(
            Environment(
                _approval_recovery_environment_spec(),
                knowledge_store=knowledge,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=_automatic_recall_recovery_policy(),
        )
        session_id = "cancel-post-preparation-cleanup"
        cleanup_started = asyncio.Event()
        release_cleanup = asyncio.Event()
        original_load_exposure = store.load_context_exposure
        original_finish = model_step_module._raise_after_model_pre_dispatch_cleanup
        original_failure = BudgetReservationLeaseLost("lease lost before dispatch")
        prepared_stage = None
        cleanup_observed = False
        snapshots = []

        async def fail_dispatch(_session_id, *, stage, **kwargs):
            nonlocal prepared_stage
            prepared_stage = stage
            raise original_failure

        async def load_exposure(*args, **kwargs):
            nonlocal cleanup_observed
            if prepared_stage is not None and not cleanup_observed:
                cleanup_observed = True
                cleanup_started.set()
                await release_cleanup.wait()
                if termination_fails:
                    raise RuntimeError("exposure store unavailable")
            return await original_load_exposure(*args, **kwargs)

        async def observe_finish(cleanup, **kwargs):
            try:
                await original_finish(cleanup, **kwargs)
            except asyncio.CancelledError as cancellation:
                # Observe the outcome before the session finalizer can repair
                # the retained stage, using real durable storage reads.
                exposures = (
                    await store.list_context_exposures(RecallEvidenceQuery(session_id=session_id))
                ).items
                stage = await store.load_model_completion_stage(
                    session_id,
                    prepared_stage.stage_id,
                )
                snapshots.append((cancellation, exposures, stage))
                raise

        monkeypatch.setattr(store, "mark_model_completion_stage_dispatched", fail_dispatch)
        monkeypatch.setattr(store, "load_context_exposure", load_exposure)
        monkeypatch.setattr(
            model_step_module, "_raise_after_model_pre_dispatch_cleanup", observe_finish
        )
        consumer = asyncio.create_task(
            _collect_events(
                app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "When is Atlas released?")],
                    )
                )
            )
        )
        try:
            await asyncio.wait_for(cleanup_started.wait(), 10)
            consumer.cancel("caller cancellation during exposure termination")
            await asyncio.sleep(0)
            assert not consumer.done()
            release_cleanup.set()
            await asyncio.wait_for(asyncio.gather(consumer, return_exceptions=True), 10)
            assert len(snapshots) == 1
            cancellation, exposures, stage = snapshots[0]
            assert cancellation.args == ("caller cancellation during exposure termination",)
            assert isinstance(cancellation.__cause__, BudgetReservationLeaseLostBeforeModelDispatch)
            assert cancellation.__cause__.__cause__ is original_failure
            assert len(exposures) == 1
            assert not provider.requests
            if termination_fails:
                assert exposures[0].state is ContextExposureState.DISPATCH_STARTED
                assert stage is not None
                assert any(
                    "termination is not durable" in note for note in original_failure.__notes__
                )
            else:
                assert exposures[0].state is ContextExposureState.FAILED
                assert stage is None
        finally:
            release_cleanup.set()
            if not consumer.done():
                consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)
            await app.aclose()
            await _close_store(store)

    asyncio.run(run())
