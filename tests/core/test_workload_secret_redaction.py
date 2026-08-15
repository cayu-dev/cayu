from __future__ import annotations

import asyncio
import contextlib
import copy
import sys
from datetime import UTC, datetime
from typing import Any, cast

import pytest
from tests.core._execution_profile_fixtures import profiled_session_identity
from tests.core._execution_unit_fixtures import tool_round_identity
from tests.core._workload_secret_support import (
    FakeProvider,
    collect_events,
    collect_fork_events,
    collect_resume_events,
)
from tests.provider_traceback_assertions import is_cayu_source_filename

import cayu.runtime._invocation_secrets as invocation_secrets_module
import cayu.runtime.execution_profiles as execution_profiles_module
import cayu.runtime.sessions as sessions_module
from cayu import (
    InMemoryKnowledgeStore,
    KnowledgeIndexer,
    KnowledgeIndexRequest,
    ListKnowledgeTool,
    LocalRunner,
    LocalWorkspace,
    ReadFileTool,
    SearchKnowledgeTool,
)
from cayu.core import (
    AgentSpec,
    Event,
    EventType,
    Message,
    MessageRole,
    ToolCallPart,
)
from cayu.core.messages import FilePart, ProviderStatePart
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.environments import (
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    EnvironmentSpec,
)
from cayu.providers import ModelStreamEvent
from cayu.runners import (
    ExecCommand,
    ExecResult,
    Runner,
    RunnerCancelledError,
    RunnerExecutionError,
    attach_cancellation_artifacts,
)
from cayu.runtime import (
    CayuApp,
    ForkSessionRequest,
    InMemorySessionStore,
    InterruptSessionRequest,
    ResumeRequest,
    RunRequest,
    Session,
    SessionIdentity,
    SessionStatus,
    StructuredOutputSpec,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
)
from cayu.runtime.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
)
from cayu.vaults import REDACTED_SECRET, SecretRedactor, SecretRef, StaticVault


def checkpoint_without_active_invocation_profile(
    checkpoint: dict[str, Any] | None,
) -> dict[str, Any]:
    assert checkpoint is not None
    assert (
        execution_profiles_module.active_invocation_execution_profile_from_checkpoint(checkpoint)
        is not None
    )
    copied = dict(checkpoint)
    copied.pop(execution_profiles_module.ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY)
    return copied


def _assert_cayu_traceback_does_not_retain_text(error: BaseException, text: str) -> None:
    traceback = error.__traceback__
    while traceback is not None:
        if is_cayu_source_filename(traceback.tb_frame.f_code.co_filename):
            retained = {
                name: type(value).__name__
                for name, value in traceback.tb_frame.f_locals.items()
                if text in repr(value)
            }
            assert retained == {}, (
                traceback.tb_frame.f_code.co_filename,
                traceback.tb_frame.f_code.co_name,
                retained,
            )
        traceback = traceback.tb_next


@pytest.mark.parametrize(
    "secret_checkpoint",
    [
        {"custom": {"value": "context-checkpoint-secret-canary"}},
        {"context-checkpoint-secret-canary": {"value": "safe"}},
    ],
)
def test_runtime_managed_context_rejects_secret_checkpoint_before_publication(
    secret_checkpoint: dict[str, Any],
) -> None:
    from cayu.runtime.context import (
        ContextBuildResult,
        ContextRequest,
        RuntimeManagedContextPolicy,
    )

    secret = "context-checkpoint-secret-canary"

    class SecretCheckpointPolicy(RuntimeManagedContextPolicy):
        def __init__(self) -> None:
            self.result = None

        async def build_with_checkpoint(
            self,
            request: ContextRequest,
            *,
            checkpoint: dict[str, Any] | None,
        ) -> ContextBuildResult:
            del checkpoint
            self.result = ContextBuildResult(
                messages=[Message.text("assistant", f"unsafe result {secret}")],
                checkpoint=secret_checkpoint,
                checkpoint_event_payload={
                    "checkpoint": "custom",
                    "custom_detail": secret,
                },
            )
            return self.result

    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("must not dispatch"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    store = InMemorySessionStore()
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    policy = SecretCheckpointPolicy()
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=policy,
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="secret_context_checkpoint",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert provider.requests == []
    assert checkpoint_without_active_invocation_profile(
        asyncio.run(store.load_checkpoint("secret_context_checkpoint"))
    ) == {
        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
    }
    assert all(event.type is not EventType.SESSION_CHECKPOINTED for event in events)
    assert secret not in repr(events)
    assert events[-1].type is EventType.SESSION_FAILED
    assert policy.result is not None
    assert policy.result.messages == []
    assert policy.result.checkpoint == {}
    assert policy.result.checkpoint_event_payload == {}


@pytest.mark.parametrize("secret_location", ["key", "value", "nested_typed_key"])
def test_runtime_managed_context_rejects_secret_checkpoint_event_payload_before_publication(
    secret_location: str,
) -> None:
    from cayu.runtime.context import (
        ContextBuildResult,
        ContextRequest,
        RuntimeManagedContextPolicy,
    )

    secret = (
        "checkpoint"
        if secret_location == "nested_typed_key"
        else "context-event-payload-secret-canary"
    )

    class SecretEventPayloadPolicy(RuntimeManagedContextPolicy):
        def __init__(self) -> None:
            self.result: ContextBuildResult | None = None

        async def build_with_checkpoint(
            self,
            request: ContextRequest,
            *,
            checkpoint: dict[str, Any] | None,
        ) -> ContextBuildResult:
            del checkpoint
            if secret_location == "nested_typed_key":
                unsafe_payload = {"custom": {"checkpoint": "safe"}}
            else:
                unsafe_field = secret if secret_location == "key" else "custom_detail"
                unsafe_value = "safe" if secret_location == "key" else secret
                unsafe_payload = {unsafe_field: unsafe_value}
            self.result = ContextBuildResult(
                messages=[Message.text("assistant", f"unsafe result {secret}")],
                checkpoint={"custom": {"value": "safe"}},
                checkpoint_event_payload={
                    "checkpoint": "custom",
                    **unsafe_payload,
                },
            )
            return self.result

    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("must not dispatch"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    store = InMemorySessionStore()
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    policy = SecretEventPayloadPolicy()
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=policy,
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=f"secret_context_event_{secret_location}",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert provider.requests == []
    assert checkpoint_without_active_invocation_profile(
        asyncio.run(store.load_checkpoint(f"secret_context_event_{secret_location}"))
    ) == {CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION}
    assert all(event.type is not EventType.SESSION_CHECKPOINTED for event in events)
    assert secret not in repr(events)
    assert events[-1].type is EventType.SESSION_FAILED
    assert policy.result is not None
    assert policy.result.messages == []
    assert policy.result.checkpoint == {}
    assert policy.result.checkpoint_event_payload == {}


def test_runtime_managed_context_discards_secret_checkpoint_carried_by_failure() -> None:
    from cayu.runtime.context import (
        ContextBuildError,
        ContextRequest,
        RuntimeManagedContextPolicy,
    )

    secret = "context-failure-checkpoint-secret-canary"

    class FailingCheckpointPolicy(RuntimeManagedContextPolicy):
        async def build_with_checkpoint(
            self,
            request: ContextRequest,
            *,
            checkpoint: dict[str, Any] | None,
        ):
            del request, checkpoint
            raise ContextBuildError(
                "context build failed",
                compaction_telemetry=[],
                checkpoint={"custom": {"value": secret}},
                checkpoint_event_payload={"checkpoint": "custom"},
                cause=RuntimeError("context policy failed"),
            )

    provider = FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})])
    store = InMemorySessionStore()
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=FailingCheckpointPolicy(),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="secret_context_failure_checkpoint",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert provider.requests == []
    assert checkpoint_without_active_invocation_profile(
        asyncio.run(store.load_checkpoint("secret_context_failure_checkpoint"))
    ) == {
        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
    }
    assert all(event.type is not EventType.SESSION_CHECKPOINTED for event in events)
    assert secret not in repr(events)
    assert events[-1].type is EventType.SESSION_FAILED


def test_context_failure_checkpoint_event_preserves_typed_keys_for_short_secret() -> None:
    from cayu.runtime.context import (
        ContextBuildError,
        sanitize_context_build_error_checkpoint,
    )

    error = ContextBuildError(
        "context build failed",
        compaction_telemetry=[],
        checkpoint={"context_compaction": {"summary": "safe"}},
        checkpoint_event_payload={
            "checkpoint": "context_compaction",
            "compacted_transcript_cursor": 2,
        },
        cause=RuntimeError("safe policy failure"),
    )

    sanitize_context_build_error_checkpoint(
        error,
        redactor=SecretRedactor("point"),
    )

    assert error.checkpoint == {"context_compaction": {"summary": "safe"}}
    assert error.checkpoint_event_payload == {
        "checkpoint": "context_compaction",
        "compacted_transcript_cursor": 2,
    }


def test_environment_factory_child_cancellation_is_redacted_ordinary_failure() -> None:
    secret = "environment-child-cancellation-secret-canary"

    class CancellingFactory(EnvironmentFactory):
        async def create(
            self,
            request: EnvironmentFactoryRequest,
        ) -> EnvironmentFactoryResult:
            del request
            raise asyncio.CancelledError(f"factory exposed {secret}")

    store = InMemorySessionStore()
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(
        FakeProvider(
            [
                ModelStreamEvent.text_delta("must not dispatch"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ),
        default=True,
    )
    app.register_environment_factory(
        EnvironmentSpec(name="dynamic"),
        CancellingFactory(),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="environment_child_cancellation",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert events[-1].type is EventType.SESSION_FAILED
    assert secret not in repr(events)
    assert "cancelled without caller cancellation" in repr(events)


def test_environment_factory_child_cancellation_ignores_historical_task_cancellation() -> None:
    secret = "environment-historical-cancellation-secret-canary"

    class CancellingFactory(EnvironmentFactory):
        async def create(
            self,
            request: EnvironmentFactoryRequest,
        ) -> EnvironmentFactoryResult:
            del request
            raise asyncio.CancelledError(f"later child cancellation {secret}")

    app = CayuApp(
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(
        FakeProvider(
            [
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ),
        default=True,
    )
    app.register_environment_factory(
        EnvironmentSpec(name="dynamic"),
        CancellingFactory(),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def scenario() -> tuple[list[Event], int]:
        current_task = asyncio.current_task()
        assert current_task is not None
        current_task.cancel("historical cancellation already handled")
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.sleep(0)
        assert current_task.cancelling() == 1
        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="environment_historical_cancellation",
                messages=[Message.text("user", "hello")],
            ),
        )
        return events, current_task.cancelling()

    events, cancelling = asyncio.run(scenario())

    assert cancelling == 1
    assert events[-1].type is EventType.SESSION_FAILED
    assert secret not in repr(events)
    assert "cancelled without caller cancellation" in repr(events)


def test_environment_factory_nested_child_cancellation_is_redacted_failure() -> None:
    secret = "environment-group-cancellation-secret-canary"

    class GroupFailingFactory(EnvironmentFactory):
        async def create(
            self,
            request: EnvironmentFactoryRequest,
        ) -> EnvironmentFactoryResult:
            del request
            raise BaseExceptionGroup(
                "factory group",
                [
                    asyncio.CancelledError(f"group child exposed {secret}"),
                    RuntimeError(f"factory allocation failed with {secret}"),
                ],
            )

    store = InMemorySessionStore()
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(
        FakeProvider(
            [
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ),
        default=True,
    )
    app.register_environment_factory(
        EnvironmentSpec(name="dynamic"),
        GroupFailingFactory(),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def scenario() -> tuple[BaseExceptionGroup, list[Event]]:
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="environment_group_cancellation",
                    messages=[Message.text("user", "hello")],
                ),
            )
        return exc_info.value, await store.load_events("environment_group_cancellation")

    failure, events = asyncio.run(scenario())

    rendered = repr((failure, events))
    assert secret not in rendered
    assert events[-1].type is EventType.SESSION_INTERRUPTED
    assert isinstance(failure.exceptions[0], asyncio.CancelledError)
    assert "factory group" in rendered
    assert failure.__cause__ is None
    assert failure.__context__ is None


def test_environment_failure_after_real_cancellation_is_detached_and_redacted() -> None:
    secret = "environment-post-cancellation-failure-secret-canary"

    class CancellationSuppressingFactory(EnvironmentFactory):
        def __init__(self) -> None:
            self.started: asyncio.Event | None = None

        async def create(
            self,
            request: EnvironmentFactoryRequest,
        ) -> EnvironmentFactoryResult:
            del request
            assert self.started is not None
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise RuntimeError(f"factory failed after cancellation with {secret}") from None

    factory = CancellationSuppressingFactory()
    app = CayuApp(
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(
        FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]),
        default=True,
    )
    app.register_environment_factory(
        EnvironmentSpec(name="dynamic"),
        factory,
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def scenario() -> tuple[BaseExceptionGroup, int, bool]:
        factory.started = asyncio.Event()
        run_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="environment_post_cancel_failure",
                    messages=[Message.text("user", "hello")],
                ),
            )
        )
        await factory.started.wait()
        run_task.cancel(f"operator cancellation with {secret}")
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await run_task
        return exc_info.value, run_task.cancelling(), run_task.cancelled()

    failure, cancelling, cancelled = asyncio.run(scenario())

    assert secret not in repr(failure)
    assert REDACTED_SECRET in repr(failure)
    assert isinstance(failure.exceptions[0], RuntimeError)
    assert isinstance(failure.exceptions[1], asyncio.CancelledError)
    assert failure.__cause__ is None
    assert failure.__context__ is None
    # The session finalizer consumes the task's cancellation request after
    # preserving it as an ordered leaf beside the extension failure.
    assert cancelling == 0
    assert cancelled is False
    _assert_cayu_traceback_does_not_retain_text(failure, secret)


def test_os_timeout_after_real_cancellation_does_not_retain_secret_slots() -> None:
    from cayu.runtime._binding_cleanup import (
        binding_cleanup_status,
        record_binding_cleanup_failure,
    )
    from cayu.runtime._environment_operation_boundary import (
        await_environment_operation as _await_environment_operation,
    )

    secret = "environment-timeout-slot-secret-canary"
    started = asyncio.Event()
    original: TimeoutError | None = None

    async def extension_operation() -> None:
        nonlocal original
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            original = TimeoutError(110, secret, f"/tmp/{secret}")

            async def retry() -> None:
                return None

            record_binding_cleanup_failure(
                original,
                RuntimeError(f"cleanup exposed {secret}"),
                retry=retry,
            )
            raise original from None

    async def scenario() -> tuple[BaseExceptionGroup, int, bool]:
        task = asyncio.create_task(
            _await_environment_operation(
                extension_operation,
                operation_name="Environment extension",
                redactor=SecretRedactor(secret),
            )
        )
        await started.wait()
        task.cancel(f"operator cancellation with {secret}")
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await task
        return exc_info.value, task.cancelling(), task.cancelled()

    failure, cancelling, cancelled = asyncio.run(scenario())

    assert original is not None
    timeout = failure.exceptions[0]
    assert type(timeout) is TimeoutError
    assert timeout is not original
    assert timeout.filename is None
    assert timeout.strerror is None
    assert secret not in str(timeout)
    assert secret not in repr(failure)
    assert REDACTED_SECRET in repr(failure)
    cleanup_status = binding_cleanup_status(timeout)
    assert cleanup_status is not None
    assert secret not in repr(cleanup_status)
    assert isinstance(failure.exceptions[1], asyncio.CancelledError)
    assert cancelling == 1
    assert cancelled is False
    _assert_cayu_traceback_does_not_retain_text(failure, secret)


def test_environment_operation_preserves_factory_cleanup_owner_after_cancellation() -> None:
    from cayu.environments.factory import (
        attach_environment_factory_cleanup_settlement_task,
        environment_factory_cleanup_settlement_task,
    )
    from cayu.runtime._environment_operation_boundary import (
        await_environment_operation as _await_environment_operation,
    )

    started = asyncio.Event()
    allow_cleanup = asyncio.Event()

    async def extension_operation() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:

            async def settle() -> None:
                await allow_cleanup.wait()

            settlement_task = asyncio.create_task(settle())
            error = RuntimeError("factory cleanup is still running")
            attach_environment_factory_cleanup_settlement_task(
                error,
                settlement_task,
            )
            raise error from None

    async def scenario() -> tuple[BaseExceptionGroup, asyncio.Task[None]]:
        task = asyncio.create_task(
            _await_environment_operation(
                extension_operation,
                operation_name="Environment factory creation",
                redactor=SecretRedactor(),
            )
        )
        await started.wait()
        task.cancel("caller stopped waiting")
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await task
        settlement_task = environment_factory_cleanup_settlement_task(exc_info.value)
        assert settlement_task is not None
        assert not settlement_task.done()
        allow_cleanup.set()
        await settlement_task
        return exc_info.value, settlement_task

    failure, settlement_task = asyncio.run(scenario())

    assert isinstance(failure.exceptions[0], RuntimeError)
    assert isinstance(failure.exceptions[1], asyncio.CancelledError)
    assert settlement_task.done()
    assert not settlement_task.cancelled()


def test_environment_operation_combines_group_leaf_cleanup_owners() -> None:
    from cayu.environments.factory import (
        attach_environment_factory_cleanup_settlement_task,
        environment_factory_cleanup_settlement_task,
    )
    from cayu.runtime._environment_operation_boundary import (
        await_environment_operation as _await_environment_operation,
    )

    started = asyncio.Event()
    allow_first_cleanup = asyncio.Event()
    allow_second_cleanup = asyncio.Event()
    first_cleanup_finished = asyncio.Event()
    second_cleanup_finished = asyncio.Event()

    async def extension_operation() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as cancellation:

            async def settle_first() -> None:
                await allow_first_cleanup.wait()
                first_cleanup_finished.set()

            async def settle_second() -> None:
                await allow_second_cleanup.wait()
                second_cleanup_finished.set()

            first_error = RuntimeError("first factory cleanup is still running")
            attach_environment_factory_cleanup_settlement_task(
                first_error,
                asyncio.create_task(settle_first()),
            )
            second_error = RuntimeError("second factory cleanup is still running")
            attach_environment_factory_cleanup_settlement_task(
                second_error,
                asyncio.create_task(settle_second()),
            )
            raise BaseExceptionGroup(
                "factory cleanup leaves",
                [
                    first_error,
                    BaseExceptionGroup(
                        "nested factory cleanup leaf",
                        [second_error, cancellation],
                    ),
                ],
            ) from None

    async def scenario() -> BaseExceptionGroup:
        task = asyncio.create_task(
            _await_environment_operation(
                extension_operation,
                operation_name="Environment factory creation",
                redactor=SecretRedactor(),
            )
        )
        await started.wait()
        task.cancel("caller stopped grouped factory creation")
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await task
        assert task.cancelling() == 1
        assert not task.cancelled()
        settlement_task = environment_factory_cleanup_settlement_task(exc_info.value)
        assert settlement_task is not None
        assert not settlement_task.done()

        allow_first_cleanup.set()
        await first_cleanup_finished.wait()
        await asyncio.sleep(0)
        assert not settlement_task.done()

        allow_second_cleanup.set()
        await settlement_task
        assert second_cleanup_finished.is_set()
        return exc_info.value

    failure = asyncio.run(scenario())
    assert environment_factory_cleanup_settlement_task(failure) is not None


def test_environment_operation_preserves_grouped_cleanup_retry_owners() -> None:
    from cayu.environments import factory as factory_module
    from cayu.environments.factory import (
        attach_environment_factory_cleanup_settlement_task,
        environment_factory_cleanup_settlement_task,
        register_environment_factory_cleanup_retry,
        retry_environment_factory_cleanup_settlement_task,
    )
    from cayu.runtime._environment_operation_boundary import (
        await_environment_operation as _await_environment_operation,
    )

    secret = "grouped-cleanup-retry-secret-canary"
    allow_cleanup = False
    cleanup_calls = [0, 0]
    cleanup_tasks: list[asyncio.Task[None]] = []

    def cleanup_attempt(index: int) -> asyncio.Task[None]:
        async def cleanup() -> None:
            cleanup_calls[index] += 1
            if not allow_cleanup:
                raise PermissionError(f"{secret}: cleanup {index} denied")

        task = asyncio.create_task(cleanup())
        cleanup_tasks.append(task)
        return task

    async def extension_operation() -> None:
        failures: list[BaseException] = []
        for index in range(2):
            task = cleanup_attempt(index)
            register_environment_factory_cleanup_retry(
                task,
                lambda index=index: cleanup_attempt(index),
            )
            error = RuntimeError(f"{secret}: cleanup {index} retained")
            attach_environment_factory_cleanup_settlement_task(error, task)
            failures.append(error)
        raise BaseExceptionGroup(
            f"{secret}: grouped cleanup failure",
            [*failures, asyncio.CancelledError("extension child cancelled")],
        )

    async def scenario() -> None:
        nonlocal allow_cleanup
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await _await_environment_operation(
                extension_operation,
                operation_name="Environment factory creation",
                redactor=SecretRedactor([secret]),
            )
        assert secret not in repr(exc_info.value)
        settlement_task = environment_factory_cleanup_settlement_task(exc_info.value)
        assert settlement_task is not None
        with pytest.raises(BaseExceptionGroup):
            await settlement_task

        allow_cleanup = True
        retry_task = retry_environment_factory_cleanup_settlement_task(settlement_task)
        assert retry_task is not settlement_task
        await retry_task
        await asyncio.sleep(0)

    asyncio.run(scenario())
    assert cleanup_calls == [2, 2]
    assert all(
        task not in factory_module._ENVIRONMENT_FACTORY_CLEANUP_OWNERS_BY_TASK
        for task in cleanup_tasks
    )


def test_factory_cleanup_settlement_handoff_ignores_exception_descriptors_and_forgery() -> None:
    from cayu.environments.factory import (
        attach_environment_factory_cleanup_settlement_task,
        environment_factory_cleanup_settlement_task,
    )

    attribute_name = "_cayu_environment_factory_cleanup_settlement_task"

    class DescriptorControlledError(RuntimeError):
        descriptor_reads = 0

        @property
        def _cayu_environment_factory_cleanup_settlement_task(self) -> object:
            type(self).descriptor_reads += 1
            raise RuntimeError("workload-secret-from-factory-descriptor")

    async def scenario() -> None:
        task = asyncio.create_task(asyncio.sleep(0))
        unattached = DescriptorControlledError("unattached provider failure")
        assert environment_factory_cleanup_settlement_task(unattached) is None

        forged = DescriptorControlledError("forged provider failure")
        forged.__dict__[attribute_name] = task
        assert environment_factory_cleanup_settlement_task(forged) is None

        attached = DescriptorControlledError("attached provider failure")
        attach_environment_factory_cleanup_settlement_task(attached, task)
        assert environment_factory_cleanup_settlement_task(attached) is task
        await task

    asyncio.run(scenario())
    assert DescriptorControlledError.descriptor_reads == 0


def test_suppressed_cancellation_result_is_not_retained_by_safe_cancellation() -> None:
    from cayu.runtime._environment_operation_boundary import (
        await_environment_operation as _await_environment_operation,
    )

    secret = "environment-suppressed-cancellation-result-secret-canary"
    started = asyncio.Event()

    async def extension_operation() -> dict[str, str]:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return {"credential": secret}

    async def scenario() -> tuple[asyncio.CancelledError, int, bool]:
        task = asyncio.create_task(
            _await_environment_operation(
                extension_operation,
                operation_name="Environment extension",
                redactor=SecretRedactor(secret),
            )
        )
        await started.wait()
        task.cancel(f"operator cancellation with {secret}")
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        return exc_info.value, task.cancelling(), task.cancelled()

    failure, cancelling, cancelled = asyncio.run(scenario())

    assert secret not in repr(failure)
    assert str(failure) == "Environment operation cancelled"
    assert cancelling == 1
    assert cancelled is True
    _assert_cayu_traceback_does_not_retain_text(failure, secret)


def test_detached_builtin_environment_failure_remains_bounded_after_redaction() -> None:
    from cayu.runtime._environment_lifecycle import FAILURE_DIAGNOSTIC_TEXT_MAX_BYTES
    from cayu.runtime._environment_operation_boundary import (
        await_environment_operation as _await_environment_operation,
    )

    secret = "x"
    started = asyncio.Event()

    async def extension_operation() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise RuntimeError(secret * 10_000) from None

    async def scenario() -> BaseExceptionGroup:
        task = asyncio.create_task(
            _await_environment_operation(
                extension_operation,
                operation_name="Environment extension",
                redactor=SecretRedactor(secret),
            )
        )
        await started.wait()
        task.cancel("operator cancellation")
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await task
        return exc_info.value

    failure = asyncio.run(scenario())

    detached_failure = failure.exceptions[0]
    assert type(detached_failure) is RuntimeError
    assert secret not in str(detached_failure)
    assert len(str(detached_failure).encode("utf-8")) <= FAILURE_DIAGNOSTIC_TEXT_MAX_BYTES
    assert detached_failure.args == (str(detached_failure),)
    assert isinstance(failure.exceptions[1], asyncio.CancelledError)


@pytest.mark.parametrize(
    "fatal_type",
    [KeyboardInterrupt, SystemExit, GeneratorExit, BaseException],
)
def test_scalar_fatal_environment_failure_after_real_cancellation_is_detached(
    fatal_type: type[BaseException],
) -> None:
    from cayu.runtime._environment_operation_boundary import (
        await_environment_operation as _await_environment_operation,
    )

    secret = f"environment-cancelled-{fatal_type.__name__}-secret-canary"
    started = asyncio.Event()
    original: BaseException | None = None

    async def extension_operation() -> None:
        nonlocal original
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            original = fatal_type(f"extension failed after cancellation with {secret}")
            raise original from None

    async def scenario() -> tuple[BaseExceptionGroup, int, bool]:
        task = asyncio.create_task(
            _await_environment_operation(
                extension_operation,
                operation_name="Environment extension",
                redactor=SecretRedactor(secret),
            )
        )
        await started.wait()
        task.cancel(f"operator cancellation with {secret}")
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await task
        return exc_info.value, task.cancelling(), task.cancelled()

    failure, cancelling, cancelled = asyncio.run(scenario())

    assert original is not None
    assert failure.exceptions[0] is not original
    assert isinstance(failure.exceptions[0], fatal_type)
    assert isinstance(failure.exceptions[1], asyncio.CancelledError)
    assert secret not in repr(failure)
    assert REDACTED_SECRET in repr(failure)
    assert cancelling == 1
    assert cancelled is False
    _assert_cayu_traceback_does_not_retain_text(failure, secret)


def test_scalar_fatal_environment_failure_without_cancellation_is_unchanged() -> None:
    from cayu.runtime._environment_operation_boundary import (
        await_environment_operation as _await_environment_operation,
    )

    original = SystemExit("ordinary fatal signal")

    async def extension_operation() -> None:
        raise original

    async def scenario() -> None:
        with pytest.raises(SystemExit) as exc_info:
            await _await_environment_operation(
                extension_operation,
                operation_name="Environment extension",
                redactor=SecretRedactor("ordinary-fatal-secret-canary"),
            )
        assert exc_info.value is original

    asyncio.run(scenario())


def test_binding_cleanup_handoff_survives_real_cancellation_without_retaining_secrets() -> None:
    from cayu.environments import BoundWorkspace, WorkspaceBinding
    from cayu.runtime._binding_cleanup import binding_cleanup_status, record_binding_cleanup_failure

    secret = "cancelled-binding-cleanup-secret-canary"

    class CancellationSuppressingBinding(WorkspaceBinding):
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.retry_calls = 0

        async def bind(self, workspace, runner, **_kwargs) -> BoundWorkspace:
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                bind_error = ExceptionGroup(
                    f"bind failed after cancellation with {secret}",
                    [RuntimeError(secret)],
                )
                cleanup_error = RuntimeError(f"rollback failed with {secret}")

                async def retry() -> None:
                    self.retry_calls += 1

                record_binding_cleanup_failure(
                    bind_error,
                    cleanup_error,
                    retry=retry,
                )
                raise bind_error from None

        async def finalize(self, bound, *, outcome=None, metadata=None):
            raise AssertionError("finalize should not run")

    binding = CancellationSuppressingBinding()
    app = CayuApp(
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(
        FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]),
        default=True,
    )
    app.register_environment(
        Environment(
            EnvironmentSpec(name="cancelled-binding"),
            binding=binding,
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def scenario() -> BaseExceptionGroup:
        run_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="cancelled_binding_cleanup_handoff",
                    messages=[Message.text("user", "hello")],
                ),
            )
        )
        await binding.started.wait()
        run_task.cancel(f"operator cancellation with {secret}")
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await run_task
        return exc_info.value

    failure = asyncio.run(scenario())

    assert binding.retry_calls == 1
    cleanup_status = binding_cleanup_status(failure)
    assert cleanup_status is not None
    assert cleanup_status.retry_attempted is True
    assert cleanup_status.retry_error is None
    assert secret not in repr(failure)
    assert secret not in repr(cleanup_status)
    assert cleanup_status.retry.__name__ == "_completed_environment_operation"
    _assert_cayu_traceback_does_not_retain_text(failure, secret)


def test_factory_backed_binding_cancellation_does_not_retain_raw_failure() -> None:
    from cayu.environments import BoundWorkspace, WorkspaceBinding

    secret = "factory-binding-attempt-secret-canary"

    class CancellationSuppressingBinding(WorkspaceBinding):
        def __init__(self) -> None:
            self.started = asyncio.Event()

        async def bind(self, workspace, runner, **_kwargs) -> BoundWorkspace:
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise ExceptionGroup(
                    f"factory binding failed after cancellation with {secret}",
                    [RuntimeError(secret)],
                ) from None

        async def finalize(self, bound, *, outcome=None, metadata=None):
            raise AssertionError("finalize should not run")

    binding = CancellationSuppressingBinding()

    class BindingFactory(EnvironmentFactory):
        async def create(
            self,
            request: EnvironmentFactoryRequest,
        ) -> EnvironmentFactoryResult:
            return EnvironmentFactoryResult(
                Environment(
                    EnvironmentSpec(name=request.environment_name),
                    binding=binding,
                )
            )

    app = CayuApp(
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(
        FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]),
        default=True,
    )
    app.register_environment_factory(
        EnvironmentSpec(name="dynamic"),
        BindingFactory(),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def scenario() -> BaseExceptionGroup:
        run_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="factory_binding_attempt_traceback",
                    messages=[Message.text("user", "hello")],
                ),
            )
        )
        await binding.started.wait()
        run_task.cancel(f"operator cancellation with {secret}")
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await run_task
        return exc_info.value

    failure = asyncio.run(scenario())

    assert secret not in repr(failure)
    assert REDACTED_SECRET in repr(failure)
    _assert_cayu_traceback_does_not_retain_text(failure, secret)


def test_factory_backed_sync_bind_cancellation_hides_resource_keys_from_traceback(
    tmp_path,
) -> None:
    from cayu.environments import SyncBinding

    secret = "sync-binding-resource-key-secret-canary"
    source_started = asyncio.Event()

    class SecretKeyWorkspace(LocalWorkspace):
        def __init__(self, *args, resource_key, **kwargs):  # type: ignore[no-untyped-def]
            super().__init__(*args, **kwargs)
            self._secret_resource_key = resource_key

        @property
        def resource_key(self) -> tuple[object, ...]:
            return self._secret_resource_key

    class BlockingSource(SecretKeyWorkspace):
        async def list(self, pattern="**/*", *, limit=None):  # type: ignore[no-untyped-def]
            del pattern, limit
            source_started.set()
            await asyncio.Event().wait()

    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    source = BlockingSource(
        source_root,
        workspace_id="source",
        resource_key=("source", secret),
    )
    target = SecretKeyWorkspace(
        target_root,
        workspace_id="target",
        resource_key=("target", secret),
    )

    class BindingFactory(EnvironmentFactory):
        async def create(
            self,
            request: EnvironmentFactoryRequest,
        ) -> EnvironmentFactoryResult:
            async def release(_action) -> None:  # type: ignore[no-untyped-def]
                return None

            return EnvironmentFactoryResult(
                Environment(
                    EnvironmentSpec(name=request.environment_name),
                    workspace=source,
                    binding=SyncBinding(target_workspace=target),
                ),
                release=release,
            )

    app = CayuApp(
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(
        FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]),
        default=True,
    )
    app.register_environment_factory(
        EnvironmentSpec(name="dynamic"),
        BindingFactory(),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def scenario() -> tuple[asyncio.CancelledError, int, bool]:
        run_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="factory_sync_resource_key_traceback",
                    messages=[Message.text("user", "hello")],
                ),
            )
        )
        await source_started.wait()
        run_task.cancel("operator cancellation")
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await run_task
        return exc_info.value, run_task.cancelling(), run_task.cancelled()

    cancellation, cancelling, cancelled = asyncio.run(scenario())

    assert cancelling == 1
    assert cancelled is True
    assert secret not in repr(cancellation)
    _assert_cayu_traceback_does_not_retain_text(cancellation, secret)


def test_scalar_cancellation_preserves_binding_cleanup_handoff_and_retry() -> None:
    from cayu.environments import BoundWorkspace, WorkspaceBinding
    from cayu.runtime._binding_cleanup import binding_cleanup_status, record_binding_cleanup_failure

    secret = "scalar-cancellation-cleanup-secret-canary"

    class ScalarCancellationBinding(WorkspaceBinding):
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.retry_calls = 0

        async def bind(self, workspace, runner, **_kwargs) -> BoundWorkspace:
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as cancellation:

                async def retry() -> None:
                    self.retry_calls += 1

                record_binding_cleanup_failure(
                    cancellation,
                    RuntimeError(f"rollback failed with {secret}"),
                    retry=retry,
                )
                raise

        async def finalize(self, bound, *, outcome=None, metadata=None):
            raise AssertionError("finalize should not run")

    binding = ScalarCancellationBinding()
    app = CayuApp(
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(
        FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]),
        default=True,
    )
    app.register_environment(
        Environment(
            EnvironmentSpec(name="scalar-cancelled-binding"),
            binding=binding,
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def scenario() -> tuple[asyncio.CancelledError, int, bool]:
        run_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="scalar_cancelled_binding_cleanup_handoff",
                    messages=[Message.text("user", "hello")],
                ),
            )
        )
        await binding.started.wait()
        run_task.cancel(f"operator cancellation with {secret}")
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await run_task
        return exc_info.value, run_task.cancelling(), run_task.cancelled()

    cancellation, cancelling, cancelled = asyncio.run(scenario())

    assert binding.retry_calls == 1
    cleanup_status = binding_cleanup_status(cancellation)
    assert cleanup_status is not None
    assert cleanup_status.retry_attempted is True
    assert cleanup_status.retry_error is None
    assert cleanup_status.retry.__name__ == "_completed_environment_operation"
    assert secret not in repr(cancellation)
    assert secret not in repr(cleanup_status)
    assert cancelling == 1
    assert cancelled is True
    _assert_cayu_traceback_does_not_retain_text(cancellation, secret)


def test_child_cancellation_preserves_binding_cleanup_handoff_and_retry() -> None:
    from cayu.environments import BoundWorkspace, WorkspaceBinding
    from cayu.runtime._binding_cleanup import record_binding_cleanup_failure

    secret = "child-cancellation-cleanup-secret-canary"

    class ChildCancellingBinding(WorkspaceBinding):
        def __init__(self) -> None:
            self.retry_calls = 0

        async def bind(self, workspace, runner, **_kwargs) -> BoundWorkspace:
            cancellation = asyncio.CancelledError(f"child cancellation exposed {secret}")

            async def retry() -> None:
                self.retry_calls += 1

            record_binding_cleanup_failure(
                cancellation,
                RuntimeError(f"rollback failed with {secret}"),
                retry=retry,
            )
            raise cancellation

        async def finalize(self, bound, *, outcome=None, metadata=None):
            raise AssertionError("finalize should not run")

    binding = ChildCancellingBinding()
    app = CayuApp(
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(
        FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]),
        default=True,
    )
    app.register_environment(
        Environment(
            EnvironmentSpec(name="child-cancelled-binding"),
            binding=binding,
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="child_cancelled_binding_cleanup_handoff",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert binding.retry_calls == 1
    assert events[-1].type is EventType.SESSION_FAILED
    assert secret not in repr(events)
    assert "cancelled without caller cancellation" in repr(events)


def test_scalar_binding_finalize_supplemental_redactor_survives_real_cancellation() -> None:
    from cayu.runtime._binding_cleanup import (
        BindingFinalizeFailure,
        binding_finalize_status,
        record_binding_finalize_failures,
    )
    from cayu.runtime._environment_operation_boundary import (
        await_environment_operation as _await_environment_operation,
    )

    secret = "cancelled-scalar-finalize-secret-canary"
    started = asyncio.Event()

    async def extension_operation() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation = asyncio.CancelledError(secret)
            record_binding_finalize_failures(
                cancellation,
                (
                    BindingFinalizeFailure(
                        phase="cancellation",
                        error=cancellation,
                    ),
                ),
                supplemental_redactor=SecretRedactor(secret),
            )
            raise cancellation from None

    async def scenario() -> asyncio.CancelledError:
        task = asyncio.create_task(
            _await_environment_operation(
                extension_operation,
                operation_name="Environment binding finalization",
                redactor=SecretRedactor(),
            )
        )
        await started.wait()
        task.cancel(secret)
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        assert task.cancelling() == 1
        assert task.cancelled()
        return exc_info.value

    cancellation = asyncio.run(scenario())

    finalize_status = binding_finalize_status(cancellation)
    assert finalize_status is not None
    assert finalize_status.supplemental_redactor is None
    assert secret not in repr(cancellation)
    assert REDACTED_SECRET in repr(cancellation)
    _assert_cayu_traceback_does_not_retain_text(cancellation, secret)


def test_grouped_binding_finalize_handoff_detaches_failures_after_real_cancellation() -> None:
    from cayu.runtime._binding_cleanup import (
        BindingFinalizeFailure,
        binding_finalize_status,
        record_binding_finalize_failures,
    )
    from cayu.runtime._environment_operation_boundary import (
        await_environment_operation as _await_environment_operation,
    )

    secret = "cancelled-finalize-status-secret-canary"
    started = asyncio.Event()

    async def extension_operation() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation = asyncio.CancelledError(secret)
            failure = BaseExceptionGroup(
                secret,
                [RuntimeError(secret), cancellation],
            )
            record_binding_finalize_failures(
                failure,
                (
                    BindingFinalizeFailure(
                        phase="workspace_finalize",
                        error=RuntimeError(secret),
                    ),
                ),
                supplemental_redactor=SecretRedactor(secret),
            )
            raise failure from None

    async def scenario() -> BaseExceptionGroup:
        task = asyncio.create_task(
            _await_environment_operation(
                extension_operation,
                operation_name="Environment binding finalization",
                redactor=SecretRedactor(),
            )
        )
        await started.wait()
        task.cancel(secret)
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await task
        return exc_info.value

    failure = asyncio.run(scenario())

    finalize_status = binding_finalize_status(failure)
    assert finalize_status is not None
    assert finalize_status.supplemental_redactor is None
    assert secret not in repr(failure)
    assert secret not in repr(finalize_status)
    _assert_cayu_traceback_does_not_retain_text(failure, secret)


def test_environment_factory_hostile_cancellation_accessors_cannot_bypass_redaction() -> None:
    secret = "environment-hostile-cancellation-secret-canary"

    class HostileCancellation(asyncio.CancelledError):
        @property
        def args(self):
            raise AssertionError("subclass args accessor must not run")

    class HostileFactory(EnvironmentFactory):
        async def create(
            self,
            request: EnvironmentFactoryRequest,
        ) -> EnvironmentFactoryResult:
            del request
            raise HostileCancellation(secret)

    app = CayuApp(
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(
        FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]),
        default=True,
    )
    app.register_environment_factory(
        EnvironmentSpec(name="dynamic"),
        HostileFactory(),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="environment_hostile_cancellation",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert events[-1].type is EventType.SESSION_FAILED
    assert secret not in repr(events)
    assert "cancelled without caller cancellation" in repr(events)


def test_environment_factory_real_task_cancellation_remains_cancelled_and_redacted() -> None:
    secret = "environment-caller-cancellation-secret-canary"

    class BlockingFactory(EnvironmentFactory):
        def __init__(self) -> None:
            self.started: asyncio.Event | None = None
            self.cancelled: asyncio.Event | None = None

        async def create(
            self,
            request: EnvironmentFactoryRequest,
        ) -> EnvironmentFactoryResult:
            del request
            assert self.started is not None
            assert self.cancelled is not None
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise

    factory = BlockingFactory()
    app = CayuApp(
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(
        FakeProvider(
            [
                ModelStreamEvent.text_delta("must not dispatch"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ),
        default=True,
    )
    app.register_environment_factory(
        EnvironmentSpec(name="dynamic"),
        factory,
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def scenario() -> tuple[asyncio.CancelledError, int, bool, bool]:
        factory.started = asyncio.Event()
        factory.cancelled = asyncio.Event()
        run_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="environment_caller_cancellation",
                    messages=[Message.text("user", "hello")],
                ),
            )
        )
        await factory.started.wait()
        run_task.cancel(f"operator requested stop with {secret}")
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await run_task
        return (
            exc_info.value,
            run_task.cancelling(),
            run_task.cancelled(),
            factory.cancelled.is_set(),
        )

    cancellation, cancelling, cancelled, factory_cancelled = asyncio.run(scenario())

    assert secret not in str(cancellation)
    assert REDACTED_SECRET in str(cancellation)
    assert cancellation.__cause__ is None
    assert cancellation.__context__ is None
    assert cancelling == 1
    assert cancelled is True
    assert factory_cancelled is True


def _structured_output_spec_with_secret(
    secret: str,
    secret_location: str,
) -> StructuredOutputSpec:
    if secret_location == "schema_value":
        return StructuredOutputSpec(json_schema={"type": "string", "const": secret})
    if secret_location == "schema_key":
        return StructuredOutputSpec(
            json_schema={
                "type": "object",
                "properties": {f"field-{secret}": {"type": "string"}},
            }
        )
    if secret_location == "name":
        return StructuredOutputSpec(json_schema={"type": "string"}, name=secret)
    if secret_location == "repair_prompt":
        return StructuredOutputSpec(
            json_schema={"type": "string"},
            repair_prompt=f"repair with {secret}",
        )
    raise AssertionError(f"Unsupported test secret location: {secret_location}")


@pytest.mark.parametrize(
    "secret_location",
    ["schema_value", "schema_key", "name", "repair_prompt"],
)
def test_run_rejects_secret_bearing_structured_output_before_session_creation(
    secret_location: str,
) -> None:
    from cayu.vaults import SecretRedactor

    secret = "run-structured-output-schema-secret-canary"
    session_id = "sess_run_structured_output_schema_secret"
    store = InMemorySessionStore()
    provider = FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})])
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    with pytest.raises(ValueError, match="workload secret"):
        asyncio.run(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "return structured output")],
                    structured_output=_structured_output_spec_with_secret(
                        secret,
                        secret_location,
                    ),
                ),
            )
        )

    assert asyncio.run(store.load(session_id)) is None
    assert provider.requests == []


@pytest.mark.parametrize(
    "secret_location",
    ["schema_value", "schema_key", "name", "repair_prompt"],
)
def test_resume_rejects_secret_bearing_structured_output_before_session_claim(
    secret_location: str,
) -> None:
    from cayu.vaults import SecretRedactor

    secret = "resume-structured-output-schema-secret-canary"
    session_id = "sess_resume_structured_output_schema_secret"
    store = InMemorySessionStore()
    provider = FakeProvider([])
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def scenario() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "initial turn")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.append_transcript_messages(
            session_id,
            [Message.text("assistant", "initial answer")],
        )
        await store.update_status(session_id, SessionStatus.COMPLETED)

        with pytest.raises(ValueError, match="workload secret"):
            await collect_resume_events(
                app,
                ResumeRequest(
                    session_id=session_id,
                    messages=[Message.text("user", "continue")],
                    structured_output=_structured_output_spec_with_secret(
                        secret,
                        secret_location,
                    ),
                ),
            )

    asyncio.run(scenario())
    session = asyncio.run(store.load(session_id))

    assert session is not None
    assert session.status is SessionStatus.COMPLETED
    assert session.run_epoch == 0
    assert provider.requests == []


@pytest.mark.parametrize("authority_field", ["tool_call_id", "provider"])
def test_cayu_app_rejects_every_secret_bearing_message_linkage_authority(
    authority_field: str,
) -> None:
    from cayu.vaults import SecretRedactor

    secret = f"secret-message-{authority_field}-authority-canary"
    part = (
        ToolCallPart(
            tool_call_id=secret,
            tool_name="safe_tool",
            arguments={},
        )
        if authority_field == "tool_call_id"
        else ProviderStatePart(provider=secret, state={"opaque": "safe"})
    )
    message = Message(role=MessageRole.ASSISTANT, content=(part,))
    store = InMemorySessionStore()
    provider = FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})])
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    with pytest.raises(ValueError, match="execution authority"):
        asyncio.run(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=f"sess_secret_{authority_field}_authority",
                    messages=[Message.text("user", "continue"), message],
                ),
            )
        )

    assert provider.requests == []
    assert asyncio.run(store.load(f"sess_secret_{authority_field}_authority")) is None


@pytest.mark.parametrize("authority_field", ["tool_call_id", "provider"])
def test_resume_rejects_legacy_secret_linkage_before_claiming_session(
    authority_field: str,
) -> None:
    from cayu.vaults import SecretRedactor

    secret = f"legacy-resume-{authority_field}-authority-canary"
    session_id = f"sess_legacy_resume_{authority_field}_authority"
    part = (
        ToolCallPart(
            tool_call_id=secret,
            tool_name="safe_tool",
            arguments={},
        )
        if authority_field == "tool_call_id"
        else ProviderStatePart(provider=secret, state={"opaque": "safe"})
    )
    store = InMemorySessionStore()
    provider = FakeProvider([])
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def scenario() -> BaseException:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "legacy")],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
            ),
        )
        await store.append_transcript_messages(
            session_id,
            [Message(role=MessageRole.ASSISTANT, content=(part,))],
        )
        await store.update_status(session_id, SessionStatus.COMPLETED)

        with pytest.raises(ValueError, match="cannot be resumed") as exc_info:
            await collect_resume_events(
                app,
                ResumeRequest(
                    session_id=session_id,
                    messages=[Message.text("user", "continue")],
                ),
            )
        return exc_info.value

    error = asyncio.run(scenario())
    session = asyncio.run(store.load(session_id))

    assert session is not None
    assert session.status is SessionStatus.COMPLETED
    assert session.run_epoch == 0
    assert provider.requests == []
    traceback = error.__traceback__
    while traceback is not None:
        if is_cayu_source_filename(traceback.tb_frame.f_code.co_filename):
            retained = {
                name: type(value).__name__
                for name, value in traceback.tb_frame.f_locals.items()
                if secret in repr(value)
            }
            assert retained == {}, (
                traceback.tb_frame.f_code.co_filename,
                traceback.tb_frame.f_code.co_name,
                retained,
            )
        traceback = traceback.tb_next


def test_resume_cleans_up_when_legacy_secret_linkage_arrives_after_preflight() -> None:
    from cayu.vaults import SecretRedactor

    secret = "legacy-resume-post-claim-authority-canary"
    session_id = "sess_legacy_resume_post_claim_authority"

    class PostPreflightTranscriptStore(InMemorySessionStore):
        def __init__(self) -> None:
            super().__init__()
            self.preflight_loaded = False
            self.injected = False

        async def load_transcript_snapshot(self, session_id: str):
            snapshot = await super().load_transcript_snapshot(session_id)
            self.preflight_loaded = True
            return snapshot

        async def load_transcript(self, session_id: str) -> list[Message]:
            if self.preflight_loaded and not self.injected:
                self.injected = True
                await self.append_transcript_messages(
                    session_id,
                    [
                        Message(
                            role=MessageRole.ASSISTANT,
                            content=(
                                ProviderStatePart(
                                    provider=secret,
                                    state={"opaque": "safe"},
                                ),
                            ),
                        )
                    ],
                )
            return await super().load_transcript(session_id)

    store = PostPreflightTranscriptStore()
    provider = FakeProvider([])
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def scenario() -> tuple[list[Event], Session | None, int | None]:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "legacy")],
            ),
            identity=profiled_session_identity(
                provider_name="fake",
                model="fake-model",
            ),
        )
        await store.append_transcript_messages(
            session_id,
            [Message.text("assistant", "safe history")],
        )
        await store.update_status(session_id, SessionStatus.COMPLETED)

        events = await collect_resume_events(
            app,
            ResumeRequest(
                session_id=session_id,
                messages=[Message.text("user", "continue")],
            ),
        )
        return (
            events,
            await store.load(session_id),
            sessions_module._current_session_run_epoch(session_id),
        )

    events, session, active_run_epoch = asyncio.run(scenario())

    assert session is not None
    assert session.status is SessionStatus.FAILED
    assert active_run_epoch is None
    assert provider.requests == []
    assert events[-1].type is EventType.SESSION_FAILED
    assert secret not in repr([event.model_dump(mode="json") for event in events])


def test_cayu_app_redacts_direct_vault_secrets_for_the_whole_tool_invocation() -> None:
    from cayu.vaults import REDACTED_SECRET

    secret_value = "direct-vault-invocation-secret-canary"

    class VaultLeakingTool(Tool):
        spec = ToolSpec(
            name="vault_leak",
            description="Resolve a vault secret and accidentally return it.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            assert ctx.vault is not None
            resolved = await ctx.vault.resolve(SecretRef(name="api_key"))
            raw_secret = resolved.value.get_secret_value()
            return ToolResult(
                content=f"connected with {raw_secret}",
                structured={"token": raw_secret},
                artifacts=[{"type": "debug", "token": raw_secret}],
            )

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_vault_leak",
                    name="vault_leak",
                    arguments={},
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
            vault=StaticVault({"api_key": secret_value}),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[VaultLeakingTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_direct_vault_secret_redaction",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript("sess_direct_vault_secret_redaction"))
    rendered_events = repr([event.model_dump(mode="json") for event in events])
    rendered_transcript = repr([message.model_dump(mode="json") for message in transcript])
    rendered_followup = repr(
        [message.model_dump(mode="json") for message in provider.requests[1].messages]
    )

    assert secret_value not in rendered_events
    assert secret_value not in rendered_transcript
    assert secret_value not in rendered_followup
    assert REDACTED_SECRET in rendered_events


def test_runtime_read_file_discards_pretruncated_secret_prefix_at_every_publication(
    tmp_path,
) -> None:
    secret = "workload-secret-canary-ABCDEFGHIJKLMNOP"
    session_id = "sess_read_file_pretruncated_secret"
    workspace = LocalWorkspace(tmp_path, workspace_id="local")
    asyncio.run(workspace.write_bytes("secret.txt", secret.encode()))
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_read_secret",
                    name="read_file",
                    arguments={"path": "secret.txt", "max_bytes": 16},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            workspace=workspace,
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[ReadFileTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "read the file")],
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript(session_id))
    rendered = repr(
        (
            [event.model_dump(mode="json") for event in events],
            [message.model_dump(mode="json") for message in transcript],
            [message.model_dump(mode="json") for message in provider.requests[1].messages],
        )
    )

    assert secret not in rendered
    assert secret[:16] not in rendered


@pytest.mark.parametrize(
    ("tool_name", "tool", "arguments"),
    [
        (
            "search_knowledge",
            SearchKnowledgeTool(),
            {"query": "workload", "preview_bytes": 16},
        ),
        (
            "list_knowledge",
            ListKnowledgeTool(),
            {"preview_bytes": 16},
        ),
    ],
)
@pytest.mark.parametrize("store_max_bytes", [16, 10_000])
def test_runtime_knowledge_preview_redacts_before_bound_at_every_publication(
    tool_name: str,
    tool: Tool,
    arguments: dict[str, Any],
    store_max_bytes: int,
) -> None:
    secret = "workload-secret-canary-ABCDEFGHIJKLMNOP"
    session_id = "sess_knowledge_preview_secret"
    store = InMemorySessionStore()
    knowledge_store = InMemoryKnowledgeStore()
    asyncio.run(
        KnowledgeIndexer(knowledge_store).index_text(
            KnowledgeIndexRequest(
                entry_id="secret",
                text=secret,
            )
        )
    )
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_knowledge_secret",
                    name=tool_name,
                    arguments={**arguments, "max_bytes": store_max_bytes},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            knowledge_store=knowledge_store,
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "search knowledge")],
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript(session_id))
    rendered = repr(
        (
            [event.model_dump(mode="json") for event in events],
            [message.model_dump(mode="json") for message in transcript],
            [message.model_dump(mode="json") for message in provider.requests[1].messages],
        )
    )

    assert secret not in rendered
    assert secret[:16] not in rendered


def test_custom_tool_runner_uses_secret_resolved_after_context_creation(tmp_path) -> None:
    secret = "workload-secret-canary-ABCDEFGHIJKLMNOP"
    session_id = "sess_custom_runner_dynamic_secret"

    class DynamicRunnerTool(Tool):
        spec = ToolSpec(
            name="dynamic_runner",
            description="Resolve a vault secret and execute through the documented runner handle.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del args
            assert ctx.vault is not None
            assert ctx.runner is not None
            resolved = await ctx.vault.resolve(SecretRef(name="api_key"))
            raw_secret = resolved.value.get_secret_value()
            result = await ctx.runner.exec(
                ExecCommand.process(
                    sys.executable,
                    "-c",
                    "import os,sys; sys.stdout.write(os.environ['TOKEN'])",
                ),
                env={"TOKEN": raw_secret},
                output_limit_bytes=16,
            )
            return ToolResult(
                content=result.stdout,
                structured={
                    "stdout": result.stdout,
                    "stdout_bytes": result.stdout_bytes,
                    "stdout_truncated": result.stdout_truncated,
                },
            )

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_dynamic_runner",
                    name="dynamic_runner",
                    arguments={},
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
            runner=LocalRunner(tmp_path),
            vault=StaticVault({"api_key": secret}),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[DynamicRunnerTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "run")],
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript(session_id))
    rendered = repr(
        (
            [event.model_dump(mode="json") for event in events],
            [message.model_dump(mode="json") for message in transcript],
            [message.model_dump(mode="json") for message in provider.requests[1].messages],
        )
    )

    assert secret not in rendered
    assert secret[:16] not in rendered


@pytest.mark.parametrize(
    ("output_limit_bytes", "expected_stdout", "expected_truncated"),
    [
        (128, REDACTED_SECRET, False),
        (16, "", True),
    ],
    ids=["complete-output", "truncated-output"],
)
def test_custom_tool_runner_rechecks_secret_resolved_during_real_local_dispatch(
    tmp_path,
    output_limit_bytes: int,
    expected_stdout: str,
    expected_truncated: bool,
) -> None:
    secret = "workload-secret-canary-ABCDEFGHIJKLMNOP"
    session_id = f"sess_runner_inflight_secret_{output_limit_bytes}"
    started_path = tmp_path / f"started-{output_limit_bytes}"
    release_path = tmp_path / f"release-{output_limit_bytes}"

    class ConcurrentSecretRunnerTool(Tool):
        spec = ToolSpec(
            name="concurrent_secret_runner",
            description="Resolve a secret while a real local command is running.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del args
            assert ctx.runner is not None
            assert ctx.vault is not None
            script = (
                "import os,pathlib,sys,time\n"
                f"pathlib.Path({str(started_path)!r}).write_text('started')\n"
                f"release = pathlib.Path({str(release_path)!r})\n"
                "while not release.exists():\n"
                "    time.sleep(0.01)\n"
                "sys.stdout.write(os.environ['TOKEN'])\n"
            )
            command_task = asyncio.create_task(
                ctx.runner.exec(
                    ExecCommand.process(sys.executable, "-c", script),
                    env={"TOKEN": secret},
                    output_limit_bytes=output_limit_bytes,
                )
            )
            while not started_path.exists():
                await asyncio.sleep(0)
            await ctx.vault.resolve(SecretRef(name="api_key"))
            release_path.write_text("release")
            result = await command_task
            return ToolResult(
                content=result.stdout,
                structured={
                    "stdout": result.stdout,
                    "stdout_bytes": result.stdout_bytes,
                    "stdout_truncated": result.stdout_truncated,
                },
            )

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_concurrent_secret_runner",
                    name="concurrent_secret_runner",
                    arguments={},
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
            runner=LocalRunner(tmp_path),
            vault=StaticVault({"api_key": secret}),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[ConcurrentSecretRunnerTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "run")],
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript(session_id))
    tool_event = next(event for event in events if event.type is EventType.TOOL_CALL_COMPLETED)
    result = tool_event.payload["result"]
    rendered = repr(
        (
            [event.model_dump(mode="json") for event in events],
            [message.model_dump(mode="json") for message in transcript],
            [message.model_dump(mode="json") for message in provider.requests[1].messages],
        )
    )

    assert result["structured"]["stdout"] == expected_stdout
    assert result["structured"]["stdout_truncated"] is expected_truncated
    assert secret not in rendered
    assert secret[:16] not in rendered
    if expected_stdout:
        assert REDACTED_SECRET in rendered


def test_runtime_fails_closed_when_secret_resolves_after_bounded_runner_completion(
    tmp_path,
) -> None:
    secret = "late-registered-secret-canary-ABCDEFGHIJKLMNOP"
    session_id = "sess_runner_late_secret"

    class LateSecretRunnerTool(Tool):
        spec = ToolSpec(
            name="late_secret_runner",
            description="Resolve a secret after a bounded command completes.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del args
            assert ctx.runner is not None
            assert ctx.vault is not None
            result = await ctx.runner.exec(
                ExecCommand.process(
                    sys.executable,
                    "-c",
                    "import os,sys; sys.stdout.write(os.environ['TOKEN'])",
                ),
                env={"TOKEN": secret},
                output_limit_bytes=16,
            )
            assert result.stdout_truncated is True
            await ctx.vault.resolve(SecretRef(name="api_key"))
            return ToolResult(
                content=result.stdout,
                structured={
                    "stdout": result.stdout,
                    "stdout_truncated": result.stdout_truncated,
                },
            )

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_late_secret_runner",
                    name="late_secret_runner",
                    arguments={},
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
            runner=LocalRunner(tmp_path),
            vault=StaticVault({"api_key": secret}),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[LateSecretRunnerTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "run")],
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript(session_id))
    failed = next(event for event in events if event.type is EventType.TOOL_CALL_FAILED)
    rendered = repr(
        (
            [event.model_dump(mode="json") for event in events],
            [message.model_dump(mode="json") for message in transcript],
            [message.model_dump(mode="json") for message in provider.requests[1].messages],
        )
    )

    assert failed.payload["terminal_outcome"] == "invalid_tool_output"
    assert failed.payload["result"]["is_error"] is True
    assert "late-registered-" not in rendered
    assert secret not in rendered


@pytest.mark.parametrize(
    "failure_message",
    [
        "workload-secret-canary-ABCDEFGHIJKLMNOP",
        "workload-secret-",
    ],
    ids=["complete-secret", "recoverable-prefix"],
)
def test_runtime_runner_failure_omits_opaque_diagnostic_at_every_publication(
    failure_message: str,
) -> None:
    secret = "workload-secret-canary-ABCDEFGHIJKLMNOP"
    session_id = f"sess_runner_failure_{len(failure_message)}"

    class OpaqueFailureRunner(Runner):
        isolation = "microsandbox"

        async def exec(self, command: ExecCommand, **kwargs) -> ExecResult:
            del command, kwargs
            raise RuntimeError(failure_message)

    class FailingRunnerTool(Tool):
        spec = ToolSpec(
            name="failing_runner",
            description="Exercise a runner operational failure.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del args
            assert ctx.runner is not None
            await ctx.runner.exec(ExecCommand.process("fail"))
            return ToolResult(content="unexpected")

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_failing_runner",
                    name="failing_runner",
                    arguments={},
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
            runner=OpaqueFailureRunner(),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[FailingRunnerTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "run")],
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript(session_id))
    rendered = repr(
        (
            [event.model_dump(mode="json") for event in events],
            [message.model_dump(mode="json") for message in transcript],
            [message.model_dump(mode="json") for message in provider.requests[1].messages],
        )
    )

    assert failure_message not in rendered
    assert secret[:16] not in rendered
    assert "Runner command execution failed." in rendered
    assert "'error': 'runner_execution_failed'" in rendered
    assert "'error_type': 'RuntimeError'" in rendered


@pytest.mark.parametrize(
    "cleanup_message",
    [
        "workload-secret-canary-ABCDEFGHIJKLMNOP",
        "workload-secret-",
    ],
    ids=["complete-secret", "recoverable-prefix"],
)
def test_operator_interrupt_runner_cleanup_omits_opaque_diagnostic(
    cleanup_message: str,
) -> None:
    secret = "workload-secret-canary-ABCDEFGHIJKLMNOP"
    session_id = f"sess_runner_cleanup_{len(cleanup_message)}"

    class CleanupFailureRunner(Runner):
        isolation = "microsandbox"

        def __init__(self) -> None:
            self.started: asyncio.Event | None = None

        async def exec(self, command: ExecCommand, **kwargs) -> ExecResult:
            del command, kwargs
            assert self.started is not None
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as exc:
                attach_cancellation_artifacts(
                    exc,
                    [
                        {
                            "type": "cayu.runner_cleanup.v1",
                            "adapter": "microsandbox",
                            "action": "kill_command",
                            "status": "failed",
                            "timeout_s": 5.0,
                            "error_type": "RuntimeError",
                            "error": cleanup_message,
                        }
                    ],
                )
                raise
            return ExecResult()

    class CancelledRunnerTool(Tool):
        spec = ToolSpec(
            name="cancelled_runner",
            description="Exercise runner cleanup during operator interruption.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del args
            assert ctx.runner is not None
            assert ctx.vault is not None
            await ctx.vault.resolve(SecretRef(name="api_key"))
            await ctx.runner.exec(ExecCommand.process("blocked"))
            return ToolResult(content="unexpected")

    runner = CleanupFailureRunner()
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_cancelled_runner",
                    name="cancelled_runner",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        ]
    )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            runner=runner,
            vault=StaticVault({"api_key": secret}),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[CancelledRunnerTool()],
    )

    async def scenario():
        runner.started = asyncio.Event()
        run_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "run")],
                ),
            )
        )
        await runner.started.wait()
        interrupt_events = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id=session_id,
                    reason="operator stop",
                )
            )
        ]
        run_events = await run_task
        stored_events = await store.load_events(session_id)
        transcript = await store.load_transcript(session_id)
        return interrupt_events, run_events, stored_events, transcript

    interrupt_events, run_events, stored_events, transcript = asyncio.run(scenario())
    rendered = repr(
        (
            [event.model_dump(mode="json") for event in interrupt_events],
            [event.model_dump(mode="json") for event in run_events],
            [event.model_dump(mode="json") for event in stored_events],
            [message.model_dump(mode="json") for message in transcript],
        )
    )

    assert run_events[-1].type is EventType.SESSION_INTERRUPTED
    assert cleanup_message not in rendered
    assert secret[:16] not in rendered
    assert "'error_type': 'RuntimeError'" in rendered
    assert "'error':" not in rendered


def test_tool_failure_redacts_dynamically_resolved_secret_before_diagnostic_bound() -> None:
    from cayu.runtime._diagnostics import MAX_DIAGNOSTIC_UTF8_BYTES

    secret = "dynamic-tool-boundary-secret-canary"
    prefix = "d" * (MAX_DIAGNOSTIC_UTF8_BYTES - len(secret.encode("utf-8")) // 2)

    class FailingDynamicSecretTool(Tool):
        spec = ToolSpec(
            name="dynamic_failure",
            description="Resolve a secret and fail.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del args
            assert ctx.vault is not None
            resolved = await ctx.vault.resolve(SecretRef(name="api_key"))
            raise RuntimeError(prefix + resolved.value.get_secret_value())

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_dynamic_failure",
                    name="dynamic_failure",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [ModelStreamEvent.completed({"finish_reason": "stop"})],
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
        tools=[FailingDynamicSecretTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_dynamic_failure_boundary",
                messages=[Message.text("user", "run")],
            ),
        )
    )
    failed = next(event for event in events if event.type is EventType.TOOL_CALL_FAILED)
    rendered = repr(failed.payload)

    assert secret not in rendered
    assert secret[: len(secret) // 2] not in rendered
    assert len(failed.payload["result"]["content"].encode("utf-8")) <= (MAX_DIAGNOSTIC_UTF8_BYTES)


def test_short_secret_in_message_argument_key_fails_closed() -> None:
    from cayu.runtime._message_redaction import redact_untrusted_message_for_boundary
    from cayu.vaults import SecretRedactor

    secret = "k9"
    message = Message.tool_call(
        tool_call_id="call-safe",
        tool_name="safe-tool",
        arguments={f"prefix-{secret}-suffix": "value"},
    )

    with pytest.raises(ValueError, match="workload secret in an object key"):
        redact_untrusted_message_for_boundary(
            message,
            redactor=SecretRedactor(secret),
            field_name="message",
        )


def test_short_secret_substring_in_typed_attachment_key_remains_valid() -> None:
    from cayu.artifacts import FileAttachment, FileAttachmentKind
    from cayu.runtime._message_redaction import redact_untrusted_message_for_boundary

    attachment = FileAttachment(
        artifact_id="artifact-safe",
        kind=FileAttachmentKind.IMAGE,
        filename="safe.png",
        content_type="image/png",
        size_bytes=1,
    )
    message = Message(
        role=MessageRole.USER,
        content=(FilePart(attachment=attachment.model_dump(mode="json")),),
    )

    redacted = redact_untrusted_message_for_boundary(
        message,
        redactor=SecretRedactor("id"),
        field_name="message",
    )

    assert redacted == message


@pytest.mark.parametrize(
    "secret",
    [
        "artifact_id",
        "content_type",
        "filename",
        "kind",
        "metadata",
        "size_bytes",
        "type",
    ],
)
def test_attachment_schema_key_exemption_is_scoped_to_typed_attachment(
    secret: str,
) -> None:
    from cayu.artifacts import FileAttachment, FileAttachmentKind
    from cayu.core.messages import ToolResultPart
    from cayu.runtime._message_redaction import redact_untrusted_message_for_boundary

    attachment = FileAttachment(
        artifact_id="artifact-safe",
        kind=FileAttachmentKind.IMAGE,
        filename="safe.png",
        content_type="image/png",
        size_bytes=1,
    )
    file_message = Message(
        role=MessageRole.USER,
        content=(FilePart(attachment=attachment.model_dump(mode="json")),),
    )

    assert (
        redact_untrusted_message_for_boundary(
            file_message,
            redactor=SecretRedactor(secret),
            field_name="message",
        )
        == file_message
    )

    untrusted_message = Message(
        role=MessageRole.TOOL,
        content=(
            ToolResultPart(
                tool_call_id="call-safe",
                tool_name="safe-tool",
                structured={secret: "safe-value"},
            ),
        ),
    )
    with pytest.raises(ValueError, match="workload secret in an object key"):
        redact_untrusted_message_for_boundary(
            untrusted_message,
            redactor=SecretRedactor(secret),
            field_name="message",
        )


@pytest.mark.parametrize("secret", ["decision", "error", "metadata", "reason"])
def test_runtime_result_schema_key_exemption_rejects_untyped_lookalike(
    secret: str,
) -> None:
    from cayu.core.messages import ToolResultPart
    from cayu.runtime._message_redaction import redact_untrusted_message_for_boundary

    message = Message(
        role=MessageRole.TOOL,
        content=(
            ToolResultPart(
                tool_call_id="call-safe",
                tool_name="safe-tool",
                structured={secret: "safe-value"},
            ),
        ),
    )

    with pytest.raises(ValueError, match="workload secret in an object key"):
        redact_untrusted_message_for_boundary(
            message,
            redactor=SecretRedactor(secret),
            field_name="message",
        )


def test_invalid_terminal_control_is_treated_as_untrusted_tool_data() -> None:
    from cayu.core.messages import ToolResultPart
    from cayu.runtime._message_redaction import redact_untrusted_message_for_boundary

    message = Message(
        role=MessageRole.TOOL,
        content=(
            ToolResultPart(
                tool_call_id="call-safe",
                tool_name="safe-tool",
                structured={
                    "terminal_outcome": "ordinary-tool-value",
                    "detail": "safe",
                },
            ),
        ),
    )

    assert (
        redact_untrusted_message_for_boundary(
            message,
            redactor=SecretRedactor("unrelated-secret"),
            field_name="message",
        )
        == message
    )


@pytest.mark.parametrize("secret", ["decision", "metadata", "reason"])
def test_approval_denial_schema_keys_do_not_block_model_continuation(secret: str) -> None:
    class ApprovalTool(Tool):
        spec = ToolSpec(
            name="approval_tool",
            description="Must be denied in this test.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del ctx, args
            raise AssertionError("denied tool must not execute")

    class ApprovalPolicy(ToolPolicy):
        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            del request
            return ToolPolicyResult(
                decision=ToolPolicyDecision.REQUIRE_APPROVAL,
                reason="operator approval required",
            )

    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call-approval",
                    name="approval_tool",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("denial handled"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    store = InMemorySessionStore()
    pause_app = CayuApp(
        session_store=store,
        enable_logging=False,
    )
    resume_app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    for app in (pause_app, resume_app):
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[ApprovalTool()],
            tool_policy=ApprovalPolicy(),
        )

    async def scenario() -> list[Event]:
        session_id = "approval_schema_1"
        interrupted = await collect_events(
            pause_app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "use the tool")],
            ),
        )
        approval = next(
            event for event in interrupted if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
        ).payload["approval"]
        return [
            event
            async for event in resume_app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id=session_id,
                    approval_id=approval["approval_id"],
                    tool_round_id=approval["tool_round_id"],
                    tool_call_id=approval["tool_call_id"],
                    decision=ToolApprovalDecision.DENY,
                    reason="operator denied execution",
                    metadata={"safe": "value"},
                )
            )
        ]

    resumed = asyncio.run(scenario())

    assert resumed[-1].type is EventType.SESSION_COMPLETED
    assert len(provider.requests) == 2
    denial_message = provider.requests[1].messages[-1]
    assert denial_message.role is MessageRole.TOOL
    assert denial_message.content[0].structured["denied_by_approval"] is True


def test_short_secret_in_attachment_metadata_key_fails_closed() -> None:
    from cayu.artifacts import FileAttachment, FileAttachmentKind
    from cayu.runtime._message_redaction import redact_untrusted_message_for_boundary

    secret = "k9"
    attachment = FileAttachment(
        artifact_id="artifact-safe",
        kind=FileAttachmentKind.IMAGE,
        filename="safe.png",
        content_type="image/png",
        size_bytes=1,
        metadata={f"prefix-{secret}-suffix": "value"},
    )
    message = Message(
        role=MessageRole.USER,
        content=(FilePart(attachment=attachment.model_dump(mode="json")),),
    )

    with pytest.raises(ValueError, match="workload secret in an object key"):
        redact_untrusted_message_for_boundary(
            message,
            redactor=SecretRedactor(secret),
            field_name="message",
        )


def test_short_secret_in_model_tool_schema_key_blocks_provider_dispatch() -> None:
    from cayu.vaults import SecretRedactor

    secret = "k9"

    class SecretKeySchemaTool(Tool):
        spec = ToolSpec(
            name="safe_tool",
            description="Schema contains an unsafe caller-controlled key.",
            input_schema={
                "type": "object",
                "properties": {
                    f"field-{secret}-suffix": {"type": "string"},
                },
            },
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del ctx, args
            return ToolResult(content="unused")

    provider = FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})])
    app = CayuApp(
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[SecretKeySchemaTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_short_schema_key",
                messages=[Message.text("user", "run")],
            ),
        )
    )

    assert provider.requests == []
    assert events[-1].type is EventType.SESSION_FAILED


def test_short_secret_substring_in_json_schema_keyword_allows_provider_dispatch() -> None:
    class SafeSchemaTool(Tool):
        spec = ToolSpec(
            name="safe_tool",
            description="Safe schema.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del ctx, args
            return ToolResult(content="unused")

    provider = FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})])
    app = CayuApp(
        secret_redactor=SecretRedactor("yp"),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[SafeSchemaTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_short_schema_keyword",
                messages=[Message.text("user", "run")],
            ),
        )
    )

    assert len(provider.requests) == 1
    assert provider.requests[0].tools[0]["input_schema"] == {
        "type": "object",
        "properties": {},
    }
    assert events[-1].type is EventType.SESSION_COMPLETED


def test_short_secret_substring_in_legacy_tuple_schema_allows_provider_dispatch() -> None:
    legacy_schema = {
        "type": "array",
        "items": [
            {"type": "string"},
            {"type": "number"},
        ],
    }

    class LegacyTupleSchemaTool(Tool):
        spec = ToolSpec(
            name="legacy_tuple_tool",
            description="Legacy draft-07 tuple schema.",
            input_schema=legacy_schema,
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del ctx, args
            return ToolResult(content="unused")

    provider = FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})])
    app = CayuApp(
        secret_redactor=SecretRedactor("typ"),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[LegacyTupleSchemaTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_legacy_tuple_schema_keyword",
                messages=[Message.text("user", "run")],
            ),
        )
    )

    assert len(provider.requests) == 1
    assert provider.requests[0].tools[0]["input_schema"] == legacy_schema
    assert events[-1].type is EventType.SESSION_COMPLETED


@pytest.mark.parametrize(
    ("secret", "legacy_schema"),
    [
        ("id", {"id": "urn:example:safe", "type": "object"}),
        (
            "recursive",
            {
                "$recursiveAnchor": True,
                "$recursiveRef": "#",
                "type": "object",
            },
        ),
        (
            "extends",
            {
                "extends": [{"type": "object"}],
                "type": "object",
            },
        ),
    ],
)
def test_legacy_json_schema_keywords_allow_provider_dispatch(
    secret: str,
    legacy_schema: dict[str, Any],
) -> None:
    class LegacySchemaTool(Tool):
        spec = ToolSpec(
            name="legacy_schema_tool",
            description="Legacy schema.",
            input_schema=legacy_schema,
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del ctx, args
            return ToolResult(content="unused")

    provider = FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})])
    app = CayuApp(
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[LegacySchemaTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_legacy_schema_protocol_keyword",
                messages=[Message.text("user", "run")],
            ),
        )
    )

    assert len(provider.requests) == 1
    assert provider.requests[0].tools[0]["input_schema"] == legacy_schema
    assert events[-1].type is EventType.SESSION_COMPLETED


def test_short_secret_in_dispatch_metadata_key_fails_closed() -> None:
    from cayu.runtime.dispatch import DispatchRequest, redact_dispatch_request
    from cayu.vaults import SecretRedactor

    secret = "k9"
    request = DispatchRequest(
        session_id="sess-safe",
        dispatch_id="dispatch-safe",
        messages=[Message.text("user", "queue")],
        metadata={f"prefix-{secret}-suffix": "value"},
    )

    with pytest.raises(ValueError, match="workload secret in an object key"):
        redact_dispatch_request(
            request,
            redactor=SecretRedactor(secret),
        )


def test_short_secret_substring_in_typed_dispatch_key_remains_valid() -> None:
    from cayu.core.thinking import ThinkingConfig
    from cayu.runtime.dispatch import DispatchRequest, redact_dispatch_request
    from cayu.vaults import SecretRedactor

    redacted = redact_dispatch_request(
        DispatchRequest(
            session_id="sess-safe",
            dispatch_id="dispatch-safe",
            messages=[Message.text("user", "queue")],
            thinking=ThinkingConfig(effort="low"),
        ),
        redactor=SecretRedactor("fort"),
    )

    assert redacted.thinking is not None
    assert redacted.thinking.effort == "low"


def test_dispatch_rejects_secret_pricing_dimension_key() -> None:
    from decimal import Decimal

    from cayu.runtime.budgets import BudgetLimit
    from cayu.runtime.costs import ModelPrice, PriceBook
    from cayu.runtime.dispatch import DispatchRequest, redact_dispatch_request

    secret = "dispatch-budget-dimension-secret-canary"
    price = ModelPrice.fixed(
        provider_name="custom",
        model="safe-model",
        input_per_million=Decimal("1"),
        output_per_million=Decimal("1"),
        match="exact",
        pricing_context={secret: ("safe-value",)},
    )
    request = DispatchRequest(
        session_id="sess-safe",
        messages=[Message.text("user", "queue")],
        budget_limits=[
            BudgetLimit(
                max_estimated_cost=Decimal("5"),
                pricing=PriceBook(prices=(price,)),
            )
        ],
    )

    with pytest.raises(ValueError, match="workload secret in an object key"):
        redact_dispatch_request(
            request,
            redactor=SecretRedactor(secret),
        )


@pytest.mark.parametrize("contaminated_state", ["labels", "transcript", "checkpoint"])
def test_cayu_app_refuses_to_fork_legacy_secret_bearing_source_state(
    contaminated_state: str,
) -> None:
    from cayu.vaults import SecretRedactor

    secret = f"legacy-fork-{contaminated_state}-secret-canary"
    source_id = f"sess_legacy_fork_{contaminated_state}_source"
    child_id = f"sess_legacy_fork_{contaminated_state}_child"
    store = InMemorySessionStore()
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(FakeProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def scenario() -> BaseException:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=source_id,
                labels=({f"owner-{secret}": "unsafe"} if contaminated_state == "labels" else {}),
                messages=[Message.text("user", "not copied by create")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        if contaminated_state == "transcript":
            await store.append_transcript_messages(
                source_id,
                [Message.text("user", f"legacy message {secret}")],
            )
        if contaminated_state == "checkpoint":
            await store.checkpoint(
                source_id,
                {"pending_execution": {"credential": secret}},
            )
        await store.update_status(source_id, SessionStatus.COMPLETED)

        with pytest.raises(ValueError, match="workload secret") as exc_info:
            await collect_fork_events(
                app,
                ForkSessionRequest(
                    source_session_id=source_id,
                    session_id=child_id,
                    copy_checkpoint=contaminated_state == "checkpoint",
                ),
            )
        return exc_info.value

    error = asyncio.run(scenario())

    assert asyncio.run(store.load(child_id)) is None
    traceback = error.__traceback__
    while traceback is not None:
        if is_cayu_source_filename(traceback.tb_frame.f_code.co_filename):
            retained = {
                name: type(value).__name__
                for name, value in traceback.tb_frame.f_locals.items()
                if secret in repr(value)
            }
            assert retained == {}, (
                traceback.tb_frame.f_code.co_filename,
                traceback.tb_frame.f_code.co_name,
                retained,
            )
        traceback = traceback.tb_next


def test_cayu_app_can_fork_without_copying_contaminated_legacy_checkpoint() -> None:
    from cayu.vaults import SecretRedactor

    secret = "legacy-fork-ignored-checkpoint-secret-canary"
    store = InMemorySessionStore()
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(FakeProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def scenario() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_legacy_checkpoint_not_copied_source",
                messages=[Message.text("user", "source")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.checkpoint(
            "sess_legacy_checkpoint_not_copied_source",
            {"legacy_diagnostic": secret},
        )
        await store.update_status(
            "sess_legacy_checkpoint_not_copied_source",
            SessionStatus.COMPLETED,
        )
        await collect_fork_events(
            app,
            ForkSessionRequest(
                source_session_id="sess_legacy_checkpoint_not_copied_source",
                session_id="sess_legacy_checkpoint_not_copied_child",
                copy_checkpoint=False,
            ),
        )

    asyncio.run(scenario())

    assert asyncio.run(store.load_checkpoint("sess_legacy_checkpoint_not_copied_child")) is None


def test_fork_validates_only_checkpoint_state_copied_to_child() -> None:
    from cayu.vaults import SecretRedactor

    secret = "legacy-fork-excluded-operation-history-secret-canary"
    source_id = "sess_legacy_operation_history_source"
    child_id = "sess_legacy_operation_history_child"
    store = InMemorySessionStore()
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(FakeProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def scenario() -> dict[str, Any] | None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=source_id,
                messages=[Message.text("user", "source")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.checkpoint(
            source_id,
            {
                "session_operations": {
                    "version": 1,
                    "active_operation_id": None,
                    "records": {
                        "legacy-operation": {
                            "status": "completed",
                            "diagnostic": secret,
                        }
                    },
                },
                "safe_state": {"value": "copied"},
            },
        )
        await store.update_status(source_id, SessionStatus.COMPLETED)

        await collect_fork_events(
            app,
            ForkSessionRequest(
                source_session_id=source_id,
                session_id=child_id,
                copy_checkpoint=True,
            ),
        )
        return await store.load_checkpoint(child_id)

    child_checkpoint = asyncio.run(scenario())

    assert child_checkpoint == {
        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
        "safe_state": {"value": "copied"},
    }
    assert secret not in repr(child_checkpoint)


def test_fork_failure_does_not_retain_excluded_legacy_checkpoint_secret() -> None:
    from cayu.vaults import SecretRedactor

    secret = "legacy-fork-excluded-checkpoint-traceback-canary"
    source_id = "sess_legacy_excluded_checkpoint_failure_source"
    store = InMemorySessionStore()
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def scenario() -> BaseException:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=source_id,
                messages=[Message.text("user", "source")],
            ),
            identity=SessionIdentity(provider_name="missing-provider", model="fake-model"),
        )
        await store.checkpoint(source_id, {"legacy_diagnostic": secret})
        await store.update_status(source_id, SessionStatus.COMPLETED)

        with pytest.raises(KeyError, match="Provider not registered") as exc_info:
            await collect_fork_events(
                app,
                ForkSessionRequest(
                    source_session_id=source_id,
                    session_id="sess_legacy_excluded_checkpoint_failure_child",
                    copy_checkpoint=False,
                ),
            )
        return exc_info.value

    error = asyncio.run(scenario())

    traceback = error.__traceback__
    while traceback is not None:
        if is_cayu_source_filename(traceback.tb_frame.f_code.co_filename):
            assert all(secret not in repr(value) for value in traceback.tb_frame.f_locals.values())
        traceback = traceback.tb_next


def test_fork_validates_concurrently_added_transcript_inside_atomic_store_copy() -> None:
    from cayu.vaults import SecretRedactor

    secret = "legacy-fork-concurrent-transcript-secret-canary"
    source_id = "sess_legacy_concurrent_transcript_source"
    child_id = "sess_legacy_concurrent_transcript_child"

    class ConcurrentAppendBeforeForkStore(InMemorySessionStore):
        async def create_fork_with_transcript_validation(self, *args, **kwargs):
            await self.append_transcript_messages(
                source_id,
                [Message.text("user", f"concurrent legacy message {secret}")],
            )
            return await super().create_fork_with_transcript_validation(*args, **kwargs)

    store = ConcurrentAppendBeforeForkStore()
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(FakeProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def scenario() -> BaseException:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=source_id,
                messages=[Message.text("user", "source")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.append_transcript_messages(
            source_id,
            [Message.text("user", "safe prefix")],
        )
        await store.update_status(source_id, SessionStatus.COMPLETED)

        with pytest.raises(ValueError, match="workload secret") as exc_info:
            await collect_fork_events(
                app,
                ForkSessionRequest(
                    source_session_id=source_id,
                    session_id=child_id,
                ),
            )
        return exc_info.value

    error = asyncio.run(scenario())

    assert asyncio.run(store.load(child_id)) is None
    traceback = error.__traceback__
    while traceback is not None:
        if is_cayu_source_filename(traceback.tb_frame.f_code.co_filename):
            assert all(secret not in repr(value) for value in traceback.tb_frame.f_locals.values())
        traceback = traceback.tb_next


def test_fork_cursor_excluded_secret_is_not_retained_by_later_failure() -> None:
    from cayu.vaults import SecretRedactor

    secret = "legacy-fork-excluded-transcript-traceback-canary"
    source_id = "sess_legacy_excluded_transcript_source"
    child_id = "sess_legacy_excluded_transcript_existing_child"
    store = InMemorySessionStore()
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(FakeProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def scenario() -> BaseException:
        for session_id in (source_id, child_id):
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "source")],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
            await store.update_status(session_id, SessionStatus.COMPLETED)
        await store.append_transcript_messages(
            source_id,
            [
                Message.text("user", "safe prefix"),
                Message.text("user", f"excluded legacy message {secret}"),
            ],
        )

        with pytest.raises(ValueError, match="Session already exists") as exc_info:
            await collect_fork_events(
                app,
                ForkSessionRequest(
                    source_session_id=source_id,
                    session_id=child_id,
                    transcript_cursor=1,
                    copy_checkpoint=False,
                ),
            )
        return exc_info.value

    error = asyncio.run(scenario())

    traceback = error.__traceback__
    while traceback is not None:
        if is_cayu_source_filename(traceback.tb_frame.f_code.co_filename):
            assert all(secret not in repr(value) for value in traceback.tb_frame.f_locals.values())
        traceback = traceback.tb_next


def test_parallel_generic_cancellation_preserves_caller_signal_without_secret_evidence() -> None:
    secret_value = "generic-cancellation-dynamic-secret-canary"

    class DynamicSecretBlockingTool(Tool):
        spec = ToolSpec(
            name="dynamic_secret_blocking_tool",
            description="Resolve a secret and block until caller cancellation.",
            input_schema={"type": "object", "properties": {}},
        )

        def __init__(self) -> None:
            self.all_started: asyncio.Event | None = None
            self.arrivals = 0

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            assert ctx.vault is not None
            assert self.all_started is not None
            secret = await ctx.vault.resolve(SecretRef(name="api_key"))
            raw_secret = secret.value.get_secret_value()
            self.arrivals += 1
            if self.arrivals == 2:
                self.all_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as exc:
                cancellation = RunnerCancelledError(
                    f"tool cancelled near {raw_secret}",
                    artifacts=[
                        {
                            "type": "cayu.runner_cleanup.v1",
                            "status": "failed",
                            "stderr": raw_secret,
                        }
                    ],
                )
                cancellation.__dict__.update(
                    {
                        "_cayu_cancellation_tool_call_id": "spoofed-call",
                    }
                )
                raise cancellation from exc
            return ToolResult(content="unexpected")

    tool = DynamicSecretBlockingTool()
    app = CayuApp(enable_logging=False, max_parallel_tool_calls=2)
    app.register_provider(
        FakeProvider(
            [
                ModelStreamEvent.tool_call(
                    id="call_dynamic_secret_block",
                    name="dynamic_secret_blocking_tool",
                    arguments={},
                ),
                ModelStreamEvent.tool_call(
                    id="call_dynamic_secret_block_2",
                    name="dynamic_secret_blocking_tool",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        ),
        default=True,
    )
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            vault=StaticVault({"api_key": secret_value}),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
    )

    async def scenario() -> tuple[asyncio.CancelledError, int, bool]:
        tool.all_started = asyncio.Event()
        run_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_generic_dynamic_secret_cancel",
                    messages=[Message.text("user", "use tool")],
                ),
            )
        )
        await tool.all_started.wait()
        run_task.cancel("caller cancelled")
        cancelling = run_task.cancelling()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await run_task
        return exc_info.value, cancelling, run_task.cancelled()

    cancellation, cancelling, cancelled = asyncio.run(scenario())

    assert cancelling == 1
    assert cancelled is True
    assert cancellation.args == ("caller cancelled",)
    assert secret_value not in repr((cancellation.args, cancellation.__dict__))
    assert cancellation.__dict__ == {}
    assert cancellation.__cause__ is None
    assert cancellation.__context__ is None
    traceback = cancellation.__traceback__
    while traceback is not None:
        if is_cayu_source_filename(traceback.tb_frame.f_code.co_filename):
            assert all(
                secret_value not in repr(value) for value in traceback.tb_frame.f_locals.values()
            )
        traceback = traceback.tb_next


def test_runner_cancellation_drops_secret_context_added_by_tool_exception_handler() -> None:
    secret_value = "runner-tool-context-secret-canary-ABCDEFGHIJKLMNOP"

    class BlockingRunner(Runner):
        def __init__(self) -> None:
            self.started: asyncio.Event | None = None

        async def exec(
            self,
            command: ExecCommand,
            **kwargs: Any,
        ) -> ExecResult:
            # Intentionally retain the complete request in this adapter frame;
            # the invocation handle must keep it out of the escaped traceback.
            request = (command, kwargs)
            assert request
            assert self.started is not None
            self.started.set()
            await asyncio.Event().wait()
            return ExecResult()

    class ExceptionHandlingRunnerTool(Tool):
        spec = ToolSpec(
            name="exception_handling_runner_tool",
            description="Await a runner while handling a secret-bearing exception.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del args
            assert ctx.vault is not None
            assert ctx.runner is not None
            resolved = await ctx.vault.resolve(SecretRef(name="api_key"))
            raw_secret = resolved.value.get_secret_value()
            try:
                raise RuntimeError(raw_secret)
            except RuntimeError:
                await ctx.runner.exec(
                    ExecCommand.process("blocked", raw_secret),
                    env={"WORKLOAD_TOKEN": raw_secret},
                    stdin=raw_secret,
                )
            return ToolResult(content="unexpected")

    runner = BlockingRunner()
    app = CayuApp(enable_logging=False)
    app.register_provider(
        FakeProvider(
            [
                ModelStreamEvent.tool_call(
                    id="call_exception_handling_runner",
                    name="exception_handling_runner_tool",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        ),
        default=True,
    )
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            runner=runner,
            vault=StaticVault({"api_key": secret_value}),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[ExceptionHandlingRunnerTool()],
    )

    async def scenario() -> tuple[asyncio.CancelledError, int, bool]:
        runner.started = asyncio.Event()
        run_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_runner_tool_context_cancel",
                    messages=[Message.text("user", "use tool")],
                ),
            )
        )
        await runner.started.wait()
        run_task.cancel("caller cancelled")
        cancelling = run_task.cancelling()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await run_task
        return exc_info.value, cancelling, run_task.cancelled()

    cancellation, cancelling, cancelled = asyncio.run(scenario())

    assert cancelling == 1
    assert cancelled is True
    assert cancellation.args == ("caller cancelled",)
    assert cancellation.__cause__ is None
    assert cancellation.__context__ is None
    assert cancellation.__dict__ == {}
    _assert_cayu_traceback_does_not_retain_text(cancellation, secret_value)


def test_runtime_sanitizes_grouped_runner_failure_from_real_caller_cancellation() -> None:
    secret_value = "grouped-runner-cancellation-secret-canary-ABCDEFGHIJKLMNOP"

    class GroupedCancellationRunner(Runner):
        def __init__(self) -> None:
            self.started: asyncio.Event | None = None

        async def exec(self, command: ExecCommand, **kwargs) -> ExecResult:
            del command, kwargs
            assert self.started is not None
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as cancellation:
                attach_cancellation_artifacts(
                    cancellation,
                    [
                        {
                            "type": "cayu.runner_cleanup.v1",
                            "adapter": "microsandbox",
                            "action": "kill_command",
                            "status": "failed",
                            "timeout_s": 5.0,
                            "error": secret_value,
                        }
                    ],
                )
                cleanup = RuntimeError(f"cleanup exposed {secret_value}")
                raise BaseExceptionGroup(
                    f"runner cleanup exposed {secret_value}",
                    [cancellation, cleanup],
                ) from cleanup
            return ExecResult()

    class GroupedCancellationTool(Tool):
        spec = ToolSpec(
            name="grouped_cancellation",
            description="Resolve a secret and block in the runner.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del args
            assert ctx.vault is not None
            assert ctx.runner is not None
            await ctx.vault.resolve(SecretRef(name="api_key"))
            await ctx.runner.exec(ExecCommand.process("blocked"))
            return ToolResult(content="unexpected")

    runner = GroupedCancellationRunner()
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(
        FakeProvider(
            [
                ModelStreamEvent.tool_call(
                    id="call_grouped_cancellation",
                    name="grouped_cancellation",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        ),
        default=True,
    )
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            runner=runner,
            vault=StaticVault({"api_key": secret_value}),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[GroupedCancellationTool()],
    )

    async def scenario():
        runner.started = asyncio.Event()
        run_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_grouped_runner_cancel",
                    messages=[Message.text("user", "run")],
                ),
            )
        )
        await runner.started.wait()
        run_task.cancel("caller cancellation")
        cancelling = run_task.cancelling()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await run_task
        stored_events = await store.load_events("sess_grouped_runner_cancel")
        transcript = await store.load_transcript("sess_grouped_runner_cancel")
        return (
            exc_info.value,
            cancelling,
            run_task.cancelled(),
            stored_events,
            transcript,
        )

    failure, cancelling, cancelled, stored_events, transcript = asyncio.run(scenario())

    assert cancelling == 1
    assert cancelled is True
    assert failure.args == ("caller cancellation",)
    assert failure.artifacts == [
        {
            "type": "cayu.runner_cleanup.v1",
            "adapter": "microsandbox",
            "action": "kill_command",
            "status": "failed",
            "timeout_s": 5.0,
        }
    ]
    assert isinstance(failure.__cause__, ExceptionGroup)
    assert len(failure.__cause__.exceptions) == 1
    assert isinstance(failure.__cause__.exceptions[0], RunnerExecutionError)
    assert [event.type for event in stored_events].count(EventType.SESSION_INTERRUPTED) == 1
    assert all(event.type is not EventType.SESSION_COMPLETED for event in stored_events)
    interrupted_events = [
        event
        for event in stored_events
        if event.type is EventType.TOOL_CALL_FAILED
        and event.payload.get("tool_call_id") == "call_grouped_cancellation"
    ]
    assert len(interrupted_events) == 1
    assert interrupted_events[0].payload["result"]["artifacts"] == failure.artifacts
    rendered = repr(
        (
            failure,
            [event.model_dump(mode="json") for event in stored_events],
            [message.model_dump(mode="json") for message in transcript],
        )
    )
    assert secret_value not in rendered
    assert "cleanup exposed" not in rendered


def test_caller_cancellation_cannot_spoof_runtime_cleanup_artifacts() -> None:
    cancellation = RunnerCancelledError(
        "caller cancellation",
        artifacts=[{"type": "caller-controlled", "value": "not runtime evidence"}],
    )
    cancellation.__dict__.update(
        {
            "_cayu_cancellation_tool_call_id": "spoofed-call",
        }
    )

    assert invocation_secrets_module.cancellation_artifacts(cancellation) == []
    assert invocation_secrets_module.cancellation_tool_call_id(cancellation) is None
    assert invocation_secrets_module.cancellation_artifacts_by_id(cancellation) is None


def test_operator_interrupt_redacts_secret_resolved_during_cancelled_tool() -> None:
    from cayu.vaults import REDACTED_SECRET

    secret_value = "cancelled-tool-dynamic-vault-secret-canary"

    class DynamicallySecretTool(Tool):
        spec = ToolSpec(
            name="dynamic_secret_cancel",
            description="Resolve a secret and wait for operator interruption.",
            input_schema={"type": "object", "properties": {}},
        )

        def __init__(self) -> None:
            self.started: asyncio.Event | None = None

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            assert ctx.vault is not None
            assert self.started is not None
            secret = await ctx.vault.resolve(SecretRef(name="api_key"))
            raw_secret = secret.value.get_secret_value()
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as exc:
                raise RunnerCancelledError(
                    artifacts=[
                        {
                            "type": "cayu.runner_cleanup.v1",
                            "status": "failed",
                            "stderr": f"cleanup exposed {raw_secret}",
                            "metadata": {"token": raw_secret},
                        }
                    ]
                ) from exc
            return ToolResult(content="unexpected")

    tool = DynamicallySecretTool()
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(
        FakeProvider(
            [
                ModelStreamEvent.tool_call(
                    id="call_dynamic_secret_cancel",
                    name="dynamic_secret_cancel",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        ),
        default=True,
    )
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            vault=StaticVault({"api_key": secret_value}),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
    )

    async def scenario():
        tool.started = asyncio.Event()
        run_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_dynamic_secret_cancel",
                    messages=[Message.text("user", "use tool")],
                ),
            )
        )
        await tool.started.wait()
        interrupt_events = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id="sess_dynamic_secret_cancel",
                    reason="operator stop",
                )
            )
        ]
        run_events = await run_task
        stored_events = await store.load_events("sess_dynamic_secret_cancel")
        transcript = await store.load_transcript("sess_dynamic_secret_cancel")
        return interrupt_events, run_events, stored_events, transcript

    interrupt_events, run_events, stored_events, transcript = asyncio.run(scenario())
    rendered = repr(
        (
            [event.model_dump(mode="json") for event in interrupt_events],
            [event.model_dump(mode="json") for event in run_events],
            [event.model_dump(mode="json") for event in stored_events],
            [message.model_dump(mode="json") for message in transcript],
        )
    )

    assert run_events[-1].type is EventType.SESSION_INTERRUPTED
    assert secret_value not in rendered
    assert REDACTED_SECRET in rendered


@pytest.mark.parametrize(
    ("event_type", "authority_field"),
    [
        (EventType.TOOL_CALL_APPROVAL_REQUESTED, "approval_id"),
        (EventType.RUNTIME_SINK_FAILED, "event_id"),
        (EventType.SESSION_AWAITING_USER_INPUT, "input_id"),
        (EventType.MODEL_STARTED, "model_attempt_id"),
        (EventType.BUDGET_RESERVED, "session_id"),
        (EventType.TASK_CREATED, "task_id"),
        (EventType.TOOL_CALL_STARTED, "tool_call_id"),
        (EventType.TOOL_CALL_STARTED, "tool_round_id"),
        (EventType.TOOL_CALL_STARTED, "idempotency_key"),
    ],
)
def test_runtime_event_rejects_secret_in_event_type_owned_payload_authority(
    event_type: EventType,
    authority_field: str,
) -> None:
    from cayu.runtime._event_writer import prepare_runtime_event
    from cayu.vaults import SecretRedactor

    secret = f"event-{authority_field}-secret-canary"
    event = Event(
        type=event_type,
        session_id="sess-safe",
        payload={authority_field: secret},
    )

    with pytest.raises(ValueError, match=rf"event\.payload\.{authority_field}"):
        prepare_runtime_event(
            event,
            redactor=SecretRedactor(secret),
        )


@pytest.mark.parametrize("descriptive_field", ["model", "provider", "policy_name"])
def test_runtime_event_redacts_secret_descriptive_values_without_granting_authority(
    descriptive_field: str,
) -> None:
    from cayu.runtime._event_writer import prepare_runtime_event

    secret = f"event-{descriptive_field}-description-canary"
    event = Event(
        type=EventType.MODEL_STARTED,
        session_id="sess-safe",
        payload={descriptive_field: secret},
    )

    prepared = prepare_runtime_event(event, redactor=SecretRedactor(secret))

    assert prepared.payload[descriptive_field] == REDACTED_SECRET


def test_runtime_event_does_not_borrow_authority_from_unrelated_event_type() -> None:
    from cayu.runtime._event_writer import prepare_runtime_event

    secret = "unrelated-tool-call-id-secret-canary"
    prepared = prepare_runtime_event(
        Event(
            type=EventType.SESSION_STARTED,
            session_id="sess-safe",
            payload={"tool_call_id": secret},
        ),
        redactor=SecretRedactor(secret),
    )

    assert prepared.payload == {"tool_call_id": REDACTED_SECRET}


def test_runtime_event_rejects_secret_interaction_authority() -> None:
    from cayu.runtime._event_writer import prepare_runtime_event

    secret = "event-interaction-id-secret-canary"
    event = Event(
        type=EventType.MODEL_STARTED,
        session_id="sess-safe",
        interaction_id=secret,
    )

    with pytest.raises(ValueError, match=r"event\.interaction_id"):
        prepare_runtime_event(event, redactor=SecretRedactor(secret))


def test_runtime_tool_event_cannot_restore_secret_linkage_control() -> None:
    from cayu.runtime._event_writer import prepare_runtime_event
    from cayu.vaults import SecretRedactor

    secret = "runtime-tool-linkage-secret-canary"
    event = Event(
        type=EventType.TOOL_CALL_STARTED,
        session_id="sess-safe",
        payload={
            "tool_call_id": secret,
            "tool_name": "safe-tool",
            "arguments": {},
        },
    )

    with pytest.raises(ValueError, match=r"event\.payload\.tool_call_id"):
        prepare_runtime_event(
            event,
            redactor=SecretRedactor(secret),
        )


def test_interaction_lifecycle_schema_keys_survive_secret_name_collisions() -> None:
    from cayu.runtime import InteractionStatus, InteractionSummaryEvidence
    from cayu.runtime._event_writer import prepare_runtime_event
    from cayu.vaults import SecretRedactor

    now = datetime.now(UTC)
    evidence = InteractionSummaryEvidence(
        status=InteractionStatus.COMPLETED,
        start_event_id="interaction-anchor-safe",
        source_transcript_start=0,
        source_transcript_end=0,
        result_transcript_start=1,
        result_transcript_end=1,
        started_at=now,
        completed_at=now,
        active_duration_ms=1,
        wall_duration_ms=1,
        model_step_count=1,
        tool_call_count=1,
        provider_names=["safe-provider"],
        models=["safe-model"],
    )

    prepared = prepare_runtime_event(
        Event(
            type=EventType.INTERACTION_COMPLETED,
            session_id="sess-safe",
            interaction_id="interaction-safe",
            payload=evidence.model_dump(mode="json"),
        ),
        redactor=SecretRedactor(
            [
                "active",
                "cache",
                "completed",
                "input",
                "model",
                "output",
                "provider",
                "result",
                "source",
                "start",
                "tool",
                "write",
            ]
        ),
    )

    restored = InteractionSummaryEvidence.model_validate(prepared.payload)
    assert restored.status is InteractionStatus.COMPLETED
    assert restored.start_event_id == "interaction-anchor-safe"


def test_pending_tool_round_rejects_secret_authority_on_write_and_legacy_load() -> None:
    from cayu.runtime import _runtime_records as runtime_records
    from cayu.runtime import _tool_round_recovery as tool_round_recovery
    from cayu.vaults import SecretRedactor

    secret = "pending-round-authority-secret-canary"
    tool_calls = [
        runtime_records.ToolCallRequest(
            id="call-safe",
            name="safe-tool",
            arguments={},
        )
    ]

    with pytest.raises(ValueError, match="workload secret"):
        tool_round_recovery.checkpoint_with_pending_tool_round(
            None,
            agent_name=secret,
            environment_name=None,
            task_id=None,
            tool_calls=tool_calls,
            policy_outcomes=None,
            structured_output=None,
            tool_round_identity=tool_round_identity(),
            redactor=SecretRedactor(secret),
        )

    legacy_checkpoint, _ = tool_round_recovery.checkpoint_with_pending_tool_round(
        None,
        agent_name=secret,
        environment_name=None,
        task_id=None,
        tool_calls=tool_calls,
        policy_outcomes=None,
        structured_output=None,
        tool_round_identity=tool_round_identity(),
    )
    with pytest.raises(ValueError, match="cannot be executed") as exc_info:
        tool_round_recovery.pending_tool_round_from_checkpoint(
            legacy_checkpoint,
            redactor=SecretRedactor(secret),
        )
    assert (
        legacy_checkpoint[tool_round_recovery.PENDING_TOOL_ROUND_CHECKPOINT_KEY]["agent_name"]
        == secret
    )
    _assert_cayu_traceback_does_not_retain_text(exc_info.value, secret)


def test_legacy_pending_approval_rejects_secret_authority_before_recovery() -> None:
    from cayu.runtime import _approval_support as approval_support
    from cayu.runtime.approvals import PendingToolApproval, PendingToolCallApproval
    from cayu.vaults import SecretRedactor

    secret = "legacy-approval-authority-secret-canary"
    pending = PendingToolApproval(
        **tool_round_identity().payload(),
        approval_id=secret,
        tool_call_id="call-safe",
        tool_name="safe-tool",
        agent_name="assistant",
        publish_arguments=True,
        tool_calls=[
            PendingToolCallApproval(
                tool_call_id="call-safe",
                tool_name="safe-tool",
            )
        ],
    )
    checkpoint = {
        approval_support.PENDING_TOOL_APPROVAL_CHECKPOINT_KEY: pending.model_dump(mode="json")
    }

    with pytest.raises(ValueError, match="workload secret") as exc_info:
        approval_support.pending_approval_from_checkpoint(
            checkpoint,
            redactor=SecretRedactor(secret),
        )
    _assert_cayu_traceback_does_not_retain_text(exc_info.value, secret)


def test_legacy_pending_approval_rejects_secret_argument_key_without_mutating_input() -> None:
    from cayu.runtime import _approval_support as approval_support
    from cayu.runtime.approvals import PendingToolApproval, PendingToolCallApproval
    from cayu.vaults import SecretRedactor

    secret = "legacy-approval-key-secret-canary"
    secret_key = f"prefix-{secret}-suffix"
    pending = PendingToolApproval(
        **tool_round_identity().payload(),
        approval_id="approval-safe",
        tool_call_id="call-safe",
        tool_name="safe-tool",
        arguments={secret_key: "top-level"},
        agent_name="assistant",
        publish_arguments=True,
        tool_calls=[
            PendingToolCallApproval(
                tool_call_id="call-safe",
                tool_name="safe-tool",
                arguments={secret_key: "top-level"},
            )
        ],
    )
    checkpoint = {
        approval_support.PENDING_TOOL_APPROVAL_CHECKPOINT_KEY: pending.model_dump(mode="json")
    }

    with pytest.raises(ValueError, match="workload secret") as exc_info:
        approval_support.pending_approval_from_checkpoint(
            checkpoint,
            redactor=SecretRedactor(secret),
        )

    assert checkpoint[approval_support.PENDING_TOOL_APPROVAL_CHECKPOINT_KEY]["arguments"] == {
        secret_key: "top-level"
    }
    _assert_cayu_traceback_does_not_retain_text(exc_info.value, secret)


def test_legacy_pending_user_input_rejects_secret_authority_before_recovery() -> None:
    from cayu.runtime.approvals import PendingToolCallApproval
    from cayu.runtime.user_input import (
        PENDING_USER_INPUT_CHECKPOINT_KEY,
        PendingUserInput,
        pending_user_input_from_checkpoint,
    )
    from cayu.vaults import SecretRedactor

    secret = "legacy-user-input-authority-secret-canary"
    pending = PendingUserInput(
        **tool_round_identity().payload(),
        input_id=secret,
        tool_call_id="call-safe",
        tool_name="safe-tool",
        question="Continue?",
        agent_name="assistant",
        tool_calls=[
            PendingToolCallApproval(
                tool_call_id="call-safe",
                tool_name="safe-tool",
            )
        ],
    )
    checkpoint = {
        PENDING_USER_INPUT_CHECKPOINT_KEY: pending.model_dump(mode="json"),
    }

    with pytest.raises(ValueError, match="workload secret") as exc_info:
        pending_user_input_from_checkpoint(
            checkpoint,
            redactor=SecretRedactor(secret),
        )
    _assert_cayu_traceback_does_not_retain_text(exc_info.value, secret)


def test_legacy_pending_user_input_rejects_secret_argument_key_without_mutating_input() -> None:
    from cayu.runtime.approvals import PendingToolCallApproval
    from cayu.runtime.user_input import (
        PENDING_USER_INPUT_CHECKPOINT_KEY,
        PendingUserInput,
        pending_user_input_from_checkpoint,
    )
    from cayu.vaults import SecretRedactor

    secret = "legacy-user-input-key-secret-canary"
    secret_key = f"prefix-{secret}-suffix"
    pending = PendingUserInput(
        **tool_round_identity().payload(),
        input_id="input-safe",
        tool_call_id="call-safe",
        tool_name="safe-tool",
        question="Continue?",
        arguments={secret_key: "top-level"},
        agent_name="assistant",
        tool_calls=[
            PendingToolCallApproval(
                tool_call_id="call-safe",
                tool_name="safe-tool",
                arguments={secret_key: "top-level"},
            )
        ],
    )
    checkpoint = {
        PENDING_USER_INPUT_CHECKPOINT_KEY: pending.model_dump(mode="json"),
    }

    with pytest.raises(ValueError, match="workload secret") as exc_info:
        pending_user_input_from_checkpoint(
            checkpoint,
            redactor=SecretRedactor(secret),
        )

    assert checkpoint[PENDING_USER_INPUT_CHECKPOINT_KEY]["arguments"] == {secret_key: "top-level"}
    _assert_cayu_traceback_does_not_retain_text(exc_info.value, secret)


def test_explicit_compaction_rejects_secret_bearing_legacy_pending_checkpoint() -> None:
    from cayu.runtime import _approval_support as approval_support
    from cayu.runtime import _session_engine as session_engine
    from cayu.runtime.approvals import PendingToolApproval, PendingToolCallApproval
    from cayu.vaults import SecretRedactor

    secret = "legacy-compaction-pending-secret-canary"
    pending = PendingToolApproval(
        **tool_round_identity().payload(),
        approval_id=secret,
        tool_call_id="call-safe",
        tool_name="safe-tool",
        agent_name="assistant",
        publish_arguments=True,
        tool_calls=[
            PendingToolCallApproval(
                tool_call_id="call-safe",
                tool_name="safe-tool",
            )
        ],
    )
    checkpoint = {
        approval_support.PENDING_TOOL_APPROVAL_CHECKPOINT_KEY: pending.model_dump(mode="json")
    }

    with pytest.raises(ValueError, match="workload secret") as exc_info:
        session_engine._reject_unresumable_session_checkpoint(
            cast("Session", object()),
            checkpoint,
            redactor=SecretRedactor(secret),
        )

    assert checkpoint == {}
    assert secret not in str(exc_info.value)
    _assert_cayu_traceback_does_not_retain_text(exc_info.value, secret)


def test_explicit_compaction_public_flow_rejects_legacy_secret_without_traceback_retention() -> (
    None
):
    from cayu.runtime import CompactSessionRequest
    from cayu.runtime import _approval_support as approval_support
    from cayu.runtime.approvals import PendingToolApproval, PendingToolCallApproval
    from cayu.vaults import SecretRedactor

    async def run() -> None:
        secret = "legacy-compaction-public-flow-secret-canary"
        session_id = "sess_legacy_compaction_secret"
        store = InMemorySessionStore()
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        pending = PendingToolApproval(
            **tool_round_identity().payload(),
            approval_id=secret,
            tool_call_id="call-safe",
            tool_name="safe-tool",
            agent_name="assistant",
            publish_arguments=True,
            tool_calls=[
                PendingToolCallApproval(
                    tool_call_id="call-safe",
                    tool_name="safe-tool",
                )
            ],
        )
        await store.transform_checkpoint(
            session_id,
            lambda _session, _checkpoint: {
                approval_support.PENDING_TOOL_APPROVAL_CHECKPOINT_KEY: pending.model_dump(
                    mode="json"
                )
            },
        )
        completed = await store.update_status(session_id, SessionStatus.COMPLETED)
        app = CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )

        with pytest.raises(ValueError, match="workload secret") as exc_info:
            async for _event in app.compact_session(
                CompactSessionRequest(
                    session_id=session_id,
                    idempotency_key="reject-secret-checkpoint",
                    expected_run_epoch=completed.run_epoch,
                    expected_transcript_cursor=0,
                )
            ):
                pass

        assert secret not in str(exc_info.value)
        _assert_cayu_traceback_does_not_retain_text(exc_info.value, secret)
        durable_checkpoint = await store.load_checkpoint(session_id)
        assert durable_checkpoint is not None
        assert (
            durable_checkpoint[approval_support.PENDING_TOOL_APPROVAL_CHECKPOINT_KEY]["approval_id"]
            == secret
        )

    asyncio.run(run())


def test_checkpoint_schema_keys_remain_valid_inside_typed_collections() -> None:
    from cayu.runtime._checkpoint_redaction import durable_value_contains_secret
    from cayu.vaults import SecretRedactor

    assert not durable_value_contains_secret(
        {
            "pending_user_input": {
                "budget_limits": [
                    {
                        "pricing": {
                            "prices": [
                                {
                                    "model": "safe-name",
                                    "input_per_million": "1",
                                }
                            ],
                            "resource_mappings": [
                                {
                                    "resource_id": "safe-resource",
                                    "model": "safe-name",
                                }
                            ],
                        }
                    }
                ]
            }
        },
        redactor=SecretRedactor("model"),
    )

    quarantined_message = {
        "pending_tool_round": {
            "quarantined_assistant_message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "harmless prose"},
                    {
                        "type": "provider_state",
                        "provider": "vendor",
                        "state": {
                            "safe-extension": "safe-value",
                            "status": "safe-value",
                        },
                    },
                    {
                        "type": "thinking",
                        "text": "safe thought",
                        "provider_state": {
                            "safe-extension": "safe-value",
                            "status": "safe-value",
                        },
                    },
                    {
                        "type": "tool_call",
                        "tool_call_id": "safe-call",
                        "tool_name": "safe-tool",
                        "arguments": {"safe-argument": "safe-value"},
                    },
                ],
            }
        }
    }
    for structural_key in (
        "content",
        "role",
        "type",
        "text",
        "provider",
        "state",
        "provider_state",
        "tool_call_id",
        "tool_name",
        "arguments",
    ):
        assert not durable_value_contains_secret(
            quarantined_message,
            redactor=SecretRedactor(structural_key),
        )
    assert durable_value_contains_secret(
        quarantined_message,
        redactor=SecretRedactor("safe-extension"),
    )
    assert durable_value_contains_secret(
        quarantined_message,
        redactor=SecretRedactor("status"),
    )
    assert durable_value_contains_secret(
        quarantined_message,
        redactor=SecretRedactor("safe-argument"),
    )

    assert durable_value_contains_secret(
        {
            "prices": [{"model": "safe-name", "input_per_million": "1"}],
        },
        redactor=SecretRedactor("model"),
    )
    assert durable_value_contains_secret(
        {"custom": {"model": "safe-name"}},
        redactor=SecretRedactor("model"),
    )
    assert durable_value_contains_secret(
        {"model": "safe-name"},
        redactor=SecretRedactor("model"),
    )
    assert durable_value_contains_secret(
        {
            "session_operations": {
                "records": {
                    "status": {
                        "status": "completed",
                    }
                }
            }
        },
        redactor=SecretRedactor("status"),
    )
    assert durable_value_contains_secret(
        {
            "environment_factory_reconnect": {
                "safe-environment": {
                    "metadata": "safe",
                }
            }
        },
        redactor=SecretRedactor("metadata"),
    )
    assert durable_value_contains_secret(
        {
            "environment_factory_reconnect": {
                "safe-environment": {
                    "nested": {
                        "status": "safe",
                    }
                }
            }
        },
        redactor=SecretRedactor("status"),
    )
    for environment_checkpoint_key in (
        "environment_factory_reconnect",
        "environment_factory_allocation_owner",
    ):
        assert durable_value_contains_secret(
            {
                environment_checkpoint_key: {
                    "status": (
                        {"safe": "value"}
                        if environment_checkpoint_key == "environment_factory_reconnect"
                        else "safe-session"
                    ),
                }
            },
            redactor=SecretRedactor("status"),
        )
    assert not durable_value_contains_secret(
        {
            "session_operations": {
                "records": {
                    "safe-operation": {
                        "status": "completed",
                    }
                }
            }
        },
        redactor=SecretRedactor("status"),
    )


def test_active_invocation_profile_unknown_extension_remains_secret_scanned() -> None:
    from cayu.runtime._checkpoint_redaction import durable_value_contains_secret
    from cayu.vaults import SecretRedactor

    secret = "active-profile-extension-secret"
    checkpoint = {
        execution_profiles_module.ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY: {
            "unknown_extension": {
                "session_id": secret,
            }
        }
    }

    assert durable_value_contains_secret(
        checkpoint,
        redactor=SecretRedactor(secret),
    )


def test_approval_resolution_digest_is_typed_private_checkpoint_state() -> None:
    from cayu.runtime import _approval_support as approval_support
    from cayu.runtime._checkpoint_redaction import durable_value_contains_secret
    from cayu.runtime.approvals import ToolApprovalDecision, ToolApprovalRequest
    from cayu.vaults import SecretRedactor

    request = ToolApprovalRequest(
        session_id="session-1",
        approval_id="approval-1",
        tool_round_id=f"tround_{'3' * 32}",
        tool_call_id="call-1",
        decision=ToolApprovalDecision.APPROVE,
    )
    digest = approval_support.approval_resolution_request_digest(request)
    checkpoint = {
        approval_support.APPROVAL_RESOLUTION_INTENT_CHECKPOINT_KEY: {
            "approval_id": request.approval_id,
            "tool_call_id": request.tool_call_id,
            "tool_round_id": request.tool_round_id,
            "model_step_id": f"mstep_{'1' * 32}",
            "model_attempt_id": f"matt_{'2' * 32}",
            "decision": request.decision.value,
            "resolution_request_digest": digest,
        }
    }

    for secret in ("resolution", "digest", next(iter(digest))):
        assert not durable_value_contains_secret(
            checkpoint,
            redactor=SecretRedactor(secret),
        )


@pytest.mark.parametrize(
    "checkpoint_kind",
    ["approval", "user_input", "tool_round"],
)
@pytest.mark.parametrize("consume_on_rejection", [False, True])
def test_malformed_legacy_pending_checkpoint_is_rejected_without_traceback_secret(
    checkpoint_kind: str,
    consume_on_rejection: bool,
) -> None:
    from cayu.runtime import _approval_support as approval_support
    from cayu.runtime import _tool_round_recovery as tool_round_recovery
    from cayu.runtime.user_input import (
        PENDING_USER_INPUT_CHECKPOINT_KEY,
        pending_user_input_from_checkpoint,
    )

    secret = "model"
    if checkpoint_kind == "approval":
        checkpoint = {
            approval_support.PENDING_TOOL_APPROVAL_CHECKPOINT_KEY: {
                "approval_id": {secret: "safe"},
            }
        }
        load = approval_support.pending_approval_from_checkpoint
    elif checkpoint_kind == "user_input":
        checkpoint = {
            PENDING_USER_INPUT_CHECKPOINT_KEY: {
                "question": {secret: "safe"},
            }
        }
        load = pending_user_input_from_checkpoint
    else:
        checkpoint = {
            tool_round_recovery.PENDING_TOOL_ROUND_CHECKPOINT_KEY: {
                "agent_name": {secret: "safe"},
            }
        }
        load = tool_round_recovery.pending_tool_round_from_checkpoint

    original_checkpoint = copy.deepcopy(checkpoint)
    with pytest.raises(ValueError, match="invalid and cannot be executed") as exc_info:
        load(
            checkpoint,
            redactor=SecretRedactor(secret),
            consume_on_rejection=consume_on_rejection,
        )

    assert checkpoint == ({} if consume_on_rejection else original_checkpoint)
    assert secret not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    _assert_cayu_traceback_does_not_retain_text(exc_info.value, secret)


def test_short_secret_substring_in_typed_thinking_key_does_not_block_provider() -> None:
    from cayu.core.thinking import ThinkingConfig
    from cayu.vaults import SecretRedactor

    provider = FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})])
    app = CayuApp(
        secret_redactor=SecretRedactor("fort"),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_typed_thinking_key_collision",
                messages=[Message.text("user", "think")],
                thinking=ThinkingConfig(effort="low"),
            ),
        )
    )

    assert events[-1].type is EventType.SESSION_COMPLETED
    assert provider.requests[0].options["thinking"]["effort"] == "low"


def test_short_secret_substring_in_typed_structured_output_key_allows_pending_round() -> None:
    from cayu.runtime import _runtime_records as runtime_records
    from cayu.runtime import _tool_round_recovery as tool_round_recovery
    from cayu.runtime.structured_output import StructuredOutputStrategy
    from cayu.vaults import SecretRedactor

    checkpoint, pending_round = tool_round_recovery.checkpoint_with_pending_tool_round(
        None,
        agent_name="assistant",
        environment_name=None,
        task_id=None,
        tool_calls=[
            runtime_records.ToolCallRequest(
                id="call-safe",
                name="safe-tool",
                arguments={},
            )
        ],
        policy_outcomes=None,
        structured_output=StructuredOutputSpec(
            name="answer",
            json_schema={"type": "object"},
            strategy=StructuredOutputStrategy.NATIVE,
        ),
        tool_round_identity=tool_round_identity(),
        redactor=SecretRedactor("rate"),
    )

    assert pending_round.structured_output is not None
    assert pending_round.structured_output.strategy is StructuredOutputStrategy.NATIVE
    assert checkpoint["pending_tool_round"]["structured_output"]["strategy"] == "native"


def test_json_schema_keyword_overlap_allows_pending_round_checkpoint_and_reload() -> None:
    from cayu.runtime import _runtime_records as runtime_records
    from cayu.runtime import _tool_round_recovery as tool_round_recovery
    from cayu.runtime.structured_output import StructuredOutputStrategy
    from cayu.vaults import SecretRedactor

    redactor = SecretRedactor("typ")
    checkpoint, pending_round = tool_round_recovery.checkpoint_with_pending_tool_round(
        None,
        agent_name="assistant",
        environment_name=None,
        task_id=None,
        tool_calls=[
            runtime_records.ToolCallRequest(
                id="call-safe",
                name="safe-tool",
                arguments={},
            )
        ],
        policy_outcomes=None,
        structured_output=StructuredOutputSpec(
            name="answer",
            json_schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
            },
            strategy=StructuredOutputStrategy.NATIVE,
        ),
        tool_round_identity=tool_round_identity(),
        redactor=redactor,
    )

    restored = tool_round_recovery.pending_tool_round_from_checkpoint(
        checkpoint,
        redactor=redactor,
    )

    assert restored == pending_round
    assert checkpoint["pending_tool_round"]["structured_output"]["json_schema"] == {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
    }


def test_schema_aware_checkpoint_still_rejects_data_owned_schema_keys() -> None:
    from cayu.runtime import _runtime_records as runtime_records
    from cayu.runtime import _tool_round_recovery as tool_round_recovery
    from cayu.vaults import SecretRedactor

    with pytest.raises(ValueError, match="contains a workload secret"):
        tool_round_recovery.checkpoint_with_pending_tool_round(
            None,
            agent_name="assistant",
            environment_name=None,
            task_id=None,
            tool_calls=[
                runtime_records.ToolCallRequest(
                    id="call-safe",
                    name="safe-tool",
                    arguments={},
                )
            ],
            policy_outcomes=None,
            structured_output=StructuredOutputSpec(
                name="answer",
                json_schema={
                    "type": "object",
                    "properties": {"typ": {"type": "string"}},
                },
            ),
            tool_round_identity=tool_round_identity(),
            redactor=SecretRedactor("typ"),
        )


def test_short_secret_substring_in_tool_round_id_key_allows_pending_round() -> None:
    from cayu.runtime import _runtime_records as runtime_records
    from cayu.runtime import _tool_round_recovery as tool_round_recovery
    from cayu.vaults import SecretRedactor

    identity = tool_round_identity()
    checkpoint, pending_round = tool_round_recovery.checkpoint_with_pending_tool_round(
        None,
        agent_name="assistant",
        environment_name=None,
        task_id=None,
        tool_calls=[
            runtime_records.ToolCallRequest(
                id="call-safe",
                name="safe-tool",
                arguments={},
            )
        ],
        policy_outcomes=None,
        structured_output=None,
        tool_round_identity=identity,
        redactor=SecretRedactor("id"),
    )

    assert checkpoint["pending_tool_round"]["tool_round_id"] == pending_round.tool_round_id
    assert pending_round.tool_round_id == identity.tool_round_id


def test_business_approval_rejects_legacy_secret_before_routing_or_traceback_exposure() -> None:
    from cayu.runtime import _approval_support as approval_support
    from cayu.runtime.approvals import PendingToolApproval, PendingToolCallApproval
    from cayu.runtime.business_approvals import (
        BusinessApprovalOutcome,
        business_approval_routing_metadata,
        resolve_business_approval,
    )
    from cayu.vaults import SecretRedactor

    async def run() -> None:
        secret = "legacy-business-approval-routing-secret-canary"
        session_id = "sess_legacy_business_approval_secret"
        store = InMemorySessionStore()
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        pending = PendingToolApproval(
            **tool_round_identity().payload(),
            approval_id="approval-safe",
            tool_call_id="call-safe",
            tool_name="safe-tool",
            agent_name="assistant",
            publish_arguments=True,
            metadata=business_approval_routing_metadata(
                required_tier="team",
                chain=("team",),
                metadata={"routing_note": secret},
            ),
            tool_calls=[
                PendingToolCallApproval(
                    tool_call_id="call-safe",
                    tool_name="safe-tool",
                )
            ],
        )
        await store.transform_checkpoint(
            session_id,
            lambda _session, _checkpoint: {
                approval_support.PENDING_TOOL_APPROVAL_CHECKPOINT_KEY: pending.model_dump(
                    mode="json"
                )
            },
        )
        app = CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )

        with pytest.raises(ValueError, match="workload secret") as exc_info:
            await resolve_business_approval(
                app,
                session_id=session_id,
                approval_id=pending.approval_id,
                approver_id="approver-safe",
                approver_tier="team",
                outcome=BusinessApprovalOutcome.APPROVED,
            )

        assert secret not in str(exc_info.value)
        _assert_cayu_traceback_does_not_retain_text(exc_info.value, secret)

    asyncio.run(run())


def test_parallel_operator_interrupt_keeps_each_invocation_secret_scope() -> None:
    from cayu.vaults import REDACTED_SECRET

    secret_values = {
        "secret_a": "parallel-cancel-secret-a-canary",
        "secret_b": "parallel-cancel-secret-b-canary",
    }

    class ParallelSecretTool(Tool):
        spec = ToolSpec(
            name="parallel_secret_cancel",
            description="Resolve one call-scoped secret and wait.",
            input_schema={
                "type": "object",
                "properties": {"secret_name": {"type": "string"}},
                "required": ["secret_name"],
            },
        )

        def __init__(self) -> None:
            self.arrivals = 0
            self.all_started: asyncio.Event | None = None

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            assert ctx.vault is not None
            assert self.all_started is not None
            secret = await ctx.vault.resolve(SecretRef(name=args["secret_name"]))
            raw_secret = secret.value.get_secret_value()
            self.arrivals += 1
            if self.arrivals == 2:
                self.all_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as exc:
                raise RunnerCancelledError(
                    artifacts=[
                        {
                            "type": "cayu.runner_cleanup.v1",
                            "status": "failed",
                            "stderr": raw_secret,
                        }
                    ]
                ) from exc
            return ToolResult(content="unexpected")

    tool = ParallelSecretTool()
    store = InMemorySessionStore()
    app = CayuApp(
        session_store=store,
        enable_logging=False,
        max_parallel_tool_calls=2,
    )
    app.register_provider(
        FakeProvider(
            [
                ModelStreamEvent.tool_call(
                    id="call_a",
                    name="parallel_secret_cancel",
                    arguments={"secret_name": "secret_a"},
                ),
                ModelStreamEvent.tool_call(
                    id="call_b",
                    name="parallel_secret_cancel",
                    arguments={"secret_name": "secret_b"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        ),
        default=True,
    )
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            vault=StaticVault(secret_values),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
    )

    async def scenario():
        tool.all_started = asyncio.Event()
        run_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_parallel_dynamic_secret_cancel",
                    messages=[Message.text("user", "use both")],
                ),
            )
        )
        await tool.all_started.wait()
        _ = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id="sess_parallel_dynamic_secret_cancel",
                    reason="operator stop",
                )
            )
        ]
        await run_task
        return await store.load_events("sess_parallel_dynamic_secret_cancel")

    stored_events = asyncio.run(scenario())
    failed_events = [event for event in stored_events if event.type is EventType.TOOL_CALL_FAILED]

    assert [event.payload["tool_call_id"] for event in failed_events] == [
        "call_a",
        "call_b",
    ]
    for event in failed_events:
        rendered = repr(event.payload)
        assert all(secret not in rendered for secret in secret_values.values())
        assert REDACTED_SECRET in rendered
