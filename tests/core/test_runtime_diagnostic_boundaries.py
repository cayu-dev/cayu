from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from tests.core._workload_secret_support import FakeProvider

from cayu import (
    EXECUTION_PROFILE_FINGERPRINT_FIELD,
    AgentSpec,
    BoundWorkspace,
    CayuApp,
    DurableValueError,
    Environment,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryResult,
    EnvironmentSpec,
    Event,
    EventType,
    InMemoryKnowledgeStore,
    InMemorySessionStore,
    InMemoryTaskStore,
    KnowledgeAccessScope,
    KnowledgeInjectionPolicy,
    Message,
    ModelStreamEvent,
    RunRequest,
    RuntimeHook,
    RuntimeHookContext,
    ScriptedModelProvider,
    SessionStatus,
    TaskCreate,
    TaskStatus,
    Tool,
    ToolCallHookContext,
    ToolContext,
    ToolEffect,
    ToolResult,
    ToolSpec,
    WorkspaceBinding,
)
from cayu._exception_groups import (
    exception_cause,
    exception_context,
    exception_tree_contains,
    set_exception_cause,
)
from cayu.environments import EnvironmentFactory, EnvironmentFactoryRequest
from cayu.runtime._binding_cleanup import (
    BindingCleanupStatus,
    BindingFinalizeFailure,
    BindingFinalizeStatus,
    append_binding_finalize_cancellation,
    attach_binding_finalize_safe_payload,
    binding_cleanup_status,
    binding_finalize_cancellation,
    binding_finalize_explicit_cancellation,
    binding_finalize_fatal_signal,
    binding_finalize_safe_payload,
    binding_finalize_status,
    is_containable_cleanup_error,
    record_binding_cleanup_failure,
)
from cayu.runtime._diagnostics import MAX_DIAGNOSTIC_UTF8_BYTES
from cayu.runtime._environment_lifecycle import (
    FAILURE_DIAGNOSTIC_TEXT_MAX_BYTES,
    _release_unclaimed_factory_result,
    exception_failure_payload,
)
from cayu.runtime._event_projection import PRIVATE_EVENT_AUTHORITY
from cayu.runtime._tool_round_executor import _parallel_tool_round_exception
from cayu.runtime.egress import _contains_timeout, _split_cleanup_cancellation
from cayu.vaults import REDACTED_SECRET, SecretRedactor


async def _collect(app: CayuApp, request: RunRequest) -> list[Event]:
    return [event async for event in app.run(request)]


class _RecordingEnvironmentFactory(EnvironmentFactory):
    def __init__(
        self,
        environment: Environment,
        *,
        reconnect_metadata: dict[str, Any] | None = None,
        release: (Callable[[EnvironmentFactoryReleaseAction], Awaitable[None]] | None) = None,
    ) -> None:
        self.environment = environment
        self.reconnect_metadata = reconnect_metadata or {}
        self.release = release
        self.requests: list[EnvironmentFactoryRequest] = []

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        self.requests.append(request)
        return EnvironmentFactoryResult(
            environment=self.environment,
            reconnect_metadata=self.reconnect_metadata,
            release=self.release,
        )


def _deep_group(leaf: BaseException, *, depth: int = 1_500) -> BaseExceptionGroup:
    failure: BaseException = leaf
    for _ in range(depth):
        failure = BaseExceptionGroup("nested failure", [failure])
    assert isinstance(failure, BaseExceptionGroup)
    return failure


async def _run_binding_finalize_failure(
    error: Exception,
    *,
    session_id: str,
) -> tuple[list[Event], object]:
    class FailingFinalizeBinding(WorkspaceBinding):
        async def bind(self, workspace, runner, **_kwargs) -> BoundWorkspace:
            return BoundWorkspace(
                workspace=workspace,
                source_workspace=workspace,
                runner=runner,
                path="/bound",
            )

        async def finalize(self, bound, *, outcome=None, metadata=None):
            del bound, outcome, metadata
            raise error

    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            ]
        ),
        default=True,
    )
    app.register_environment(
        Environment(
            EnvironmentSpec(name="binding-finalize-boundary"),
            binding=FailingFinalizeBinding(),
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="scripted-model"))
    events = await _collect(
        app,
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=[Message.text("user", "run")],
        ),
    )
    return events, await store.load(session_id)


def _run_linked_task_failure(error: Exception) -> tuple[list[Event], object, object]:
    class FailingSessionStore(InMemorySessionStore):
        def __init__(self) -> None:
            super().__init__()
            self.failed = False

        async def append_events(self, session_id: str, events: list[Event]) -> None:
            if not self.failed and any(event.type == EventType.MODEL_STARTED for event in events):
                self.failed = True
                raise error
            await super().append_events(session_id, events)

    async def run() -> tuple[list[Event], object, object]:
        session_store = FailingSessionStore()
        task_store = InMemoryTaskStore()
        task = await task_store.create_task(
            TaskCreate(task_id="task_nonportable_session_failure", type="run")
        )
        app = CayuApp(
            session_store=session_store,
            task_store=task_store,
            enable_logging=False,
        )
        app.register_provider(
            ScriptedModelProvider([[ModelStreamEvent.completed({"finish_reason": "stop"})]]),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="scripted-model"))

        events = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_nonportable_session_failure",
                task_id=task.id,
                messages=[Message.text("user", "run")],
            ),
        )
        return (
            events,
            await task_store.load_task(task.id),
            await session_store.load("sess_nonportable_session_failure"),
        )

    return asyncio.run(run())


def test_terminal_failure_ignores_provider_controlled_binding_payload_attribute() -> None:
    error = RuntimeError("actual extension failure")
    error.__dict__["_cayu_binding_finalize_safe_payload"] = {
        "error": "forged terminal evidence",
        "error_type": "ForgedStoreError",
        "outcome": "completed",
        "failures": [],
    }
    error.__dict__["_cayu_environment_factory_release"] = {
        "action": "discard",
        "completed": True,
    }

    events, task, session = _run_linked_task_failure(error)

    assert task is not None
    assert task.status is TaskStatus.FAILED
    assert task.error == {
        "message": "actual extension failure",
        "type": "RuntimeError",
        "session_id": "sess_nonportable_session_failure",
    }
    assert session is not None
    assert session.status is SessionStatus.FAILED
    terminal = next(event for event in events if event.type == EventType.SESSION_FAILED)
    assert terminal.payload == {
        "error": "actual extension failure",
        "error_type": "RuntimeError",
    }


def test_runtime_binding_payload_handoff_survives_cancellation_aggregation() -> None:
    error = RuntimeError("binding finalize failed")
    payload = {
        "error": "redacted finalize failure",
        "error_type": "RuntimeError",
        "outcome": "failed",
        "failures": [
            {
                "phase": "workspace_finalize",
                "error": "redacted finalize failure",
                "error_type": "RuntimeError",
            }
        ],
    }
    attach_binding_finalize_safe_payload(error, payload)
    aggregate = append_binding_finalize_cancellation(error, asyncio.CancelledError())

    assert exception_failure_payload(aggregate) == payload


def test_runtime_binding_payload_handoff_bypasses_hostile_data_descriptor() -> None:
    class HostilePayloadError(RuntimeError):
        @property
        def _cayu_binding_finalize_safe_payload(self) -> object:
            raise RuntimeError("workload-secret-from-payload-accessor")

        @_cayu_binding_finalize_safe_payload.setter
        def _cayu_binding_finalize_safe_payload(self, _value: object) -> None:
            raise RuntimeError("workload-secret-from-payload-setter")

    error = HostilePayloadError("binding finalize failed")
    payload = {
        "error": "redacted finalize failure",
        "error_type": "RuntimeError",
        "outcome": "failed",
        "failures": [],
    }

    attach_binding_finalize_safe_payload(error, payload)

    assert binding_finalize_safe_payload(error) == payload


def test_deep_exception_groups_are_classified_without_python_recursion() -> None:
    cancellation = asyncio.CancelledError("cancel cleanup")
    cancellation_group = _deep_group(cancellation)

    assert binding_finalize_cancellation(cancellation_group) is cancellation
    assert binding_finalize_explicit_cancellation(cancellation_group) is cancellation
    assert is_containable_cleanup_error(cancellation_group) is True

    failure = RuntimeError("binding finalize failed")
    payload = {
        "error": "redacted finalize failure",
        "error_type": "RuntimeError",
        "outcome": "failed",
        "failures": [],
    }
    attach_binding_finalize_safe_payload(failure, payload)
    failure_group = _deep_group(failure)

    assert exception_failure_payload(failure_group) == payload
    assert _parallel_tool_round_exception(failure_group) is failure


@pytest.mark.parametrize("stage", ["request", "completion"])
def test_grouped_billing_hook_failure_is_detached_and_terminalized(stage: str) -> None:
    canary = f"provider-secret-grouped-billing-{stage}"

    def grouped_failure() -> BaseExceptionGroup:
        return BaseExceptionGroup(
            f"billing hook failed near {canary}",
            [
                asyncio.CancelledError(f"billing cancelled near {canary}"),
                RuntimeError(f"billing cleanup failed near {canary}"),
            ],
        )

    class GroupedBillingProvider(ScriptedModelProvider):
        async def billing_identity_for_request(self, request):
            del request
            if stage == "request":
                raise grouped_failure()
            return None

        def billing_identity_for_completion(self, identity, payload):
            del payload
            if stage == "completion":
                raise grouped_failure()
            return identity

    store = InMemorySessionStore()
    provider = GroupedBillingProvider(
        [
            [
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 7,
                            "output_tokens": 3,
                            "total_tokens": 10,
                        },
                    }
                )
            ]
        ]
    )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="scripted-model"))

    events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=f"sess_grouped_billing_{stage}",
                messages=[Message.text("user", "run")],
            ),
        )
    )
    session = asyncio.run(store.load(f"sess_grouped_billing_{stage}"))

    assert canary not in repr(events)
    assert session is not None
    assert session.status is SessionStatus.FAILED
    if stage == "request":
        failed = next(event for event in events if event.type is EventType.MODEL_ERROR)
        assert failed.payload["stage"] == "billing_identity_for_request"
    else:
        completed = next(event for event in events if event.type is EventType.MODEL_COMPLETED)
        assert completed.payload["completion_outcome"] == ("billing_identity_resolution_failed")
        assert completed.payload["usage_metrics"]["input_tokens"] == 7
        assert completed.payload["usage_metrics"]["output_tokens"] == 3
        assert completed.payload["usage_metrics"]["total_tokens"] == 10
        usage = asyncio.run(app.get_session_usage(f"sess_grouped_billing_{stage}"))
        assert usage.model_steps == 1
        assert usage.usage.input_tokens == 7
        assert usage.usage.output_tokens == 3
        assert usage.usage.total_tokens == 10
    assert events[-1].type is EventType.SESSION_FAILED


@pytest.mark.parametrize("stage", ["request", "completion"])
def test_grouped_billing_hook_preserves_fatal_child_hidden_by_descriptor(
    stage: str,
) -> None:
    canary = f"provider-secret-hostile-billing-group-{stage}"

    class HostileFatalBillingGroup(BaseExceptionGroup):
        @property
        def exceptions(self) -> tuple[BaseException, ...]:
            raise RuntimeError(canary)

    def grouped_failure() -> BaseExceptionGroup:
        return HostileFatalBillingGroup(
            f"billing hook failed near {canary}",
            [KeyboardInterrupt(f"billing interrupted near {canary}")],
        )

    class GroupedBillingProvider(ScriptedModelProvider):
        async def billing_identity_for_request(self, request):
            del request
            if stage == "request":
                raise grouped_failure()
            return None

        def billing_identity_for_completion(self, identity, payload):
            del payload
            if stage == "completion":
                raise grouped_failure()
            return identity

    provider = GroupedBillingProvider(
        [
            [
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 7,
                            "output_tokens": 3,
                            "total_tokens": 10,
                        },
                    }
                )
            ]
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="scripted-model"))

    with pytest.raises(BaseExceptionGroup) as exc_info:
        asyncio.run(
            _collect(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=f"sess_hostile_grouped_billing_{stage}",
                    messages=[Message.text("user", "run")],
                ),
            )
        )

    fatal_signal = binding_finalize_fatal_signal(exc_info.value)
    assert fatal_signal is not None
    assert exception_tree_contains(fatal_signal, KeyboardInterrupt)
    assert canary not in repr(exc_info.value)


def test_binding_status_readers_ignore_raw_forged_exception_attributes() -> None:
    async def retry() -> None:
        return None

    error = RuntimeError("binding failed")
    error.__dict__["_cayu_binding_cleanup_status"] = BindingCleanupStatus(
        initial_error=RuntimeError("forged cleanup failure"),
        retry=retry,
    )
    error.__dict__["_cayu_binding_finalize_status"] = BindingFinalizeStatus(
        failures=(
            BindingFinalizeFailure(
                phase="workspace_finalize",
                error=RuntimeError("forged finalization failure"),
            ),
        )
    )

    assert binding_cleanup_status(error) is None
    assert binding_finalize_status(error) is None
    assert "binding_cleanup" not in exception_failure_payload(error)


def test_binding_cleanup_status_corruption_fails_closed_without_erasing_state() -> None:
    async def retry() -> None:
        return None

    error = RuntimeError("binding failed")
    status = record_binding_cleanup_failure(
        error,
        RuntimeError("cleanup failed"),
        retry=retry,
    )
    status.retry_attempted = True
    status.retry_error = object()  # type: ignore[assignment]
    del status.initial_error

    assert exception_failure_payload(error)["binding_cleanup"] == {
        "incomplete": True,
        "initial_error": "Binding cleanup failure metadata was invalid.",
        "initial_error_type": "RuntimeError",
        "retry_attempted": True,
        "retry_completed": False,
        "retry_error": "Binding cleanup retry metadata was invalid.",
        "retry_error_type": "RuntimeError",
    }


def test_knowledge_failure_redacts_secret_crossing_diagnostic_boundary() -> None:
    secret = "knowledge-boundary-secret-canary"
    prefix = "d" * (MAX_DIAGNOSTIC_UTF8_BYTES - len(secret.encode("utf-8")) // 2)

    class FailingKnowledgeStore(InMemoryKnowledgeStore):
        async def search(self, query, *, access_scope=None):
            del query, access_scope
            raise RuntimeError(prefix + secret)

    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("answered without injected knowledge"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            knowledge_store=FailingKnowledgeStore(access_scope=KnowledgeAccessScope.privileged()),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=KnowledgeInjectionPolicy(fail_open=True),
    )

    events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_knowledge_boundary_redaction",
                messages=[Message.text("user", "search")],
            ),
        )
    )

    failed_event = next(
        event for event in events if event.type == EventType.KNOWLEDGE_SEARCH_FAILED
    )
    rendered = str(failed_event.payload)
    assert secret not in rendered
    assert secret[: len(secret) // 2] not in rendered
    assert len(failed_event.payload["error"].encode("utf-8")) <= MAX_DIAGNOSTIC_UTF8_BYTES
    assert len(provider.requests) == 1


@pytest.mark.parametrize("failure_kind", ["factory", "binding"])
def test_cayu_app_redacts_environment_setup_failure_payloads(failure_kind: str) -> None:
    secret = f"environment-{failure_kind}-boundary-canary"

    class SecretFailingFactory(_RecordingEnvironmentFactory):
        async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
            self.requests.append(request)
            raise RuntimeError(f"factory failed with {secret}")

    class SecretFailingBinding(WorkspaceBinding):
        async def bind(self, workspace, runner, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError(f"binding failed with {secret}")

        async def finalize(self, bound, *, outcome=None, metadata=None):  # type: ignore[no-untyped-def]
            raise AssertionError("finalize should not run")

    async def run() -> tuple[list[Event], list[Event]]:
        store = InMemorySessionStore()
        app = CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_provider(FakeProvider([]), default=True)
        if failure_kind == "factory":
            app.register_environment_factory(
                EnvironmentSpec(name="dynamic"),
                SecretFailingFactory(Environment(EnvironmentSpec(name="dynamic"))),
                default=True,
            )
        else:
            app.register_environment(
                Environment(
                    EnvironmentSpec(name="local"),
                    binding=SecretFailingBinding(),
                ),
                default=True,
            )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = f"sess_environment_{failure_kind}_redaction"
        emitted = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "run")],
            ),
        )
        return emitted, await store.load_events(session_id)

    emitted, persisted = asyncio.run(run())
    serialized = str([event.model_dump(mode="json") for event in [*emitted, *persisted]])
    assert secret not in serialized
    assert REDACTED_SECRET in serialized


def test_cayu_app_rejects_secret_bearing_factory_reconnect_metadata_before_checkpoint() -> None:
    secret = "factory-reconnect-boundary-canary"

    async def run():
        release_actions: list[EnvironmentFactoryReleaseAction] = []

        async def release(action: EnvironmentFactoryReleaseAction) -> None:
            release_actions.append(action)

        store = InMemorySessionStore()
        factory = _RecordingEnvironmentFactory(
            Environment(EnvironmentSpec(name="dynamic")),
            reconnect_metadata={"allocation_id": f"sandbox-{secret}"},
            release=release,
        )
        provider = FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})])
        app = CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="dynamic"),
            factory,
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_factory_reconnect_redaction",
                messages=[Message.text("user", "run")],
            ),
        )
        return (
            events,
            await store.load_checkpoint("sess_factory_reconnect_redaction"),
            release_actions,
            provider.requests,
        )

    events, checkpoint, release_actions, provider_requests = asyncio.run(run())

    assert events[-1].type is EventType.SESSION_FAILED
    assert secret not in str([event.model_dump(mode="json") for event in events])
    assert secret not in str(checkpoint)
    assert release_actions == [EnvironmentFactoryReleaseAction.DISCARD]
    assert provider_requests == []


def test_cayu_app_redacts_and_bounds_task_store_failure_details() -> None:
    secret = "task-store-failure-canary"
    error_message = f"provider rejected {secret} " + ("x" * 5000)
    session_store = InMemorySessionStore()
    task_store = InMemoryTaskStore()
    provider = FakeProvider([ModelStreamEvent.error(error_message)])

    async def run():
        await task_store.create_task(TaskCreate(task_id="task_redacted_failure", type="respond"))
        app = CayuApp(
            session_store=session_store,
            task_store=task_store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        events = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_task_redacted_failure",
                task_id="task_redacted_failure",
                messages=[Message.text("user", "hi")],
            ),
        )
        return events, await task_store.load_task("task_redacted_failure")

    events, task = asyncio.run(run())

    assert task is not None
    assert task.status is TaskStatus.FAILED
    assert task.error is not None
    assert secret not in str(task.error)
    assert REDACTED_SECRET in task.error["message"]
    assert len(task.error["message"].encode()) <= FAILURE_DIAGNOSTIC_TEXT_MAX_BYTES
    assert secret not in str([event.model_dump(mode="json") for event in events])


@pytest.mark.parametrize(
    "rejected_text",
    [
        "finalize failed\u0000workload-secret",
        "finalize failed\ud800workload-secret",
    ],
    ids=["nul", "surrogate"],
)
def test_binding_finalize_nonportable_diagnostic_remains_terminal(
    rejected_text: str,
) -> None:
    events, session = asyncio.run(
        _run_binding_finalize_failure(
            RuntimeError(rejected_text),
            session_id="sess_binding_finalize_nonportable",
        )
    )

    assert session is not None
    assert session.status is SessionStatus.COMPLETED
    assert [event.type for event in events[-3:]] == [
        EventType.ENVIRONMENT_BINDING_FINALIZE_STARTED,
        EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED,
        EventType.SESSION_COMPLETED,
    ]
    expected = {
        "error": "Binding finalization failed with a non-portable diagnostic.",
        "error_type": "RuntimeError",
        "outcome": "completed",
        "failures": [
            {
                "phase": "workspace_finalize",
                "error": "Binding finalization failed with a non-portable diagnostic.",
                "error_type": "RuntimeError",
            }
        ],
    }
    assert {
        key: events[-2].payload[key] for key in ("error", "error_type", "outcome", "failures")
    } == expected
    assert events[-1].payload["binding_finalize_error"] == expected
    assert "workload-secret" not in repr(events[-2].payload) + repr(events[-1].payload)


def test_binding_finalize_hostile_exception_access_cannot_escape() -> None:
    class HostileFinalizeError(RuntimeError):
        def __getattribute__(self, name: str):
            if name == "_cayu_binding_finalize_status":
                raise RuntimeError("workload-secret-from-finalize-accessor")
            return super().__getattribute__(name)

        def __str__(self) -> str:
            raise RuntimeError("workload-secret-from-finalize-rendering")

    events, session = asyncio.run(
        _run_binding_finalize_failure(
            HostileFinalizeError("ignored"),
            session_id="sess_binding_finalize_hostile",
        )
    )

    assert session is not None
    assert session.status is SessionStatus.COMPLETED
    failed = next(
        event for event in events if event.type is EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED
    )
    terminal = events[-1]
    assert failed.payload["error"] == "HostileFinalizeError: binding finalization failed"
    assert failed.payload["error_type"] == "HostileFinalizeError"
    assert terminal.type is EventType.SESSION_COMPLETED
    assert terminal.payload["binding_finalize_error"]["error"] == (
        "HostileFinalizeError: binding finalization failed"
    )
    assert "workload-secret" not in repr(failed.payload) + repr(terminal.payload)


def test_binding_finalize_classifiers_bypass_hostile_group_and_causal_accessors() -> None:
    class HostileGroup(BaseExceptionGroup):
        @property
        def exceptions(self) -> tuple[BaseException, ...]:
            raise RuntimeError("workload-secret-from-exceptions-accessor")

        def subgroup(self, _condition):
            raise RuntimeError("workload-secret-from-subgroup")

        def split(self, _condition):
            raise RuntimeError("workload-secret-from-split")

    cancellation = asyncio.CancelledError("caller cancelled")
    grouped = HostileGroup("hostile cleanup", [cancellation, RuntimeError("cleanup failed")])

    assert binding_finalize_explicit_cancellation(grouped) is cancellation
    assert binding_finalize_cancellation(grouped) is cancellation
    assert is_containable_cleanup_error(grouped) is True
    assert binding_finalize_fatal_signal(grouped) is None

    class HostileCausalError(RuntimeError):
        @property
        def __cause__(self) -> BaseException | None:
            raise RuntimeError("workload-secret-from-cause-accessor")

        @__cause__.setter
        def __cause__(self, _value: BaseException | None) -> None:
            raise RuntimeError("workload-secret-from-cause-setter")

        @property
        def __context__(self) -> BaseException | None:
            raise RuntimeError("workload-secret-from-context-accessor")

        @__context__.setter
        def __context__(self, _value: BaseException | None) -> None:
            raise RuntimeError("workload-secret-from-context-setter")

    linked = HostileCausalError("linked cleanup failed")
    assert set_exception_cause(linked, cancellation) is True
    assert exception_cause(linked) is cancellation
    assert binding_finalize_cancellation(linked) is cancellation

    contextual = HostileCausalError("contextual cleanup failed")
    BaseException.__context__.__set__(contextual, cancellation)
    assert exception_context(contextual) is cancellation


def test_egress_cleanup_classification_bypasses_hostile_group_methods() -> None:
    class HostileCleanupGroup(BaseExceptionGroup):
        def __getattribute__(self, name: str):
            if name == "exceptions":
                raise RuntimeError("workload-secret-from-exceptions-accessor")
            return super().__getattribute__(name)

        def subgroup(self, _condition):
            raise RuntimeError("workload-secret-from-subgroup")

        def split(self, _condition):
            raise RuntimeError("workload-secret-from-split")

    cancellation = asyncio.CancelledError("caller cancelled")
    timeout = TimeoutError("cleanup timed out")
    grouped = HostileCleanupGroup("hostile cleanup", [cancellation, timeout])

    classified_cancellation, classified_error = _split_cleanup_cancellation(grouped)

    assert classified_cancellation is cancellation
    assert classified_error is timeout
    assert _contains_timeout(grouped) is True


def test_binding_finalize_hostile_exception_group_remains_terminal() -> None:
    class HostileFinalizeGroup(ExceptionGroup):
        def __getattribute__(self, name: str):
            if name in {"exceptions", "__cause__", "__context__"}:
                raise RuntimeError(f"workload-secret-from-{name}-accessor")
            return super().__getattribute__(name)

        def subgroup(self, _condition):
            raise RuntimeError("workload-secret-from-subgroup")

        def split(self, _condition):
            raise RuntimeError("workload-secret-from-split")

    events, session = asyncio.run(
        _run_binding_finalize_failure(
            HostileFinalizeGroup("hostile finalize group", [RuntimeError("cleanup failed")]),
            session_id="sess_binding_finalize_hostile_group",
        )
    )

    assert session is not None
    assert session.status is SessionStatus.COMPLETED
    failed = next(
        event for event in events if event.type is EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED
    )
    terminal = events[-1]
    assert failed.payload["error"] == "hostile finalize group (1 sub-exception)"
    assert failed.payload["error_type"] == "HostileFinalizeGroup"
    assert terminal.type is EventType.SESSION_COMPLETED
    assert terminal.payload["binding_finalize_error"]["error"] == failed.payload["error"]
    assert "workload-secret" not in repr(failed.payload) + repr(terminal.payload)


def test_binding_cleanup_nonportable_diagnostics_preserve_retry_evidence() -> None:
    class HostileBindError(RuntimeError):
        @property
        def _cayu_binding_cleanup_status(self) -> object:
            raise RuntimeError("workload-secret-from-cleanup-accessor")

        @_cayu_binding_cleanup_status.setter
        def _cayu_binding_cleanup_status(self, _value: object) -> None:
            raise RuntimeError("workload-secret-from-cleanup-setter")

    class HostileRetryError(RuntimeError):
        def __str__(self) -> str:
            raise RuntimeError("workload-secret-from-cleanup-rendering")

    class CleanupFailingBinding(WorkspaceBinding):
        async def bind(self, workspace, runner, **_kwargs):
            bind_error = HostileBindError("bind failed")

            async def retry() -> None:
                raise HostileRetryError("ignored")

            record_binding_cleanup_failure(
                bind_error,
                RuntimeError("initial cleanup failed\u0000workload-secret"),
                retry=retry,
            )
            raise bind_error

        async def finalize(self, bound, *, outcome=None, metadata=None):
            raise AssertionError("finalize should not run")

    async def run() -> tuple[list[Event], object]:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(ScriptedModelProvider([]), default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="binding-cleanup-boundary"),
                binding=CleanupFailingBinding(),
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="scripted-model"))
        events = await _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_binding_cleanup_nonportable",
                messages=[Message.text("user", "run")],
            ),
        )
        return events, await store.load("sess_binding_cleanup_nonportable")

    events, session = asyncio.run(run())

    assert session is not None
    assert session.status is SessionStatus.FAILED
    expected_cleanup = {
        "incomplete": True,
        "initial_error": "Binding cleanup failed with a non-portable diagnostic.",
        "initial_error_type": "RuntimeError",
        "initial_durable_value_error_code": "nul_character",
        "initial_durable_value_error_path": "$",
        "retry_attempted": True,
        "retry_completed": False,
        "retry_error": "HostileRetryError: binding cleanup retry failed",
        "retry_error_type": "HostileRetryError",
    }
    binding_failed = next(
        event for event in events if event.type is EventType.ENVIRONMENT_BINDING_FAILED
    )
    session_failed = next(event for event in events if event.type is EventType.SESSION_FAILED)
    assert binding_failed.payload["binding_cleanup"] == expected_cleanup
    assert session_failed.payload["binding_cleanup"] == expected_cleanup
    assert "workload-secret" not in repr(binding_failed.payload) + repr(session_failed.payload)


def test_environment_factory_release_rejects_nonportable_diagnostic_safely() -> None:
    async def release(_action: EnvironmentFactoryReleaseAction) -> None:
        raise RuntimeError("release failed\u0000workload-secret")

    async def run() -> dict:
        return await _release_unclaimed_factory_result(
            EnvironmentFactoryResult(
                environment=Environment(EnvironmentSpec(name="release-boundary")),
                release=release,
            ),
            action=EnvironmentFactoryReleaseAction.DISCARD,
            original_error=RuntimeError("factory setup failed"),
        )

    assert asyncio.run(run()) == {
        "action": "discard",
        "callback_provided": True,
        "completed": False,
        "error": "Environment factory release failed with a non-portable diagnostic.",
        "error_type": "RuntimeError",
        "durable_value_error_code": "nul_character",
        "durable_value_error_path": "$",
        "timeout_s": 15.0,
    }


def test_environment_factory_release_diagnostic_rendering_cannot_escape() -> None:
    class HostileReleaseError(RuntimeError):
        def __str__(self) -> str:
            raise RuntimeError("workload-secret-from-release-rendering")

    async def release(_action: EnvironmentFactoryReleaseAction) -> None:
        raise HostileReleaseError()

    async def run() -> tuple[dict, RuntimeError]:
        original_error = RuntimeError("factory setup failed")
        payload = await _release_unclaimed_factory_result(
            EnvironmentFactoryResult(
                environment=Environment(EnvironmentSpec(name="release-render-boundary")),
                release=release,
            ),
            action=EnvironmentFactoryReleaseAction.DISCARD,
            original_error=original_error,
        )
        return payload, original_error

    payload, original_error = asyncio.run(run())

    assert payload == {
        "action": "discard",
        "callback_provided": True,
        "completed": False,
        "error": "HostileReleaseError: environment factory release failed",
        "error_type": "HostileReleaseError",
        "timeout_s": 15.0,
    }
    rendered = repr(payload) + repr(getattr(original_error, "__notes__", ()))
    assert "workload-secret" not in rendered


def test_environment_factory_release_hostile_exception_group_cannot_escape() -> None:
    class HostileReleaseGroup(ExceptionGroup):
        def __getattribute__(self, name: str):
            if name in {"exceptions", "__cause__", "__context__"}:
                raise RuntimeError(f"workload-secret-from-{name}-accessor")
            return super().__getattribute__(name)

        def subgroup(self, _condition):
            raise RuntimeError("workload-secret-from-subgroup")

        def split(self, _condition):
            raise RuntimeError("workload-secret-from-split")

    async def release(_action: EnvironmentFactoryReleaseAction) -> None:
        raise HostileReleaseGroup("hostile release group", [RuntimeError("release failed")])

    async def run() -> tuple[dict, RuntimeError]:
        original_error = RuntimeError("factory setup failed")
        payload = await _release_unclaimed_factory_result(
            EnvironmentFactoryResult(
                environment=Environment(EnvironmentSpec(name="release-group-boundary")),
                release=release,
            ),
            action=EnvironmentFactoryReleaseAction.DISCARD,
            original_error=original_error,
        )
        return payload, original_error

    payload, original_error = asyncio.run(run())

    assert payload == {
        "action": "discard",
        "callback_provided": True,
        "completed": False,
        "error": "hostile release group (1 sub-exception)",
        "error_type": "HostileReleaseGroup",
        "timeout_s": 15.0,
    }
    rendered = repr(payload) + repr(getattr(original_error, "__notes__", ()))
    assert "workload-secret" not in rendered


def test_environment_factory_missing_release_ignores_hostile_note_override() -> None:
    class HostileNoteError(RuntimeError):
        def add_note(self, note: str) -> None:
            raise RuntimeError("workload-secret-from-release-note")

    async def run() -> dict:
        return await _release_unclaimed_factory_result(
            EnvironmentFactoryResult(
                environment=Environment(EnvironmentSpec(name="missing-release-boundary")),
            ),
            action=EnvironmentFactoryReleaseAction.PRESERVE,
            original_error=HostileNoteError("factory setup failed"),
        )

    assert asyncio.run(run()) == {
        "action": "preserve",
        "callback_provided": False,
        "completed": False,
        "error": "Durable factory result has no release callback.",
        "error_type": "MissingEnvironmentFactoryRelease",
    }


def test_environment_factory_fallback_release_ignores_hostile_note_override() -> None:
    class HostileNoteError(RuntimeError):
        def add_note(self, note: str) -> None:
            raise RuntimeError("workload-secret-from-fallback-note")

    class FailingCloseBinding(WorkspaceBinding):
        async def bind(self, workspace, runner, **_kwargs):
            raise AssertionError("bind should not run")

        async def finalize(self, bound, *, outcome=None, metadata=None):
            raise AssertionError("finalize should not run")

        async def close(self) -> None:
            raise RuntimeError("binding fallback close failed")

    async def run() -> dict:
        return await _release_unclaimed_factory_result(
            EnvironmentFactoryResult(
                environment=Environment(
                    EnvironmentSpec(name="fallback-release-boundary"),
                    binding=FailingCloseBinding(),
                ),
            ),
            action=EnvironmentFactoryReleaseAction.DISCARD,
            original_error=HostileNoteError("factory setup failed"),
        )

    assert asyncio.run(run()) == {
        "action": "discard",
        "callback_provided": False,
        "completed": False,
        "error": "binding: RuntimeError: binding fallback close failed",
        "error_type": "RuntimeError",
    }


@pytest.mark.parametrize(
    ("rejected_text", "error_code"),
    [
        ("store failure\u0000after task start", "nul_character"),
        ("store failure\ud800after task start", "unicode_surrogate"),
    ],
)
def test_session_failure_terminalizes_linked_task_with_nonportable_store_error(
    rejected_text: str,
    error_code: str,
) -> None:
    events, task, session = _run_linked_task_failure(RuntimeError(rejected_text))

    assert task is not None
    assert task.status is TaskStatus.FAILED
    assert task.error == {
        "message": "Operation failed with a non-portable diagnostic.",
        "type": "RuntimeError",
        "durable_value_error_code": error_code,
        "durable_value_error_path": "$",
        "session_id": "sess_nonportable_session_failure",
    }
    assert session is not None
    assert session.status is SessionStatus.FAILED
    task_failed = [event for event in events if event.type == EventType.TASK_FAILED]
    session_failed = [event for event in events if event.type == EventType.SESSION_FAILED]
    assert len(task_failed) == 1
    assert len(session_failed) == 1
    assert session_failed[0].payload == {
        "error": "Operation failed with a non-portable diagnostic.",
        "error_type": "RuntimeError",
        "durable_value_error_code": error_code,
        "durable_value_error_path": "$",
    }


@pytest.mark.parametrize("rejected_name", ["bad\u0000hook", "bad\ud800hook"])
def test_runtime_hook_name_must_be_portable_at_registration(rejected_name: str) -> None:
    class InvalidNameHook(RuntimeHook):
        @property
        def name(self) -> str:
            return rejected_name

    with pytest.raises(DurableValueError) as app_error:
        CayuApp(runtime_hooks=[InvalidNameHook()], enable_logging=False)

    assert app_error.value.code in {"nul_character", "unicode_surrogate"}
    app = CayuApp(enable_logging=False)
    with pytest.raises(DurableValueError) as agent_error:
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            runtime_hooks=[InvalidNameHook()],
        )
    assert agent_error.value.code == app_error.value.code


def test_runtime_hook_identity_is_frozen_before_external_tool_execution() -> None:
    effects: list[str] = []
    hook_calls: list[tuple[str, str]] = []

    class ExternalEffectTool(Tool):
        spec = ToolSpec(
            name="external_effect",
            effect=ToolEffect.EXTERNAL,
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del ctx, args
            effects.append("executed")
            return ToolResult(content="effect completed")

    class SingleReadNameHook(RuntimeHook):
        def __init__(self) -> None:
            self.name_reads = 0

        @property
        def name(self) -> str:
            self.name_reads += 1
            if self.name_reads == 1:
                return "frozen-hook"
            return "bad\u0000hook"

        async def after_tool_call(self, context: ToolCallHookContext) -> None:
            hook_calls.append((context.phase.value, context.hook_name))

        async def after_session_completed(self, context: RuntimeHookContext) -> None:
            hook_calls.append((context.phase.value, context.hook_name))

    hook = SingleReadNameHook()
    store = InMemorySessionStore()
    app = CayuApp(
        session_store=store,
        runtime_hooks=[hook],
        enable_logging=False,
    )
    assert app.describe().runtime.runtime_hooks == ("SingleReadNameHook",)
    assert hook.name_reads == 1
    app.register_provider(
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_external",
                        name="external_effect",
                        arguments={},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="scripted-model"),
        tools=[ExternalEffectTool()],
    )

    events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_frozen_runtime_hook_identity",
                messages=[Message.text("user", "perform the external effect")],
            ),
        )
    )
    session = asyncio.run(store.load("sess_frozen_runtime_hook_identity"))

    assert effects == ["executed"]
    assert hook.name_reads == 1
    assert hook_calls == [
        ("after_tool_call", "frozen-hook"),
        ("after_session_completed", "frozen-hook"),
    ]
    assert session is not None
    assert session.status is SessionStatus.COMPLETED
    assert sum(event.type == EventType.TOOL_CALL_COMPLETED for event in events) == 1
    assert EventType.TOOL_CALL_FAILED not in [event.type for event in events]
    assert EventType.SESSION_FAILED not in [event.type for event in events]
    hook_events = [
        event
        for event in events
        if event.type in {EventType.HOOK_STARTED, EventType.HOOK_COMPLETED}
    ]
    assert len(hook_events) == 4
    assert {event.payload["hook_name"] for event in hook_events} == {"frozen-hook"}


@pytest.mark.parametrize(
    ("rejected_text", "error_code"),
    [
        ("hook failure\u0000with invalid text", "nul_character"),
        ("hook failure\ud800with invalid text", "unicode_surrogate"),
    ],
)
def test_terminal_hook_nonportable_failure_is_recorded_without_rewriting_session(
    rejected_text: str,
    error_code: str,
) -> None:
    calls: list[str] = []

    class FailingHook(RuntimeHook):
        async def after_session_completed(self, context: RuntimeHookContext) -> None:
            calls.append(context.hook_name)
            raise RuntimeError(rejected_text)

    class LaterHook(RuntimeHook):
        async def after_session_completed(self, context: RuntimeHookContext) -> None:
            calls.append(context.hook_name)

    store = InMemorySessionStore()
    app = CayuApp(
        session_store=store,
        runtime_hooks=[FailingHook(), LaterHook()],
        enable_logging=False,
    )
    app.register_provider(
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            ]
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="scripted-model"))

    events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_nonportable_terminal_hook",
                messages=[Message.text("user", "finish")],
            ),
        )
    )
    session = asyncio.run(store.load("sess_nonportable_terminal_hook"))

    assert session is not None
    assert session.status is SessionStatus.COMPLETED
    assert EventType.SESSION_FAILED not in [event.type for event in events]
    assert calls == ["FailingHook", "LaterHook"]
    failed = next(event for event in events if event.type == EventType.HOOK_FAILED)
    assert failed.payload == {
        "hook_name": "FailingHook",
        "scope": "app",
        "phase": "after_session_completed",
        "terminal_event_id": PRIVATE_EVENT_AUTHORITY,
        "terminal_event_type": "session.completed",
        "error": "Runtime hook failed with a non-portable diagnostic.",
        "error_type": "RuntimeError",
        "durable_value_error_code": error_code,
        "durable_value_error_path": "$",
        "actions": [],
        EXECUTION_PROFILE_FINGERPRINT_FIELD: failed.payload[EXECUTION_PROFILE_FINGERPRINT_FIELD],
    }
    assert any(
        event.type == EventType.HOOK_COMPLETED and event.payload["hook_name"] == "LaterHook"
        for event in events
    )


@pytest.mark.parametrize(
    ("rejected_text", "error_code"),
    [
        ("knowledge failure\u0000with invalid text", "nul_character"),
        ("knowledge failure\ud800with invalid text", "unicode_surrogate"),
    ],
)
def test_fail_open_knowledge_search_publishes_portable_failure_and_calls_provider(
    rejected_text: str,
    error_code: str,
) -> None:
    class FailingKnowledgeStore(InMemoryKnowledgeStore):
        async def search(self, query, *, access_scope=None):
            del query, access_scope
            raise RuntimeError(rejected_text)

    store = InMemorySessionStore()
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.text_delta("answered without knowledge"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ]
    )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            knowledge_store=FailingKnowledgeStore(access_scope=KnowledgeAccessScope.privileged()),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="scripted-model"),
        context_policy=KnowledgeInjectionPolicy(fail_open=True),
    )

    events = asyncio.run(
        _collect(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_nonportable_knowledge_failure",
                messages=[Message.text("user", "answer from available context")],
            ),
        )
    )
    session = asyncio.run(store.load("sess_nonportable_knowledge_failure"))

    assert session is not None
    assert session.status is SessionStatus.COMPLETED
    assert len(provider.requests) == 1
    failed = next(event for event in events if event.type == EventType.KNOWLEDGE_SEARCH_FAILED)
    assert failed.payload["error"] == ("Knowledge search failed with a non-portable diagnostic.")
    assert failed.payload["error_type"] == "RuntimeError"
    assert failed.payload["durable_value_error_code"] == error_code
    assert failed.payload["durable_value_error_path"] == "$"
    assert EventType.MODEL_STARTED in [event.type for event in events]
    assert EventType.SESSION_FAILED not in [event.type for event in events]
