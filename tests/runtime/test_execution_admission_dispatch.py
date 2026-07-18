from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryOperation,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    EnvironmentSpec,
    Event,
    EventType,
    ExecutionAdmissionCandidate,
    ExecutionCapabilityClaim,
    ExecutionCapabilityEvidence,
    ExecutionRequirements,
    LocalRunner,
    Message,
    ResumeRequest,
    RunRequest,
    Workspace,
    WorkspaceSnapshot,
)
from cayu.environments import BoundWorkspace, WorkspaceBinding
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runners import DEFAULT_EXEC_OUTPUT_LIMIT_BYTES, ExecCommand, ExecResult, Runner


def _candidate(name: str, *, state: str) -> ExecutionAdmissionCandidate:
    claim = (
        ExecutionCapabilityClaim.declared("confirmed_cleanup")
        if state == "declared"
        else ExecutionCapabilityClaim.available("confirmed_cleanup")
    )
    return ExecutionAdmissionCandidate(
        candidate=name,
        evidence=ExecutionCapabilityEvidence(subject=name, claims=(claim,)),
    )


class _EvidenceRunner(Runner):
    def __init__(self, candidate: str) -> None:
        self.candidate = candidate

    def execution_admission_candidate(self) -> ExecutionAdmissionCandidate:
        return _candidate(self.candidate, state="available")

    async def exec(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ) -> ExecResult:
        del command, cwd, env, timeout_s, stdin, output_limit_bytes
        return ExecResult()


class _NoEvidenceRunner(Runner):
    async def exec(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ) -> ExecResult:
        del command, cwd, env, timeout_s, stdin, output_limit_bytes
        return ExecResult()


class _RecordingProvider(ModelProvider):
    name = "fake"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _HostedFactory(EnvironmentFactory):
    def __init__(
        self,
        *,
        pre_create_candidate: str | None,
        runner: Runner,
        binding: WorkspaceBinding | None = None,
    ) -> None:
        self.pre_create_candidate = pre_create_candidate
        self.runner = runner
        self.binding = binding
        self.requests: list[EnvironmentFactoryRequest] = []

    def execution_admission_candidate(
        self,
        request: EnvironmentFactoryRequest,
    ) -> ExecutionAdmissionCandidate | None:
        del request
        if self.pre_create_candidate is None:
            return None
        return _candidate(self.pre_create_candidate, state="declared")

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        self.requests.append(request)
        return EnvironmentFactoryResult(
            environment=Environment(
                EnvironmentSpec(name=request.environment_name),
                runner=self.runner,
                binding=self.binding,
            )
        )


class _SwitchingBinding(WorkspaceBinding):
    def __init__(self, runner: Runner, *, lifecycle: list[str] | None = None) -> None:
        self.runner = runner
        self.lifecycle = lifecycle
        self.finalize_calls = 0
        self.finalize_outcomes: list[str | None] = []
        self.source_runner: Runner | None = None

    async def bind(
        self,
        workspace: Workspace | None,
        runner: Runner | None,
        *,
        session_id: str,
        agent_name: str | None = None,
        environment_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BoundWorkspace:
        del session_id, agent_name, environment_name, metadata
        self.source_runner = runner
        return BoundWorkspace(
            workspace=workspace,
            source_workspace=workspace,
            runner=self.runner,
        )

    async def finalize(
        self,
        bound: BoundWorkspace,
        *,
        outcome: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkspaceSnapshot | None:
        del metadata
        self.finalize_calls += 1
        self.finalize_outcomes.append(outcome)
        if self.lifecycle is not None:
            self.lifecycle.append("binding.finalize")
        if bound.runner is not None:
            await bound.runner.close()
        if self.source_runner is not None and self.source_runner is not bound.runner:
            await self.source_runner.close()
        return None


class _FailingBinding(WorkspaceBinding):
    async def bind(
        self,
        workspace: Workspace | None,
        runner: Runner | None,
        **kwargs: Any,
    ) -> BoundWorkspace:
        del workspace, runner, kwargs
        raise RuntimeError("bind failed")

    async def finalize(
        self,
        bound: BoundWorkspace,
        *,
        outcome: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkspaceSnapshot | None:
        del bound, outcome, metadata
        raise AssertionError("finalize should not run after bind failure")


class _ReleasableHostedFactory(_HostedFactory):
    def __init__(
        self,
        *,
        pre_create_candidate: str,
        runner: Runner,
        binding: WorkspaceBinding,
        lifecycle: list[str],
    ) -> None:
        super().__init__(
            pre_create_candidate=pre_create_candidate,
            runner=runner,
            binding=binding,
        )
        self.lifecycle = lifecycle
        self.release_actions: list[EnvironmentFactoryReleaseAction] = []

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        self.requests.append(request)

        async def release(action: EnvironmentFactoryReleaseAction) -> None:
            self.lifecycle.append(f"factory.release:{action.value}")
            self.release_actions.append(action)
            await self.runner.close()

        return EnvironmentFactoryResult(
            environment=Environment(
                EnvironmentSpec(name=request.environment_name),
                runner=self.runner,
                binding=self.binding,
            ),
            reconnect_metadata={"allocation_id": request.session_id},
            release=release,
        )


class _RecoveringFactory(EnvironmentFactory):
    def __init__(self) -> None:
        self.requests: list[EnvironmentFactoryRequest] = []
        self.first_runner = _EvidenceRunner("hosted-b")
        self.second_runner = _EvidenceRunner("hosted-a")
        self.release_actions: list[EnvironmentFactoryReleaseAction] = []

    def execution_admission_candidate(
        self,
        request: EnvironmentFactoryRequest,
    ) -> ExecutionAdmissionCandidate:
        del request
        return _candidate("hosted-a", state="declared")

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        self.requests.append(request)
        runner = self.first_runner if len(self.requests) == 1 else self.second_runner

        async def release(action: EnvironmentFactoryReleaseAction) -> None:
            self.release_actions.append(action)
            await runner.close()

        return EnvironmentFactoryResult(
            environment=Environment(
                EnvironmentSpec(name=request.environment_name),
                runner=runner,
            ),
            reconnect_metadata={"allocation_id": f"allocation-{len(self.requests)}"},
            release=release,
        )


class _BindingRunnerFactory(EnvironmentFactory):
    def __init__(self, binding: WorkspaceBinding) -> None:
        self.binding = binding
        self.requests: list[EnvironmentFactoryRequest] = []

    def execution_admission_candidate(
        self,
        request: EnvironmentFactoryRequest,
    ) -> ExecutionAdmissionCandidate:
        del request
        return _candidate("hosted", state="declared")

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        self.requests.append(request)
        return EnvironmentFactoryResult(
            environment=Environment(
                EnvironmentSpec(name=request.environment_name),
                binding=self.binding,
            ),
            reconnect_metadata={"allocation_id": request.session_id},
        )


async def _run(app: CayuApp, session_id: str) -> list[Event]:
    return [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "run")],
            )
        )
    ]


def _requirements() -> ExecutionRequirements:
    return ExecutionRequirements.trusted(
        cleanup="confirmed",
        minimum_evidence="available",
    )


def _bound_factory_app() -> tuple[
    CayuApp,
    _ReleasableHostedFactory,
    _SwitchingBinding,
    _EvidenceRunner,
    _EvidenceRunner,
    list[str],
]:
    lifecycle: list[str] = []
    source_runner = _EvidenceRunner("hosted")
    bound_runner = _EvidenceRunner("hosted")
    binding = _SwitchingBinding(bound_runner, lifecycle=lifecycle)
    factory = _ReleasableHostedFactory(
        pre_create_candidate="hosted",
        runner=source_runner,
        binding=binding,
        lifecycle=lifecycle,
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(_RecordingProvider(), default=True)
    app.register_environment_factory(
        EnvironmentSpec(name="hosted"),
        factory,
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        execution_requirements=_requirements(),
    )
    return app, factory, binding, source_runner, bound_runner, lifecycle


def test_static_local_runner_is_refused_for_untrusted_workload(tmp_path) -> None:
    async def run() -> tuple[list[Event], _RecordingProvider]:
        provider = _RecordingProvider()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                runner=LocalRunner(tmp_path),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            execution_requirements=ExecutionRequirements.untrusted(),
        )
        return await _run(app, "sess_static_local_refused"), provider

    events, provider = asyncio.run(run())

    assert provider.requests == []
    failed = next(event for event in events if event.type is EventType.SESSION_FAILED)
    assert failed.payload["error_type"] == "ExecutionAdmissionError"


def test_generic_factory_is_admitted_through_common_dispatch() -> None:
    async def run() -> tuple[list[Event], _RecordingProvider, _HostedFactory]:
        provider = _RecordingProvider()
        factory = _HostedFactory(
            pre_create_candidate="hosted",
            runner=_EvidenceRunner("hosted"),
        )
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="hosted"),
            factory,
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            execution_requirements=_requirements(),
        )
        return await _run(app, "sess_generic_admitted"), provider, factory

    events, provider, factory = asyncio.run(run())

    assert len(provider.requests) == 1
    assert len(factory.requests) == 1
    assert factory.requests[0].execution_requirements == _requirements()
    assert EventType.SESSION_COMPLETED in {event.type for event in events}


def test_factory_binding_can_supply_the_admitted_runner() -> None:
    async def run() -> tuple[list[Event], _RecordingProvider, _BindingRunnerFactory]:
        provider = _RecordingProvider()
        factory = _BindingRunnerFactory(_SwitchingBinding(_EvidenceRunner("hosted")))
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="hosted"),
            factory,
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            execution_requirements=_requirements(),
        )
        return await _run(app, "sess_binding_supplies_runner"), provider, factory

    events, provider, factory = asyncio.run(run())

    assert len(factory.requests) == 1
    assert len(provider.requests) == 1
    assert EventType.SESSION_COMPLETED in {event.type for event in events}


def test_generic_factory_without_evidence_is_refused_before_create() -> None:
    async def run() -> tuple[list[Event], _RecordingProvider, _HostedFactory]:
        provider = _RecordingProvider()
        factory = _HostedFactory(
            pre_create_candidate=None,
            runner=_EvidenceRunner("hosted"),
        )
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="hosted"),
            factory,
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            execution_requirements=_requirements(),
        )
        return await _run(app, "sess_generic_precreate_refused"), provider, factory

    events, provider, factory = asyncio.run(run())

    assert factory.requests == []
    assert provider.requests == []
    assert EventType.ENVIRONMENT_FACTORY_FAILED in {event.type for event in events}


def test_binding_cannot_switch_the_admitted_execution_candidate() -> None:
    async def run() -> tuple[
        list[Event],
        _RecordingProvider,
        _SwitchingBinding,
        _ReleasableHostedFactory,
        _EvidenceRunner,
        _EvidenceRunner,
        list[str],
        dict[str, Any],
    ]:
        provider = _RecordingProvider()
        lifecycle: list[str] = []
        original_runner = _EvidenceRunner("hosted-a")
        replacement_runner = _EvidenceRunner("hosted-b")
        binding = _SwitchingBinding(replacement_runner, lifecycle=lifecycle)
        factory = _ReleasableHostedFactory(
            pre_create_candidate="hosted-a",
            runner=original_runner,
            binding=binding,
            lifecycle=lifecycle,
        )
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="hosted"),
            factory,
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            execution_requirements=_requirements(),
        )
        events = await _run(app, "sess_binding_candidate_switch")
        checkpoint = await app.session_store.load_checkpoint("sess_binding_candidate_switch")
        assert checkpoint is not None
        return (
            events,
            provider,
            binding,
            factory,
            original_runner,
            replacement_runner,
            lifecycle,
            checkpoint,
        )

    (
        events,
        provider,
        binding,
        factory,
        original_runner,
        replacement_runner,
        lifecycle,
        checkpoint,
    ) = asyncio.run(run())

    assert provider.requests == []
    assert binding.finalize_calls == 1
    assert binding.finalize_outcomes == ["interrupted"]
    assert factory.release_actions == []
    assert lifecycle == ["binding.finalize"]
    assert original_runner.is_closed is True
    assert replacement_runner.is_closed is True
    assert checkpoint["environment_factory_reconnect"] == {
        "hosted": {"allocation_id": "sess_binding_candidate_switch"}
    }
    assert checkpoint["environment_factory_allocation_owner"] == {
        "hosted": "sess_binding_candidate_switch"
    }
    finalize_started = next(
        event for event in events if event.type is EventType.ENVIRONMENT_BINDING_FINALIZE_STARTED
    )
    assert finalize_started.payload["outcome"] == "interrupted"
    assert finalize_started.payload["terminal_outcome"] == "failed"
    assert finalize_started.payload["factory_allocation_action"] == "preserve"
    failed = next(event for event in events if event.type is EventType.SESSION_FAILED)
    assert failed.payload["error_type"] == "ExecutionAdmissionError"
    assert "environment_factory_release" not in failed.payload


def test_bind_failure_releases_unadopted_factory_result_once() -> None:
    async def run() -> tuple[
        list[Event],
        _RecordingProvider,
        _ReleasableHostedFactory,
        _EvidenceRunner,
        list[str],
        dict[str, Any],
    ]:
        provider = _RecordingProvider()
        lifecycle: list[str] = []
        runner = _EvidenceRunner("hosted")
        factory = _ReleasableHostedFactory(
            pre_create_candidate="hosted",
            runner=runner,
            binding=_FailingBinding(),
            lifecycle=lifecycle,
        )
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="hosted"),
            factory,
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            execution_requirements=_requirements(),
        )
        events = await _run(app, "sess_bind_release_once")
        checkpoint = await app.session_store.load_checkpoint("sess_bind_release_once")
        assert checkpoint is not None
        return events, provider, factory, runner, lifecycle, checkpoint

    events, provider, factory, runner, lifecycle, checkpoint = asyncio.run(run())

    assert provider.requests == []
    assert factory.release_actions == [EnvironmentFactoryReleaseAction.PRESERVE]
    assert lifecycle == ["factory.release:preserve"]
    assert runner.is_closed is True
    assert checkpoint["environment_factory_allocation_owner"] == {
        "hosted": "sess_bind_release_once"
    }
    binding_failed = next(
        event for event in events if event.type is EventType.ENVIRONMENT_BINDING_FAILED
    )
    failed = next(event for event in events if event.type is EventType.SESSION_FAILED)
    expected_release = {
        "action": "preserve",
        "callback_provided": True,
        "completed": True,
    }
    assert binding_failed.payload["environment_factory_release"] == expected_release
    assert failed.payload["environment_factory_release"] == expected_release


def test_bind_cancellation_releases_unadopted_factory_result_once() -> None:
    class _BlockingBinding(WorkspaceBinding):
        def __init__(self, started: asyncio.Event) -> None:
            self.started = started

        async def bind(
            self,
            workspace: Workspace | None,
            runner: Runner | None,
            **kwargs: Any,
        ) -> BoundWorkspace:
            del workspace, runner, kwargs
            self.started.set()
            await asyncio.Event().wait()
            raise AssertionError("cancelled bind unexpectedly resumed")

        async def finalize(
            self,
            bound: BoundWorkspace,
            *,
            outcome: str | None = None,
            metadata: dict[str, Any] | None = None,
        ) -> WorkspaceSnapshot | None:
            del bound, outcome, metadata
            raise AssertionError("finalize should not run after bind cancellation")

    async def run() -> tuple[
        _ReleasableHostedFactory,
        _EvidenceRunner,
        list[str],
        dict[str, Any],
    ]:
        provider = _RecordingProvider()
        lifecycle: list[str] = []
        runner = _EvidenceRunner("hosted")
        bind_started = asyncio.Event()
        factory = _ReleasableHostedFactory(
            pre_create_candidate="hosted",
            runner=runner,
            binding=_BlockingBinding(bind_started),
            lifecycle=lifecycle,
        )
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="hosted"),
            factory,
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            execution_requirements=_requirements(),
        )
        run_task = asyncio.create_task(_run(app, "sess_bind_release_cancel"))
        await asyncio.wait_for(bind_started.wait(), timeout=1)
        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run_task
        checkpoint = await app.session_store.load_checkpoint("sess_bind_release_cancel")
        assert checkpoint is not None
        assert provider.requests == []
        return factory, runner, lifecycle, checkpoint

    factory, runner, lifecycle, checkpoint = asyncio.run(run())

    assert factory.release_actions == [EnvironmentFactoryReleaseAction.PRESERVE]
    assert lifecycle == ["factory.release:preserve"]
    assert runner.is_closed is True
    assert checkpoint["environment_factory_allocation_owner"] == {
        "hosted": "sess_bind_release_cancel"
    }


def test_abandoned_factory_result_is_released_before_binding() -> None:
    async def run() -> tuple[
        _ReleasableHostedFactory,
        _EvidenceRunner,
        list[str],
        str,
    ]:
        app, factory, _binding, runner, _bound_runner, lifecycle = _bound_factory_app()
        stream = app.run(
            RunRequest(
                agent_name="assistant",
                session_id="sess_factory_abandoned_before_bind",
                messages=[Message.text("user", "run")],
            )
        )
        async for event in stream:
            if event.type is EventType.ENVIRONMENT_FACTORY_COMPLETED:
                break
        await stream.aclose()
        session = await app.session_store.load("sess_factory_abandoned_before_bind")
        assert session is not None
        return factory, runner, lifecycle, session.status.value

    factory, runner, lifecycle, status = asyncio.run(run())

    assert factory.release_actions == [EnvironmentFactoryReleaseAction.PRESERVE]
    assert lifecycle == ["factory.release:preserve"]
    assert runner.is_closed is True
    assert status == "interrupted"


def test_binding_completion_publication_failure_finalizes_adopted_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[_SwitchingBinding, _EvidenceRunner, _EvidenceRunner, list[str]]:
        app, _factory, binding, source_runner, bound_runner, lifecycle = _bound_factory_app()
        original_emit = app._event_writer.emit

        async def fail_binding_completion(event: Event) -> Event:
            if event.type is EventType.ENVIRONMENT_BINDING_COMPLETED:
                raise RuntimeError("binding completion publication failed")
            return await original_emit(event)

        monkeypatch.setattr(app._event_writer, "emit", fail_binding_completion)
        await _run(app, "sess_binding_completion_publication_failed")
        return binding, source_runner, bound_runner, lifecycle

    binding, source_runner, bound_runner, lifecycle = asyncio.run(run())

    assert binding.finalize_calls == 1
    assert binding.finalize_outcomes == ["interrupted"]
    assert lifecycle == ["binding.finalize"]
    assert source_runner.is_closed is True
    assert bound_runner.is_closed is True


def test_finalize_started_publication_failure_does_not_skip_binding_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[_SwitchingBinding, _EvidenceRunner, _EvidenceRunner]:
        app, _factory, binding, source_runner, bound_runner, _lifecycle = _bound_factory_app()
        original_emit = app._event_writer.emit

        async def fail_finalize_started(event: Event) -> Event:
            if event.type is EventType.ENVIRONMENT_BINDING_FINALIZE_STARTED:
                raise RuntimeError("finalize start publication failed")
            return await original_emit(event)

        monkeypatch.setattr(app._event_writer, "emit", fail_finalize_started)
        events = await _run(app, "sess_finalize_start_publication_failed")
        terminal = next(event for event in events if event.type is EventType.SESSION_COMPLETED)
        assert terminal.payload["binding_finalize_publication_error"]["failures"] == [
            {
                "phase": "finalize_started_event",
                "error": "finalize start publication failed",
                "error_type": "RuntimeError",
            }
        ]
        return binding, source_runner, bound_runner

    binding, source_runner, bound_runner = asyncio.run(run())

    assert binding.finalize_calls == 1
    assert source_runner.is_closed is True
    assert bound_runner.is_closed is True


def test_binding_completion_publication_cancellation_finalizes_adopted_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[_SwitchingBinding, _EvidenceRunner, _EvidenceRunner]:
        app, _factory, binding, source_runner, bound_runner, _lifecycle = _bound_factory_app()
        publication_started = asyncio.Event()
        original_emit = app._event_writer.emit

        async def block_binding_completion(event: Event) -> Event:
            if event.type is EventType.ENVIRONMENT_BINDING_COMPLETED:
                publication_started.set()
                await asyncio.Event().wait()
            return await original_emit(event)

        monkeypatch.setattr(app._event_writer, "emit", block_binding_completion)
        run_task = asyncio.create_task(_run(app, "sess_binding_completion_publication_cancelled"))
        await asyncio.wait_for(publication_started.wait(), timeout=1)
        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run_task
        return binding, source_runner, bound_runner

    binding, source_runner, bound_runner = asyncio.run(run())

    assert binding.finalize_calls == 1
    assert binding.finalize_outcomes == ["interrupted"]
    assert source_runner.is_closed is True
    assert bound_runner.is_closed is True


def test_factory_candidate_switch_is_refused_and_discarded_before_checkpoint() -> None:
    async def run() -> tuple[
        list[Event],
        _RecordingProvider,
        _EvidenceRunner,
        dict[str, Any] | None,
    ]:
        provider = _RecordingProvider()
        runner = _EvidenceRunner("hosted-b")
        factory = _HostedFactory(
            pre_create_candidate="hosted-a",
            runner=runner,
        )
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="hosted"),
            factory,
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            execution_requirements=_requirements(),
        )
        events = await _run(app, "sess_factory_candidate_switch")
        checkpoint = await app.session_store.load_checkpoint("sess_factory_candidate_switch")
        return events, provider, runner, checkpoint

    events, provider, runner, checkpoint = asyncio.run(run())

    assert provider.requests == []
    assert runner.is_closed is True
    failed = next(event for event in events if event.type is EventType.SESSION_FAILED)
    assert failed.payload["environment_factory_release"] == {
        "action": "discard",
        "callback_provided": False,
        "completed": True,
    }
    assert checkpoint is None or "environment_factory_reconnect" not in checkpoint
    assert checkpoint is None or "environment_factory_allocation_owner" not in checkpoint


def test_failed_pre_checkpoint_factory_setup_recreates_on_resume() -> None:
    async def run() -> tuple[list[Event], list[Event], _RecordingProvider, _RecoveringFactory]:
        provider = _RecordingProvider()
        factory = _RecoveringFactory()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="hosted"),
            factory,
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            execution_requirements=_requirements(),
        )
        initial_events = await _run(app, "sess_factory_recreate")
        resumed_events = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id="sess_factory_recreate",
                    messages=[Message.text("user", "retry")],
                )
            )
        ]
        return initial_events, resumed_events, provider, factory

    initial_events, resumed_events, provider, factory = asyncio.run(run())

    assert EventType.SESSION_FAILED in {event.type for event in initial_events}
    assert EventType.SESSION_COMPLETED in {event.type for event in resumed_events}
    assert [request.operation for request in factory.requests] == [
        EnvironmentFactoryOperation.CREATE,
        EnvironmentFactoryOperation.CREATE,
    ]
    assert [request.reconnect_metadata for request in factory.requests] == [{}, {}]
    assert factory.release_actions == [EnvironmentFactoryReleaseAction.DISCARD]
    assert factory.first_runner.is_closed is True
    assert len(provider.requests) == 1


def test_missing_final_runner_evidence_returns_structured_refusal() -> None:
    async def run() -> list[Event]:
        app = CayuApp(enable_logging=False)
        app.register_provider(_RecordingProvider(), default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="hosted"),
            _HostedFactory(
                pre_create_candidate="hosted",
                runner=_NoEvidenceRunner(),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            execution_requirements=_requirements(),
        )
        return await _run(app, "sess_missing_final_evidence")

    events = asyncio.run(run())

    failed = next(event for event in events if event.type is EventType.SESSION_FAILED)
    assert failed.payload["error_type"] == "ExecutionAdmissionError"
    decision = failed.payload["execution_admission"]
    assert decision["status"] == "refused"
    assert decision["candidate"] == "hosted"
    assert decision["stage"] == "pre_exposure"
    assert decision["refusals"] == [
        {
            "code": "missing_capability",
            "capability": "confirmed_cleanup",
            "required_state": "available",
            "observed_state": "missing",
            "reason_code": None,
            "remediation_code": None,
        }
    ]
