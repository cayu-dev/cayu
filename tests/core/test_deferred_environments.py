"""Deferred environment materialization through the real runtime lifecycle."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cayu import (
    AgentSpec,
    AlwaysRequireApprovalToolPolicy,
    CayuApp,
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryResult,
    EnvironmentSpec,
    EventType,
    ExecCommand,
    ExecutionProfileBehaviorIdentity,
    InMemorySessionStore,
    LocalRunner,
    LocalWorkspace,
    Message,
    ModelStreamEvent,
    PendingToolApprovalEventView,
    ResumeRequest,
    RunRequest,
    ScriptedModelProvider,
    StaticToolExposurePolicy,
    StaticToolPolicy,
    SyncBinding,
    Tool,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolEffect,
    ToolExecutableRequirement,
    ToolExecutionRequirement,
    ToolResult,
    ToolSpec,
    run_to_completion,
)
from cayu.environments.admission import ExecutionRequirements
from cayu.environments.deferred import (
    DeferredMaterialization,
    EnvironmentMaterializationError,
)
from cayu.workspaces.revisions import (
    WorkspaceRevisionObservationLimits,
    observe_deterministic_workspace,
)
from cayu.workspaces.runner import RunnerWorkspace


def _identity(name: str) -> ExecutionProfileBehaviorIdentity:
    return ExecutionProfileBehaviorIdentity(
        name=name, behavior_version="1", implementation_version="1"
    )


class ObservedSyncBinding(SyncBinding):
    """Copy-in/copy-back binding that observes its target like the Docker binding."""

    async def observe_revision(self, bound):
        return await observe_deterministic_workspace(
            bound.workspace,
            observer=type(self).__name__,
            limits=WorkspaceRevisionObservationLimits(),
        )


def _placeholder(runner) -> RunnerWorkspace:
    # Like the Docker factory: a workspace over the deferred runner itself.
    return RunnerWorkspace(runner, workspace_id="deferred-target", python_executable=sys.executable)


class DeferredLocalFactory(EnvironmentFactory):
    """Deferred factory over local directories: a stand-in for a container."""

    def __init__(self, source: Path, target: Path, *, failures: int = 0) -> None:
        self.source = source
        self.target = target
        self.failures = failures
        self.created = 0
        self.disposed_unmaterialized = 0

    @property
    def secret_resolution_scope(self):
        return "static"

    @property
    def execution_profile_identity(self):
        return _identity("test.deferred_local_factory")

    @property
    def deferred_materialization(self) -> bool:
        return True

    async def is_allocation_disposed(self, request) -> bool:
        return True

    async def create(self, request) -> EnvironmentFactoryResult:
        async def materialize(mode):
            await asyncio.sleep(0.01)
            if self.failures:
                self.failures -= 1
                raise RuntimeError("container runtime unavailable")
            self.created += 1
            return LocalRunner(self.target), ObservedSyncBinding(
                target_workspace=LocalWorkspace(self.target, workspace_id="target"),
                path=str(self.target),
            )

        async def dispose_unmaterialized() -> None:
            self.disposed_unmaterialized += 1

        materialization = DeferredMaterialization(
            materialize,
            default_cwd=str(self.target),
            placeholder_workspace=_placeholder,
            dispose_unmaterialized=dispose_unmaterialized,
        )
        return EnvironmentFactoryResult(
            environment=Environment(
                EnvironmentSpec(name=request.environment_name),
                workspace=LocalWorkspace(self.source, workspace_id="source"),
                runner=materialization.runner,
                binding=materialization.binding,
            ),
            reconnect_metadata={"kind": "deferred-local"},
        )


class RunnerTool(Tool):
    def __init__(self, name: str = "write", *, mutation: bool = False) -> None:
        self.spec = ToolSpec(
            name=name,
            description="Use the sandbox.",
            input_schema={"type": "object", "properties": {}},
            effect=ToolEffect.EXTERNAL if mutation else ToolEffect.NONE,
            workspace_mutation=mutation,
            parallel_safe=not mutation,
            execution_profile_identity=_identity(f"test.{name}"),
        )

    async def run(self, ctx, args) -> ToolResult:
        result = await ctx.runner.exec(
            ExecCommand(argv=["sh", "-c", f"cat input.txt; echo out > {self.spec.name}.txt"]),
            timeout_s=10,
        )
        return ToolResult(content=result.stdout.strip())


def _turns(*tool_names: str) -> list[list[ModelStreamEvent]]:
    turns: list[list[ModelStreamEvent]] = []
    if tool_names:
        turns.append(
            [
                *(
                    ModelStreamEvent.tool_call(id=f"call-{name}", name=name, arguments={})
                    for name in tool_names
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        )
    turns.append(
        [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({"finish_reason": "stop"})]
    )
    return turns


def _app(
    factory: DeferredLocalFactory,
    store: InMemorySessionStore,
    turns: list[list[ModelStreamEvent]],
    *,
    tools: tuple[str, ...] = ("write",),
    approval: bool = False,
    mutation: bool = False,
) -> CayuApp:
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(ScriptedModelProvider(turns), default=True)
    app.register_environment_factory(
        EnvironmentSpec(name="sandbox", execution_profile_identity=_identity("test.sandbox")),
        factory,
        default=True,
    )
    app.register_agent(
        AgentSpec(name="agent", model="scripted"),
        tools=[RunnerTool(name, mutation=mutation) for name in tools],
        tool_exposure_policy=StaticToolExposurePolicy(profile_id="deferred", tools=tools),
        tool_policy=AlwaysRequireApprovalToolPolicy(tools=list(tools))
        if approval
        else StaticToolPolicy(),
    )
    return app


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / "input.txt").write_text("hello\n", encoding="utf-8")
    return source, target


async def _types(store: InMemorySessionStore, session_id: str) -> list[str]:
    return [str(event.type) for event in await store.load_events(session_id)]


def test_a_run_that_never_uses_the_runner_never_creates_it(dirs) -> None:
    source, target = dirs
    factory = DeferredLocalFactory(source, target)
    store = InMemorySessionStore()

    async def run():
        outcome = await run_to_completion(
            _app(factory, store, _turns()),
            RunRequest(agent_name="agent", messages=[Message.text("user", "hi")]),
        )
        return outcome, await _types(store, outcome.session_id)

    outcome, types = asyncio.run(run())

    assert outcome.ok, outcome.error
    assert factory.created == 0
    assert "environment.deferred" in types
    assert not any(t.startswith("environment.materialization") for t in types)
    # No bind happened, so no binding start/completion events.
    assert "environment.binding.started" not in types
    assert "environment.binding.completed" not in types
    # Finalization still settles a resource a crashed process could have left.
    assert factory.disposed_unmaterialized == 1
    assert sorted(path.name for path in source.iterdir()) == ["input.txt"]


def test_first_runner_use_materializes_once_and_binds_through_the_real_binding(dirs) -> None:
    source, target = dirs
    factory = DeferredLocalFactory(source, target)
    store = InMemorySessionStore()

    async def run():
        outcome = await run_to_completion(
            _app(factory, store, _turns("write", "second"), tools=("write", "second")),
            RunRequest(agent_name="agent", messages=[Message.text("user", "go")]),
        )
        return outcome, await store.load_events(outcome.session_id)

    outcome, events = asyncio.run(run())

    assert outcome.ok, outcome.error
    assert factory.created == 1
    started = [e for e in events if e.type is EventType.ENVIRONMENT_MATERIALIZATION_STARTED]
    completed = [e for e in events if e.type is EventType.ENVIRONMENT_MATERIALIZATION_COMPLETED]
    assert len(started) == len(completed) == 1
    assert started[0].payload["trigger_tool_call_id"] in {"call-write", "call-second"}
    assert started[0].payload["trigger_tool_name"] in {"write", "second"}
    # Copy-in happened before the command, copy-back at finalization.
    tool_results = [e for e in events if e.type is EventType.TOOL_CALL_COMPLETED]
    assert all("hello" in str(e.payload) for e in tool_results)
    assert (source / "write.txt").read_text() == "out\n"
    assert (source / "second.txt").read_text() == "out\n"


def test_parallel_first_uses_share_one_materialization(tmp_path: Path) -> None:
    created = 0

    async def materialize(mode):
        nonlocal created
        await asyncio.sleep(0.05)
        created += 1
        return LocalRunner(tmp_path), None

    async def run():
        state = DeferredMaterialization(materialize, default_cwd=str(tmp_path))
        runners = await asyncio.gather(*(state.materialize() for _ in range(8)))
        return runners

    runners = asyncio.run(run())
    assert created == 1
    assert len({id(runner) for runner in runners}) == 1


def test_materialization_failure_fails_the_call_and_the_session_stays_usable(dirs) -> None:
    source, target = dirs
    factory = DeferredLocalFactory(source, target, failures=1)
    store = InMemorySessionStore()

    async def run():
        first = await run_to_completion(
            _app(factory, store, _turns("write")),
            RunRequest(agent_name="agent", messages=[Message.text("user", "go")]),
        )
        events = await store.load_events(first.session_id)
        second = await run_to_completion(
            _app(factory, store, _turns("write")),
            ResumeRequest(session_id=first.session_id, messages=[Message.text("user", "again")]),
        )
        return first, events, second

    first, events, second = asyncio.run(run())

    failed_calls = [e for e in events if e.type is EventType.TOOL_CALL_FAILED]
    # The typed runner-unavailable failure carries the materialization diagnostic.
    assert failed_calls and "RunnerUnavailableError" in str(failed_calls[0].payload)
    failed = [e for e in events if e.type is EventType.ENVIRONMENT_MATERIALIZATION_FAILED]
    assert failed and failed[0].payload["reason"] == "RuntimeError"
    assert failed[0].payload["trigger_tool_call_id"] == "call-write"
    assert first.ok  # the model saw a failed tool call and finished the turn
    assert second.ok, second.error
    assert factory.created == 1
    assert (source / "write.txt").read_text() == "out\n"


def test_admission_uses_configured_evidence_until_live_evidence_is_required(
    tmp_path: Path,
) -> None:
    created = 0
    configured = object()

    async def materialize(mode):
        nonlocal created
        created += 1
        return LocalRunner(tmp_path), None

    async def run():
        state = DeferredMaterialization(
            materialize,
            default_cwd=str(tmp_path),
            configured_candidate=lambda requirements: configured,
        )
        runner = state.runner
        trusted = await runner.execution_admission_observer(
            ExecutionRequirements.trusted()
        ).collect()
        assert trusted is configured
        assert created == 0
        live = ExecutionRequirements.model_validate(
            {
                **ExecutionRequirements.trusted().model_dump(mode="python", warnings=False),
                "required_executables": ("sh",),
            }
        )
        await runner.execution_admission_observer(live).collect()
        return created

    assert asyncio.run(run()) == 1


def test_admission_refusal_at_materialization_is_a_typed_error(tmp_path: Path) -> None:
    async def materialize(mode):
        raise EnvironmentMaterializationError("refused", reason="admission_refused")

    async def run():
        state = DeferredMaterialization(materialize, default_cwd=str(tmp_path))
        with pytest.raises(EnvironmentMaterializationError) as raised:
            await state.runner.exec(ExecCommand(argv=["true"]))
        return raised.value

    error = asyncio.run(run())
    assert error.reason == "admission_refused"
    assert error.diagnostic["kind"] == "environment_materialization_failed"


def test_pause_before_materialization_resumes_deferred_then_materializes(dirs) -> None:
    source, target = dirs
    factory = DeferredLocalFactory(source, target)
    store = InMemorySessionStore()

    async def run():
        app = _app(factory, store, _turns("write"), approval=True)
        paused = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="agent",
                    session_id="paused",
                    messages=[Message.text("user", "go")],
                )
            )
        ]
        assert paused[-1].type is not EventType.SESSION_COMPLETED
        created_while_paused = factory.created
        approval_event = next(
            event
            for event in await store.load_events("paused")
            if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        pending = PendingToolApprovalEventView.from_event(approval_event)
        resumed_app = _app(factory, store, _turns(), approval=True)
        resumed = [
            event
            async for event in resumed_app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id="paused",
                    approval_id=pending.approval_id,
                    tool_round_id=pending.tool_round_id,
                    tool_call_id=pending.tool_call_id,
                    decision=ToolApprovalDecision.APPROVE,
                )
            )
        ]
        return created_while_paused, resumed, await _types(store, "paused")

    created_while_paused, resumed, types = asyncio.run(run())

    assert created_while_paused == 0
    assert resumed[-1].type is EventType.SESSION_COMPLETED
    assert factory.created == 1
    assert types.count("environment.materialization.completed") == 1
    assert (source / "write.txt").read_text() == "out\n"


def test_pause_after_materialization_resumes_and_uses_the_environment_again(dirs) -> None:
    source, target = dirs
    factory = DeferredLocalFactory(source, target)
    store = InMemorySessionStore()

    def app(turns):
        built = CayuApp(session_store=store, enable_logging=False)
        built.register_provider(ScriptedModelProvider(turns), default=True)
        built.register_environment_factory(
            EnvironmentSpec(name="sandbox", execution_profile_identity=_identity("test.sandbox")),
            factory,
            default=True,
        )
        built.register_agent(
            AgentSpec(name="agent", model="scripted"),
            tools=[RunnerTool("write"), RunnerTool("second")],
            tool_exposure_policy=StaticToolExposurePolicy(
                profile_id="deferred", tools=("write", "second")
            ),
            tool_policy=AlwaysRequireApprovalToolPolicy(tools=["second"]),
        )
        return built

    async def run():
        first_turn = [
            [
                ModelStreamEvent.tool_call(id="call-write", name="write", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(id="call-second", name="second", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
        ]
        [
            event
            async for event in app(first_turn).run(
                RunRequest(
                    agent_name="agent", session_id="mid", messages=[Message.text("user", "go")]
                )
            )
        ]
        created_before_pause = factory.created
        approval_event = next(
            event
            for event in await store.load_events("mid")
            if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        pending = PendingToolApprovalEventView.from_event(approval_event)
        resumed = [
            event
            async for event in app(_turns()).resolve_tool_approval(
                ToolApprovalRequest(
                    session_id="mid",
                    approval_id=pending.approval_id,
                    tool_round_id=pending.tool_round_id,
                    tool_call_id=pending.tool_call_id,
                    decision=ToolApprovalDecision.APPROVE,
                )
            )
        ]
        return created_before_pause, resumed

    created_before_pause, resumed = asyncio.run(run())

    assert created_before_pause == 1
    assert resumed[-1].type is EventType.SESSION_COMPLETED
    # The resumed run reconnects a deferred environment and materializes it again.
    assert factory.created == 2
    assert (source / "write.txt").read_text() == "out\n"
    assert (source / "second.txt").read_text() == "out\n"


def test_manifest_and_inspect_report_deferred_environments(dirs) -> None:
    from cayu.cli.inspect import _render_human

    source, target = dirs
    app = _app(DeferredLocalFactory(source, target), InMemorySessionStore(), _turns())
    manifest = app.describe()

    assert manifest.environments[0].deferred is True
    assert "Environments: sandbox (deferred)" in _render_human(manifest)


class EagerLocalFactory(DeferredLocalFactory):
    @property
    def deferred_materialization(self) -> bool:
        return False

    async def create(self, request) -> EnvironmentFactoryResult:
        self.created += 1
        runner = LocalRunner(self.target)
        return EnvironmentFactoryResult(
            environment=Environment(
                EnvironmentSpec(name=request.environment_name),
                workspace=LocalWorkspace(self.source, workspace_id="source"),
                runner=runner,
                binding=SyncBinding(
                    target_workspace=LocalWorkspace(self.target, workspace_id="target"),
                    path=str(self.target),
                ),
            ),
        )


def test_eager_factories_emit_no_deferral_events_and_bind_at_session_start(dirs) -> None:
    source, target = dirs
    factory = EagerLocalFactory(source, target)
    store = InMemorySessionStore()

    async def run():
        app = _app(factory, store, _turns())
        outcome = await run_to_completion(
            app, RunRequest(agent_name="agent", messages=[Message.text("user", "hi")])
        )
        return outcome, await _types(store, outcome.session_id), app.describe()

    outcome, types, manifest = asyncio.run(run())

    assert outcome.ok, outcome.error
    assert factory.created == 1
    assert (target / "input.txt").read_text() == "hello\n"  # copy-in at session start
    assert not any("deferred" in t or "materialization" in t for t in types)
    assert manifest.environments[0].deferred is False


def test_materialization_events_reach_the_live_stream_with_the_bound_workspace(dirs) -> None:
    source, target = dirs
    factory = DeferredLocalFactory(source, target)

    async def run():
        return await run_to_completion(
            _app(factory, InMemorySessionStore(), _turns("write")),
            RunRequest(agent_name="agent", messages=[Message.text("user", "go")]),
        )

    outcome = asyncio.run(run())

    types = [event.type for event in outcome.events]
    started = types.index(EventType.ENVIRONMENT_MATERIALIZATION_STARTED)
    binding_started = types.index(EventType.ENVIRONMENT_BINDING_STARTED)
    binding_completed = types.index(EventType.ENVIRONMENT_BINDING_COMPLETED)
    completed = types.index(EventType.ENVIRONMENT_MATERIALIZATION_COMPLETED)
    terminal = types.index(EventType.TOOL_CALL_COMPLETED)
    assert started < binding_started < binding_completed < completed < terminal
    assert types.count(EventType.ENVIRONMENT_BINDING_STARTED) == 1
    assert types.count(EventType.ENVIRONMENT_BINDING_COMPLETED) == 1
    # The normal binding events now describe the real bind into the runner.
    bound = outcome.events[binding_completed].payload
    assert bound["bound_path"] == str(target)
    assert (
        bound["binding_generation_id"]
        == outcome.events[binding_started].payload["binding_generation_id"]
    )
    payload = outcome.events[completed].payload
    # Tool-call linkage is projected like the call's runner.exec events.
    runner_started = next(e for e in outcome.events if e.type is EventType.RUNNER_EXEC_STARTED)
    assert payload["trigger_tool_call_id"] == runner_started.payload["tool_call_id"]
    assert payload["trigger_tool_name"] == "write"


class _NoEvidenceRunner(LocalRunner):
    async def collect_execution_admission_candidate_for(self, requirements):
        return None


class DeclaredLocalFactory(DeferredLocalFactory):
    """Declares the local candidate, so live evidence is checked at materialization."""

    def __init__(self, source: Path, target: Path, *, withhold_evidence: bool) -> None:
        super().__init__(source, target)
        self.withhold_evidence = withhold_evidence
        self.runners: list[LocalRunner] = []

    def execution_admission_candidate(self, request):
        return LocalRunner(self.target).execution_admission_candidate_for(
            request.execution_requirements
        )

    async def create(self, request) -> EnvironmentFactoryResult:
        async def materialize(mode):
            self.created += 1
            runner_type = _NoEvidenceRunner if self.withhold_evidence else LocalRunner
            runner = runner_type(self.target)
            self.runners.append(runner)
            return runner, SyncBinding(
                target_workspace=LocalWorkspace(self.target, workspace_id="target"),
                path=str(self.target),
            )

        materialization = DeferredMaterialization(
            materialize,
            default_cwd=str(self.target),
            configured_candidate=LocalRunner(self.target).execution_admission_candidate_for,
        )
        return EnvironmentFactoryResult(
            environment=Environment(
                EnvironmentSpec(name=request.environment_name),
                workspace=LocalWorkspace(self.source, workspace_id="source"),
                runner=materialization.runner,
                binding=materialization.binding,
            ),
            reconnect_metadata={"kind": "deferred-local"},
        )


@pytest.mark.parametrize("withhold_evidence", [False, True])
def test_runtime_admission_runs_on_the_materialized_runner(dirs, withhold_evidence) -> None:
    source, target = dirs
    factory = DeclaredLocalFactory(source, target, withhold_evidence=withhold_evidence)

    async def run():
        return await run_to_completion(
            _app(factory, InMemorySessionStore(), _turns("write")),
            RunRequest(agent_name="agent", messages=[Message.text("user", "go")]),
        )

    outcome = asyncio.run(run())

    types = [event.type for event in outcome.events]
    started = types.index(EventType.ENVIRONMENT_MATERIALIZATION_STARTED)
    admissions = [
        event
        for event in outcome.events[started:]
        if event.type is EventType.ENVIRONMENT_LIFECYCLE_TRANSITION
        and event.payload["phase"] == "admission"
    ]
    assert len(admissions) == 1
    if withhold_evidence:
        assert admissions[0].payload["outcome"] == "refused"
        assert "missing_final_evidence" in admissions[0].payload["refusal_codes"]
        failed = next(
            e for e in outcome.events if e.type is EventType.ENVIRONMENT_MATERIALIZATION_FAILED
        )
        assert failed.payload["reason"] == "admission_refused"
        assert EventType.TOOL_CALL_FAILED in types
        # Refused before copy-in: the runner never received workspace content.
        assert not (target / "input.txt").exists()
        assert not (source / "write.txt").exists()
    else:
        assert admissions[0].payload["outcome"] == "admitted"
        assert (source / "write.txt").read_text() == "out\n"


def test_workspace_observations_after_materialization_carry_the_deferred_identity(
    dirs,
) -> None:
    source, target = dirs
    factory = DeferredLocalFactory(source, target)
    turns = [
        [
            ModelStreamEvent.tool_call(id="call-write", name="write", arguments={}),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ],
        [
            ModelStreamEvent.tool_call(id="call-second", name="second", arguments={}),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ],
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ],
    ]

    async def run():
        outcome = await run_to_completion(
            _app(factory, InMemorySessionStore(), turns, tools=("write", "second"), mutation=True),
            RunRequest(agent_name="agent", messages=[Message.text("user", "go")]),
        )
        return outcome

    outcome = asyncio.run(run())

    assert outcome.ok, outcome.error
    observed = [
        event.payload
        for event in outcome.events
        if event.type is EventType.WORKSPACE_REVISION_OBSERVED
    ]
    statuses = [payload["status"] for payload in observed]
    # Before the first call nothing exists yet; every later observation is complete.
    assert statuses[0] == "unsupported"
    assert statuses[1:] and set(statuses[1:]) == {"supported"}, statuses
    receipts = [
        event.payload
        for event in outcome.events
        if event.type is EventType.WORKSPACE_MUTATION_RECORDED
    ]
    assert receipts[-1]["status"] == "changed", receipts


class _CrashingFactory(DeferredLocalFactory):
    """Exits the process right after creating its resource, before binding.

    The ``leftover`` file stands for the crashed run's container. Each
    materialization logs the mode the runtime asked for and whether a leftover
    was present; removing it in ``use`` mode is the factory's job, covered
    against the real Docker code in tests/environments/test_docker_deferred.py.
    """

    def __init__(self, source: Path, target: Path, *, release: str = "ok") -> None:
        super().__init__(source, target)
        self.release = release

    async def create(self, request) -> EnvironmentFactoryResult:
        result = await super().create(request)
        materialization = result.environment.runner.materialization
        real = materialization._materialize

        async def materialize(mode):
            leftover = self.target / "leftover"
            state = "leftover" if leftover.exists() else "clean"
            with (self.target.parent / "materializations.log").open("a") as log:
                log.write(f"{mode}:{state}\n")
            if mode == "use":
                leftover.unlink(missing_ok=True)
            created = await real(mode)
            if (self.target / "crash").exists():
                leftover.write_text("crashed-container", encoding="utf-8")
                os._exit(73)
            return created

        materialization._materialize = materialize
        return result

    async def release_deferred_materialization(self, request) -> bool:
        if self.release == "daemon_down":
            raise RuntimeError("container daemon unavailable")
        leftover = self.target / "leftover"
        if not leftover.exists():
            return False
        leftover.unlink()
        return True


async def _crash_child(root: Path, mode: str) -> None:
    from cayu import IncompleteSessionRecoveryRequest, SQLiteSessionStore

    source, target = root / "source", root / "target"
    store = SQLiteSessionStore(root / "sessions.db")
    try:
        app = _app(
            _CrashingFactory(source, target, release="daemon_down" if "down" in mode else "ok"),
            store,
            _turns("write"),
        )
        if mode == "run":
            (target / "crash").write_text("", encoding="utf-8")
            await run_to_completion(
                app,
                RunRequest(
                    agent_name="agent", session_id="crashed", messages=[Message.text("user", "go")]
                ),
            )
            return
        if mode == "resume":
            outcome = await run_to_completion(
                app,
                ResumeRequest(session_id="crashed", messages=[Message.text("user", "again")]),
            )
            (root / "resume.json").write_text(
                json.dumps({"ok": outcome.ok, "error": str(outcome.error)}), encoding="utf-8"
            )
            return
        result = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id="crashed", reason="process_killed")
        )
        (root / f"{mode}.json").write_text(result.model_dump_json(), encoding="utf-8")
    finally:
        await store.close()


def _crash_process(root: Path, mode: str) -> subprocess.CompletedProcess:
    script = (
        "import asyncio,sys; from pathlib import Path; "
        "from tests.core.test_deferred_environments import _crash_child; "
        "asyncio.run(_crash_child(Path(sys.argv[1]), sys.argv[2]))"
    )
    return subprocess.run(
        [sys.executable, "-c", script, str(root), mode],
        text=True,
        capture_output=True,
        timeout=60,
        cwd=Path(__file__).resolve().parents[2],
    )


@pytest.mark.parametrize("daemon", ["up", "down"])
def test_crash_cleanup_in_recovery_is_best_effort(tmp_path: Path, daemon: str) -> None:
    (tmp_path / "source").mkdir()
    (tmp_path / "target").mkdir()
    (tmp_path / "source" / "input.txt").write_text("hello\n", encoding="utf-8")

    crashed = _crash_process(tmp_path, "run")
    assert crashed.returncode == 73, crashed.stderr[-2000:]
    assert (tmp_path / "target" / "leftover").exists()
    (tmp_path / "target" / "crash").unlink()

    mode = f"recover_{daemon}"
    recovered = _crash_process(tmp_path, mode)
    assert recovered.returncode == 0, recovered.stderr[-2000:]
    result = json.loads((tmp_path / f"{mode}.json").read_text(encoding="utf-8"))
    assert result["status"] == "interrupted"
    if daemon == "up":
        assert "reaped_allocation" in result["actions"]
        assert not (tmp_path / "target" / "leftover").exists()
    else:
        # The cleanup failure is logged, not raised: recovery still completes.
        assert "reaped_allocation" not in result["actions"]
        assert (tmp_path / "target" / "leftover").exists()

    # The daemon is back. The resumed run asks for a fresh resource ("use"),
    # never recovery adoption, even though a failed cleanup left the old one.
    resumed = _crash_process(tmp_path, "resume")
    assert resumed.returncode == 0, resumed.stderr[-2000:]
    resume = json.loads((tmp_path / "resume.json").read_text(encoding="utf-8"))
    assert resume["ok"], resume["error"]
    log = (tmp_path / "materializations.log").read_text().split()
    assert log == ["use:clean", "use:clean" if daemon == "up" else "use:leftover"]
    assert (tmp_path / "source" / "write.txt").read_text() == "out\n"


def test_a_cancelled_first_materialization_records_its_outcome(tmp_path: Path) -> None:
    attempts = 0
    release = asyncio.Event()

    async def materialize(mode):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            await release.wait()  # cancelled while creating
        return LocalRunner(tmp_path), None

    async def run():
        state = DeferredMaterialization(materialize, default_cwd=str(tmp_path))
        notified: list[tuple[str, dict]] = []

        async def observer(kind, payload, bound):
            notified.append((kind, payload))

        state.attach_observer(observer)
        first = asyncio.create_task(state.materialize())
        await asyncio.sleep(0.01)
        waiter = asyncio.create_task(state.materialize())
        await asyncio.sleep(0.01)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        runner = await waiter  # a waiter retries instead of hanging
        return notified, runner

    notified, runner = asyncio.run(run())

    kinds = [kind for kind, _ in notified]
    assert kinds == ["started", "failed", "started", "completed"]
    assert notified[1][1]["reason"] == "cancelled"
    assert isinstance(runner, LocalRunner)


class ExecutableRunnerTool(RunnerTool):
    def __init__(self) -> None:
        super().__init__("write")
        self.spec = self.spec.model_copy(
            update={
                "execution_requirements": (
                    ToolExecutionRequirement(
                        name="shell",
                        alternatives=(ToolExecutableRequirement(executable="sh"),),
                    ),
                )
            }
        )


def test_session_start_materialization_collects_admission_evidence_once(dirs) -> None:
    source, target = dirs
    factory = DeclaredLocalFactory(source, target, withhold_evidence=False)

    async def run():
        app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
        app.register_provider(ScriptedModelProvider(_turns("write")), default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="sandbox", execution_profile_identity=_identity("test.sandbox")),
            factory,
            default=True,
        )
        app.register_agent(
            AgentSpec(name="agent", model="scripted"),
            tools=[ExecutableRunnerTool()],
            tool_exposure_policy=StaticToolExposurePolicy(profile_id="deferred", tools=("write",)),
        )
        return await run_to_completion(
            app, RunRequest(agent_name="agent", messages=[Message.text("user", "go")])
        )

    outcome = asyncio.run(run())

    assert outcome.ok, outcome.error
    assert factory.created == 1
    phases = [
        event.payload["phase"]
        for event in outcome.events
        if event.type is EventType.ENVIRONMENT_LIFECYCLE_TRANSITION
    ]
    # Executables need live evidence, so the runner materializes at session
    # start; its evidence is collected and admitted once, as for an eager bind.
    assert phases.count("final_evidence") == 1, phases
    assert phases.count("admission") == 1, phases
    assert (source / "write.txt").read_text() == "out\n"
    # One materialization record for the one runner: created at session start,
    # completed when the first use after admission bound it.
    types = [event.type for event in outcome.events]
    assert types.count(EventType.ENVIRONMENT_MATERIALIZATION_STARTED) == 1
    assert types.count(EventType.ENVIRONMENT_MATERIALIZATION_COMPLETED) == 1
    assert (
        types.index(EventType.ENVIRONMENT_MATERIALIZATION_STARTED)
        < types.index(EventType.ENVIRONMENT_BINDING_STARTED)
        < types.index(EventType.ENVIRONMENT_BINDING_COMPLETED)
        < types.index(EventType.ENVIRONMENT_MATERIALIZATION_COMPLETED)
    )


def test_a_session_start_admission_refusal_copies_nothing_into_the_runner(dirs) -> None:
    source, target = dirs
    (source / "input.txt").write_text("private-workspace-canary\n", encoding="utf-8")
    factory = DeclaredLocalFactory(source, target, withhold_evidence=True)

    store = InMemorySessionStore()

    async def run():
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(ScriptedModelProvider(_turns("write")), default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="sandbox", execution_profile_identity=_identity("test.sandbox")),
            factory,
            default=True,
        )
        app.register_agent(
            AgentSpec(name="agent", model="scripted"),
            tools=[ExecutableRunnerTool()],
            tool_exposure_policy=StaticToolExposurePolicy(profile_id="deferred", tools=("write",)),
        )
        outcome = await run_to_completion(
            app, RunRequest(agent_name="agent", messages=[Message.text("user", "go")])
        )
        return outcome, await store.load_events(outcome.session_id)

    outcome, stored = asyncio.run(run())

    assert not outcome.ok
    # The one materialization record closes as failed: the runner was never bound.
    records = [
        (event.type, event.payload.get("reason"))
        for event in stored
        if event.type
        in {
            EventType.ENVIRONMENT_MATERIALIZATION_STARTED,
            EventType.ENVIRONMENT_MATERIALIZATION_COMPLETED,
            EventType.ENVIRONMENT_MATERIALIZATION_FAILED,
        }
    ]
    assert records == [
        (EventType.ENVIRONMENT_MATERIALIZATION_STARTED, None),
        (EventType.ENVIRONMENT_MATERIALIZATION_FAILED, "not_bound"),
    ], records
    refusals = [
        event.payload
        for event in outcome.events
        if event.type is EventType.ENVIRONMENT_LIFECYCLE_TRANSITION
        and event.payload["phase"] == "admission"
    ]
    assert [payload["outcome"] for payload in refusals] == ["refused"]
    assert factory.created == 1  # created to collect live evidence...
    # ...but refused before anything was copied in or bound.
    assert not (target / "input.txt").exists()
    assert not any(event.type is EventType.ENVIRONMENT_BINDING_STARTED for event in outcome.events)
    assert [runner.is_closed for runner in factory.runners] == [True]  # removed, not leaked


class UndeclaredDeferredFactory(DeferredLocalFactory):
    @property
    def deferred_materialization(self) -> bool:
        return False


def test_a_factory_returning_a_deferred_environment_must_declare_it(dirs) -> None:
    source, target = dirs
    factory = UndeclaredDeferredFactory(source, target)

    async def run():
        return await run_to_completion(
            _app(factory, InMemorySessionStore(), _turns()),
            RunRequest(agent_name="agent", messages=[Message.text("user", "hi")]),
        )

    outcome = asyncio.run(run())

    assert not outcome.ok
    assert "deferred_materialization = True" in str(outcome.error)
    assert factory.created == 0


class _IdleResourceFactory(DeferredLocalFactory):
    def __init__(self, source: Path, target: Path) -> None:
        super().__init__(source, target)
        self.calls: list[str] = []

    async def release_idle_resources(self) -> None:
        self.calls.append("trim")

    async def close_idle_resources(self) -> None:
        self.calls.append("close")


def test_drain_trims_idle_factory_resources_and_only_close_releases_them(dirs) -> None:
    source, target = dirs
    factory = _IdleResourceFactory(source, target)

    async def run():
        app = _app(factory, InMemorySessionStore(), _turns("write"))
        assert await app.drain_environment_cleanups(timeout_s=5)
        outcome = await run_to_completion(
            app, RunRequest(agent_name="agent", messages=[Message.text("user", "go")])
        )
        assert outcome.ok, outcome.error
        assert await app.close_idle_environment_resources()
        return app

    asyncio.run(run())

    assert factory.calls == ["trim", "close"]
    assert factory.created == 1  # the app kept working after an idle drain


class _BlockingIdleFactory(DeferredLocalFactory):
    def __init__(self, source: Path, target: Path) -> None:
        super().__init__(source, target)
        self.calls: list[str] = []
        self.gate: asyncio.Event | None = None
        self.fail = False
        self.fail_kinds: set[str] = set()

    async def _release(self, kind: str) -> None:
        self.calls.append(f"{kind}:start")
        if self.gate is not None:
            await self.gate.wait()
        self.calls.append(f"{kind}:end")
        if self.fail or kind in self.fail_kinds:
            raise RuntimeError("docker rm failed")

    async def release_idle_resources(self) -> None:
        await self._release("trim")

    async def close_idle_resources(self) -> None:
        await self._release("close")


def test_drain_bounds_idle_release_and_keeps_owning_it(dirs) -> None:
    source, target = dirs
    factory = _BlockingIdleFactory(source, target)

    async def run():
        app = _app(factory, InMemorySessionStore(), _turns())
        factory.gate = asyncio.Event()
        loop = asyncio.get_running_loop()
        started = loop.time()
        first = await app.drain_environment_cleanups(timeout_s=0.05)
        elapsed = loop.time() - started
        (_, task) = app._idle_resource_releases["sandbox"]
        still_owned = not task.done() and not task.cancelled()
        # A second drain waits for the same release instead of starting another.
        second = await app.drain_environment_cleanups(timeout_s=0.05)
        factory.gate.set()
        third = await app.drain_environment_cleanups(timeout_s=5)
        return first, elapsed, still_owned, second, third, task

    first, elapsed, still_owned, second, third, task = asyncio.run(run())

    assert first is False and elapsed < 1.0
    assert still_owned and second is False
    assert third is True and task.done() and not task.cancelled()
    assert factory.calls[:2] == ["trim:start", "trim:end"]  # one release, never cancelled


def test_close_is_bounded_runs_after_a_running_trim_and_reports_late_failures(dirs) -> None:
    source, target = dirs
    factory = _BlockingIdleFactory(source, target)

    async def run():
        app = _app(factory, InMemorySessionStore(), _turns())
        factory.gate = asyncio.Event()
        drained = await app.drain_environment_cleanups(timeout_s=0.05)
        closed = await app.close_idle_environment_resources(timeout_s=0.05)
        factory.fail = True
        factory.gate.set()
        await asyncio.sleep(0.05)  # both settle after their callers stopped waiting
        factory.gate = None
        factory.fail = False
        reported = await app.close_idle_environment_resources(timeout_s=5)
        again = await app.close_idle_environment_resources(timeout_s=5)
        return drained, closed, reported, again

    drained, closed, reported, again = asyncio.run(run())

    assert (drained, closed) == (False, False)
    assert factory.calls[:4] == ["trim:start", "trim:end", "close:start", "close:end"]
    assert reported is False  # the late failure is reported once...
    assert again is True  # ...and a clean close afterwards succeeds


def test_a_close_chained_after_a_failed_trim_reports_the_trim_failure(dirs) -> None:
    source, target = dirs
    factory = _BlockingIdleFactory(source, target)

    async def run():
        app = _app(factory, InMemorySessionStore(), _turns())
        factory.gate = asyncio.Event()
        drained = await app.drain_environment_cleanups(timeout_s=0.05)
        factory.fail_kinds = {"trim"}
        asyncio.get_running_loop().call_later(0.05, factory.gate.set)
        closed = await app.close_idle_environment_resources(timeout_s=5)
        factory.fail_kinds = set()
        factory.gate = None
        again = await app.close_idle_environment_resources(timeout_s=5)
        return drained, closed, again

    drained, closed, again = asyncio.run(run())

    assert drained is False
    assert factory.calls[:4] == ["trim:start", "trim:end", "close:start", "close:end"]
    assert closed is False  # the close itself succeeded, but the trim it waited for failed
    assert again is True


def test_application_shutdown_closes_idle_factories_after_trimming(dirs) -> None:
    factory = _BlockingIdleFactory(*dirs)

    async def run():
        app = _app(factory, InMemorySessionStore(), _turns())
        outcome = await app.aclose(timeout_s=1)
        assert outcome.settled
        assert outcome.step("idle_environment_resources").status == "settled"
        assert app.lifecycle_state == "closed"

    asyncio.run(run())

    assert factory.calls == ["trim:start", "trim:end", "close:start", "close:end"]


def test_application_shutdown_retains_blocked_idle_cleanup_until_retry(dirs) -> None:
    factory = _BlockingIdleFactory(*dirs)

    async def run():
        app = _app(factory, InMemorySessionStore(), _turns())
        factory.gate = asyncio.Event()
        first = await app.aclose(timeout_s=0.05)
        assert not first.settled
        assert app.lifecycle_state == "closing"
        assert factory.calls == ["trim:start"]
        _, task = app._idle_resource_releases["sandbox"]
        assert not task.done() and not task.cancelled()

        factory.gate.set()
        second = await app.aclose(timeout_s=1)
        assert second.settled
        assert app.lifecycle_state == "closed"
        assert task.done() and not task.cancelled()
        assert factory.calls[-2:] == ["close:start", "close:end"]

    asyncio.run(run())


class _FlakyCloseRunner(LocalRunner):
    """A runner whose first ``close_failures`` closes fail."""

    def __init__(self, root: Path, *, close_failures: int) -> None:
        super().__init__(root)
        self.close_failures = close_failures
        self.close_attempts = 0

    async def close(self) -> None:
        self.close_attempts += 1
        if self.close_failures:
            self.close_failures -= 1
            raise RuntimeError("container removal failed")
        await super().close()


def _flaky_materialization(dirs, runners: list[_FlakyCloseRunner]) -> DeferredMaterialization:
    target = dirs[1]

    async def materialize(mode):
        runner = _FlakyCloseRunner(target, close_failures=1 if not runners else 0)
        runners.append(runner)
        return runner, SyncBinding(
            target_workspace=LocalWorkspace(target, workspace_id="target"), path=str(target)
        )

    return DeferredMaterialization(materialize, default_cwd=str(target))


async def _placeholder_bind(materialization: DeferredMaterialization, source: Path):
    return await materialization.binding.bind(
        LocalWorkspace(source, workspace_id="source"),
        materialization.runner,
        session_id="session",
        agent_name="agent",
        environment_name="sandbox",
    )


@pytest.mark.parametrize("retry", ["finalize", "wrapper_close"])
def test_an_unbound_runner_whose_close_fails_stays_owned_until_closed(dirs, retry) -> None:
    runners: list[_FlakyCloseRunner] = []
    materialization = _flaky_materialization(dirs, runners)
    records: list[tuple[str, object]] = []

    async def observe(kind, payload, bound) -> None:
        records.append((kind, payload.get("reason")))

    materialization.attach_observer(observe)

    async def run():
        bound = await _placeholder_bind(materialization, dirs[0])
        # Created for an admission collection, then never bound (for example refused).
        await materialization._materialize_once("use", run_admission=False, bind=False)
        with pytest.raises(RuntimeError, match="container removal failed"):
            await materialization.binding.finalize(bound, outcome="interrupted")
        # The failed close must not drop the runner: nothing may use it again...
        assert materialization.materialized_runner is None
        assert not runners[0].is_closed
        # ...but finalization or the wrapper's close still owns and retries it.
        if retry == "finalize":
            await materialization.binding.finalize(bound, outcome="interrupted")
        else:
            await materialization.runner.close()

    asyncio.run(run())

    assert [runner.is_closed for runner in runners] == [True]
    assert runners[0].close_attempts == 2
    # Its one materialization record still ends exactly once.
    assert records == [("started", None), ("failed", "not_bound")], records


def test_a_refused_runner_whose_close_fails_is_never_reused_and_still_closed(dirs) -> None:
    runners: list[_FlakyCloseRunner] = []
    materialization = _flaky_materialization(dirs, runners)
    refusals = iter([True, False])

    async def admit(runner) -> None:
        if next(refusals):
            raise RuntimeError("refused by policy")

    materialization.attach_admission(admit)

    async def run():
        bound = await _placeholder_bind(materialization, dirs[0])
        with pytest.raises(EnvironmentMaterializationError):
            await materialization.materialize("use")
        assert not runners[0].is_closed  # its close failed
        # The next use gets a fresh runner, never the refused one.
        second = await materialization.materialize("use")
        assert second is runners[1]
        # Finalization retries the refused runner's close before anything else.
        await materialization.binding.finalize(bound, outcome="completed")
        assert runners[0].is_closed
        await materialization.runner.close()

    asyncio.run(run())

    assert [runner.is_closed for runner in runners] == [True, True]
    assert runners[0].close_attempts == 2
