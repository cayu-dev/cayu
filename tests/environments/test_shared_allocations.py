"""Sessions of one allocation scope share one factory allocation."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.environments import (
    AllocationSharingScope,
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    EnvironmentSpec,
    NativeBinding,
)
from cayu.environments.process_ownership import current_process_local_owner
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.storage import InMemoryKnowledgeStore, KnowledgeAccessScope
from cayu.workflows import StepError, StepRunOptions, WorkflowBase, WorkflowSpec, step
from cayu.workspaces import LocalWorkspace


def _text(delta: str) -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.text_delta(delta),
        ModelStreamEvent.completed({"finish_reason": "stop"}),
    ]


class _RecordingBinding(NativeBinding):
    def __init__(self, log: list[tuple[str, str | None]], failures: list[int]) -> None:
        super().__init__()
        self._log = log
        self._failures = failures

    async def bind(self, workspace, runner, *, session_id, **options):
        self._log.append(("bind", session_id))
        return await super().bind(workspace, runner, session_id=session_id, **options)

    def abandon(self, bound):
        self._log.append(("abandon", None))
        return super().abandon(bound)

    async def finalize(self, bound, *, outcome=None, metadata=None):
        self._log.append(("finalize", outcome))
        if self._failures and self._failures[0] > 0:
            self._failures[0] -= 1
            raise RuntimeError("copy-back conflict")
        return await super().finalize(bound, outcome=outcome, metadata=metadata)


class _RecordingFactory(EnvironmentFactory):
    """A factory that counts allocations, binds and finalizations."""

    def __init__(self, root: Path, *, finalize_failures: int = 0) -> None:
        self.root = root
        self.requests: list[EnvironmentFactoryRequest] = []
        self.log: list[tuple[str, str | None]] = []
        self.discards = 0
        self._failures = [finalize_failures]

    @property
    def secret_resolution_scope(self):
        return "static"

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        self.requests.append(request)

        async def release(action: EnvironmentFactoryReleaseAction) -> None:
            if action is EnvironmentFactoryReleaseAction.DISCARD:
                self.discards += 1

        return EnvironmentFactoryResult(
            environment=Environment(
                EnvironmentSpec(name=request.environment_name),
                workspace=LocalWorkspace(self.root, workspace_id="shared-root"),
                binding=_RecordingBinding(self.log, self._failures),
            ),
            release=release,
        )

    def finalizations(self) -> list[str | None]:
        return [outcome for kind, outcome in self.log if kind == "finalize"]


def _owners(factory: _RecordingFactory) -> list[str]:
    """The scope owners a factory's shared allocations were requested for."""

    return [request.session_id.split(".shared-")[0] for request in factory.requests]


def _app(
    tmp_path: Path,
    *,
    turns: int,
    allocation_scope: str = "session",
    finalize_failures: int = 0,
) -> tuple[CayuApp, _RecordingFactory]:
    factory = _RecordingFactory(tmp_path, finalize_failures=finalize_failures)
    app = CayuApp(enable_logging=False)
    app.register_provider(
        ScriptedModelProvider([_text(f"turn {index}") for index in range(turns)]),
        default=True,
    )
    app.register_environment_factory(
        EnvironmentSpec(name="shared"),
        factory,
        default=True,
        allocation_scope=allocation_scope,
    )
    for name in ("planner", "solver"):
        app.register_agent(AgentSpec(name=name, model="scripted"))
    return app, factory


class _PlanThenSolve(WorkflowBase):
    spec = WorkflowSpec(name="plan-then-solve")

    def __init__(self, app, *, options=None, observe=None, fail_after_plan=False) -> None:
        super().__init__(app)
        self._options = options
        self._observe = observe or (lambda label: None)
        self._fail_after_plan = fail_after_plan

    async def run(self, session_id: str):
        ctx = self.context(session_id)
        yield await ctx.start()
        await step(ctx, agent="planner", step_id="plan", prompt="plan", run_options=self._options)
        self._observe("after_plan")
        if self._fail_after_plan:
            raise RuntimeError("workflow failed after planning")
        await step(ctx, agent="solver", step_id="solve", prompt="solve", run_options=self._options)
        self._observe("after_solve")
        yield await ctx.completed()


def test_steps_share_one_allocation_released_when_the_workflow_completes(tmp_path):
    app, factory = _app(tmp_path, turns=2, allocation_scope="parent")
    observed: dict[str, list[str | None]] = {}

    def observe(label: str) -> None:
        observed[label] = factory.finalizations()

    async def scenario():
        workflow = _PlanThenSolve(app, observe=observe)
        return [event async for event in workflow.run("run-shared")]

    events = asyncio.run(scenario())
    assert events[-1].type == EventType.WORKFLOW_COMPLETED
    # One allocation for both steps, requested for the workflow run and owned
    # by this process.
    assert len(factory.requests) == 1
    (request,) = factory.requests
    # Named after the workflow run, but fresh for this allocation, so another
    # process never reattaches it.
    assert request.session_id.startswith("run-shared.shared-")
    assert request.parent_session_id is None
    assert request.process_local_owner == current_process_local_owner()
    assert factory.log[0] == ("bind", request.session_id)
    # Neither step's end released it; the workflow's completion did.
    assert observed == {"after_plan": [], "after_solve": []}
    assert factory.finalizations() == ["completed"]


def test_the_default_scope_keeps_one_allocation_per_step(tmp_path):
    app, factory = _app(tmp_path, turns=2)
    observed: dict[str, list[str | None]] = {}

    async def scenario():
        workflow = _PlanThenSolve(
            app, observe=lambda label: observed.setdefault(label, factory.finalizations())
        )
        return [event async for event in workflow.run("run-default")]

    events = asyncio.run(scenario())
    assert events[-1].type == EventType.WORKFLOW_COMPLETED
    assert len(factory.requests) == 2
    assert {request.session_id for request in factory.requests} != {"run-default"}
    assert observed["after_plan"] == ["completed"]
    assert factory.finalizations() == ["completed", "completed"]


def test_a_step_option_shares_an_environment_registered_per_session(tmp_path):
    app, factory = _app(tmp_path, turns=2)

    async def scenario():
        workflow = _PlanThenSolve(app, options=StepRunOptions(allocation_scope="parent"))
        return [event async for event in workflow.run("run-option")]

    events = asyncio.run(scenario())
    assert events[-1].type == EventType.WORKFLOW_COMPLETED
    assert _owners(factory) == ["run-option"]
    assert factory.finalizations() == ["completed"]


def test_a_failed_release_is_kept_and_retried(tmp_path):
    app, factory = _app(tmp_path, turns=2, allocation_scope="parent", finalize_failures=1)

    async def scenario():
        workflow = _PlanThenSolve(app)
        with pytest.raises(RuntimeError, match="copy-back conflict"):
            async for _ in workflow.run("run-retry"):
                pass
        # The allocation is still held; the next release finalizes it.
        assert app._environment_lifecycle.shared_allocations.live_keys()
        await app._environment_lifecycle.shared_allocations.end_owner(
            "run-retry", outcome="completed"
        )
        assert not app._environment_lifecycle.shared_allocations.live_keys()

    asyncio.run(scenario())
    assert factory.finalizations() == ["completed", "completed"]
    assert factory.discards == 0


def test_execute_releases_a_run_that_raised(tmp_path):
    app, factory = _app(tmp_path, turns=1, allocation_scope="causal_budget")

    async def scenario():
        workflow = _PlanThenSolve(app, fail_after_plan=True)
        with pytest.raises(RuntimeError, match="workflow failed after planning"):
            async for _ in workflow.execute("run-raise"):
                pass

    asyncio.run(scenario())
    assert _owners(factory) == ["run-raise"]
    assert factory.finalizations() == ["failed"]


def test_shutdown_discards_a_release_that_keeps_failing(tmp_path):
    app, factory = _app(tmp_path, turns=1, allocation_scope="parent", finalize_failures=5)

    async def scenario():
        workflow = _PlanThenSolve(app, fail_after_plan=True)
        with pytest.raises(RuntimeError, match="workflow failed after planning"):
            async for _ in workflow.run("run-shutdown"):
                pass
        assert factory.finalizations() == []
        assert await app.close_idle_environment_resources() is False
        assert not app._environment_lifecycle.shared_allocations.live_keys()

    asyncio.run(scenario())
    assert factory.finalizations() == ["interrupted"]
    assert factory.discards == 1


def test_an_agent_session_owns_its_scope_until_its_run_ends(tmp_path):
    app, factory = _app(tmp_path, turns=1, allocation_scope="parent")

    async def scenario():
        return [
            event
            async for event in app.run(
                RunRequest(
                    session_id="agent-root",
                    agent_name="planner",
                    messages=[Message.text("user", "plan")],
                )
            )
        ]

    events = asyncio.run(scenario())
    assert events[-1].type == EventType.SESSION_COMPLETED, events[-1].payload
    assert _owners(factory) == ["agent-root"]
    # The root's own binding detached; its terminal event released the allocation.
    assert factory.finalizations() == ["completed"]
    assert not app._environment_lifecycle.shared_allocations.live_keys()


def test_scope_values_are_validated():
    with pytest.raises(ValueError, match="allocation_scope must be one of"):
        StepRunOptions(allocation_scope="sibling")
    assert "allocation_scope" not in StepRunOptions().model_dump()
    assert StepRunOptions(allocation_scope="parent").allocation_scope is (
        AllocationSharingScope.PARENT
    )


def test_registration_refuses_an_unknown_scope(tmp_path):
    app = CayuApp(enable_logging=False)
    with pytest.raises(ValueError, match="allocation_scope must be one of"):
        app.register_environment_factory(
            EnvironmentSpec(name="shared"), _RecordingFactory(tmp_path), allocation_scope="tree"
        )


def test_a_session_whose_scope_owner_is_not_running_is_refused(tmp_path):
    app, factory = _app(tmp_path, turns=2, allocation_scope="parent")

    async def scenario():
        parent = [
            event
            async for event in app.run(
                RunRequest(
                    session_id="finished-parent",
                    agent_name="planner",
                    messages=[Message.text("user", "plan")],
                )
            )
        ]
        assert parent[-1].type == EventType.SESSION_COMPLETED
        return [
            event
            async for event in app.run(
                RunRequest(
                    session_id="late-child",
                    agent_name="solver",
                    parent_session_id="finished-parent",
                    messages=[Message.text("user", "solve")],
                )
            )
        ]

    events = asyncio.run(scenario())
    assert events[-1].type == EventType.SESSION_FAILED
    assert "is not running in this process" in str(events[-1].payload)
    # Only the parent's own allocation, released at its end, was ever created.
    assert _owners(factory) == ["finished-parent"]
    assert factory.finalizations() == ["completed"]


def test_run_metadata_cannot_opt_a_session_into_sharing(tmp_path):
    app, factory = _app(tmp_path, turns=2)

    async def scenario():
        for session_id in ("first", "second"):
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="planner",
                        messages=[Message.text("user", "plan")],
                        # What an earlier draft read; a model can put it in
                        # subagent metadata, so it must mean nothing.
                        metadata={"cayu.environment.allocation_scope": "causal_budget"},
                    )
                )
            ]
            assert events[-1].type == EventType.SESSION_COMPLETED

    asyncio.run(scenario())
    assert [request.session_id for request in factory.requests] == ["first", "second"]


class _GatedFactory(_RecordingFactory):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.started = asyncio.Event()
        self.gate = asyncio.Event()

    async def create(self, request):
        self.started.set()
        await self.gate.wait()
        return await super().create(request)


def test_sessions_waiting_on_creation_never_outlive_their_owner(tmp_path):
    from cayu.workflows import parallel

    class Parallel(WorkflowBase):
        spec = WorkflowSpec(name="parallel-shared")

        async def run(self, session_id):
            ctx = self.context(session_id)
            yield await ctx.start()
            outcome = await parallel(
                [
                    step(ctx, agent="planner", step_id="a", prompt="a"),
                    step(ctx, agent="solver", step_id="b", prompt="b"),
                ]
            )
            yield ctx.event(EventType.WORKFLOW_COMPLETED, payload={"ok": outcome.ok})

    async def scenario():
        factory = _GatedFactory(tmp_path)
        app = CayuApp(enable_logging=False)
        app.register_provider(
            ScriptedModelProvider(response_factory=lambda request: _text("done")), default=True
        )
        app.register_environment_factory(
            EnvironmentSpec(name="shared"), factory, default=True, allocation_scope="parent"
        )
        for name in ("planner", "solver"):
            app.register_agent(AgentSpec(name=name, model="scripted"))
        registry = app._environment_lifecycle.shared_allocations
        task = asyncio.ensure_future(asyncio.wait_for(_collect(Parallel(app).run("run-race")), 30))
        await asyncio.wait_for(factory.started.wait(), 30)
        # The owner ends while one step creates and the other waits for it.
        await registry.end_owner("run-race", outcome="interrupted")
        factory.gate.set()
        events = await task
        return factory, registry, events

    factory, registry, events = asyncio.run(scenario())
    assert events[-1].payload["ok"] is False
    # One creation, discarded; the waiting step didn't create another.
    assert len(factory.requests) == 1
    assert factory.discards == 1
    assert registry.live_keys() == ()


async def _collect(stream):
    return [event async for event in stream]


class _KnowledgeFactory(_RecordingFactory):
    async def create(self, request):
        result = await super().create(request)
        environment = result.environment
        store = InMemoryKnowledgeStore()
        return EnvironmentFactoryResult(
            environment=Environment(
                environment.spec,
                workspace=environment.workspace,
                binding=environment.binding,
                knowledge_store=store,
                knowledge_access_scope=KnowledgeAccessScope.for_namespace(
                    f"private:{request.agent_name}"
                ),
            ),
            release=result.release,
        )


def test_an_allocation_whose_authority_depends_on_the_request_is_not_shared(tmp_path):
    factory = _KnowledgeFactory(tmp_path)
    app = CayuApp(enable_logging=False)
    app.register_provider(ScriptedModelProvider([_text("plan")]), default=True)
    app.register_environment_factory(
        EnvironmentSpec(name="shared"), factory, default=True, allocation_scope="parent"
    )
    app.register_agent(AgentSpec(name="planner", model="scripted"))

    async def scenario():
        return [
            event
            async for event in app.run(
                RunRequest(
                    session_id="knowledge-root",
                    agent_name="planner",
                    messages=[Message.text("user", "plan")],
                )
            )
        ]

    events = asyncio.run(scenario())
    assert events[-1].type == EventType.SESSION_FAILED
    assert "knowledge store or scope" in str(events[-1].payload)
    assert factory.discards == 1
    assert app._environment_lifecycle.shared_allocations.live_keys() == ()


def test_each_shared_allocation_gets_a_fresh_identity(tmp_path):
    identities = []
    for _ in range(2):
        # The same run id in two applications, as when a run resumes in a new
        # process after its first one ended.
        app, factory = _app(tmp_path, turns=2, allocation_scope="parent")

        async def scenario(app=app):
            async for _event in _PlanThenSolve(app).run("run-again"):
                pass

        asyncio.run(scenario())
        assert _owners(factory) == ["run-again"]
        identities.append(factory.requests[0].session_id)
    assert identities[0] != identities[1]


class _ArtifactFactory(_RecordingFactory):
    async def create(self, request):
        from cayu.artifacts.local import LocalArtifactStore

        result = await super().create(request)
        environment = result.environment
        return EnvironmentFactoryResult(
            environment=Environment(
                environment.spec,
                workspace=environment.workspace,
                binding=environment.binding,
                artifact_store=LocalArtifactStore(
                    self.root / f"artifacts-{request.agent_name}",
                    store_id=f"private-{request.agent_name}",
                ),
            ),
            release=result.release,
        )


def test_an_allocation_with_a_request_specific_artifact_store_is_not_shared(tmp_path):
    factory = _ArtifactFactory(tmp_path)
    app = CayuApp(enable_logging=False)
    app.register_provider(ScriptedModelProvider([_text("plan")]), default=True)
    app.register_environment_factory(
        EnvironmentSpec(name="shared"), factory, default=True, allocation_scope="parent"
    )
    app.register_agent(AgentSpec(name="planner", model="scripted"))

    async def scenario():
        return [
            event
            async for event in app.run(
                RunRequest(
                    session_id="artifact-root",
                    agent_name="planner",
                    messages=[Message.text("user", "plan")],
                )
            )
        ]

    events = asyncio.run(scenario())
    assert events[-1].type == EventType.SESSION_FAILED
    assert "artifact store" in str(events[-1].payload)
    assert factory.discards == 1


class _StalledBinding(_RecordingBinding):
    def __init__(self, log, failures, gate: asyncio.Event) -> None:
        super().__init__(log, failures)
        self._gate = gate

    async def finalize(self, bound, *, outcome=None, metadata=None):
        await self._gate.wait()
        return await super().finalize(bound, outcome=outcome, metadata=metadata)


def test_closing_respects_its_timeout_and_keeps_the_running_release(tmp_path):
    app, factory = _app(tmp_path, turns=1, allocation_scope="parent")

    async def scenario():
        gate = asyncio.Event()
        workflow = _PlanThenSolve(app, fail_after_plan=True)
        with pytest.raises(RuntimeError, match="workflow failed after planning"):
            async for _ in workflow.run("run-stalled"):
                pass
        registry = app._environment_lifecycle.shared_allocations
        (entry,) = registry._live.values()
        entry.result.environment.binding.__class__ = _StalledBinding
        entry.result.environment.binding._gate = gate
        started = asyncio.get_running_loop().time()
        assert await app.close_idle_environment_resources(timeout_s=0.2) is False
        assert asyncio.get_running_loop().time() - started < 5
        # The copy-back still runs; a later close waits for that same release.
        assert registry.live_keys()
        gate.set()
        assert await app.close_idle_environment_resources(timeout_s=5) is True
        assert registry.live_keys() == ()

    asyncio.run(scenario())
    assert factory.finalizations() == ["interrupted"]


class _FailOnceFactory(_RecordingFactory):
    """Fails its first create after the second step started waiting for it."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.gate = asyncio.Event()

    async def create(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            await self.gate.wait()
            raise RuntimeError("post-create probe failed; the allocation stays reserved")
        self.requests.pop()
        return await super().create(request)


def test_a_retry_after_a_failed_creation_reuses_its_allocation_identity(tmp_path):
    from cayu.workflows import parallel

    class Parallel(WorkflowBase):
        spec = WorkflowSpec(name="parallel-retry")

        async def run(self, session_id):
            ctx = self.context(session_id)
            yield await ctx.start()
            outcome = await parallel(
                [
                    step(ctx, agent="planner", step_id="a", prompt="a"),
                    step(ctx, agent="solver", step_id="b", prompt="b"),
                ]
            )
            yield await ctx.completed({"failures": len(outcome.failures)})

    async def scenario():
        factory = _FailOnceFactory(tmp_path)
        app = CayuApp(enable_logging=False)
        app.register_provider(
            ScriptedModelProvider(response_factory=lambda request: _text("done")), default=True
        )
        app.register_environment_factory(
            EnvironmentSpec(name="shared"), factory, default=True, allocation_scope="parent"
        )
        for name in ("planner", "solver"):
            app.register_agent(AgentSpec(name=name, model="scripted"))
        registry = app._environment_lifecycle.shared_allocations
        task = asyncio.ensure_future(
            asyncio.wait_for(_collect(Parallel(app).run("run-retry-id")), 30)
        )
        while not registry._creating:
            await asyncio.sleep(0.01)
        # Let the other step reach the creation it waits for, then fail it.
        await asyncio.sleep(0.2)
        factory.gate.set()
        events = await task
        return factory, events

    factory, events = asyncio.run(scenario())
    assert events[-1].payload["failures"] == 1
    # The failed attempt and the retry asked for the same allocation, so the
    # retry adopts (or cleans up) whatever the failure left reserved.
    failed, adopted = factory.requests
    assert failed.session_id == adopted.session_id
    assert adopted.session_id.startswith("run-retry-id.shared-")
    assert factory.finalizations() == ["completed"]


def test_a_workflow_steps_subagent_joins_the_steps_allocation(tmp_path):
    from cayu.tools import SubagentTool

    factory = _RecordingFactory(tmp_path)
    app = CayuApp(enable_logging=False)
    planner = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="spawn", name="subagent", arguments={"agent": "helper", "task": "help"}
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            _text("planned"),
        ],
        name="planner-provider",
    )
    helper = ScriptedModelProvider([_text("helped")], name="helper-provider")
    app.register_provider(planner, default=True)
    app.register_provider(helper)
    app.register_environment_factory(
        EnvironmentSpec(name="shared"), factory, default=True, allocation_scope="parent"
    )
    app.register_agent(AgentSpec(name="helper", model="scripted", provider_name=helper.name))
    app.register_agent(
        AgentSpec(name="planner", model="scripted", provider_name=planner.name),
        tools=[SubagentTool(app, agents={"helper": "helper"})],
    )

    class PlanWithHelper(WorkflowBase):
        spec = WorkflowSpec(name="plan-with-helper")

        async def run(self, session_id):
            ctx = self.context(session_id)
            yield await ctx.start()
            await step(ctx, agent="planner", step_id="plan", prompt="plan")
            yield await ctx.completed()

    async def scenario():
        return [event async for event in PlanWithHelper(app).run("run-subagent")]

    events = asyncio.run(scenario())
    assert events[-1].type == EventType.WORKFLOW_COMPLETED
    assert helper.requests, "the subagent ran"
    # The step and its subagent used the run's one allocation, copied back once.
    assert _owners(factory) == ["run-subagent"]
    assert factory.finalizations() == ["completed"]


def test_an_older_attempt_ending_leaves_the_newer_attempts_allocation(tmp_path):
    app, factory = _app(tmp_path, turns=2, allocation_scope="parent")
    paused = asyncio.Event()
    resume = asyncio.Event()

    class PlanThenWait(WorkflowBase):
        spec = WorkflowSpec(name="plan-then-wait")

        async def run(self, session_id):
            ctx = self.context(session_id)
            yield await ctx.start()
            await step(ctx, agent="planner", step_id="plan", prompt="plan")
            paused.set()
            await resume.wait()
            yield await ctx.completed()

    async def scenario():
        workflow = PlanThenWait(app)
        task = asyncio.ensure_future(_collect(workflow.execute("run-takeover")))
        await asyncio.wait_for(paused.wait(), 30)
        # A newer attempt takes the run over in this process, on the same
        # workflow instance, as a re-dispatch would.
        newer = workflow.context("run-takeover")
        await newer.start()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        registry = app._environment_lifecycle.shared_allocations
        assert registry.live_keys(), "the newer attempt keeps the sandbox"
        assert factory.finalizations() == []
        await newer.completed()
        assert registry.live_keys() == ()

    asyncio.run(scenario())
    assert factory.finalizations() == ["completed"]


def test_closing_bounds_a_stalled_discard_after_a_failed_release(tmp_path):
    app, factory = _app(tmp_path, turns=1, allocation_scope="parent", finalize_failures=5)
    gate = asyncio.Event()

    async def stalled_release(action):
        await gate.wait()
        factory.discards += 1

    async def scenario():
        workflow = _PlanThenSolve(app, fail_after_plan=True)
        with pytest.raises(RuntimeError, match="workflow failed after planning"):
            async for _ in workflow.run("run-stalled-discard"):
                pass
        registry = app._environment_lifecycle.shared_allocations
        (entry,) = registry._live.values()
        object.__setattr__(entry.result, "release", stalled_release)
        loop = asyncio.get_running_loop()
        started = loop.time()
        assert await app.close_idle_environment_resources(timeout_s=0.2) is False
        assert loop.time() - started < 5
        assert registry.live_keys(), "the running discard stays owned"
        gate.set()
        # A later close waits for that same discard; it doesn't release again.
        assert await app.close_idle_environment_resources(timeout_s=5) is True
        assert registry.live_keys() == ()

    asyncio.run(scenario())
    assert factory.finalizations() == ["interrupted"]
    assert factory.discards == 1


def test_a_failed_discard_stays_owned_and_is_retried(tmp_path):
    app, factory = _app(tmp_path, turns=1, allocation_scope="parent", finalize_failures=5)
    attempts = []

    async def flaky_release(action):
        attempts.append(action)
        if len(attempts) == 1:
            raise RuntimeError("docker API unavailable")

    async def scenario():
        workflow = _PlanThenSolve(app, fail_after_plan=True)
        with pytest.raises(RuntimeError, match="workflow failed after planning"):
            async for _ in workflow.run("run-flaky-discard"):
                pass
        registry = app._environment_lifecycle.shared_allocations
        (entry,) = registry._live.values()
        object.__setattr__(entry.result, "release", flaky_release)
        assert await app.close_idle_environment_resources(timeout_s=5) is False
        # Still owned, so its runner stays in the exit hook and a later close
        # retries the discard instead of re-running the failed copy-back.
        assert registry.live_keys()
        assert await app.close_idle_environment_resources(timeout_s=5) is True
        assert registry.live_keys() == ()

    asyncio.run(scenario())
    assert len(attempts) == 2
    assert factory.finalizations() == ["interrupted"]
    # Once disposed, the binding's retained state (a SyncBinding's source
    # reservation, for example) is released.
    assert factory.log[-1] == ("abandon", None)


class _BindFailsFactory(_RecordingFactory):
    """Creates an allocation whose copy-in fails and whose first discard fails too."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.discard_attempts = 0

    async def create(self, request):
        result = await super().create(request)

        class _CopyInFails(_RecordingBinding):
            async def bind(self, workspace, runner, *, session_id, **options):
                raise RuntimeError("copy-in failed")

        factory = self

        async def release(action):
            factory.discard_attempts += 1
            if factory.discard_attempts == 1:
                raise RuntimeError("docker API unavailable")

        return EnvironmentFactoryResult(
            environment=Environment(
                result.environment.spec,
                workspace=result.environment.workspace,
                binding=_CopyInFails(self.log, [0]),
            ),
            release=release,
        )


def test_a_failed_creation_whose_discard_fails_stays_owned(tmp_path):
    factory = _BindFailsFactory(tmp_path)
    app = CayuApp(enable_logging=False)
    app.register_provider(ScriptedModelProvider([_text("plan")]), default=True)
    app.register_environment_factory(
        EnvironmentSpec(name="shared"), factory, default=True, allocation_scope="parent"
    )
    app.register_agent(AgentSpec(name="planner", model="scripted"))
    app.register_agent(AgentSpec(name="solver", model="scripted"))
    registry = app._environment_lifecycle.shared_allocations

    async def scenario():
        with pytest.raises(StepError, match="copy-in failed"):
            async for _ in _PlanThenSolve(app).run("run-copy-in"):
                pass
        # The allocation exists but couldn't be discarded: it stays owned, so
        # shutdown can see it and its owner's end retries the discard.
        assert factory.discard_attempts == 1
        assert registry.holds_environment("shared")
        await registry.end_owner("run-copy-in", outcome="failed")
        assert factory.discard_attempts == 2
        assert not registry.holds_environment("shared")
        assert await app.close_idle_environment_resources(timeout_s=5) is True

    asyncio.run(scenario())


def test_an_owner_ending_before_creation_starts_gets_no_allocation(tmp_path):
    from cayu.runtime._shared_environment_allocations import (
        SharedAllocationKey,
        SharedAllocationScopeClosedError,
    )

    app, factory = _app(tmp_path, turns=0, allocation_scope="parent")
    registry = app._environment_lifecycle.shared_allocations
    key = SharedAllocationKey(
        scope=AllocationSharingScope.PARENT,
        scope_id="run-gen",
        owner_session_id="run-gen",
        environment_name="shared",
    )
    request = EnvironmentFactoryRequest(
        session_id="run-gen.step", agent_name="planner", environment_name="shared"
    )

    async def scenario():
        registry.open_owner("run-gen")
        acquiring = asyncio.ensure_future(registry._acquire(key, factory, request, None))
        # Let the acquire register its creation, but not run it yet.
        while key not in registry._creating:
            await asyncio.sleep(0)
        await registry.end_owner("run-gen", outcome="interrupted")
        with pytest.raises(SharedAllocationScopeClosedError):
            await acquiring

    asyncio.run(scenario())
    # The creation ran for an owner that had already ended, so it was discarded
    # instead of published.
    assert registry.live_keys() == ()
    assert factory.discards == len(factory.requests)


class _CopyBackBinding(_RecordingBinding):
    """A binding whose completed work exists only until it copies back."""

    def _completion_requires_successful_finalization(self, bound):
        return True


class _CopyBackFactory(_RecordingFactory):
    async def create(self, request):
        result = await super().create(request)
        environment = result.environment
        return EnvironmentFactoryResult(
            environment=Environment(
                environment.spec,
                workspace=environment.workspace,
                binding=_CopyBackBinding(self.log, self._failures),
            ),
            release=result.release,
        )


def _copy_back_app(tmp_path: Path, *, turns: int) -> tuple[CayuApp, _CopyBackFactory]:
    factory = _CopyBackFactory(tmp_path)
    app = CayuApp(enable_logging=False)
    app.register_provider(
        ScriptedModelProvider([_text(f"turn {index}") for index in range(turns)]),
        default=True,
    )
    app.register_environment_factory(
        EnvironmentSpec(name="shared"), factory, default=True, allocation_scope="parent"
    )
    for name in ("planner", "solver"):
        app.register_agent(AgentSpec(name=name, model="scripted"))
    return app, factory


def test_an_agent_session_cannot_own_a_copy_back_allocation(tmp_path):
    app, factory = _copy_back_app(tmp_path, turns=1)

    async def scenario():
        return [
            event
            async for event in app.run(
                RunRequest(
                    session_id="agent-copy-back",
                    agent_name="planner",
                    messages=[Message.text("user", "plan")],
                )
            )
        ]

    events = asyncio.run(scenario())
    # The agent's success would be published before its scope's copy-back, so
    # a failed copy-back couldn't fail it: sharing is refused up front.
    assert events[-1].type == EventType.SESSION_FAILED
    assert "only a workflow run can own" in str(events[-1].payload)
    assert factory.discards == 1
    assert not app._environment_lifecycle.shared_allocations.live_keys()


def test_a_workflow_run_can_own_a_copy_back_allocation(tmp_path):
    app, factory = _copy_back_app(tmp_path, turns=2)

    async def scenario():
        return await _collect(_PlanThenSolve(app).run("run-copy-back"))

    events = asyncio.run(scenario())
    assert events[-1].type == EventType.WORKFLOW_COMPLETED
    assert _owners(factory) == ["run-copy-back"]
    assert factory.finalizations() == ["completed"]


def test_a_forks_descendants_never_resolve_to_the_sources_causal_scope(tmp_path):
    from cayu.sessions import ForkSessionRequest

    app, _ = _app(tmp_path, turns=1)
    registry = app._environment_lifecycle.shared_allocations

    async def scenario():
        source = [
            event
            async for event in app.run(
                RunRequest(
                    session_id="fork-source",
                    agent_name="planner",
                    messages=[Message.text("user", "plan")],
                )
            )
        ]
        assert source[-1].type == EventType.SESSION_COMPLETED
        forked = [
            event
            async for event in app.fork_session(
                ForkSessionRequest(source_session_id="fork-source", session_id="fork-child")
            )
        ]
        assert forked[-1].type == EventType.SESSION_FORKED
        fork = await app.session_store.load("fork-child")
        assert fork is not None and fork.causal_budget_id is not None
        # A subagent the fork spawns: child of the fork, same causal budget.
        helper = fork.model_copy(
            update={"id": "fork-helper", "parent_session_id": fork.id, "metadata": {}}
        )
        key = await registry.resolve_key(helper, "shared", AllocationSharingScope.CAUSAL_BUDGET)
        source_key = await registry.resolve_key(
            await app.session_store.load("fork-source"),
            "shared",
            AllocationSharingScope.CAUSAL_BUDGET,
        )
        return key, source_key

    key, source_key = asyncio.run(scenario())
    assert key.owner_session_id == "fork-child"
    assert key != source_key


class _OwnerEndsOnPublishFactory(_RecordingFactory):
    """Ends the scope's owner right after the allocation is published."""

    registry = None

    async def create(self, request):
        result = await super().create(request)
        owner = request.session_id.split(".shared-")[0]
        registry = self.registry

        async def end_soon():
            await asyncio.sleep(0)
            await registry.end_owner(owner, outcome="interrupted")

        self.ending = asyncio.ensure_future(end_soon())
        return result


def test_the_creating_session_rechecks_its_owner_after_creation(tmp_path):
    from cayu.runtime._shared_environment_allocations import (
        SharedAllocationKey,
        SharedAllocationScopeClosedError,
    )

    factory = _OwnerEndsOnPublishFactory(tmp_path)
    app = CayuApp(enable_logging=False)
    registry = app._environment_lifecycle.shared_allocations
    factory.registry = registry
    key = SharedAllocationKey(
        scope=AllocationSharingScope.PARENT,
        scope_id="run-recheck",
        owner_session_id="run-recheck",
        environment_name="shared",
    )
    request = EnvironmentFactoryRequest(
        session_id="run-recheck.step", agent_name="planner", environment_name="shared"
    )

    async def scenario():
        registry.open_owner("run-recheck")
        step_session = SimpleNamespace(id="run-recheck.step", run_epoch=0)
        acquiring = asyncio.ensure_future(registry._acquire(key, factory, request, step_session))
        # The owner ends after the creation task publishes, before the
        # creating session resumes.
        with pytest.raises(SharedAllocationScopeClosedError):
            await acquiring
        await factory.ending

    asyncio.run(scenario())
    assert factory.finalizations() == ["interrupted"]
    assert registry.live_keys() == ()


class _RetainingBinding(_RecordingBinding):
    """Refuses abandonment while a failed finalize's cleanup is pending, like Docker coding."""

    def __init__(self, log, failures) -> None:
        super().__init__(log, failures)
        self.pending_cleanup = False

    async def finalize(self, bound, *, outcome=None, metadata=None):
        if self.pending_cleanup:
            if self.factory.discards:
                # Like Docker coding marking released immutable inputs closing.
                raise RuntimeError("attachment already released by the discard")
            self._log.append(("cleanup-retry", outcome))
            self.pending_cleanup = False
            return None
        self._log.append(("finalize", outcome))
        self.pending_cleanup = True
        raise RuntimeError("container removal failed after copy-back")

    def abandon(self, bound):
        self._log.append(("abandon", None))
        return not self.pending_cleanup


class _RetainingFactory(_RecordingFactory):
    async def create(self, request):
        result = await super().create(request)
        environment = result.environment
        binding = _RetainingBinding(self.log, self._failures)
        binding.factory = self
        return EnvironmentFactoryResult(
            environment=Environment(
                environment.spec,
                workspace=environment.workspace,
                binding=binding,
            ),
            release=result.release,
        )


def test_a_discard_finishes_the_bindings_pending_cleanup(tmp_path):
    factory = _RetainingFactory(tmp_path)
    app = CayuApp(enable_logging=False)
    app.register_provider(ScriptedModelProvider([_text("plan")]), default=True)
    app.register_environment_factory(
        EnvironmentSpec(name="shared"), factory, default=True, allocation_scope="parent"
    )
    app.register_agent(AgentSpec(name="planner", model="scripted"))
    app.register_agent(AgentSpec(name="solver", model="scripted"))
    registry = app._environment_lifecycle.shared_allocations

    async def scenario():
        workflow = _PlanThenSolve(app, fail_after_plan=True)
        with pytest.raises(RuntimeError, match="workflow failed after planning"):
            async for _ in workflow.run("run-retained"):
                pass
        # Shutdown: the release (finalize) fails after copy-back, the discard
        # disposes of the container, and the binding's refused abandonment is
        # finished by its own cleanup retry instead of being forgotten.
        assert await app.close_idle_environment_resources(timeout_s=5) is False
        return registry.live_keys(), registry.holds_environment("shared")

    live, held = asyncio.run(scenario())
    assert factory.discards == 1
    assert ("cleanup-retry", "interrupted") in factory.log
    assert live == () and held is False
