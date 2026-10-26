"""register_agent(environment_name=...) picks a new run's environment."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest

from cayu import AgentSpec, CayuApp, EventType, Message, ResumeRequest, RunRequest
from cayu.artifacts import LocalArtifactStore
from cayu.environments import NO_ENVIRONMENT, Environment, EnvironmentSpec
from cayu.evals import (
    CorpusExecutionLimits,
    CorpusTarget,
    EvalCase,
    EvalOutcome,
    EvalScenarioDocumentV2,
    ScenarioInitialInputEventV2,
    ScenarioInputV2,
    ScenarioTextPartV2,
    ScenarioUserMessageV2,
    ScriptedModelProvider,
    WorkspaceFileExists,
    preflight_eval_scenario,
    run_eval_case,
)
from cayu.evals.benchmark_campaign import _materialize_inputs
from cayu.evals.benchmark_package import benchmark_package_scenarios, load_benchmark_package
from cayu.evals.benchmark_synthetic import write_synthetic_benchmark_package
from cayu.providers import ModelStreamEvent
from cayu.runtime import check_manifest
from cayu.sessions import (
    ForkSessionRequest,
    InMemorySessionStore,
    SessionQuery,
    SessionStatus,
)
from cayu.tasks import DispatchRequest, InMemoryTaskStore, TaskStoreDispatcher
from cayu.tools import SubagentSpec, SubagentTool
from cayu.workspaces.local import LocalWorkspace


def _text(text: str) -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.text_delta(text),
        ModelStreamEvent.completed({"finish_reason": "stop"}),
    ]


def _spawn(agent: str) -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.tool_call(
            id=f"call_{agent}",
            name="subagent",
            arguments={"agent": agent, "task": "help"},
        ),
        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
    ]


def _app(
    responses: list[list[ModelStreamEvent]],
    *,
    store: InMemorySessionStore | None = None,
    **kwargs,
) -> CayuApp:
    app = CayuApp(session_store=store or InMemorySessionStore(), enable_logging=False, **kwargs)
    app.register_provider(ScriptedModelProvider(responses), default=True)
    app.register_environment(Environment(EnvironmentSpec(name="coding")), default=True)
    app.register_environment(Environment(EnvironmentSpec(name="filing")))
    return app


async def _run(app: CayuApp, request: RunRequest | ResumeRequest) -> list:
    stream = app.resume(request) if isinstance(request, ResumeRequest) else app.run(request)
    return [event async for event in stream]


def _environment(app: CayuApp, session_id: str) -> str | None:
    session = asyncio.run(app.session_store.load(session_id))
    assert session is not None
    assert session.status is SessionStatus.COMPLETED
    return session.environment_name


def test_request_then_agent_default_then_application_default() -> None:
    app = _app([_text("done")] * 5)
    app.register_agent(AgentSpec(name="coder", model="scripted-model"))
    app.register_agent(
        AgentSpec(name="filer", model="scripted-model"), environment_name=NO_ENVIRONMENT
    )
    app.register_agent(AgentSpec(name="named", model="scripted-model"), environment_name="filing")

    cases = (
        ("coder", None, "coding"),
        ("filer", None, None),
        ("named", None, "filing"),
        ("filer", "filing", "filing"),
        ("named", "coding", "coding"),
    )
    for index, (agent, requested, expected) in enumerate(cases):
        session_id = f"sess_{index}"
        events = asyncio.run(
            _run(
                app,
                RunRequest(
                    agent_name=agent,
                    session_id=session_id,
                    environment_name=requested,
                    messages=[Message.text("user", "go")],
                ),
            )
        )
        assert events[-1].type is EventType.SESSION_COMPLETED
        assert _environment(app, session_id) == expected, (agent, requested)


def test_resume_dispatch_and_fork_keep_the_session_environment() -> None:
    tasks = InMemoryTaskStore()
    dispatcher = TaskStoreDispatcher(tasks)
    app = _app(
        [_text("first"), _text("resumed"), _text("queued"), _text("forked")],
        task_store=tasks,
        dispatcher=dispatcher,
    )
    app.register_agent(
        AgentSpec(name="filer", model="scripted-model"), environment_name=NO_ENVIRONMENT
    )
    first = RunRequest(
        agent_name="filer",
        session_id="sess_filer",
        messages=[Message.text("user", "file it")],
    )

    async def scenario() -> None:
        assert (await _run(app, first))[-1].type is EventType.SESSION_COMPLETED
        resumed = await _run(
            app,
            ResumeRequest(session_id="sess_filer", messages=[Message.text("user", "again")]),
        )
        assert resumed[-1].type is EventType.SESSION_COMPLETED
        await dispatcher.submit(
            app,
            DispatchRequest(session_id="sess_filer", messages=[Message.text("user", "queued")]),
        )
        handle = await dispatcher.process_next(app, worker_id="worker")
        assert handle is not None
        assert handle.status.value == "completed"
        async for _event in app.fork_session(
            ForkSessionRequest(source_session_id="sess_filer", session_id="sess_fork")
        ):
            pass
        forked = await _run(
            app,
            ResumeRequest(session_id="sess_fork", messages=[Message.text("user", "fork")]),
        )
        assert forked[-1].type is EventType.SESSION_COMPLETED

    asyncio.run(scenario())

    assert _environment(app, "sess_filer") is None
    assert _environment(app, "sess_fork") is None


def test_subagent_child_of_a_parent_without_environment_gets_none() -> None:
    app = _app([_spawn("helper"), _text("helped"), _text("parent done")])
    app.register_agent(
        AgentSpec(name="filer", model="scripted-model"),
        tools=[SubagentTool(app, agents={"helper": SubagentSpec(agent_name="helper")})],
        environment_name=NO_ENVIRONMENT,
    )
    app.register_agent(AgentSpec(name="helper", model="scripted-model"))

    events = asyncio.run(
        _run(
            app,
            RunRequest(
                agent_name="filer",
                session_id="sess_parent",
                messages=[Message.text("user", "delegate")],
            ),
        )
    )

    assert events[-1].type is EventType.SESSION_COMPLETED
    (child,) = asyncio.run(
        app.session_store.list_sessions(SessionQuery(parent_session_id="sess_parent"))
    ).sessions
    assert _environment(app, "sess_parent") is None
    # The child inherits its parent's environment, not the application default.
    assert _environment(app, child.id) is None


def test_subagent_child_inherits_the_parent_environment_over_its_agent_default() -> None:
    app = _app([_spawn("filer"), _text("filed"), _text("parent done")])
    app.register_agent(
        AgentSpec(name="coder", model="scripted-model"),
        tools=[SubagentTool(app, agents={"filer": SubagentSpec(agent_name="filer")})],
    )
    app.register_agent(
        AgentSpec(name="filer", model="scripted-model"), environment_name=NO_ENVIRONMENT
    )

    events = asyncio.run(
        _run(
            app,
            RunRequest(
                agent_name="coder",
                session_id="sess_coder",
                messages=[Message.text("user", "delegate")],
            ),
        )
    )

    assert events[-1].type is EventType.SESSION_COMPLETED
    (child,) = asyncio.run(
        app.session_store.list_sessions(SessionQuery(parent_session_id="sess_coder"))
    ).sessions
    assert _environment(app, "sess_coder") == "coding"
    assert _environment(app, child.id) == "coding"


def test_unregistered_agent_default_fails_the_run_and_the_check() -> None:
    app = _app([_text("unused")])
    app.register_agent(AgentSpec(name="lost", model="scripted-model"), environment_name="missing")

    with pytest.raises(KeyError, match="Environment not registered: missing"):
        asyncio.run(
            _run(
                app,
                RunRequest(agent_name="lost", messages=[Message.text("user", "go")]),
            )
        )
    (finding,) = [
        item
        for item in check_manifest(app.describe()).diagnostics
        if item.code == "AGENT_ENVIRONMENT_NOT_FOUND"
    ]
    assert finding.parameters == {"agent": "lost", "environment": "missing"}
    assert finding.documentation_anchor == "cayu guide diagnostics#agent-environment-not-found"


def test_manifest_reports_only_agent_owned_defaults() -> None:
    app = _app([])
    app.register_agent(AgentSpec(name="coder", model="scripted-model"))
    app.register_agent(
        AgentSpec(name="filer", model="scripted-model"), environment_name=NO_ENVIRONMENT
    )
    app.register_agent(AgentSpec(name="named", model="scripted-model"), environment_name="filing")

    agents = {agent["name"]: agent for agent in app.describe().model_dump(mode="json")["agents"]}

    assert "default_environment" not in agents["coder"]
    assert "runs_without_environment" not in agents["coder"]
    assert agents["filer"]["runs_without_environment"] is True
    assert "default_environment" not in agents["filer"]
    assert agents["named"]["default_environment"] == "filing"


@pytest.mark.parametrize(
    ("value", "error"),
    [("", ValueError), (" coding", ValueError), (object(), TypeError), (1, TypeError)],
)
def test_register_agent_rejects_invalid_environment_names(
    value: object, error: type[Exception]
) -> None:
    app = _app([])
    with pytest.raises(error):
        app.register_agent(
            AgentSpec(name="bad", model="scripted-model"),
            environment_name=value,  # ty: ignore[invalid-argument-type]
        )


def test_eval_probes_do_not_read_the_default_workspace_for_an_agent_without_one(
    tmp_path: Path,
) -> None:
    (tmp_path / "result.txt").write_text("unrelated", encoding="utf-8")
    app = CayuApp(enable_logging=False)
    app.register_provider(ScriptedModelProvider([_text("done")]), default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="coding"), workspace=LocalWorkspace(tmp_path)),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="filer", model="scripted-model"), environment_name=NO_ENVIRONMENT
    )
    case = EvalCase(
        id="no-workspace",
        request=RunRequest(agent_name="filer", messages=[Message.text("user", "go")], max_steps=1),
        assertions=[WorkspaceFileExists("result.txt")],
    )

    result = asyncio.run(run_eval_case(app, case, suite_id="s"))

    (assertion,) = result.trials[0].assertions
    assert assertion.outcome is EvalOutcome.UNAVAILABLE


def _scenario_target(app: CayuApp, agent_name: str) -> CorpusTarget:
    return CorpusTarget(
        key="assistant.default",
        app=app,
        request_base=RunRequest(agent_name=agent_name, messages=[], max_steps=8),
        application_release_id="release-current",
        limits=CorpusExecutionLimits(max_trials=4, max_concurrency=2, max_timeout_seconds=600),
    )


def test_scenario_preflight_binds_the_agent_default_environment() -> None:
    app = _app([])
    app.register_agent(AgentSpec(name="coder", model="scripted-model"))
    app.register_agent(
        AgentSpec(name="filer", model="scripted-model"), environment_name=NO_ENVIRONMENT
    )
    app.register_agent(AgentSpec(name="named", model="scripted-model"), environment_name="filing")
    scenario = EvalScenarioDocumentV2.create(
        id="retained-request",
        target_key="assistant.default",
        name="Retained request",
        events=(
            ScenarioInitialInputEventV2(
                sequence=0,
                id="initial",
                input=ScenarioInputV2.create(
                    (ScenarioUserMessageV2.create([ScenarioTextPartV2(text="Go.")]),)
                ),
            ),
        ),
    )

    def bound_environment(agent_name: str) -> str | None:
        result = asyncio.run(
            preflight_eval_scenario(
                scenario, _scenario_target(app, agent_name), actor_authorized=True
            )
        )
        assert result.ready is True, result.diagnostics
        assert result.binding is not None
        return result.binding.environment_name

    assert bound_environment("coder") == "coding"
    assert bound_environment("filer") is None
    assert bound_environment("named") == "filing"


def test_benchmark_inputs_go_to_the_agent_default_environment(tmp_path: Path) -> None:
    loaded = load_benchmark_package(write_synthetic_benchmark_package(tmp_path / "package"))
    default_store = LocalArtifactStore(tmp_path / "default", store_id="default")
    agent_store = LocalArtifactStore(tmp_path / "agent", store_id="agent")
    app = CayuApp(enable_logging=False)
    app.register_provider(ScriptedModelProvider([]), default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="coding"), artifact_store=default_store), default=True
    )
    app.register_environment(
        Environment(EnvironmentSpec(name="filing"), artifact_store=agent_store)
    )
    app.register_agent(AgentSpec(name="named", model="scripted-model"), environment_name="filing")
    app.register_agent(
        AgentSpec(name="filer", model="scripted-model"), environment_name=NO_ENVIRONMENT
    )
    scenarios = benchmark_package_scenarios(loaded.package)

    references = asyncio.run(_materialize_inputs(loaded, _scenario_target(app, "named"), scenarios))

    (artifact_id,) = [item for refs in references.values() for item in refs.values()]
    content = loaded.file_contents[0][1]
    stored = asyncio.run(agent_store.read_bytes(artifact_id, max_bytes=len(content)))
    assert hashlib.sha256(stored.content).digest() == hashlib.sha256(content).digest()
    with pytest.raises(FileNotFoundError):
        asyncio.run(default_store.read_bytes(artifact_id, max_bytes=len(content)))
    with pytest.raises(ValueError, match="existing environment with an ArtifactStore"):
        asyncio.run(_materialize_inputs(loaded, _scenario_target(app, "filer"), scenarios))
