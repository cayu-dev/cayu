from __future__ import annotations

import asyncio

import pytest

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.environments import (
    EnvironmentFactoryRequest,
    EnvironmentSpec,
    ExecutionRequirements,
    ExecutionToolRequirement,
    LocalWorkspaceEnvironmentFactory,
    NativeBinding,
    evaluate_execution_admission,
    require_allowed_workspace_root,
    workspace_root_from_metadata,
)
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.runners import DockerRunner, LocalRunner
from cayu.tools import SearchTextTool, Tool, ToolEffect, ToolResult, ToolSpec


def _request(root, **kwargs):
    return EnvironmentFactoryRequest(
        session_id="workspace",
        agent_name="reader",
        environment_name="local",
        metadata={"workspace_root": str(root)},
        **kwargs,
    )


def test_concurrent_runs_read_only_their_own_workspace(tmp_path):
    roots = [tmp_path / name for name in ("one", "two")]
    for root in roots:
        root.mkdir()
        (root / "owner.txt").write_text(root.name)

    async def scenario():
        entered = set()
        both_entered = asyncio.Event()
        observed = {}

        class ReadOwn(Tool):
            spec = ToolSpec(name="read_own", effect=ToolEffect.NONE)

            async def run(self, ctx, arguments):
                entered.add(ctx.session_id)
                if len(entered) == 2:
                    both_entered.set()
                await asyncio.wait_for(both_entered.wait(), 10)
                observed[ctx.session_id] = (await ctx.workspace.read_bytes("owner.txt")).content
                with pytest.raises(ValueError):
                    await ctx.workspace.read_bytes("../owner.txt")
                return ToolResult(content=observed[ctx.session_id].decode())

        def response(request):
            if any(message.role == "tool" for message in request.messages):
                return [ModelStreamEvent.completed()]
            return [
                ModelStreamEvent.tool_call(id="read", name="read_own", arguments={}),
                ModelStreamEvent.completed(),
            ]

        app = CayuApp(enable_logging=False)
        app.register_provider(ScriptedModelProvider(response_factory=response), default=True)
        app.register_agent(AgentSpec(name="reader", model="scripted"), tools=[ReadOwn()])
        app.register_environment_factory(
            EnvironmentSpec(name="local"),
            LocalWorkspaceEnvironmentFactory(
                allowed_roots=[tmp_path],
                authorize_root=lambda request, root: root == tmp_path / request.session_id,
            ),
            default=True,
        )

        async def run(root):
            return [
                event
                async for event in app.run(
                    RunRequest(
                        session_id=root.name,
                        agent_name="reader",
                        messages=[Message.text("user", "read my file")],
                        metadata={"workspace_root": str(root)},
                    )
                )
            ]

        batches = await asyncio.wait_for(asyncio.gather(*(run(root) for root in roots)), 15)
        assert observed == {"one": b"one", "two": b"two"}
        assert all(any(e.type is EventType.SESSION_COMPLETED for e in batch) for batch in batches)
        assert not any(e.type is EventType.SESSION_FAILED for batch in batches for e in batch)

    asyncio.run(scenario())


def test_local_factory_forwards_search_admission_for_the_selected_root(tmp_path):
    requirements = ExecutionRequirements.trusted(
        tool_requirements=tuple(
            ExecutionToolRequirement(tool_name="search_text", requirement=requirement)
            for requirement in SearchTextTool().spec.execution_requirements
        )
    )
    request = _request(tmp_path, execution_requirements=requirements)
    factory = LocalWorkspaceEnvironmentFactory(
        allowed_roots=[tmp_path], root_from_request=lambda request: tmp_path
    )
    candidate = factory.execution_admission_candidate(request)
    expected = LocalRunner(tmp_path).execution_admission_candidate_for(requirements)
    assert candidate.candidate == expected.candidate
    assert candidate.evidence.environment_fingerprint == expected.evidence.environment_fingerprint
    assert (
        evaluate_execution_admission(
            candidate=candidate.candidate,
            requirements=requirements,
            evidence=candidate.evidence,
            stage="pre_create",
        ).status
        == "admitted"
    )
    result = asyncio.run(factory.create(request))
    assert result.environment.runner.root == tmp_path.resolve()
    assert result.environment.workspace.root == tmp_path.resolve()


def test_factory_supports_attached_docker_runner_and_guest_binding(tmp_path):
    runner = DockerRunner(
        "prepared-container",
        docker_path="/usr/bin/docker",
        default_cwd="/work",
        _container_id="a" * 64,
    )
    selected = []

    def runner_factory(root, request):
        selected.append((root, request.session_id))
        return runner

    factory = LocalWorkspaceEnvironmentFactory(
        allowed_roots=[tmp_path],
        root_from_request=lambda request: tmp_path,
        runner_factory=runner_factory,
        binding_factory=lambda root, request: NativeBinding(default_path="/work"),
    )
    request = _request(tmp_path)
    assert factory.execution_admission_candidate(
        request
    ) == runner.execution_admission_candidate_for(request.execution_requirements)
    result = asyncio.run(factory.create(request))
    assert result.environment.runner is runner
    bound = asyncio.run(
        result.environment.binding.bind(
            result.environment.workspace,
            runner,
            session_id=request.session_id,
        )
    )
    assert bound.path == "/work"
    assert selected == [(tmp_path.resolve(), "workspace")] * 2


def test_root_selection_and_allowed_root_check_are_reusable(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    request = _request(second)
    assert workspace_root_from_metadata(request) == second
    assert require_allowed_workspace_root(second, allowed_roots=[first, second]) == second
    factory = LocalWorkspaceEnvironmentFactory(
        allowed_roots=[second],
        root_from_request=lambda request: second,
    )
    assert factory.resolve_root(request) == second


@pytest.mark.parametrize("kind", ["outside", "symlink", "missing", "relative", "unset"])
def test_invalid_root_is_refused_before_runner_creation(tmp_path, kind):
    allowed, outside = tmp_path / "allowed", tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    link = allowed / "escape"
    link.symlink_to(outside, target_is_directory=True)
    root = {
        "outside": outside,
        "symlink": link,
        "missing": allowed / "missing",
        "relative": "relative",
        "unset": "",
    }[kind]
    calls = []
    factory = LocalWorkspaceEnvironmentFactory(
        allowed_roots=[allowed],
        authorize_root=lambda request, resolved: resolved == allowed / request.session_id,
        runner_factory=lambda *args: calls.append(args),
    )
    with pytest.raises(ValueError) as error:
        factory.execution_admission_candidate(_request(root))
    assert "workspace root" in str(error.value).lower()
    if kind in {"outside", "symlink"}:
        assert str(outside) in str(error.value)
    assert calls == []
    assert not (allowed / "missing").exists()


def test_metadata_selection_requires_explicit_ownership_authorization(tmp_path):
    with pytest.raises(ValueError, match="requires authorize_root"):
        LocalWorkspaceEnvironmentFactory(allowed_roots=[tmp_path])


@pytest.mark.parametrize("target", ["victim", "parent"])
def test_allowed_parent_does_not_authorize_a_foreign_workspace(tmp_path, target):
    own = tmp_path / "workspace"
    victim = tmp_path / "victim"
    own.mkdir()
    victim.mkdir()
    (victim / "secret.txt").write_text("private")
    calls = []
    factory = LocalWorkspaceEnvironmentFactory(
        allowed_roots=[tmp_path],
        authorize_root=lambda request, root: root == tmp_path / request.session_id,
        runner_factory=lambda *args: calls.append(args),
    )
    request = _request(victim if target == "victim" else tmp_path)
    with pytest.raises(ValueError, match="not authorized"):
        factory.execution_admission_candidate(request)
    with pytest.raises(ValueError, match="not authorized"):
        asyncio.run(factory.create(request))
    assert calls == []


@pytest.mark.parametrize("action", ["discard", "preserve"])
def test_unadopted_external_runner_attachment_does_not_close_its_owner(tmp_path, action):
    from cayu.environments.factory import EnvironmentFactoryReleaseAction
    from cayu.runtime._environment_lifecycle import _release_unclaimed_factory_result

    class OwnedRunner(LocalRunner):
        closed = False

        async def close(self):
            self.closed = True

    async def scenario():
        runner = OwnedRunner(tmp_path)
        factory = LocalWorkspaceEnvironmentFactory(
            allowed_roots=[tmp_path],
            root_from_request=lambda request: tmp_path,
            runner_factory=lambda root, request: runner,
        )
        request = _request(tmp_path)
        result = await factory.create(request)
        await _release_unclaimed_factory_result(
            result,
            action=EnvironmentFactoryReleaseAction(action),
            original_error=RuntimeError("setup interrupted before checkpoint"),
        )
        assert not runner.closed
        retried = await factory.create(request)
        assert retried.environment.runner is runner
        assert not runner.closed

    asyncio.run(scenario())
