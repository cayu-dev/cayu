"""Credential-free allocation lifetime qualification for issue #1412."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest
from tests.docker_toolchain import docker_toolchain_profile
from tests.environments.test_docker_coding_live import _configuration_or_skip

from cayu import (
    AgentSpec,
    CayuApp,
    DockerCodingEnvironmentFactory,
    DockerImageIdentity,
    EnvironmentSpec,
    EventType,
    ExecutionProfileBehaviorIdentity,
    ImmutableInputStore,
    LocalWorkspace,
    Message,
    ModelStreamEvent,
    ResumeRequest,
    RunRequest,
    ScriptedModelProvider,
    SQLiteSessionStore,
    SQLiteTaskStore,
    TaskCreate,
    TaskHandlerOutcome,
    TaskQuery,
    TaskStatus,
    Tool,
    ToolEffect,
    ToolResult,
    ToolSpec,
    inspect_local_immutable_input,
    run_task_worker,
)

pytestmark = pytest.mark.process


@pytest.mark.parametrize("pending_disposal", [False, True])
def test_model_step_handoff_recreates_docker_without_repeating_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pending_disposal: bool,
    record_property,
) -> None:
    docker_path, image, image_id = _configuration_or_skip()
    identity = DockerImageIdentity(reference=image, content_digest=image_id)
    architecture = subprocess.check_output(
        [docker_path, "image", "inspect", "--format", "{{.Architecture}}", image],
        text=True,
    ).strip()
    source = tmp_path / "workspace"
    source.mkdir()
    inputs = tmp_path / "input"
    inputs.mkdir()
    (inputs / "data").write_text("immutable")
    projection = inspect_local_immutable_input(
        inputs,
        target_path="/evidence",
        policy_fingerprint="sha256:" + "a" * 64,
        runtime_compatibility_fingerprint=identity.fingerprint,
        authorization_scope_fingerprint="sha256:" + "c" * 64,
    )
    store = ImmutableInputStore(tmp_path / "managed")
    results = []
    effect_path = tmp_path / "effects"

    class Effect(Tool):
        spec = ToolSpec(
            name="record",
            description="Record one completed effect.",
            input_schema={"type": "object", "properties": {}},
            effect=ToolEffect.EXTERNAL,
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name="tests:docker-lifetime:effect",
                behavior_version="1",
                implementation_version="1",
            ),
        )

        async def run(self, ctx, args):
            with effect_path.open("a") as handle:
                handle.write("effect\n")
            return ToolResult(content="recorded")

    class Factory(DockerCodingEnvironmentFactory):
        async def create_recoverable(self, request, allocation):
            result = await super().create_recoverable(request, allocation)
            results.append(result)
            return result

    def app(scripts):
        sessions = SQLiteSessionStore(tmp_path / "sessions.db")
        tasks = SQLiteTaskStore(tmp_path / "tasks.db")
        runtime = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
        runtime.register_provider(ScriptedModelProvider(scripts), default=True)
        runtime.register_agent(AgentSpec(name="worker", model="scripted-model"), tools=[Effect()])
        runtime.register_environment_factory(
            EnvironmentSpec(
                name="coding",
                execution_profile_identity=ExecutionProfileBehaviorIdentity(
                    name="tests:lifetime:environment",
                    behavior_version="1",
                    implementation_version="1",
                ),
            ),
            Factory(
                source_workspace=LocalWorkspace(source, workspace_id="lifetime-workspace"),
                toolchain_profile=docker_toolchain_profile(
                    image_identity=identity,
                    platform_architecture=architecture,
                ),
                immutable_inputs=(projection,),
                immutable_input_store=store,
                immutable_input_runtime_compatibility_fingerprint=identity.fingerprint,
                docker_path=docker_path,
            ),
            default=True,
        )
        return runtime, sessions, tasks

    def absent(container_id):
        assert (
            subprocess.check_output(
                [
                    docker_path,
                    "ps",
                    "-a",
                    "--no-trunc",
                    "--filter",
                    f"id={container_id}",
                    "--format",
                    "{{.ID}}",
                ],
                text=True,
                timeout=15,
            ).strip()
            == ""
        )

    async def run():
        first, sessions, tasks = app(
            [
                [
                    ModelStreamEvent.tool_call(id="effect-once", name="record", arguments={}),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
            ]
        )
        if pending_disposal:
            # Exercise recovery from the durable state between confirmed physical
            # disposal and retirement, without rerunning the completed tool effect.
            async def defer_retirement(*args, **kwargs):
                pass

            monkeypatch.setattr(
                first._environment_lifecycle, "_retire_disposed_allocation", defer_retirement
            )
        await first.create_task(TaskCreate(task_id="job", type="job", assigned_agent_name="worker"))

        async def start(runtime, task, worker_id):
            async for _ in runtime.run(
                RunRequest(
                    agent_name="worker",
                    session_id="session",
                    task_id=task.id,
                    task_worker_id=worker_id,
                    task_lease_expires_at=task.lease_expires_at,
                    messages=[Message.text("user", "Record then finish")],
                    max_steps=1,
                )
            ):
                pass
            return TaskHandlerOutcome.SESSION_INTERRUPTED

        assert (
            await run_task_worker(
                first,
                tasks,
                start,
                worker_id="first",
                max_tasks=1,
                reclaim=False,
            )
            == 1
        )
        events = await sessions.load_events("session")
        interrupted = [event for event in events if event.type is EventType.SESSION_INTERRUPTED]
        assert len(interrupted) == 1
        assert interrupted[0].payload["limit"] == "model_steps"
        assert any(event.type is EventType.TASK_INTERRUPTED_HANDOFF for event in events)
        assert effect_path.read_text() == "effect\n"
        assert len(results) == 1
        absent(results[0].metadata["container_id"])
        assert store.inspect()[0].reference_count == 0
        checkpoint = await sessions.load_checkpoint("session")
        assert bool(checkpoint.get("environment_factory_pending_disposals")) is pending_disposal
        await tasks.close()
        await sessions.close()

        recovered, sessions, tasks = app(
            [
                [
                    ModelStreamEvent.text_delta("Done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            ]
        )
        continuation = (
            await tasks.claim_interrupted_task_continuation(
                "second",
                TaskQuery(type="job"),
                handoff_id="continue",
            )
        ).task
        assert continuation is not None
        async for _ in recovered.resume(
            ResumeRequest(
                session_id="session",
                task_worker_id="second",
                task_handoff_id=continuation.interrupted_handoff_id,
                messages=[Message.text("user", "Continue and finish")],
                max_steps=1,
            )
        ):
            pass
        task = await tasks.load_task("job")
        assert task.status is TaskStatus.COMPLETED, task.error
        assert len(results) == 2
        assert results[0].metadata["container_id"] != results[1].metadata["container_id"]
        assert (
            results[0].reconnect_metadata["allocation_id"]
            != results[1].reconnect_metadata["allocation_id"]
        )
        assert effect_path.read_text() == "effect\n"
        for result in results:
            absent(result.metadata["container_id"])
        assert store.inspect()[0].reference_count == 0
        assert store.collect(projection.projection.fingerprint)
        await tasks.close()
        await sessions.close()

    asyncio.run(asyncio.wait_for(run(), timeout=90))
    record_property("tasks_completed", 1)
    record_property("tool_calls", 1)
    record_property("remaining_owned_containers", 0)
    record_property("immutable_input_references", 0)


async def _allocation_process(root: Path, action: str, metadata: dict):
    """Fresh interpreter entry point; only public Runtime factory/store APIs."""
    docker_path, image, image_id = _configuration_or_skip()
    identity = DockerImageIdentity(reference=image, content_digest=image_id)
    projection = inspect_local_immutable_input(
        root / "input",
        target_path="/evidence",
        policy_fingerprint="sha256:" + "a" * 64,
        runtime_compatibility_fingerprint=identity.fingerprint,
        authorization_scope_fingerprint="sha256:" + "c" * 64,
    )
    factory = DockerCodingEnvironmentFactory(
        source_workspace=LocalWorkspace(root / "workspace", workspace_id="process-workspace"),
        toolchain_profile=docker_toolchain_profile(
            image_identity=identity,
            platform_architecture=subprocess.check_output(
                [docker_path, "image", "inspect", "--format", "{{.Architecture}}", image],
                text=True,
            ).strip(),
        ),
        immutable_inputs=(projection,),
        immutable_input_store=ImmutableInputStore(root / "managed"),
        immutable_input_runtime_compatibility_fingerprint=identity.fingerprint,
        docker_path=docker_path,
    )
    from cayu import EnvironmentFactoryOperation, EnvironmentFactoryRequest

    request = EnvironmentFactoryRequest(
        session_id="process-session",
        agent_name="worker",
        environment_name="coding",
        operation=EnvironmentFactoryOperation.RECONNECT
        if metadata
        else EnvironmentFactoryOperation.CREATE,
        reconnect_metadata=metadata.get("reconnect", {}),
    )
    if action == "dispose":
        await factory.recover_finalization_disposal(
            request,
            {
                "version": 1,
                "kind": "docker_coding_disposal",
                "container_id": metadata["metadata"]["container_id"],
                "attachment_ids": [
                    value["attachment_id"] for value in metadata["metadata"]["immutable_inputs"]
                ],
            },
        )
        return {}
    result = await factory.create(request)
    return {"metadata": result.metadata, "reconnect": result.reconnect_metadata}


def test_fresh_concurrent_processes_share_and_replace_exact_allocation(
    tmp_path: Path, record_property
) -> None:
    import json
    import sys

    docker_path, _, _ = _configuration_or_skip()
    (tmp_path / "workspace").mkdir()
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "data").write_text("shared")
    script = """
import asyncio, json, sys
from pathlib import Path
from tests.environments.test_docker_allocation_lifetime_live import _allocation_process
print(json.dumps(asyncio.run(_allocation_process(Path(sys.argv[1]), sys.argv[2], json.loads(sys.argv[3])))))
"""
    children = []
    retained = []

    def start(action="create", metadata=None):
        child = subprocess.Popen(
            [sys.executable, "-c", script, str(tmp_path), action, json.dumps(metadata or {})],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        children.append(child)
        return child

    def finish(child):
        stdout, stderr = child.communicate(timeout=60)
        assert child.returncode == 0, stderr
        return json.loads(stdout)

    def absent(metadata):
        assert (
            subprocess.check_output(
                [
                    docker_path,
                    "ps",
                    "-a",
                    "--no-trunc",
                    "--filter",
                    "id=" + metadata["metadata"]["container_id"],
                    "--format",
                    "{{.ID}}",
                ],
                text=True,
                timeout=15,
            ).strip()
            == ""
        )

    store = ImmutableInputStore(tmp_path / "managed")
    try:
        first_workers = [start(), start()]
        for worker in first_workers:
            retained.append(finish(worker))
        assert retained[0]["reconnect"] == retained[1]["reconnect"]
        assert store.inspect()[0].reference_count == 1
        finish(start("dispose", retained[0]))
        absent(retained[0])
        assert store.inspect()[0].reference_count == 0
        successor = finish(start())
        retained.append(successor)
        assert successor["reconnect"]["container_id"] != retained[0]["reconnect"]["container_id"]
        assert successor["reconnect"]["allocation_id"] != retained[0]["reconnect"]["allocation_id"]
        assert successor["metadata"]["immutable_inputs"][0]["reused"]
        finish(start("dispose", retained[0]))
        assert store.inspect()[0].reference_count == 1
        finish(start("dispose", successor))
        absent(successor)
        assert store.inspect()[0].reference_count == 0
        assert store.collect(store.inspect()[0].projection_fingerprint)
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=10)
        for metadata in retained:
            finish(start("dispose", metadata))

    record_property("remaining_owned_containers", 0)
    record_property("immutable_input_references", 0)
