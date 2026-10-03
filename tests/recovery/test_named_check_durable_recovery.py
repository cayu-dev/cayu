"""Public recovery integration with real processes and a controlled Docker transport.

Only Docker admission/transport is substituted; Runtime, tool journaling,
SQLite, guest supervisor, planner, executor and signals are the shipped owners.
Actual Docker/PostgreSQL qualification is separate and remains required.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
from tests.runners.test_docker_durable_commands import install_transport, make_runner

from cayu import (
    AgentSpec,
    CayuApp,
    DockerCodingCommandAuthority,
    DockerCodingEnvironmentFactory,
    DockerCodingToolchainProfile,
    DockerImageIdentity,
    Environment,
    EnvironmentSpec,
    ExecCommand,
    ExecutionProfileBehaviorIdentity,
    LocalWorkspace,
    Message,
    ModelStreamEvent,
    NamedCheck,
    ProcessCommandPolicy,
    ReadFileTool,
    RunCheckTool,
    RunRequest,
    ScriptedModelProvider,
    SQLiteSessionStore,
    SQLiteTaskStore,
    StaticToolPolicy,
    TaskCreate,
    TaskQuery,
    ToolExecutableRequirement,
    run_task_worker,
)
from cayu.runtime import (
    RecoveryDecision,
    RecoveryExecutionRequest,
    RecoveryPlanAction,
    RecoveryPlanRequest,
    RecoveryPlanSelection,
)

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux supervisor and SIGKILL")


class RecoveryProvider(ScriptedModelProvider):
    @property
    def execution_profile_identity(self):
        return ExecutionProfileBehaviorIdentity(
            name="tests.named-check-recovery", behavior_version="1", implementation_version="1"
        )


def build_app(root: Path, patch):
    import cayu.tools.named_checks as named_checks

    source = root / "source"
    source.mkdir(exist_ok=True)
    install_transport(patch, root)
    runner = make_runner(source)
    runner._runtime_evidence = replace(
        runner._runtime_evidence,
        required_executables=tuple(sorted(("python3", sys.executable))),
        executable_availability=tuple(sorted((("python3", True), (sys.executable, True)))),
        executable_probes=(ToolExecutableRequirement(executable=sys.executable),),
        valid_until=runner._runtime_evidence.observed_at + timedelta(seconds=60),
    )
    code = (
        "import pathlib,time\n"
        "pathlib.Path('checked.txt').write_text('retained-command-effect')\n"
        f"pathlib.Path({str(root / 'started')!r}).touch()\n"
        f"while not pathlib.Path({str(root / 'release')!r}).exists(): time.sleep(0.01)\n"
        "print('checked-once')\n"
    )
    authority = DockerCodingCommandAuthority(
        selector="test",
        revision="1",
        description="Test",
        exposure="named_check",
        executable=sys.executable,
        fixed_arguments=("-c", code),
        max_arguments=0,
        dependency_sensitive=False,
        timeout_seconds=30,
        max_output_bytes=1024,
    )
    profile = DockerCodingToolchainProfile(
        profile_id="durable-test",
        revision="1",
        image_identity=DockerImageIdentity(reference="test@sha256:" + "a" * 64),
        platform_architecture="amd64",
        command_authorities=(authority,),
    )

    async def admitted(*args, **kwargs):
        return None

    patch.setattr(named_checks, "ensure_docker_coding_toolchain_runner_admission", admitted)

    async def collect_admission():
        return runner.execution_admission_candidate()

    patch.setattr(runner, "collect_execution_admission_candidate", collect_admission)
    check = NamedCheck(
        name="test",
        description="Test",
        command=ExecCommand.process(sys.executable, "-c", code),
        timeout_s=30,
        max_output_bytes=1024,
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests.durable-check", behavior_version="1", implementation_version="1"
        ),
    )
    target = source
    factory = None
    if (root / "factory-mode").exists():
        import cayu.environments.docker_coding as docker_coding

        target = root / "guest"
        target.mkdir(exist_ok=True)
        factory = DockerCodingEnvironmentFactory(
            source_workspace=LocalWorkspace(source, workspace_id="source"),
            toolchain_profile=profile,
            docker_path="/usr/bin/true",
        )
        runner.default_cwd = str(target)
        vars(runner)["_cayu_execution_environment_authority"] = (
            factory.execution_environment_authority()
        )

        async def allocated(_name, *, requirements, **kwargs):
            runner._runtime_evidence = replace(
                runner._runtime_evidence,
                default_cwd=str(target),
                toolchain_profile_fingerprint=profile.fingerprint,
                required_executables=requirements.executable_names(),
                executable_availability=tuple(
                    (name, True) for name in requirements.executable_names()
                ),
                executable_probes=requirements.executable_probes(),
            )
            return runner

        async def create(name, **kwargs):
            assert not (root / "allocated").exists(), "Recovery must not create another guest"
            (root / "allocated").touch()
            return await allocated(name, **kwargs)

        async def reconnect(container, **kwargs):
            assert container == runner.container_id
            assert (root / "allocated").exists()
            if (root / "cancel-reconnect").exists():
                (root / "reconnect-waiting").touch()
                while not (root / "release-reconnect").exists():
                    await asyncio.sleep(0.01)
            (root / "reconnected").touch()
            return await allocated(container, **kwargs)

        async def close():
            (root / "disposed").touch()

        patch.setattr(factory, "_create_or_recover_runner", create)
        patch.setattr(factory, "_reconnect_runner", reconnect)
        patch.setattr(runner, "close", close)
        patch.setattr(docker_coding, "_run_toolchain_admission_probes", admitted)

    tool = RunCheckTool(
        checks=(check,),
        toolchain_profile=profile,
        command_policy=ProcessCommandPolicy(
            allowed_executables=(sys.executable,), allowed_cwds=(str(target),)
        ),
    )
    if (root / "postgres-dsn").exists():
        from cayu import PostgresSessionStore, PostgresTaskStore
        from cayu.storage.migrations import SchemaMode

        dsn = (root / "postgres-dsn").read_text().strip()
        store = PostgresSessionStore(dsn, schema_mode=SchemaMode.CREATE)
        task_store = PostgresTaskStore(dsn, schema_mode=SchemaMode.CREATE)
    else:
        store = SQLiteSessionStore(root / "sessions.sqlite")
        task_store = SQLiteTaskStore(root / "tasks.sqlite")
    app = CayuApp(session_store=store, task_store=task_store, enable_logging=False)
    provider = RecoveryProvider(
        [
            [
                *(
                    [
                        ModelStreamEvent.tool_call(
                            id="read", name="read_file", arguments={"path": "seed.txt"}
                        )
                    ]
                    if (root / "mixed-round").exists()
                    else []
                ),
                ModelStreamEvent.tool_call(
                    id="check", name="run_check", arguments={"check": "test"}
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app.register_provider(provider, default=True)
    if factory is not None:
        app.register_environment_factory(
            EnvironmentSpec(
                name="coding", execution_profile_identity=factory.execution_profile_identity
            ),
            factory,
            default=True,
        )
    else:
        app.register_environment(
            Environment(
                EnvironmentSpec(
                    name="coding",
                    execution_profile_identity=ExecutionProfileBehaviorIdentity(
                        name="tests.check-environment",
                        behavior_version="1",
                        implementation_version="1",
                    ),
                ),
                runner=runner,
                workspace=LocalWorkspace(source, workspace_id="source"),
            ),
            default=True,
        )
    app.register_agent(
        AgentSpec(name="coding", model="test-model"),
        tools=[ReadFileTool(), tool] if (root / "mixed-round").exists() else [tool],
        tool_policy=StaticToolPolicy(allow=("run_check", "read_file")),
    )
    if (root / "pause-policy").exists():
        authorize = StaticToolPolicy.authorize

        async def pause_policy(self, request):
            (root / "policy-started").touch()
            while not (root / "release-policy").exists():
                await asyncio.sleep(0.01)
            return await authorize(self, request)

        patch.setattr(StaticToolPolicy, "authorize", pause_policy)
    return app, store, provider


async def start(root: Path):
    with pytest.MonkeyPatch.context() as patch:
        app, _store, _provider = build_app(root, patch)
        await app.task_store.create_task(TaskCreate(task_id="check-task", type="check"))

        async def handle(app, task, worker_id):
            async for _event in app.run(
                RunRequest(
                    agent_name="coding",
                    session_id="recovery-check",
                    task_id=task.id,
                    task_worker_id=worker_id,
                    task_lease_expires_at=task.lease_expires_at,
                    messages=[Message.text("user", "Run test")],
                )
            ):
                pass

        await run_task_worker(
            app,
            app.task_store,
            handle,
            worker_id="original-worker",
            query=TaskQuery(type="check"),
            lease_seconds=3,
            max_tasks=1,
            recover_interrupted_handoffs=False,
        )


def wait_file(path, process):
    deadline = time.monotonic() + 20
    while not path.exists():
        if process.poll() is not None:
            pytest.fail("Worker exited before reaching the real command barrier")
        if time.monotonic() >= deadline:
            pytest.fail("Worker did not reach command barrier")
        time.sleep(0.02)


@pytest.mark.parametrize(
    "factory_mode,receipt_removed,cancel_reconnect,binding_conflict,mixed_round",
    [
        (False, False, False, False, False),
        (True, False, False, False, False),
        (False, True, False, False, False),
        (True, True, False, False, False),
        (True, False, True, False, False),
        (True, False, False, True, False),
        pytest.param(True, False, False, False, True, id="mixed-round"),
        pytest.param(True, True, False, False, True, id="mixed-round-missing-receipt"),
    ],
)
def test_public_recovery_after_mid_command_worker_death(
    tmp_path, receipt_removed, factory_mode, cancel_reconnect, binding_conflict, mixed_round
):
    if factory_mode:
        (tmp_path / "factory-mode").touch()
    if mixed_round:
        (tmp_path / "mixed-round").touch()
        (tmp_path / "source").mkdir()
        (tmp_path / "source" / "seed.txt").write_text("read-once")
    worker_pid = tmp_path / "worker.pid"
    witness = f"""
import ctypes,os,pathlib,subprocess,sys
assert ctypes.CDLL(None).prctl(36,1,0,0,0) == 0
p=subprocess.Popen([sys.executable,'-m','tests.recovery.test_named_check_durable_recovery',{str(tmp_path)!r}])
pathlib.Path({str(worker_pid)!r}).write_text(str(p.pid))
status=p.wait()
if status != -9: raise SystemExit(status or 1)
while True:
    try: os.waitpid(-1,0)
    except ChildProcessError: break
"""
    process = subprocess.Popen([sys.executable, "-c", witness])
    try:
        wait_file(tmp_path / "started", process)
        os.kill(int(worker_pid.read_text()), signal.SIGKILL)

        async def recover():
            with pytest.MonkeyPatch.context() as patch:
                app, store, provider = build_app(tmp_path, patch)
                try:
                    request = RecoveryPlanRequest(
                        selection=RecoveryPlanSelection(
                            session_ids=("recovery-check",), inactive_for_seconds=0
                        )
                    )
                    before = await store.load_checkpoint("recovery-check")
                    prior_terminals = [
                        event
                        for event in await store.load_events("recovery-check")
                        if event.type.value == "tool.call.completed"
                    ]
                    if mixed_round:
                        assert not prior_terminals
                        from cayu.runtime._tool_round_recovery import (
                            pending_tool_round_from_checkpoint,
                        )

                        retained_round = pending_tool_round_from_checkpoint(before)
                        assert retained_round is not None
                        retained_read = [
                            item
                            for item in retained_round.staged_terminals
                            if item.tool_call_id == "read"
                        ]
                        assert len(retained_read) == 1
                        assert "read-once" in retained_read[0].event.payload["result"]["content"]

                        async def prohibit_read(*args, **kwargs):
                            pytest.fail("Recovery must not redispatch the completed sibling")

                        patch.setattr(ReadFileTool, "run", prohibit_read)
                    pending = await app.plan_recovery(request)
                    assert pending.items[0].allowed_actions == (RecoveryPlanAction.LEAVE_INTACT,)
                    assert await store.load_checkpoint("recovery-check") == before
                    (tmp_path / "release").touch()
                    await asyncio.to_thread(process.wait, timeout=10)
                    from datetime import UTC, datetime

                    task = await app.task_store.load_task("check-task")
                    remaining = (task.lease_expires_at - datetime.now(UTC)).total_seconds()
                    if remaining > 0:
                        await asyncio.sleep(remaining + 0.05)
                    plan = await app.plan_recovery(request)
                    assert RecoveryPlanAction.AUTOMATIC_REPAIR in plan.items[0].allowed_actions, (
                        plan.model_dump_json()
                    )
                    assert await store.load_checkpoint("recovery-check") == before
                    receipt_path = next((tmp_path / "receipts").glob("*/receipt.json"))
                    assert '"key"' not in plan.model_dump_json()
                    if receipt_removed:
                        receipt_path.rename(receipt_path.with_suffix(".retained"))
                    if binding_conflict:
                        load = store.load_session_operation

                        async def conflicting_load(session_id, key, **kwargs):
                            value = await load(session_id, key, **kwargs)
                            if key.startswith("command-binding:") and value is not None:
                                value["allocation"] = "conflicting-allocation"
                            return value

                        patch.setattr(store, "load_session_operation", conflicting_load)
                    execution = RecoveryExecutionRequest(
                        plan=plan,
                        decisions=(
                            RecoveryDecision(
                                item_id=plan.items[0].item_id,
                                action=RecoveryPlanAction.AUTOMATIC_REPAIR,
                            ),
                        ),
                        execution_id="recover-check",
                    )
                    if cancel_reconnect:
                        (tmp_path / "cancel-reconnect").touch()
                        operation = asyncio.create_task(app.execute_recovery(execution))
                        async with asyncio.timeout(10):
                            while not (tmp_path / "reconnect-waiting").exists():
                                if operation.done():
                                    pytest.fail(
                                        f"Recovery returned before reconnect: {operation.result()}"
                                    )
                                await asyncio.sleep(0.01)
                        operation.cancel()
                        await asyncio.sleep(0.05)
                        (tmp_path / "release-reconnect").touch()
                        with pytest.raises(asyncio.CancelledError):
                            await operation
                        assert operation.cancelled() and operation.cancelling() == 1
                        assert not provider.requests
                        # Public recovery owns the in-flight mutation through
                        # settlement; cancelling its waiter is not an abort.
                        assert (
                            tmp_path / "source" / "checked.txt"
                        ).read_text() == "retained-command-effect"
                        terminal = [
                            event
                            for event in await store.load_events("recovery-check")
                            if event.type.value == "tool.call.completed"
                        ]
                        assert len(terminal) == 1 and terminal[0].payload["recovered"] is True
                        assert (tmp_path / "disposed").exists()
                        assert (
                            tmp_path / "guest" / "checked.txt"
                        ).read_text() == "retained-command-effect"
                        return
                    result = await app.execute_recovery(execution)
                    assert not provider.requests
                    if binding_conflict:
                        assert result.items[0].error_code is not None
                        assert not (tmp_path / "reconnected").exists()
                        assert not (tmp_path / "disposed").exists()
                        assert (
                            tmp_path / "guest" / "checked.txt"
                        ).read_text() == "retained-command-effect"
                        return
                    if receipt_removed:
                        if mixed_round:
                            # The staged sibling permits repairing its workspace
                            # observation, not settling the unknown command.
                            assert "pending_tool_effect" in result.items[0].recovery_actions
                            assert result.items[0].final_session_status.value == "interrupted"
                            retained = pending_tool_round_from_checkpoint(
                                await store.load_checkpoint("recovery-check")
                            )
                            assert retained is not None
                            assert retained.tool_round_id == retained_round.tool_round_id
                            assert not (tmp_path / "reconnected").exists()
                            assert not (tmp_path / "disposed").exists()
                            assert not (tmp_path / "source" / "checked.txt").exists()
                            assert not any(
                                event.type.value in {"tool.call.completed", "tool.call.failed"}
                                for event in await store.load_events("recovery-check")
                            )
                            return
                        assert result.items[0].status.value != "executed", result.model_dump_json()
                        assert await store.load_checkpoint("recovery-check") == before
                        assert not any(
                            event.payload.get("recovered")
                            for event in await store.load_events("recovery-check")
                        )
                        return
                    assert result.items[0].error_code is None, result
                    task = await app.task_store.load_task("check-task")
                    assert task.worker_id is None
                    assert task.lease_expires_at is None
                    assert task.interrupted_handoff_id is not None
                    if factory_mode:
                        assert (tmp_path / "reconnected").exists()
                        assert (tmp_path / "disposed").exists()
                    events = await store.load_events("recovery-check")
                    terminal = [
                        event for event in events if event.type.value == "tool.call.completed"
                    ]
                    if mixed_round:
                        reads = [
                            event for event in terminal if event.payload["tool_call_id"] == "read"
                        ]
                        assert len(reads) == 1
                        assert "read-once" in reads[0].payload["result"]["content"]
                    terminal = [
                        event for event in terminal if event.payload["tool_call_id"] == "check"
                    ]
                    assert len(terminal) == 1
                    assert terminal[0].payload["recovered"] is True
                    assert terminal[0].payload["result"]["structured"]["status"] == "passed"
                    assert terminal[0].payload["result"]["structured"]["stdout"] == "checked-once\n"
                    assert (
                        tmp_path / "source" / "checked.txt"
                    ).read_text() == "retained-command-effect"
                    exported = await store.load_session_export_snapshot("recovery-check")
                    assert "runner_receipt" not in str(exported.document())
                    replay = await app.execute_recovery(execution)
                    assert replay.items[0].replayed is True
                    assert await store.load_events("recovery-check") == events
                    assert not provider.requests
                finally:
                    assert await app.drain_recovery_cleanups()
                    assert await app.drain_environment_cleanups()
                    await store.close()
                    await app.task_store.close()

        asyncio.run(recover())
    finally:
        (tmp_path / "release").touch()
        if process.poll() is None and worker_pid.exists():
            with suppress(ProcessLookupError):
                os.kill(int(worker_pid.read_text()), signal.SIGKILL)
        process.wait(timeout=35)


if __name__ == "__main__":
    asyncio.run(start(Path(sys.argv[1])))
