"""Real local check, SIGKILL and fresh-process application settlement.

The stopped-owner observer is a controlled host-process observer, not Docker
qualification. No network, provider service, socket or container is used.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

_WORKER = r"""
import asyncio, json, os, sys
from pathlib import Path
from hashlib import sha256
from cayu import (
    AgentSpec, CayuApp, Environment, EnvironmentSpec, ExecCommand,
    ExecutionProfileBehaviorIdentity, LocalRunner, LocalWorkspace,
    ModelStreamEvent, NamedCheck, ProcessCommandPolicy, RunCheckTool,
    RunRequest, Message, ScriptedModelProvider, SQLiteSessionStore, SQLiteTaskStore,
    TaskCreate, TaskQuery, run_task_worker, ResolutionActor, ResolutionActorSource,
    PriceBook, ModelPrice,
)
from cayu.artifacts import LocalArtifactStore
from cayu.coding_products import CodingProductRunner, CodingProductArtifactRepository
from cayu.guides.coding_host import BusinessStore, Reservation, SettlementConflict, settle_cancelled
from cayu.workspaces.revisions import observe_deterministic_workspace, WorkspaceRevisionObservationLimits
from decimal import Decimal

root, action, old_pid = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
generated = len(sys.argv) > 4 and sys.argv[4] == "generated"
listing = len(sys.argv) > 5 and sys.argv[5] == "listing"

async def main():
    source = root / "source"
    source.mkdir(exist_ok=True)
    workspace_options = {}
    if generated:
        sys.path.insert(0, str(root / "generated-coder"))
        from workflows.coding_product import CodingProductApplication
        from domain.coding_product import CodingProductTask
        from operations import coding as composition
        from cayu.guides.coding_host import extend_generated_application
        from cayu import CodingSettlementPolicy
        from cayu.runners.docker_workload import DockerImageIdentity
        workspace_options = dict(
            excluded_directory_names=composition._SOURCE_EXCLUDED_DIRECTORY_NAMES,
            excluded_path_patterns=composition._SOURCE_EXCLUDED_PATH_PATTERNS,
        )
    workspace = LocalWorkspace(source, workspace_id="source-workspace", **workspace_options)
    sessions = SQLiteSessionStore(root / "sessions.sqlite")
    tasks = SQLiteTaskStore(root / "tasks.sqlite")
    app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
    usage = {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}
    provider = ScriptedModelProvider([
        [*([ModelStreamEvent.tool_call(id="listing", name="list_files",
             arguments={"pattern":"**/*.py", "limit":1})] if listing else []),
         ModelStreamEvent.tool_call(id="check", name="run_check", arguments={"check":"test"}),
         ModelStreamEvent.completed({"finish_reason":"tool_calls", "usage":usage})],
        [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({"finish_reason":"stop", "usage":usage})],
    ], name="fixture")
    if action == "start-unknown":
        async def unknown_response(request):
            (root / "ready").write_text(str(os.getpid()))
            await asyncio.Event().wait()
            yield ModelStreamEvent.completed({"finish_reason":"stop"})
        provider.stream = unknown_response
    app.register_provider(provider, default=True)
    command_text = "print('checked once')"
    if generated:
        command_text += "; from pathlib import Path; p=Path(" + repr(str(root / "check-dispatches")) + "); p.open('a').write('check\\n')"
    command = ExecCommand.process(sys.executable, "-c", command_text)
    check = RunCheckTool(
        checks=(NamedCheck(name="test", description="Local controlled check", command=command,
            execution_profile_identity=ExecutionProfileBehaviorIdentity(name="local-test", behavior_version="1", implementation_version="1")),),
        command_policy=ProcessCommandPolicy(allowed_executables=(sys.executable,), allowed_cwds=(str(source),)),
    )
    app.register_environment(Environment(
        EnvironmentSpec(name="coding", execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="example-local-environment", behavior_version="1", implementation_version="1")),
        workspace=workspace, runner=LocalRunner(source),
    ))
    from cayu import ListFilesTool
    app.register_agent(AgentSpec(name="coder", model="fixture"),
        tools=[*([ListFilesTool()] if listing else []), check])
    repository = CodingProductArtifactRepository(LocalArtifactStore(root / "artifacts", store_id="example"))
    business = BusinessStore(root / "business.sqlite")
    pricing = PriceBook(prices=(ModelPrice.fixed(provider_name="fixture", model="fixture", input_per_million=Decimal("1"), output_per_million=Decimal("1")),))
    if not generated:
        from tests.core.test_coding_products import _request
        from cayu.sessions.base import session_input_messages_sha256
        run = RunRequest(agent_name="coder", session_id="session-1", environment_name="coding",
                         messages=[Message.text("user", "repair the project")])
        observation = await observe_deterministic_workspace(workspace, observer="cayu-coding-product-source", limits=WorkspaceRevisionObservationLimits())
        request = _request(baseline=observation.revision)
        request = request.model_copy(update={
            "runtime": request.runtime.model_copy(update={"execution_profile_fingerprint": await app.inspect_run_execution_profile(run)}),
            "task": request.task.model_copy(update={"instruction_sha256": "sha256:" + session_input_messages_sha256(run.messages)}),
        })
        async def git_guard(expected):
            assert expected == request.source.git_baseline
        runner = CodingProductRunner(app, source_workspace=workspace, repository=repository, source_git_authority_validator=git_guard)
        expected = Reservation(tenant="example", public_id="example", request=request, pricing=pricing, cost_basis="synthetic")
    if generated:
        application = CodingProductApplication(
            app, source_workspace=workspace, artifact_store=repository.store,
            toolchain_profile=composition._python_toolchain_profile(
                DockerImageIdentity(reference="fixture@sha256:" + "a" * 64)),
            agent_name="coder", project_root=source,
        )
        wrapped = extend_generated_application(application, business=business,
            tenant="example", pricing=pricing, cost_basis="synthetic")
        original_task = CodingProductTask(product_run_id="generated-product",
            session_id="session-1", task_id="task-1", instruction="repair the project",
            settlement=CodingSettlementPolicy(required_checks=("test",),
                reviewer_required=False, human_approval_required=False))
        async def retained_reservation():
            saved = await repository.load_request(original_task.product_run_id,
                session_id=original_task.session_id)
            assert saved.task.task_id == original_task.task_id
            assert saved.session_id == original_task.session_id
            assert saved.product_run_id == original_task.product_run_id
            return Reservation(tenant="example", public_id=original_task.product_run_id,
                request=saved, pricing=pricing, cost_basis="synthetic")
        async def session_events():
            return [event.model_dump(mode="json") for event in
                await sessions.load_events(original_task.session_id)]
        if action != "start":
            expected = await retained_reservation()
            request = expected.request
            assert request.fingerprint == (root / "original-fingerprint").read_text()
            assert (root / "check-dispatches").read_text() == "check\n"
        def competitor(reservation):
            return reservation.model_copy(update={"public_id":"next-source-owner",
                "request":reservation.request.model_copy(update={"product_run_id":"next-product"})})
        def require_source_fenced(reservation):
            try:
                business.reserve(competitor(reservation))
            except SettlementConflict:
                return
            raise AssertionError("source released before positive application settlement")
    try:
        if action in {"start", "start-unknown"}:
            if not generated:
                business.reserve(expected)
            task_id = original_task.task_id if generated else request.task.task_id
            await tasks.create_task(TaskCreate(task_id=task_id, type="example"))
            async def handler(_app, task, worker):
                if generated:
                    await wrapped.run(original_task)
                    retained = await retained_reservation()
                    assert business.read(retained) is None
                    require_source_fenced(retained)
                    (root / "original-fingerprint").write_text(retained.request.fingerprint)
                    (root / "original-events.json").write_text(json.dumps(await session_events()))
                    if listing:
                        results = [event.payload["result"]["structured"] for event in
                            await sessions.load_events(original_task.session_id)
                            if event.type.value == "tool.call.completed" and event.tool_name == "list_files"]
                        assert len(results) == 1
                        assert results[0]["total_files"] is None and results[0]["truncated"] is True
                    assert len(provider.requests) == 2
                else:
                    await runner.run(request, run)
                (root / "ready").write_text(str(os.getpid()))
                await asyncio.Event().wait()
            await run_task_worker(app, tasks, handler, worker_id="original-process", query=TaskQuery(type="example"), lease_seconds=1, max_tasks=1)
        else:
            for _ in range(100):
                task = await tasks.load_task(request.task.task_id)
                from datetime import datetime, UTC
                if task.lease_expires_at is None or task.lease_expires_at <= datetime.now(UTC):
                    break
                await asyncio.sleep(.05)
            await tasks.reclaim_expired(query=TaskQuery(type="example"))
            if generated and business.read(expected) is None:
                require_source_fenced(expected)
            async def stopped(worker):
                # Parent killed and reaped the exact process before launching us.
                # Only host observation is substituted; this proves no Docker claim.
                assert worker == "original-process"
                try:
                    os.kill(old_pid, 0)
                except ProcessLookupError:
                    return {"worker_id":worker, "container_id":"a"*64, "generation":"b"*64}
                raise AssertionError("original worker is still live")
            actor = ResolutionActor(subject="test-operator", tenant="example", source=ResolutionActorSource.HTTP_AUTH)
            if action == "native-ack-loss":
                original = tasks.reconcile_task_cancellation
                async def lost(request):
                    await original(request)
                    raise ConnectionError("native acknowledgement lost")
                tasks.reconcile_task_cancellation = lost
            if action == "application-ack-loss":
                original = business.settle
                def lost(expected, result):
                    original(expected, result)
                    raise ConnectionError("application acknowledgement lost")
                business.settle = lost
            try:
                kwargs = dict(actor=actor,
                    reconciliation_id="changed" if action == "identity-conflict" else "one-operation", inspect_stopped_worker=stopped,
                    pricing=pricing, cost_basis="observed" if action == "basis-conflict" else "synthetic")
                if generated:
                    from dataclasses import replace
                    recovery_task = (replace(original_task, instruction="conflicting instruction")
                        if action == "request-conflict" else original_task)
                    result = await wrapped.settle_cancelled(recovery_task, **kwargs)
                else:
                    result = await settle_cancelled(runner, business, expected, **kwargs)
            except __import__("cayu").CodingProductAdmissionError:
                assert generated and action == "request-conflict"
                result = {"classification":"conflict_refused"}
            except SettlementConflict:
                assert action in {"identity-conflict", "basis-conflict"}
                result = {"classification":"conflict_refused"}
            except __import__("cayu").CodingProductReconstructionRequiredError:
                assert action == "unknown"
                assert business.read(expected) is None and business.pending(expected) is None
                task = await tasks.load_task(request.task.task_id)
                assert task.status_reason == "cancellation_requested"
                result = {"classification":"unknown_fenced", "task_status":task.status.value}
            except ConnectionError as error:
                assert action.endswith("ack-loss")
                if generated:
                    native_task = await tasks.load_task(original_task.task_id)
                    assert native_task.status.value == "cancelled"
                    assert business.pending(expected).task_id == original_task.task_id
                    if action == "native-ack-loss":
                        assert business.read(expected) is None
                        require_source_fenced(expected)
                    else:
                        assert business.read(expected)["classification"] == "cancelled"
                        business.reserve(competitor(expected))
                result = {"retained_failure":str(error)}
            if generated:
                assert (root / "check-dispatches").read_text() == "check\n"
                assert expected.request.fingerprint == (root / "original-fingerprint").read_text()
                assert await session_events() == json.loads((root / "original-events.json").read_text())
                if result.get("classification") == "cancelled":
                    assert result == business.read(expected)
                    assert result["task_id"] == original_task.task_id
                    assert result["session_id"] == original_task.session_id
                    assert result["request_fingerprint"] == expected.request.fingerprint
                    assert business.read(competitor(expected)) is None
                    # Old receipt replay must not clear the newer source owner.
                    third = competitor(expected).model_copy(update={"public_id":"third-owner"})
                    try:
                        business.reserve(third)
                    except SettlementConflict:
                        pass
                    else:
                        raise AssertionError("replay released a newer source owner")
            assert not provider.requests
            print(json.dumps(result))
    finally:
        await sessions.close()
        await tasks.close()

asyncio.run(main())
"""


@pytest.mark.skipif(os.name != "posix", reason="SIGKILL requires POSIX")
@pytest.mark.parametrize("unknown", [False, True])
def test_local_worker_loss_exact_settlement_across_three_fresh_processes(tmp_path, unknown):
    _exercise_worker_loss(tmp_path, unknown=unknown)


@pytest.mark.skipif(os.name != "posix", reason="SIGKILL requires POSIX")
@pytest.mark.parametrize("listing", [False, True])
def test_generated_wrapper_worker_loss_and_fresh_process_settlement(tmp_path, listing):
    from cayu.cli import main

    assert (
        main(
            [
                "new",
                "generated-coder",
                "--preset",
                "coding",
                "--execution",
                "docker",
                "--coding-toolchain",
                "python",
                "--dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    source = tmp_path / "source"
    source.mkdir()
    (source / "example.py").write_text("value = 1\n")
    if listing:
        (source / "second.py").write_text("value = 2\n")
    for arguments in (
        ["init"],
        ["add", "."],
        [
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-m",
            "fixture",
        ],
    ):
        subprocess.run(["git", "-C", str(source), *arguments], check=True, capture_output=True)
    _exercise_worker_loss(tmp_path, unknown=False, generated=True, listing=listing)


def _exercise_worker_loss(tmp_path, *, unknown, generated=False, listing=False):
    installed = os.environ.get("CAYU_GUIDE_INSTALLED_PYTHON")
    pythonpath = (
        str(Path.cwd()) if installed else str(Path.cwd() / "src") + os.pathsep + str(Path.cwd())
    )
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": pythonpath}
    command = [installed or sys.executable, "-c", _WORKER, str(tmp_path)]
    mode = [*(["generated"] if generated else []), *(["listing"] if listing else [])]
    log = tmp_path / "producer.log"
    with log.open("w") as stream:
        producer = subprocess.Popen(
            [
                *command,
                "start-unknown" if unknown else "start",
                "0",
                *mode,
            ],
            env=env,
            stdout=stream,
            stderr=stream,
        )
        try:
            deadline = time.monotonic() + 30
            while not (tmp_path / "ready").exists():
                if producer.poll() is not None or time.monotonic() > deadline:
                    pytest.fail(log.read_text())
                time.sleep(0.05)
            producer.kill()
            producer.wait(timeout=5)
            results = []
            for action in (
                ("unknown", "unknown")
                if unknown
                else (
                    "native-ack-loss",
                    "application-ack-loss",
                    "replay",
                    "replay",
                    "identity-conflict",
                    "basis-conflict",
                    *(["request-conflict"] if generated else []),
                )
            ):
                completed = subprocess.run(
                    [*command, action, str(producer.pid), *mode],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                assert completed.returncode == 0, completed.stdout + completed.stderr
                results.append(json.loads(completed.stdout.strip().splitlines()[-1]))
            if unknown:
                assert (
                    results == [{"classification": "unknown_fenced", "task_status": "claimed"}] * 2
                )
                return
            assert results[0] == {"retained_failure": "native acknowledgement lost"}
            assert results[1] == {"retained_failure": "application acknowledgement lost"}
            assert results[2] == results[3]
            assert results[4:] == [{"classification": "conflict_refused"}] * (3 if generated else 2)
            assert results[2]["classification"] == "cancelled"
            assert results[2]["cost"]["basis"] == "synthetic"
            assert results[2]["cost"]["availability"] == "recorded"
            from decimal import Decimal

            assert Decimal(results[2]["cost"]["estimated_total"]) == Decimal("0.000020")
            assert results[2]["cost"]["model_steps"] == 2
        finally:
            if producer.poll() is None:
                producer.kill()
                producer.wait(timeout=5)
