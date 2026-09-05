from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.process,
    pytest.mark.skipif(os.name != "posix", reason="POSIX process backend"),
]


def _project(root: Path, extra: str = "") -> None:
    (root / "pyproject.toml").write_text("""[tool.cayu]
factory = "process_project:build_app"
eval_target = "process_project:build_eval"
[tool.cayu.workers]
once = "process_project:run_once"
wait = "process_project:run_wait"
fail = "process_project:run_fail"
stubborn = "process_project:run_stubborn"
""")
    (root / "process_project.py").write_text(
        """import asyncio,json,os,time
from pathlib import Path
from cayu import AgentSpec,CayuApp,EvalCase,EvalPlan,EvalSuite,FinalOutputContains,Message,RunRequest
from cayu import ModelProvider,ModelStreamEvent,ExecutionProfileBehaviorIdentity

class Provider(ModelProvider):
    name = "process-proof"
    @property
    def execution_profile_identity(self):
        return ExecutionProfileBehaviorIdentity(name="process-proof", behavior_version="1", implementation_version="1")
    async def stream(self, request):
        case_id = request.messages[-1].content[0].text
        started=time.monotonic(); cpu=time.process_time()
        while time.process_time()-cpu < .1:
            sum(i*i for i in range(200))
        with Path(f"dispatch-{case_id}.json").open("x") as f:
            json.dump({"pid":os.getpid(),"started":started,"ended":time.monotonic()},f)
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason":"stop"})

def build_app():
    Path(f"factory-{os.getpid()}.json").write_text(json.dumps({"pid":os.getpid()}))
    app=CayuApp(enable_logging=False)
    app.register_provider(Provider(),default=True)
    app.register_agent(AgentSpec(name="agent",model="test"),tools=[])
    return app

def build_eval():
    return EvalPlan(app=build_app(),suite=EvalSuite(id="process-proof",cases=[
        EvalCase(id=f"case-{i}",request=RunRequest(agent_name="agent",messages=[Message.text("user",f"case-{i}")]),assertions=[FinalOutputContains("done")])
        for i in range(4)]))

async def run_once(app,stop):
    Path(f"worker-{os.getpid()}.json").write_text(json.dumps({"index":os.environ["CAYU_WORKER_INDEX"]}))
    while len(list(Path('.').glob('worker-*.json'))) < int(os.environ['CAYU_WORKER_COUNT']):
        await asyncio.sleep(.01)

async def run_wait(app,stop):
    Path(f"worker-{os.getpid()}.json").write_text('{}')
    await stop.wait()
    Path(f"stopped-{os.getpid()}.json").write_text('{}')

async def run_fail(app,stop):
    if os.environ['CAYU_WORKER_INDEX']=='0':
        await asyncio.sleep(.5)
        raise RuntimeError('injected failure')
    await run_wait(app,stop)

async def run_stubborn(app,stop):
    Path(f"worker-{os.getpid()}.json").write_text('{}')
    while True:
        try: await asyncio.sleep(.1)
        except asyncio.CancelledError: pass
"""
        + extra
    )


def _start(root: Path, *args: str) -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    return subprocess.Popen(
        [sys.executable, "-m", "cayu", *args],
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _finished(process: subprocess.Popen, timeout: float = 30):
    try:
        return process.communicate(timeout=timeout)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)


def _wait_markers(root: Path, count: int) -> list[int]:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        markers = list(root.glob("worker-*.json"))
        if len(markers) == count:
            return [int(p.stem.split("-")[1]) for p in markers]
        time.sleep(0.02)
    pytest.fail("workers did not start")


def _assert_dead(pids):
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_named_workers_construct_apps_in_distinct_child_processes(tmp_path):
    _project(tmp_path)
    process = _start(tmp_path, "worker", "once", "--processes", "2")
    stdout, stderr = _finished(process)
    assert process.returncode == 0, (stdout, stderr)
    factories = list(tmp_path.glob("factory-*.json"))
    assert len(factories) == 2
    assert not (tmp_path / f"factory-{process.pid}.json").exists()
    assert {json.loads(p.read_text())["index"] for p in tmp_path.glob("worker-*.json")} == {
        "0",
        "1",
    }
    _assert_dead([int(p.stem.split("-")[1]) for p in factories])


@pytest.mark.parametrize(
    "name,signum,expected",
    [
        ("wait", signal.SIGTERM, 143),
        ("wait", signal.SIGINT, 130),
        ("stubborn", signal.SIGTERM, 124),
    ],
)
def test_process_group_signal_stops_and_reaps_children(tmp_path, name, signum, expected):
    _project(tmp_path)
    process = _start(tmp_path, "worker", name, "--processes", "2", "--shutdown-grace-seconds", ".3")
    try:
        pids = _wait_markers(tmp_path, 2)
        process.send_signal(signum)
        stdout, stderr = _finished(process)
        assert process.returncode == expected, (stdout, stderr)
        _assert_dead(pids)
        if name == "wait":
            assert len(list(tmp_path.glob("stopped-*.json"))) == 2
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)


def test_worker_failure_stops_siblings(tmp_path):
    _project(tmp_path)
    process = _start(
        tmp_path, "worker", "fail", "--processes", "2", "--shutdown-grace-seconds", ".5"
    )
    stdout, stderr = _finished(process)
    assert process.returncode == 1, (stdout, stderr)
    assert len(list(tmp_path.glob("stopped-*.json"))) == 1
    _assert_dead([int(p.stem.split("-")[1]) for p in tmp_path.glob("factory-*.json")])


def test_parent_loss_stops_workers(tmp_path):
    _project(tmp_path)
    process = _start(
        tmp_path, "worker", "wait", "--processes", "2", "--shutdown-grace-seconds", ".5"
    )
    try:
        pids = _wait_markers(tmp_path, 2)
        process.kill()
        _finished(process)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if len(list(tmp_path.glob("stopped-*.json"))) == 2:
                break
            time.sleep(0.02)
        assert len(list(tmp_path.glob("stopped-*.json"))) == 2
        # Orphans are reaped by the host, independently of the killed supervisor.
        while time.monotonic() < deadline:
            alive = []
            for pid in pids:
                try:
                    os.kill(pid, 0)
                    alive.append(pid)
                except ProcessLookupError:
                    pass
            if not alive:
                return
            time.sleep(0.02)
        _assert_dead(pids)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)


def test_cancellation_during_spawn_reaps_registered_child(tmp_path, monkeypatch):
    import asyncio

    from cayu.runtime._process_workers import ProcessWorkerCommand, supervise_process_workers

    async def scenario():
        spawned = asyncio.Event()
        release = asyncio.Event()
        children = []
        original = asyncio.create_subprocess_exec

        async def delayed_spawn(*args, **kwargs):
            child = await original(*args, **kwargs)
            children.append(child)
            spawned.set()
            await release.wait()
            return child

        monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_spawn)
        task = asyncio.create_task(
            supervise_process_workers(
                [
                    ProcessWorkerCommand(
                        (sys.executable, "-c", "import time; time.sleep(60)"),
                        tmp_path,
                        os.environ.copy(),
                    )
                ],
                shutdown_grace_seconds=0.2,
            )
        )
        await asyncio.wait_for(spawned.wait(), 5)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        assert children[0].returncode is not None
        _assert_dead([children[0].pid])

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["spawn", "admission"])
def test_startup_failure_reaps_already_started_children(tmp_path, failure):
    import asyncio

    from cayu.runtime._process_workers import ProcessWorkerCommand, supervise_process_workers

    async def scenario():
        script = "import os,time; from pathlib import Path; Path('child').write_text(str(os.getpid())); time.sleep(60)"
        commands = [
            ProcessWorkerCommand((sys.executable, "-c", script), tmp_path, os.environ.copy())
        ]
        if failure == "spawn":
            commands.append(
                ProcessWorkerCommand(
                    (str(tmp_path / "missing-executable"),), tmp_path, os.environ.copy()
                )
            )
        children = []

        async def reject(processes):
            children.extend(processes)
            raise ValueError("injected admission failure")

        with pytest.raises(FileNotFoundError if failure == "spawn" else ValueError):
            await supervise_process_workers(commands, shutdown_grace_seconds=0.2, ready=reject)
        for child in children:
            assert child.returncode is not None
            _assert_dead([child.pid])
        if (tmp_path / "child").exists():
            _assert_dead([int((tmp_path / "child").read_text())])

    asyncio.run(scenario())


def test_native_eval_processes_preserve_complete_ordered_results(tmp_path):
    _project(tmp_path)
    process = _start(
        tmp_path,
        "eval",
        "run",
        "--processes",
        "2",
        "--max-concurrency",
        "2",
        "--process-directory",
        "workers",
        "--output",
        "result.json",
    )
    stdout, stderr = _finished(process, 60)
    assert process.returncode == 0, (
        stdout,
        stderr,
        [(p.name, p.read_text()) for p in (tmp_path / "workers").glob("*.log")],
    )
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "passed"
    assert [c["case_id"] for c in result["cases"]] == [f"case-{i}" for i in range(4)]
    provenance = result["metadata"]["cayu_process_execution"]["workers"]
    assert len({w["pid"] for w in provenance}) == 2
    assert len(list(tmp_path.glob("dispatch-*.json"))) == 4
    assert len(list(tmp_path.glob("factory-*.json"))) == 2
    assert (tmp_path / "workers/completed.json").exists()
    _assert_dead([w["pid"] for w in provenance])


def test_eval_plan_drift_fails_before_dispatch(tmp_path):
    _project(
        tmp_path,
        """
_original=build_eval
def build_eval():
    plan=_original()
    plan.suite.metadata['drift']=os.environ['CAYU_WORKER_INDEX']
    return plan
""",
    )
    process = _start(
        tmp_path,
        "eval",
        "run",
        "--processes",
        "2",
        "--max-concurrency",
        "2",
        "--process-directory",
        "workers",
        "--output",
        "result.json",
    )
    stdout, stderr = _finished(process, 60)
    assert process.returncode == 2, (stdout, stderr)
    assert not list(tmp_path.glob("dispatch-*.json"))
    assert not (tmp_path / "result.json").exists()
    assert (tmp_path / "workers/incomplete.json").exists()


def test_missing_case_result_rejects_aggregate(tmp_path, monkeypatch):
    import asyncio

    from cayu.cli import _eval_processes
    from cayu.cli.project import project_context

    _project(tmp_path)
    original = _eval_processes._read_json

    def omit_case(path):
        document = original(path)
        if path.name == "result-0.json":
            document["cases"] = document["cases"][1:]
        return document

    monkeypatch.setattr(_eval_processes, "_read_json", omit_case)
    with project_context(tmp_path), pytest.raises(RuntimeError, match="admitted cases"):
        asyncio.run(
            _eval_processes.run_process_eval(
                target="process_project:build_eval",
                project_root=tmp_path,
                directory=tmp_path / "workers",
                processes=2,
                max_concurrency=2,
                case_timeout_seconds=None,
            )
        )
    assert not (tmp_path / "workers/completed.json").exists()
    assert (tmp_path / "workers/incomplete.json").exists()


def test_named_workers_share_durable_task_claims_without_duplicates(tmp_path):
    import asyncio

    from cayu import SQLiteTaskStore, TaskCreate

    _project(
        tmp_path,
        """
from cayu import SQLiteTaskStore,TaskQuery
async def run_claims(app,stop):
    store=SQLiteTaskStore(Path('tasks.sqlite'))
    worker=f'worker-{os.getpid()}'
    try:
        while not stop.is_set():
            task=await store.claim_task(worker,TaskQuery(type='cpu-work'),lease_seconds=30)
            if task is None:break
            with Path(f'claim-{task.id}.json').open('x') as f:json.dump({'pid':os.getpid()},f)
            await asyncio.sleep(.03)
            await store.complete_task(task.id,{'done':True},worker_id=worker,
                lease_expires_at=task.lease_expires_at)
    finally:await store.close()
""",
    )
    with (tmp_path / "pyproject.toml").open("a") as f:
        f.write('claims = "process_project:run_claims"\n')

    async def seed():
        store = SQLiteTaskStore(tmp_path / "tasks.sqlite")
        try:
            for i in range(12):
                await store.create_task(
                    TaskCreate(task_id=f"task-{i}", type="cpu-work", title="synthetic work")
                )
        finally:
            await store.close()

    asyncio.run(seed())
    process = _start(tmp_path, "worker", "claims", "--processes", "2")
    stdout, stderr = _finished(process)
    assert process.returncode == 0, (stdout, stderr)
    records = [json.loads(p.read_text()) for p in tmp_path.glob("claim-*.json")]
    assert len(records) == 12
    assert len({r["pid"] for r in records}) == 2


def test_multiple_workers_execute_cpu_work_in_parallel(tmp_path):
    _project(
        tmp_path,
        """
async def run_cpu(app,stop):
    Path(f'worker-{os.getpid()}.json').write_text('{}')
    while len(list(Path('.').glob('worker-*.json'))) < int(os.environ['CAYU_WORKER_COUNT']):
        await asyncio.sleep(.01)
    start=time.monotonic();cpu=time.process_time()
    while time.process_time()-cpu < 1.5:sum(i*i for i in range(1000))
    Path(f'cpu-{os.getpid()}.json').write_text(json.dumps({'start':start,'end':time.monotonic(),'cpu':time.process_time()-cpu}))
""",
    )
    with (tmp_path / "pyproject.toml").open("a") as f:
        f.write('cpu = "process_project:run_cpu"\n')
    process = _start(tmp_path, "worker", "cpu", "--processes", "2")
    stdout, stderr = _finished(process)
    assert process.returncode == 0, (stdout, stderr)
    records = [json.loads(p.read_text()) for p in tmp_path.glob("cpu-*.json")]
    assert len(records) == 2
    wall = max(r["end"] for r in records) - min(r["start"] for r in records)
    cpu = sum(r["cpu"] for r in records)
    assert max(r["start"] for r in records) < min(r["end"] for r in records)
    print(
        json.dumps(
            {
                "cpu_proof_processes": 2,
                "wall_seconds": wall,
                "aggregate_cpu_seconds": cpu,
                "effective_cpu_cores": cpu / wall,
            }
        )
    )
    if os.environ.get("CAYU_REQUIRE_MULTICORE_PROOF") == "1":
        assert cpu / wall > 1.2


def test_native_eval_worker_crash_never_publishes_partial_success(tmp_path):
    _project(
        tmp_path,
        """
_original_stream=Provider.stream
async def crash_stream(self,request):
    if os.environ['CAYU_WORKER_INDEX']=='1':os._exit(17)
    async for event in _original_stream(self,request):yield event
Provider.stream=crash_stream
""",
    )
    process = _start(
        tmp_path,
        "eval",
        "run",
        "--processes",
        "2",
        "--max-concurrency",
        "2",
        "--process-directory",
        "workers",
        "--output",
        "result.json",
    )
    stdout, stderr = _finished(process, 60)
    assert process.returncode == 2, (stdout, stderr)
    assert not (tmp_path / "result.json").exists()
    assert (tmp_path / "workers/incomplete.json").exists()
    assert len(list(tmp_path.glob("factory-*.json"))) == 2
    _assert_dead([int(p.stem.split("-")[1]) for p in tmp_path.glob("factory-*.json")])


def test_process_workflow_eval_keeps_native_child_evidence(tmp_path):
    _project(
        tmp_path,
        """
from cayu import WorkflowEvalTarget,WorkflowEvalExecution,WorkflowEvalResult,WorkflowSpec,EvaluationEvidencePolicySpec
from cayu.workflows import WorkflowBase,step
class Workflow(WorkflowBase):
    spec=WorkflowSpec(name='process-workflow')
    async def run(self,session_id):
        ctx=self.context(session_id)
        yield await ctx.start()
        result=await step(ctx,agent='agent',step_id='child',prompt=session_id)
        yield await ctx.completed({'answer':result.text})

def build_eval():
    app=build_app()
    def execution(invocation):
        child=build_app()
        return WorkflowEvalExecution(app=child,workflow=Workflow(child))
    revision='sha256:'+'1'*64
    target=WorkflowEvalTarget(key='process-proof',app=app,
        request_base=RunRequest(agent_name='agent',messages=[]),application_release_id='test',
        evidence_policy=EvaluationEvidencePolicySpec.standard(),workflow_spec=Workflow.spec,
        implementation_revision=revision,result_projector_revision=revision,execution_scope_revision=revision,
        workflow_factory=execution,result_projector=lambda e:WorkflowEvalResult(final_output=e.completion_event.payload['answer']))
    return EvalPlan(workflow_target=target,suite=EvalSuite(id='workflow-process-proof',cases=[
        EvalCase(id=f'case-{i}',request=RunRequest(agent_name='agent',messages=[Message.text('user','synthetic')]),assertions=[FinalOutputContains('done')])
        for i in range(2)]))
""",
    )
    process = _start(
        tmp_path,
        "eval",
        "run",
        "--processes",
        "2",
        "--max-concurrency",
        "2",
        "--process-directory",
        "workers",
        "--output",
        "result.json",
    )
    stdout, stderr = _finished(process, 60)
    assert process.returncode == 0, (
        stdout,
        stderr,
        [(p.name, p.read_text()) for p in (tmp_path / "workers").glob("*.log")],
    )
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "passed"
    for case in result["cases"]:
        trial = case["trials"][0]
        assert trial["evidence_complete"] is True
        assert trial["execution_status"] == "completed"


@pytest.mark.parametrize("leader_crashes", [True, False])
def test_shutdown_stops_descendants_after_worker_leader_exits(tmp_path, leader_crashes):
    import asyncio

    from cayu.runtime._process_workers import ProcessWorkerCommand, supervise_process_workers

    descendant = """import os, signal, time
from pathlib import Path
IGNORE_TERM
Path('descendant').write_text(str(os.getpid()))
while True:
    time.sleep(.01)
""".replace(
        "IGNORE_TERM",
        "" if leader_crashes else "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
    )
    wait_ready = "\nwhile not Path('descendant').exists(): time.sleep(.01)\n"
    leader = (
        "import subprocess, sys, time\nfrom pathlib import Path\n"
        f"subprocess.Popen([sys.executable, '-c', {descendant!r}])\n"
        + wait_ready
        + ("sys.exit(17)" if leader_crashes else "time.sleep(60)")
    )
    sibling = (
        "import sys, time\nfrom pathlib import Path\n"
        + wait_ready
        + ("time.sleep(60)" if leader_crashes else "sys.exit(17)")
    )

    def descendant_running(pid):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        # An orphan's zombie can await host PID 1 reaping in Linux containers.
        # It cannot execute work; reaping our direct children is checked below.
        stat = Path(f"/proc/{pid}/stat")
        try:
            return stat.read_text().rsplit(")", 1)[1].split()[0] != "Z"
        except FileNotFoundError:
            return sys.platform != "linux"

    async def scenario():
        children = []

        async def remember(processes):
            children.extend(processes)

        try:
            outcome = await asyncio.wait_for(
                supervise_process_workers(
                    [
                        ProcessWorkerCommand(
                            (sys.executable, "-c", script),
                            tmp_path,
                            os.environ.copy(),
                            tmp_path / f"worker-{index}.log",
                        )
                        for index, script in enumerate((leader, sibling))
                    ],
                    shutdown_grace_seconds=0.2,
                    ready=remember,
                ),
                timeout=10,
            )
            assert outcome.exit_code != 0
            assert all(child.returncode is not None for child in children)
            _assert_dead(outcome.pids)
            if not leader_crashes:
                assert outcome.forced_shutdown
            pid = int((tmp_path / "descendant").read_text())
            deadline = time.monotonic() + 5
            while descendant_running(pid) and time.monotonic() < deadline:
                await asyncio.sleep(0.02)
            assert not descendant_running(pid), "descendant survived supervisor cleanup"
        finally:
            for child in children:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGKILL)
            marker = tmp_path / "descendant"
            if marker.exists():
                pid = int(marker.read_text())
                if descendant_running(pid):
                    os.kill(pid, signal.SIGKILL)

    asyncio.run(scenario())
