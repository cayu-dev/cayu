"""Free native workers drain queued work while another trial remains active."""

from __future__ import annotations

import asyncio
import json
import os
from itertools import pairwise

import pytest
from test_process_execution import _finished, _project, _start

pytestmark = [
    pytest.mark.process,
    pytest.mark.skipif(os.name != "posix", reason="POSIX process backend"),
]


@pytest.mark.parametrize("concurrency", [2, 3])
def test_free_worker_drains_queue_before_long_case_finishes(tmp_path, concurrency):
    _project(
        tmp_path,
        """
async def uneven_stream(self, request):
    case_id = request.messages[-1].content[0].text
    started=time.monotonic()
    Path(f"entered-{case_id}").write_text(str(os.getpid()))
    if case_id == "case-0":
        while len(list(Path('.').glob('dispatch-case-*.json'))) < 3:
            await asyncio.sleep(.01)
    else:
        await asyncio.sleep(.05)
    with Path(f"dispatch-{case_id}.json").open("x") as output:
        json.dump({"pid":os.getpid(),"started":started,"ended":time.monotonic()},output)
    yield ModelStreamEvent.text_delta("done")
    yield ModelStreamEvent.completed({"finish_reason":"stop"})
Provider.stream=uneven_stream
""",
    )
    process = _start(
        tmp_path,
        "eval",
        "run",
        "--processes",
        "2",
        "--max-concurrency",
        str(concurrency),
        "--stagger-seconds",
        ".02",
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
    records = [
        json.loads((tmp_path / f"dispatch-case-{index}.json").read_text()) for index in range(4)
    ]
    assert all(item["ended"] < records[0]["ended"] for item in records[1:])
    if concurrency == 2:
        assert all(item["pid"] != records[0]["pid"] for item in records[1:])
    points = sorted(
        [(item["started"], 1) for item in records] + [(item["ended"], -1) for item in records]
    )
    active = 0
    for _, delta in points:
        active += delta
        assert active <= concurrency
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "passed"
    assert [case["case_id"] for case in result["cases"]] == [f"case-{i}" for i in range(4)]
    stamps = [
        item["monotonic_seconds"]
        for item in result["metadata"]["cayu_launch_scheduling"]["admissions"]
    ]
    assert all(b - a >= 0.02 for a, b in pairwise(stamps))
    from cayu.evals.process_inspection import inspect_process_eval_run

    inspection = asyncio.run(inspect_process_eval_run(tmp_path / "workers"))
    assert inspection.phase == "completed"
    assert inspection.counts["passed"] == 4
    assert all(case.claim_state == "completed" for case in inspection.cases)
    aggregate = next((tmp_path / "workers").glob("result-*.json"))
    original = aggregate.read_text()
    changed = json.loads(original)
    changed["metadata"]["unadmitted"] = "changed aggregate metadata"
    aggregate.write_text(json.dumps(changed))
    try:
        with pytest.raises(ValueError, match="aggregate differs from committed case evidence"):
            asyncio.run(inspect_process_eval_run(tmp_path / "workers"))
    finally:
        aggregate.write_text(original)


@pytest.mark.parametrize("boundary", ["claimed", "dispatching", "completed"])
def test_worker_death_retains_claim_without_automatic_replay(tmp_path, boundary):
    _project(
        tmp_path,
        f"""
from cayu.evals._process_claims import ProcessClaims
_original_claim = ProcessClaims.claim
_original_dispatch = ProcessClaims.dispatch
_original_complete = ProcessClaims.complete
async def interrupted_claim(self, slot):
    result = await _original_claim(self, slot)
    if result is not None and self.worker == 0 and {boundary!r} == "claimed": os._exit(23)
    return result
async def interrupted_dispatch(self, case_id, slot):
    result = await _original_dispatch(self, case_id, slot)
    if self.worker == 0 and {boundary!r} == "dispatching": os._exit(23)
    return result
async def interrupted_complete(self, case_id, slot, result):
    value = await _original_complete(self, case_id, slot, result)
    if self.worker == 0 and {boundary!r} == "completed": os._exit(23)
    return value
ProcessClaims.claim = interrupted_claim
ProcessClaims.dispatch = interrupted_dispatch
ProcessClaims.complete = interrupted_complete
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
    snapshot = json.loads((tmp_path / "workers/claims.json").read_text())
    crashed = [row for row in snapshot["cases"] if row["worker"] == 0]
    assert len(crashed) == 1
    assert crashed[0]["state"] == boundary
    case_id = crashed[0]["case_id"]
    assert (tmp_path / f"dispatch-{case_id}.json").exists() == (boundary == "completed")
    from cayu.evals.process_inspection import export_process_eval_run, inspect_process_eval_run

    before = {path.name: path.read_bytes() for path in (tmp_path / "workers").glob("*.json")}
    inspection = asyncio.run(inspect_process_eval_run(tmp_path / "workers"))
    assert inspection.phase == "incomplete"
    assert inspection.automatic_replay is False
    case = next(item for item in inspection.cases if item.case_id == case_id)
    assert case.worker_index == 0 and case.claim_state == boundary
    assert (case.result_status is not None) == (boundary == "completed")
    worker = next(item for item in inspection.workers if item.index == 0)
    assert worker.result_run_id is None  # A per-case result does not invent a worker aggregate.
    exported = export_process_eval_run(tmp_path / "workers", tmp_path / "evidence.zip")
    assert exported.phase == "incomplete"
    assert {
        path.name: path.read_bytes() for path in (tmp_path / "workers").glob("*.json")
    } == before


@pytest.mark.parametrize("concurrency", [1, 4])
@pytest.mark.parametrize("reuse_instances", [False, True])
def test_workflow_claims_preserve_per_trial_instance_isolation(
    tmp_path, concurrency, reuse_instances
):
    _project(
        tmp_path,
        f"""
from cayu import WorkflowEvalTarget,WorkflowEvalExecution,WorkflowEvalResult,WorkflowSpec,EvaluationEvidencePolicySpec
from cayu.workflows import WorkflowBase,step
class Workflow(WorkflowBase):
    spec=WorkflowSpec(name='process-workflow')
    async def run(self,session_id):
        ctx=self.context(session_id)
        yield await ctx.start()
        await asyncio.sleep(.05)
        result=await step(ctx,agent='agent',step_id='child',prompt=session_id)
        yield await ctx.completed({{'answer':result.text}})

def build_eval():
    app=build_app()
    reused_app=build_app()
    reused_workflow=Workflow(reused_app)
    def execution(invocation):
        if {reuse_instances!r}:
            return WorkflowEvalExecution(app=reused_app,workflow=reused_workflow)
        child=build_app()
        return WorkflowEvalExecution(app=child,workflow=Workflow(child))
    revision='sha256:'+'1'*64
    target=WorkflowEvalTarget(key='process-proof',app=app,
        request_base=RunRequest(agent_name='agent',messages=[]),application_release_id='test',
        evidence_policy=EvaluationEvidencePolicySpec.standard(),workflow_spec=Workflow.spec,
        implementation_revision=revision,result_projector_revision=revision,execution_scope_revision=revision,
        workflow_factory=execution,result_projector=lambda e:WorkflowEvalResult(final_output=e.completion_event.payload['answer']))
    return EvalPlan(workflow_target=target,suite=EvalSuite(id='workflow-process-proof',cases=[
        EvalCase(id=f'case-{{i}}',request=RunRequest(agent_name='agent',messages=[Message.text('user','synthetic')]),assertions=[FinalOutputContains('done')])
        for i in range(4)]))
""",
    )
    process = _start(
        tmp_path,
        "eval",
        "run",
        "--processes",
        "2",
        "--max-concurrency",
        str(concurrency),
        "--process-directory",
        "workers",
        "--output",
        "result.json",
    )
    stdout, stderr = _finished(process, 60)
    assert process.returncode == (2 if reuse_instances else 0), (stdout, stderr)
    result = json.loads((tmp_path / "result.json").read_text())
    assert len(result["cases"]) == 4
    errors = [case for case in result["cases"] if case["status"] == "error"]
    if reuse_instances:
        assert errors
        assert all(
            "Per-trial workflow target reused application or workflow state."
            in case["trials"][0]["error"]
            for case in errors
        )
    else:
        assert result["status"] == "passed"
    from cayu.evals.process_inspection import inspect_process_eval_run

    inspection = asyncio.run(inspect_process_eval_run(tmp_path / "workers"))
    assert inspection.phase == "completed"
    assert inspection.counts["error"] == len(errors)
    assert all(case.claim_state == "completed" for case in inspection.cases)
