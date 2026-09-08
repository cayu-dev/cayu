"""Fresh-process execution for trusted native direct eval target factories."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import signal
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from cayu.build_provenance import current_runtime_build_provenance
from cayu.evals import EvalPlan, EvalRun, EvalSuite, eval_run_to_json
from cayu.evals._inspection_documents import write_process_document as _write_json
from cayu.evals._process_progress import ProcessEvalProgress
from cayu.evals.capacity import EVAL_MAX_CONCURRENCY
from cayu.evals.models import aggregate_eval_score, aggregate_eval_status
from cayu.evals.runner import run_eval_suite, run_workflow_eval_suite
from cayu.evals.trial_policy import EvalSuiteTrialPolicyV1
from cayu.runtime._process_workers import (
    ProcessWorkerCommand,
    process_worker_environment,
    supervise_process_workers,
    supervisor_watchdog,
    watch_supervisor,
)

_MAX_DOCUMENT_BYTES = 64 * 1024 * 1024
_METADATA_KEY = "cayu_process_execution"


def _read_json(path: Path):
    with path.open("rb") as stream:
        data = stream.read(_MAX_DOCUMENT_BYTES + 1)
    if len(data) > _MAX_DOCUMENT_BYTES:
        raise ValueError("Process eval document exceeds its 64 MiB bound.")
    return json.loads(data)


async def _plan_identity(plan: EvalPlan) -> dict:
    if plan.suite is None or plan.corpus_target is not None:
        raise ValueError("Process execution currently requires a native direct EvalSuite.")
    if _METADATA_KEY in plan.suite.metadata:
        raise ValueError(f"Suite metadata reserves {_METADATA_KEY!r} for process provenance.")
    target = plan.workflow_target
    app = target.app if target is not None else plan.app
    if app is None:
        raise ValueError("Process eval requires an application.")
    if target is not None and target.instance_scope.value != "per_trial":
        raise ValueError("Process workflow eval requires per_trial instance scope.")
    cases = []
    for case in plan.suite.cases:
        assertions = []
        for assertion in case.assertions:
            cls = type(assertion)
            revision = assertion.assertion_revision
            definition = None
            if revision is None:
                if cls.__module__ != "cayu.evals.assertions":
                    raise ValueError(
                        "Custom assertions in process evals require assertion_revision."
                    )
                # Built-in declarative assertions may use JSON-representable state.
                # Do not pickle or guess at arbitrary application objects.
                try:
                    definition = json.loads(json.dumps(vars(assertion), allow_nan=False))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Assertion {cls.__name__} has no process-portable definition."
                    ) from exc
            assertions.append(
                {
                    "type": f"{cls.__module__}:{cls.__qualname__}",
                    "name": assertion.name,
                    "revision": revision,
                    "definition": definition,
                }
            )
        cases.append(
            {
                "id": case.id,
                "request": case.request.model_dump(mode="json"),
                "assertions": assertions,
                "metadata": case.metadata,
            }
        )
    requests = (
        [target.request_base] if target is not None else [c.request for c in plan.suite.cases]
    )
    profiles = [await app.inspect_run_execution_profile(request) for request in requests]
    material = {
        "suite_id": plan.suite.id,
        "metadata": plan.suite.metadata,
        "cases": cases,
        "application": app.describe().fingerprint,
        "target": None if target is None else target.identity().model_dump(mode="json"),
        "profiles": profiles,
    }
    fingerprint = hashlib.sha256(
        json.dumps(material, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    return {
        "fingerprint": fingerprint,
        "suite_id": plan.suite.id,
        "case_ids": [case.id for case in plan.suite.cases],
        "metadata": plan.suite.metadata,
    }


async def run_process_eval(
    *,
    target: str,
    project_root: Path,
    directory: Path,
    processes: int,
    max_concurrency: int,
    case_timeout_seconds: float | None,
    startup_timeout_seconds: float = 120,
    shutdown_grace_seconds: float = 30,
) -> EvalRun:
    if os.name != "posix":
        raise ValueError("Multi-process Cayu execution currently requires POSIX.")
    if type(processes) is not int or not 1 <= processes <= 256:
        raise ValueError("processes must be between 1 and 256.")
    if type(max_concurrency) is not int or not 1 <= max_concurrency <= EVAL_MAX_CONCURRENCY:
        raise ValueError(f"max_concurrency must be between 1 and {EVAL_MAX_CONCURRENCY}.")
    from math import isfinite

    for value in (case_timeout_seconds, startup_timeout_seconds, shutdown_grace_seconds):
        if value is not None and (not isfinite(value) or value <= 0):
            raise ValueError("Process eval timeouts must be finite and positive.")
    count = min(processes, max_concurrency)
    directory = directory.resolve()
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    launch_id = str(uuid4())
    started_at = datetime.now(UTC)
    manifest = {
        "schema_version": 2,
        "launch_id": launch_id,
        "started_at": started_at.isoformat(),
        "supervisor_pid": os.getpid(),
        "automatic_replay": False,
        "python_version": sys.version.split()[0],
        "runtime_build_provenance": current_runtime_build_provenance().model_dump(mode="json"),
        "target": target,
        "processes": count,
        "max_concurrency": max_concurrency,
        "case_timeout_seconds": case_timeout_seconds,
        "startup_timeout_seconds": startup_timeout_seconds,
        "shutdown_grace_seconds": shutdown_grace_seconds,
    }
    _write_json(directory / "launch.json", manifest)
    print(f"Eval process launch {launch_id}: {directory}", file=sys.stderr, flush=True)
    commands = [
        ProcessWorkerCommand(
            argv=(sys.executable, "-m", "cayu.cli._eval_processes", str(directory), str(index)),
            cwd=project_root,
            environment=process_worker_environment(index, count),
            log_path=directory / f"worker-{index}.log",
        )
        for index in range(count)
    ]
    identity = None
    assignments = []

    async def admit(children) -> None:
        nonlocal identity, assignments
        async with asyncio.timeout(startup_timeout_seconds):
            paths = [directory / f"ready-{index}.json" for index in range(count)]
            while not all(path.is_file() for path in paths):
                if any(child.returncode is not None for child in children):
                    raise RuntimeError("An eval worker exited during admission.")
                await asyncio.sleep(0.02)
            prepared = [_read_json(path) for path in paths]
            identity = prepared[0]["identity"]
            for index, item in enumerate(prepared):
                if (
                    item["launch_id"] != launch_id
                    or item["index"] != index
                    or item["pid"] != children[index].pid
                    or item["identity"] != identity
                ):
                    raise RuntimeError(
                        "Eval worker target/case admission differs; nothing dispatched."
                    )
            assignments = [identity["case_ids"][index::count] for index in range(count)]
            _write_json(
                directory / "start.json",
                {
                    "launch_id": launch_id,
                    "fingerprint": identity["fingerprint"],
                    "assignments": assignments,
                },
            )

    try:
        outcome = await supervise_process_workers(
            commands,
            shutdown_grace_seconds=shutdown_grace_seconds + 1,
            ready=admit,
        )
        if outcome.exit_code:
            raise RuntimeError(
                f"Process eval incomplete (exit {outcome.exit_code}); "
                f"worker evidence retained in {directory}. No automatic replay."
            )
        if identity is None:
            raise RuntimeError("Process eval never reached admission.")
        cases = {}
        provenance = []
        for index, expected in enumerate(assignments):
            if not expected:
                continue
            run = EvalRun.model_validate(_read_json(directory / f"result-{index}.json"))
            if (
                [case.case_id for case in run.cases] != expected
                or run.suite_id != identity["suite_id"]
                or run.metadata != identity["metadata"]
                or run.run_contract is not None
            ):
                raise RuntimeError("Eval worker result does not match its admitted cases.")
            for case in run.cases:
                if case.case_id in cases:
                    raise RuntimeError("Duplicate case in process eval results.")
                cases[case.case_id] = case
            provenance.append(
                {
                    "index": index,
                    "pid": outcome.pids[index],
                    "run_id": run.run_id,
                    "case_ids": expected,
                }
            )
        if set(cases) != set(identity["case_ids"]):
            raise RuntimeError("Process eval results are incomplete.")
        ordered = tuple(cases[key] for key in identity["case_ids"])
        completed_at = datetime.now(UTC)
        result = EvalRun(
            run_id=launch_id,
            suite_id=identity["suite_id"],
            cases=ordered,
            status=aggregate_eval_status(case.status for case in ordered),
            score=aggregate_eval_score(case.score for case in ordered),
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=int((completed_at - started_at).total_seconds() * 1000),
            metadata={
                **identity["metadata"],
                _METADATA_KEY: {
                    "schema_version": 1,
                    "plan_fingerprint": identity["fingerprint"],
                    "max_concurrency": max_concurrency,
                    "workers": provenance,
                },
            },
        )
        _write_json(directory / "completed.json", {"launch_id": launch_id})
        return result
    except BaseException as exc:
        _write_json(
            directory / "incomplete.json",
            {
                "launch_id": launch_id,
                "exception_type": type(exc).__name__,
                "automatic_replay": False,
            },
        )
        raise


async def _worker(directory: Path, index: int) -> None:
    from cayu.cli.evals import _load_eval_plan

    launch = _read_json(directory / "launch.json")
    current = asyncio.current_task()
    assert current is not None
    loop = asyncio.get_running_loop()
    stop_requested = False

    def stop(_signum: int) -> None:
        nonlocal stop_requested
        if stop_requested:
            return
        stop_requested = True
        current.cancel()
        loop.call_later(launch["shutdown_grace_seconds"], os._exit, 124)

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop, sig)
    watcher = asyncio.create_task(watch_supervisor(stop))
    try:
        plan = await _load_eval_plan(launch["target"], label="Process eval target")
        identity = await _plan_identity(plan)
        _write_json(
            directory / f"ready-{index}.json",
            {
                "launch_id": launch["launch_id"],
                "index": index,
                "pid": os.getpid(),
                "identity": identity,
            },
        )
        async with asyncio.timeout(launch["startup_timeout_seconds"]):
            while not (directory / "start.json").exists():
                await asyncio.sleep(0.02)
        admission = _read_json(directory / "start.json")
        if (
            admission["launch_id"] != launch["launch_id"]
            or admission["fingerprint"] != identity["fingerprint"]
        ):
            raise RuntimeError("Process eval admission identity changed.")
        assert plan.suite is not None
        assigned = admission["assignments"][index]
        expected = identity["case_ids"][index :: launch["processes"]]
        if assigned != expected:
            raise RuntimeError("Process eval case assignment changed.")
        if not assigned:
            return
        by_id = {case.id: case for case in plan.suite.cases}
        suite = EvalSuite(
            id=plan.suite.id,
            cases=[by_id[key] for key in assigned],
            metadata=plan.suite.metadata,
        )
        base, extra = divmod(launch["max_concurrency"], launch["processes"])
        capacity = base + (index < extra)
        policy = EvalSuiteTrialPolicyV1.create(
            trial_count=1,
            max_concurrency=launch["max_concurrency"],
        )
        progress = ProcessEvalProgress(
            directory,
            launch_id=launch["launch_id"],
            index=index,
            fingerprint=identity["fingerprint"],
            case_ids=tuple(assigned),
        )
        with progress.activate():
            if plan.workflow_target is not None:
                result = await run_workflow_eval_suite(
                    plan.workflow_target,
                    suite,
                    max_concurrency=capacity,
                    case_timeout_seconds=launch["case_timeout_seconds"],
                    trial_policy=policy,
                )
            else:
                assert plan.app is not None
                result = await run_eval_suite(
                    plan.app,
                    suite,
                    max_concurrency=capacity,
                    case_timeout_seconds=launch["case_timeout_seconds"],
                    trial_policy=policy,
                )
            if await _plan_identity(plan) != identity:
                raise RuntimeError("Process eval target changed during execution.")
            _write_json(directory / f"result-{index}.json", json.loads(eval_run_to_json(result)))
    finally:
        watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watcher


if __name__ == "__main__":
    try:
        launch_directory = Path(sys.argv[1])
        grace = _read_json(launch_directory / "launch.json")["shutdown_grace_seconds"]
        with supervisor_watchdog(grace):
            asyncio.run(_worker(launch_directory, int(sys.argv[2])))
    except BaseException as error:
        print(f"Process eval worker failed ({type(error).__name__}).", file=sys.stderr)
        sys.exit(2)
