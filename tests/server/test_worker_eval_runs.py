"""Eval runs a project's eval worker runs, so the server only admits them."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi.testclient import TestClient
from tests.evals.test_corpus_execution import _corpus, _provider, _target
from tests.server.test_server_evals import _AUTH_HEADERS, _authenticate

import cayu.cli.worker as worker_cli
from cayu.cloud_reporting import CloudSchedule, EvalRunScheduleSource, read_cloud_schedule
from cayu.evals.execution_profiles import EvalExecutionProfilePolicyV1
from cayu.evals.store import EvalRunInvocation, EvalRunRequest, EvalRunStatus
from cayu.project_control_plane import (
    ProjectControlPlaneAccess,
    ProjectEvalTarget,
    _create_project_control_plane_context,
)
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.server import ServerConfig, create_server
from cayu.server.cloud_reporting import CLOUD_READER_TOKEN_SHA256_VARIABLE
from cayu.server.evals_registry import (
    EvalTargetRegistry,
    declared_eval_target_registrations,
    resolved_evals_runtime,
    target_for_eval_invocation,
)
from cayu.server.evals_worker import EvalRunCoordinator
from cayu.storage.evals_sqlite import SQLiteEvalStore
from cayu.tasks.until_idle import TaskWorkerIdleResult

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_READER_TOKEN = "cloud-reader-token"
_TERMINAL = {EvalRunStatus.COMPLETED, EvalRunStatus.FAILED, EvalRunStatus.CANCELLED}


class _GatedProvider(ModelProvider):
    """Answers "Approved" once released, so a run can be held in progress."""

    name = "gated-eval-provider"

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        self.started.set()
        await self.release.wait()
        yield ModelStreamEvent.text_delta("Approved")
        yield ModelStreamEvent.completed(
            {
                "finish_reason": "stop",
                "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
            }
        )


def _declared(key: str, provider: ModelProvider | None = None) -> ProjectEvalTarget:
    return ProjectEvalTarget(
        reference=f"evals:{key.replace('-', '_')}",
        target=_target(_provider(trials=1) if provider is None else provider, key=key),
        execution_profile_policy=EvalExecutionProfilePolicyV1.safe_default(),
    )


def _runtime(store, *declared: ProjectEvalTarget, claimed: tuple[str, ...] | None = None):
    runtime = resolved_evals_runtime(
        explicit=None,
        registry=EvalTargetRegistry(
            declared_eval_target_registrations(declared, project_root=_PROJECT_ROOT)
        ),
        automatic_store=store,
    )
    assert runtime is not None
    return replace(
        runtime, poll_interval_seconds=0.01, lease_seconds=5, claimed_target_keys=claimed
    )


async def _admit(runtime, store, declared: ProjectEvalTarget, run_id: str) -> str:
    target = declared.target
    corpus = _corpus(target_key=target.key, trials=1)
    registration = runtime.registry.registration(target.key)
    invocation = EvalRunInvocation()
    prepared = await runtime.registry.prepare_execution_profile(
        target.key,
        effective_target=target_for_eval_invocation(registration.execution_target(), invocation),
    )
    await store.save_corpus(corpus, redact_json=target.app.redact_json)
    await store.admit_run(
        EvalRunRequest(
            run_id=run_id,
            idempotency_key="sha256:" + hashlib.sha256(run_id.encode()).hexdigest(),
            corpus_revision=corpus.revision,
            target_key=target.key,
            suite_id=corpus.suites[0].id,
            suite_revision=corpus.suites[0].revision,
            max_concurrency=1,
            invocation=invocation.model_copy(
                update={
                    "execution_profile": prepared.binding,
                    "execution_profile_snapshot": prepared.snapshot,
                },
                deep=True,
            ),
        ),
        redact_json=target.app.redact_json,
    )
    return run_id


async def _status(store, run_id: str) -> EvalRunStatus:
    record = await store.load_run(run_id)
    assert record is not None
    return record.status


def test_running_until_idle_takes_only_the_claimed_targets_runs(tmp_path: Path) -> None:
    async def run() -> None:
        store = SQLiteEvalStore(tmp_path / "evals.db")
        worker, served = _declared("worker-agent"), _declared("served-agent")
        runtime = _runtime(store, worker, served, claimed=("worker-agent",))
        try:
            worker_run = await _admit(runtime, store, worker, "worker-run")
            served_run = await _admit(runtime, store, served, "served-run")

            assert await EvalRunCoordinator(runtime).run_until_idle() == (1, True)

            assert await _status(store, worker_run) is EvalRunStatus.COMPLETED
            assert await _status(store, served_run) is EvalRunStatus.QUEUED
        finally:
            await store.close()

    asyncio.run(run())


def test_a_drained_coordinator_finishes_the_run_it_holds_and_claims_no_other(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        store = SQLiteEvalStore(tmp_path / "evals.db")
        provider = _GatedProvider()
        declared = _declared("worker-agent", provider)
        runtime = _runtime(store, declared)
        coordinator = EvalRunCoordinator(runtime)
        try:
            held = await _admit(runtime, store, declared, "held-run")
            coordinator.start()
            await asyncio.wait_for(provider.started.wait(), timeout=10)
            waiting = await _admit(runtime, store, declared, "waiting-run")

            draining = asyncio.create_task(coordinator.drain())
            await asyncio.sleep(0.2)
            assert not draining.done()
            provider.release.set()
            await asyncio.wait_for(draining, timeout=10)

            assert await _status(store, held) is EvalRunStatus.COMPLETED
            assert await _status(store, waiting) is EvalRunStatus.QUEUED
        finally:
            provider.release.set()
            await coordinator.stop()
            await store.close()

    asyncio.run(run())


def test_the_schedule_counts_the_runs_waiting_for_the_eval_worker(tmp_path: Path) -> None:
    async def run() -> None:
        store = SQLiteEvalStore(tmp_path / "evals.db")
        first, second = _declared("first-agent"), _declared("second-agent")
        runtime = _runtime(store, first, second)
        app = first.target.app
        try:
            await _admit(runtime, store, first, "first-run")
            await _admit(runtime, store, second, "second-run")
            await _admit(runtime, store, second, "third-run")
            await store.request_cancel("third-run")

            schedule = await read_cloud_schedule(
                app,
                eval_runs=EvalRunScheduleSource(
                    store=store, target_keys=("first-agent", "second-agent")
                ),
            )
            assert schedule.eval_runs is not None
            assert (schedule.eval_runs.queued, schedule.eval_runs.running) == (2, 0)
            assert (await read_cloud_schedule(app)).eval_runs is None
        finally:
            await store.close()

    asyncio.run(run())


def test_the_server_admits_the_eval_workers_runs_and_leaves_them_to_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        CLOUD_READER_TOKEN_SHA256_VARIABLE, hashlib.sha256(_READER_TOKEN.encode()).hexdigest()
    )
    declared = _declared("worker-agent")
    context = _create_project_control_plane_context(
        project_root=_PROJECT_ROOT,
        project_id="worker-evals",
        eval_store=SQLiteEvalStore(tmp_path / "evals.db"),
        store_backend="sqlite",
        store_source="project",
        access=ProjectControlPlaneAccess.AUTHENTICATED_PRODUCTION,
        eval_targets=(declared,),
        eval_worker="evals",
    )
    try:
        server = create_server(
            _target(_provider(trials=1)).app,
            config=ServerConfig.protected(_authenticate),
            project_context=context,
        )
        with TestClient(server) as client:
            catalog = client.get("/api/evals/targets", headers=_AUTH_HEADERS).json()
            item = next(i for i in catalog["items"] if i["target_key"] == "worker-agent")
            corpus = _corpus(target_key="worker-agent", trials=1)
            assert (
                client.post(
                    "/api/evals/corpora", headers=_AUTH_HEADERS, json=corpus.model_dump(mode="json")
                ).status_code
                == 201
            )
            admitted = client.post(
                "/api/evals/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "worker-run"},
                json={
                    "corpus_revision": corpus.revision,
                    "suite_id": corpus.suites[0].id,
                    "expected_execution_profile_revision": item["execution_profile"]["revision"],
                },
            )
            assert admitted.status_code == 202
            run_id = admitted.json()["spec"]["run_id"]

            # The server's own coordinator would have finished it well within this.
            for _ in range(20):
                assert (
                    client.get(f"/api/evals/runs/{run_id}", headers=_AUTH_HEADERS).json()["status"]
                    == "queued"
                )
                time.sleep(0.05)
            schedule = client.get(
                "/api/cloud/v1/schedule", headers={"Authorization": f"Bearer {_READER_TOKEN}"}
            )
            assert schedule.status_code == 200
            eval_runs = CloudSchedule.model_validate(schedule.json()).eval_runs
            assert eval_runs is not None
            assert (eval_runs.queued, eval_runs.running) == (1, 0)
            contract = client.get(
                "/api/cloud/contract", headers={"Authorization": f"Bearer {_READER_TOKEN}"}
            ).json()
            assert "eval_runs" in contract["features"]
    finally:
        asyncio.run(context.close())


def test_a_worker_that_runs_until_idle_serves_its_tasks_while_an_eval_run_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SQLiteEvalStore(tmp_path / "evals.db")
    provider = _GatedProvider()
    declared = _declared("worker-agent", provider)
    runtime = _runtime(store, declared)
    run_id = asyncio.run(_admit(runtime, store, declared, "worker-run"))
    entrypoint_runs = 0

    async def tasks_idle(app, stop) -> TaskWorkerIdleResult:
        nonlocal entrypoint_runs
        entrypoint_runs += 1
        # The eval run's workflow could create a task for this worker at any time.
        if entrypoint_runs == 3:
            provider.release.set()
        return TaskWorkerIdleResult(handled=0, idle=True)

    monkeypatch.setattr(worker_cli, "_EVAL_TASK_SERVICE_SECONDS", 0.05)
    try:
        exit_code = asyncio.run(
            worker_cli._invoke_worker(
                tasks_idle,
                _target(_provider(trials=1)).app,
                name="evals",
                shutdown_grace_seconds=5,
                until_idle=True,
                evals=EvalRunCoordinator(runtime),
                eval_apps=(declared.target.app,),
            )
        )
        assert exit_code == 0
        assert entrypoint_runs >= 4
        assert asyncio.run(_status(store, run_id)) is EvalRunStatus.COMPLETED
        assert declared.target.app.lifecycle_state == "closed"
    finally:
        asyncio.run(store.close())


def test_a_worker_that_runs_until_idle_reports_the_eval_runs_it_took(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = SQLiteEvalStore(tmp_path / "evals.db")
    declared = _declared("worker-agent")
    runtime = _runtime(store, declared)
    run_id = asyncio.run(_admit(runtime, store, declared, "worker-run"))

    handled = iter((2,))

    async def tasks_idle(app, stop) -> TaskWorkerIdleResult:
        # Two tasks the first time; it runs again while and after the eval runs go on.
        return TaskWorkerIdleResult(handled=next(handled, 0), idle=True)

    try:
        exit_code = asyncio.run(
            worker_cli._invoke_worker(
                tasks_idle,
                _target(_provider(trials=1)).app,
                name="evals",
                shutdown_grace_seconds=5,
                until_idle=True,
                evals=EvalRunCoordinator(runtime),
            )
        )
        assert exit_code == 0
        assert asyncio.run(_status(store, run_id)) is EvalRunStatus.COMPLETED
        result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert (result["handled"], result["idle"]) == (3, True)
    finally:
        asyncio.run(store.close())


def test_a_drained_worker_finishes_its_eval_run_before_it_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SQLiteEvalStore(tmp_path / "evals.db")
    provider = _GatedProvider()
    declared = _declared("worker-agent", provider)
    runtime = _runtime(store, declared)
    run_id = asyncio.run(_admit(runtime, store, declared, "worker-run"))

    async def cloud_drains_once_the_run_started(on_drain) -> None:
        await provider.started.wait()
        on_drain(
            worker_cli.DrainRequest(release_id="dep_old", replaced_by="dep_new", deadline_at=None)
        )
        # The drained worker must keep its run going until it finishes.
        await asyncio.sleep(0.2)
        provider.release.set()

    async def tasks_until_stopped(app, stop) -> None:
        await stop.wait()

    monkeypatch.setattr(worker_cli, "watch_cloud_drain", cloud_drains_once_the_run_started)
    try:
        exit_code = asyncio.run(
            worker_cli._invoke_worker(
                tasks_until_stopped,
                _target(_provider(trials=1)).app,
                name="evals",
                shutdown_grace_seconds=5,
                evals=EvalRunCoordinator(runtime),
            )
        )
        assert exit_code == 0
        assert asyncio.run(_status(store, run_id)) is EvalRunStatus.COMPLETED
    finally:
        asyncio.run(store.close())


def test_a_drained_coordinator_running_until_idle_takes_no_run(tmp_path: Path) -> None:
    async def run() -> None:
        store = SQLiteEvalStore(tmp_path / "evals.db")
        declared = _declared("worker-agent")
        runtime = _runtime(store, declared)
        coordinator = EvalRunCoordinator(runtime)
        try:
            queued = await _admit(runtime, store, declared, "queued-run")
            coordinator.request_drain()

            assert await coordinator.run_until_idle() == (0, False)
            assert await _status(store, queued) is EvalRunStatus.QUEUED
        finally:
            await store.close()

    asyncio.run(run())


def test_running_until_idle_takes_over_a_run_whose_worker_died(tmp_path: Path) -> None:
    async def run() -> None:
        store = SQLiteEvalStore(tmp_path / "evals.db")
        declared = _declared("worker-agent")
        runtime = _runtime(store, declared)
        try:
            orphaned = await _admit(runtime, store, declared, "orphaned-run")
            # A worker claimed it and died: nothing renews its one-second lease.
            assert await store.claim_run(target_key="worker-agent", lease_seconds=1) is not None

            assert await EvalRunCoordinator(runtime).run_until_idle() == (1, True)
            assert await _status(store, orphaned) is EvalRunStatus.COMPLETED
        finally:
            await store.close()

    asyncio.run(run())


def test_a_worker_drained_while_its_entrypoint_still_works_takes_no_new_eval_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SQLiteEvalStore(tmp_path / "evals.db")
    declared = _declared("worker-agent")
    runtime = _runtime(store, declared)
    finished = asyncio.Event()

    async def cloud_drains_at_once(on_drain) -> None:
        on_drain(
            worker_cli.DrainRequest(release_id="dep_old", replaced_by="dep_new", deadline_at=None)
        )
        # Admitted for the replacement while this worker's entrypoint still finishes a task.
        await _admit(runtime, store, declared, "replacement-run")
        await asyncio.sleep(0.3)
        finished.set()

    async def entrypoint_finishing_a_task(app, stop) -> None:
        await stop.wait()
        await finished.wait()

    monkeypatch.setattr(worker_cli, "watch_cloud_drain", cloud_drains_at_once)
    try:
        exit_code = asyncio.run(
            worker_cli._invoke_worker(
                entrypoint_finishing_a_task,
                _target(_provider(trials=1)).app,
                name="evals",
                shutdown_grace_seconds=5,
                evals=EvalRunCoordinator(runtime),
            )
        )
        assert exit_code == 0
        assert asyncio.run(_status(store, "replacement-run")) is EvalRunStatus.QUEUED
    finally:
        asyncio.run(store.close())


def test_a_coordinator_stopped_before_it_started_takes_no_run(tmp_path: Path) -> None:
    async def run() -> None:
        store = SQLiteEvalStore(tmp_path / "evals.db")
        declared = _declared("worker-agent")
        runtime = _runtime(store, declared)
        coordinator = EvalRunCoordinator(runtime)
        try:
            queued = await _admit(runtime, store, declared, "queued-run")
            await coordinator.stop()

            assert await coordinator.run_until_idle() == (0, False)
            assert await _status(store, queued) is EvalRunStatus.QUEUED
        finally:
            await store.close()

    asyncio.run(run())


def test_a_worker_that_runs_until_idle_takes_an_eval_run_its_last_task_pass_saw_admitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SQLiteEvalStore(tmp_path / "evals.db")
    declared = _declared("worker-agent")
    runtime = _runtime(store, declared)
    calls = 0

    async def a_task_admits_an_eval_run(app, stop) -> TaskWorkerIdleResult:
        nonlocal calls
        calls += 1
        if calls == 2:
            # The second pass runs after an empty eval pass; its task gets a run admitted.
            await _admit(runtime, store, declared, "late-run")
            return TaskWorkerIdleResult(handled=1, idle=True)
        return TaskWorkerIdleResult(handled=0, idle=True)

    monkeypatch.setattr(worker_cli, "_EVAL_TASK_SERVICE_SECONDS", 0.05)
    try:
        exit_code = asyncio.run(
            worker_cli._invoke_worker(
                a_task_admits_an_eval_run,
                _target(_provider(trials=1)).app,
                name="evals",
                shutdown_grace_seconds=5,
                until_idle=True,
                evals=EvalRunCoordinator(runtime),
            )
        )
        assert exit_code == 0
        assert asyncio.run(_status(store, "late-run")) is EvalRunStatus.COMPLETED
    finally:
        asyncio.run(store.close())


def test_the_worker_closes_its_eval_store_in_the_loop_that_used_it(tmp_path: Path) -> None:
    # A PostgreSQL pool's connections belong to the loop that opened them.
    used: list[asyncio.AbstractEventLoop] = []
    closed: list[asyncio.AbstractEventLoop] = []

    class LoopBoundEvalStore(SQLiteEvalStore):
        async def claim_run(self, **kwargs):
            used.append(asyncio.get_running_loop())
            return await super().claim_run(**kwargs)

        async def close(self) -> None:
            closed.append(asyncio.get_running_loop())
            await super().close()

    store = LoopBoundEvalStore(tmp_path / "evals.db")
    declared = _declared("worker-agent")
    context = _create_project_control_plane_context(
        project_root=_PROJECT_ROOT,
        project_id=None,
        eval_store=store,
        store_backend="sqlite",
        store_source="project",
        access=ProjectControlPlaneAccess.AUTHENTICATED_PRODUCTION,
        eval_targets=(declared,),
        eval_worker="evals",
    )

    async def tasks_idle(app, stop) -> TaskWorkerIdleResult:
        return TaskWorkerIdleResult(handled=0, idle=True)

    exit_code = asyncio.run(
        worker_cli._invoke_worker(
            tasks_idle,
            _target(_provider(trials=1)).app,
            name="evals",
            shutdown_grace_seconds=5,
            until_idle=True,
            evals=EvalRunCoordinator(_runtime(store, declared)),
            eval_context=context,
        )
    )

    assert exit_code == 0
    assert used and closed == [used[0]]
