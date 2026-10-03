"""A stopped worker's durable non-success binding cleanup remains discoverable."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from tests.core.test_completion_verifier_adapters import _contract
from tests.core.test_verified_task_worker import _StaticHandler
from tests.core.test_verified_worker_limit_cleanup import _limit_app
from tests.core.verified_worker_fixtures import (
    VerifiedWorkerStoreFactory,
    wait_for_verified_worker_lease_expiry,
)
from tests.core.verified_worker_fixtures import (
    verified_work_postgres_dsn as verified_work_postgres_dsn,
)
from tests.core.verified_worker_fixtures import (
    verified_worker_store_factory as verified_worker_store_factory,
)

import cayu
from cayu import (
    Environment,
    EnvironmentSpec,
    LocalWorkspace,
    SyncBinding,
    TaskCreate,
    TaskQuery,
    TaskStatus,
    VerifiedTaskWorker,
)


def _app_with_binding(sessions, tasks, root, reason):
    app, provider, verifier = _limit_app(sessions, tasks, reason)
    source, target = root / "source", root / "target"
    source.mkdir(exist_ok=True)
    target.mkdir(exist_ok=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="coding"),
            workspace=LocalWorkspace(source, workspace_id="source"),
            binding=SyncBinding(
                target_workspace=LocalWorkspace(target, workspace_id="target"),
                sync_back="always",
                source_conflict_policy="require_revision",
                max_file_bytes=1024,
                max_total_bytes=4096,
            ),
        ),
        default=True,
    )
    return app, provider, verifier


async def _crash_during_finalization(root, reason, dsn):
    sessions, tasks = VerifiedWorkerStoreFactory("postgres" if dsn else "sqlite", root, dsn)()
    app, provider, _ = _app_with_binding(sessions, tasks, root, reason)

    async def crash(self, bound, *, outcome=None, metadata=None):
        (task,) = await tasks.list_tasks(TaskQuery())
        admission = await tasks.load_latest_work_attempt_admission(task.id)
        checkpoint = await sessions.load_checkpoint(admission.session_id)
        marker = checkpoint["pending_completion_finalization"]
        assert marker["outcome"] in {"failed", "interrupted"}
        assert admission.execution_stop.request.reason == reason
        assert len(provider.requests) == int(reason == "elapsed_limit")
        # Native marker publication precedes this binding effect. Skip all
        # finally blocks, just as process death does; no synthetic checkpoint.
        os._exit(79)

    SyncBinding.finalize = crash
    await tasks.publish_work_contract(_contract())
    await tasks.create_task(TaskCreate(type="verified", work_contract=_contract().reference()))
    async with VerifiedTaskWorker(
        app,
        _StaticHandler(),
        worker_id="terminal-cleanup-crash",
        lease_seconds=5,
        callback_timeout_seconds=1,
        max_elapsed_seconds=3 if reason == "elapsed_limit" else 3600,
    ) as worker:
        await worker.run(max_tasks=1)


async def _assert_rejected_marker_scans(app, tasks, admission, monkeypatch):
    """Untrusted/conflicting observations cannot turn cleanup into execution."""
    engine = app._session_engine
    snapshot = engine._load_work_attempt_execution_snapshot
    task = await tasks.load_task(admission.task_id)
    for field, value, error in (
        ("outcome", "completed", None),
        ("execution_profile_fingerprint", "0" * 64, None),
        ("task_id", admission.task_id, None),
        ("version", True, "unsupported format"),
        ("outcome", "future-outcome", "unsupported format"),
    ):

        async def conflicting_snapshot(candidate, field=field, value=value):
            session, checkpoint = await snapshot(candidate)
            checkpoint = deepcopy(checkpoint)
            checkpoint["pending_completion_finalization"][field] = value
            return session, checkpoint

        stop = asyncio.Event()
        handler = _StaticHandler()
        with monkeypatch.context() as faults:
            faults.setattr(engine, "_load_work_attempt_execution_snapshot", conflicting_snapshot)
            async with VerifiedTaskWorker(
                app, handler, worker_id="reject-conflicting-finalization"
            ) as worker:
                discover = worker._discover_unfinished_attempt

                async def one_scan(discover=discover, stop=stop):
                    try:
                        return await discover()
                    finally:
                        stop.set()

                faults.setattr(worker, "_discover_unfinished_attempt", one_scan)
                if error is None:
                    assert await asyncio.wait_for(worker.run(stop=stop), 10) == 0
                else:
                    with pytest.raises(ValueError, match=error):
                        await asyncio.wait_for(worker.run(stop=stop), 10)
        assert await tasks.load_task(task.id) == task
        assert await tasks.load_latest_work_attempt_admission(task.id) == admission
        assert await tasks.load_work_attempt_lifecycle_receipt(admission.admission_id) is None
        assert handler.preparations == handler.proposals == []


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize("reason", ["budget_limit", "elapsed_limit"])
def test_replacement_worker_recovers_terminal_binding_marker(
    backend, reason, verified_worker_store_factory, monkeypatch
):
    factory = verified_worker_store_factory
    repository = Path(__file__).resolve().parents[2]
    # Preserve the selected parent import for installed-wheel qualification;
    # never silently switch a subprocess back to the checkout's src directory.
    environment = {
        **os.environ,
        "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent),
    }
    environment.pop("CAYU_TEST_VERIFIED_WORKER_DSN", None)
    if factory.postgres_dsn is not None:
        environment["CAYU_TEST_VERIFIED_WORKER_DSN"] = factory.postgres_dsn
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "import asyncio, os, sys, cayu; from pathlib import Path; "
            "assert Path(cayu.__file__).resolve() == Path(sys.argv[3]).resolve(); "
            "from tests.core.test_verified_worker_terminal_finalization "
            "import _crash_during_finalization; "
            "asyncio.run(_crash_during_finalization(Path(sys.argv[1]), sys.argv[2], "
            "os.environ.get('CAYU_TEST_VERIFIED_WORKER_DSN')))",
            str(factory.directory),
            reason,
            cayu.__file__,
        ],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert child.returncode == 79, (child.stdout, child.stderr)

    async def scenario():
        sessions, tasks = factory()
        try:
            (task,) = await tasks.list_tasks(TaskQuery())
            admission = await tasks.load_latest_work_attempt_admission(task.id)
            app, provider, verifier = _app_with_binding(sessions, tasks, factory.directory, reason)
            assert (
                await app._session_engine.load_work_attempt_released_recovery_evidence(admission)
                is None
            )
            assert await tasks.load_work_attempt_lifecycle_receipt(admission.admission_id) is None
            await wait_for_verified_worker_lease_expiry(tasks, admission.claim.lease_expires_at)
            checkpoint = await sessions.load_checkpoint(admission.session_id)
            await _assert_rejected_marker_scans(app, tasks, admission, monkeypatch)
            assert await sessions.load_checkpoint(admission.session_id) == checkpoint
            handler = _StaticHandler()
            async with VerifiedTaskWorker(
                app,
                handler,
                worker_id="terminal-cleanup-replacement",
                lease_seconds=5,
                callback_timeout_seconds=1,
            ) as worker:
                assert await asyncio.wait_for(worker.run(max_tasks=1), 30) == 1
            current = await tasks.load_latest_work_attempt_admission(task.id)
            receipt = await tasks.load_work_attempt_lifecycle_receipt(admission.admission_id)
            final = await tasks.load_task(task.id)
            assert final.status is TaskStatus.NEEDS_ATTENTION
            assert final.status_reason == "work_contract_" + reason
            assert final.worker_id is final.lease_expires_at is None
            assert receipt.task == final
            assert current.execution_stop == admission.execution_stop
            assert current.execution_entry == admission.execution_entry
            assert current.claim.generation == 2
            assert "pending_completion_finalization" not in await sessions.load_checkpoint(
                admission.session_id
            )
            assert provider.requests == verifier.requests == []
            assert handler.preparations == handler.proposals == []
            assert await tasks.settle_work_attempt_lifecycle(receipt.request) == receipt
            assert not app._environment_lifecycle._active_environment_setups
        finally:
            await tasks.close()
            await sessions.close()

    asyncio.run(scenario())
