"""Terminal interruption must not discard a killed worker's environment owner."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from tests.recovery.test_named_check_durable_recovery import build_app, wait_file

from cayu import InterruptSessionRequest
from cayu.runtime import (
    RecoveryDecision,
    RecoveryExecutionRequest,
    RecoveryPlanAction,
    RecoveryPlanRequest,
    RecoveryPlanSelection,
)

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Real SIGKILL worker loss")


@pytest.mark.parametrize(
    ("backend", "fault"),
    [
        ("sqlite", None),
        ("sqlite", "missing"),
        ("sqlite", "allocation"),
        ("sqlite", "marker_ack_loss"),
        ("sqlite", "cancel_reconnect"),
        ("postgres", "marker_ack_loss"),
    ],
)
def test_terminal_recovery_disposes_allocation_after_policy_planning_worker_loss(
    tmp_path, fault, backend
):
    if backend == "postgres":
        dsn = os.environ.get("CAYU_TEST_TERMINAL_RECOVERY_POSTGRES_DSN")
        if not dsn:
            pytest.skip("Requires a dedicated disposable terminal-recovery PostgreSQL database")
        (tmp_path / "postgres-dsn").write_text(dsn)
    source = tmp_path / "source"
    source.mkdir()
    (source / "preserved.txt").write_text("original source remains owned")
    (tmp_path / "factory-mode").touch()
    (tmp_path / "pause-policy").touch()
    worker = subprocess.Popen(
        [sys.executable, "-m", "tests.recovery.test_named_check_durable_recovery", str(tmp_path)]
    )
    try:
        wait_file(tmp_path / "policy-started", worker)
        worker.kill()
        assert worker.wait(timeout=10) == -signal.SIGKILL
        (tmp_path / "release-policy").touch()

        async def recover():
            with pytest.MonkeyPatch.context() as patch:
                app, store, provider = build_app(tmp_path, patch)
                try:
                    task = await app.task_store.load_task("check-task")
                    remaining = (task.lease_expires_at - datetime.now(UTC)).total_seconds()
                    if remaining > 0:
                        await asyncio.sleep(remaining + 0.05)
                    request = RecoveryPlanRequest(
                        selection=RecoveryPlanSelection(
                            session_ids=("recovery-check",), inactive_for_seconds=0
                        )
                    )

                    async def execute(identity):
                        plan = await app.plan_recovery(request)
                        assert (
                            RecoveryPlanAction.AUTOMATIC_REPAIR in plan.items[0].allowed_actions
                        ), plan.model_dump_json()
                        return await app.execute_recovery(
                            RecoveryExecutionRequest(
                                plan=plan,
                                decisions=(
                                    RecoveryDecision(
                                        item_id=plan.items[0].item_id,
                                        action=RecoveryPlanAction.AUTOMATIC_REPAIR,
                                    ),
                                ),
                                execution_id=identity,
                            )
                        )

                    first = await execute("recover-policy")
                    assert first.items[0].error_code is None, first
                    assert not provider.requests
                    assert not (tmp_path / "started").exists()
                    async for _ in app.interrupt_session(
                        InterruptSessionRequest(session_id="recovery-check", reason="Operator stop")
                    ):
                        pass
                    assert await app.drain_background_interruptions(timeout_s=10)
                    if fault in {"missing", "allocation"}:
                        load = store.load_session_operation

                        async def altered(session_id, key):
                            value = await load(session_id, key)
                            if key.startswith("command-binding:"):
                                if fault == "missing":
                                    return None
                                value["allocation"] = "different-allocation"
                            return value

                        patch.setattr(store, "load_session_operation", altered)
                        before = await store.load_checkpoint("recovery-check")
                        plan = await app.plan_recovery(request)
                        assert plan.items[0].allowed_actions == (RecoveryPlanAction.LEAVE_INTACT,)
                        assert await store.load_checkpoint("recovery-check") == before
                        assert not (tmp_path / "disposed").exists()
                        assert not (tmp_path / "reconnected").exists()
                        assert not provider.requests
                        return
                    if fault == "marker_ack_loss":
                        from cayu.runtime._environment_lifecycle import EnvironmentLifecycle

                        checkpoint_marker = (
                            EnvironmentLifecycle.checkpoint_terminal_binding_finalization
                        )

                        async def lost_ack(self, **kwargs):
                            await checkpoint_marker(self, **kwargs)
                            raise ConnectionError("lost terminal marker acknowledgement")

                        patch.setattr(
                            EnvironmentLifecycle,
                            "checkpoint_terminal_binding_finalization",
                            lost_ack,
                        )
                    if fault == "cancel_reconnect":
                        (tmp_path / "cancel-reconnect").touch()
                        operation = asyncio.create_task(execute("recover-terminal"))
                        async with asyncio.timeout(10):
                            while not (tmp_path / "reconnect-waiting").exists():
                                if operation.done():
                                    pytest.fail(
                                        f"Recovery exited before reconnect: {operation.result()}"
                                    )
                                await asyncio.sleep(0.01)
                        operation.cancel()
                        assert operation.cancelling() == 1
                        (tmp_path / "release-reconnect").touch()
                        with pytest.raises(asyncio.CancelledError):
                            await operation
                        assert operation.cancelled() and operation.cancelling() == 1
                        assert not provider.requests
                        assert not (tmp_path / "started").exists()
                        return
                    second = await execute("recover-terminal")
                    if fault == "marker_ack_loss":
                        assert second.items[0].error_code == "ConnectionError"
                        checkpoint = await store.load_checkpoint("recovery-check")
                        assert (
                            checkpoint["pending_completion_finalization"]["outcome"]
                            == "interrupted"
                        )
                        assert not (tmp_path / "disposed").exists()
                        assert not (tmp_path / "reconnected").exists()
                        return
                    assert second.items[0].error_code is None, second
                    assert not provider.requests
                    assert not (tmp_path / "started").exists()
                    assert (tmp_path / "disposed").exists()
                    events = await store.load_events("recovery-check")
                    assert any(e.type == "environment.binding.finalize_completed" for e in events)
                    assert (await store.load_state("recovery-check")).status.value == "interrupted"
                    final_plan = await app.plan_recovery(request)
                    assert final_plan.items[0].registration.status.value == "ready", final_plan
                    assert final_plan.items[0].allowed_actions == (RecoveryPlanAction.LEAVE_INTACT,)
                    assert (source / "preserved.txt").read_text() == "original source remains owned"
                finally:
                    assert await app.drain_recovery_cleanups()
                    assert await app.drain_environment_cleanups()
                    await store.close()
                    await app.task_store.close()

        asyncio.run(recover())
        if fault in {"marker_ack_loss", "cancel_reconnect"}:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tests.recovery.test_terminal_factory_worker_loss",
                    str(tmp_path),
                ],
                check=True,
                timeout=30,
            )
    finally:
        if worker.poll() is None:
            os.kill(worker.pid, signal.SIGKILL)
        worker.wait(timeout=10)


async def finish_after_restart(root: Path):
    with pytest.MonkeyPatch.context() as patch:
        app, store, provider = build_app(root, patch)
        try:
            plan = await app.plan_recovery(
                RecoveryPlanRequest(
                    selection=RecoveryPlanSelection(
                        session_ids=("recovery-check",),
                        inactive_for_seconds=0,
                    )
                )
            )
            if (root / "cancel-reconnect").exists():
                assert plan.items[0].registration.status.value == "ready", plan
                assert (root / "disposed").exists()
                assert not provider.requests
                assert not (root / "started").exists()
                checkpoint = await store.load_checkpoint("recovery-check")
                assert "pending_completion_finalization" not in checkpoint
                events = await store.load_events("recovery-check")
                assert any(e.type == "environment.binding.finalize_completed" for e in events)
                return
            assert RecoveryPlanAction.AUTOMATIC_REPAIR in plan.items[0].allowed_actions, plan
            request = RecoveryExecutionRequest(
                plan=plan,
                decisions=(
                    RecoveryDecision(
                        item_id=plan.items[0].item_id, action=RecoveryPlanAction.AUTOMATIC_REPAIR
                    ),
                ),
                execution_id="finish-terminal-after-restart",
            )
            receipt = await app.execute_recovery(request)
            assert receipt.items[0].error_code is None, receipt
            assert (root / "disposed").exists()
            assert not (root / "started").exists()
            assert not provider.requests
            events = await store.load_events("recovery-check")
            replay = await app.execute_recovery(request)
            assert replay.items[0].replayed
            assert await store.load_events("recovery-check") == events
            assert (await store.load_state("recovery-check")).status.value == "interrupted"
            assert "pending_completion_finalization" not in await store.load_checkpoint(
                "recovery-check"
            )
            final_plan = await app.plan_recovery(plan.request)
            assert final_plan.items[0].registration.status.value == "ready", final_plan
            assert final_plan.items[0].allowed_actions == (RecoveryPlanAction.LEAVE_INTACT,)
        finally:
            await store.close()
            await app.task_store.close()


if __name__ == "__main__":
    asyncio.run(finish_after_restart(Path(sys.argv[1])))
