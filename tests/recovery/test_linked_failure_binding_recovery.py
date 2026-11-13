"""Committed linked failures retain their environment cleanup owner."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta

import pytest
from tests.recovery.test_model_terminal_binding_recovery import (
    _application,
    _database,
)
from tests.runtime.test_orphaned_session_reconciliation import database_url as database_url

from cayu import EventType, Message, ResumeRequest, RunRequest, SessionStatus
from cayu.providers.base import ModelStreamEvent
from cayu.tasks import TaskCreate, TaskStatus, run_task_worker, run_task_worker_until_idle


def _crash_linked_failure(root):
    async def scenario():
        with pytest.MonkeyPatch.context() as patch:
            app, _store, provider = _application(root, patch)

            async def fail(_request):
                assert (root / "allocated").exists()
                yield ModelStreamEvent.error("linked failure with allocated environment")
                yield ModelStreamEvent.completed({"finish_reason": "error"})

            patch.setattr(provider, "stream", fail)
            if os.environ.get("CAYU_LINKED_FAILURE_MARKER") == "v0.19.1":
                import cayu.runtime._session_outcome_publication as publication

                write_marker = publication.runtime_task_failure_payload

                def v0_19_1_marker(*args, **kwargs):
                    # v0.19.1 wrote the same v2 marker without the selected event id.
                    payload = write_marker(*args, **kwargs)
                    del payload["runtime_task_failure"]["session_terminal_event_id"]
                    return payload

                patch.setattr(publication, "runtime_task_failure_payload", v0_19_1_marker)
            await app.create_task(TaskCreate(task_id="linked", type="work"))
            fail_task = app._session_finalization.fail_task
            elect = app._session_outcomes.claim_invocation_terminal_decision
            loss = os.environ.get("CAYU_LINKED_FAILURE_LOSS", "task")

            async def die_after_task(**kwargs):
                result = await fail_task(**kwargs)
                if loss == "task":
                    os._exit(137)
                return result

            async def die_after_decision(**kwargs):
                await elect(**kwargs)
                if loss == "decision":
                    os._exit(137)

            emit = app._event_writer.emit

            async def die_after_task_event(event):
                result = await emit(event)
                if loss == "task_event" and event.type is EventType.TASK_FAILED:
                    os._exit(137)
                return result

            patch.setattr(app._event_writer, "emit", die_after_task_event)
            patch.setattr(app._session_finalization, "fail_task", die_after_task)
            patch.setattr(
                app._session_outcomes, "claim_invocation_terminal_decision", die_after_decision
            )

            async def handler(application, task, worker_id):
                async for _ in application.run(
                    RunRequest(
                        agent_name="coding",
                        session_id="linked-failure",
                        task_id=task.id,
                        task_worker_id=worker_id,
                        task_lease_expires_at=task.lease_expires_at,
                        messages=[Message.text("user", "work")],
                    )
                ):
                    pass

            if loss == "decision":
                await run_task_worker(
                    app,
                    app.task_store,
                    handler,
                    worker_id="original",
                    lease_seconds=1,
                    max_tasks=1,
                    reclaim=False,
                )
            else:
                async for _ in app.run(
                    RunRequest(
                        agent_name="coding",
                        session_id="linked-failure",
                        task_id="linked",
                        messages=[Message.text("user", "work")],
                    )
                ):
                    pass
            raise AssertionError("Linked failure did not reach task settlement")

    asyncio.run(scenario())


def _lost_linked_failure(tmp_path, database_url, *, loss="task", marker="current"):
    _database(tmp_path, database_url)
    worker = subprocess.run(
        [
            sys.executable,
            "-c",
            f"from pathlib import Path; from {__name__} import _crash_linked_failure; import sys; _crash_linked_failure(Path(sys.argv[1]))",
            str(tmp_path),
        ],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(item for item in sys.path if item),
            "CAYU_LINKED_FAILURE_LOSS": loss,
            "CAYU_LINKED_FAILURE_MARKER": marker,
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert worker.returncode == 137, (worker.stdout, worker.stderr)


def _crash_after_binding_disposal(root):
    """Recover the lost failure, then die after its allocation is disposed and retired but
    before the pending finalization marker is cleared."""

    import cayu.runtime._recovery_ownership as recovery_ownership

    recovery_ownership._INCOMPLETE_RECOVERY_CLAIM_LEASE = timedelta(seconds=1)

    async def scenario():
        with pytest.MonkeyPatch.context() as patch:
            app, _store, _provider = _application(root, patch)
            clear = app._environment_lifecycle.clear_completion_finalization

            async def die_after_disposal(**kwargs):
                if (root / "disposed").exists():
                    os._exit(137)
                return await clear(**kwargs)

            patch.setattr(
                app._environment_lifecycle, "clear_completion_finalization", die_after_disposal
            )
            await app.reconcile_orphaned_sessions()
            raise AssertionError("Binding cleanup did not reach its marker")

    asyncio.run(scenario())


async def _recovery_claim_expired(store):
    from cayu.sessions.recovery import _incomplete_recovery_claim_from_checkpoint

    async with asyncio.timeout(10):
        while True:
            claim = _incomplete_recovery_claim_from_checkpoint(
                await store.load_checkpoint("linked-failure")
            )
            if claim is None or claim[1] <= datetime.now(UTC):
                return
            await asyncio.sleep(0.05)


def _lost_binding_cleanup(tmp_path):
    worker = subprocess.run(
        [
            sys.executable,
            "-c",
            f"from pathlib import Path; from {__name__} import _crash_after_binding_disposal; import sys; _crash_after_binding_disposal(Path(sys.argv[1]))",
            str(tmp_path),
        ],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(item for item in sys.path if item)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert worker.returncode == 137, (worker.stdout, worker.stderr)


@pytest.mark.parametrize("loss", ["decision", "task"])
def test_linked_failure_written_by_v0_19_1_settles_after_upgrade(tmp_path, database_url, loss):
    _lost_linked_failure(tmp_path, database_url, loss=loss, marker="v0.19.1")

    async def scenario():
        with pytest.MonkeyPatch.context() as patch:
            app, store, provider = _application(tmp_path, patch)
            try:
                if loss == "task":
                    await app.reconcile_orphaned_sessions()
                else:
                    # Only the elected decision survived: the replacement worker's
                    # expired-task handoff rebuilds the task failure from it.
                    async def no_handler(*_args):
                        raise AssertionError("Failure recovery must not redispatch task work")

                    async with asyncio.timeout(10):
                        while (
                            await app.task_store.load_expired_interrupted_task_handoff_candidate(
                                "linked"
                            )
                            is None
                        ):
                            await asyncio.sleep(0.02)
                    failures = []
                    recover_task = app._recover_expired_attached_task_session

                    async def observe_task_recovery(*args):
                        try:
                            return await recover_task(*args)
                        except Exception as exc:
                            failures.append(exc)
                            raise

                    patch.setattr(
                        app, "_recover_expired_attached_task_session", observe_task_recovery
                    )
                    await run_task_worker_until_idle(
                        app, app.task_store, no_handler, worker_id="replacement", reclaim=False
                    )
                    if failures:
                        raise failures[0]
                assert (await store.load("linked-failure")).status is SessionStatus.FAILED
                task = await app.task_store.load_task("linked")
                assert task.status is TaskStatus.FAILED
                marker = task.error["runtime_task_failure"]
                assert "session_terminal_event_id" not in marker
                events = await store.load_events("linked-failure")
                terminal = [event for event in events if event.type is EventType.SESSION_FAILED]
                assert [event.id for event in terminal] == [
                    marker["failure_id"] + ":session_failed"
                ]
                assert not provider.requests
                # A second pass finds nothing left to settle instead of conflicting evidence.
                await app.reconcile_orphaned_sessions()
                assert await store.load_events("linked-failure") == events
            finally:
                await app.aclose()
                await store.close()
                await app.task_store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "entrance",
    ["sweep", "sweep_after_disposal_crash", "public", "public_cleanup_failure", "task_handoff"],
)
def test_linked_failure_recovery_disposes_saved_binding(tmp_path, database_url, entrance):
    _lost_linked_failure(
        tmp_path, database_url, loss="decision" if entrance == "task_handoff" else "task"
    )
    if entrance == "sweep_after_disposal_crash":
        _lost_binding_cleanup(tmp_path)

    async def scenario():
        with pytest.MonkeyPatch.context() as patch:
            app, store, provider = _application(tmp_path, patch)
            hook_calls = []
            run_hooks = app._terminal_event_publication._run_runtime_hooks

            async def observe_hooks(**kwargs):
                hook_calls.append(True)
                async for event in run_hooks(**kwargs):
                    yield event

            patch.setattr(app._terminal_event_publication, "_run_runtime_hooks", observe_hooks)
            try:
                if entrance.startswith("sweep"):
                    if entrance == "sweep_after_disposal_crash":
                        await _recovery_claim_expired(store)
                    await app.reconcile_orphaned_sessions()
                elif entrance.startswith("public"):
                    from cayu.sessions import IncompleteSessionRecoveryRequest

                    if entrance == "public_cleanup_failure":
                        from cayu.sessions._completion_finalization import (
                            pending_completion_finalization_from_checkpoint,
                        )

                        async def failed_cleanup(**_kwargs):
                            raise OSError("retained binding cleanup unavailable")

                        with pytest.MonkeyPatch.context() as cleanup_fault:
                            cleanup_fault.setattr(
                                app._environment_lifecycle, "resolve_factory", failed_cleanup
                            )
                            with pytest.raises(
                                OSError, match="retained binding cleanup unavailable"
                            ):
                                await app.recover_incomplete_session(
                                    IncompleteSessionRecoveryRequest(
                                        session_id="linked-failure",
                                        inactive_for_seconds=0,
                                    )
                                )
                        assert (await store.load("linked-failure")).status is SessionStatus.FAILED
                        assert not (tmp_path / "disposed").exists()
                        assert (
                            pending_completion_finalization_from_checkpoint(
                                await store.load_checkpoint("linked-failure")
                            )
                            is not None
                        )
                        assert (
                            sum(
                                e.type is EventType.SESSION_FAILED
                                for e in await store.load_events("linked-failure")
                            )
                            == 1
                        )
                    await app.recover_incomplete_session(
                        IncompleteSessionRecoveryRequest(
                            session_id="linked-failure",
                            inactive_for_seconds=0,
                        )
                    )
                else:

                    async def no_handler(*_args):
                        raise AssertionError("Failure recovery must not redispatch task work")

                    async with asyncio.timeout(10):
                        while (
                            await app.task_store.load_expired_interrupted_task_handoff_candidate(
                                "linked"
                            )
                            is None
                        ):
                            await asyncio.sleep(0.02)
                    failures = []
                    recover_task = app._recover_expired_attached_task_session

                    async def observe_task_recovery(*args):
                        try:
                            return await recover_task(*args)
                        except Exception as exc:
                            failures.append(exc)
                            raise

                    patch.setattr(
                        app, "_recover_expired_attached_task_session", observe_task_recovery
                    )
                    await run_task_worker_until_idle(
                        app, app.task_store, no_handler, worker_id="replacement", reclaim=False
                    )
                    if failures:
                        raise failures[0]
                assert (await store.load("linked-failure")).status is SessionStatus.FAILED
                assert (tmp_path / "disposed").exists()
                from cayu.sessions._completion_finalization import (
                    pending_completion_finalization_from_checkpoint as pending_marker,
                )

                assert pending_marker(await store.load_checkpoint("linked-failure")) is None
                if entrance != "sweep_after_disposal_crash":
                    # The crashed recovery already published through the runtime hooks.
                    assert hook_calls
                assert (await app.task_store.load_task("linked")).status is TaskStatus.FAILED
                events = await store.load_events("linked-failure")
                assert sum(e.type is EventType.SESSION_FAILED for e in events) == 1
                await app.reconcile_orphaned_sessions()
                assert await store.load_events("linked-failure") == events
                if entrance == "task_handoff":
                    assert not provider.requests
                    return
                from cayu.runtime import _session_task_lifecycle as task_lifecycle
                from cayu.sessions._invocation_lifecycle import (
                    INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
                )
                from cayu.tasks.records import TaskClaimLost

                native_lineage = task_lifecycle.require_invocation_rebind_lineage
                attempts = []

                def reject_missing_lineage(checkpoint, **kwargs):
                    attempts.append(True)
                    damaged = dict(checkpoint)
                    damaged.pop(INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY, None)
                    return native_lineage(damaged, **kwargs)

                with pytest.MonkeyPatch.context() as fault:
                    fault.setattr(
                        task_lifecycle, "require_invocation_rebind_lineage", reject_missing_lineage
                    )
                    with pytest.raises(TaskClaimLost):
                        [
                            event
                            async for event in app.resume(
                                ResumeRequest(
                                    session_id="linked-failure",
                                    messages=[Message.text("user", "retry failure")],
                                )
                            )
                        ]
                assert attempts == [True]
                assert await store.load_events("linked-failure") == events
                assert not provider.requests
                replay = [
                    event
                    async for event in app.resume(
                        ResumeRequest(
                            session_id="linked-failure",
                            messages=[Message.text("user", "retry failure")],
                        )
                    )
                ]
                assert any(event.type is EventType.SESSION_FAILED for event in replay)
                assert await store.load_events("linked-failure") == events
                assert not provider.requests
            finally:
                await app.aclose()
                await store.close()
                await app.task_store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "variant", ["runtime_changed", "missing_environment", "missing_environment_after_task_event"]
)
def test_elected_linked_failure_keeps_original_identity_after_deploy(
    tmp_path, database_url, variant
):
    _lost_linked_failure(
        tmp_path,
        database_url,
        loss="task_event" if variant.endswith("after_task_event") else "task",
    )
    missing_environment = variant.startswith("missing_environment")

    async def scenario():
        with pytest.MonkeyPatch.context() as patch:
            import cayu.runtime._execution_profile_continuation as continuation
            from cayu.runtime.execution_profiles import ExecutionProfileMismatchError

            if variant == "runtime_changed":
                patch.setattr(continuation, "_runtime_version", lambda: "999.0.0")
            app, store, provider = _application(
                tmp_path, patch, register_environment=not missing_environment
            )

            async def no_changed_runtime_hooks(**_kwargs):
                raise AssertionError("Elected failure settlement must not execute changed hooks")
                yield

            patch.setattr(
                app._terminal_event_publication, "_run_runtime_hooks", no_changed_runtime_hooks
            )
            try:
                if variant.endswith("after_task_event"):
                    before = await store.load_events("linked-failure")
                    assert sum(event.type is EventType.TASK_FAILED for event in before) == 1
                    assert not any(event.type is EventType.SESSION_FAILED for event in before)
                    assert (await store.load("linked-failure")).status is SessionStatus.RUNNING
                await app.reconcile_orphaned_sessions()
                assert (await store.load("linked-failure")).status is SessionStatus.FAILED
                marker = (await app.task_store.load_task("linked")).error["runtime_task_failure"]
                events = await store.load_events("linked-failure")
                terminal = [event for event in events if event.type is EventType.SESSION_FAILED]
                assert len(terminal) == 1
                assert terminal[0].id == marker["failure_id"] + ":session_failed"
                assert terminal[0].payload["runtime_task_failure_id"] == marker["failure_id"]
                task_events = [event for event in events if event.type is EventType.TASK_FAILED]
                assert len(task_events) == 1
                assert (
                    task_events[0].environment_name
                    == (await store.load("linked-failure")).environment_name
                )
                assert (tmp_path / "disposed").exists() == (not missing_environment)
                assert not provider.requests
                if variant == "runtime_changed":
                    # The ordinary public continuation entrance still requires its
                    # current execution profile; equal durable input grants no override.
                    with pytest.raises(ExecutionProfileMismatchError):
                        [
                            event
                            async for event in app.resume(
                                ResumeRequest(
                                    session_id="linked-failure",
                                    messages=[Message.text("user", "retry failure")],
                                )
                            )
                        ]
                    assert await store.load_events("linked-failure") == events
            finally:
                await app.aclose()
                await store.close()
                await app.task_store.close()
        if missing_environment:
            with pytest.MonkeyPatch.context() as patch:
                app, store, provider = _application(tmp_path, patch)
                try:
                    await app.reconcile_orphaned_sessions()
                    assert (tmp_path / "disposed").exists()
                    assert (
                        sum(
                            event.type is EventType.SESSION_FAILED
                            for event in await store.load_events("linked-failure")
                        )
                        == 1
                    )
                    assert not provider.requests
                finally:
                    await app.aclose()
                    await store.close()
                    await app.task_store.close()

    asyncio.run(scenario())
