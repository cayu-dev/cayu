"""Expired task authority fences only the run observed before claim validation."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from tests.core._execution_profile_fixtures import (
    admit_test_invocation,
    interrupt_and_release_test_invocation,
    profiled_session_identity,
    runtime_interaction_started_event,
    versioned_test_provider_identity,
)
from tests.core.task_invocation_fixtures import stored_session_invocation

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.evals.testing import ScriptedModelProvider
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.runtime._task_lease_authority import TaskLeaseAuthority
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import ResumeRequest, RunRequest, run_request_with_task_invocation
from cayu.storage.sqlite import SQLiteSessionStore, SQLiteTaskStore
from cayu.tasks.creation import TaskCreate, TaskInvocationSnapshot
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.records import TaskStatus
from cayu.tasks.worker import (
    _handoff_interrupted_session,
    _recover_expired_interrupted_task_handoffs,
    run_task_worker,
)
from cayu.tools.exposure import ToolCapabilityCeiling


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("mode", ["expired", "cancelled", "task_read", "reservation"])
def test_expired_owner_fences_only_observed_run(backend, mode, tmp_path):
    async def scenario():
        now = [datetime.now(UTC)]
        paused, continue_scan = asyncio.Event(), asyncio.Event()
        provider_entered, release_provider = asyncio.Event(), asyncio.Event()
        task_base = InMemoryTaskStore if backend == "memory" else SQLiteTaskStore
        session_base = InMemorySessionStore if backend == "memory" else SQLiteSessionStore

        class TaskStore(task_base):
            supports_interrupted_task_handoffs = True
            verified_work_mutations_are_cancellation_quiescent = True
            block_read = False

            async def load_task(self, task_id):
                task = await super().load_task(task_id)
                if self.block_read:
                    self.block_read = False
                    paused.set()
                    await continue_scan.wait()
                return task

        class SessionStore(session_base):
            invocation_lifecycle_command_version = 1
            terminal_interaction_publication_version = 1
            durable_model_terminalization_version = 1
            block_reservation = False

            async def reserve_stalled_run_recovery(self, *args, **kwargs):
                if self.block_reservation:
                    self.block_reservation = False
                    paused.set()
                    await continue_scan.wait()
                return await super().reserve_stalled_run_recovery(*args, **kwargs)

        class WaitingProvider(ScriptedModelProvider):
            @property
            def execution_profile_identity(self):
                return versioned_test_provider_identity(self)

            async def stream(self, request):
                provider_entered.set()
                await release_provider.wait()
                yield ModelStreamEvent.text_delta("done")
                yield ModelStreamEvent.completed()

        store = TaskStore(
            *(() if backend == "memory" else (tmp_path / "tasks.sqlite",)),
            ownership_clock=lambda: now[0],
        )
        sessions = SessionStore(*(() if backend == "memory" else (tmp_path / "sessions.sqlite",)))
        provider = WaitingProvider([])
        apps = [
            CayuApp(task_store=store, session_store=sessions, enable_logging=False)
            for _ in range(2)
        ]
        for app in apps:
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="worker-agent", model="scripted-model"))
        scanner_app, replacement_app = apps
        scanner = replacement = None
        try:
            created = await store.create_task(TaskCreate(task_id="job", type="job"))
            await sessions.create(
                run_request_with_task_invocation(
                    RunRequest(
                        agent_name="worker-agent",
                        session_id="session",
                        task_id="job",
                        messages=[Message.text("user", "pause")],
                        tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
                    ),
                    TaskInvocationSnapshot(
                        id=created.id, session_id=None, invocation=created.invocation
                    ),
                ),
                identity=profiled_session_identity(
                    max_steps=None,
                    provider_name=provider.name,
                    model="scripted-model",
                    agent_name="worker-agent",
                    app=scanner_app,
                ),
            )
            await admit_test_invocation(
                sessions,
                "session",
                interaction_started_event=runtime_interaction_started_event(
                    scanner_app,
                    session_id="session",
                    interaction_id="original-interaction",
                    agent_name="worker-agent",
                ),
                interaction_source_messages=(Message.text("user", "pause"),),
            )
            claim = await store.claim_task("original", lease_seconds=1)
            assert claim is not None and claim.lease_expires_at is not None
            await store.attach_task(
                claim.id,
                session_id="session",
                session_invocation=await stored_session_invocation(sessions, "session"),
                worker_id="original",
                claim_generation=claim.claim_generation,
            )
            # Presence can remain live independently of an expired task lease.
            owner = await sessions._claim_session_execution(
                "session",
                token="original-token",
                owner_id="original",
                owner_kind="task_worker",
                owner_label=None,
                lease_seconds=300,
            )
            original = await sessions.load("session")
            assert original.status is SessionStatus.RUNNING
            assert (await sessions.inspect_session_execution("session")).state == "executing"
            if mode == "cancelled":
                await store.cancel_task("job", error={"code": "operator_cancelled"})
            now[0] = claim.lease_expires_at + timedelta(seconds=1)
            store.block_read = mode == "task_read"
            sessions.block_reservation = mode == "reservation"
            scanner = asyncio.create_task(
                _recover_expired_interrupted_task_handoffs(
                    scanner_app, store, worker_id="scanner", after=None, limit=10, stop=None
                )
            )
            if mode in {"expired", "cancelled"}:
                page = await asyncio.wait_for(scanner, 10)
                assert page.recovered == 1
                assert not page.failed and not page.manual_recovery_required
                current = await sessions.load("session")
                assert current.status is SessionStatus.INTERRUPTED
                assert current.run_epoch > original.run_epoch
                task = await store.load_task("job")
                assert task.worker_id is None and task.lease_expires_at is None
                assert task.status is (
                    TaskStatus.CANCELLED if mode == "cancelled" else TaskStatus.RUNNING
                )
                return

            await asyncio.wait_for(paused.wait(), 10)
            # Finish the predecessor and start a genuine continuation while the
            # scanner holds either an old task read or a pending reservation.
            await sessions._release_session_execution(owner)
            await interrupt_and_release_test_invocation(sessions, "session")
            await _handoff_interrupted_session(
                scanner_app,
                store,
                TaskLeaseAuthority(
                    claim.lease_expires_at,
                    task_id=claim.id,
                    worker_id="original",
                    claim_generation=claim.claim_generation,
                ),
            )

            async def fresh(*args):
                raise AssertionError("Expected a continuation")

            async def resume(app, task, worker_id):
                async for _ in app.resume(
                    ResumeRequest(
                        session_id=task.session_id,
                        task_worker_id=worker_id,
                        task_handoff_id=task.interrupted_handoff_id,
                        messages=[Message.text("user", "continue")],
                    )
                ):
                    pass

            replacement = asyncio.create_task(
                run_task_worker(
                    replacement_app,
                    store,
                    fresh,
                    worker_id="replacement",
                    max_tasks=1,
                    reclaim=False,
                    recover_interrupted_handoffs=False,
                    recovered_interrupted_task_handler=resume,
                )
            )
            await asyncio.wait_for(provider_entered.wait(), 10)
            active = await sessions.load("session")
            assert active.status is SessionStatus.RUNNING
            assert active.run_epoch > original.run_epoch
            assert (await sessions.inspect_session_execution("session")).state == "executing"
            replacement_claim = await store.load_task("job")
            assert replacement_claim.worker_id == "replacement"
            assert replacement_claim.lease_expires_at > now[0]
            continue_scan.set()
            page = await asyncio.wait_for(scanner, 10)
            assert page.recovered == 0
            current = await sessions.load("session")
            assert (current.status, current.run_epoch) == (active.status, active.run_epoch)
            assert await store.load_task("job") == replacement_claim
            release_provider.set()
            assert await asyncio.wait_for(replacement, 10) == 1
            assert (await store.load_task("job")).status is TaskStatus.COMPLETED
        finally:
            continue_scan.set()
            release_provider.set()
            await asyncio.gather(
                *(task for task in (scanner, replacement) if task is not None),
                return_exceptions=True,
            )
            for app in apps:
                await app.aclose()
            if backend == "sqlite":
                await store.close()
                await sessions.close()

    asyncio.run(scenario())
