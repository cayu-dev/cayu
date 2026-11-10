"""An expired-owner scan must not fence a replacement's live session."""

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
@pytest.mark.parametrize("stale_scan", [False, True])
def test_expired_scan_preserves_live_replacement(backend, stale_scan, tmp_path):
    async def scenario():
        now = [datetime.now(UTC)]
        scanned = asyncio.Event()
        continue_scan = asyncio.Event()
        provider_entered = asyncio.Event()
        release_provider = asyncio.Event()
        base = InMemoryTaskStore if backend == "memory" else SQLiteTaskStore

        class ScanBarrierStore(base):
            supports_interrupted_task_handoffs = True
            verified_work_mutations_are_cancellation_quiescent = True

            async def list_expired_interrupted_task_handoff_candidates(self, **kwargs):
                candidates = await super().list_expired_interrupted_task_handoff_candidates(
                    **kwargs
                )
                scanned.set()
                await continue_scan.wait()
                return candidates

        class WaitingProvider(ScriptedModelProvider):
            @property
            def execution_profile_identity(self):
                return versioned_test_provider_identity(self)

            async def stream(self, request):
                provider_entered.set()
                await release_provider.wait()
                yield ModelStreamEvent.text_delta("done")
                yield ModelStreamEvent.completed()

        store = ScanBarrierStore(
            *(() if backend == "memory" else (tmp_path / "tasks.sqlite",)),
            ownership_clock=lambda: now[0],
        )
        sessions = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "sessions.sqlite")
        )
        provider = WaitingProvider([])
        # Separate apps model a scanner and a continuation in different workers:
        # the scanner has no process-local active-session record for its peer.
        apps = [
            CayuApp(task_store=store, session_store=sessions, enable_logging=False)
            for _ in range(2)
        ]
        for app in apps:
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="worker-agent", model="scripted-model"))
        scanner_app, replacement_app = apps
        scanner = replacement = None

        async def scan():
            return await _recover_expired_interrupted_task_handoffs(
                scanner_app,
                store,
                worker_id="scanner",
                after=None,
                limit=10,
                stop=None,
            )

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
            await interrupt_and_release_test_invocation(sessions, "session")
            claim = await store.claim_task("original", lease_seconds=1)
            assert claim is not None and claim.lease_expires_at is not None
            await store.attach_task(
                claim.id,
                session_id="session",
                session_invocation=await stored_session_invocation(sessions, "session"),
                worker_id="original",
                claim_generation=claim.claim_generation,
            )
            now[0] = claim.lease_expires_at + timedelta(seconds=1)
            if stale_scan:
                scanner = asyncio.create_task(scan())
                await asyncio.wait_for(scanned.wait(), 10)

            # The original callback finishes its delayed disposition. This real
            # worker path upgrades its expired live handoff to exact recovery;
            # no second elected scanner is needed to produce the race.
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
                raise AssertionError("The task is a continuation, not fresh work")

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
            assert active is not None and active.status is SessionStatus.RUNNING
            presence = await sessions.inspect_session_execution("session")
            assert presence.state == "executing"
            replacement_claim = await store.load_task("job")
            assert replacement_claim.worker_id == "replacement"
            assert replacement_claim.claim_generation != claim.claim_generation
            assert replacement_claim.lease_expires_at > now[0]

            if not stale_scan:
                # Control: a fresh scan at this point sees no expired candidate.
                scanner = asyncio.create_task(scan())
                await asyncio.wait_for(scanned.wait(), 10)
            continue_scan.set()
            await asyncio.wait_for(scanner, 10)
            current = await sessions.load("session")
            assert (current.status, current.run_epoch) == (active.status, active.run_epoch), (
                "A stale expired-task candidate fenced a replacement's live session",
                current.status,
                current.run_epoch,
                active.run_epoch,
            )
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
