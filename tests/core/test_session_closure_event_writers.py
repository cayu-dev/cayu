"""Specialized event publication respects the closure's native inventory owner."""

import asyncio
from uuid import uuid4

import pytest
from tests.core.test_active_invocation_execution_profiles import BlockingCompletionHook
from tests.core.test_execution_profile_store_conformance import _profile, _rejection_event
from tests.core.test_tool_effect_state import _event, _intent, _receipt, _terminal
from tests.core.test_tool_effect_store_conformance import _stores

from cayu import (
    AgentSpec,
    CayuApp,
    Message,
    ModelStreamEvent,
    ResumeRequest,
    RunRequest,
    ScriptedModelProvider,
)
from cayu.events import Event, EventType
from cayu.runtime._checkpoint_store import _RuntimeCheckpointSessionStore
from cayu.runtime._tool_effect_conflicts import ToolEffectConflictAudit
from cayu.runtime._tool_effect_state import ToolEffectStateOwner
from cayu.runtime.execution_profiles import ExecutionProfileMismatchError
from cayu.runtime.session_closure import (
    SessionClosureDisposition,
    SessionClosureRecord,
)
from cayu.sessions.base import SessionIdentity, SessionStatus
from cayu.tools import ReadFileTool, WriteFileTool


class _PublicationBarrier:
    store_id = "publication-barrier"

    def __init__(self):
        self.entered, self.release = asyncio.Event(), asyncio.Event()

    async def inspect_session_closure(self, session_id, *, policy):
        return SessionClosureRecord(
            store_id=self.store_id,
            record_class="records",
            disposition=SessionClosureDisposition.OWNED_ELIGIBLE,
        )

    async def export_session_closure(self, session_id, *, policy):
        return {"records": []}

    async def erase_session_closure(self, session_id, *, policy, plan_id):
        self.entered.set()
        await self.release.wait()
        return SessionClosureRecord(
            store_id=self.store_id,
            record_class="records",
            disposition=SessionClosureDisposition.ERASED,
        )

    async def wait(self, closing):
        entered = asyncio.create_task(self.entered.wait())
        try:
            done, _ = await asyncio.wait(
                (entered, closing), timeout=10, return_when=asyncio.FIRST_COMPLETED
            )
            if closing in done:
                raise AssertionError(f"Closure stopped before barrier: {await closing}")
            assert entered in done, "Closure did not reach the publication barrier."
        finally:
            entered.cancel()
            await asyncio.gather(entered, return_exceptions=True)


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_profile_rejection_cannot_publish_into_claimed_closure(backend, tmp_path, request):
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    async def scenario():
        async with _stores(backend, tmp_path / "profile-closure.sqlite", dsn) as open_store:
            store, competitor = open_store(), open_store()
            barrier = _PublicationBarrier()
            provider = ScriptedModelProvider(
                [[ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({})]]
            )
            apps = []
            for index in range(3):
                app = CayuApp(
                    enable_logging=False,
                    session_store=store if index == 0 else competitor,
                    session_closure_stores=(barrier,) if index == 0 else (),
                )
                app.register_provider(provider, default=True)
                app.register_agent(
                    AgentSpec(name="worker", model="scripted-model"),
                    tools=([], [ReadFileTool()], [WriteFileTool()])[index],
                )
                apps.append(app)
            session_id = f"closure-rejection-{uuid4().hex}"
            async for _ in apps[0].run(
                RunRequest(
                    session_id=session_id,
                    agent_name="worker",
                    messages=[Message.text("user", "go")],
                )
            ):
                pass
            with pytest.raises(ExecutionProfileMismatchError):
                async for _ in apps[1].resume(
                    ResumeRequest(session_id=session_id, messages=[Message.text("user", "resume")])
                ):
                    pass
            inspection = await apps[0].inspect_session_closure(session_id)
            assert inspection.complete, inspection
            closing = asyncio.create_task(apps[0].erase_session_closure(session_id))
            try:
                await barrier.wait(closing)
                before = await competitor.load_session_closure_records(
                    session_id, max_records=1000, max_bytes=1_000_000
                )
                # Public resume's earlier checkpoint guard also denies replay attempts.
                with pytest.raises(ValueError, match="owned by"):
                    async for _ in apps[1].resume(
                        ResumeRequest(
                            session_id=session_id, messages=[Message.text("user", "resume")]
                        )
                    ):
                        pass
                with pytest.raises(ValueError, match="owned by"):
                    async for _ in apps[2].resume(
                        ResumeRequest(
                            session_id=session_id, messages=[Message.text("user", "resume")]
                        )
                    ):
                        pass
                assert len(provider.requests) == 1
                assert (
                    await competitor.load_session_closure_records(
                        session_id, max_records=1000, max_bytes=1_000_000
                    )
                    == before
                )
                barrier.release.set()
                assert (await closing).complete
                assert await store.load(session_id) is None
            finally:
                barrier.release.set()
                if not closing.done():
                    closing.cancel()
                await asyncio.gather(closing, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_closure_waits_for_terminal_hook_quiescence(backend, tmp_path, request):
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    async def scenario():
        async with _stores(backend, tmp_path / "hook-closure.sqlite", dsn) as open_store:
            store = open_store()
            hook = BlockingCompletionHook("closure-hook")
            barrier = _PublicationBarrier()
            barrier.release.set()
            app = CayuApp(session_store=store, enable_logging=False)
            closer = CayuApp(
                session_store=open_store(),
                enable_logging=False,
                session_closure_stores=(barrier,),
            )
            provider = ScriptedModelProvider(
                [[ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({})]]
            )
            app.register_provider(provider, default=True)
            app.register_agent(
                AgentSpec(name="worker", model="scripted-model"), runtime_hooks=[hook]
            )
            session_id = f"closure-hook-{uuid4().hex}"

            async def run():
                return [
                    event
                    async for event in app.run(
                        RunRequest(
                            session_id=session_id,
                            agent_name="worker",
                            messages=[Message.text("user", "go")],
                        )
                    )
                ]

            running = asyncio.create_task(run())
            try:
                await asyncio.wait_for(hook.started.wait(), 10)
                session = await store.load(session_id)
                assert session is not None and session.status is SessionStatus.COMPLETED
                before = await store.load_session_closure_records(
                    session_id, max_records=1000, max_bytes=1_000_000
                )
                for operation in (closer.validate_session_closure, closer.erase_session_closure):
                    with pytest.raises(ValueError, match="terminal hooks or trailing cleanup"):
                        await operation(session_id)
                assert not barrier.entered.is_set()
                assert not running.done()
                assert (
                    await store.load_session_closure_records(
                        session_id, max_records=1000, max_bytes=1_000_000
                    )
                    == before
                )
                hook.release.set()
                await running
                assert (await closer.erase_session_closure(session_id)).complete
                assert barrier.entered.is_set()
                assert await store.load(session_id) is None
            finally:
                hook.release.set()
                if not running.done():
                    running.cancel()
                await asyncio.gather(running, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_specialized_writers_preserve_claimed_inventory(backend, tmp_path, request):
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    async def scenario():
        async with _stores(backend, tmp_path / "specialized-closure.sqlite", dsn) as open_store:
            store = open_store()
            intent = await _intent(store, session_id=f"specialized-{uuid4().hex}")
            session_id = intent.session_id
            owner = ToolEffectStateOwner(store)
            audits = []
            for index in range(2):
                current = intent.model_copy(
                    update={"idempotency_key": f"key-{index}", "tool_call_id": f"call-{index}"}
                )
                executing = await owner.begin(current, run_epoch=0)
                unknown = await owner.transition(executing, state="outcome_unknown", run_epoch=0)
                event = _event(current, event_id=f"terminal-{index}")
                selected = await owner.transition(
                    unknown,
                    state="reconciled_completed",
                    run_epoch=0,
                    terminal=_terminal(event, _receipt(current)),
                    events=(event,),
                )
                audits.append(ToolEffectConflictAudit(executing, selected))
            prior_audit = await store.append_tool_effect_conflict(audits[0])
            await store.append_event(
                session_id,
                Event(
                    type="custom.cayu.workflow.attempt",
                    session_id=session_id,
                    workflow_name="maintenance",
                    payload={"attempt_id": "attempt"},
                ),
            )
            step = Event(
                type=EventType.WORKFLOW_STEP_STARTED,
                session_id=session_id,
                workflow_name="maintenance",
                payload={"attempt_id": "attempt", "step_id": "step"},
            )
            assert await store.append_workflow_step_started(
                session_id, step, workflow_name="maintenance", attempt_id="attempt"
            )
            await store.update_status(session_id, SessionStatus.COMPLETED)
            await store.append_event(
                session_id, Event(type=EventType.SESSION_COMPLETED, session_id=session_id)
            )
            barrier = _PublicationBarrier()
            app = CayuApp(
                enable_logging=False, session_store=store, session_closure_stores=(barrier,)
            )
            inspection = await app.inspect_session_closure(session_id)
            assert inspection.complete, inspection
            closing = asyncio.create_task(app.erase_session_closure(session_id))
            try:
                await barrier.wait(closing)
                competitor = _RuntimeCheckpointSessionStore(open_store())
                before = await store.load_session_closure_records(
                    session_id, max_records=1000, max_bytes=1_000_000
                )
                assert await competitor.append_tool_effect_conflict(audits[0]) == prior_audit
                assert not await competitor.append_workflow_step_started(
                    session_id, step, workflow_name="maintenance", attempt_id="attempt"
                )
                with pytest.raises(ValueError, match="owned by"):
                    await competitor.append_tool_effect_conflict(audits[1])
                with pytest.raises(ValueError, match="owned by"):
                    await competitor.append_workflow_step_started(
                        session_id,
                        step.model_copy(update={"id": "late-step"}),
                        workflow_name="maintenance",
                        attempt_id="attempt",
                    )
                history_key = "sha256:" + "1" * 64
                with pytest.raises(ValueError, match="owned by"):
                    await competitor.compare_and_publish_mcp_manifest_checks(
                        session_id,
                        expected_generations={history_key: None},
                        baseline_updates={},
                        events=[
                            Event(
                                type=EventType.MCP_MANIFEST_BLOCKED,
                                session_id=session_id,
                                payload={
                                    "history_key": history_key,
                                    "status": "first_seen",
                                    "outcome": "blocked",
                                    "manifest_identity": "sha256:" + "2" * 64,
                                    "manifest_hash": "sha256:" + "3" * 64,
                                    "source_manifest_hash": "sha256:" + "4" * 64,
                                    "server_hash": "sha256:" + "5" * 64,
                                },
                            )
                        ],
                    )
                assert not (await competitor.load_mcp_manifest_baselines((history_key,))).baselines
                assert (
                    await store.load_session_closure_records(
                        session_id, max_records=1000, max_bytes=1_000_000
                    )
                    == before
                )
                barrier.release.set()
                assert (await closing).complete
            finally:
                barrier.release.set()
                if not closing.done():
                    closing.cancel()
                await asyncio.gather(closing, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_profile_rejection_store_replay_is_read_only_during_closure(backend, tmp_path, request):
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    async def scenario():
        async with _stores(backend, tmp_path / "rejection-replay.sqlite", dsn) as open_store:
            store = open_store()
            session_id = f"profile-replay-{uuid4().hex}"
            expected, candidate = _profile(tool_name="original"), _profile(tool_name="changed")
            await store.create(
                RunRequest(session_id=session_id, agent_name="assistant", messages=[]),
                identity=SessionIdentity(
                    provider_name="fake", model="fake-model", execution_profile=expected
                ),
            )
            await store.update_status(session_id, SessionStatus.COMPLETED)
            await store.append_event(
                session_id, Event(type=EventType.SESSION_COMPLETED, session_id=session_id)
            )
            event = _rejection_event(session_id=session_id, expected=expected, candidate=candidate)
            arguments = dict(
                expected_statuses={SessionStatus.COMPLETED},
                expected_run_epoch=0,
                expected_profile=expected,
                candidate_profile=candidate,
            )
            original = await store.reject_execution_profile_resume(
                session_id, event=event, **arguments
            )
            assert not original.replayed
            barrier = _PublicationBarrier()
            app = CayuApp(
                enable_logging=False, session_store=store, session_closure_stores=(barrier,)
            )
            closing = asyncio.create_task(app.erase_session_closure(session_id))
            try:
                await barrier.wait(closing)
                competitor = open_store()
                before = await store.load_session_closure_records(
                    session_id, max_records=1000, max_bytes=1_000_000
                )
                replay = await competitor.reject_execution_profile_resume(
                    session_id, event=event, **arguments
                )
                assert replay.replayed and replay.event == original.event
                with pytest.raises(ValueError, match="owned by"):
                    await competitor.reject_execution_profile_resume(
                        session_id,
                        event=event.model_copy(update={"id": "new-rejection"}),
                        **arguments,
                    )
                assert (
                    await store.load_session_closure_records(
                        session_id, max_records=1000, max_bytes=1_000_000
                    )
                    == before
                )
                barrier.release.set()
                assert (await closing).complete
            finally:
                barrier.release.set()
                if not closing.done():
                    closing.cancel()
                await asyncio.gather(closing, return_exceptions=True)

    asyncio.run(scenario())
