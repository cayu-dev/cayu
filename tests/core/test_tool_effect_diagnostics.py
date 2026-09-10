from __future__ import annotations

import asyncio

import pytest
from tests.core.test_tool_effect_state import _intent

from cayu.core import EventType
from cayu.runtime import InMemorySessionStore
from cayu.runtime._diagnostics import MAX_DIAGNOSTIC_UTF8_BYTES
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._tool_effect_diagnostics import persist_cleanup_diagnostics
from cayu.runtime._tool_effect_state import ToolEffectStateOwner
from cayu.runtime.budgets import InMemoryBudgetStore
from cayu.runtime.event_sinks import EventSink
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.vaults import SecretRedactor


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("ack_loss", [False, True, "concurrent"])
def test_cleanup_diagnostic_replay_preserves_unknown_effect(tmp_path, backend, ack_loss):
    async def scenario():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "effects.db")
        )
        try:
            intent = await _intent(store)
            owner = ToolEffectStateOwner(store)
            executing = await owner.begin(intent, run_epoch=intent.source_run_epoch)
            unknown = await owner.transition(
                executing, state="outcome_unknown", run_epoch=intent.source_run_epoch
            )
            writer = RuntimeEventWriter(
                session_store=store, budget_store=InMemoryBudgetStore(), event_sinks=[]
            )
            original_append = store.append_event
            failed = False

            async def append(session_id, event):
                nonlocal failed
                await original_append(session_id, event)
                if (
                    ack_loss is True
                    and not failed
                    and event.type == EventType.TOOL_EFFECT_CLEANUP_OBSERVED
                ):
                    failed = True
                    raise RuntimeError("cleanup acknowledgement lost")

            store.append_event = append
            arguments = dict(
                store=store,
                writer=writer,
                records=(unknown,),
                redactor=SecretRedactor("canary"),
                attributed=True,
            )
            artifacts = [
                {"type": "cayu.runner_cleanup.v1", "status": "timeout", "detail": "canary"}
            ]
            if ack_loss == "concurrent":
                original_query = store.query_events
                admitted = asyncio.Event()
                initial_queries = 0

                async def query(query):
                    nonlocal initial_queries
                    rows = await original_query(query)
                    if (
                        query.event_id
                        and query.event_id.startswith("tool-effect-cleanup:")
                        and initial_queries < 2
                    ):
                        initial_queries += 1
                        if initial_queries == 2:
                            admitted.set()
                        await admitted.wait()
                    return rows

                store.query_events = query
                first, repeated = await asyncio.gather(
                    persist_cleanup_diagnostics(**arguments, artifacts=artifacts),
                    persist_cleanup_diagnostics(**arguments, artifacts=artifacts),
                )
            else:
                first = await persist_cleanup_diagnostics(**arguments, artifacts=artifacts)
                repeated = await persist_cleanup_diagnostics(**arguments, artifacts=artifacts)
            assert first == repeated
            assert first is not None
            assert "canary" not in repr(first.payload)
            later = await persist_cleanup_diagnostics(**arguments, artifacts=[{"status": "failed"}])
            assert later is not None and later.id != first.id
            assert await owner.load(intent) == unknown
            events = await store.load_events(intent.session_id)
            assert sum(e.type == EventType.TOOL_EFFECT_OUTCOME_UNKNOWN for e in events) == 1
            assert sum(e.type == EventType.TOOL_EFFECT_CLEANUP_OBSERVED for e in events) == 2
            assert not any(
                e.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
                for e in events
            )
        finally:
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("phase", ["before_commit", "after_commit", "readback"])
def test_cleanup_publication_preserves_real_task_cancellation(tmp_path, backend, phase):
    async def scenario():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "cancel.db")
        )
        task = None
        try:
            intent = await _intent(store)
            owner = ToolEffectStateOwner(store)
            executing = await owner.begin(intent, run_epoch=intent.source_run_epoch)
            unknown = await owner.transition(
                executing, state="outcome_unknown", run_epoch=intent.source_run_epoch
            )
            writer = RuntimeEventWriter(
                session_store=store, budget_store=InMemoryBudgetStore(), event_sinks=[]
            )
            original_append, original_query = store.append_event, store.query_events
            reached = asyncio.Event()
            attempted = False

            async def append(session_id, event):
                nonlocal attempted
                attempted = True
                if phase == "readback":
                    await original_append(session_id, event)
                    raise OSError("acknowledgement lost before cancellation")
                if phase == "after_commit":
                    await original_append(session_id, event)
                reached.set()
                await asyncio.Event().wait()

            async def query(query):
                if phase == "readback" and attempted:
                    reached.set()
                    await asyncio.Event().wait()
                return await original_query(query)

            store.append_event, store.query_events = append, query
            arguments = dict(
                store=store,
                writer=writer,
                records=(unknown,),
                artifacts=[{"status": "timeout"}],
                redactor=SecretRedactor(),
                attributed=True,
            )
            task = asyncio.create_task(persist_cleanup_diagnostics(**arguments))
            await asyncio.wait_for(reached.wait(), timeout=5)
            assert task.cancelling() == 0
            task.cancel()
            assert task.cancelling() == 1
            handled = False
            try:
                await task
            except asyncio.CancelledError:
                handled = True
            assert handled and task.cancelled() and task.cancelling() == 1
            store.append_event, store.query_events = original_append, original_query
            assert await owner.load(intent) == unknown
            events = await store.load_events(intent.session_id)
            assert sum(e.type == EventType.TOOL_EFFECT_CLEANUP_OBSERVED for e in events) == (
                0 if phase == "before_commit" else 1
            )
            replay = await persist_cleanup_diagnostics(**arguments)
            assert replay is not None
            events = await store.load_events(intent.session_id)
            assert sum(e.type == EventType.TOOL_EFFECT_CLEANUP_OBSERVED for e in events) == 1
            assert await owner.load(intent) == unknown
        finally:
            if task is not None:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("delivery", ["sink_failure", "deferred"])
def test_cleanup_diagnostic_uses_existing_delivery_owner(tmp_path, backend, delivery):
    class Recorder(EventSink):
        def __init__(self):
            self.events = []

        async def emit(self, event):
            self.events.append(event)

    class FailingSink(EventSink):
        async def emit(self, event):
            raise RuntimeError("sink unavailable")

    async def scenario():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "delivery.db")
        )
        try:
            intent = await _intent(store)
            owner = ToolEffectStateOwner(store)
            executing = await owner.begin(intent, run_epoch=intent.source_run_epoch)
            unknown = await owner.transition(
                executing, state="outcome_unknown", run_epoch=intent.source_run_epoch
            )
            recorder = Recorder()
            writer = RuntimeEventWriter(
                session_store=store,
                budget_store=InMemoryBudgetStore(),
                event_sinks=[FailingSink(), recorder] if delivery == "sink_failure" else [recorder],
            )
            event = await persist_cleanup_diagnostics(
                store=store,
                writer=writer,
                records=(unknown,),
                artifacts=[{"status": "timeout"}],
                redactor=SecretRedactor(),
                attributed=True,
            )
            assert event is not None
            original_claim = store.claim_persisted_event_side_effect
            if delivery == "deferred":

                async def decline(**kwargs):
                    return None

                store.claim_persisted_event_side_effect = decline
            await writer.fan_out_persisted([event])
            if delivery == "deferred":
                assert recorder.events == []
                store.claim_persisted_event_side_effect = original_claim
                await writer.recover_persisted_side_effects()
            await writer.fan_out_persisted([event])
            observed = [
                e for e in recorder.events if e.type == EventType.TOOL_EFFECT_CLEANUP_OBSERVED
            ]
            assert len(observed) == 1
            assert observed[0].payload["artifacts"] == [{"status": "timeout"}]
            assert await owner.load(intent) == unknown
            persisted = await store.load_events(intent.session_id)
            if delivery == "sink_failure":
                assert sum(e.type == EventType.RUNTIME_SINK_FAILED for e in persisted) == 1
        finally:
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("malformed", ["item", "nested", "container"])
def test_cleanup_diagnostic_rejects_hostile_artifacts_without_output(
    malformed, capsys, caplog, recwarn
):
    class Hostile:
        def __repr__(self):
            return "secret-diagnostic-canary"

        __str__ = __repr__

    async def scenario():
        store = InMemorySessionStore()
        intent = await _intent(store)
        owner = ToolEffectStateOwner(store)
        executing = await owner.begin(intent, run_epoch=intent.source_run_epoch)
        writer = RuntimeEventWriter(
            session_store=store, budget_store=InMemoryBudgetStore(), event_sinks=[]
        )
        before = await store.load_events(intent.session_id)
        artifacts = (
            [Hostile()]
            if malformed == "item"
            else [{"detail": Hostile(), "sibling": "secret-diagnostic-canary"}]
            if malformed == "nested"
            else Hostile()
        )
        with pytest.raises((TypeError, ValueError)) as failure:
            await persist_cleanup_diagnostics(
                store=store,
                writer=writer,
                records=(executing,),
                artifacts=artifacts,
                redactor=SecretRedactor("secret-diagnostic-canary"),
                attributed=True,
            )
        assert "secret-diagnostic-canary" not in str(failure.value)
        assert "secret-diagnostic-canary" not in repr(failure.value)
        assert await store.load_events(intent.session_id) == before
        assert await owner.load(intent) == executing

    asyncio.run(scenario())
    output = capsys.readouterr()
    assert "secret-diagnostic-canary" not in output.out + output.err + caplog.text
    assert not recwarn.list


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("committed", [False, True])
def test_cleanup_append_and_readback_failures_preserve_evidence(tmp_path, backend, committed):
    async def scenario():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "failure.db")
        )
        try:
            intent = await _intent(store)
            owner = ToolEffectStateOwner(store)
            executing = await owner.begin(intent, run_epoch=intent.source_run_epoch)
            unknown = await owner.transition(
                executing, state="outcome_unknown", run_epoch=intent.source_run_epoch
            )
            writer = RuntimeEventWriter(
                session_store=store, budget_store=InMemoryBudgetStore(), event_sinks=[]
            )
            append_error = OSError("append failed")
            readback_errors = [OSError("first readback failed"), OSError("second readback failed")]
            original_append, original_query = store.append_event, store.query_events
            attempted = False
            readbacks = 0

            async def append(session_id, event):
                nonlocal attempted
                if committed:
                    await original_append(session_id, event)
                attempted = True
                raise append_error

            async def query(query):
                nonlocal readbacks
                if attempted:
                    error = readback_errors[readbacks]
                    readbacks += 1
                    raise error
                return await original_query(query)

            store.append_event, store.query_events = append, query
            arguments = dict(
                store=store,
                writer=writer,
                records=(unknown,),
                artifacts=[{"status": "timeout"}],
                redactor=SecretRedactor(),
                attributed=True,
            )
            with pytest.raises(ExceptionGroup) as failure:
                await persist_cleanup_diagnostics(**arguments)
            assert failure.value.exceptions == (append_error, readback_errors[1])
            assert append_error.__cause__ is readback_errors[0]
            assert readbacks == 2
            store.append_event, store.query_events = original_append, original_query
            assert await owner.load(intent) == unknown
            recovered = await persist_cleanup_diagnostics(**arguments)
            assert recovered is not None
            events = await store.load_events(intent.session_id)
            assert sum(e.type == EventType.TOOL_EFFECT_CLEANUP_OBSERVED for e in events) == 1
            assert await owner.load(intent) == unknown
        finally:
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_cleanup_diagnostic_truncation_does_not_collapse_distinct_reports(tmp_path, backend):
    async def scenario():
        path = tmp_path / "bounded.db"
        store = InMemorySessionStore() if backend == "memory" else SQLiteSessionStore(path)
        try:
            intent = await _intent(store)
            owner = ToolEffectStateOwner(store)
            executing = await owner.begin(intent, run_epoch=intent.source_run_epoch)
            unknown = await owner.transition(
                executing, state="outcome_unknown", run_epoch=intent.source_run_epoch
            )
            writer = RuntimeEventWriter(
                session_store=store, budget_store=InMemoryBudgetStore(), event_sinks=[]
            )
            arguments = dict(
                store=store,
                writer=writer,
                records=(unknown,),
                redactor=SecretRedactor("secret-canary"),
                attributed=False,
            )
            prefix = {"status": "timeout", "detail": "secret-canary"}
            first = await persist_cleanup_diagnostics(
                **arguments, artifacts=[prefix, {"error": "x" * MAX_DIAGNOSTIC_UTF8_BYTES}]
            )
            second = await persist_cleanup_diagnostics(
                **arguments, artifacts=[prefix, {"error": "y" * MAX_DIAGNOSTIC_UTF8_BYTES}]
            )
            assert first is not None and second is not None
            assert first.id != second.id
            assert first.payload == second.payload
            assert first.payload["truncated"] is True
            assert len(first.payload["artifacts"]) == 1
            assert "tool_call_id" not in first.payload
            assert "secret-canary" not in repr(first.payload)
            if backend == "sqlite":
                await store.close()
                store = SQLiteSessionStore(path)
            events = await store.load_events(intent.session_id)
            assert [e.id for e in events if e.type == EventType.TOOL_EFFECT_CLEANUP_OBSERVED] == [
                first.id,
                second.id,
            ]
            assert await ToolEffectStateOwner(store).load(intent) == unknown
        finally:
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())
