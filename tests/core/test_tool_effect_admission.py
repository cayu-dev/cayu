from __future__ import annotations

import asyncio

import pytest
from tests.core.test_tool_effect_state import _event, _intent, _terminal

from cayu.core.events import Event, EventType
from cayu.runtime._tool_effect_state import ToolEffectConflict, ToolEffectStateOwner, _digest
from cayu.runtime.sessions import InMemorySessionStore
from cayu.storage.sqlite import SQLiteSessionStore


def _start(record, event_id="start", digest="c" * 64):
    intent = record.intent
    return Event(
        id=event_id,
        type=EventType.TOOL_EFFECT_RECONCILIATION_STARTED,
        session_id=intent.session_id,
        interaction_id=intent.interaction_id,
        agent_name=intent.agent_name,
        tool_name=intent.tool_name,
        payload={
            "schema_version": 1,
            "request_digest": digest,
            "intent_digest": _digest(intent.model_dump(mode="json")),
            "dispatch_id": record.dispatch_id,
            "expected_revision": record.revision,
            "expected_run_epoch": 0,
            "lookup": True,
            "execution_profile_fingerprint": intent.execution_profile_fingerprint,
            **{
                name: getattr(intent, name)
                for name in (
                    "model_step_id",
                    "model_attempt_id",
                    "tool_round_id",
                    "tool_call_id",
                    "approval_id",
                )
            },
        },
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("fault", ["none", "precommit", "lost_ack"])
@pytest.mark.parametrize("successor", ["terminal", "new_request"])
def test_admission_and_supersession_share_the_effect_transaction(
    backend, fault, successor, tmp_path, monkeypatch
):
    async def scenario():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(str(tmp_path / "admission.db"))
        )
        try:
            intent = await _intent(store)
            owner = ToolEffectStateOwner(store)
            executing = await owner.begin(intent, run_epoch=0)
            unknown = await owner.transition(executing, state="outcome_unknown", run_epoch=0)
            started = _start(unknown)
            publish = store.publish_session_operation
            remaining = fault != "none"

            async def fail_publication(*args, **kwargs):
                nonlocal remaining
                if remaining:
                    remaining = False
                    if fault == "lost_ack":
                        await publish(*args, **kwargs)
                    raise OSError("admission write acknowledgement unavailable")
                return await publish(*args, **kwargs)

            async def admit(record, event):
                return await owner.start_reconciliation(
                    record,
                    source_run_epoch=0,
                    run_epoch=0,
                    request_digest=event.payload["request_digest"],
                    lookup=True,
                    event=event,
                )

            # Invalid event material cannot create either side of admission.
            with pytest.raises(ToolEffectConflict):
                await admit(unknown, started.model_copy(update={"tool_name": "another-tool"}))
            for changes in (
                {"lookup": 1},
                {"expected_run_epoch": False},
                {"schema_version": True},
                {"request_digest": "e" * 64},
                {"tool_call_id": "another-call"},
            ):
                invalid = started.model_copy(update={"payload": {**started.payload, **changes}})
                with pytest.raises(ToolEffectConflict):
                    await owner.start_reconciliation(
                        unknown,
                        source_run_epoch=0,
                        run_epoch=0,
                        request_digest="c" * 64,
                        lookup=True,
                        event=invalid,
                    )
            assert await owner.load(intent) == unknown
            before = await store.load_events(intent.session_id)
            monkeypatch.setattr(store, "publish_session_operation", fail_publication)
            if fault == "precommit":
                with pytest.raises(OSError):
                    await admit(unknown, started)
                assert await owner.load(intent) == unknown
                assert await store.load_events(intent.session_id) == before
            admitted = await admit(unknown, started)
            assert admitted.revision == unknown.revision + 1
            assert admitted.reconciliation_attempt is not None
            assert admitted.reconciliation_attempt.event_id == started.id
            assert len(await store.load_events(intent.session_id)) == len(before) + 1
            assert await admit(unknown, started) == admitted
            assert len(await store.load_events(intent.session_id)) == len(before) + 1
            monkeypatch.setattr(store, "publish_session_operation", publish)
            # Reconstruction retains the exact admitted attempt, not a process flag.
            if backend == "sqlite":
                await store.close()
                store = SQLiteSessionStore(str(tmp_path / "admission.db"))
            owner = ToolEffectStateOwner(store)
            assert await owner.load(intent) == admitted
            publish = store.publish_session_operation
            remaining = fault != "none"
            monkeypatch.setattr(store, "publish_session_operation", fail_publication)
            event = (
                _event(intent)
                if successor == "terminal"
                else _start(admitted, "second-start", "d" * 64)
            )

            async def select():
                if successor == "terminal":
                    return await owner.transition(
                        admitted,
                        state="completed",
                        run_epoch=0,
                        terminal=_terminal(event),
                        events=(event,),
                    )
                return await admit(admitted, event)

            before_selection = await store.load_events(intent.session_id)
            if fault == "precommit":
                with pytest.raises(OSError):
                    await select()
                assert await owner.load(intent) == admitted
                assert await store.load_events(intent.session_id) == before_selection
            selected = await select()
            if successor == "terminal":
                assert selected.reconciliation_attempt is None
            else:
                assert selected.reconciliation_attempt.request_digest == "d" * 64
            history = await store.load_events(intent.session_id)
            audits = [e for e in history if e.type == EventType.TOOL_EFFECT_RECONCILIATION_CONFLICT]
            assert len(audits) == 1
            assert audits[0].payload["kind"] == "reconciliation_superseded"
            assert audits[0].payload["request_digest"] == "c" * 64
            assert audits[0].payload["attempt_digest"] == _digest(
                admitted.reconciliation_attempt.model_dump(mode="json")
            )
            assert await owner.load(intent) == selected
            if successor == "terminal":
                assert (
                    await owner.transition(
                        admitted,
                        state="completed",
                        run_epoch=0,
                        terminal=_terminal(event),
                        events=(event,),
                    )
                    == selected
                )
            else:
                assert await admit(admitted, event) == selected
            assert await store.load_events(intent.session_id) == history
            with pytest.raises(ToolEffectConflict):
                await admit(unknown, _start(unknown, "stale-start", "e" * 64))
            assert await owner.load(intent) == selected
            assert await store.load_events(intent.session_id) == history
        finally:
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())
