from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from tests.core.test_tool_effect_state import _event, _intent, _receipt, _terminal
from tests.core.test_tool_effect_store_conformance import _stores

from cayu.runtime._checkpoint_store import _RuntimeCheckpointSessionStore
from cayu.runtime._tool_effect_conflicts import ToolEffectConflictAudit
from cayu.runtime._tool_effect_state import ToolEffectStateOwner
from cayu.runtime.sessions import (
    SessionRunFenced,
    _activate_session_run_fence,
    _deactivate_session_run_fence,
)
from cayu.runtime.tool_effects import ToolEffectConflict


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("lose_acknowledgement", [False, True])
def test_conflict_audit_is_exact_evidence_not_execution_authority(
    backend, lose_acknowledgement, tmp_path, request, monkeypatch
):
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    async def scenario():
        async with _stores(backend, tmp_path / "audit.db", dsn) as open_store:
            store = open_store()
            intent = await _intent(store, session_id=f"audit-{uuid4().hex}")
            owner = ToolEffectStateOwner(store)
            executing = await owner.begin(intent, run_epoch=0)
            unknown = await owner.transition(executing, state="outcome_unknown", run_epoch=0)
            terminal = _event(intent)
            selected = await owner.transition(
                unknown,
                state="reconciled_completed",
                run_epoch=0,
                terminal=_terminal(terminal, _receipt(intent)),
                events=(terminal,),
            )
            audit = ToolEffectConflictAudit(executing, selected)
            before = await store.load(intent.session_id)
            checkpoint = await store.load_checkpoint(intent.session_id)
            # Contract-level fence test. Actual ownership transfer and delayed
            # Tool.run return are covered separately by the public runtime test.
            _activate_session_run_fence(before.model_copy(update={"run_epoch": 1}))
            try:
                with pytest.raises(SessionRunFenced):
                    await store.append_event(
                        intent.session_id, _event(intent, event_id="forbidden")
                    )
                wrapped = _RuntimeCheckpointSessionStore(store)
                if lose_acknowledgement:
                    append = store.append_tool_effect_conflict

                    async def commit_then_raise(value):
                        await append(value)
                        raise OSError("audit acknowledgement lost")

                    monkeypatch.setattr(store, "append_tool_effect_conflict", commit_then_raise)
                    with pytest.raises(OSError, match="audit acknowledgement lost"):
                        await wrapped.append_tool_effect_conflict(audit)
                    monkeypatch.setattr(store, "append_tool_effect_conflict", append)
                first, duplicate = await asyncio.gather(
                    wrapped.append_tool_effect_conflict(audit),
                    wrapped.append_tool_effect_conflict(audit),
                )
                assert first == duplicate
                assert first.type.value == "tool.effect.reconciliation.conflict"
                assert first.payload["kind"] == "late_dispatch"
                assert "key" not in first.payload
                assert await store.load(intent.session_id) == before
                assert await store.load_checkpoint(intent.session_id) == checkpoint
                assert await owner.load(intent) == selected
                with pytest.raises(SessionRunFenced):
                    await store.append_event(
                        intent.session_id, _event(intent, event_id="still-forbidden")
                    )
                assert len(await store.load_events(intent.session_id)) == 3
                delivery = await store.get_persisted_event_side_effect_delivery(
                    session_id=intent.session_id, event_id=first.id
                )
                assert delivery is not None
                reconstructed = open_store()
                assert await reconstructed.append_tool_effect_conflict(audit) == first
                altered = selected.model_copy(update={"publication_digest": "f" * 64})
                with pytest.raises(ToolEffectConflict):
                    await store.append_tool_effect_conflict(
                        ToolEffectConflictAudit(executing, altered)
                    )
                other_incarnation = intent.model_copy(update={"session_instance_id": "another"})
                with pytest.raises(ToolEffectConflict):
                    await store.append_tool_effect_conflict(
                        ToolEffectConflictAudit(
                            executing.model_copy(update={"intent": other_incarnation}),
                            selected.model_copy(update={"intent": other_incarnation}),
                        )
                    )
                with pytest.raises(TypeError):
                    await store.append_tool_effect_conflict(object())
                assert len(await store.load_events(intent.session_id)) == 3
            finally:
                _deactivate_session_run_fence(intent.session_id)

    asyncio.run(scenario())
