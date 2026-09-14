"""Native admission preserves unresolved external-effect recovery evidence."""

import asyncio
from hashlib import sha256
from uuid import uuid4

import pytest
from tests.core.test_session_closure_admission import RecordingDependent
from tests.core.test_tool_effect_state import _event, _intent, _receipt, _terminal
from tests.core.test_tool_effect_store_conformance import _stores
from tests.core.test_tool_round_execution_identities import _SequencedProvider, _tool_call_response

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu._validation import canonical_durable_json_bytes
from cayu.events import Event, EventType
from cayu.runtime._tool_effect_state import ToolEffectStateOwner
from cayu.sessions.base import SessionStatus
from cayu.tools.base import Tool, ToolEffect, ToolSpec


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_interrupted_external_effect_prevents_public_closure(backend, tmp_path, request):
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    async def scenario():
        async with _stores(backend, tmp_path / "unknown.db", dsn) as open_store:
            store = open_store()
            calls = []

            class External(Tool):
                spec = ToolSpec(name="record", effect=ToolEffect.EXTERNAL)

                async def run(self, ctx, args):
                    calls.append(ctx.idempotency_key)
                    raise RuntimeError("external acknowledgement lost")

            dependent = RecordingDependent()
            app = CayuApp(
                session_store=store, enable_logging=False, session_closure_stores=(dependent,)
            )
            app.register_provider(_SequencedProvider([_tool_call_response(7)]), default=True)
            app.register_agent(AgentSpec(name="agent", model="test"), tools=[External()])
            session_id = uuid4().hex
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="agent",
                        messages=[Message.text("user", "go")],
                    )
                )
            ]
            assert len(calls) == 1
            assert events[-1].type is EventType.SESSION_INTERRUPTED
            # Reconstruct the closure entrance after the invocation released its fence.
            observer = open_store()
            closer = CayuApp(
                session_store=observer, enable_logging=False, session_closure_stores=(dependent,)
            )
            before = await observer.load_session_closure_records(
                session_id, max_records=1000, max_bytes=4_000_000
            )
            for operation in (closer.validate_session_closure, closer.erase_session_closure):
                with pytest.raises(ValueError, match="settled protected tool effects"):
                    await operation(session_id)
                assert dependent.deleted == []
                assert (
                    await observer.load_session_closure_records(
                        session_id, max_records=1000, max_bytes=4_000_000
                    )
                    == before
                )

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("late", [False, True])
def test_protected_effect_admission_and_reconciled_retry(
    backend, late, tmp_path, request, monkeypatch
):
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    async def scenario():
        async with _stores(backend, tmp_path / "reconcile.db", dsn) as open_store:
            store = open_store()
            intent = await _intent(store, session_id=uuid4().hex)
            # Durable child arguments are not limited to the smaller effect
            # publication-digest material. Settled large records remain erasable.
            arguments = {"task": "x" * (300 * 1024)}
            intent = intent.model_copy(
                update={
                    "arguments_digest": sha256(
                        canonical_durable_json_bytes(arguments, "arguments")
                    ).hexdigest()
                }
            )
            owner = ToolEffectStateOwner(store)
            await store.update_status(intent.session_id, SessionStatus.INTERRUPTED)
            await store.append_event(
                intent.session_id,
                Event(type=EventType.SESSION_INTERRUPTED, session_id=intent.session_id),
            )
            dependent = RecordingDependent()
            app = CayuApp(
                session_store=store, enable_logging=False, session_closure_stores=(dependent,)
            )
            unknown = None

            async def publish_unknown():
                nonlocal unknown
                executing = await owner.begin(
                    intent, run_epoch=0, child_recovery_arguments=arguments
                )
                unknown = await owner.transition(executing, state="outcome_unknown", run_epoch=0)

            original = store.claim_session_closure_progress
            proposed = []

            async def claim(progress):
                proposed.append(progress)
                if late:
                    await publish_unknown()
                return await original(progress)

            monkeypatch.setattr(store, "claim_session_closure_progress", claim)
            if not late:
                await publish_unknown()
            with pytest.raises(ValueError, match="settled protected tool effects"):
                await app.erase_session_closure(intent.session_id)
            assert dependent.deleted == []
            for progress in proposed:
                assert (
                    await store.load_session_closure_progress(
                        intent.session_id, progress["plan_id"]
                    )
                    is None
                )
            assert await owner.load(intent) == unknown
            event = _event(intent)
            await owner.transition(
                unknown,
                state="reconciled_completed",
                run_epoch=0,
                terminal=_terminal(event, _receipt(intent)),
                events=(event,),
            )
            monkeypatch.setattr(store, "claim_session_closure_progress", original)
            assert (await app.erase_session_closure(intent.session_id)).complete
            assert await store.load(intent.session_id) is None
            assert dependent.deleted == [intent.session_id]

    asyncio.run(scenario())
