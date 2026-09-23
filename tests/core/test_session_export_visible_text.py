"""Public visible-text selection, exact publication and diagnostic boundaries."""

import asyncio
import threading

import pytest
from tests.core.test_session_exports import CONTEXT, harness, published
from tests.core.test_session_exports import backend as backend

from cayu.collaboration.exports import SessionExportConflict, SessionExportDenied
from cayu.events import EventType
from cayu.messages import Message, ProviderStatePart, TextPart, ThinkingPart, ToolCallPart
from cayu.sessions.base import InMemorySessionStore
from cayu.storage import PostgresSessionStore, SQLiteSessionStore

PRIVATE = "private-validation-canary"


def test_visible_selection_is_explicit_versioned_and_immutable():
    from tests.core.test_session_export_contracts import request

    from cayu.collaboration.exports import SessionExportRequest

    original = request()
    assert original.source_selection == "whole_records"
    selected = SessionExportRequest.model_validate(
        {**original.model_dump(), "source_selection": "assistant_visible_text_v1"}
    )
    with pytest.raises(ValueError):
        selected.source_selection = "whole_records"
    for changes in (
        {"source_selection": "assistant_visible_text_v2"},
        {"source_selection": True},
        {"source_indices": ()},
        {"mode": "reviewed_prose"},
    ):
        with pytest.raises(ValueError):
            SessionExportRequest.model_validate({**selected.model_dump(), **changes})


def mixed():
    return Message(
        role="assistant",
        content=(
            TextPart(text="approved"),
            ProviderStatePart(provider="openai", state={"private": PRIVATE}),
            ThinkingPart(text=PRIVATE),
        ),
    )


@pytest.mark.anyio
@pytest.mark.parametrize("kind", ["hidden_only", "tool", "wrong_role"])
async def test_visible_selection_rejects_ineligible_record(backend, kind, caplog, capsys, recwarn):
    async with harness(backend) as case:
        current, store, _, projector = case.app()
        await case.create(store)
        message = {
            "hidden_only": Message(role="assistant", content=(ThinkingPart(text=PRIVATE),)),
            "tool": Message(
                role="assistant",
                content=(
                    TextPart(text="approved"),
                    ToolCallPart(tool_call_id="call", tool_name="tool", arguments={}),
                ),
            ),
            "wrong_role": Message.text("user", "approved"),
        }[kind]
        await store.append_transcript_messages(case.session_id, [message])
        expected = (await case.request(current)).model_copy(
            update={"source_indices": (1,), "source_selection": "assistant_visible_text_v1"}
        )
        with pytest.raises(SessionExportDenied) as failure:
            await current.export_session(expected, context=CONTEXT)
        assert projector.calls == 0 and not await published(store, case.session_id)
        captured = capsys.readouterr()
        assert PRIVATE not in caplog.text + captured.out + captured.err + repr(failure.value) + str(
            [str(w.message) for w in recwarn]
        )


@pytest.mark.anyio
async def test_visible_selection_lost_ack_and_reopen(backend):
    base = {
        "memory": InMemorySessionStore,
        "sqlite": SQLiteSessionStore,
        "postgres": PostgresSessionStore,
    }[backend[0]]

    class LostAck(base):
        session_export_version = 1
        commits = 0

        async def publish_session_operation_guarded_with_store_time(self, *args, **kwargs):
            result = await super().publish_session_operation_guarded_with_store_time(
                *args, **kwargs
            )
            if any(event.type == EventType.SESSION_EXPORT_PUBLISHED for event in kwargs["events"]):
                self.commits += 1
                raise OSError("Acknowledgement lost after native commit")
            return result

    async with harness(backend) as case:
        current, store, _, projector = case.app(store_type=LostAck)
        await case.create(store)
        await store.append_transcript_messages(case.session_id, [mixed()])
        expected = (await case.request(current)).model_copy(
            update={"source_indices": (1,), "source_selection": "assistant_visible_text_v1"}
        )
        receipt = await current.export_session(expected, context=CONTEXT)
        assert store.commits == 1 and projector.calls == 1
        assert await current.export_session(expected, context=CONTEXT) == receipt
        assert len(await published(store, case.session_id)) == 1
        if backend[0] != "memory":
            await current.drain_session_exports()
            await store.close()
            case.stores.remove(store)
            current, _, _, _ = case.app(projectors=())
        assert await current.export_session(expected, context=CONTEXT) == receipt
        assert await current.read_session_export(expected, context=CONTEXT) == {"count": 8}
        assert PRIVATE not in receipt.model_dump_json()


@pytest.mark.anyio
async def test_private_source_change_conflicts_before_publication(backend):
    async with harness(backend) as case:
        current, store, _, projector = case.app()
        await case.create(store)
        original = mixed()
        await store.append_transcript_messages(case.session_id, [original])
        expected = (await case.request(current)).model_copy(
            update={"source_indices": (1,), "source_selection": "assistant_visible_text_v1"}
        )
        projector.release = threading.Event()
        task = asyncio.create_task(current.export_session(expected, context=CONTEXT))
        try:
            assert await asyncio.to_thread(projector.entered.wait, 5)
            changed = Message(
                role="assistant",
                content=(
                    TextPart(text="approved"),
                    ProviderStatePart(provider="openai", state={"private": "changed-private"}),
                    ThinkingPart(text=PRIVATE),
                ),
            )
            # Deliberately alter only private source bytes after selection. The
            # public text commitment is unchanged; native full-row CAS must fail.
            if backend[0] == "memory":
                async with store._lock:
                    store._transcripts[case.session_id][1] = changed
            elif backend[0] == "sqlite":
                store._connection.execute(
                    "UPDATE cayu_transcript_messages SET message_json = ? WHERE session_id = ? AND session_order = 2",
                    (changed.model_dump_json(), case.session_id),
                )
                store._connection.commit()
            else:
                async with store._connection() as connection:
                    await connection.execute(
                        "UPDATE cayu_transcript_messages SET message = %s::jsonb WHERE session_id = %s AND session_order = 2",
                        (changed.model_dump_json(), case.session_id),
                    )
                    await connection.commit()
            projector.release.set()
            with pytest.raises(SessionExportConflict):
                await task
            assert not await published(store, case.session_id)
            assert (
                await current.lookup_session_export(expected, context=CONTEXT)
            ).status == "not_found"
        finally:
            projector.release.set()
            await asyncio.gather(task, return_exceptions=True)
