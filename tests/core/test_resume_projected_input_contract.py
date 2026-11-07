"""Resume input evidence must address durable rows, not projected model positions."""

import asyncio

import pytest
from tests.core.test_session_model_switch import _app, _NamedProvider, _profiled_source_identity

from cayu import EventType, Message, ResumeRequest, RunRequest, SessionStatus
from cayu.evals import EvaluationSourceIdentityV1, capture_eval_scenario_from_session
from cayu.providers import ModelStreamEvent
from cayu.sessions import ModelTarget
from cayu.sessions.transcript_input import (
    parse_session_input_contract_evidence,
    session_input_messages_sha256,
)
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools import ToolCapabilityCeiling


@pytest.mark.parametrize("opaque_only_answer", [False, True], ids=["text-bearing", "opaque-only"])
def test_model_switch_resume_input_contract_uses_durable_cursor(opaque_only_answer):
    async def scenario():
        source_batch = [] if opaque_only_answer else [ModelStreamEvent.text_delta("first")]
        source_batch.append(
            ModelStreamEvent.completed(
                {"provider_state": [{"provider": "source", "state": {"id": "response-1"}}]}
            )
        )
        source = _NamedProvider("source", [source_batch])
        target = _NamedProvider(
            "target",
            [[ModelStreamEvent.text_delta("continued"), ModelStreamEvent.completed()]],
        )
        app, store = _app(source, target)
        session_id = "switch-input-contract"
        try:
            initial = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "opening")],
                    )
                )
            ]
            assert initial[-1].type is EventType.SESSION_COMPLETED
            before = await store.load_transcript_snapshot(session_id)
            assert before.cursor == 2
            new_input = [Message.text("user", "continue on the target model")]
            resumed = [
                event
                async for event in app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=new_input,
                        target=ModelTarget(provider_name="target", model="target-model"),
                    )
                )
            ]
            assert resumed[-1].type is EventType.SESSION_COMPLETED
            captured = await capture_eval_scenario_from_session(
                app,
                session_id,
                target_key="assistant.default",
                source_agent_name="assistant",
                source=EvaluationSourceIdentityV1(
                    application_release_id="release-current",
                    app_manifest_schema_version="1",
                    app_manifest_fingerprint="a" * 64,
                    evidence_revision="sha256:" + "b" * 64,
                ),
            )
            assert captured.available, captured.diagnostics
            durable_events = await store.load_events(session_id)
            event = next(
                event for event in durable_events if event.type is EventType.SESSION_RESUMED
            )
            contract = parse_session_input_contract_evidence(event.payload["input_contract"])
            assert contract.message_start_index == before.cursor
            after = await store.load_transcript_snapshot(session_id)
            recorded_input = [
                record.message
                for record in after.records
                if contract.message_start_index
                <= record.index
                < contract.message_start_index + contract.message_count
            ]
            assert recorded_input == new_input
            assert session_input_messages_sha256(recorded_input) == contract.messages_sha256
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_resume_input_contract_uses_durable_cursor_after_retention(tmp_path):
    async def scenario():
        source = _NamedProvider(
            "source", [[ModelStreamEvent.text_delta("second"), ModelStreamEvent.completed()]]
        )
        store = SQLiteSessionStore(tmp_path / "retained-input-contract.sqlite")
        app, _ = _app(source, _NamedProvider("target", []), store=store)
        session_id = "retained-input-contract"
        try:
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[],
                    tool_capability_ceiling=ToolCapabilityCeiling(tool_names=("echo",)),
                ),
                identity=_profiled_source_identity(),
            )
            await store.append_transcript_messages(
                session_id, [Message.text("user", "opening"), Message.text("assistant", "first")]
            )
            await store.update_status(session_id, SessionStatus.COMPLETED)
            assert await store.compact_transcript(session_id, keep_last=1) == 1
            before = await store.load_transcript_snapshot(session_id)
            assert before.cursor == 2
            new_input = [Message.text("user", "continue")]
            resumed = [
                event
                async for event in app.resume(
                    ResumeRequest(session_id=session_id, messages=new_input)
                )
            ]
            assert resumed[-1].type is EventType.SESSION_COMPLETED
            event = next(
                event
                for event in await store.load_events(session_id)
                if event.type is EventType.SESSION_RESUMED
            )
            contract = parse_session_input_contract_evidence(event.payload["input_contract"])
            assert contract.message_start_index == before.cursor
            after = await store.load_transcript_snapshot(session_id)
            recorded_input = [
                record.message
                for record in after.records
                if contract.message_start_index
                <= record.index
                < contract.message_start_index + contract.message_count
            ]
            assert session_input_messages_sha256(recorded_input) == contract.messages_sha256
        finally:
            await app.aclose()
            await store.close()

    asyncio.run(scenario())
