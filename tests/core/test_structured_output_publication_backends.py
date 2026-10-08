"""Public structured-output publication across the durable store backends."""

from __future__ import annotations

import asyncio
import base64
from contextlib import aclosing

import pytest
from pydantic import SecretStr
from tests.core.test_resource_execution_access import Policy
from tests.core.test_structured_output_tool_round_recovery import (
    _answer_spec,
    _RecordingProvider,
    _register_runtime,
    _SideEffectTool,
)
from tests.core.test_tool_round_publication_failure_matrix import (
    _WATCHDOG_SECONDS,
    _finish_tasks,
    _LostAcknowledgementStore,
    _ProcessLossStore,
    _PublicationBarrierStore,
    _SimulatedProcessLoss,
)
from tests.core.test_tool_round_publication_failure_matrix import (
    store_factory as store_factory,
)

from cayu import AgentSpec, CayuApp
from cayu.context.structured_output import STRUCTURED_OUTPUT_TOOL_NAME, StructuredOutputSpec
from cayu.events import EventType
from cayu.messages import Message, ToolResultPart
from cayu.providers import ModelStreamEvent
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.sessions import _pending_tool_round_reader as pending_round_reader
from cayu.sessions.access import SessionAccessDenied, SessionAccessScope
from cayu.sessions.base import IncompleteSessionRecoveryRequest, RunRequest
from cayu.sessions.records import SessionStatus
from cayu.vaults import REDACTED_SECRET, SecretRedactor


@pytest.fixture
def structured_store_factory(store_factory):
    # PostgreSQL shares a module database, so every opener uses the same test keyring.
    key = base64.urlsafe_b64encode(bytes([41]) * 32).decode().rstrip("=")
    codec = PublicAuthorityAliasCodec(
        PublicAuthorityAliasKeyring(active_key_id="test", keys={"test": SecretStr(key)})
    )

    def open_store(fault_type, **options):
        return store_factory(fault_type, public_authority_alias_codec=codec, **options)

    return open_store


class _RecoveryPublicationBarrierStore(_ProcessLossStore, _PublicationBarrierStore):
    """Stage a model round before enabling the ordinary publication barrier."""


def _response(output, *, mixed=False):
    events = [
        ModelStreamEvent.tool_call(
            id="final-call",
            name=STRUCTURED_OUTPUT_TOOL_NAME,
            arguments={"output": output},
        )
    ]
    if mixed:
        events.append(
            ModelStreamEvent.tool_call(
                id="side-effect-call", name="side_effect", arguments={"value": "unused"}
            )
        )
    return [*events, ModelStreamEvent.completed({"finish_reason": "tool_calls"})]


def _request(session_id, spec):
    return RunRequest(
        agent_name="assistant",
        session_id=session_id,
        messages=[Message.text("user", "Return the structured answer.")],
        structured_output=spec,
    )


async def _run(app, request):
    async with aclosing(app.run(request)) as events:
        return [event async for event in events]


async def _leave_pending_round(app, store, request):
    with pytest.raises(_SimulatedProcessLoss):
        await _run(app, request)
    assert store.tool_publication_attempted.is_set()
    assert pending_round_reader.pending_tool_round_from_checkpoint(
        await store.load_checkpoint(request.session_id)
    )
    store.fail_before_tool_publication = False
    await store.release_run_fence(request.session_id)
    await store.update_status(request.session_id, SessionStatus.INTERRUPTED)


async def _assert_round(store, session_id, *, valid):
    events = await store.load_events(session_id)
    round_events = [
        event
        for event in events
        if event.type
        in {
            EventType.TOOL_CALL_COMPLETED,
            EventType.TOOL_CALL_FAILED,
            EventType.STRUCTURED_OUTPUT_VALIDATING,
            EventType.STRUCTURED_OUTPUT_VALIDATED,
            EventType.STRUCTURED_OUTPUT_FAILED,
            EventType.STRUCTURED_OUTPUT_RETRY,
        }
    ]
    assert [event.type for event in round_events] == [
        EventType.TOOL_CALL_COMPLETED if valid else EventType.TOOL_CALL_FAILED,
        EventType.STRUCTURED_OUTPUT_VALIDATING,
        EventType.STRUCTURED_OUTPUT_VALIDATED if valid else EventType.STRUCTURED_OUTPUT_FAILED,
    ]
    terminal, *auxiliary = round_events
    round_id = terminal.payload["tool_round_id"]
    assert all(event.payload["tool_round_id"] == round_id for event in auxiliary)
    assert all(
        event.payload["execution_profile_fingerprint"]
        == terminal.payload["execution_profile_fingerprint"]
        for event in auxiliary
    )
    receipt = await store.load_runtime_publication_receipt(session_id, f"tool-round:{round_id}")
    assert receipt is not None
    assert receipt.appended_event_ids == tuple(event.id for event in auxiliary)
    assert tuple(reference.event_id for reference in receipt.referenced_events) == (terminal.id,)
    assert receipt.intent["auxiliary"] == {
        "schema_version": 1,
        "kind": "structured-output-validation",
        "step": 1,
        "attempt": 1,
        "valid": valid,
        "retry_scheduled": False,
        "event_ids": [event.id for event in auxiliary],
    }
    transcript = await store.load_transcript(session_id)
    assert [message.role.value for message in transcript] == ["user", "assistant", "tool"]
    result = transcript[-1].content[0]
    assert isinstance(result, ToolResultPart)
    assert result.tool_call_id == "final-call"
    assert result.is_error is (not valid)
    assert (
        pending_round_reader.pending_tool_round_from_checkpoint(
            await store.load_checkpoint(session_id)
        )
        is None
    )
    assert len({event.id for event in events}) == len(events)
    return events, transcript


@pytest.mark.parametrize("entrance", ["live", "recovery"])
@pytest.mark.parametrize("valid", [True, False], ids=["valid-secret", "invalid"])
def test_structured_publication_replays_exact_request_and_auxiliary_events(
    structured_store_factory, entrance, valid
):
    async def scenario():
        async with structured_store_factory(_LostAcknowledgementStore) as store:
            secret = "structured-publication-secret-longer-than-thirty-characters"
            spec = StructuredOutputSpec(
                json_schema={
                    "type": "object",
                    "properties": {"answer": {"type": "string", "minLength": 30}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
                max_retries=0,
            )
            provider = _RecordingProvider([_response({"answer": secret} if valid else {})])
            redactor = SecretRedactor(secret)
            app = _register_runtime(store, provider, secret_redactor=redactor)
            request = _request(f"structured-ack-{entrance}-{valid}", spec)
            if entrance == "recovery":
                await _leave_pending_round(app, store, request)
                app = _register_runtime(store, provider, secret_redactor=redactor)
                await app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(session_id=request.session_id)
                )
            else:
                store.fail_before_tool_publication = False
                returned = await _run(app, request)
                assert returned[-1].type is (
                    EventType.SESSION_COMPLETED if valid else EventType.SESSION_FAILED
                )
            assert len(provider.requests) == 1
            assert len(store.publication_requests) == 2
            assert store.publication_requests[0] == store.publication_requests[1]
            committed, replayed = store.publication_results
            assert not committed.replayed and replayed.replayed
            assert committed.receipt == replayed.receipt
            events, transcript = await _assert_round(store, request.session_id, valid=valid)
            assert secret not in str([event.model_dump(mode="json") for event in events])
            assert secret not in str([message.model_dump(mode="json") for message in transcript])
            if valid:
                assert transcript[-1].content[0].structured == {
                    "output": {"answer": REDACTED_SECRET}
                }
            if entrance == "recovery":
                await app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(session_id=request.session_id)
                )
                assert await store.load_transcript(request.session_id) == transcript
                assert len(store.publication_requests) == 2

    asyncio.run(scenario())


@pytest.mark.parametrize("entrance", ["live", "recovery"])
@pytest.mark.parametrize("boundary", ["before-commit", "after-commit"])
def test_structured_publication_retains_repeated_cancellation(
    structured_store_factory, entrance, boundary
):
    async def scenario():
        async with structured_store_factory(
            _RecoveryPublicationBarrierStore, boundary=boundary
        ) as store:
            provider = _RecordingProvider([_response({"answer": "done"})])
            app = _register_runtime(store, provider)
            request = _request(f"structured-cancel-{entrance}-{boundary}", _answer_spec())
            if entrance == "recovery":
                await _leave_pending_round(app, store, request)
                app = _register_runtime(store, provider)
                operation = app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(session_id=request.session_id)
                )
            else:
                store.fail_before_tool_publication = False
                operation = _run(app, request)
            running = asyncio.create_task(operation)
            try:
                await asyncio.wait_for(store.boundary_reached.wait(), _WATCHDOG_SECONDS)
                running.cancel("structured publication cancelled")
                running.cancel("structured publication cancelled")
                store.release_publication.set()
                with pytest.raises(
                    asyncio.CancelledError, match="structured publication cancelled"
                ):
                    await asyncio.wait_for(running, _WATCHDOG_SECONDS)
            finally:
                store.release_publication.set()
                await _finish_tasks(running)
                assert await app.drain_background_interruptions()
            assert running.cancelled()
            assert len(provider.requests) == 1
            await _assert_round(store, request.session_id, valid=True)

    asyncio.run(scenario())


def test_mixed_structured_round_retries_without_dispatching_sibling(structured_store_factory):
    async def scenario():
        async with structured_store_factory(_LostAcknowledgementStore) as store:
            store.fail_before_tool_publication = False
            provider = _RecordingProvider(
                [_response({"answer": "first"}, mixed=True), _response({"answer": "second"})]
            )
            tool = _SideEffectTool()
            app = _register_runtime(store, provider, tools=[tool])
            request = _request("structured-mixed-retry", _answer_spec())
            returned = await _run(app, request)
            assert returned[-1].type is EventType.SESSION_COMPLETED
            assert not tool.calls
            assert len(provider.requests) == 2
            events = await store.load_events(request.session_id)
            retries = [event for event in events if event.type is EventType.STRUCTURED_OUTPUT_RETRY]
            assert len(retries) == 1
            validated = [
                event for event in events if event.type is EventType.STRUCTURED_OUTPUT_VALIDATED
            ]
            assert len(validated) == 1
            assert validated[0].payload["tool_round_id"] != retries[0].payload["tool_round_id"]
            transcript = await store.load_transcript(request.session_id)
            results = [message for message in transcript if message.role.value == "tool"]
            assert [[part.tool_call_id for part in message.content] for message in results] == [
                ["final-call", "side-effect-call"],
                ["final-call"],
            ]
            assert all(part.is_error for part in results[0].content)
            assert results[1].content[0].structured == {"output": {"answer": "second"}}
            assert len({event.id for event in events}) == len(events)

    asyncio.run(scenario())


@pytest.mark.parametrize("mixed", [False, True], ids=["valid", "mixed"])
def test_closed_structured_stream_recovers_partial_terminal_evidence(
    structured_store_factory, mixed
):
    async def scenario():
        async with structured_store_factory(_ProcessLossStore) as store:
            store.fail_before_tool_publication = False
            provider = _RecordingProvider([_response({"answer": "done"}, mixed=mixed)])
            tool = _SideEffectTool()
            app = _register_runtime(store, provider, tools=[tool])
            request = _request(f"structured-stream-close-{mixed}", _answer_spec())
            async with aclosing(app.run(request)) as stream:
                async for event in stream:
                    if event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}:
                        break
                else:
                    raise AssertionError("Structured round never reached its first terminal.")
            assert await app.drain_background_interruptions()
            assert len(provider.requests) == 1
            assert not tool.calls
            assert (await store.load(request.session_id)).status is SessionStatus.INTERRUPTED
            assert [
                message.role.value for message in await store.load_transcript(request.session_id)
            ] == ["user"]
            pending = pending_round_reader.pending_tool_round_from_checkpoint(
                await store.load_checkpoint(request.session_id)
            )
            assert pending is not None
            assert (
                await store.load_runtime_publication_receipt(
                    request.session_id, f"tool-round:{pending.tool_round_id}"
                )
                is None
            )
            app = _register_runtime(store, provider, tools=[tool])
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=request.session_id)
            )
            assert len(provider.requests) == 1
            assert not tool.calls
            transcript = await store.load_transcript(request.session_id)
            assert [message.role.value for message in transcript] == ["user", "assistant", "tool"]
            assert [part.tool_call_id for part in transcript[-1].content] == (
                ["final-call", "side-effect-call"] if mixed else ["final-call"]
            )
            events = await store.load_events(request.session_id)
            terminals = [
                event
                for event in events
                if event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
            ]
            assert len(terminals) == (2 if mixed else 1)
            round_id = terminals[0].payload["tool_round_id"]
            receipt = await store.load_runtime_publication_receipt(
                request.session_id, f"tool-round:{round_id}"
            )
            assert receipt is not None
            assert receipt.intent["auxiliary"]["valid"] is (not mixed)
            assert receipt.intent["auxiliary"]["retry_scheduled"] is False
            assert len({event.id for event in events}) == len(events)
            assert (
                pending_round_reader.pending_tool_round_from_checkpoint(
                    await store.load_checkpoint(request.session_id)
                )
                is None
            )

    asyncio.run(scenario())


def test_denied_structured_run_does_not_create_a_round(structured_store_factory):
    async def scenario():
        async with structured_store_factory(_ProcessLossStore) as store:
            policy = Policy()
            policy.current = SessionAccessScope()
            provider = _RecordingProvider([_response({"answer": "unused"})])
            app = CayuApp(session_store=store, resource_access_policy=policy, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="assistant", model="fake-model"))
            access = await app.access("alice")
            request = _request("structured-denied", _answer_spec())
            with pytest.raises(SessionAccessDenied):
                await _run(access, request)
            assert not provider.requests
            assert await store.load(request.session_id) is None
            assert not store.tool_publication_attempted.is_set()

    asyncio.run(scenario())
