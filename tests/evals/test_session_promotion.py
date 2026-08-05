from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from tests.core.postgres_contention_support import drop_cayu_tables

from cayu import (
    AgentSpec,
    CayuApp,
    Event,
    EventType,
    FilePart,
    InMemorySessionStore,
    Message,
    MessageRole,
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    PostgresSessionStore,
    RunRequest,
    ScriptedModelProvider,
    SessionIdentity,
    SessionPromotionError,
    SessionPromotionErrorCode,
    SessionStatus,
    SessionStore,
    SQLiteSessionStore,
    StructuredOutputSpec,
    TextPart,
    ToolResultPart,
    file_attachment,
    promotable_run_input,
    scripted_structured_output,
    trajectory_from_session,
)
from cayu.evals.models import _trajectory_promotion_capture_sha256
from cayu.runtime.sessions import (
    SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY,
    parse_session_input_contract_evidence,
)
from cayu.storage.migrations import SchemaMode
from cayu.vaults import REDACTED_SECRET, SecretRedactor


class _FailingModelProvider(ModelProvider):
    name = "fake"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        yield ModelStreamEvent.error("captured failure")


def test_versioned_input_contract_parser_rejects_noncanonical_markers():
    messages_sha256 = "a" * 64
    evidence = parse_session_input_contract_evidence(
        f"v1:1:2:redacted:structured:sha256:{messages_sha256}"
    )
    assert evidence.message_start_index == 1
    assert evidence.message_count == 2
    assert evidence.redactions_applied is True
    assert evidence.structured_output_requested is True
    assert evidence.messages_sha256 == messages_sha256

    for invalid in (
        None,
        True,
        f"v2:1:2:redacted:structured:sha256:{messages_sha256}",
        f"v1:01:2:redacted:structured:sha256:{messages_sha256}",
        f"v1:-1:2:redacted:structured:sha256:{messages_sha256}",
        f"v1:1:02:redacted:structured:sha256:{messages_sha256}",
        f"v1:1:-1:redacted:structured:sha256:{messages_sha256}",
        f"v1:1:2:none:structured:sha256:{messages_sha256}",
        f"v1:1:2:redacted:none:sha256:{messages_sha256}",
        f"v1:1:2:redacted:structured:sha1:{messages_sha256}",
        "v1:1:2:redacted:structured:sha256:abc",
        f"v1:1:2:redacted:structured:sha256:{'A' * 64}",
        f"v1:1:2:redacted:structured:sha256:{messages_sha256}:extra",
    ):
        with pytest.raises(ValueError):
            parse_session_input_contract_evidence(invalid)


@pytest.fixture(params=("memory", "sqlite", "postgres"))
def promotion_store_case(request, tmp_path):
    if request.param == "postgres":
        return request.param, tmp_path, request.getfixturevalue("postgres_dsn")
    return request.param, tmp_path, None


async def _open_store(case) -> SessionStore:
    kind, tmp_path, postgres_dsn = case
    if kind == "memory":
        return InMemorySessionStore()
    if kind == "sqlite":
        return SQLiteSessionStore(tmp_path / "session-promotion.sqlite")
    await drop_cayu_tables(postgres_dsn)
    return PostgresSessionStore(
        postgres_dsn,
        min_size=1,
        max_size=4,
        schema_mode=SchemaMode.CREATE,
    )


async def _close_store(store: SessionStore) -> None:
    close = getattr(store, "close", None)
    if close is not None:
        await close()


async def _run_trajectory(
    store: SessionStore,
    *,
    session_id: str = "promotion-root",
    messages: list[Message] | None = None,
    agent_system_prompt: str | None = "Answer precisely.",
    secret_redactor: SecretRedactor | None = None,
    fail: bool = False,
    structured_output: StructuredOutputSpec | None = None,
):
    app = CayuApp(
        session_store=store,
        secret_redactor=secret_redactor,
        enable_logging=False,
    )
    provider: ModelProvider
    if fail:
        provider = _FailingModelProvider()
    elif structured_output is not None:
        provider = ScriptedModelProvider(
            scripted_structured_output({"answer": "captured answer"}),
            name="fake",
        )
    else:
        provider = ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("captured answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            name="fake",
        )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(
            name="assistant",
            model="fake-model",
            system_prompt=agent_system_prompt,
        )
    )
    async for _ in app.run(
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=([Message.text("user", "promote this run")] if messages is None else messages),
            structured_output=structured_output,
        )
    ):
        pass
    return app, await trajectory_from_session(app, session_id)


def _assert_rejection(
    app: CayuApp,
    trajectory,
    expected: SessionPromotionErrorCode,
) -> None:
    with pytest.raises(SessionPromotionError) as captured:
        promotable_run_input(app, trajectory, source_agent_name="assistant")
    assert captured.value.code is expected


def _runtime_attested_trajectory_copy(trajectory, **update):
    """Construct one internally attested fixture after deliberate public-state edits."""

    copied = trajectory.model_copy(update=update)
    copied._promotion_capture_sha256 = _trajectory_promotion_capture_sha256(copied)
    return copied


def _event_before_terminal(trajectory, event_type: EventType):
    assert trajectory.session is not None
    event = Event(
        type=event_type,
        session_id=trajectory.session.id,
        interaction_id=(
            "another-interaction" if event_type is EventType.INTERACTION_STARTED else None
        ),
    )
    return _runtime_attested_trajectory_copy(
        trajectory,
        events=(*trajectory.events[:-1], event, trajectory.events[-1]),
    )


def test_promotable_input_survives_every_builtin_store_and_restart(promotion_store_case):
    async def scenario():
        store = await _open_store(promotion_store_case)
        app, trajectory = await _run_trajectory(
            store,
            messages=[
                Message.text("user", "first request"),
                Message.text("user", "second request"),
            ],
        )
        promoted = promotable_run_input(
            app,
            trajectory,
            source_agent_name="assistant",
        )
        await _close_store(store)
        return trajectory, promoted

    trajectory, promoted = asyncio.run(scenario())
    assert trajectory.initial_input_message_count == 2
    assert trajectory.structured_output_requested is False
    assert trajectory.input_redactions_applied is False
    assert all(
        SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY not in event.payload
        for event in trajectory.events
    )
    assert [message.text for message in promoted.messages] == [
        "first request",
        "second request",
    ]
    assert promoted.redactions_applied is False
    assert promoted.to_run_input_spec().messages == promoted.messages


def test_failed_session_input_is_still_eligible_for_a_regression_case():
    async def scenario():
        app, trajectory = await _run_trajectory(InMemorySessionStore(), fail=True)
        return app, trajectory

    app, trajectory = asyncio.run(scenario())
    assert trajectory.session is not None
    assert trajectory.session.status == SessionStatus.FAILED
    promoted = promotable_run_input(app, trajectory, source_agent_name="assistant")
    assert [message.text for message in promoted.messages] == ["promote this run"]


def test_promotion_rejects_source_mismatch_and_incomplete_descendants():
    async def scenario():
        return await _run_trajectory(InMemorySessionStore())

    app, trajectory = asyncio.run(scenario())
    with pytest.raises(SessionPromotionError) as mismatch:
        promotable_run_input(app, trajectory, source_agent_name="different-agent")
    assert mismatch.value.code == SessionPromotionErrorCode.SOURCE_AGENT_MISMATCH

    _assert_rejection(
        app,
        _runtime_attested_trajectory_copy(trajectory, children_incomplete=True),
        SessionPromotionErrorCode.DESCENDANT_EVIDENCE_UNSUPPORTED,
    )


def test_runtime_attested_structured_output_is_ineligible_without_event_guessing():
    async def scenario():
        app, trajectory = await _run_trajectory(
            InMemorySessionStore(),
            structured_output=StructuredOutputSpec(
                json_schema={
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                }
            ),
        )
        return app, trajectory

    app, trajectory = asyncio.run(scenario())
    assert trajectory.structured_output_requested is True
    _assert_rejection(
        app,
        trajectory,
        SessionPromotionErrorCode.STRUCTURED_OUTPUT_UNSUPPORTED,
    )


def test_promotion_redacts_input_before_returning_a_public_model():
    secret = "promotion-secret-value"

    async def scenario():
        store = InMemorySessionStore()
        app, trajectory = await _run_trajectory(
            store,
            messages=[Message.text("user", f"do not expose {secret}")],
            secret_redactor=SecretRedactor(secret),
        )
        return app, trajectory

    app, trajectory = asyncio.run(scenario())
    promoted = promotable_run_input(app, trajectory, source_agent_name="assistant")
    assert secret not in promoted.messages[0].text
    assert REDACTED_SECRET in promoted.messages[0].text
    assert promoted.redactions_applied is True


def test_promotion_rejects_multiple_text_parts_instead_of_changing_replay_input():
    secret = "split-secret"
    split_message = Message(
        role=MessageRole.USER,
        content=(TextPart(text="split-"), TextPart(text="secret")),
    )

    async def scenario():
        return await _run_trajectory(
            InMemorySessionStore(),
            messages=[split_message],
            secret_redactor=SecretRedactor(secret),
        )

    app, trajectory = asyncio.run(scenario())
    with pytest.raises(SessionPromotionError) as captured:
        promotable_run_input(app, trajectory, source_agent_name="assistant")
    assert captured.value.code is SessionPromotionErrorCode.INPUT_PART_UNSUPPORTED
    assert str(captured.value) == (
        "Portable corpus v1 requires exactly one text part per caller-supplied message."
    )


def test_caller_system_input_without_runtime_bootstrap_uses_role_rejection():
    async def scenario():
        return await _run_trajectory(
            InMemorySessionStore(),
            messages=[Message.text("system", "caller-authored system state")],
            agent_system_prompt=None,
            fail=True,
        )

    app, trajectory = asyncio.run(scenario())
    assert [message.role for message in trajectory.transcript] == [MessageRole.SYSTEM]
    _assert_rejection(
        app,
        trajectory,
        SessionPromotionErrorCode.INPUT_ROLE_UNSUPPORTED,
    )


def test_failed_user_input_without_runtime_bootstrap_remains_eligible():
    async def scenario():
        return await _run_trajectory(
            InMemorySessionStore(),
            messages=[Message.text("user", "caller-authored user input")],
            agent_system_prompt=None,
            fail=True,
        )

    app, trajectory = asyncio.run(scenario())
    promoted = promotable_run_input(app, trajectory, source_agent_name="assistant")
    assert [message.text for message in promoted.messages] == ["caller-authored user input"]


@pytest.mark.parametrize("secret", ["_", "original", "text"])
def test_runtime_input_contract_survives_nonsecret_schema_collisions(secret):
    async def scenario():
        return await _run_trajectory(
            InMemorySessionStore(),
            secret_redactor=SecretRedactor(secret),
        )

    app, trajectory = asyncio.run(scenario())
    promoted = promotable_run_input(app, trajectory, source_agent_name="assistant")

    assert [message.text for message in promoted.messages] == ["promote this run"]
    assert promoted.redactions_applied is False


def test_caller_authored_input_markers_are_stripped_across_builtin_stores(
    promotion_store_case,
):
    async def scenario():
        store = await _open_store(promotion_store_case)
        session_id = "untrusted-promotion-root"
        interaction_id = "untrusted-promotion-interaction"
        user_message = Message.text("user", "caller supplied marker")
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[user_message],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
            interaction_started_event=Event(
                type=EventType.INTERACTION_STARTED,
                session_id=session_id,
                interaction_id=interaction_id,
            ),
            interaction_source_messages=[user_message],
        )
        await store.append_event(
            session_id,
            Event(
                type=EventType.SESSION_STARTED,
                session_id=session_id,
                payload={
                    "agent_name": "assistant",
                    SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY: "v1:1:original:text",
                },
            ),
        )
        await store.replace_initial_transcript_messages(
            session_id,
            [user_message],
            [Message.text("system", "runtime bootstrap"), user_message],
            interaction_id=interaction_id,
        )
        await store.append_transcript_messages(
            session_id,
            [Message.text("assistant", "answer")],
            interaction_id=interaction_id,
        )
        await store.publish_interaction_transition(
            session_id,
            event=Event(
                type=EventType.INTERACTION_COMPLETED,
                session_id=session_id,
                interaction_id=interaction_id,
            ),
            from_statuses={SessionStatus.RUNNING},
            to_status=SessionStatus.COMPLETED,
        )
        await store.append_event(
            session_id,
            Event(type=EventType.SESSION_COMPLETED, session_id=session_id),
        )
        app = CayuApp(session_store=store, enable_logging=False)
        trajectory = await trajectory_from_session(app, session_id)
        durable_started = next(
            event for event in trajectory.events if event.type == EventType.SESSION_STARTED
        )
        await _close_store(store)
        return app, trajectory, durable_started

    app, trajectory, durable_started = asyncio.run(scenario())
    assert SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY not in durable_started.payload
    assert trajectory.initial_input_message_count is None
    assert trajectory.structured_output_requested is None
    assert trajectory.input_redactions_applied is None
    _assert_rejection(
        app,
        trajectory,
        SessionPromotionErrorCode.INPUT_EVIDENCE_UNAVAILABLE,
    )


@pytest.mark.parametrize(
    ("event_type", "expected"),
    [
        (
            EventType.TOOL_CALL_APPROVAL_REQUESTED,
            SessionPromotionErrorCode.APPROVAL_CONTINUATION_UNSUPPORTED,
        ),
        (EventType.SESSION_RESUMED, SessionPromotionErrorCode.SESSION_RESUME_UNSUPPORTED),
        (EventType.SESSION_MESSAGE_QUEUED, SessionPromotionErrorCode.QUEUED_INPUT_UNSUPPORTED),
        (
            EventType.INTERACTION_STARTED,
            SessionPromotionErrorCode.LATER_INTERACTION_UNSUPPORTED,
        ),
        (
            EventType.STRUCTURED_OUTPUT_VALIDATED,
            SessionPromotionErrorCode.STRUCTURED_OUTPUT_UNSUPPORTED,
        ),
    ],
)
def test_promotion_rejects_nonportable_runtime_phases_with_stable_codes(
    event_type,
    expected,
):
    async def scenario():
        app, trajectory = await _run_trajectory(InMemorySessionStore())
        return app, _event_before_terminal(trajectory, event_type)

    app, trajectory = asyncio.run(scenario())
    _assert_rejection(app, trajectory, expected)


@pytest.mark.parametrize(
    ("source_message", "expected"),
    [
        (
            Message.text("system", "caller-authored system state"),
            SessionPromotionErrorCode.INPUT_ROLE_UNSUPPORTED,
        ),
        (
            Message.text("assistant", "caller-authored assistant state"),
            SessionPromotionErrorCode.INPUT_ROLE_UNSUPPORTED,
        ),
        (
            Message(
                role=MessageRole.TOOL,
                content=(
                    ToolResultPart(
                        tool_call_id="caller-call",
                        tool_name="caller-tool",
                        content="caller-authored tool state",
                    ),
                ),
            ),
            SessionPromotionErrorCode.INPUT_ROLE_UNSUPPORTED,
        ),
        (
            Message(
                role=MessageRole.USER,
                content=(
                    FilePart(
                        attachment=file_attachment(
                            artifact_id="artifact-1",
                            kind="document",
                            filename="input.pdf",
                            content_type="application/pdf",
                            size_bytes=1,
                        )
                    ),
                ),
            ),
            SessionPromotionErrorCode.INPUT_PART_UNSUPPORTED,
        ),
    ],
)
def test_promotion_rejects_unsupported_caller_input(source_message, expected):
    async def scenario():
        return await _run_trajectory(
            InMemorySessionStore(),
            messages=[source_message],
        )

    app, trajectory = asyncio.run(scenario())
    _assert_rejection(app, trajectory, expected)


def test_serialized_trajectory_cannot_forge_runtime_input_attestation():
    async def scenario():
        return await _run_trajectory(InMemorySessionStore())

    app, trajectory = asyncio.run(scenario())
    restored = type(trajectory).model_validate(trajectory.model_dump(mode="python"))
    assert restored.initial_input_message_count is None
    assert restored.structured_output_requested is None
    assert restored.input_redactions_applied is None
    _assert_rejection(
        app,
        restored,
        SessionPromotionErrorCode.INPUT_EVIDENCE_UNAVAILABLE,
    )
