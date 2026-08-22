from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from collections.abc import AsyncIterator
from decimal import Decimal
from pathlib import Path

import pytest
from tests.core.postgres_contention_support import drop_cayu_tables

from cayu import (
    AgentSpec,
    CayuApp,
    EvaluationEvidencePolicySpec,
    Event,
    EventType,
    FilePart,
    InMemorySessionStore,
    MaxEstimatedCostAssertionSpec,
    Message,
    MessageRole,
    ModelPrice,
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    PostgresSessionStore,
    PriceBook,
    PromotionCandidateV1,
    PromotionWarningCode,
    RootStatusAssertionSpec,
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
    Trajectory,
    build_captured_evaluation_candidate,
    build_promotion_candidate,
    corpus_from_captured_evaluation_candidate,
    corpus_from_promotion_candidate,
    eval_corpus_from_json,
    export_promotion_corpus,
    file_attachment,
    promotable_run_input,
    score_captured_evaluation_candidate,
    score_promotion_candidate,
    scripted_structured_output,
    session_usage_summary,
    trajectory_from_session,
)
from cayu.evals.models import _trajectory_promotion_capture_sha256
from cayu.runtime.sessions import (
    SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY,
    parse_session_input_contract_evidence,
    session_input_messages_sha256,
)
from cayu.storage.migrations import SchemaMode
from cayu.vaults import REDACTED_SECRET, SecretRedactor


class _FailingModelProvider(ModelProvider):
    name = "fake"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        yield ModelStreamEvent.error("captured failure")


class _RepeatableModelProvider(ModelProvider):
    name = "fake"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        yield ModelStreamEvent.text_delta("same answer")
        yield ModelStreamEvent.completed(
            {
                "finish_reason": "stop",
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                },
            }
        )


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


def _price_book(*, input_rate: str = "1", currency: str = "USD") -> PriceBook:
    return PriceBook(
        price_book_version="2026-08-05",
        generated_at="2026-08-05T00:00:00Z",
        prices=(
            ModelPrice.fixed(
                provider_name="fake",
                model="fake-model",
                input_per_million=Decimal(input_rate),
                output_per_million=Decimal("2"),
                currency=currency,
            ),
        ),
    )


@pytest.fixture(params=("memory", "sqlite", pytest.param("postgres", marks=pytest.mark.postgres)))
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
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 10,
                            "output_tokens": 5,
                            "total_tokens": 15,
                        },
                    }
                ),
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


async def _run_repeatable_trajectories(
    runs: tuple[tuple[str, str, str], ...],
) -> tuple[CayuApp, dict[str, Trajectory]]:
    app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
    app.register_provider(_RepeatableModelProvider(), default=True)
    for agent_name in sorted({agent_name for _, agent_name, _ in runs}):
        app.register_agent(
            AgentSpec(
                name=agent_name,
                model="fake-model",
                system_prompt="Same prompt.",
            )
        )

    trajectories: dict[str, Trajectory] = {}
    for session_id, agent_name, input_text in runs:
        async for _ in app.run(
            RunRequest(
                agent_name=agent_name,
                session_id=session_id,
                messages=[Message.text("user", input_text)],
            )
        ):
            pass
        trajectories[session_id] = await trajectory_from_session(app, session_id)
    return app, trajectories


def _assert_rejection(
    app: CayuApp,
    trajectory,
    expected: SessionPromotionErrorCode,
) -> None:
    with pytest.raises(SessionPromotionError) as captured:
        promotable_run_input(app, trajectory, source_agent_name="assistant")
    assert captured.value.code is expected


def _runtime_attested_trajectory_copy(trajectory: Trajectory, **update) -> Trajectory:
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


def _completed_child_tree(
    trajectory: Trajectory,
    event_type: EventType | None = None,
) -> Trajectory:
    """Build a contract-valid completed child, optionally with one extra event."""

    assert trajectory.session is not None
    child_id = "promotion-child"
    child_events = tuple(
        event.model_copy(
            update={
                "id": f"child-{event.id}",
                "session_id": child_id,
            }
        )
        for event in trajectory.events
    )
    if event_type is not None:
        extra_event = Event(
            id=f"child-extra-{event_type.value}",
            type=event_type,
            session_id=child_id,
            interaction_id=(
                "child-later-interaction" if event_type is EventType.INTERACTION_STARTED else None
            ),
        )
        child_events = (*child_events[:-1], extra_event, child_events[-1])
    child = trajectory.model_copy(
        deep=True,
        update={
            "session": trajectory.session.model_copy(
                update={
                    "id": child_id,
                    "parent_session_id": trajectory.session.id,
                }
            ),
            "events": child_events,
            "usage_summary": session_usage_summary(child_id, list(child_events)),
            "children": (),
        },
    )
    return _runtime_attested_trajectory_copy(trajectory, children=(child,))


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
    assert trajectory.initial_input_message_start_index == 1
    assert trajectory.initial_input_message_count == 2
    assert trajectory.initial_input_messages_sha256 is not None
    assert len(trajectory.initial_input_messages_sha256) == 64
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
    assert promoted.revision.startswith("sha256:")
    assert promoted.redactions_applied is False
    assert promoted.to_run_input_spec().messages == promoted.messages

    forged = promoted.model_copy(update={"revision": "sha256:" + "0" * 64})
    with pytest.raises(ValueError, match="revision does not match"):
        type(promoted).model_validate(forged.model_dump(mode="python"))


def test_promotion_rejects_copied_trajectory_with_replaced_attested_input():
    async def scenario():
        return await _run_trajectory(InMemorySessionStore())

    app, trajectory = asyncio.run(scenario())
    transcript = list(trajectory.transcript)
    source_index = next(
        index for index, message in enumerate(transcript) if message.role is MessageRole.USER
    )
    transcript[source_index] = Message.text("user", "forged caller input")
    changed = trajectory.model_copy(update={"transcript": tuple(transcript)})

    assert changed.initial_input_messages_sha256 == trajectory.initial_input_messages_sha256
    _assert_rejection(
        app,
        changed,
        SessionPromotionErrorCode.INPUT_EVIDENCE_INCONSISTENT,
    )
    with pytest.raises(SessionPromotionError) as captured:
        build_promotion_candidate(
            app,
            changed,
            target_key="assistant",
            source_agent_name="assistant",
            application_release_id="release-forged-input",
            evidence_policy=EvaluationEvidencePolicySpec.standard(),
        )
    assert captured.value.code is SessionPromotionErrorCode.INPUT_EVIDENCE_INCONSISTENT


def test_promotion_capture_binds_the_selected_input_range_across_every_entry_point():
    async def scenario():
        return await _run_trajectory(
            InMemorySessionStore(),
            messages=[
                Message.text("user", "first retained input"),
                Message.text("user", "second silently dropped"),
            ],
        )

    app, trajectory = asyncio.run(scenario())
    candidate = build_promotion_candidate(
        app,
        trajectory,
        target_key="assistant",
        source_agent_name="assistant",
        application_release_id="release-attestation-range",
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
    )
    start_index = trajectory.initial_input_message_start_index
    assert start_index is not None
    for selected_offset in (0, 1):
        selected_start = start_index + selected_offset
        forged = trajectory.model_copy(
            update={
                "_initial_input_message_start_index": selected_start,
                "_initial_input_message_count": 1,
                "_initial_input_messages_sha256": session_input_messages_sha256(
                    trajectory.transcript[selected_start : selected_start + 1]
                ),
            }
        )
        assert forged.initial_input_message_start_index == selected_start
        assert forged.initial_input_message_count == 1
        assert forged._promotion_capture_sha256 == trajectory._promotion_capture_sha256

        _assert_rejection(
            app,
            forged,
            SessionPromotionErrorCode.INPUT_EVIDENCE_INCONSISTENT,
        )
        with pytest.raises(SessionPromotionError) as candidate_error:
            build_promotion_candidate(
                app,
                forged,
                target_key="assistant",
                source_agent_name="assistant",
                application_release_id="release-attestation-range",
                evidence_policy=EvaluationEvidencePolicySpec.standard(),
            )
        assert candidate_error.value.code is SessionPromotionErrorCode.INPUT_EVIDENCE_INCONSISTENT
        with pytest.raises(SessionPromotionError) as score_error:
            score_promotion_candidate(
                app,
                forged,
                candidate,
                target_key="assistant",
                source_agent_name="assistant",
                application_release_id="release-attestation-range",
            )
        assert score_error.value.code is SessionPromotionErrorCode.INPUT_EVIDENCE_INCONSISTENT


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    [
        ("_input_redactions_applied", True),
        ("_structured_output_requested", True),
    ],
)
def test_promotion_capture_binds_private_input_modes(field_name, changed_value):
    async def scenario():
        return await _run_trajectory(InMemorySessionStore())

    app, trajectory = asyncio.run(scenario())
    changed = trajectory.model_copy(update={field_name: changed_value})
    assert getattr(changed, field_name) is changed_value
    assert changed._promotion_capture_sha256 == trajectory._promotion_capture_sha256

    _assert_rejection(
        app,
        changed,
        SessionPromotionErrorCode.INPUT_EVIDENCE_INCONSISTENT,
    )


def test_promotion_rejects_attestation_transplanted_between_same_input_captures():
    async def scenario():
        return await _run_repeatable_trajectories(
            (
                ("attestation-donor", "assistant", "same production input"),
                ("detached-target", "assistant", "same production input"),
            )
        )

    app, trajectories = asyncio.run(scenario())
    donor = trajectories["attestation-donor"]
    target = trajectories["detached-target"]
    candidate = build_promotion_candidate(
        app,
        target,
        target_key="assistant",
        source_agent_name="assistant",
        application_release_id="release-attestation-transplant",
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
    )
    detached_target = Trajectory.model_validate(target.model_dump(mode="python"))
    assert detached_target.initial_input_message_count is None
    transplanted = donor.model_copy(
        update={
            field_name: getattr(detached_target, field_name)
            for field_name in Trajectory.model_fields
        }
    )
    assert transplanted.session is not None
    assert transplanted.session.id == "detached-target"
    assert transplanted.initial_input_messages_sha256 == donor.initial_input_messages_sha256

    _assert_rejection(
        app,
        transplanted,
        SessionPromotionErrorCode.INPUT_EVIDENCE_INCONSISTENT,
    )
    with pytest.raises(SessionPromotionError) as candidate_error:
        build_promotion_candidate(
            app,
            transplanted,
            target_key="assistant",
            source_agent_name="assistant",
            application_release_id="release-attestation-transplant",
            evidence_policy=EvaluationEvidencePolicySpec.standard(),
        )
    assert candidate_error.value.code is SessionPromotionErrorCode.INPUT_EVIDENCE_INCONSISTENT
    with pytest.raises(SessionPromotionError) as score_error:
        score_promotion_candidate(
            app,
            transplanted,
            candidate,
            target_key="assistant",
            source_agent_name="assistant",
            application_release_id="release-attestation-transplant",
        )
    assert score_error.value.code is SessionPromotionErrorCode.INPUT_EVIDENCE_INCONSISTENT


def test_promotion_rejects_malformed_copied_trajectory_as_invalid():
    async def scenario():
        return await _run_trajectory(InMemorySessionStore())

    app, trajectory = asyncio.run(scenario())
    candidate = build_promotion_candidate(
        app,
        trajectory,
        target_key="assistant",
        source_agent_name="assistant",
        application_release_id="release-malformed-trajectory",
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
    )
    malformed = trajectory.model_copy(update={"transcript": ("not-a-message",)})

    _assert_rejection(
        app,
        malformed,
        SessionPromotionErrorCode.INVALID_TRAJECTORY,
    )
    with pytest.raises(SessionPromotionError) as captured:
        build_promotion_candidate(
            app,
            malformed,
            target_key="assistant",
            source_agent_name="assistant",
            application_release_id="release-malformed-trajectory",
            evidence_policy=EvaluationEvidencePolicySpec.standard(),
        )
    assert captured.value.code is SessionPromotionErrorCode.INVALID_TRAJECTORY
    with pytest.raises(SessionPromotionError) as score_error:
        score_promotion_candidate(
            app,
            malformed,
            candidate,
            target_key="assistant",
            source_agent_name="assistant",
            application_release_id="release-malformed-trajectory",
        )
    assert score_error.value.code is SessionPromotionErrorCode.INVALID_TRAJECTORY


def test_failed_session_input_is_still_eligible_for_a_regression_case():
    async def scenario():
        app, trajectory = await _run_trajectory(InMemorySessionStore(), fail=True)
        return app, trajectory

    app, trajectory = asyncio.run(scenario())
    assert trajectory.session is not None
    assert trajectory.session.status == SessionStatus.FAILED
    promoted = promotable_run_input(app, trajectory, source_agent_name="assistant")
    assert [message.text for message in promoted.messages] == ["promote this run"]
    candidate = build_promotion_candidate(
        app,
        trajectory,
        target_key="assistant",
        source_agent_name="assistant",
        application_release_id="release-failed",
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
    )
    assert candidate.warnings == (PromotionWarningCode.SOURCE_RUN_FAILED,)
    forged = candidate.model_copy(update={"warnings": ()})
    with pytest.raises(ValueError, match="warnings do not match"):
        PromotionCandidateV1.model_validate(forged.model_dump(mode="python"))
    assert len(candidate.case.assertions) == 1
    default_assertion = candidate.case.assertions[0]
    assert type(default_assertion) is RootStatusAssertionSpec
    assert default_assertion.expected == "completed"
    score = score_promotion_candidate(
        app,
        trajectory,
        candidate,
        target_key="assistant",
        source_agent_name="assistant",
        application_release_id="release-failed",
    )
    assert score.status == "failed"
    assert score.score == 0.0
    assert score.assertions[0].outcome == "failed"


def test_candidate_projection_is_deterministic_editable_and_identity_free():
    async def scenario():
        return await _run_trajectory(
            InMemorySessionStore(),
            session_id="private-session-must-not-leak",
        )

    app, trajectory = asyncio.run(scenario())
    policy = EvaluationEvidencePolicySpec.standard()
    candidate = build_promotion_candidate(
        app,
        trajectory,
        target_key="assistant",
        source_agent_name="assistant",
        application_release_id="release-2026-08-05",
        evidence_policy=policy,
    )
    repeated = build_promotion_candidate(
        app,
        trajectory,
        target_key="assistant",
        source_agent_name="assistant",
        application_release_id="release-2026-08-05",
        evidence_policy=policy,
    )
    assert repeated == candidate
    assert repeated.revision == candidate.revision
    captured_input = promotable_run_input(app, trajectory, source_agent_name="assistant")
    assert candidate.source.input_revision == captured_input.revision
    assert candidate.source.input_redactions_applied is False
    assert candidate.source.evidence_revision == candidate.evidence.revision
    assert candidate.source.evidence_policy_revision == policy.revision
    assert candidate.case.source == candidate.source.case_source()
    assert candidate.case.suite_id == candidate.suite.id
    assert candidate.suite.id == "assistant.regressions"
    serialized = candidate.model_dump_json()
    assert "private-session-must-not-leak" not in serialized
    score = score_promotion_candidate(
        app,
        trajectory,
        candidate,
        target_key="assistant",
        source_agent_name="assistant",
        application_release_id="release-2026-08-05",
    )
    repeated_score = score_promotion_candidate(
        app,
        trajectory,
        candidate,
        target_key="assistant",
        source_agent_name="assistant",
        application_release_id="release-2026-08-05",
    )
    assert score == repeated_score
    assert score.status == "passed"
    assert score.score == 1.0
    assert "private-session-must-not-leak" not in score.model_dump_json()
    changed_trajectory = trajectory.model_copy(
        update={
            "transcript": (
                *trajectory.transcript[:-1],
                Message.text("assistant", "changed after preview"),
            ),
            "final_output": "changed after preview",
        }
    )
    _assert_rejection(
        app,
        changed_trajectory,
        SessionPromotionErrorCode.INPUT_EVIDENCE_INCONSISTENT,
    )
    with pytest.raises(SessionPromotionError) as changed_candidate_error:
        build_promotion_candidate(
            app,
            changed_trajectory,
            target_key="assistant",
            source_agent_name="assistant",
            application_release_id="release-2026-08-05",
            evidence_policy=policy,
        )
    assert (
        changed_candidate_error.value.code is SessionPromotionErrorCode.INPUT_EVIDENCE_INCONSISTENT
    )
    with pytest.raises(SessionPromotionError) as changed_evidence_error:
        score_promotion_candidate(
            app,
            changed_trajectory,
            candidate,
            target_key="assistant",
            source_agent_name="assistant",
            application_release_id="release-2026-08-05",
        )
    assert (
        changed_evidence_error.value.code is SessionPromotionErrorCode.INPUT_EVIDENCE_INCONSISTENT
    )

    changed_transcript = list(trajectory.transcript)
    assert changed_transcript[1].role is MessageRole.USER
    changed_transcript[1] = Message.text("user", "changed after preview")
    changed_input = trajectory.model_copy(update={"transcript": tuple(changed_transcript)})
    with pytest.raises(SessionPromotionError) as changed_input_error:
        score_promotion_candidate(
            app,
            changed_input,
            candidate,
            target_key="assistant",
            source_agent_name="assistant",
            application_release_id="release-2026-08-05",
        )
    assert changed_input_error.value.code is SessionPromotionErrorCode.INPUT_EVIDENCE_INCONSISTENT

    restored = PromotionCandidateV1.model_validate_json(serialized)
    assert restored == candidate
    edited_input = type(candidate.case.input)(
        messages=(type(candidate.case.input.messages[0])(text="operator-edited replay input"),)
    )
    edited_case = type(candidate.case).create(
        id=candidate.case.id,
        suite_id=candidate.case.suite_id,
        name="Operator-edited regression",
        description=candidate.case.description,
        source=candidate.case.source,
        input=edited_input,
        assertions=candidate.case.assertions,
    )
    edited = PromotionCandidateV1.create(
        target_key=candidate.target_key,
        source=candidate.source,
        evidence_policy=candidate.evidence_policy,
        pricing_profile=candidate.pricing_profile,
        evidence=candidate.evidence,
        suite=candidate.suite,
        case=edited_case,
    )
    assert edited.case.id == candidate.case.id
    assert edited.case.revision != candidate.case.revision
    assert edited.revision != candidate.revision
    edited_score = score_promotion_candidate(
        app,
        trajectory,
        edited,
        target_key="assistant",
        source_agent_name="assistant",
        application_release_id="release-2026-08-05",
    )
    assert edited_score.status == "passed"

    forged_revision = candidate.model_copy(update={"revision": "sha256:" + "0" * 64})
    with pytest.raises(ValueError, match="revision does not match"):
        PromotionCandidateV1.model_validate(forged_revision.model_dump(mode="python"))

    forged_warning = candidate.model_copy(
        update={"warnings": (PromotionWarningCode.SOURCE_RUN_FAILED,)}
    )
    with pytest.raises(ValueError, match="warnings do not match"):
        PromotionCandidateV1.model_validate(forged_warning.model_dump(mode="python"))

    contradictory_source = candidate.source.model_copy(
        update={"evidence_revision": "sha256:" + "0" * 64}
    )
    with pytest.raises(ValueError, match="source and evidence revisions"):
        PromotionCandidateV1.create(
            target_key=candidate.target_key,
            source=contradictory_source,
            evidence_policy=candidate.evidence_policy,
            pricing_profile=candidate.pricing_profile,
            evidence=candidate.evidence,
            suite=candidate.suite,
            case=candidate.case,
        )


def test_scoring_rejects_a_changed_application_redaction_boundary():
    async def scenario():
        app, trajectory = await _run_trajectory(InMemorySessionStore())
        changed_app, _ = await _run_trajectory(
            InMemorySessionStore(),
            session_id="redaction-control",
            messages=[Message.text("user", "unrelated control input")],
            secret_redactor=SecretRedactor("promote this run"),
        )
        return app, changed_app, trajectory

    app, changed_app, trajectory = asyncio.run(scenario())
    candidate = build_promotion_candidate(
        app,
        trajectory,
        target_key="assistant",
        source_agent_name="assistant",
        application_release_id="release-redaction-boundary",
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
    )

    assert app.describe() == changed_app.describe()
    with pytest.raises(ValueError, match="input evidence changed"):
        score_promotion_candidate(
            changed_app,
            trajectory,
            candidate,
            target_key="assistant",
            source_agent_name="assistant",
            application_release_id="release-redaction-boundary",
        )


def test_scoring_rejects_a_different_source_agent_with_equal_evidence():
    async def scenario():
        return await _run_repeatable_trajectories(
            (
                ("session-agent-a", "agent-a", "same input"),
                ("session-agent-b", "agent-b", "same input"),
            )
        )

    app, trajectories = asyncio.run(scenario())
    candidate = build_promotion_candidate(
        app,
        trajectories["session-agent-a"],
        target_key="shared-target",
        source_agent_name="agent-a",
        application_release_id="release",
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
    )
    assert candidate.source.source_agent_name == "agent-a"

    with pytest.raises(ValueError, match="source agent does not match"):
        score_promotion_candidate(
            app,
            trajectories["session-agent-b"],
            candidate,
            target_key="shared-target",
            source_agent_name="agent-b",
            application_release_id="release",
        )


def test_different_captured_inputs_get_distinct_default_case_ids():
    async def scenario():
        return await _run_repeatable_trajectories(
            (
                ("first-session", "agent", "first input"),
                ("second-session", "agent", "second input"),
            )
        )

    app, trajectories = asyncio.run(scenario())
    candidates = tuple(
        build_promotion_candidate(
            app,
            trajectories[session_id],
            target_key="shared-target",
            source_agent_name="agent",
            application_release_id="release",
            evidence_policy=EvaluationEvidencePolicySpec.standard(),
        )
        for session_id in ("first-session", "second-session")
    )

    assert candidates[0].evidence == candidates[1].evidence
    assert candidates[0].source.input_revision != candidates[1].source.input_revision
    assert candidates[0].case.id != candidates[1].case.id
    assert candidates[0].case.revision != candidates[1].case.revision


def test_long_target_key_uses_a_bounded_deterministic_suite_id():
    async def scenario():
        return await _run_trajectory(InMemorySessionStore())

    app, trajectory = asyncio.run(scenario())
    candidate = build_promotion_candidate(
        app,
        trajectory,
        target_key="a" * 128,
        source_agent_name="assistant",
        application_release_id="release-2026-08-05",
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
    )
    assert candidate.suite.id.startswith("regressions-")
    assert len(candidate.suite.id) <= 128


def test_promotion_export_is_canonical_across_processes_and_omits_preview_evidence():
    async def scenario():
        return await _run_trajectory(
            InMemorySessionStore(),
            session_id="private-export-session",
        )

    app, trajectory = asyncio.run(scenario())
    candidate = build_promotion_candidate(
        app,
        trajectory,
        target_key="assistant",
        source_agent_name="assistant",
        application_release_id="release-export",
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
        source_label="Exported regression",
    )
    corpus = corpus_from_promotion_candidate(candidate)
    exported = export_promotion_corpus(candidate)
    assert eval_corpus_from_json(exported.decode("utf-8")) == corpus
    assert exported.endswith(b"\n")
    assert b"private-export-session" not in exported
    assert b"captured answer" not in exported
    assert b'"warnings"' not in exported
    assert b'"source_label"' not in exported

    repo_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    source_path = str(repo_root / "src")
    existing_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_path if not existing_python_path else source_path + os.pathsep + existing_python_path
    )
    script = (
        "import sys\n"
        "from cayu import PromotionCandidateV1, export_promotion_corpus\n"
        "candidate = PromotionCandidateV1.model_validate_json(sys.stdin.read())\n"
        "sys.stdout.buffer.write(export_promotion_corpus(candidate))\n"
    )
    process = subprocess.run(
        [sys.executable, "-c", script],
        input=candidate.model_dump_json(),
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
        env=environment,
        cwd=repo_root,
    )
    assert process.stdout.encode("utf-8") == exported

    probe = Path(__file__).with_name("promotion_determinism_probe.py")
    probe_outputs = tuple(
        subprocess.run(
            [sys.executable, str(probe)],
            capture_output=True,
            check=True,
            timeout=30,
            env=environment,
            cwd=repo_root,
        ).stdout
        for _ in range(2)
    )
    assert probe_outputs[0] == probe_outputs[1]
    independent_document = json.loads(probe_outputs[0])
    assert independent_document["candidate"]["revision"].startswith("sha256:")
    assert independent_document["score"]["revision"].startswith("sha256:")
    assert independent_document["corpus"].endswith("\n")

    cost_case = type(candidate.case).create(
        id=candidate.case.id,
        suite_id=candidate.case.suite_id,
        name=candidate.case.name,
        description=candidate.case.description,
        source=candidate.case.source,
        input=candidate.case.input,
        assertions=(
            *candidate.case.assertions,
            MaxEstimatedCostAssertionSpec(
                id="missing-price",
                maximum="1",
                currency="USD",
            ),
        ),
    )
    unpriced_candidate = PromotionCandidateV1.create(
        target_key=candidate.target_key,
        source=candidate.source,
        evidence_policy=candidate.evidence_policy,
        evidence=candidate.evidence,
        suite=candidate.suite,
        case=cost_case,
    )
    with pytest.raises(ValueError, match="Cost assertions require a pricing profile"):
        export_promotion_corpus(unpriced_candidate)


def test_descriptive_label_and_pricing_do_not_change_default_case_identity():
    async def scenario():
        return await _run_trajectory(InMemorySessionStore())

    app, trajectory = asyncio.run(scenario())
    policy = EvaluationEvidencePolicySpec.standard()
    plain = build_promotion_candidate(
        app,
        trajectory,
        target_key="assistant",
        source_agent_name="assistant",
        application_release_id="release-2026-08-05",
        evidence_policy=policy,
    )
    described = build_promotion_candidate(
        app,
        trajectory,
        target_key="assistant",
        source_agent_name="assistant",
        application_release_id="release-2026-08-05",
        evidence_policy=policy,
        pricing=_price_book(),
        source_label="Refund approval regression",
    )
    assert described.case.id == plain.case.id
    assert described.case.name == "Refund approval regression"
    assert described.pricing_profile is not None
    assert described.source.pricing_profile_fingerprint == described.pricing_profile.fingerprint


def test_captured_cost_scoring_requires_the_reviewed_pricing_profile():
    async def scenario():
        return await _run_trajectory(InMemorySessionStore())

    app, trajectory = asyncio.run(scenario())
    pricing = _price_book()
    candidate = build_promotion_candidate(
        app,
        trajectory,
        target_key="assistant",
        source_agent_name="assistant",
        application_release_id="release-priced",
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
        pricing=pricing,
    )
    priced_case = type(candidate.case).create(
        id=candidate.case.id,
        suite_id=candidate.case.suite_id,
        name=candidate.case.name,
        description=candidate.case.description,
        source=candidate.case.source,
        input=candidate.case.input,
        assertions=(
            *candidate.case.assertions,
            MaxEstimatedCostAssertionSpec(
                id="cost-budget",
                maximum="1",
                currency="USD",
            ),
        ),
    )
    priced_candidate = PromotionCandidateV1.create(
        target_key=candidate.target_key,
        source=candidate.source,
        evidence_policy=candidate.evidence_policy,
        pricing_profile=candidate.pricing_profile,
        evidence=candidate.evidence,
        suite=candidate.suite,
        case=priced_case,
    )
    scored = score_promotion_candidate(
        app,
        trajectory,
        priced_candidate,
        target_key="assistant",
        source_agent_name="assistant",
        application_release_id="release-priced",
        pricing=pricing,
    )
    assert scored.status == "passed"
    assert scored.score == 1.0
    assert candidate.pricing_profile is not None
    assert scored.pricing_profile_fingerprint == candidate.pricing_profile.fingerprint

    missing_pricing = score_promotion_candidate(
        app,
        trajectory,
        priced_candidate,
        target_key="assistant",
        source_agent_name="assistant",
        application_release_id="release-priced",
    )
    assert missing_pricing.status == "unavailable"
    assert missing_pricing.score is None
    assert missing_pricing.assertions[-1].outcome == "unavailable"

    with pytest.raises(ValueError, match="pricing profile no longer matches"):
        score_promotion_candidate(
            app,
            trajectory,
            priced_candidate,
            target_key="assistant",
            source_agent_name="assistant",
            application_release_id="release-priced",
            pricing=_price_book(input_rate="2"),
        )


@pytest.mark.parametrize(
    "field_name",
    ["target_key", "source_agent_name", "application_release_id", "source_label"],
)
def test_candidate_diagnostic_text_rejects_workload_secrets(field_name):
    secret = "assistant" if field_name == "source_agent_name" else "candidate-diagnostic-secret"

    async def scenario():
        app, trajectory = await _run_trajectory(
            InMemorySessionStore(),
            secret_redactor=(None if field_name == "source_agent_name" else SecretRedactor(secret)),
        )
        if field_name == "source_agent_name":
            app = CayuApp(
                secret_redactor=SecretRedactor(secret),
                enable_logging=False,
            )
        return app, trajectory

    app, trajectory = asyncio.run(scenario())
    target_key = "assistant"
    if field_name == "target_key":
        target_key = secret
    elif field_name == "source_agent_name":
        target_key = "safe-target"
    source_agent_name = secret if field_name == "source_agent_name" else "assistant"
    with pytest.raises(ValueError, match="workload secret"):
        build_promotion_candidate(
            app,
            trajectory,
            target_key=target_key,
            source_agent_name=source_agent_name,
            application_release_id=(
                secret if field_name == "application_release_id" else "safe-release"
            ),
            evidence_policy=EvaluationEvidencePolicySpec.standard(),
            source_label=secret if field_name == "source_label" else None,
        )


@pytest.mark.parametrize(
    ("field_name", "secret"),
    [
        ("price_book_version", "private-pricing-identity"),
        ("generated_at", "private-pricing-identity"),
        ("currency", "ZZZ"),
    ],
)
def test_candidate_pricing_identity_rejects_workload_secrets(field_name, secret):

    async def scenario():
        return await _run_trajectory(
            InMemorySessionStore(),
            secret_redactor=SecretRedactor(secret),
        )

    app, trajectory = asyncio.run(scenario())
    if field_name == "currency":
        pricing = _price_book(currency=secret)
    else:
        pricing = _price_book()
        pricing = pricing.model_copy(update={field_name: secret})
    with pytest.raises(ValueError, match="workload secret"):
        build_promotion_candidate(
            app,
            trajectory,
            target_key="assistant",
            source_agent_name="assistant",
            application_release_id="safe-release",
            evidence_policy=EvaluationEvidencePolicySpec.standard(),
            pricing=pricing,
        )


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

    promoted = promotable_run_input(
        app,
        _completed_child_tree(trajectory),
        source_agent_name="assistant",
    )
    assert [message.text for message in promoted.messages] == ["promote this run"]


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

    policy = EvaluationEvidencePolicySpec.standard()
    candidate = build_captured_evaluation_candidate(
        app,
        trajectory,
        target_key="assistant",
        source_agent_name="assistant",
        application_release_id="structured-release",
        evidence_policy=policy,
    )
    assert candidate.case.input is None
    score = score_captured_evaluation_candidate(
        app,
        trajectory,
        candidate,
        target_key="assistant",
        source_agent_name="assistant",
        application_release_id="structured-release",
    )
    assert score.status == "passed"
    corpus = corpus_from_captured_evaluation_candidate(candidate)
    assert corpus.cases[0].input is None


def test_captured_failed_session_defaults_to_a_regression_to_fix():
    async def scenario():
        return await _run_trajectory(InMemorySessionStore(), fail=True)

    app, trajectory = asyncio.run(scenario())
    candidate = build_captured_evaluation_candidate(
        app,
        trajectory,
        target_key="assistant",
        source_agent_name="assistant",
        application_release_id="failed-release",
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
    )
    assertion = candidate.case.assertions[0]
    assert isinstance(assertion, RootStatusAssertionSpec)
    assert assertion.expected == SessionStatus.COMPLETED
    assert candidate.warnings == ("source_run_failed",)
    score = score_captured_evaluation_candidate(
        app,
        trajectory,
        candidate,
        target_key="assistant",
        source_agent_name="assistant",
        application_release_id="failed-release",
    )
    assert score.status == "failed"


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


def test_caller_system_input_without_runtime_bootstrap_uses_role_rejection(
    promotion_store_case,
):
    async def scenario():
        store = await _open_store(promotion_store_case)
        try:
            return await _run_trajectory(
                store,
                messages=[Message.text("system", "caller-authored system state")],
                agent_system_prompt=None,
                fail=False,
            )
        finally:
            await _close_store(store)

    app, trajectory = asyncio.run(scenario())
    assert [message.role for message in trajectory.transcript] == [
        MessageRole.SYSTEM,
        MessageRole.ASSISTANT,
    ]
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
                    SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY: (
                        "v1:0:1:original:text:sha256:" + "0" * 64
                    ),
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
    assert trajectory.initial_input_messages_sha256 is None
    assert trajectory.structured_output_requested is None
    assert trajectory.input_redactions_applied is None
    _assert_rejection(
        app,
        trajectory,
        SessionPromotionErrorCode.INPUT_EVIDENCE_UNAVAILABLE,
    )


def test_sql_input_contract_requires_persisted_runtime_proof(promotion_store_case):
    kind, tmp_path, postgres_dsn = promotion_store_case
    if kind == "memory":
        pytest.skip("In-memory authority stays attached to the Event instance.")

    async def scenario():
        store = await _open_store(promotion_store_case)
        _app, trusted = await _run_trajectory(store)
        assert trusted.initial_input_message_count == 1
        await _close_store(store)

        if kind == "sqlite":
            import sqlite3

            connection = sqlite3.connect(tmp_path / "session-promotion.sqlite")
            try:
                proof = connection.execute(
                    "SELECT input_contract_runtime_owned FROM cayu_events "
                    "WHERE session_id = ? AND event_type = 'session.started'",
                    ("promotion-root",),
                ).fetchone()
                assert proof == (1,)
                connection.execute(
                    "UPDATE cayu_events SET input_contract_runtime_owned = 0 "
                    "WHERE session_id = ? AND event_type = 'session.started'",
                    ("promotion-root",),
                )
                connection.commit()
            finally:
                connection.close()
            store = SQLiteSessionStore(tmp_path / "session-promotion.sqlite")
        else:
            import psycopg

            async with (
                await psycopg.AsyncConnection.connect(postgres_dsn) as connection,
                connection.cursor() as cursor,
            ):
                await cursor.execute(
                    "SELECT input_contract_runtime_owned FROM cayu_events "
                    "WHERE session_id = %s AND event_type = 'session.started'",
                    ("promotion-root",),
                )
                assert await cursor.fetchone() == (True,)
                await cursor.execute(
                    "UPDATE cayu_events SET input_contract_runtime_owned = FALSE "
                    "WHERE session_id = %s AND event_type = 'session.started'",
                    ("promotion-root",),
                )
                await connection.commit()
            store = PostgresSessionStore(
                postgres_dsn,
                min_size=1,
                max_size=4,
                schema_mode=SchemaMode.CREATE,
            )

        try:
            reloaded_app = CayuApp(session_store=store, enable_logging=False)
            untrusted = await trajectory_from_session(reloaded_app, "promotion-root")
            return reloaded_app, untrusted
        finally:
            await _close_store(store)

    app, trajectory = asyncio.run(scenario())
    assert trajectory.initial_input_message_start_index is None
    assert trajectory.initial_input_message_count is None
    assert trajectory.initial_input_messages_sha256 is None
    _assert_rejection(
        app,
        trajectory,
        SessionPromotionErrorCode.INPUT_EVIDENCE_UNAVAILABLE,
    )


def test_sqlite_ordinary_event_batch_skips_input_contract_proof_updates(tmp_path):
    async def scenario() -> list[str]:
        store = SQLiteSessionStore(tmp_path / "ordinary-events.sqlite")
        try:
            session_id = "ordinary-event-batch"
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
            statements: list[str] = []
            store._connection.set_trace_callback(statements.append)
            try:
                await store.append_events(
                    session_id,
                    [
                        Event(
                            id=f"ordinary-event-{index}",
                            type=EventType.MODEL_TEXT_DELTA,
                            session_id=session_id,
                            payload={"delta": "x"},
                        )
                        for index in range(32)
                    ],
                )
            finally:
                store._connection.set_trace_callback(None)
            return statements
        finally:
            await store.close()

    statements = asyncio.run(scenario())
    proof_updates = [
        statement
        for statement in statements
        if "UPDATE cayu_events" in statement and "input_contract_runtime_owned" in statement
    ]
    assert proof_updates == []


def test_postgres_ordinary_event_batch_skips_input_contract_proof_update():
    class RecordingCursor:
        def __init__(self) -> None:
            self.statements: list[str] = []

        async def execute(self, statement, params) -> None:
            del params
            self.statements.append(str(statement))

    async def scenario() -> list[str]:
        cursor = RecordingCursor()
        await PostgresSessionStore._enqueue_persisted_event_side_effects(
            cursor,
            "ordinary-event-batch",
            [
                Event(
                    id=f"ordinary-event-{index}",
                    type=EventType.MODEL_TEXT_DELTA,
                    session_id="ordinary-event-batch",
                    payload={"delta": "x"},
                )
                for index in range(32)
            ],
        )
        return cursor.statements

    statements = asyncio.run(scenario())
    assert len(statements) == 1
    assert "INSERT INTO cayu_persisted_event_side_effects" in statements[0]
    assert "UPDATE cayu_events" not in statements[0]


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
    ],
)
def test_promotion_rejects_nonportable_phases_in_descendants(event_type, expected):
    async def scenario():
        app, trajectory = await _run_trajectory(InMemorySessionStore())
        return app, _completed_child_tree(trajectory, event_type)

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
    assert restored.initial_input_messages_sha256 is None
    assert restored.structured_output_requested is None
    assert restored.input_redactions_applied is None
    assert restored._promotion_capture_sha256 is None
    _assert_rejection(
        app,
        restored,
        SessionPromotionErrorCode.INPUT_EVIDENCE_UNAVAILABLE,
    )
