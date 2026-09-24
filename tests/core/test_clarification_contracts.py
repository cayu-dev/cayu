"""Clarification value/election characterization; not public service qualification."""

import pytest

from cayu.collaboration._clarification_state import (
    ClarificationQuestionState,
    accept_clarification_reply,
    clarification_commitment,
)
from cayu.collaboration._contracts import (
    CollaborationConflict,
    InitiatorBinding,
    ObjectRef,
    OperationRef,
    OwnerRef,
)
from cayu.collaboration._preparation import prepare_contract
from cayu.collaboration.clarifications import (
    ClarificationLineageUsage,
    ClarificationPolicy,
    ClarificationQuestion,
    ClarificationReply,
    ClarificationSource,
)
from cayu.collaboration.exports import SessionExportRef
from cayu.collaboration.participants import ParticipantRef
from cayu.collaboration.requests import ClarificationFrontier, RequestRef
from cayu.vaults.redaction import SecretRedactor

OWNER = OwnerRef(application_scope="app", owner_id="owner", incarnation="one")
REDACTOR = SecretRedactor()


@pytest.mark.parametrize(
    "changes",
    [
        {"generation": True},
        {"generation": 33},
        {"generation": 1},
        {"input_revision": 1},
        {"input_sha256": "a" * 64},
        {"generation": 1, "lineage": None},
    ],
)
def test_clarification_frontier_rejects_inconsistent_or_unbounded_state(changes):
    with pytest.raises(ValueError):
        ClarificationFrontier.model_validate(changes)


def operation(key):
    return OperationRef(
        application_scope="app", namespace_incarnation="namespace", generation=1, caller_key=key
    )


def reference(kind):
    return ObjectRef(owner=OWNER, kind=kind, object_id=kind, incarnation="one", revision=1)


def policy(**changes):
    return ClarificationPolicy.model_validate(
        dict(
            reference=reference("policy"),
            max_questions=32,
            max_depth=4,
            max_service_turns=32,
            max_question_bytes=16384,
            max_reply_bytes=16384,
            max_content_bytes=1048576,
            max_pending=64,
            service_timeout_ms=60000,
            max_spend_usd="1.00",
            failure_policy="return_and_report",
            return_policy="original_wait_or_terminal_exclusion",
        )
        | changes
    )


def fixture():
    responder = ParticipantRef(owner=OWNER, participant_id="responder", incarnation="one")
    initiator = InitiatorBinding(
        issuer=OWNER,
        principal="host",
        participant=reference("participant"),
        mandate=reference("mandate"),
        invocation_id="invocation",
        interaction_id="interaction",
    )
    source = ClarificationSource(
        export=SessionExportRef(
            session_id="source",
            session_instance_id="one",
            operation=operation("export"),
        ),
        export_receipt_sha256="a" * 64,
        producer=reference("turn"),
        content_sha256="b" * 64,
        content_bytes=32,
        selection="assistant_visible_text_v1",
        projector=reference("projector"),
        policy=reference("export_policy"),
        audience=reference("participant"),
    )
    question = ClarificationQuestion(
        operation=operation("question"),
        request=RequestRef(owner=OWNER, request_id="request", incarnation="one"),
        request_sha256="c" * 64,
        admission=operation("admission"),
        initiator=initiator,
        responder=responder,
        receiver=reference("receiver"),
        generation=1,
        input_revision=0,
        input_sha256="d" * 64,
        lineage=operation("lineage"),
        parent_question=None,
        depth=1,
        source=source,
        policy=policy(),
        budget_binding=reference("budget"),
        budget_authority_sha256="e" * 64,
        deadline_at_ms=200,
    )
    reply = ClarificationReply(
        operation=operation("reply"),
        question=question.operation,
        question_sha256=clarification_commitment(question, REDACTOR),
        question_generation=1,
        question_input_revision=0,
        expected_input_revision=0,
        expected_input_sha256=question.input_sha256,
        initiator=initiator,
        responder=responder,
        service=operation("service"),
        service_generation=1,
        service_session=reference("session"),
        invocation=reference("invocation"),
        admission_sha256="f" * 64,
        production_stage_id="reply-stage",
        production_sha256="a" * 64,
        source=source,
    )
    return ClarificationQuestionState(question=question, opened_at_ms=100, state="open"), reply


def accept(state, reply, **changes):
    return accept_clarification_reply(
        state,
        reply,
        **(
            dict(
                request_is_open=True,
                current_input_revision=0,
                current_input_sha256="d" * 64,
                now_ms=150,
                redactor=REDACTOR,
            )
            | changes
        ),
    )


@pytest.mark.parametrize(
    "field,maximum",
    [
        ("max_questions", 32),
        ("max_depth", 4),
        ("max_service_turns", 32),
        ("max_question_bytes", 16384),
        ("max_reply_bytes", 16384),
        ("max_content_bytes", 1048576),
        ("max_pending", 64),
    ],
)
def test_policy_has_strict_finite_ceilings(field, maximum):
    assert getattr(policy(**{field: maximum - 1}), field) == maximum - 1
    assert getattr(policy(**{field: maximum}), field) == maximum
    for invalid in (0, maximum + 1, True, str(maximum), None):
        with pytest.raises(ValueError):
            policy(**{field: invalid})


@pytest.mark.parametrize(
    "value", ["NaN", "Infinity", "-Infinity", "0", "-1", "bad", " 1", "1_0", "1e9", True, 1]
)
def test_policy_rejects_invalid_spending(value):
    with pytest.raises(ValueError):
        policy(max_spend_usd=value)


def test_reply_election_preserves_request_and_replays_after_terminal_deadline():
    original, reply = fixture()
    before = original.model_dump_json()
    answered = accept(original, reply)
    assert original.model_dump_json() == before
    assert answered.question == original.question
    assert answered.input.revision == 1
    restored = ClarificationQuestionState.model_validate_json(answered.model_dump_json())
    assert (
        accept(restored, reply, request_is_open=False, now_ms=300, current_input_revision=4)
        == answered
    )


@pytest.mark.parametrize("now", [99, 200, 201])
def test_reply_outside_admission_interval_refuses_without_changing_state(now):
    state, reply = fixture()
    before = state.model_dump_json()
    with pytest.raises(CollaborationConflict):
        accept(state, reply, now_ms=now)
    assert state.model_dump_json() == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("question", operation("other-question")),
        ("question_sha256", "0" * 64),
        ("question_generation", 2),
        ("expected_input_revision", 1),
        ("expected_input_sha256", "0" * 64),
    ],
)
def test_stale_or_changed_reply_refuses(field, value):
    state, reply = fixture()
    with pytest.raises(CollaborationConflict):
        accept(state, reply.model_copy(update={field: value}))


def test_fixed_reply_key_compares_full_service_evidence_on_replay():
    state, reply = fixture()
    answered = accept(state, reply)
    for field, value in (
        ("service_generation", 2),
        ("admission_sha256", "0" * 64),
        ("production_stage_id", "different-stage"),
        ("production_sha256", "0" * 64),
        ("invocation", reference("other_invocation")),
    ):
        with pytest.raises(CollaborationConflict):
            accept(answered, reply.model_copy(update={field: value}))


def test_terminal_parent_cannot_accept_new_reply():
    state, reply = fixture()
    with pytest.raises(CollaborationConflict):
        accept(state, reply, request_is_open=False)


@pytest.mark.parametrize("disposition", ["superseded", "cancelled", "expired", "request_terminal"])
def test_closed_question_cannot_be_reopened(disposition):
    state, reply = fixture()
    closed = ClarificationQuestionState.model_validate(
        state.model_dump()
        | {"state": disposition, "closed_at_ms": 200, "closure_operation": operation("close")}
    )
    with pytest.raises(CollaborationConflict):
        accept(closed, reply, now_ms=201)


@pytest.mark.parametrize("version", [True, "1", 1.0, 2])
def test_question_version_is_strict(version):
    state, _ = fixture()
    with pytest.raises(ValueError):
        ClarificationQuestion.model_validate(
            state.question.model_dump() | {"schema_version": version}
        )


def test_reply_before_deadline_accepts_and_input_cas_refuses_stale_operation():
    state, reply = fixture()
    assert accept(state, reply, now_ms=199).state == "answered"
    with pytest.raises(CollaborationConflict):
        accept(state, reply, current_input_revision=1)


def test_reconstructed_answer_rejects_corrupt_ancestry():
    state, reply = fixture()
    raw = accept(state, reply).model_dump(mode="json")
    raw["input"]["reply_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        ClarificationQuestionState.model_validate(raw)


def test_usage_includes_reserved_capacity_at_exact_limit():
    selected = policy(max_questions=2)
    ClarificationLineageUsage(questions=1).require_within(selected)
    ClarificationLineageUsage(questions=2).require_within(selected)
    with pytest.raises(ValueError):
        ClarificationLineageUsage(questions=3).require_within(selected)


def test_hostile_postconstruction_value_is_not_serialized(capsys, caplog):
    class Hostile:
        def __repr__(self):
            raise AssertionError("secret-canary")

        __str__ = __repr__

    value = policy().model_copy(update={"max_questions": Hostile()})
    with pytest.raises(ValueError) as caught:
        prepare_contract(ClarificationPolicy, value, redactor=REDACTOR)
    output = capsys.readouterr()
    assert "secret-canary" not in str(caught.value) + output.out + output.err + caplog.text
