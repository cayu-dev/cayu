from __future__ import annotations

import pytest

from cayu.providers._stream_lifecycle import (
    StreamLifecycle,
    StreamLifecycleError,
    StreamPhase,
    StreamPolicy,
    StreamTerminal,
    StreamTransition,
    StreamViolation,
)
from cayu.providers._stream_lifecycle import (
    StreamTransitionKind as Kind,
)
from cayu.providers.deadlines import ProviderProgressKind


def terminal(metadata: bytes = b"stop") -> StreamTransition:
    return StreamTransition(Kind.TERMINAL, terminal=StreamTerminal("stop", metadata))


@pytest.mark.parametrize("require_start", [False, True])
def test_clean_stream_has_one_terminal_and_one_completion(require_start: bool) -> None:
    lifecycle = StreamLifecycle(StreamPolicy(require_start=require_start))
    assert lifecycle.phase is StreamPhase.PRE_START
    assert lifecycle.accept(StreamTransition(Kind.IGNORE)).phase is StreamPhase.PRE_START
    started = lifecycle.accept(StreamTransition(Kind.START, response_id="response", model="model"))
    assert started.identity_started
    assert started.phase is StreamPhase.ACTIVE
    accepted = lifecycle.accept(
        StreamTransition(Kind.SEMANTIC, progress=ProviderProgressKind.CONTENT)
    )
    assert accepted.progress is ProviderProgressKind.CONTENT
    assert lifecycle.accept(terminal()).terminal_started
    assert lifecycle.phase is StreamPhase.TERMINAL
    lifecycle.claim_completion()
    with pytest.raises(StreamLifecycleError) as failure:
        lifecycle.claim_completion()
    assert failure.value.violation is StreamViolation.REPEATED_COMPLETION


@pytest.mark.parametrize("kind", [Kind.SEMANTIC, Kind.USAGE, Kind.METADATA, Kind.TERMINAL])
def test_strict_start_rejection_does_not_bind_identity(kind: Kind) -> None:
    lifecycle = StreamLifecycle(StreamPolicy(require_start=True))
    transition = StreamTransition(
        kind,
        response_id="unaccepted",
        model="unaccepted",
        terminal=StreamTerminal("stop") if kind is Kind.TERMINAL else None,
    )
    with pytest.raises(StreamLifecycleError) as failure:
        lifecycle.accept(transition)
    assert failure.value.violation is StreamViolation.BEFORE_START
    assert lifecycle.phase is StreamPhase.PRE_START
    assert lifecycle.response_id is None
    assert lifecycle.model is None
    lifecycle.accept(StreamTransition(Kind.START, response_id="accepted", model="accepted"))


@pytest.mark.parametrize("first", [Kind.START, Kind.SEMANTIC, Kind.TERMINAL])
def test_start_cannot_restart_an_active_or_terminal_response(first: Kind) -> None:
    lifecycle = StreamLifecycle(StreamPolicy())
    lifecycle.accept(terminal() if first is Kind.TERMINAL else StreamTransition(first))
    with pytest.raises(StreamLifecycleError) as failure:
        lifecycle.accept(StreamTransition(Kind.START))
    assert failure.value.violation is (
        StreamViolation.AFTER_TERMINAL if first is Kind.TERMINAL else StreamViolation.REPEATED_START
    )


@pytest.mark.parametrize("field", ["response_id", "model"])
def test_identity_conflict_is_atomic_and_omission_does_not_unfreeze(field: str) -> None:
    lifecycle = StreamLifecycle(StreamPolicy(max_usage_tails=1, max_total_tails=1))
    lifecycle.accept(StreamTransition(Kind.START, response_id="response", model="model"))
    lifecycle.accept(StreamTransition(Kind.SEMANTIC))
    lifecycle.accept(terminal())
    with pytest.raises(StreamLifecycleError) as failure:
        lifecycle.accept(StreamTransition(Kind.USAGE, **{field: "foreign"}))
    assert failure.value.violation is (
        StreamViolation.RESPONSE_CONFLICT
        if field == "response_id"
        else StreamViolation.MODEL_CONFLICT
    )
    assert lifecycle.response_id == "response"
    assert lifecycle.model == "model"
    # Rejection did not spend the one permitted tail or overwrite identity.
    lifecycle.accept(StreamTransition(Kind.USAGE, response_id="response", model="model"))


@pytest.mark.parametrize("first_field", ["response_id", "model"])
def test_conflicting_second_field_cannot_commit_first_field(first_field: str) -> None:
    existing = "model" if first_field == "response_id" else "response_id"
    lifecycle = StreamLifecycle(StreamPolicy())
    lifecycle.accept(StreamTransition(Kind.START, **{existing: "accepted"}))
    with pytest.raises(StreamLifecycleError):
        lifecycle.accept(
            StreamTransition(Kind.SEMANTIC, **{existing: "foreign", first_field: "new"})
        )
    assert getattr(lifecycle, first_field) is None


def test_exact_terminal_repeat_uses_complete_immutable_metadata_without_progress() -> None:
    lifecycle = StreamLifecycle(StreamPolicy(max_terminal_repeats=1))
    lifecycle.accept(terminal(b'{"reason":"stop","metadata":1}'))
    with pytest.raises(StreamLifecycleError) as failure:
        lifecycle.accept(terminal(b'{"reason":"stop","metadata":2}'))
    assert failure.value.violation is StreamViolation.TERMINAL_CONFLICT
    repeated = lifecycle.accept(terminal(b'{"reason":"stop","metadata":1}'))
    assert repeated.repeated
    assert not repeated.terminal_started
    assert repeated.progress is None
    with pytest.raises(StreamLifecycleError) as failure:
        lifecycle.accept(terminal(b'{"reason":"stop","metadata":1}'))
    assert failure.value.violation is StreamViolation.REPEATED_TERMINAL


@pytest.mark.parametrize("kind", [Kind.START, Kind.IDENTITY, Kind.SEMANTIC])
def test_semantic_or_identity_transition_cannot_follow_terminal(kind: Kind) -> None:
    lifecycle = StreamLifecycle(StreamPolicy())
    lifecycle.accept(terminal())
    with pytest.raises(StreamLifecycleError) as failure:
        lifecycle.accept(StreamTransition(kind))
    assert failure.value.violation is StreamViolation.AFTER_TERMINAL


def test_tail_limits_compose_and_rejection_does_not_bind_identity() -> None:
    lifecycle = StreamLifecycle(
        StreamPolicy(max_usage_tails=1, max_metadata_tails=1, max_total_tails=1)
    )
    lifecycle.accept(terminal())
    lifecycle.accept(StreamTransition(Kind.USAGE, progress=ProviderProgressKind.USAGE))
    with pytest.raises(StreamLifecycleError) as failure:
        lifecycle.accept(StreamTransition(Kind.METADATA, response_id="late"))
    assert failure.value.violation is StreamViolation.TAIL_LIMIT
    assert lifecycle.response_id is None


def test_metadata_limit_includes_preterminal_metadata() -> None:
    lifecycle = StreamLifecycle(
        StreamPolicy(max_metadata_tails=1, max_total_tails=1, max_metadata_events=1)
    )
    lifecycle.accept(StreamTransition(Kind.METADATA))
    lifecycle.accept(terminal())
    with pytest.raises(StreamLifecycleError) as failure:
        lifecycle.accept(StreamTransition(Kind.METADATA))
    assert failure.value.violation is StreamViolation.METADATA_LIMIT


def test_reconstructed_model_identity_rejects_conflict_without_replacing_model() -> None:
    lifecycle = StreamLifecycle(
        StreamPolicy(require_start=True),
        resumed_response_id="operation",
        resumed_model="original-model",
    )
    with pytest.raises(StreamLifecycleError) as failure:
        lifecycle.accept(StreamTransition(Kind.SEMANTIC, model="foreign-model"))
    assert failure.value.violation is StreamViolation.MODEL_CONFLICT
    assert lifecycle.model == "original-model"
    lifecycle.accept(StreamTransition(Kind.SEMANTIC, model="original-model"))
    assert lifecycle.model == "original-model"


@pytest.mark.parametrize("value", [True, 1, [], {}, "", " ", "\ud800"])
def test_invalid_identities_are_content_free(value: object, capsys, caplog) -> None:
    lifecycle = StreamLifecycle(StreamPolicy())
    with pytest.raises(StreamLifecycleError) as failure:
        lifecycle.accept(StreamTransition(Kind.START, response_id=value))
    assert failure.value.violation is StreamViolation.INVALID_IDENTITY
    assert lifecycle.phase is StreamPhase.PRE_START
    assert capsys.readouterr() == ("", "")
    assert not caplog.records


def test_hostile_identity_is_not_formatted_or_serialized(capsys, caplog) -> None:
    class HostileIdentity:
        def __repr__(self) -> str:
            raise AssertionError("SECRET_CANARY")

        def __str__(self) -> str:
            raise AssertionError("SECRET_CANARY")

    lifecycle = StreamLifecycle(StreamPolicy())
    with pytest.raises(StreamLifecycleError) as failure:
        lifecycle.accept(StreamTransition(Kind.START, model=HostileIdentity()))
    assert "SECRET_CANARY" not in str(failure.value)
    assert "SECRET_CANARY" not in repr(failure.value)
    assert capsys.readouterr() == ("", "")
    assert not caplog.records


@pytest.mark.parametrize(
    "transition",
    [
        StreamTransition(Kind.TERMINAL),
        StreamTransition(Kind.SEMANTIC, terminal=StreamTerminal("stop")),
        StreamTransition(Kind.IGNORE, response_id="discarded"),
        StreamTransition(Kind.IGNORE, progress=ProviderProgressKind.CONTENT),
        StreamTransition(Kind.SEMANTIC, progress=ProviderProgressKind.TERMINAL),
    ],
)
def test_invalid_transition_shapes_cannot_mutate_state(transition: StreamTransition) -> None:
    lifecycle = StreamLifecycle(StreamPolicy())
    with pytest.raises(StreamLifecycleError) as failure:
        lifecycle.accept(transition)
    assert failure.value.violation is StreamViolation.INVALID_TRANSITION
    assert lifecycle.phase is StreamPhase.PRE_START


def test_missing_terminal_cannot_claim_completion() -> None:
    lifecycle = StreamLifecycle(StreamPolicy())
    with pytest.raises(StreamLifecycleError) as failure:
        lifecycle.claim_completion()
    assert failure.value.violation is StreamViolation.MISSING_TERMINAL


def test_native_failure_can_precede_start_but_cannot_follow_terminal() -> None:
    lifecycle = StreamLifecycle(StreamPolicy(require_start=True))
    lifecycle.accept(StreamTransition(Kind.FAILURE))
    assert lifecycle.phase is StreamPhase.PRE_START
    lifecycle.accept(StreamTransition(Kind.START))
    lifecycle.accept(terminal())
    with pytest.raises(StreamLifecycleError) as failure:
        lifecycle.accept(StreamTransition(Kind.FAILURE))
    assert failure.value.violation is StreamViolation.AFTER_TERMINAL


@pytest.mark.parametrize("value", [True, -1, 1.5, "1"])
def test_policy_limits_require_exact_nonnegative_integers(value: object) -> None:
    with pytest.raises(ValueError):
        StreamPolicy(max_terminal_repeats=value)


def test_separate_streams_do_not_share_identity_or_completion() -> None:
    for _ in range(2):
        lifecycle = StreamLifecycle(StreamPolicy())
        lifecycle.accept(StreamTransition(Kind.START, response_id="same-id"))
        lifecycle.accept(terminal())
        lifecycle.claim_completion()
