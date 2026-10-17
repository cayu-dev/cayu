from __future__ import annotations

import base64
import json
from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from cayu.approvals import ResolutionActorSource
from cayu.sessions import (
    SessionAggregateFilter,
    SupervisionEventCursorInvalid,
    SupervisionEventKind,
    SupervisionEventQuery,
    decode_supervision_event_cursor,
    encode_supervision_event_cursor,
)
from cayu.sessions.supervision_events import (
    SUPERVISION_EVENT_MAX_CURSOR_BYTES,
    SUPERVISION_EVENT_MAX_LIMIT,
    SUPERVISION_EVENT_MAX_RESULT_BYTES,
    SUPERVISION_EVENT_MAX_WINDOW,
)

START = datetime(2026, 10, 1, tzinfo=UTC)
END = datetime(2026, 10, 8, tzinfo=UTC)
FINGERPRINT = "a" * 64


def _query(**changes) -> SupervisionEventQuery:
    return SupervisionEventQuery.model_validate({"start_at": START, "end_at": END, **changes})


def test_query_normalizes_window_to_utc_and_applies_bounded_defaults():
    plus_six = timezone(timedelta(hours=6))
    query = SupervisionEventQuery(
        start_at=datetime(2026, 10, 1, 6, tzinfo=plus_six),
        end_at=datetime(2026, 10, 8, 6, tzinfo=plus_six),
    )

    assert query.start_at == START
    assert query.start_at.tzinfo is UTC
    assert query.end_at == END
    assert query.sessions == SessionAggregateFilter()
    assert query.kinds == ()
    assert query.actor_sources == ()
    assert query.exclude_forked_sessions is False
    assert query.execution_profile_fingerprint is None
    assert 1 <= query.limit <= SUPERVISION_EVENT_MAX_LIMIT
    assert 1 <= query.max_bytes <= SUPERVISION_EVENT_MAX_RESULT_BYTES
    assert query.cursor is None


@pytest.mark.parametrize(
    ("start_at", "end_at", "message"),
    [
        (datetime(2026, 10, 1), END, "timezone-aware"),
        (START, datetime(2026, 10, 8), "timezone-aware"),
        (END, START, "before end_at"),
        (START, START, "before end_at"),
        (START, START + SUPERVISION_EVENT_MAX_WINDOW + timedelta(seconds=1), "cannot exceed"),
    ],
)
def test_query_rejects_naive_empty_inverted_and_oversized_windows(start_at, end_at, message):
    with pytest.raises(ValidationError, match=message):
        SupervisionEventQuery(start_at=start_at, end_at=end_at)


def test_query_accepts_the_maximum_window():
    query = SupervisionEventQuery(start_at=START, end_at=START + SUPERVISION_EVENT_MAX_WINDOW)

    assert query.end_at - query.start_at == SUPERVISION_EVENT_MAX_WINDOW


@pytest.mark.parametrize(
    "changes",
    [
        {"limit": 0},
        {"limit": SUPERVISION_EVENT_MAX_LIMIT + 1},
        {"limit": True},
        {"limit": 10.0},
        {"max_bytes": 0},
        {"max_bytes": SUPERVISION_EVENT_MAX_RESULT_BYTES + 1},
        {"max_bytes": True},
        {"exclude_forked_sessions": 1},
        {"execution_profile_fingerprint": "A" * 64},
        {"execution_profile_fingerprint": "a" * 63},
        {"execution_profile_fingerprint": ""},
        {"kinds": ("approve", "approve")},
        {"kinds": ("take_over",)},
        {"actor_sources": ("http_auth", "http_auth")},
        {"actor_sources": ("operator",)},
        {"cursor": ""},
        {"cursor": "has space"},
        {"cursor": "x" * (SUPERVISION_EVENT_MAX_CURSOR_BYTES + 1)},
        {"unexpected": True},
    ],
)
def test_query_rejects_malformed_wrong_type_and_unknown_fields(changes):
    with pytest.raises(ValidationError):
        _query(**changes)


def test_query_accepts_every_kind_and_actor_source_filter():
    query = _query(
        kinds=tuple(SupervisionEventKind),
        actor_sources=tuple(ResolutionActorSource),
        execution_profile_fingerprint=FINGERPRINT,
        exclude_forked_sessions=True,
        sessions=SessionAggregateFilter(agent_name="close-agent", labels={"org": "acme"}),
    )

    assert set(query.kinds) == {
        SupervisionEventKind.APPROVE,
        SupervisionEventKind.EDIT,
        SupervisionEventKind.DENY,
        SupervisionEventKind.EXPIRE,
        SupervisionEventKind.ANSWER,
        SupervisionEventKind.RECOVER,
        SupervisionEventKind.INTERRUPT,
        SupervisionEventKind.STEER,
    }
    assert set(query.actor_sources) == set(ResolutionActorSource)


def test_query_does_not_echo_rejected_values_in_validation_errors():
    canary = "secret-canary-7f3a"
    with pytest.raises(ValidationError) as excinfo:
        _query(execution_profile_fingerprint=canary)

    assert canary not in str(excinfo.value)
    assert canary not in repr(excinfo.value)


def test_cursor_round_trips_the_resume_sequence_for_the_same_query():
    query = _query(kinds=(SupervisionEventKind.APPROVE,), limit=10)
    cursor = encode_supervision_event_cursor(query, after_sequence=41)

    assert len(cursor) <= SUPERVISION_EVENT_MAX_CURSOR_BYTES
    assert decode_supervision_event_cursor(query.model_copy(update={"cursor": cursor})) == 41


def test_cursor_is_independent_of_page_size_and_byte_limit():
    cursor = encode_supervision_event_cursor(_query(limit=10), after_sequence=7)

    resumed = _query(limit=200, max_bytes=4096, cursor=cursor)
    assert decode_supervision_event_cursor(resumed) == 7


def test_query_without_cursor_decodes_to_the_start():
    assert decode_supervision_event_cursor(_query()) is None


@pytest.mark.parametrize(
    "changes",
    [
        {"start_at": START + timedelta(seconds=1)},
        {"end_at": END - timedelta(seconds=1)},
        {"kinds": (SupervisionEventKind.DENY,)},
        {"actor_sources": (ResolutionActorSource.SYSTEM,)},
        {"execution_profile_fingerprint": FINGERPRINT},
        {"exclude_forked_sessions": True},
        {"sessions": SessionAggregateFilter(agent_name="other-agent")},
        {"sessions": SessionAggregateFilter(labels={"org": "globex"})},
    ],
)
def test_cursor_from_another_query_is_rejected(changes):
    cursor = encode_supervision_event_cursor(_query(), after_sequence=5)

    with pytest.raises(ValueError, match="another query"):
        decode_supervision_event_cursor(_query(**changes, cursor=cursor))


def test_cursor_binds_content_visibility():
    query = _query()
    cursor = encode_supervision_event_cursor(query, after_sequence=5, include_content=False)

    with pytest.raises(ValueError, match="another query"):
        decode_supervision_event_cursor(
            query.model_copy(update={"cursor": cursor}), include_content=True
        )


@pytest.mark.parametrize(
    "cursor",
    [
        "not*base64!",
        base64.urlsafe_b64encode(b"[]").decode(),
        base64.urlsafe_b64encode(json.dumps({"version": 1}).encode()).decode(),
        base64.urlsafe_b64encode(
            json.dumps(
                {"version": 99, "query_sha256": "0" * 64, "after_sequence": 1},
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).decode(),
        base64.urlsafe_b64encode(
            json.dumps(
                {"after_sequence": True, "query_sha256": "0" * 64, "version": 1},
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).decode(),
        base64.urlsafe_b64encode(
            json.dumps(
                {"after_sequence": 1, "query_sha256": "0" * 64, "version": True},
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).decode(),
        base64.urlsafe_b64encode(
            json.dumps(
                {"after_sequence": -1, "query_sha256": "0" * 64, "version": 1},
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).decode(),
    ],
)
def test_malformed_cursor_is_rejected_without_echoing_it(cursor):
    with pytest.raises(ValueError, match="Invalid supervision event cursor") as excinfo:
        decode_supervision_event_cursor(_query(cursor=cursor))

    assert cursor not in str(excinfo.value)


@pytest.mark.parametrize("after_sequence", [-1, True, 1.5])
def test_cursor_encoding_rejects_invalid_sequences(after_sequence):
    with pytest.raises((TypeError, ValueError)):
        encode_supervision_event_cursor(_query(), after_sequence=after_sequence)


def test_cursor_binds_the_reader_audience():
    query = _query()
    cursor = encode_supervision_event_cursor(query, after_sequence=9, audience="tenant-a")
    resumed = query.model_copy(update={"cursor": cursor})

    assert decode_supervision_event_cursor(resumed, audience="tenant-a") == 9
    for other in ("tenant-b", None):
        with pytest.raises(ValueError, match="another query"):
            decode_supervision_event_cursor(resumed, audience=other)


@pytest.mark.parametrize("digest", ["é" * 64, "A" * 64, "0" * 63, "g" * 64])
def test_cursor_with_a_malformed_digest_is_rejected_as_invalid(digest):
    cursor = base64.urlsafe_b64encode(
        json.dumps(
            {"after_sequence": 0, "query_sha256": digest, "version": 1},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).decode()

    with pytest.raises(SupervisionEventCursorInvalid) as excinfo:
        decode_supervision_event_cursor(_query(cursor=cursor))

    assert cursor not in str(excinfo.value)
