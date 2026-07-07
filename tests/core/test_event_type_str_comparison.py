"""Regression guard for #125 footgun 1: EventType is a StrEnum — compare with ==, never is.

A known event type validates back to the enum member through the ``Event`` model (so cayu's own
store-sourced comparisons work), but an event deserialized from JSON elsewhere (webhook / SSE /
JSONL, or ``payload["type"]``) carries ``.type`` as a plain str. Identity (``is``) silently fails on
that str form, so framework and consumer code must use ``==``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from cayu import (
    Event,
    EventType,
    RunRequest,
    SessionIdentity,
    SQLiteSessionStore,
)


def test_event_type_str_form_matches_enum_by_equality_not_identity() -> None:
    # model_construct bypasses validation -> simulates a JSON-sourced event whose .type is a plain str.
    event = Event.model_construct(type="session.failed", session_id="s")
    assert type(event.type) is str  # not coerced to the enum member
    assert event.type == EventType.SESSION_FAILED  # `==` detects it
    assert event.type is not EventType.SESSION_FAILED  # `is` would silently miss it — the footgun


def test_event_type_known_string_coerces_to_enum_through_the_model() -> None:
    # Constructed/validated through the Event model, a known type IS the enum member — this is why
    # cayu's store path (which rehydrates via the model) is safe; the == audit keeps JSON paths safe too.
    event = Event(type="session.failed", session_id="s")
    assert event.type is EventType.SESSION_FAILED
    assert event.type == EventType.SESSION_FAILED


def test_store_round_tripped_failure_event_is_detected(tmp_path: Path) -> None:
    # #125 footgun-1 acceptance: persist a session.failed event, reload it from the SQLite store,
    # and assert detection succeeds. `_event_from_row` rebuilds via the Event model, so its
    # validator coerces `.type` back to the enum member — the round-trip itself is safe (both `==`
    # and `is` match). This pins that the store path is NOT where the footgun bites.
    store = SQLiteSessionStore(tmp_path / "sessions.sqlite")

    async def scenario() -> Event:
        await store.create(
            RunRequest(
                session_id="s1", agent_name="assistant", environment_name="local-dev", messages=[]
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.append_event(
            "s1",
            Event(
                type=EventType.SESSION_FAILED,
                session_id="s1",
                environment_name="local-dev",
                payload={"error": "boom"},
            ),
        )
        return (await store.load_events("s1"))[-1]

    reloaded = asyncio.run(scenario())
    assert reloaded.type == EventType.SESSION_FAILED
    assert reloaded.type is EventType.SESSION_FAILED  # store coerced back to the enum member
    assert reloaded.payload["error"] == "boom"
