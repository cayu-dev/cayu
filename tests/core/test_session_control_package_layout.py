"""Session control values compose independently of stores and execution workers."""

from __future__ import annotations

import importlib
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

import cayu


def _assert_store_independent(code: str, public_module: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import importlib.abc
import pickle
import sys
from typing import get_type_hints

class RejectStores(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("cayu.storage") or fullname in {
            "cayu.sessions.base", "cayu.tasks.memory", "cayu.runtime.execution",
        }:
            raise AssertionError(f"Session control contract imported {fullname}")
sys.meta_path.insert(0, RejectStores())
public = importlib.import_module(sys.argv[1])
"""
            + code,
            public_module,
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _assert_public_identity(
    owner: str,
    old_module: str,
    names: tuple[str, ...],
    public_modules: tuple[str, ...] = ("cayu", "cayu.runtime"),
) -> None:
    module = importlib.import_module(owner)
    historical = importlib.import_module(old_module)
    for name in names:
        canonical = getattr(module, name)
        assert getattr(historical, name) is canonical
        for public_module in public_modules:
            assert getattr(importlib.import_module(public_module), name) is canonical
        assert pickle.loads(f"c{old_module}\n{name}\n.".encode()) is canonical


def test_session_message_public_imports_and_historical_pickle_globals():
    _assert_public_identity(
        "cayu.sessions.messaging",
        "cayu.sessions.base",
        (
            "EnqueueSessionMessageRequest",
            "EnqueueSessionMessageResult",
            "SessionQueuedMessage",
            "SessionMessageDeliveryBatch",
            "SessionMessageDeliveryMode",
            "SessionMessageInspection",
            "SessionMessageInspectionRecord",
            "SessionMessageActionResult",
        ),
    )
    _assert_public_identity(
        "cayu.sessions.messaging",
        "cayu.runtime.session_message_lifecycle",
        (
            "SessionMessageConditions",
            "SessionMessageSource",
            "SessionMessageTarget",
            "SessionMessageQuery",
            "SessionMessageCursor",
            "SessionMessageQueueStatus",
            "SessionMessageActionRequest",
            "SessionMessageAccessPolicy",
        ),
    )


@pytest.mark.parametrize("public_module", ("cayu", "cayu.sessions.messaging", "cayu.runtime"))
def test_session_message_values_and_rules_without_stores(public_module):
    _assert_store_independent(
        """
from datetime import UTC, datetime
from pydantic import ValidationError
from cayu.messages import Message
from cayu.sessions.messaging import (
    copy_enqueue_session_message_request, enqueue_session_message_input,
    session_message_rejection, SESSION_MESSAGE_CONTENT_MAX_BYTES,
)
stamp = datetime(2026, 1, 1, tzinfo=UTC)
message = Message.text("user", "Authoritative input")
conditions = public.SessionMessageConditions(
    target=public.SessionMessageTarget(session_instance_id="instance", run_epoch=2, transcript_cursor=3),
    expires_at=stamp,
)
request = public.EnqueueSessionMessageRequest(session_id="session", idempotency_key="key",
    content="Summary", message=message, delivery_mode="next_turn", conditions=conditions)
assert request.message is not message
assert enqueue_session_message_input(request).content[0].text == "Authoritative input"
copy = copy_enqueue_session_message_request(request)
assert copy == request and copy is not request and copy.message is not request.message
assert session_message_rejection(conditions, session_instance_id="wrong", run_epoch=0,
    transcript_cursor=0, now=stamp) is public.SessionMessageQueueStatus.EXPIRED
assert pickle.loads(pickle.dumps(request)) == request
assert get_type_hints(copy_enqueue_session_message_request)["return"] is type(request)
assert type(request).model_json_schema()["title"] == "EnqueueSessionMessageRequest"
try:
    public.EnqueueSessionMessageRequest(session_id="session", idempotency_key="key",
        content="é" * (SESSION_MESSAGE_CONTENT_MAX_BYTES // 2 + 1), delivery_mode="next_turn")
except ValidationError:
    pass
else:
    raise AssertionError("Encoded content bound was not enforced")
""",
        public_module,
    )


def test_event_delivery_public_imports_and_historical_pickle_globals():
    _assert_public_identity(
        "cayu.sessions.event_delivery",
        "cayu.sessions.base",
        (
            "PersistedEventSideEffectClaim",
            "PersistedEventSideEffectDelivery",
            "PersistedEventSideEffectStatus",
            "PersistedEventSideEffectClaimLost",
        ),
        public_modules=("cayu.runtime",),
    )
    _assert_public_identity(
        "cayu.sessions.event_delivery",
        "cayu.runtime.event_side_effect_health",
        (
            "PersistedEventSideEffectHealth",
            "PersistedEventSideEffectQuery",
            "PersistedEventSideEffectInspection",
            "PersistedEventSideEffectPage",
        ),
    )


@pytest.mark.parametrize("public_module", ("cayu", "cayu.sessions.event_delivery", "cayu.runtime"))
def test_event_delivery_values_and_projections_without_stores(public_module):
    _assert_store_independent(
        """
from datetime import UTC, datetime
from cayu.events import Event, EventType
from cayu.sessions.event_delivery import (
    PERSISTED_EVENT_SIDE_EFFECT_ERROR_MAX_BYTES, cursor_key, page, page_sql,
    PersistedEventSideEffectClaim, PersistedEventSideEffectDelivery,
    health_sql, portable_persisted_event_side_effect_error,
    validate_persisted_event_side_effect_error,
)
stamp = datetime(2026, 1, 1, tzinfo=UTC)
event = Event(id="event", session_id="session", type=EventType.SESSION_STARTED,
    timestamp=stamp, payload={"nested": {"value": 1}})
claim = PersistedEventSideEffectClaim(session_id="session", event_id="event",
    event_sequence=1, event=event, attempt=1, claim_id="claim", lease_expires_at=stamp)
event.payload["nested"]["value"] = 2
assert claim.event.payload["nested"]["value"] == 1
error = portable_persisted_event_side_effect_error("é" * 4096)
assert len(error.encode("utf-8")) <= PERSISTED_EVENT_SIDE_EFFECT_ERROR_MAX_BYTES
assert validate_persisted_event_side_effect_error(error) == error
row = PersistedEventSideEffectDelivery(session_id="session", event_id="event",
    event_sequence=1, status="failed", last_error=error, updated_at=stamp)
query = public.PersistedEventSideEffectQuery(limit=1)
result = page([row, row.model_copy(update={"event_id": "second"})], query, stamp)
assert len(result.deliveries) == 1 and result.deliveries[0].claimable
assert result.deliveries[0].last_error != error
continued = query.model_copy(update={"cursor": result.next_cursor})
assert cursor_key(continued) == ("session", "event")
for placeholder in ("?", "%s"):
    sql, params = page_sql(continued, stamp, placeholder)
    assert params == [stamp, "session", "event", 2]
    assert "oldest_claimable_at" in health_sql(placeholder)
    assert "earliest_live_lease_expires_at" in health_sql(placeholder)
assert pickle.loads(pickle.dumps(claim)) == claim
assert get_type_hints(type(claim))["event"] is Event
assert type(claim).model_json_schema()["title"] == "PersistedEventSideEffectClaim"
""",
        public_module,
    )
