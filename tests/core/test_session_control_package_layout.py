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


def _assert_public_identity(owner: str, old_module: str, names: tuple[str, ...]) -> None:
    module = importlib.import_module(owner)
    historical = importlib.import_module(old_module)
    for name in names:
        canonical = getattr(module, name)
        assert getattr(historical, name) is canonical
        for public_module in ("cayu", "cayu.runtime"):
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
