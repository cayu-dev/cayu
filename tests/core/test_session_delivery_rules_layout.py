"""Shared delivery and closure rules compose without concrete store owners."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import cayu


def _without_implementations(code: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys

class RejectImplementations(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {
            "cayu.sessions.base", "cayu.storage.sqlite", "cayu.storage.postgres",
            "cayu.storage.memory", "cayu.tasks.memory", "cayu.applications",
            "cayu.runtime._session_engine", "cayu.runtime.session_closure",
            "cayu.runtime._model_step_executor", "cayu.runtime._tool_round_executor",
        }:
            raise AssertionError(f"Shared delivery rules imported {fullname}")
sys.meta_path.insert(0, RejectImplementations())

def rejected(call, error_type=ValueError):
    try:
        call()
    except error_type:
        return
    raise AssertionError("Invalid delivery or closure authority was accepted")
"""
            + code,
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_completion_event_reservations_compose_without_stores():
    _without_implementations(
        """
from datetime import UTC, datetime, timedelta
from cayu.sessions import _checkpoint_preservation as rules

now = datetime(2026, 1, 1, tzinfo=UTC)
identity = dict(publication_id="completion-result-publication:v1:" + "a" * 64,
    authority_sha256="a" * 64, owner_id="completion-result-owner:v1:" + "b" * 64)
original = {"customer": {"items": [1]}}
reserved = rules._reserve_completion_result_event_publication(original, **identity,
    owner_expires_at=now + timedelta(seconds=10), now=now)
assert original == {"customer": {"items": [1]}}
assert rules._completion_result_event_publication_delete_block_reason(reserved, now=now)
renewed = rules._renew_completion_result_event_publication(reserved, **identity,
    owner_expires_at=now + timedelta(seconds=20), now=now + timedelta(seconds=5))
assert rules._completion_result_event_publication_delete_block_reason(
    reserved, now=now + timedelta(seconds=10)) is None
assert rules._completion_result_event_publication_delete_block_reason(
    renewed, now=now + timedelta(seconds=10))
for release in (rules._release_completion_result_event_publication,
                rules._complete_completion_result_event_publication):
    cleared = release(renewed, **identity, require_present=True, now=now)
    assert cleared == original
    rejected(lambda: release(cleared, **identity, require_present=True, now=now))
    rejected(lambda: release(renewed, **{**identity, "authority_sha256": "c" * 64},
        require_present=True, now=now))
rejected(lambda: rules._reserve_completion_result_event_publication(None, **identity,
    owner_expires_at=now, now=now))
"""
    )


def test_first_event_delivery_validates_exact_unclaimed_attempt_without_stores():
    _without_implementations(
        """
from cayu.sessions.event_delivery import (
    PersistedEventSideEffectDelivery, PersistedEventSideEffectStatus,
    _copy_pending_first_event_delivery, _copy_failed_first_delivery_retirement,
    _copy_optional_event_id,
)

pending = PersistedEventSideEffectDelivery(session_id="session", event_id="event",
    event_sequence=1, status=PersistedEventSideEffectStatus.PENDING)
copied = _copy_pending_first_event_delivery(pending)
assert copied == pending and copied is not pending
failed = pending.model_copy(update={"status": PersistedEventSideEffectStatus.FAILED, "attempts": 1})
assert _copy_failed_first_delivery_retirement(failed) == failed
rejected(lambda: _copy_pending_first_event_delivery(failed))
rejected(lambda: _copy_failed_first_delivery_retirement(pending))
rejected(lambda: _copy_failed_first_delivery_retirement(failed.model_copy(update={"attempts": 2})))
rejected(lambda: _copy_pending_first_event_delivery(pending.model_copy(update={"claim_id": "owned"})))
assert _copy_optional_event_id(None, "event") is None
assert _copy_optional_event_id("event", "event") == "event"
rejected(lambda: _copy_optional_event_id("x" * 10000, "event"))
"""
    )
