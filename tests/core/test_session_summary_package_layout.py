"""Stored-session read projections compose without concrete stores."""

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
            "cayu.sessions.base", "cayu.tasks.memory",
        }:
            raise AssertionError(f"Session projection imported {fullname}")
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


def test_session_summary_public_imports_and_historical_pickle_globals():
    from cayu.sessions import base, summaries

    for name in ("SessionOperationalSnapshot", "EventSummary", "SessionOutcome"):
        canonical = getattr(summaries, name)
        assert getattr(base, name) is canonical
        for module in ("cayu", "cayu.sessions", "cayu.runtime"):
            assert getattr(importlib.import_module(module), name) is canonical
        assert pickle.loads(f"ccayu.sessions.base\n{name}\n.".encode()) is canonical


@pytest.mark.parametrize("public_module", ("cayu", "cayu.sessions", "cayu.runtime"))
def test_session_summaries_project_durable_records_without_stores(public_module):
    _assert_store_independent(
        """
from datetime import UTC, datetime
from cayu.budgets.aggregates import EXACT_AGGREGATE
from cayu.events import Event, EventType
from cayu.sessions.invocation import InvocationOrigin, SessionInvocation
from cayu.sessions.summaries import event_summary_from_records, session_outcome_from_records

stamp = datetime(2026, 1, 1, tzinfo=UTC)
session = public.Session(
    id="session", agent_name="agent", provider_name="provider", model="model",
    created_at=stamp, updated_at=stamp, last_activity_at=stamp, status="failed",
    invocation=SessionInvocation(
        origin=InvocationOrigin(trust="unattributed"),
        root_invocation_id="12345678-1234-4234-8234-123456789abc",
        root_session_id="session", source="sdk_run",
    ),
)
def record(sequence, event_type, payload=None, session_id="session"):
    return public.EventRecord(sequence=sequence, event=Event(
        id=f"event-{sequence}", type=event_type, session_id=session_id,
        timestamp=stamp, payload=payload or {},
    ))
records = [
    record(1, EventType.SESSION_FAILED, {"error": "previous"}),
    record(2, EventType.SESSION_RESUMED),
    record(3, EventType.MODEL_RETRY, {"reason": "retry", "attempt": 1}),
    record(4, EventType.SESSION_FAILED, {"error": "current", "error_type": "ValueError"}),
    record(5, EventType.SESSION_COMPLETED, session_id="other"),
]
outcome = session_outcome_from_records(session, records)
assert outcome.reason == "failed"
assert outcome.details == {"error": "current", "error_type": "ValueError"}
assert outcome.retry == {"reason": "retry", "attempt": 1}
assert outcome.terminal_event.sequence == 4
summary = event_summary_from_records("session", records)
assert summary.total_events == 4
assert summary.counts_by_type["session.failed"] == 2
assert summary.latest_event.sequence == 4
records[3].event.payload["error"] = "changed"
assert outcome.details["error"] == outcome.terminal_event.event.payload["error"] == "current"
assert summary.latest_event.event.payload["error"] == "current"
# A resumed interaction cannot inherit the prior interaction's failure.
resumed = session_outcome_from_records(session, records[:2])
assert resumed.terminal_event is None and resumed.details == {} and resumed.retry is None
counts = public.SessionStatusCounts(pending=0, running=0, interrupting=0,
    completed=0, failed=1, interrupted=0)
snapshot = public.SessionOperationalSnapshot(as_of=stamp, total_count=1,
    counts_by_status=counts, accuracy=EXACT_AGGREGATE)
counts.failed = 0
assert snapshot.counts_by_status.failed == 1
for value in (outcome, summary, snapshot):
    assert pickle.loads(pickle.dumps(value)) == value
    assert get_type_hints(type(value))
    assert type(value).model_json_schema()
""",
        public_module,
    )
