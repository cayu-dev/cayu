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


def test_session_usage_public_imports_and_historical_pickle_global():
    from cayu.sessions import base, usage

    assert base.UsageRollupQuery is usage.UsageRollupQuery
    for module in ("cayu", "cayu.sessions", "cayu.runtime"):
        assert importlib.import_module(module).UsageRollupQuery is usage.UsageRollupQuery
    assert pickle.loads(b"ccayu.sessions.base\nUsageRollupQuery\n.") is usage.UsageRollupQuery


@pytest.mark.parametrize("public_module", ("cayu", "cayu.sessions", "cayu.runtime"))
def test_session_usage_aggregates_bounded_records_without_stores(public_module):
    _assert_store_independent(
        """
from datetime import UTC, datetime, timedelta
from cayu.events import Event, EventType
from cayu.sessions.usage import _usage_rollup_from_session_records, copy_usage_rollup_query

stamp = datetime(2026, 1, 1, tzinfo=UTC)
end = stamp + timedelta(hours=1)
query = public.UsageRollupQuery(start_at=stamp, end_at=end,
    group_limit=1, session_group_limit=1, include_pricing_inputs=True,
    sessions=public.SessionAggregateFilter(labels={"team": "support"}))
copied = copy_usage_rollup_query(query)
copied.sessions.labels["team"] = "changed"
assert query.sessions.labels == {"team": "support"}
def record(sequence, session_id, tokens, provider, at=stamp, kind=EventType.MODEL_COMPLETED):
    payload = {} if tokens is None else {"usage_metrics": {
        "input_tokens": tokens, "output_tokens": 0, "total_tokens": tokens,
        "provider_name": provider, "model": "model",
    }}
    return public.EventRecord(sequence=sequence, event=Event(
        id=f"event-{session_id}-{sequence}", type=kind, session_id=session_id,
        timestamp=at, payload=payload,
    ))
rows = [
    ("a", public.SessionStatus.COMPLETED, (
        record(1, "a", 10, "provider-a"), record(2, "a", None, None),
        record(3, "a", 999, "excluded", at=end),
    )),
    ("b", public.SessionStatus.RUNNING, (
        record(1, "b", 20, "provider-b"),
        record(2, "b", None, None, kind=EventType.TOOL_CALL_STARTED),
    )),
]
result = _usage_rollup_from_session_records(session_records=lambda: iter(rows),
    query=query, as_of=end, matching_session_count=2, active_session_count=1)
assert result.totals.usage.total_tokens == 30
assert result.totals.model_steps == 3 and result.totals.model_steps_with_usage == 2
assert result.totals.tool_calls == 1 and result.totals.session_count == 2
assert result.provider_breakdown.groups[0].provider_name == "provider-b"
assert result.provider_breakdown.remainder.totals.usage.total_tokens == 10
assert result.provider_breakdown.accuracy.kind.value == "truncated"
assert result.session_breakdown.groups[0].session_id == "b"
assert result.session_breakdown.remainder.totals.usage.total_tokens == 10
assert result.session_breakdown.remainder.group_count == 1
assert result.matching_session_count == 2 and result.active_session_count == 1
assert result.pricing_inputs_included and result.session_pricing_inputs_included
assert pickle.loads(pickle.dumps(query)) == query
assert get_type_hints(type(query)) and type(query).model_json_schema()
try:
    copy_usage_rollup_query(object())
except TypeError as exc:
    assert str(exc) == "Usage aggregate queries must be UsageRollupQuery instances."
else:
    raise AssertionError("Invalid query type accepted")
""",
        public_module,
    )


def test_session_inspection_public_imports_and_historical_pickle_globals():
    from cayu.sessions import base, inspection, records

    for owner, names in (
        (
            inspection,
            (
                "SerializedRecordSummary",
                "SessionInspectionIdentity",
                "SessionInspectionUsageSummary",
                "SessionInspectionSummary",
            ),
        ),
        (records, ("PendingActionKind",)),
    ):
        for name in names:
            canonical = getattr(owner, name)
            assert getattr(base, name) is canonical
            for module in ("cayu", "cayu.sessions", "cayu.runtime"):
                assert getattr(importlib.import_module(module), name) is canonical
            assert pickle.loads(f"ccayu.sessions.base\n{name}\n.".encode()) is canonical


@pytest.mark.parametrize("public_module", ("cayu", "cayu.sessions", "cayu.runtime"))
def test_session_inspection_bounds_and_folds_records_without_stores(public_module):
    _assert_store_independent(
        """
from datetime import UTC, datetime
from cayu.budgets.base import SessionBudgetInspection
from cayu.budgets.usage import UsageMetrics
from cayu.events import Event, EventType
from cayu.sessions.inspection import (
    SESSION_INSPECTION_LABEL_LIMIT, _SESSION_INSPECTION_MAX_RETAINED_EVENT_BYTES,
    _SessionInspectionUsageAccumulator, _bounded_session_inspection_labels,
    _retain_session_inspection_event,
)

labels = {f"label-{i:04d}": str(i) for i in range(SESSION_INSPECTION_LABEL_LIMIT + 1)}
retained, total, truncated = _bounded_session_inspection_labels(labels)
assert total == SESSION_INSPECTION_LABEL_LIMIT + 1 and truncated
assert list(retained) == sorted(labels)[:SESSION_INSPECTION_LABEL_LIMIT]
retained["label-0000"] = "changed"
assert labels["label-0000"] == "0"
metrics = UsageMetrics(input_tokens=7, output_tokens=3, total_tokens=10,
    provider_name="provider", model="model")
fold = _SessionInspectionUsageAccumulator()
fold.add(EventType.MODEL_COMPLETED, metrics)
fold.add(EventType.MODEL_COMPLETED, None)
fold.add(EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED, metrics)
fold.add(EventType.MODEL_HOSTED_TOOL_CALL, metrics)
fold.add(EventType.TOOL_CALL_STARTED, None)
usage, with_usage = fold.result("session")
assert usage.usage.total_tokens == 30 and usage.model_steps == 2
assert with_usage == 1 and usage.tool_calls == 1
assert usage.provider_names == ["provider"] and usage.models == ["model"]
fold.provider_names.append("later")
assert usage.provider_names == ["provider"]
stamp = datetime(2026, 1, 1, tzinfo=UTC)
identity = public.SessionInspectionIdentity(id="session", agent_name="agent",
    provider_name="provider", model="model", parent_session_id=None,
    causal_budget_id="session", runtime_name="cayu", runtime_version=None,
    environment_name=None, status="interrupted", created_at=stamp, updated_at=stamp,
    last_activity_at=stamp, run_epoch=1, label_count=0)
sizes = public.SerializedRecordSummary(record_count=0, total_bytes=0, largest_record_bytes=0)
budget = SessionBudgetInspection(event_count=0, reservation_count=0,
    reconciliation_count=0, pending_reservation_count=0,
    completed_unsettled_reservation_count=0, reconciled_reservation_count=0,
    conservative_reconciliation_count=0, released_reservation_count=0, cost_state="unknown")
summary = public.SessionInspectionSummary(session=identity, transcript=sizes,
    events=sizes, usage=usage, model_calls=2, model_calls_with_usage=1, tool_calls=1,
    pending_action_count=1, pending_action_kinds=(public.PendingActionKind.USER_INPUT,),
    pending_action_issue_count=0, queued_message_count=0, delivered_message_count=0,
    outstanding_message_count=0, operation_event_count=0,
    terminal_failure_state="interrupted", budget=budget)
assert summary.model_dump(mode="json")["pending_action_kinds"] == ["user_input"]
for value in (identity, sizes, usage, summary):
    assert pickle.loads(pickle.dumps(value)) == value
    assert get_type_hints(type(value)) and type(value).model_json_schema()
event = Event(id="event", type=EventType.SESSION_FAILED, session_id="session",
    timestamp=stamp, payload={"error": "failure"})
assert _retain_session_inspection_event(0, event) > 0
try:
    _retain_session_inspection_event(_SESSION_INSPECTION_MAX_RETAINED_EVENT_BYTES, event)
except ValueError as exc:
    assert str(exc) == ("Session inspection exceeds the retained-event safety limit of "
        f"{_SESSION_INSPECTION_MAX_RETAINED_EVENT_BYTES} bytes.")
else:
    raise AssertionError("Oversized retained events were accepted")
""",
        public_module,
    )
