"""Bounded session discovery contracts compose without session stores."""

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
            raise AssertionError(f"Session discovery imported {fullname}")
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


def _assert_public_identity(owner: str, names: tuple[str, ...]) -> None:
    from cayu.sessions import base

    module = importlib.import_module(owner)
    for name in names:
        canonical = getattr(module, name)
        assert getattr(base, name) is canonical
        for public_module in ("cayu", "cayu.sessions", "cayu.runtime"):
            assert getattr(importlib.import_module(public_module), name) is canonical
        assert pickle.loads(f"ccayu.sessions.base\n{name}\n.".encode()) is canonical


def test_pending_action_public_imports_and_historical_pickle_globals():
    _assert_public_identity(
        "cayu.sessions.pending_action_contracts",
        (
            "PendingActionQuery",
            "PendingActionRecord",
            "PendingActionIssue",
            "PendingActionIssueCode",
            "PendingActionListResult",
            "PendingActionResultTooLarge",
            "DelegatedActionReference",
        ),
    )


@pytest.mark.parametrize("public_module", ("cayu", "cayu.sessions", "cayu.runtime"))
def test_pending_action_bounds_and_identity_without_stores(public_module):
    _assert_store_independent(
        """
from datetime import UTC, datetime
from cayu.events import Event, EventType
from cayu.sessions.pending_action_contracts import enforce_pending_action_result_size

stamp = datetime(2026, 1, 1, tzinfo=UTC)
session = public.PendingActionSession(id="session", instance_id="incarnation",
    agent_name="agent", provider_name="provider", model="model", causal_budget_id="session",
    runtime_name="cayu", status="interrupted", created_at=stamp, updated_at=stamp)
event = public.EventRecord(sequence=1, event=Event(id="event", session_id="session",
    type=EventType.SESSION_AWAITING_USER_INPUT, timestamp=stamp))
action = public.PendingActionRecord(id="action", kind="user_input", session=session,
    event=event, title="Question", input_id="input", question="Continue?")
identity = action.attention_id
assert identity is not None and identity.startswith("attention_")
session.instance_id = "changed"
event.event.payload["changed"] = True
assert action.session.instance_id == "incarnation" and not action.event.event.payload
encoded = action.model_dump(mode="json")
encoded["attention_id"] = "untrusted"
assert public.PendingActionRecord.model_validate(encoded).attention_id == identity
delegated = public.PendingActionRecord(id="delegated", kind="delegated_action",
    session=action.session, event=action.event, title="Child question",
    delegated_action=public.DelegatedActionReference(child_session_id="child",
        action_kind="user_input", action_id="child-input"))
assert delegated.attention_id is None
result = public.PendingActionListResult(actions=[action, delegated])
assert enforce_pending_action_result_size(result, max_bytes=10000) is result
try:
    enforce_pending_action_result_size(result, max_bytes=1)
except public.PendingActionResultTooLarge as exc:
    assert exc.max_bytes == 1
else:
    raise AssertionError("Oversized pending-action page accepted")
for value in (public.PendingActionQuery(session_id="session"), action, delegated, result):
    assert pickle.loads(pickle.dumps(value)) == value
    assert get_type_hints(type(value)) and type(value).model_json_schema()
""",
        public_module,
    )


def test_session_topology_public_imports_and_historical_pickle_globals():
    _assert_public_identity(
        "cayu.sessions.topology",
        (
            "SessionTopologyNode",
            "SessionTopologyQuery",
            "SessionTopologyBranch",
            "SessionTopologyStoreResult",
            "SessionTopologyCycle",
            "SessionTopologyDepthExceeded",
        ),
    )


@pytest.mark.parametrize("public_module", ("cayu", "cayu.sessions", "cayu.runtime"))
def test_session_topology_pages_and_cursors_without_stores(public_module):
    _assert_store_independent(
        """
from datetime import UTC, datetime
from cayu.sessions.topology import build_session_topology_result, decode_session_topology_cursor

stamp = datetime(2026, 1, 1, tzinfo=UTC)
def node(id, parent=None):
    return public.SessionTopologyNode(id=id, parent_session_id=parent, agent_name="agent",
        provider_name="provider", model="model", causal_budget_id="root", runtime_name="cayu",
        runtime_version=None, environment_name=None, status="completed", created_at=stamp,
        updated_at=stamp, last_activity_at=stamp)
root = node("root")
children = (node("a", "root"), node("b", "root"))
def bounded_candidates():
    yield from children
    raise AssertionError("Topology consumed beyond its page lookahead")
result = build_session_topology_result(focus=root, ancestors=(), expanded_parents=(root,),
    branch_candidates=(bounded_candidates(),), child_limit=1)
branch, = result.branches
assert branch.children == children[:1] and branch.has_more
assert decode_session_topology_cursor(branch.next_cursor, parent_session_id="root") == (stamp, "a")
try:
    decode_session_topology_cursor(branch.next_cursor, parent_session_id="other")
except ValueError as exc:
    assert str(exc) == "Invalid session topology cursor."
else:
    raise AssertionError("Cross-parent topology cursor accepted")
for value in (public.SessionTopologyQuery(focus_session_id="root"), root, result):
    assert pickle.loads(pickle.dumps(value)) == value
    assert get_type_hints(type(value)) and type(value).model_json_schema()
try:
    build_session_topology_result(focus=node("a", "b"), ancestors=(),
        expanded_parents=(node("b", "a"),), branch_candidates=((),), child_limit=1)
except public.SessionTopologyCycle:
    pass
else:
    raise AssertionError("Loaded topology cycle accepted")
""",
        public_module,
    )


def test_session_lineage_public_imports_and_historical_pickle_globals():
    _assert_public_identity(
        "cayu.sessions.lineage",
        (
            "SessionLineageOrigin",
            "SessionLineageNode",
            "SessionLineageQuery",
            "SessionLineageResult",
        ),
    )


@pytest.mark.parametrize("public_module", ("cayu", "cayu.sessions", "cayu.runtime"))
def test_session_lineage_bounds_and_cursors_without_stores(public_module):
    _assert_store_independent(
        """
from datetime import UTC, datetime
from pydantic import ValidationError
from cayu.sessions.lineage import (
    SESSION_LINEAGE_MAX_IDENTIFIER_BYTES, copy_session_lineage_query,
    encode_session_lineage_cursor, decode_session_lineage_cursor,
)

stamp = datetime(2026, 1, 1, tzinfo=UTC)
origin = public.SessionLineageOrigin(sequence=1, event_id="event", event_type="session.forked")
node = public.SessionLineageNode(id="child", parent_session_id="parent",
    created_at=stamp, origin_events=(origin,))
assert node.origin_events[0] == origin and node.origin_events[0] is not origin
cursor = encode_session_lineage_cursor("parent", node)
assert decode_session_lineage_cursor(cursor, parent_session_id="parent") == (stamp, "child")
query = public.SessionLineageQuery(parent_session_id="parent", cursor=cursor, limit=1)
copied = copy_session_lineage_query(query)
assert copied == query and copied is not query
result = public.SessionLineageResult(parent_session_id="parent", children=(node,),
    next_cursor=cursor, has_more=True)
try:
    decode_session_lineage_cursor(cursor, parent_session_id="other")
except ValueError as exc:
    assert str(exc) == "Invalid session lineage cursor."
else:
    raise AssertionError("Cross-parent lineage cursor accepted")
try:
    public.SessionLineageQuery(parent_session_id="é" * (SESSION_LINEAGE_MAX_IDENTIFIER_BYTES // 2 + 1))
except ValidationError:
    pass
else:
    raise AssertionError("Oversized UTF-8 lineage identifier accepted")
for value in (origin, node, query, result):
    assert pickle.loads(pickle.dumps(value)) == value
    assert get_type_hints(type(value)) and type(value).model_json_schema()
""",
        public_module,
    )
