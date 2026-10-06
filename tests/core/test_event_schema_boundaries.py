"""Event evidence and its data contracts stay independent of live execution."""

from __future__ import annotations

import importlib
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

import cayu
from cayu import _event_schema as schema
from cayu.events import Event, EventType, event_payload_authority_is_runtime_generated


@pytest.mark.parametrize(
    "first_module",
    (
        "cayu._event_schema",
        "cayu.providers._retry_decision",
        "cayu.tools._shared_artifact_result_schema",
        "cayu.tools._web_access_result_schema",
        "cayu.workspaces._revision_records",
    ),
)
def test_event_evidence_works_without_runtime_or_storage(first_module):
    script = """
import importlib
import importlib.abc
import sys

class RejectExecution(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname == 'cayu.runtime' or fullname.startswith('cayu.runtime.')
                or fullname == 'cayu.storage' or fullname.startswith('cayu.storage.')
                or fullname in {'cayu.sessions.base', 'cayu.tools.base', 'cayu.runners.base'}):
            raise AssertionError(f'Shared event evidence imported {fullname}')

sys.meta_path.insert(0, RejectExecution())
importlib.import_module(sys.argv[1])
import cayu
from cayu import _event_schema as schema
from cayu.events import Event, EventType
from cayu.providers import _retry_decision as retry
from cayu.workspaces import _revision_records as workspace

assert cayu.RetryDecision is retry.RetryDecision
assert cayu.WorkspacePathRevision is workspace.WorkspacePathRevision
assert set(schema.EVENT_PAYLOAD_POLICIES) == set(EventType)
event = Event(type=EventType.SESSION_INTERRUPTED, session_id='session', payload={
    'tool_round_id': 'round', 'approval': {'tool_round_id': 'round', 'tool_call_id': 'call'}
})
assert schema.private_event_linkage_value(event, field_name='tool_call_id') == 'call'
assert schema.private_event_linkage_value(event, field_name='tool_round_id') == 'round'
event.payload['approval']['tool_round_id'] = 'different'
assert schema.private_event_linkage_value(event, field_name='tool_round_id') is None
assert retry.RetryDecision(retry=False, attempt=1, max_attempts=1,
                          effective_max_attempts=1).model_dump(mode='json')['retry'] is False
assert workspace.WorkspacePathRevision(path='folder/file').path == 'folder/file'
"""
    result = subprocess.run(
        [sys.executable, "-c", script, first_module],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "payload,expected",
    (
        ({"approval": {"tool_call_id": "one", "tool_calls": [{"tool_call_id": "two"}]}}, "one"),
        ({"approval": {"tool_calls": [{"tool_call_id": "one"}]}}, "one"),
        (
            {"approval": {"tool_calls": [{"tool_call_id": "one"}, {"tool_call_id": "two"}]}},
            None,
        ),
        ({"approval": {"tool_call_id": [], "tool_calls": [{"tool_call_id": "one"}]}}, None),
        ({"approval": {"tool_call_id": "one", "tool_calls": [{"tool_call_id": " "}]}}, None),
        ({"result": {"tool_call_id": "untrusted"}}, None),
    ),
)
def test_linkage_keeps_schema_authority_and_conflict_rules(payload, expected):
    event = Event(type=EventType.SESSION_INTERRUPTED, session_id="session", payload=payload)
    assert schema.private_event_linkage_value(event, field_name="tool_call_id") == expected


def test_linkage_reads_current_evidence_and_does_not_stamp_authority():
    event = Event(
        type=EventType.TOOL_CALL_STARTED, session_id="session", payload={"tool_call_id": "one"}
    )
    original = event.model_copy(deep=True)
    assert not event_payload_authority_is_runtime_generated(
        event, field_name="tool_call_id", value="one"
    )
    assert schema.private_event_linkage_value(event, field_name="tool_call_id") == "one"
    assert event == original
    assert not event_payload_authority_is_runtime_generated(
        event, field_name="tool_call_id", value="one"
    )
    event.payload["tool_call_id"] = "two"
    assert schema.private_event_linkage_value(event, field_name="tool_call_id") == "two"


@pytest.mark.parametrize(
    "canonical,public_module,name",
    (
        ("cayu.providers._retry_decision", "cayu.runtime.retry_policy", "RetryDecision"),
        ("cayu.providers._retry_decision", "cayu.runtime.retry_policy", "RetryReason"),
        ("cayu.providers._retry_decision", "cayu.runtime.retry_policy", "RetryDisposition"),
        ("cayu.providers._retry_decision", "cayu.runtime.retry_policy", "RetrySuppression"),
        ("cayu.workspaces._revision_records", "cayu.workspaces.revisions", "WorkspacePathRevision"),
        (
            "cayu.workspaces._revision_records",
            "cayu.workspaces.revisions",
            "WorkspacePathRevisionDelta",
        ),
    ),
)
def test_public_data_imports_and_saved_pickle_paths_share_one_definition(
    canonical, public_module, name
):
    expected = getattr(importlib.import_module(canonical), name)
    assert getattr(importlib.import_module(public_module), name) is expected
    assert getattr(cayu, name) is expected
    namespace = importlib.import_module(
        "cayu.runtime" if name.startswith("Retry") else "cayu.workspaces"
    )
    assert getattr(namespace, name) is expected
    assert pickle.loads(f"c{public_module}\n{name}\n.".encode()) is expected
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        assert pickle.loads(pickle.dumps(expected, protocol=protocol)) is expected


def test_projection_uses_one_registry_without_private_forwarding_aliases():
    from cayu.runtime import _event_projection as projection

    assert projection.event_schema is schema
    for name in (
        "EventPayloadPolicy",
        "EVENT_PAYLOAD_POLICIES",
        "event_payload_policy",
        "private_event_linkage_value",
    ):
        assert not hasattr(projection, name)
