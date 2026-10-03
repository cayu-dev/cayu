"""Shared continuation values remain usable independently of live execution."""

from __future__ import annotations

import importlib
import inspect
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest
from tests.core.test_temporary_continuation_contracts import service_records

import cayu


@pytest.mark.parametrize("module_name", ("_session_continuation", "_temporary_continuation"))
def test_legacy_continuation_imports_and_pickles_share_canonical_definitions(module_name):
    canonical = importlib.import_module(f"cayu.sessions.{module_name}")
    legacy = importlib.import_module(f"cayu.runtime.{module_name}")
    for name, value in vars(canonical).items():
        if getattr(value, "__module__", None) == canonical.__name__:
            assert getattr(legacy, name) is value
            if inspect.isclass(value) or inspect.isfunction(value):
                assert pickle.loads(f"c{legacy.__name__}\n{name}\n.".encode()) is value
        elif (
            module_name == "_session_continuation" and name.startswith("CONTINUATION_")
        ) or name == "ServiceReleasedSessionStatus":
            assert getattr(legacy, name) is value
    for namespace in (cayu, cayu.collaboration, cayu.runtime):
        assert (
            namespace.ContinuationRecord is cayu.sessions._session_continuation.ContinuationRecord
        )
        assert (
            namespace.ContinuationConflict
            is cayu.sessions._session_continuation.ContinuationConflict
        )
    assert not hasattr(cayu.sessions._session_continuation, "admit_continuation")
    assert not hasattr(cayu.sessions._session_continuation, "require_operation_record_owner")


def test_continuation_records_validate_without_runtime_or_store_implementations():
    script = """
import importlib.abc
import json
import pickle
import sys

class RejectRuntimeAndStores(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname == "cayu.runtime" or fullname.startswith("cayu.runtime.")
                or fullname == "cayu.storage" or fullname.startswith("cayu.storage.")
                or fullname == "cayu.sessions.base"):
            raise AssertionError(f"Continuation contracts imported {fullname}")
sys.meta_path.insert(0, RejectRuntimeAndStores())

from cayu import ContinuationRecord
from cayu.collaboration import ContinuationConflict
from cayu.sessions._session_continuation import (
    ContinuationTicket, continuation_digest, continuation_operation_key,
    require_ticket_identity,
)
from cayu.sessions._temporary_continuation import (
    TemporaryServiceRecord, advance_temporary_service_record, reference_for_service,
    require_temporary_service_capacity, temporary_service_key,
)

reserved, admitted, returned = [TemporaryServiceRecord.model_validate(value)
                              for value in json.load(sys.stdin)]
ticket = reserved.intent.ticket
assert ContinuationRecord.__module__ == "cayu.sessions._session_continuation"
restored_ticket = ContinuationTicket.model_validate_json(ticket.model_dump_json())
require_ticket_identity(ticket, restored_ticket)
assert continuation_operation_key(ticket) == continuation_operation_key(restored_ticket)
assert advance_temporary_service_record(reserved, admitted) == admitted
assert advance_temporary_service_record(admitted, returned) == returned
try:
    advance_temporary_service_record(returned, reserved)
except ContinuationConflict:
    pass
else:
    raise AssertionError("A returned service must reject regression to reserved")
for service in (reserved, admitted, returned):
    require_temporary_service_capacity(service)
    reference = reference_for_service(service, ())
    assert reference.key == temporary_service_key(service.intent.operation)
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        restored = pickle.loads(pickle.dumps(service, protocol=protocol))
        assert type(restored) is TemporaryServiceRecord
        assert restored == service
        assert continuation_digest(restored) == continuation_digest(service)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        input=json.dumps([value.model_dump(mode="json") for value in service_records()]),
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
