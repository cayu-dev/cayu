"""Checkpoint evidence has independent owners and compatible legacy imports."""

from __future__ import annotations

import importlib
import os
import pickle
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

import cayu

_MODULES = (
    "_model_completion_publication",
    "_terminal_evidence",
    "_invocation_terminal_decision",
    "_provider_operation_cancellation_claim",
)
_CONSTANTS = {
    "_model_completion_publication": (
        "LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY",
        "MODEL_STEP_PUBLICATION_CHECKPOINT_RECORD_TYPE",
        "MODEL_STEP_PUBLICATION_CHECKPOINT_SCHEMA_VERSION",
    ),
    "_terminal_evidence": (
        "SESSION_RUN_OPERATION_ID_PAYLOAD_KEY",
        "TERMINAL_EVENT_TYPES",
        "TERMINAL_EVIDENCE_EVENT_TYPES",
        "TERMINAL_EVIDENCE_QUERY_LIMIT",
        "TERMINAL_LIFECYCLE_EVENT_TYPES",
    ),
    "_provider_operation_cancellation_claim": (
        "PROVIDER_OPERATION_CANCELLATION_CLAIM_CHECKPOINT_KEY",
    ),
}


@pytest.mark.parametrize("module_name", _MODULES)
def test_runtime_compatibility_imports_share_canonical_contracts(module_name):
    canonical = importlib.import_module(f"cayu.sessions.{module_name}")
    legacy = importlib.import_module(f"cayu.runtime.{module_name}")
    for name, value in vars(canonical).items():
        if name.startswith("_"):
            continue
        if getattr(value, "__module__", None) == canonical.__name__:
            assert getattr(legacy, name) is value
            assert pickle.loads(f"c{legacy.__name__}\n{name}\n.".encode()) is value
    for name in _CONSTANTS.get(module_name, ()):
        assert getattr(legacy, name) is getattr(canonical, name)


def test_old_pickled_evidence_preserves_serialized_authority():
    from cayu.sessions._invocation_terminal_decision import (
        InvocationTerminalOutcome,
        build_invocation_terminal_decision,
    )
    from cayu.sessions._model_completion_publication import ModelStepPublicationCheckpoint
    from cayu.sessions._provider_operation_cancellation_claim import (
        ProviderOperationCancellationClaim,
    )
    from cayu.sessions._terminal_evidence import CurrentTerminalEvidence

    recovery_id = "model-recovery:" + "a" * 64
    records = (
        ModelStepPublicationCheckpoint(
            logical_step_id="step",
            stage_id="stage",
            source_transcript_cursor=0,
            transcript_end_cursor=1,
            completion_event_id="completion",
            classification={"type": "final"},
            assistant_message_published=True,
        ),
        CurrentTerminalEvidence(events=(), latest_lifecycle_event_type=None),
        build_invocation_terminal_decision(
            outcome=InvocationTerminalOutcome.FAILED,
            session_id="session",
            session_instance_id="instance",
            run_epoch=1,
            profile_interaction_id="interaction",
            interaction_id="interaction",
            execution_profile_fingerprint="b" * 64,
            interaction_event_id=recovery_id + ":interaction_failed",
            terminal_event_id=recovery_id + ":session_failed",
            observed_at=datetime(2026, 1, 1, tzinfo=UTC),
            terminal_payload={"reason": "failure"},
            model_recovery_id=recovery_id,
        ),
        ProviderOperationCancellationClaim(
            claim_id="claim",
            stage_id="stage",
            run_epoch=1,
            operation_id="operation",
            stream_protocol="protocol",
            expires_at=datetime(2026, 1, 2, tzinfo=UTC),
        ),
    )
    for record in records:
        canonical_pickle = pickle.dumps(record, protocol=0)
        legacy_pickle = canonical_pickle.replace(b"cayu.sessions._", b"cayu.runtime._")
        assert legacy_pickle != canonical_pickle
        restored = pickle.loads(legacy_pickle)
        assert type(restored) is type(record)
        assert restored == record
        assert pickle.loads(pickle.dumps(record)) == record


@pytest.mark.parametrize("module_name", (*_MODULES, "checkpoints"))
def test_session_evidence_imports_without_runtime(module_name):
    script = """
import importlib
import importlib.abc
import sys

class RejectRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "cayu.runtime" or fullname.startswith("cayu.runtime."):
            raise AssertionError(f"Session evidence imported {fullname}")

sys.meta_path.insert(0, RejectRuntime())
importlib.import_module(sys.argv[1])
assert not any(name == "cayu.runtime" or name.startswith("cayu.runtime.") for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, f"cayu.sessions.{module_name}"],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
