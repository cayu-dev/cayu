from __future__ import annotations

import asyncio
import json
import os
import signal
import sqlite3
from contextlib import closing

import pytest
from examples.counterfactual_approval.deployment import DeploymentState
from worker_harness import BackendConfig, RecoveryHarness

from cayu.core import ToolResultPart

pytestmark = [
    pytest.mark.process,
    pytest.mark.sigkill_recovery,
    pytest.mark.skipif(os.name != "posix", reason="requires real POSIX SIGKILL"),
]


def _effect(path, session_id):
    # Read committed rows without bootstrap writes while the doomed worker may
    # hold an actual open receipt transaction.
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as connection:
        rows = connection.execute(
            "SELECT record_json FROM cayu_session_operations WHERE session_id=? "
            "AND idempotency_key LIKE 'tool-effect:v1:%'",
            (session_id,),
        ).fetchall()
    assert len(rows) == 1
    return json.loads(rows[0][0])


@pytest.mark.parametrize("approval_gate", [False, True], ids=["ordinary", "approval"])
@pytest.mark.parametrize(
    "phase",
    [
        "after_preparation",
        "before_invocation",
        "during_execution",
        "after_external_completion",
        "receipt_transaction",
        "before_continuation",
    ],
)
def test_public_receipt_recovery_after_sigkill(tmp_path, phase, approval_gate):
    backend = BackendConfig.sqlite(tmp_path)
    external_path = tmp_path / "downstream.sqlite"
    session_id = "effect-process-loss"
    with RecoveryHarness(tmp_path, backend) as harness:
        worker = harness.launch(
            scenario="tool_effect",
            action="start",
            session_id=session_id,
            crash_phase=phase,
            approval_gate=approval_gate,
            external_path=str(external_path),
        )
        reached = worker.wait_for_phase(phase)
        prior = _effect(backend.session_path, session_id)
        key = prior["intent"]["idempotency_key"]
        external = DeploymentState(external_path)
        expected_mutations = int(
            phase not in {"after_preparation", "before_invocation", "during_execution"}
        )
        assert external.mutation_count == expected_mutations
        assert len(external.receipts) == expected_mutations
        assert external.invocation_count(key) == int(
            phase not in {"after_preparation", "before_invocation"}
        )
        if phase == "after_preparation":
            assert prior["state"] == "prepared"
            assert prior["dispatch_id"] is None and prior["terminal"] is None
        elif phase == "before_continuation":
            assert prior["state"] == "reconciled_completed"
            assert prior["terminal"] is not None
        elif phase == "receipt_transaction":
            assert prior["state"] == "outcome_unknown"
            assert prior["reconciliation_attempt"] is not None
            assert prior["terminal"] is None
        else:
            assert prior["state"] == "executing"
            assert prior["terminal"] is None
        worker.sigkill()
        assert worker.process.returncode == -signal.SIGKILL
        assert _effect(backend.session_path, session_id) == prior

        def recover():
            # This worker performs incomplete-session recovery, receipt lookup,
            # continuation and exact replay in one fresh process. Successful
            # local runs already approach the harness's default 20s limit;
            # leave headroom for contended CI without changing runtime deadlines
            # or any of the durable-state/cardinality assertions below.
            return harness.launch(
                scenario="tool_effect",
                action="recover",
                session_id=session_id,
                crash_phase=phase,
                approval_gate=approval_gate,
                external_path=str(external_path),
                replay_request=reached["replay_request"]
                if phase == "before_continuation"
                else None,
            ).wait_success(timeout=60.0)

        recovered = recover()
        if phase == "after_preparation":
            assert recovered["status"] == "completed"
            assert recovered["mutation_count"] == recovered["receipt_count"] == 0
            assert external.invocation_count(key) == 0
            selected = _effect(backend.session_path, session_id)
            assert selected["state"] == "failed" and selected["revision"] == 1
            assert selected["dispatch_id"] is None and selected["terminal"]["receipt"] is None
            assert selected["intent"] == prior["intent"]
            state = asyncio.run(harness.load_session_state(session_id))
            terminals = [
                e
                for e in state.events
                if e.type.value in {"tool.call.completed", "tool.call.failed"}
            ]
            assert len(terminals) == 1 and terminals[0].id == selected["terminal"]["event_id"]
            assert terminals[0].payload["result"]["structured"]["executed"] is False
            assert (
                sum(
                    isinstance(part, ToolResultPart)
                    for message in state.transcript
                    for part in message.content
                )
                == 1
            )
            assert harness.read_marker() == [
                {"model_continuation": False},
                {"model_continuation": True},
            ]
            return
        if phase in {"before_invocation", "during_execution"}:
            assert recovered["status"] == "interrupted"
            assert recovered["mutation_count"] == recovered["receipt_count"] == 0
            assert _effect(backend.session_path, session_id)["state"] == "outcome_unknown"
            assert harness.read_marker() == [{"model_continuation": False}]
            if phase == "before_invocation":
                assert external.invocation_count(key) == 0
                return
            # The external service completes independently of the dead Cayu
            # process. A new explicit lookup observes it; no tool is invoked.
            external.complete(key)
            recovered = recover()
        assert recovered["status"] == "completed"
        assert recovered["mutation_count"] == recovered["receipt_count"] == 1
        assert external.invocation_count(key) == 1
        selected = _effect(backend.session_path, session_id)
        assert selected["state"] == "reconciled_completed"
        assert selected["intent"] == prior["intent"]
        assert selected["dispatch_id"] == prior["dispatch_id"]
        state = asyncio.run(harness.load_session_state(session_id))
        terminals = [
            e for e in state.events if e.type.value in {"tool.call.completed", "tool.call.failed"}
        ]
        assert len(terminals) == 1
        assert terminals[0].id == selected["terminal"]["event_id"]
        assert selected["terminal"]["receipt"]["receipt_id"] == external.receipts[key].receipt_id
        assert (
            sum(
                isinstance(part, ToolResultPart)
                for message in state.transcript
                for part in message.content
            )
            == 1
        )
        assert harness.read_marker() == [
            {"model_continuation": False},
            {"model_continuation": True},
        ]
