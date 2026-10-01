import json
from hashlib import sha256
from pathlib import Path

import pytest

from cayu.runtime._policy_journal import (
    acknowledge_decision,
    append_decision,
    load_journal,
)
from cayu.runtime._policy_wire import PolicyContractError, canonical

VECTORS = json.loads(
    (Path(__file__).parents[1] / "fixtures/model_policy/contract.json").read_text()
)
AUTHORITY = {"scope": VECTORS["effective"]["scope"], "incarnation": ("incarnation-1", 1)}


def report(sequence=1, **changes):
    return canonical(
        {
            **VECTORS["report"],
            "operation_id": f"policy-{sequence}",
            "installation_seq": sequence,
            "installation_id": f"install-{sequence}",
            **changes,
        }
    )


def receipt(wire):
    value = json.loads(wire)
    return canonical(
        {
            "schema_version": 1,
            "kind": "refusal_receipt" if value["kind"] == "adoption_refusal" else "report_receipt",
            "receipt_id": f"receipt-{value['operation_id']}",
            "report": value,
            "report_sha256": sha256(wire).hexdigest(),
            "accepted_at": "2026-09-28T12:00:20.000000Z",
            "knowledge_valid_until": "2026-09-28T12:05:20.000000Z",
        }
    )


def test_installation_and_pending_report_are_one_transition():
    wire = report()
    body = append_decision(b"{}", wire, **AUTHORITY)
    state = load_journal(body, **AUTHORITY)
    assert state["current"] == json.loads(wire)
    assert state["reports"] == [{"report": json.loads(wire), "receipt": None}]
    assert append_decision(body, wire, **AUTHORITY) == body
    for changed in (
        report(installed_model="model-b"),
        report(installation_id="different"),
        report(locally_committed_at="2026-09-28T12:00:11.000000Z"),
    ):
        with pytest.raises(PolicyContractError):
            append_decision(body, changed, **AUTHORITY)
    acknowledged = acknowledge_decision(body, receipt(wire), expected_report=wire, **AUTHORITY)
    assert (
        acknowledge_decision(acknowledged, receipt(wire), expected_report=wire, **AUTHORITY)
        == acknowledged
    )


def test_receipt_for_different_self_consistent_report_is_rejected():
    wire = report()
    body = append_decision(b"{}", wire, **AUTHORITY)
    with pytest.raises(PolicyContractError):
        acknowledge_decision(
            body, receipt(report(installed_model="model-b")), expected_report=wire, **AUTHORITY
        )


def test_refusal_preserves_installed_default_and_shares_sequence():
    installed = report()
    body = append_decision(b"{}", installed, **AUTHORITY)
    refusal = canonical({**VECTORS["refusal"], "operation_id": "policy-2", "decision_seq": 2})
    body = append_decision(body, refusal, **AUTHORITY)
    state = load_journal(body, **AUTHORITY)
    assert state["current"] == json.loads(installed)
    assert state["sequence"] == 2 and len(state["reports"]) == 2
    with pytest.raises(PolicyContractError):
        append_decision(body, report(2), **AUTHORITY)


@pytest.mark.parametrize("action", ["override_default", "withdraw"])
def test_local_control_pauses_automatic_adoption(action):
    body = append_decision(b"{}", report(), **AUTHORITY)
    local = report(2, action=action, snapshot=None)
    body = append_decision(body, local, **AUTHORITY)
    assert load_journal(body, **AUTHORITY)["adoption_enabled"] is False
    with pytest.raises(PolicyContractError):
        append_decision(body, report(3), **AUTHORITY)
    refusal = canonical({**VECTORS["refusal"], "operation_id": "policy-3", "decision_seq": 3})
    with pytest.raises(PolicyContractError):
        append_decision(body, refusal, **AUTHORITY)


def test_withdraw_cannot_silently_replace_installed_model():
    body = append_decision(b"{}", report(), **AUTHORITY)
    with pytest.raises(PolicyContractError):
        append_decision(
            body,
            report(2, action="withdraw", snapshot=None, installed_model="model-b"),
            **AUTHORITY,
        )


def test_retention_never_prunes_pending_and_old_sequence_cannot_replay(monkeypatch):
    monkeypatch.setattr("cayu.runtime._policy_journal.RETAINED_REPORTS", 2)
    body = append_decision(b"{}", report(), **AUTHORITY)
    body = append_decision(body, report(2), **AUTHORITY)
    with pytest.raises(PolicyContractError):
        append_decision(body, report(3), **AUTHORITY)
    body = acknowledge_decision(body, receipt(report()), expected_report=report(), **AUTHORITY)
    body = append_decision(body, report(3), **AUTHORITY)
    state = load_journal(body, **AUTHORITY)
    assert state["replay_floor"] == 1
    assert [entry["report"]["operation_id"] for entry in state["reports"]] == [
        "policy-2",
        "policy-3",
    ]
    with pytest.raises(PolicyContractError):
        append_decision(body, report(), **AUTHORITY)
    with pytest.raises(PolicyContractError):
        append_decision(body, report(4, operation_id="policy-1"), **AUTHORITY)
