"""Independent Runtime validation of the paired Cloud policy contract."""

import json
from hashlib import sha256
from pathlib import Path

import pytest

from cayu.runtime._policy_contract import _decision, _report_receipt, _snapshot
from cayu.runtime._policy_wire import PolicyContractError, canonical, decode

VECTORS = json.loads(
    (Path(__file__).parents[1] / "fixtures/model_policy/contract.json").read_text()
)


def test_paired_effective_digest_and_decisions():
    assert (
        sha256(canonical(VECTORS["effective"])).hexdigest() == VECTORS["expected_effective_sha256"]
    )
    for name in ("report", "refusal"):
        assert _decision(canonical(VECTORS[name])) == VECTORS[name]


def receipt(report):
    return {
        "schema_version": 1,
        "kind": "refusal_receipt",
        "receipt_id": "receipt-2",
        "report": report,
        "report_sha256": sha256(canonical(report)).hexdigest(),
        "accepted_at": "2026-09-28T12:02:00.000000Z",
        "knowledge_valid_until": "2026-09-28T12:07:00.000000Z",
    }


def test_receipt_requires_complete_expected_decision_not_self_consistency():
    report = VECTORS["refusal"]
    assert _report_receipt(canonical(receipt(report)), expected_report=canonical(report))
    changed = report | {"reason": "model_unsupported"}
    with pytest.raises(PolicyContractError):
        _report_receipt(canonical(receipt(changed)), expected_report=canonical(report))


@pytest.mark.parametrize(
    "changes",
    [
        {"decision_seq": True},
        {"decision_seq": 0},
        {"operation_id": "report\n"},
        {"reason": "unknown"},
        {"snapshot": None},
        {"extra": "private-canary"},
    ],
)
def test_refusal_rejects_invalid_authority(changes):
    with pytest.raises(PolicyContractError, match="Model policy contract validation failed"):
        _decision(canonical(VECTORS["refusal"] | changes))


@pytest.mark.parametrize(
    "wire",
    [
        b'{"v": 1, "v": 2}',
        b'{"v": 1.0}',
        b'{"v": -1}',
        b'{"v": NaN}',
        b'{"v": ' + b"[" * 20 + b"0" + b"]" * 20 + b"}",
    ],
)
def test_codec_fails_without_financial_schema_or_coercion(wire):
    with pytest.raises(PolicyContractError):
        decode(wire)


def test_snapshot_correlates_scope_and_exact_effective_content():
    effective = VECTORS["effective"]
    snapshot = {
        "schema_version": 1,
        "kind": "policy_snapshot",
        "snapshot_id": "snapshot-1",
        "incarnation_id": "incarnation-1",
        "incarnation_epoch": 1,
        "effective": effective,
        "config_sha256": VECTORS["expected_effective_sha256"],
        "issued_at": "2026-09-28T12:00:00.000000Z",
        "valid_until": "2026-09-28T12:01:00.000000Z",
    }
    assert _snapshot(canonical(snapshot), scope=effective["scope"]) == snapshot
    with pytest.raises(PolicyContractError):
        _snapshot(canonical(snapshot), scope=effective["scope"] | {"application_id": "other"})
