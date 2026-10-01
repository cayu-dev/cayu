"""Pure policy wire validation shared by transport, controller and journal.

Validation establishes shape and correlation, never authentication or adoption.
"""

from __future__ import annotations

import re
from contextlib import suppress
from datetime import datetime, timedelta
from hashlib import sha256
from typing import Any

from cayu.runtime._policy_wire import canonical, decode, identifier, require

_SCOPE = frozenset(
    {
        "organization_id",
        "cloud_agent_id",
        "application_id",
        "instance_id",
        "integration_id",
        "credential_family_id",
        "inference_key_id",
    }
)
_REPORT = frozenset(
    {
        "schema_version",
        "kind",
        "scope",
        "incarnation_id",
        "incarnation_epoch",
        "operation_id",
        "installation_id",
        "installation_seq",
        "target",
        "action",
        "snapshot",
        "installed_model",
        "locally_committed_at",
    }
)


def _report_receipt(wire: bytes, *, expected_report: bytes) -> dict[str, Any]:
    """Validate complete report correlation, without granting authentication."""
    receipt = decode(wire, max_bytes=65536)
    require(
        set(receipt)
        == {
            "schema_version",
            "kind",
            "receipt_id",
            "report",
            "report_sha256",
            "accepted_at",
            "knowledge_valid_until",
        }
    )
    report = _decision(expected_report)
    refusal = report["kind"] == "adoption_refusal"
    require(type(receipt["schema_version"]) is int and receipt["schema_version"] == 1)
    require(receipt["kind"] == ("refusal_receipt" if refusal else "report_receipt"))
    identifier(receipt["receipt_id"])
    body = canonical(report)
    require(canonical(receipt["report"]) == body)
    require(receipt["report_sha256"] == sha256(body).hexdigest())
    require(
        _timestamp(receipt["knowledge_valid_until"])
        == _timestamp(receipt["accepted_at"]) + timedelta(seconds=300)
    )
    return receipt


REFUSAL_REASONS = frozenset(
    {"default_absent", "default_ineligible", "model_unknown", "model_unsupported"}
)


def _decision(wire: bytes) -> dict[str, Any]:
    value = decode(wire)
    if value.get("kind") == "installation_report":
        return _report(wire)
    require(
        set(value)
        == {
            "schema_version",
            "kind",
            "scope",
            "incarnation_id",
            "incarnation_epoch",
            "operation_id",
            "decision_seq",
            "target",
            "snapshot",
            "reason",
            "observed_at",
        }
    )
    require(type(value["schema_version"]) is int and value["schema_version"] == 1)
    require(value["kind"] == "adoption_refusal")
    require(value["target"] == "application_default")
    _scope(value["scope"])
    for key in ("incarnation_id", "operation_id"):
        identifier(value[key])
    for key in ("incarnation_epoch", "decision_seq"):
        _integer(value[key])
    require(type(value["reason"]) is str and value["reason"] in REFUSAL_REASONS)
    _timestamp(value["observed_at"])
    ref = value["snapshot"]
    require(
        type(ref) is dict and set(ref) == {"snapshot_id", "effective_revision", "config_sha256"}
    )
    identifier(ref["snapshot_id"])
    _integer(ref["effective_revision"])
    _digest(ref["config_sha256"])
    return value


def _integer(value: object) -> None:
    require(type(value) is int and 1 <= value <= 2**53 - 1)


def _digest(value: object) -> None:
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None)


def _timestamp(value: object) -> datetime:
    require(
        type(value) is str
        and re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z", value)
        is not None
    )
    result = None
    assert isinstance(value, str)
    with suppress(ValueError):
        result = datetime.fromisoformat(value)
    require(result is not None)
    assert result is not None
    return result


def _scope(value: Any) -> None:
    require(type(value) is dict and set(value) == _SCOPE)
    for item in value.values():
        identifier(item)


def _report(wire: bytes) -> dict[str, Any]:
    value = decode(wire, max_bytes=65536)
    require(set(value) == _REPORT)
    require(type(value["schema_version"]) is int and value["schema_version"] == 1)
    require(value["kind"] == "installation_report" and value["target"] == "application_default")
    _scope(value["scope"])
    for key in ("incarnation_id", "operation_id", "installation_id"):
        identifier(value[key])
    for key in ("incarnation_epoch", "installation_seq"):
        _integer(value[key])
    _timestamp(value["locally_committed_at"])
    require(value["action"] in ("install_default", "override_default", "withdraw"))
    model = value["installed_model"]
    if model is not None:
        identifier(model)
    if value["action"] != "withdraw":
        require(model is not None)
    snapshot = value["snapshot"]
    if value["action"] == "install_default":
        require(
            type(snapshot) is dict
            and set(snapshot) == {"snapshot_id", "effective_revision", "config_sha256"}
        )
        identifier(snapshot["snapshot_id"])
        _integer(snapshot["effective_revision"])
        _digest(snapshot["config_sha256"])
    else:
        require(snapshot is None)
    return value


def _snapshot(wire: bytes, *, scope: dict[str, str]) -> dict:
    value = decode(wire, max_bytes=65536)
    require(
        set(value)
        == {
            "schema_version",
            "kind",
            "snapshot_id",
            "incarnation_id",
            "incarnation_epoch",
            "effective",
            "config_sha256",
            "issued_at",
            "valid_until",
        }
    )
    require(type(value["schema_version"]) is int and value["schema_version"] == 1)
    require(value["kind"] == "policy_snapshot")
    identifier(value["snapshot_id"])
    identifier(value["incarnation_id"])
    _integer(value["incarnation_epoch"])
    effective = value["effective"]
    require(
        type(effective) is dict
        and set(effective)
        == {
            "scope",
            "effective_revision",
            "source_revisions",
            "allowed_models",
            "default_model",
            "default_state",
        }
    )
    _scope(effective["scope"])
    require(effective["scope"] == scope)
    _integer(effective["effective_revision"])
    revisions = effective["source_revisions"]
    require(type(revisions) is dict and set(revisions) == {"organization", "agent", "key"})
    for revision in revisions.values():
        _integer(revision)
    models = effective["allowed_models"]
    require(type(models) is list and len(models) <= 256)
    for model in models:
        identifier(model)
    require(models == sorted(set(models)))
    default = effective["default_model"]
    if default is not None:
        identifier(default)
    state = "absent" if default is None else "eligible" if default in models else "ineligible"
    require(effective["default_state"] == state)
    _digest(value["config_sha256"])
    require(value["config_sha256"] == sha256(canonical(effective)).hexdigest())
    require(
        _timestamp(value["valid_until"]) == _timestamp(value["issued_at"]) + timedelta(seconds=60)
    )
    return value
