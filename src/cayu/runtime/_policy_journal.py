"""Atomic installation/report state transitions, with bounded replay retention.

The storage owner supplies CAS and fencing. These pure transitions never publish
an installation independently of its pending authenticated report. A sequence
floor prevents pruned operation identities from becoming fresh mutations.
"""

from __future__ import annotations

from typing import Any

from cayu.runtime._policy_contract import _decision, _report_receipt
from cayu.runtime._policy_storage import parse_state, state_bytes
from cayu.runtime._policy_wire import canonical, require

MAX_PENDING_REPORTS = 128
RETAINED_REPORTS = 256


def empty_journal() -> dict[str, Any]:
    return {
        "version": 1,
        "sequence": 0,
        "replay_floor": 0,
        "adoption_enabled": True,
        "current": None,
        "reports": [],
        "observation": None,
        "publication": None,
    }


def _sequence(report: dict[str, Any]) -> int:
    return (
        report["decision_seq"]
        if report["kind"] == "adoption_refusal"
        else report["installation_seq"]
    )


def _local_identity(report: dict[str, Any]) -> None:
    # Cloud accepts opaque operation IDs. This local producer uses a sequence-
    # bound namespace so replay remains safe after an acknowledged prefix is
    # pruned; an old identity can never be resubmitted with a new sequence.
    require(report["operation_id"] == f"policy-{_sequence(report)}")


def load_journal(
    body: bytes, *, scope: dict[str, str], incarnation: tuple[str, int]
) -> dict[str, Any]:
    value = parse_state(body)
    if value == {}:
        return empty_journal()
    require(set(value) == set(empty_journal()))
    require(type(value["version"]) is int and value["version"] == 1)
    require(type(value["adoption_enabled"]) is bool)
    if value["observation"] is not None:
        from cayu.runtime._policy_freshness import validate_observation

        validate_observation(value["observation"], scope=scope, incarnation=incarnation)
    sequence, floor = value["sequence"], value["replay_floor"]
    require(type(sequence) is int and type(floor) is int and 0 <= floor <= sequence < 2**53)
    records = value["reports"]
    require(type(records) is list and len(records) <= RETAINED_REPORTS)
    identities = set()
    prior = floor
    pending = 0
    for record in records:
        require(type(record) is dict and set(record) == {"report", "receipt"})
        wire = canonical(record["report"])
        report = _decision(wire)
        _local_identity(report)
        require(report["scope"] == scope)
        require((report["incarnation_id"], report["incarnation_epoch"]) == incarnation)
        require(_sequence(report) == prior + 1)
        prior = _sequence(report)
        require(report["operation_id"] not in identities)
        identities.add(report["operation_id"])
        if record["receipt"] is None:
            pending += 1
        else:
            _report_receipt(canonical(record["receipt"]), expected_report=wire)
    require(prior == sequence and pending <= MAX_PENDING_REPORTS)
    current = value["current"]
    if current is not None:
        report = _decision(canonical(current))
        _local_identity(report)
        require(report["kind"] == "installation_report" and report["scope"] == scope)
        require((report["incarnation_id"], report["incarnation_epoch"]) == incarnation)
        require(_sequence(report) <= sequence)
        retained = [
            item["report"] for item in records if _sequence(item["report"]) == _sequence(report)
        ]
        require(not retained or retained == [report])
    installations = [
        item["report"] for item in records if item["report"]["kind"] == "installation_report"
    ]
    require(not installations or installations[-1] == current)
    publication = value["publication"]
    if publication is not None:
        require(
            type(publication) is dict
            and set(publication) == {"prior", "operation_id", "deadline_ns", "completed_ns"}
        )
        require(records and records[-1]["report"]["operation_id"] == publication["operation_id"])
        deadline, completed = publication["deadline_ns"], publication["completed_ns"]
        require(type(deadline) is int and 0 < deadline < 2**63)
        if completed is None:
            prior_state = publication["prior"]
            require(type(prior_state) is dict and prior_state.get("publication") is None)
            prior_state = load_journal(
                state_bytes(prior_state), scope=scope, incarnation=incarnation
            )
            rebuilt = parse_state(
                append_decision(
                    state_bytes(prior_state),
                    canonical(records[-1]["report"]),
                    scope=scope,
                    incarnation=incarnation,
                )
            )
            require(rebuilt == {**value, "publication": None})
        else:
            require(type(completed) is int and 0 <= completed < deadline)
            require(publication["prior"] is None)
    return value


def append_decision(
    body: bytes,
    report_wire: bytes,
    *,
    scope: dict[str, str],
    incarnation: tuple[str, int],
) -> bytes:
    """Exact replay compares the entire report; new writes must advance once."""
    state = load_journal(body, scope=scope, incarnation=incarnation)
    require(state["publication"] is None or state["publication"]["completed_ns"] is not None)
    state["publication"] = None
    report = _decision(report_wire)
    _local_identity(report)
    require(report["scope"] == scope)
    require((report["incarnation_id"], report["incarnation_epoch"]) == incarnation)
    for record in state["reports"]:
        if record["report"]["operation_id"] == report["operation_id"]:
            require(record["report"] == report)
            return state_bytes(state)
    require(_sequence(report) == state["sequence"] + 1)
    if report["kind"] == "adoption_refusal":
        require(state["adoption_enabled"])
    elif report["action"] == "withdraw":
        current = state["current"]
        require(
            report["installed_model"] == (None if current is None else current["installed_model"])
        )
    require(sum(record["receipt"] is None for record in state["reports"]) < MAX_PENDING_REPORTS)
    # Evict only an acknowledged prefix. Pending work never loses its retry owner.
    while len(state["reports"]) >= RETAINED_REPORTS:
        require(state["reports"][0]["receipt"] is not None)
        removed = state["reports"].pop(0)
        state["replay_floor"] = _sequence(removed["report"])
    state["reports"].append({"report": report, "receipt": None})
    state["sequence"] = _sequence(report)
    if report["kind"] == "installation_report":
        state["current"] = report
        if report["action"] in ("override_default", "withdraw"):
            state["adoption_enabled"] = False
        else:
            require(state["adoption_enabled"])
    return state_bytes(state)


def acknowledge_decision(
    body: bytes,
    receipt_wire: bytes,
    *,
    expected_report: bytes,
    scope: dict[str, str],
    incarnation: tuple[str, int],
) -> bytes:
    state = load_journal(body, scope=scope, incarnation=incarnation)
    expected = _decision(expected_report)
    receipt = _report_receipt(receipt_wire, expected_report=expected_report)
    for record in state["reports"]:
        if record["report"] == expected:
            require(record["receipt"] is None or record["receipt"] == receipt)
            record["receipt"] = receipt
            return state_bytes(state)
    require(False)
    raise AssertionError("unreachable")
