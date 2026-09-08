"""Exact guest read acknowledgements without adopting unpublished authority."""

from typing import Any

from cayu._validation import canonical_durable_json_bytes


def browser_read_evidence_matches(reply: Any, expected: dict[str, Any]) -> bool:
    actual = canonical_durable_json_bytes(reply, "browser read acknowledgement")
    if actual == canonical_durable_json_bytes(expected, "browser read acknowledgement"):
        return True
    # Native observe can finish before its host terminal publication. Accept only
    # this single difference; callers keep the original durable record/fence.
    return (
        expected.get("state") == "agent_controlled"
        and expected.get("fresh_observation_required") is True
        and actual
        == canonical_durable_json_bytes(
            {**expected, "fresh_observation_required": False}, "browser read acknowledgement"
        )
    )
