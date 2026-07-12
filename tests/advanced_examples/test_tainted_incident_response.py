from __future__ import annotations

import asyncio
from pathlib import Path

from examples.tainted_incident_response.deterministic import run


def test_tainted_incident_is_quarantined_and_only_sanitized_facts_cross_boundary(
    tmp_path: Path,
) -> None:
    result = asyncio.run(run(tmp_path))

    assert result.status == "verified"
    assert result.assertions == {
        "clean_session_received_only_sanitized_artifact": True,
        "fork_inherited_taint": True,
        "hostile_instruction_not_propagated": True,
        "protected_mutation_blocked_after_restart": True,
        "quarantine_outbound_authority_restricted": True,
        "sanitized_notification_sent_once": True,
    }
    assert result.metrics["protected_mutations"] == 0
    assert result.metrics["notifications"] == 1
    assert result.metrics["inherited_taint_labels"] == ["incident-untrusted"]
    receipt = result.outputs["sanitizer_receipt"]
    assert receipt["artifact_id"] == "sanitized-incident-facts:incident-quarantine:v1"
    assert result.outputs["notification_artifact_id"] == receipt["artifact_id"]
    blocked = result.outputs["protected_mutation_block_event"]
    assert blocked["tool_name"] == "rotate_credentials"
    assert blocked["matched_taint_labels"] == ["incident-untrusted"]
