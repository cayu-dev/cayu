"""Session-bound failure evidence for paused-round continuations."""

from __future__ import annotations

from typing import Any

from cayu.failure_evidence import exception_evidence
from cayu.runtime._environment_lifecycle import exception_failure_payload
from cayu.runtime._foreground_subagent_recovery import ForegroundSubagentRecoveryRequired
from cayu.sessions.records import Session
from cayu.vaults.redaction import SecretRedactor


def _continuation_failure_payload(
    error: BaseException, *, session: Session, redactor: SecretRedactor
) -> dict[str, Any]:
    return {
        **exception_failure_payload(error, redactor=redactor),
        **(
            error.interruption_evidence()
            if isinstance(error, ForegroundSubagentRecoveryRequired)
            else {}
        ),
        "failure_evidence": exception_evidence(error)
        .model_copy(update={"session_id": session.id, "run_epoch": session.run_epoch})
        .model_dump(mode="json"),
    }
