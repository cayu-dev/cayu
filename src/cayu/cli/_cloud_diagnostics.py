"""Validation of the versioned customer-facing Cloud build diagnostic contract."""

from __future__ import annotations

import re
import unicodedata
from typing import NotRequired, TypedDict, cast


class CloudDeploymentFailure(TypedDict):
    automatic_retryable: bool
    code: str
    detail: str
    hint: str
    message: str
    phase: str
    schema_version: NotRequired[int]
    attempt: NotRequired[int | None]
    diagnostic_ref: NotRequired[str | None]
    diagnostic: NotRequired[dict[str, object]]


_PHASES = {
    "source_resolved",
    "image_built",
    "image_scanned",
    "sandbox_template_ready",
    "policy_compiled",
    "smoke_tested",
    # Cloud's service-publication phases for Agents that declare a database.
    "database_provisioned",
    "database_migrated",
}
_PRIVATE = re.compile(
    r"(?:[a-z][a-z0-9+.-]*://|arn:|\b[A-Za-z_][A-Za-z0-9_]*\s*=|"
    r"\b(?:sk|ghp|gho|github_pat|AKIA|ASIA)[_-]?[A-Za-z0-9_-]{8,}|"
    r"[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}|"
    r"\b(?:authorization|password|passwd|[A-Za-z0-9_]*api[_-]?key|[A-Za-z0-9_]*secret|[A-Za-z0-9_]*token|credential)\s*[:=]|"
    r"\b(?:CodeBuild|CloudWatch)\b|[A-Za-z0-9_+/-]{64,})",
    re.I,
)


def safe_text(
    value: object, limit: int, *, multiline: bool = False, empty: bool = False
) -> str | None:
    if not isinstance(value, str) or (not value and not empty):
        return None
    try:
        if len(value.encode()) > limit:
            return None
    except UnicodeError:
        return None
    if any(
        unicodedata.category(c) in {"Cc", "Cf"} and not (multiline and c in "\n\t") for c in value
    ):
        return None
    if _PRIVATE.search(value):
        return None
    return value


def parse_build_failure(value: object) -> CloudDeploymentFailure | None:
    if not isinstance(value, dict):
        return None
    candidate = cast("dict[str, object]", value)
    version = candidate.get("schema_version")
    if type(version) is not int or version != 1:
        return None
    code = safe_text(candidate.get("code"), 64)
    phase = candidate.get("phase")
    detail = safe_text(candidate.get("detail"), 4096, multiline=True)
    hint = safe_text(candidate.get("hint"), 1024)
    message = safe_text(candidate.get("message"), 512)
    retryable = candidate.get("automatic_retryable")
    if (
        code is None
        or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", code) is None
        or not isinstance(phase, str)
        or phase not in _PHASES
        or detail is None
        or hint is None
        or message is None
        or not isinstance(retryable, bool)
    ):
        return None
    attempt = candidate.get("attempt")
    reference = candidate.get("diagnostic_ref")
    if attempt is not None and (type(attempt) is not int or not 1 <= attempt <= 1_000_000):
        return None
    if reference is not None and (
        not isinstance(reference, str)
        or attempt is None
        or reference != f"{phase}:attempt-{attempt}"
    ):
        return None
    diagnostic = candidate.get("diagnostic")
    if not isinstance(diagnostic, dict):
        return None
    raw = cast("dict[str, object]", diagnostic)
    status, stage, reason = raw.get("status"), raw.get("stage"), raw.get("reason")
    excerpt = safe_text(raw.get("excerpt"), 4096, multiline=True, empty=True)
    exit_code = raw.get("exit_code")
    truncated = raw.get("truncated")
    if (
        not isinstance(status, str)
        or status not in {"available", "unavailable", "withheld"}
        or not isinstance(stage, str)
        or stage not in {"source_validation", "docker_build", "image_build", "database_migration"}
        or (
            reason is not None
            and (not isinstance(reason, str) or re.fullmatch(r"[a-z_]{1,64}", reason) is None)
        )
        or excerpt is None
        or not isinstance(truncated, bool)
        or (exit_code is not None and (type(exit_code) is not int or not 0 <= exit_code <= 255))
        or (status == "available" and (reason is not None or not excerpt))
        or (status != "available" and (reason is None or excerpt != ""))
    ):
        return None
    return {
        "schema_version": 1,
        "code": code,
        "phase": phase,
        "message": message,
        "detail": detail,
        "hint": hint,
        "automatic_retryable": retryable,
        "attempt": attempt,
        "diagnostic_ref": reference,
        "diagnostic": {
            "status": status,
            "stage": stage,
            "reason": reason,
            "exit_code": exit_code,
            "excerpt": excerpt,
            "truncated": truncated,
        },
    }
