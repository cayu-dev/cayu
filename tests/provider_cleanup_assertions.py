"""Check diagnostic context without coupling assertions to source line numbers."""

from __future__ import annotations

import json
from typing import Any


def without_redacted_cleanup_context(failure: dict[str, Any]) -> dict[str, Any]:
    copied = dict(failure)
    if "cleanup_diagnostic_version" not in copied:
        return copied
    assert copied.pop("cleanup_exception_message") == "redacted"
    if "cleanup_cause_type" in copied:
        assert copied.pop("cleanup_cause_type") == "CancelledError"
        assert copied.pop("cleanup_cause_message") == "redacted"
    frames = json.loads(copied.pop("cleanup_local_stack"))
    assert 1 <= len(frames) <= 8
    for filename, line in frames:
        assert filename in {"_credential_boundary.py", "_http.py"}
        assert type(line) is int and 1 <= line <= 1_000_000
    return copied
