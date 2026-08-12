"""Secret-free local evidence records for Cayu Cloud commands."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cayu.cli._cloud_api import CloudApiError

EVIDENCE_SCHEMA_VERSION = 1
_SAFE_OPERATION = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_SENSITIVE_KEY = re.compile(
    r"(?:^|_)(?:api_?key|authorization|credential|password|secret|token)(?:$|_)",
    re.IGNORECASE,
)
_SENSITIVE_VALUE = re.compile(
    r"(?:cayu_live_[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|"
    r"AIza[0-9A-Za-z_-]{20,}|AKIA[0-9A-Z]{16})"
)
_REDACTED = "[redacted]"


@dataclass(frozen=True)
class EvidenceRecorder:
    directory: Path

    def preflight(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor, probe = tempfile.mkstemp(
            dir=self.directory,
            prefix=".cayu-evidence-",
            suffix=".tmp",
        )
        os.close(descriptor)
        Path(probe).unlink()

    def redact(self, result: dict[str, Any]) -> dict[str, Any]:
        safe_result = _redact_sensitive(result)
        if not isinstance(safe_result, dict):
            raise TypeError("Evidence result must remain an object after redaction.")
        _reject_sensitive(safe_result)
        return safe_result

    def record(
        self,
        operation: str,
        *,
        result: dict[str, Any],
        correlation_id: str,
    ) -> Path:
        if _SAFE_OPERATION.fullmatch(operation) is None:
            raise ValueError("Evidence operation is invalid.")
        safe_result = self.redact(result)
        timestamp = datetime.now(UTC)
        record: dict[str, Any] = {
            "correlation_id": correlation_id,
            "operation": operation,
            "recorded_at": timestamp.isoformat(),
            "result": safe_result,
            "schema_version": EVIDENCE_SCHEMA_VERSION,
        }
        record["content_digest"] = _content_digest(record)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = self.directory / (
            f"{timestamp.strftime('%Y%m%dT%H%M%S%fZ')}-{operation}-{secrets.token_hex(4)}.json"
        )
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        path.chmod(0o600)
        return path

    def list_records(self) -> list[dict[str, Any]]:
        if not self.directory.exists():
            return []
        return [self._load(path) for path in sorted(self.directory.glob("*.json"))]

    def show(self, evidence_id: str) -> dict[str, Any]:
        if "/" in evidence_id or evidence_id in {".", ".."}:
            raise CloudApiError("evidence_id_invalid", "Evidence ID is invalid.")
        path = self.directory / evidence_id
        if path.suffix != ".json":
            path = path.with_suffix(".json")
        if path.parent != self.directory:
            raise CloudApiError("evidence_id_invalid", "Evidence ID is invalid.")
        return self._load(path)

    def verify(self) -> dict[str, Any]:
        records = self.list_records()
        return {
            "directory": str(self.directory),
            "records": len(records),
            "valid": True,
        }

    @staticmethod
    def _load(path: Path) -> dict[str, Any]:
        try:
            raw = path.read_text()
            payload = json.loads(raw)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise CloudApiError(
                "evidence_invalid",
                f"Evidence record is unreadable: {path.name}",
            ) from exc
        if not isinstance(payload, dict):
            raise CloudApiError("evidence_invalid", f"Evidence record is invalid: {path.name}")
        digest = payload.pop("content_digest", None)
        if (
            payload.get("schema_version") != EVIDENCE_SCHEMA_VERSION
            or not isinstance(digest, str)
            or digest != _content_digest(payload)
        ):
            raise CloudApiError(
                "evidence_invalid",
                f"Evidence digest is invalid: {path.name}",
            )
        _reject_sensitive(payload)
        payload["content_digest"] = digest
        payload["evidence_id"] = path.name
        return payload


def _content_digest(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _redact_sensitive(value: Any) -> Any:
    if isinstance(value, str):
        return _REDACTED if _SENSITIVE_VALUE.search(value) else value
    if isinstance(value, dict):
        return {
            str(child_key): (
                {str(environment_key): _REDACTED for environment_key in child}
                if str(child_key).casefold() == "environment" and isinstance(child, dict)
                else (
                    _REDACTED if _SENSITIVE_KEY.search(str(child_key)) else _redact_sensitive(child)
                )
            )
            for child_key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_sensitive(child) for child in value]
    return value


def _reject_sensitive(value: object, *, key: str | None = None) -> None:
    if key is not None and _SENSITIVE_KEY.search(key):
        if value == _REDACTED:
            return
        raise CloudApiError(
            "evidence_contains_secret",
            f"Evidence contains a sensitive field: {key}",
        )
    if isinstance(value, str) and _SENSITIVE_VALUE.search(value):
        raise CloudApiError("evidence_contains_secret", "Evidence contains secret-like material.")
    if isinstance(value, dict):
        for child_key, child in value.items():
            _reject_sensitive(child, key=str(child_key))
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_sensitive(child)
