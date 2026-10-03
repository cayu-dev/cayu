"""Optional runner-owned durable command capability.

The private descriptor is stored by the existing tool journal before dispatch.
It is never a public runner-operation identity or a model-facing result.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from cayu.runners.base import ExecCommand, ExecResult
from cayu.vaults import SecretRedactor


@runtime_checkable
class DurableCommandObserver(Protocol):
    async def observe_command_receipt(
        self, identity: dict[str, Any], receipt: dict[str, Any]
    ) -> ExecResult | None:
        """Read an exact saved allocation; never create, reconnect or dispose it."""
        ...


def redact_command_receipt_result(
    result: ExecResult, redactor: SecretRedactor, limit: int
) -> ExecResult:
    updates: dict[str, Any] = {}
    for stream in ("stdout", "stderr"):
        truncated = getattr(result, stream + "_truncated")
        if limit == 0:
            value, withheld = "", bool(getattr(result, stream))
        else:
            value, withheld = redactor.redact_utf8_head(
                getattr(result, stream).encode("utf-8"),
                max_bytes=limit,
                source_complete=not truncated,
            )
        updates[stream] = value
        updates[stream + "_truncated"] = truncated or withheld
    return result.model_copy(update=updates)


class DurableCommandRunner(ABC):
    @abstractmethod
    def prepare_command_receipt(
        self,
        identity: dict[str, Any],
        *,
        command: ExecCommand,
        cwd: str | None,
        env: dict[str, str] | None,
        timeout_s: int,
        output_limit_bytes: int,
    ) -> dict[str, Any] | None:
        """Return private, serializable launch authority, without dispatching."""

    @abstractmethod
    async def exec_with_receipt(
        self,
        command: ExecCommand,
        *,
        receipt: dict[str, Any],
        redactor: SecretRedactor,
        cwd: str | None,
        env: dict[str, str] | None,
        env_remove: tuple[str, ...] = (),
        timeout_s: int | None,
        stdin: str | None,
        output_limit_bytes: int | None,
    ) -> ExecResult:
        """Dispatch only the exact saved request; retain ordinary cleanup ownership."""

    @abstractmethod
    async def observe_command_receipt(
        self, identity: dict[str, Any], receipt: dict[str, Any]
    ) -> ExecResult | None:
        """Observe the exact allocation without dispatch, allocation, or cleanup.

        Missing/invalid/conflicting evidence is None, never a no-effect receipt.
        Returned raw output remains private until Runtime's redaction boundary.
        """
