"""Experimental helpers for runner adapters maintained outside Cayu.

Status: experimental. These are the same objects Cayu's built-in sandbox
runners use; names and signatures may change between releases until the
supported extension contract and compatibility tiers are finalized.

- ``register_runner_adapter_identity`` trusts an adapter name (and optional
  exception class names) in durable runner diagnostics and cleanup receipts.
  Unregistered names are reported as ``"unknown"`` and unregistered exception
  classes as ``"Exception"``. Registration is explicit and process-wide.
- Cleanup: ``cleanup_runner_command_with_diagnostic`` performs one bounded
  command or sandbox kill and returns a ``RunnerCleanupResult`` carrying a
  ``cayu.runner_cleanup.v1`` receipt; ``validate_cancel_timeout`` and
  ``validate_runner_cleanup_policy`` validate the matching constructor options.
- Output: ``RedactedOutputCapture`` captures a bounded, secret-redacted head of
  streamed output; ``redact_completed_exec_result`` projects a complete
  ``ExecResult`` through a redactor and output limit.
- Request validation: ``validate_timeout``, ``validate_output_limit``,
  ``validate_stdin``, ``copy_runner_env``, ``remove_runner_env`` and
  ``copy_exec_command`` apply the ``Runner.exec`` argument rules.

See ``docs/build-a-runner.md`` for the full description.
"""

from typing import Any as _Any

from cayu._api import resolve_export as _resolve_export
from cayu._api import wildcard_names as _wildcard_names
from cayu.extensions.runners._exports import EXPORTS as _EXPORTS
from cayu.extensions.runners._exports import PUBLIC_NAMES as _PUBLIC_NAMES

__all__ = _wildcard_names(_PUBLIC_NAMES, _EXPORTS)


def __getattr__(name: str) -> _Any:
    return _resolve_export(name, globals(), _EXPORTS)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
