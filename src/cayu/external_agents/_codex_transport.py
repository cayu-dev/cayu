"""Bounds before the pinned SDK parses or queues untrusted guest frames."""

from __future__ import annotations

import json
import threading
from contextlib import suppress
from typing import Any

from cayu.external_agents.base import ExternalAgentError

_MAX_FRAME_BYTES = 8 * 1024 * 1024
_MAX_TRANSPORT_BYTES = 64 * 1024 * 1024
_MAX_TRANSPORT_MESSAGES = 100_000


class BoundedCodexTransport:
    """Mixin for the pinned CodexClient's sole reader and stderr drain.

    Count every frame, including global and pre-initialization notifications.
    Cumulative limits conservatively bound all SDK queues even if a queue is
    never consumed. SDK observations keep their separate recorder limits.
    """

    _proc: Any = None
    _stderr_thread: threading.Thread | None
    _transport_bytes: int = 0
    _transport_messages: int = 0

    def stop_transport(self) -> None:
        """Unblock SDK writes even when guest cleanup reported a failure."""
        if self._proc is not None:
            with suppress(ProcessLookupError):
                self._proc.kill()
            self._proc.wait()

    def _read_message(self) -> dict[str, Any]:
        if self._proc is None or self._proc.stdout is None:
            raise ExternalAgentError("Codex transport is closed.")
        # Read bytes before TextIOWrapper decoding or an unbounded readline.
        line = self._proc.stdout.buffer.readline(_MAX_FRAME_BYTES + 1)
        if not line:
            raise ExternalAgentError("Codex transport closed before completion.")
        self._transport_bytes += len(line)
        self._transport_messages += 1
        if (
            len(line) > _MAX_FRAME_BYTES
            or self._transport_bytes > _MAX_TRANSPORT_BYTES
            or self._transport_messages > _MAX_TRANSPORT_MESSAGES
        ):
            raise ExternalAgentError("Codex transport exceeds its observation size limit.")
        try:
            message = json.loads(line)
        except (ValueError, RecursionError):
            raise ExternalAgentError("Codex returned an invalid transport frame.") from None
        if not isinstance(message, dict):
            raise ExternalAgentError("Codex returned an invalid transport frame.")
        return message

    def _start_stderr_drain_thread(self) -> None:
        if self._proc is None or self._proc.stderr is None:
            return
        stderr = self._proc.stderr.buffer

        def discard() -> None:
            # Guest diagnostics are not a governed publication channel. Drain
            # fixed-size chunks so neither giant lines nor secrets are retained.
            try:
                while stderr.read(64 * 1024):
                    pass
            except (OSError, ValueError):
                pass

        self._stderr_thread = threading.Thread(target=discard, daemon=True)
        self._stderr_thread.start()
