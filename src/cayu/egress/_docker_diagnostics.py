"""Conservative Docker stderr projection for durable setup failures.

Docker can echo arbitrary argv, URLs, environment values, and file contents.
A denylist of credential spellings cannot make that text safe. Retain only
literal diagnostic phrases; never copy names, addresses, paths, or payloads.
Unknown text is explicitly redacted so this projection is not mistaken for the
complete daemon response. Add new phrases only when they contain no variables.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

_MAX_STDERR_BYTES = 1024
_MAX_SCAN_CHARACTERS = 65536
_REDACTED = "[REDACTED]"
_PHRASES = (
    "Error response from daemon:",
    "all predefined address pools have been fully subnetted",
    "could not find an available, non-overlapping IPv4 address pool among the defaults",
    "Pool overlaps with other one on this address space",
    "invalid pool request",
    "address already in use",
    "failed to create network",
    "failed to add interface",
    "failed to create endpoint",
    "failed to connect to the docker API",
    "failed to connect to the Docker daemon",
    "Cannot connect to the Docker daemon",
    "Is the docker daemon running?",
    "permission denied",
    "operation not permitted",
    "connection refused",
    "connection reset by peer",
    "context deadline exceeded",
    "network not found",
    "No such network",
    "No such container",
    "already exists in network",
    "endpoint with name",
    "network is unreachable",
    "no space left on device",
    "address pool exhausted",
    "i/o timeout",
    "TLS handshake timeout",
    "certificate signed by unknown authority",
    "unauthorized",
    "authentication required",
    "invalid mount config",
)
_SAFE_PHRASE = re.compile("|".join(re.escape(phrase) for phrase in _PHRASES), re.IGNORECASE)
_OPERATIONS = frozenset({"run", "exec", "start", "stop", "rm", "pause", "unpause", "inspect"})
_NETWORK_OPERATIONS = frozenset({"create", "connect", "disconnect", "rm", "inspect", "ls"})


def docker_setup_failure(argv: Sequence[str], exit_code: int, stderr: str) -> str:
    """Format safe details without retaining the raw stderr or command arguments."""
    operation = "operation"
    if argv and argv[0] in _OPERATIONS:
        operation = argv[0]
    elif len(argv) >= 2 and argv[0] == "network" and argv[1] in _NETWORK_OPERATIONS:
        operation = f"network {argv[1]}"
    return (
        f"docker {operation} failed while preparing egress (exit_code={exit_code}); "
        f"stderr: {_safe_stderr(stderr)}"
    )


def _safe_stderr(stderr: str) -> str:
    if not stderr.strip():
        return "[unavailable]"
    source = stderr[:_MAX_SCAN_CHARACTERS]
    parts: list[str] = []
    end = 0
    for match in _SAFE_PHRASE.finditer(source):
        if source[end : match.start()].strip():
            parts.append(_REDACTED)
        parts.append(match.group())
        end = match.end()
    if source[end:].strip():
        parts.append(_REDACTED)
    projected = " ".join(parts).encode("utf-8")
    if len(projected) > _MAX_STDERR_BYTES or len(stderr) > len(source):
        marker = b"...[truncated]"
        return (
            projected[: _MAX_STDERR_BYTES - len(marker)].decode("utf-8", "ignore") + marker.decode()
        )
    return projected.decode() or _REDACTED
