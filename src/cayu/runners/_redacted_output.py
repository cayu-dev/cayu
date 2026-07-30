"""Bounded runner-output capture that redacts before discarding source bytes."""

from __future__ import annotations

import codecs

from cayu.vaults import REDACTED_SECRET, SecretRedactionCapacityError, SecretRedactor
from cayu.vaults.redaction import _bounded_redacted_head, _bounded_stream_retention_bytes

_REDACTION_FEED_CHUNK_BYTES = 4096


class RedactedOutputCapture:
    """Capture a bounded redacted head while counting original input bytes."""

    def __init__(self, *, redactor: SecretRedactor, limit: int | None) -> None:
        if not isinstance(redactor, SecretRedactor):
            raise TypeError("redactor must be a SecretRedactor.")
        if limit is not None and (type(limit) is not int or limit <= 0):
            raise ValueError("limit must be a positive integer or None.")
        self._redactor = redactor
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._stream = redactor.stream_bytes(
            max_retained_bytes=(
                None
                if limit is None
                else _bounded_stream_retention_bytes(redactor, max_bytes=limit)
            )
        )
        self._limit = limit
        self._content = bytearray()
        self._redacted_bytes = 0
        self._prefix_finalized = False
        self._sealed = False
        self.total_bytes = 0
        self.truncated = False

    def append(self, chunk: str | bytes) -> None:
        if self._sealed:
            return
        if type(chunk) is str:
            raw = chunk.encode("utf-8")
        elif type(chunk) is bytes:
            raw = chunk
        else:
            raise TypeError("Runner output chunks must be strings or bytes.")
        if not raw:
            return
        self.total_bytes += len(raw)
        if self._prefix_finalized:
            self.truncated = True
            return
        for offset in range(0, len(raw), _REDACTION_FEED_CHUNK_BYTES):
            normalized = self._decoder.decode(
                raw[offset : offset + _REDACTION_FEED_CHUNK_BYTES],
                final=False,
            ).encode("utf-8")
            try:
                self._append_redacted(self._stream.feed(normalized))
            except SecretRedactionCapacityError as exc:
                self._append_redacted(exc.released)
                # Preserve only output already proven independent of future
                # input. The unresolved tail is omitted and reported through
                # the existing truncation contract.
                self.truncated = True
                self._prefix_finalized = True
                break
            if self._prefix_finalized:
                # The stream releases only bytes that no future fragment can
                # change. Once enough stable output exists to determine the
                # marker-safe public prefix, later source bytes can be drained
                # and counted without paying to redact discarded output.
                self._stream.abort()
                break
        if self._limit is not None and self.total_bytes > self._limit:
            self.truncated = True

    def seed_if_empty(self, chunk: object) -> None:
        if self.total_bytes:
            return
        if type(chunk) is str or type(chunk) is bytes:
            self.append(chunk)

    def replace(self, chunk: bytes) -> None:
        if self._sealed:
            raise RuntimeError("Cannot replace a finalized runner output capture.")
        if type(chunk) is not bytes:
            raise TypeError("Replacement runner output must be bytes.")
        self._stream.abort()
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._stream = self._redactor.stream_bytes(
            max_retained_bytes=(
                None
                if self._limit is None
                else _bounded_stream_retention_bytes(
                    self._redactor,
                    max_bytes=self._limit,
                )
            )
        )
        self._content.clear()
        self._redacted_bytes = 0
        self._prefix_finalized = False
        self.total_bytes = 0
        self.truncated = False
        self.append(chunk)

    def finish_complete(self) -> None:
        if self._sealed:
            return
        if self._prefix_finalized:
            self._sealed = True
            return
        normalized = self._decoder.decode(b"", final=True).encode("utf-8")
        try:
            self._append_redacted(self._stream.feed(normalized))
            self._append_redacted(self._stream.finish_complete())
        except SecretRedactionCapacityError as exc:
            self._append_redacted(exc.released)
            self.truncated = True
        self._sealed = True

    def abort(self) -> None:
        if self._sealed:
            return
        self._stream.abort()
        self._sealed = True
        # Abandonment means EOF was not observed. Even without a currently
        # undecided secret prefix, late or buffered source bytes may be absent.
        self.truncated = True

    def text(self) -> str:
        if not self._sealed:
            raise RuntimeError("Runner output capture must be finalized before publication.")
        content = bytes(self._content)
        if self._limit is not None:
            if len(content) > self._limit:
                self.truncated = True
            content = _bounded_redacted_head(content, max_bytes=self._limit)
        return content.decode("utf-8", "ignore")

    def _append_redacted(self, chunk: bytes) -> None:
        if not chunk:
            return
        self._redacted_bytes += len(chunk)
        if self._limit is None:
            self._content.extend(chunk)
            return
        if self._redacted_bytes > self._limit:
            self.truncated = True
        # One complete marker beyond the public bound is sufficient because
        # this buffer contains only already-redacted bytes.
        capture_limit = self._limit + len(REDACTED_SECRET.encode("utf-8"))
        remaining = capture_limit - len(self._content)
        if remaining > 0:
            self._content.extend(chunk[:remaining])
        if len(self._content) >= capture_limit:
            self._prefix_finalized = True


def redact_completed_exec_result(
    result: object,
    *,
    redactor: SecretRedactor,
    output_limit_bytes: int | None,
    omit_pretruncated: bool,
):
    """Project a complete runner result through a safe compatibility boundary."""

    from cayu.runners.base import ExecResult

    if type(result) is not ExecResult:
        raise TypeError("Runner.exec must return an ExecResult.")
    if not isinstance(redactor, SecretRedactor):
        raise TypeError("redactor must be a SecretRedactor.")
    if output_limit_bytes is not None and (
        type(output_limit_bytes) is not int or output_limit_bytes <= 0
    ):
        raise ValueError("output_limit_bytes must be a positive integer or None.")

    updates: dict[str, object] = {}
    for channel in ("stdout", "stderr"):
        value = getattr(result, channel)
        truncated = getattr(result, f"{channel}_truncated")
        byte_count = getattr(result, f"{channel}_bytes")
        ambiguous = bool(truncated) or (
            output_limit_bytes is not None
            and type(byte_count) is int
            and byte_count > output_limit_bytes
        )
        if omit_pretruncated and ambiguous:
            updates[channel] = ""
            updates[f"{channel}_truncated"] = True
            continue
        redacted = redactor.redact_text(value)
        encoded = redacted.encode("utf-8", "replace")
        redaction_truncated = output_limit_bytes is not None and len(encoded) > output_limit_bytes
        if output_limit_bytes is not None:
            encoded = _bounded_redacted_head(encoded, max_bytes=output_limit_bytes)
        updates[channel] = encoded.decode("utf-8", "ignore")
        updates[f"{channel}_truncated"] = bool(truncated) or redaction_truncated
    return result.model_copy(update=updates)
