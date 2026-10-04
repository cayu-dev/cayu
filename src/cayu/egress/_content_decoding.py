"""Bounded streaming decoders for upstream HTTP content codings.

The egress upstream requests ``identity`` content encoding, but some origins
(for example the Internet Archive) return their stored encoding regardless.
These decoders cap returned output relative to the response byte limit, so a
decompression bomb is detected before its expansion is materialized. Zstandard
also caps its history window relative to that limit. Deflate may retain bounded
encoded input to retry a raw stream whose prefix resembles a zlib header.

Each codec call produces at most one output step. A caller that accounts memory
passes ``reserve``: the decoder calls it with the total bytes it will hold (the
decoded sink, retained encoded input, and the next step's output) before that
memory is allocated, so a small compressed chunk never inflates past the
reservation.
"""

from __future__ import annotations

import importlib
import zlib
from collections.abc import Callable
from typing import Any, Protocol

Reserve = Callable[[int], None]


class UnsupportedContentEncodingError(ValueError):
    """The response uses a coding this environment cannot decode within a bound."""


class ContentDecodingError(ValueError):
    """The encoded response body is malformed or truncated."""


class DecodedContentTooLargeError(ValueError):
    """The decoded response body exceeded its byte limit."""


class BoundedContentDecoder(Protocol):
    def feed(
        self, data: bytes, sink: bytearray, limit: int, reserve: Reserve | None = None
    ) -> None: ...

    def finish(self, sink: bytearray, limit: int, reserve: Reserve | None = None) -> None: ...


def _optional_module(name: str) -> Any | None:
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


# Only codec APIs that can cap each call's output are accepted. Brotli gained
# ``output_buffer_limit`` together with ``can_accept_more_data`` in 1.2.
_brotli = _optional_module("brotli")
if _brotli is not None and not hasattr(_brotli.Decompressor, "can_accept_more_data"):
    _brotli = None
# The standard-library Zstandard module (Python 3.14+) supports ``max_length``.
_zstd = _optional_module("compression.zstd")


def supported_content_codings() -> frozenset[str]:
    codings = {"gzip", "x-gzip", "deflate"}
    if _brotli is not None:
        codings.add("br")
    if _zstd is not None:
        codings.add("zstd")
    return frozenset(codings)


def content_decoder(
    content_encoding: str | None, *, max_response_bytes: int
) -> BoundedContentDecoder | None:
    """Return a bounded decoder, or ``None`` for an identity response.

    Only a single coding is accepted. Stacked codings and codings without a
    bounded decoder in this environment raise ``UnsupportedContentEncodingError``.
    """

    if content_encoding is None:
        return None
    coding = content_encoding.strip().lower()
    if coding == "identity":
        return None
    if coding in {"gzip", "x-gzip"}:
        return _ZlibDecoder(wbits=zlib.MAX_WBITS | 16, multi_member=True)
    if coding == "deflate":
        return _DeflateDecoder()
    if coding == "br" and _brotli is not None:
        return _BrotliDecoder(_brotli)
    if coding == "zstd" and _zstd is not None:
        return _ZstdDecoder(_zstd, max_response_bytes)
    raise UnsupportedContentEncodingError("Upstream response content encoding is unsupported.")


def _append(sink: bytearray, output: bytes, limit: int) -> None:
    if len(sink) + len(output) > limit:
        raise DecodedContentTooLargeError("Decoded response exceeded the configured byte limit.")
    sink.extend(output)


# Output produced by one codec call; also the most a reservation runs ahead.
_DECODE_OUTPUT_STEP = 1024 * 1024


def _next_output_bound(sink: bytearray, limit: int) -> int:
    # One byte past the remaining allowance proves overflow without allocating
    # more than the limit permits.
    return min(limit - len(sink) + 1, _DECODE_OUTPUT_STEP)


def _reserve(reserve: Reserve | None, total: int) -> None:
    if reserve is not None:
        reserve(total)


class _ZlibDecoder:
    def __init__(self, *, wbits: int, multi_member: bool) -> None:
        self._wbits = wbits
        self._multi_member = multi_member
        self._decompressor = zlib.decompressobj(wbits)
        self._started = False

    def feed(
        self,
        data: bytes | bytearray,
        sink: bytearray,
        limit: int,
        reserve: Reserve | None = None,
        *,
        held: int = 0,
    ) -> None:
        pending = data
        draining = False
        while pending or draining:
            if self._decompressor.eof:
                if not self._multi_member:
                    raise ContentDecodingError("Encoded response has trailing data.")
                self._decompressor = zlib.decompressobj(self._wbits)
            self._started = True
            bound = _next_output_bound(sink, limit)
            _reserve(reserve, len(sink) + held + bound)
            try:
                output = self._decompressor.decompress(pending, bound)
            except zlib.error as exc:
                raise ContentDecodingError("Encoded response is malformed.") from exc
            _append(sink, output, limit)
            if self._decompressor.eof:
                pending = self._decompressor.unused_data
                draining = False
            else:
                # A full step may leave decoded bytes inside zlib even with no
                # input left; output below the bound means it stopped only for
                # lack of input.
                pending = self._decompressor.unconsumed_tail
                draining = len(output) == bound

    @property
    def complete(self) -> bool:
        return self._decompressor.eof

    def finish(self, sink: bytearray, limit: int, reserve: Reserve | None = None) -> None:
        if not self._started or not self._decompressor.eof:
            raise ContentDecodingError("Encoded response is truncated.")


class _DeflateDecoder:
    """HTTP ``deflate`` is specified as zlib-wrapped but often sent raw."""

    def __init__(self) -> None:
        self._replay: bytearray | None = bytearray()
        self._decoder: _ZlibDecoder | None = None
        self._sink_start: int | None = None

    def _held(self) -> int:
        return 0 if self._replay is None else len(self._replay)

    def feed(
        self, data: bytes, sink: bytearray, limit: int, reserve: Reserve | None = None
    ) -> None:
        if self._sink_start is None:
            self._sink_start = len(sink)
        if self._replay is not None:
            if len(self._replay) + len(data) > limit:
                raise DecodedContentTooLargeError("Encoded response exceeded the byte limit.")
            # Retained encoded input is held memory too.
            _reserve(reserve, len(sink) + len(self._replay) + len(data))
            self._replay.extend(data)
        pending: bytes | bytearray = data
        if self._decoder is None:
            assert self._replay is not None
            if len(self._replay) < 2:
                return
            cmf, flg = self._replay[0], self._replay[1]
            zlib_wrapped = cmf & 0x0F == 8 and ((cmf << 8) | flg) % 31 == 0
            self._decoder = _ZlibDecoder(
                wbits=zlib.MAX_WBITS if zlib_wrapped else -zlib.MAX_WBITS,
                multi_member=False,
            )
            pending = self._replay
            if not zlib_wrapped:
                self._replay = None
        try:
            self._decoder.feed(pending, sink, limit, reserve, held=self._held())
        except ContentDecodingError:
            if self._replay is None:
                raise
            self._fallback_to_raw(sink, limit, reserve)
        if self._decoder.complete:
            # A complete wrapped stream passed its checksum. It must not be
            # reinterpreted as raw if a later chunk contains trailing garbage.
            self._replay = None

    def _fallback_to_raw(self, sink: bytearray, limit: int, reserve: Reserve | None) -> None:
        assert self._replay is not None and self._sink_start is not None
        replay, self._replay = self._replay, None
        del sink[self._sink_start :]
        self._decoder = _ZlibDecoder(wbits=-zlib.MAX_WBITS, multi_member=False)
        self._decoder.feed(replay, sink, limit, reserve, held=len(replay))

    def finish(self, sink: bytearray, limit: int, reserve: Reserve | None = None) -> None:
        if self._decoder is None:
            raise ContentDecodingError("Encoded response is truncated.")
        try:
            self._decoder.finish(sink, limit)
        except ContentDecodingError:
            if self._replay is None:
                raise
            self._fallback_to_raw(sink, limit, reserve)
            self._decoder.finish(sink, limit)


# Brotli grows its output buffer in blocks, so ``output_buffer_limit`` is only
# a lower bound on one call's output. Requests below this size yield at most one
# first block (just under 32 KiB), which bounds the transient overshoot that is
# produced and discarded before the limit error.
_BROTLI_OUTPUT_REQUEST = 16 * 1024
# Reserved ahead of each Brotli call: its first output block stays below this.
_BROTLI_CALL_OUTPUT_BOUND = 32 * 1024


class _BrotliDecoder:
    def __init__(self, module: Any) -> None:
        self._module = module
        self._decompressor = module.Decompressor()
        self._started = False

    def feed(
        self, data: bytes, sink: bytearray, limit: int, reserve: Reserve | None = None
    ) -> None:
        if not data:
            return
        self._started = True
        try:
            _reserve(reserve, len(sink) + _BROTLI_CALL_OUTPUT_BOUND)
            output = self._decompressor.process(
                data, output_buffer_limit=self._request(sink, limit)
            )
            _append(sink, output, limit)
            # A capped call may leave decoded output pending even after all
            # input is consumed, so drain with empty input until none remains.
            while not self._decompressor.is_finished() and (
                output or not self._decompressor.can_accept_more_data()
            ):
                _reserve(reserve, len(sink) + _BROTLI_CALL_OUTPUT_BOUND)
                output = self._decompressor.process(
                    b"", output_buffer_limit=self._request(sink, limit)
                )
                _append(sink, output, limit)
        except self._module.error as exc:
            raise ContentDecodingError("Encoded response is malformed.") from exc

    @staticmethod
    def _request(sink: bytearray, limit: int) -> int:
        return min(_next_output_bound(sink, limit), _BROTLI_OUTPUT_REQUEST)

    def finish(self, sink: bytearray, limit: int, reserve: Reserve | None = None) -> None:
        if not self._started or not self._decompressor.is_finished():
            raise ContentDecodingError("Encoded response is truncated.")


class _ZstdDecoder:
    def __init__(self, module: Any, max_response_bytes: int) -> None:
        self._module = module
        # The API expresses this cap as a power of two and permits a minimum
        # of 1 KiB. Round up so a window fitting a non-power-of-two response
        # budget stays admissible; codec bookkeeping is separate from output.
        window_log_max = max(10, (max_response_bytes - 1).bit_length())
        self._max_window_bytes = 1 << window_log_max
        self._options = {module.DecompressionParameter.window_log_max: window_log_max}
        self._decompressor = module.ZstdDecompressor(options=self._options)
        self._started = False
        self._frame_header = bytearray()
        self._window_checked = False

    def _check_window(self, data: bytes) -> None:
        if self._window_checked:
            return
        # The format's complete frame header is at most 18 bytes with magic.
        # Inspect it before giving the completing chunk to the native decoder
        # so a window-budget refusal keeps the oversized-response error code.
        self._frame_header.extend(data[: 18 - len(self._frame_header)])
        header = self._frame_header
        if len(header) < 5:
            return
        if header[:4] != b"\x28\xb5\x2f\xfd":
            # Skippable frames have no history window; invalid magic remains
            # the native decoder's responsibility.
            self._window_checked = True
            return
        descriptor = header[4]
        if descriptor & 0x20:
            # A single-segment frame's content size is also its window size.
            size_width = (1, 2, 4, 8)[descriptor >> 6]
            size_offset = 5 + (0, 1, 2, 4)[descriptor & 3]
            if len(header) < size_offset + size_width:
                return
            window = int.from_bytes(header[size_offset : size_offset + size_width], "little")
            if size_width == 2:
                window += 256
        else:
            if len(header) < 6:
                return
            window_descriptor = header[5]
            base = 1 << (10 + (window_descriptor >> 3))
            window = base + (base >> 3) * (window_descriptor & 7)
        if window > self._max_window_bytes:
            raise DecodedContentTooLargeError("Zstandard window exceeded the byte limit.")
        self._window_checked = True
        self._frame_header.clear()

    def feed(
        self, data: bytes, sink: bytearray, limit: int, reserve: Reserve | None = None
    ) -> None:
        pending = data
        while pending or not self._decompressor.needs_input:
            if self._decompressor.eof:
                # A Zstandard stream may carry several concatenated frames.
                self._decompressor = self._module.ZstdDecompressor(options=self._options)
                self._frame_header.clear()
                self._window_checked = False
            self._started = True
            self._check_window(pending)
            bound = _next_output_bound(sink, limit)
            _reserve(reserve, len(sink) + bound)
            try:
                output = self._decompressor.decompress(pending, bound)
            except (self._module.ZstdError, EOFError) as exc:
                raise ContentDecodingError("Encoded response is malformed.") from exc
            _append(sink, output, limit)
            pending = self._decompressor.unused_data if self._decompressor.eof else b""
            if self._decompressor.eof and not pending:
                return

    def finish(self, sink: bytearray, limit: int, reserve: Reserve | None = None) -> None:
        if not self._started or not self._decompressor.eof:
            raise ContentDecodingError("Encoded response is truncated.")
