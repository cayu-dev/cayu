from __future__ import annotations

import codecs
import re
from bisect import bisect_right
from collections import deque
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import SecretStr

from cayu._validation import collision_safe_json_object, copy_json_value, require_nonblank
from cayu.vaults.base import ResolvedSecret

REDACTED_SECRET = "[REDACTED_SECRET]"
_MIN_BOUNDED_STREAM_RETENTION_BYTES = 64 * 1024
_STREAM_INPUT_CHUNK_BYTES = 1024


class SecretRedactionCapacityError(RuntimeError):
    """Raised before unresolved streaming-redaction state exceeds its bound."""

    def __init__(self, *, released: bytes = b"") -> None:
        if type(released) is not bytes:
            raise TypeError("released must be bytes.")
        super().__init__("Streaming secret redaction exceeded its unresolved-source capacity.")
        self.released = released


def contains_redacted_secret(value: Any) -> bool:
    """Return whether JSON-compatible data contains the durable redaction marker."""

    copied = copy_json_value(value, "value")
    return _contains_redacted_secret(copied)


def _contains_redacted_secret(value: Any) -> bool:
    if type(value) is str:
        return REDACTED_SECRET in value
    if value is None or type(value) in {bool, int, float}:
        return False
    if type(value) is list:
        return any(_contains_redacted_secret(item) for item in value)
    if type(value) is dict:
        return any(
            REDACTED_SECRET in key or _contains_redacted_secret(item) for key, item in value.items()
        )
    raise AssertionError("copy_json_value returned non-JSON-compatible data.")


class SecretRedactor:
    """Redacts known resolved secret values from strings and JSON-like data."""

    def __init__(
        self,
        secrets: str
        | SecretStr
        | ResolvedSecret
        | Sequence[str | SecretStr | ResolvedSecret]
        | None = None,
    ) -> None:
        values: set[str] = set()
        for secret in _secret_items(secrets):
            values.add(_secret_value(secret))
        self._values = tuple(sorted(values, key=len, reverse=True))
        self._pattern = _redaction_pattern(self._values)

    @property
    def has_values(self) -> bool:
        return bool(self._values)

    @property
    def max_secret_utf8_bytes(self) -> int:
        """Return the bounded overlap needed to redact across a byte boundary."""

        return max(
            (len(value.encode("utf-8", "surrogatepass")) for value in self._values),
            default=0,
        )

    @property
    def pagination_overlap_utf8_bytes(self) -> int:
        """Return source overlap needed for source-position-aware page redaction."""

        if not self.has_values:
            return 0
        return self.max_secret_utf8_bytes + len(REDACTED_SECRET.encode("utf-8"))

    def with_secret(self, secret: str | SecretStr | ResolvedSecret) -> SecretRedactor:
        value = _secret_value(secret)
        require_nonblank(value, "secret")
        if value in self._values:
            return self
        values = set(self._values)
        values.add(value)
        return SecretRedactor._from_values(tuple(sorted(values, key=len, reverse=True)))

    def merged_with(self, other: SecretRedactor) -> SecretRedactor:
        """Combine two registries without exposing either registry's values."""

        if not isinstance(other, SecretRedactor):
            raise TypeError("other must be a SecretRedactor.")
        return SecretRedactor._from_values(
            tuple(sorted(set(self._values) | set(other._values), key=len, reverse=True))
        )

    def has_same_registry(self, other: SecretRedactor) -> bool:
        """Compare registries without exposing their secret values."""

        if not isinstance(other, SecretRedactor):
            raise TypeError("other must be a SecretRedactor.")
        return self._values == other._values

    def redact_text(self, value: str) -> str:
        if type(value) is not str:
            raise TypeError("SecretRedactor.redact_text expects a string.")
        if self._pattern is None:
            return value
        # Preserve the pre-existing total-string contract for diagnostic text
        # containing lone surrogates. Portable publication paths replace or
        # reject them later, but redaction itself must not mask the original
        # failure with UnicodeEncodeError.
        redacted = self._pattern.sub(REDACTED_SECRET, value).encode(
            "utf-8",
            "surrogatepass",
        )
        return _stabilize_redacted_bytes(
            redacted,
            secret_patterns=self._secret_byte_patterns(),
        ).decode("utf-8", "surrogatepass")

    def redact_text_bounded(self, value: str, *, max_bytes: int) -> str:
        """Redact a diagnostic string before applying its UTF-8 byte bound."""

        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer.")
        redacted = self.redact_text(value)
        encoded = redacted.encode("utf-8", "replace")
        if len(encoded) <= max_bytes:
            return redacted
        truncation_marker = b"...[truncated]"
        if len(truncation_marker) < max_bytes:
            retained = _bounded_redacted_head(
                encoded,
                max_bytes=max_bytes - len(truncation_marker),
            )
            candidate = retained + truncation_marker
            # The retained prefix and otherwise-safe marker may reconstruct a
            # registered secret at their join. Omit the truncation marker in
            # that case instead of creating a value that grows on a mandatory
            # later redaction pass.
            candidate_text = candidate.decode("utf-8", "ignore")
            if self.redact_text(candidate_text).encode("utf-8") == candidate:
                return candidate_text
        return _bounded_redacted_head(encoded, max_bytes=max_bytes).decode(
            "utf-8",
            "ignore",
        )

    def redact_text_bounded_with_marker(
        self,
        value: str,
        *,
        max_bytes: int,
        truncation_marker: str,
    ) -> tuple[str, bool]:
        """Redact complete text, then append an atomic marker when it is shortened."""

        if type(value) is not str:
            raise TypeError("SecretRedactor.redact_text_bounded_with_marker expects a string.")
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer.")
        if type(truncation_marker) is not str or not truncation_marker:
            raise ValueError("truncation_marker must be a non-empty string.")
        marker = truncation_marker.encode("utf-8")
        redacted = self.redact_text(value)
        encoded = redacted.encode("utf-8", "replace")
        if len(encoded) <= max_bytes:
            return redacted, False
        if len(marker) >= max_bytes:
            return (
                _bounded_redacted_head(encoded, max_bytes=max_bytes).decode(
                    "utf-8",
                    "ignore",
                ),
                True,
            )
        retained = _bounded_redacted_head(
            encoded,
            max_bytes=max_bytes - len(marker),
        )
        candidate = retained.rstrip() + marker
        candidate_text = candidate.decode("utf-8", "ignore")
        if self.redact_text(candidate_text).encode("utf-8") != candidate:
            candidate = _bounded_redacted_head(encoded, max_bytes=max_bytes)
            candidate_text = candidate.decode("utf-8", "ignore")
        return candidate_text, True

    def redact_text_head(self, value: str, *, max_bytes: int) -> tuple[str, bool]:
        """Redact a complete string, then return its marker-safe bounded head."""

        if type(value) is not str:
            raise TypeError("SecretRedactor.redact_text_head expects a string.")
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer.")
        redacted = self.redact_text(value)
        encoded = redacted.encode("utf-8", "replace")
        truncated = len(encoded) > max_bytes
        return (
            _bounded_redacted_head(encoded, max_bytes=max_bytes).decode(
                "utf-8",
                "ignore",
            ),
            truncated,
        )

    def redact_utf8_head(
        self,
        value: bytes,
        *,
        max_bytes: int,
        source_complete: bool,
    ) -> tuple[str, bool]:
        """Project a possibly incomplete UTF-8 source prefix without exposing its suffix.

        ``source_complete=False`` means the caller did not observe authoritative
        EOF. The streaming redactor therefore discards every suffix whose safety
        could depend on unavailable bytes instead of treating the captured prefix
        as a complete value.
        """

        if type(value) is not bytes:
            raise TypeError("SecretRedactor.redact_utf8_head expects bytes.")
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer.")
        if type(source_complete) is not bool:
            raise TypeError("source_complete must be a bool.")

        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        stream = self.stream_bytes(
            max_retained_bytes=_bounded_stream_retention_bytes(
                self,
                max_bytes=max_bytes,
            )
        )
        content = bytearray()
        truncated = False
        try:
            for offset in range(0, len(value), _STREAM_INPUT_CHUNK_BYTES):
                normalized = decoder.decode(
                    value[offset : offset + _STREAM_INPUT_CHUNK_BYTES],
                    final=False,
                ).encode("utf-8")
                content.extend(stream.feed(normalized))
            if source_complete:
                normalized = decoder.decode(b"", final=True).encode("utf-8")
                content.extend(stream.feed(normalized))
                content.extend(stream.finish_complete())
            else:
                truncated = stream.abort()
        except SecretRedactionCapacityError as exc:
            content.extend(exc.released)
            truncated = True

        if len(content) > max_bytes:
            truncated = True
        bounded = _bounded_redacted_head(bytes(content), max_bytes=max_bytes)
        return bounded.decode("utf-8", "ignore"), truncated

    def redact_bytes_head(
        self,
        value: bytes,
        *,
        max_bytes: int,
        source_complete: bool,
    ) -> tuple[bytes, bool]:
        """Project a possibly incomplete binary prefix without exposing its suffix."""

        if type(value) is not bytes:
            raise TypeError("SecretRedactor.redact_bytes_head expects bytes.")
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer.")
        if type(source_complete) is not bool:
            raise TypeError("source_complete must be a bool.")

        stream = self.stream_bytes(
            max_retained_bytes=_bounded_stream_retention_bytes(
                self,
                max_bytes=max_bytes,
            )
        )
        content = bytearray()
        truncated = False
        try:
            for offset in range(0, len(value), _STREAM_INPUT_CHUNK_BYTES):
                content.extend(stream.feed(value[offset : offset + _STREAM_INPUT_CHUNK_BYTES]))
            if source_complete:
                content.extend(stream.finish_complete())
            else:
                truncated = stream.abort()
        except SecretRedactionCapacityError as exc:
            content.extend(exc.released)
            truncated = True

        if len(content) > max_bytes:
            truncated = True
        return _bounded_redacted_head(bytes(content), max_bytes=max_bytes), truncated

    def redact_utf8_page(
        self,
        value: bytes,
        *,
        window_offset: int,
        page_offset: int,
        page_end: int,
        max_bytes: int,
        source_complete: bool,
    ) -> tuple[str, bool]:
        """Project one raw-source page without exposing cross-page secret fragments."""

        if type(value) is not bytes:
            raise TypeError("SecretRedactor.redact_utf8_page expects bytes.")
        for field_name, field_value in (
            ("window_offset", window_offset),
            ("page_offset", page_offset),
            ("page_end", page_end),
        ):
            if type(field_value) is not int or field_value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer.")
        if page_end < page_offset:
            raise ValueError("page_end must not precede page_offset.")
        window_end = window_offset + len(value)
        if page_offset < window_offset or page_end > window_end:
            raise ValueError("The requested page must be contained by the source window.")
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer.")
        if type(source_complete) is not bool:
            raise TypeError("source_complete must be a bool.")

        if not self.has_values:
            raw_page = value[page_offset - window_offset : page_end - window_offset]
            bounded = _bounded_redacted_head(raw_page, max_bytes=max_bytes)
            return bounded.decode("utf-8", "ignore"), len(bounded) < len(raw_page)

        relative_start = page_offset - window_offset
        relative_end = page_end - window_offset
        pieces = _source_redaction_pieces(
            value,
            ordered_patterns=self._ordered_byte_patterns(),
        )
        pieces = _stabilize_redacted_piece_sequence(
            pieces,
            secret_patterns=self._secret_byte_patterns(),
        )
        projected = bytearray()
        omitted_leading_atomic_piece = False
        for piece in pieces:
            if piece.source_end <= relative_start or piece.source_start >= relative_end:
                continue
            if not piece.linear:
                if piece.source_start < relative_start:
                    omitted_leading_atomic_piece = True
                    continue
                projected.extend(piece.value)
                continue
            local_start = max(relative_start, piece.source_start) - piece.source_start
            local_end = min(relative_end, piece.source_end) - piece.source_start
            projected.extend(piece.value[local_start:local_end])

        stabilized = _stabilize_redacted_bytes(
            bytes(projected),
            secret_patterns=self._secret_byte_patterns(),
        )
        text, boundary_truncated = self.redact_utf8_head(
            stabilized,
            max_bytes=max_bytes,
            source_complete=source_complete,
        )
        if not source_complete and text:
            encoded_text = text.encode("utf-8")
            safe_end = _future_sensitive_suffix_start(
                encoded_text,
                secret_patterns=self._secret_byte_patterns(),
            )
            if safe_end < len(encoded_text):
                text = encoded_text[:safe_end].decode("utf-8")
                boundary_truncated = True
        return (
            text,
            omitted_leading_atomic_piece or boundary_truncated,
        )

    def redact_bytes_page(
        self,
        value: bytes,
        *,
        window_offset: int,
        page_offset: int,
        page_end: int,
        max_bytes: int,
        source_complete: bool,
    ) -> tuple[bytes, bool]:
        """Project one raw-source byte page without exposing cross-page fragments."""

        if type(value) is not bytes:
            raise TypeError("SecretRedactor.redact_bytes_page expects bytes.")
        for field_name, field_value in (
            ("window_offset", window_offset),
            ("page_offset", page_offset),
            ("page_end", page_end),
        ):
            if type(field_value) is not int or field_value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer.")
        if page_end < page_offset:
            raise ValueError("page_end must not precede page_offset.")
        window_end = window_offset + len(value)
        if page_offset < window_offset or page_end > window_end:
            raise ValueError("The requested page must be contained by the source window.")
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer.")
        if type(source_complete) is not bool:
            raise TypeError("source_complete must be a bool.")

        if not self.has_values:
            raw_page = value[page_offset - window_offset : page_end - window_offset]
            bounded = _bounded_redacted_head(raw_page, max_bytes=max_bytes)
            return bounded, len(bounded) < len(raw_page)

        relative_start = page_offset - window_offset
        relative_end = page_end - window_offset
        pieces = _source_redaction_pieces(
            value,
            ordered_patterns=self._ordered_byte_patterns(),
        )
        pieces = _stabilize_redacted_piece_sequence(
            pieces,
            secret_patterns=self._secret_byte_patterns(),
        )
        projected = bytearray()
        omitted_leading_atomic_piece = False
        for piece in pieces:
            if piece.source_end <= relative_start or piece.source_start >= relative_end:
                continue
            if not piece.linear:
                if piece.source_start < relative_start:
                    omitted_leading_atomic_piece = True
                    continue
                projected.extend(piece.value)
                continue
            local_start = max(relative_start, piece.source_start) - piece.source_start
            local_end = min(relative_end, piece.source_end) - piece.source_start
            projected.extend(piece.value[local_start:local_end])

        stabilized = _stabilize_redacted_bytes(
            bytes(projected),
            secret_patterns=self._secret_byte_patterns(),
        )
        bounded, boundary_truncated = self.redact_bytes_head(
            stabilized,
            max_bytes=max_bytes,
            source_complete=source_complete,
        )
        if not source_complete and bounded:
            safe_end = _future_sensitive_suffix_start(
                bounded,
                secret_patterns=self._secret_byte_patterns(),
            )
            if safe_end < len(bounded):
                bounded = bounded[:safe_end]
                boundary_truncated = True
        return bounded, omitted_leading_atomic_piece or boundary_truncated

    def stream_bytes(
        self,
        *,
        max_retained_bytes: int | None = None,
    ) -> SecretRedactionStream:
        """Create an ordered byte-stream redactor for this registry.

        ``max_retained_bytes`` bounds source bytes whose final redaction is
        still dependent on future input. Exceeding the bound fails closed
        before the unresolved source is published.
        """

        return SecretRedactionStream(
            self,
            max_retained_bytes=max_retained_bytes,
        )

    def redact_json(self, value: Any) -> Any:
        copied = copy_json_value(value, "value")
        return self._redact_copied_json(copied)

    def redact_json_values(
        self,
        value: Any,
        *,
        preserve_string_fields: Collection[str] = (),
    ) -> Any:
        """Redact JSON string values without rewriting protocol object keys."""

        preserved_fields = frozenset(preserve_string_fields)
        if any(type(field_name) is not str for field_name in preserved_fields):
            raise TypeError("preserve_string_fields must contain strings.")
        copied = copy_json_value(value, "value")
        return self._redact_copied_json_values(
            copied,
            field_name=None,
            depth=0,
            preserve_string_fields=preserved_fields,
        )

    def require_no_secret_keys(
        self,
        value: Any,
        *,
        field_name: str = "value",
        preserve_keys: Collection[str] = (),
        untrusted_container_keys: Collection[str] = (),
        match_short_substrings: bool = False,
    ) -> None:
        """Fail closed when a known secret occurs in an untyped JSON object key."""

        preserved_keys = frozenset(preserve_keys)
        untrusted_keys = frozenset(untrusted_container_keys)
        if any(type(key) is not str for key in preserved_keys):
            raise TypeError("preserve_keys must contain strings.")
        if any(type(key) is not str for key in untrusted_keys):
            raise TypeError("untrusted_container_keys must contain strings.")
        copied = copy_json_value(value, field_name)
        self._require_no_secret_keys(
            copied,
            field_name=field_name,
            preserve_keys=preserved_keys,
            untrusted_container_keys=untrusted_keys,
            inside_untrusted=False,
            match_short_substrings=match_short_substrings,
        )

    def _redact_copied_json(self, value: Any) -> Any:
        if type(value) is str:
            return self.redact_text(value)
        if value is None or type(value) in {bool, int, float}:
            return value
        if type(value) is list:
            return [self._redact_copied_json(item) for item in value]
        if type(value) is dict:
            items: list[tuple[str, Any]] = []
            for key, item in value.items():
                if type(key) is not str:
                    raise AssertionError("copy_json_value returned a non-string object key.")
                redacted_key = self.redact_text(key)
                redacted_item = self._redact_copied_json(item)
                items.append((redacted_key, redacted_item))
            return collision_safe_json_object(items, preserve_input_order=True)
        raise AssertionError("copy_json_value returned non-JSON-compatible data.")

    def _redact_copied_json_values(
        self,
        value: Any,
        *,
        field_name: str | None,
        depth: int,
        preserve_string_fields: frozenset[str],
    ) -> Any:
        if type(value) is str:
            if depth == 1 and field_name in preserve_string_fields:
                return value
            return self.redact_text(value)
        if value is None or type(value) in {bool, int, float}:
            return value
        if type(value) is list:
            return [
                self._redact_copied_json_values(
                    item,
                    field_name=field_name,
                    depth=depth + 1,
                    preserve_string_fields=preserve_string_fields,
                )
                for item in value
            ]
        if type(value) is dict:
            return {
                key: self._redact_copied_json_values(
                    item,
                    field_name=key,
                    depth=depth + 1,
                    preserve_string_fields=preserve_string_fields,
                )
                for key, item in value.items()
            }
        raise AssertionError("copy_json_value returned non-JSON-compatible data.")

    def _require_no_secret_keys(
        self,
        value: Any,
        *,
        field_name: str,
        preserve_keys: frozenset[str],
        untrusted_container_keys: frozenset[str],
        inside_untrusted: bool,
        match_short_substrings: bool,
    ) -> None:
        if value is None or type(value) in {str, bool, int, float}:
            return
        if type(value) is list:
            for item in value:
                self._require_no_secret_keys(
                    item,
                    field_name=field_name,
                    preserve_keys=preserve_keys,
                    untrusted_container_keys=untrusted_container_keys,
                    inside_untrusted=inside_untrusted,
                    match_short_substrings=match_short_substrings,
                )
            return
        if type(value) is dict:
            for key, item in value.items():
                # Public schema keys are trusted only at structural boundaries.
                # Below caller-controlled containers, even a spelling that
                # happens to match a schema key is untrusted data.
                key_is_preserved = key in preserve_keys and not inside_untrusted
                if not key_is_preserved and any(
                    key == secret
                    or (secret in key and (match_short_substrings or len(secret) >= 8))
                    for secret in self._values
                ):
                    raise ValueError(
                        f"{field_name} contains a workload secret in an object key; "
                        "refusing to publish it."
                    )
                self._require_no_secret_keys(
                    item,
                    field_name=field_name,
                    preserve_keys=preserve_keys,
                    untrusted_container_keys=untrusted_container_keys,
                    inside_untrusted=(inside_untrusted or key in untrusted_container_keys),
                    match_short_substrings=match_short_substrings,
                )
            return
        raise AssertionError("copy_json_value returned non-JSON-compatible data.")

    @classmethod
    def _from_values(cls, values: tuple[str, ...]) -> SecretRedactor:
        redactor = cls()
        redactor._values = values
        redactor._pattern = _redaction_pattern(values)
        return redactor

    def _ordered_byte_patterns(self) -> tuple[tuple[bytes, bool], ...]:
        marker = REDACTED_SECRET.encode("utf-8")
        marker_prefixed = [
            value.encode("utf-8", "surrogatepass")
            for value in self._values
            if value != REDACTED_SECRET and value.startswith(REDACTED_SECRET)
        ]
        excluded = {
            value
            for value in self._values
            if value == REDACTED_SECRET or value.startswith(REDACTED_SECRET)
        }
        remaining = [
            value.encode("utf-8", "surrogatepass")
            for value in self._values
            if value not in excluded
        ]
        return (
            *((value, True) for value in marker_prefixed),
            (marker, False),
            *((value, True) for value in remaining),
        )

    def _secret_byte_patterns(self) -> tuple[bytes, ...]:
        marker = REDACTED_SECRET.encode("utf-8")
        return tuple(
            encoded
            for value in self._values
            if (encoded := value.encode("utf-8", "surrogatepass")) != marker
        )


@dataclass(slots=True)
class _PendingRedactedSegment:
    value: bytearray
    repetitions: int
    is_marker: bool


class _CompactRedactedSource:
    """Retain exact first-pass bytes without expanding adjacent marker runs."""

    def __init__(self) -> None:
        self._marker = REDACTED_SECRET.encode("utf-8")
        self._segments: deque[_PendingRedactedSegment] = deque()
        self._byte_length = 0

    def __bool__(self) -> bool:
        return bool(self._byte_length)

    def __len__(self) -> int:
        return self._byte_length

    @property
    def storage_bytes(self) -> int:
        """Return retained payload bytes, excluding fixed segment metadata."""

        return sum(len(segment.value) for segment in self._segments)

    @property
    def contains_marker(self) -> bool:
        return any(segment.is_marker for segment in self._segments)

    def stabilized_uniform_marker_run(
        self,
        *,
        collapse_threshold: int | None,
    ) -> bytes | None:
        """Return an exact compact result for a uniform marker run, if bounded."""

        if len(self._segments) != 1 or not self._segments[0].is_marker:
            return None
        repetitions = self._segments[0].repetitions
        if repetitions == 1 or (
            collapse_threshold is not None and repetitions >= collapse_threshold
        ):
            return self._marker
        return None

    def __bytes__(self) -> bytes:
        return b"".join(bytes(segment.value) * segment.repetitions for segment in self._segments)

    def append(
        self,
        chunk: bytes,
        *,
        pattern_alphabet: frozenset[int],
    ) -> tuple[bool, bool]:
        """Append bytes and report shape changes plus literal hard barriers."""

        shape_changed = False
        has_hard_barrier = False
        cursor = 0
        while cursor < len(chunk):
            marker_start = chunk.find(self._marker, cursor)
            if marker_start < 0:
                literal = chunk[cursor:]
                shape_changed = self._append_literal(literal) or shape_changed
                has_hard_barrier = (
                    any(byte not in pattern_alphabet for byte in literal) or has_hard_barrier
                )
                break
            literal = chunk[cursor:marker_start]
            shape_changed = self._append_literal(literal) or shape_changed
            has_hard_barrier = (
                any(byte not in pattern_alphabet for byte in literal) or has_hard_barrier
            )
            shape_changed = self._append_marker() or shape_changed
            cursor = marker_start + len(self._marker)
        self._byte_length += len(chunk)
        return shape_changed, has_hard_barrier

    def discard_prefix(self, byte_count: int) -> None:
        if not 0 <= byte_count <= self._byte_length:
            raise ValueError("byte_count must describe a retained source prefix.")
        remaining = byte_count
        while remaining:
            segment = self._segments[0]
            segment_length = len(segment.value) * segment.repetitions
            if remaining >= segment_length:
                remaining -= segment_length
                self._segments.popleft()
                continue
            if segment.repetitions > 1:
                unit_length = len(segment.value)
                repetitions, unit_offset = divmod(remaining, unit_length)
                segment.repetitions -= repetitions
                if unit_offset:
                    if segment.is_marker:
                        raise AssertionError("Secret redaction split a compact marker run.")
                    expanded = bytearray(segment.value[unit_offset:])
                    expanded.extend(segment.value * (segment.repetitions - 1))
                    segment.value = expanded
                    segment.repetitions = 1
            else:
                del segment.value[:remaining]
            remaining = 0
        self._byte_length -= byte_count

    def clear(self) -> None:
        self._segments.clear()
        self._byte_length = 0

    def _append_literal(self, value: bytes) -> bool:
        if not value:
            return False
        if value.count(value[:1]) == len(value):
            unit = value[:1]
            if (
                self._segments
                and not self._segments[-1].is_marker
                and bytes(self._segments[-1].value) == unit
            ):
                self._segments[-1].repetitions += len(value)
                return False
            self._segments.append(
                _PendingRedactedSegment(
                    value=bytearray(unit),
                    repetitions=len(value),
                    is_marker=False,
                )
            )
            return True
        if (
            self._segments
            and not self._segments[-1].is_marker
            and self._segments[-1].repetitions == 1
        ):
            previous_byte = self._segments[-1].value[-1]
            self._segments[-1].value.extend(value)
            return any(byte != previous_byte for byte in value)
        self._segments.append(
            _PendingRedactedSegment(
                value=bytearray(value),
                repetitions=1,
                is_marker=False,
            )
        )
        return True

    def _append_marker(self) -> bool:
        if self._segments and self._segments[-1].is_marker:
            self._segments[-1].repetitions += 1
            return False
        self._segments.append(
            _PendingRedactedSegment(
                value=bytearray(self._marker),
                repetitions=1,
                is_marker=True,
            )
        )
        return True


class SecretRedactionStream:
    """Redact registered secrets before releasing ordered byte fragments.

    A suffix that may begin either a secret or the public marker remains
    private until a later fragment makes it decidable. Callers must distinguish
    a proven normal EOF from uncertain abandonment: ``finish_complete`` may
    release an unmatched suffix, while ``abort`` discards it.
    """

    def __init__(
        self,
        redactor: SecretRedactor,
        *,
        max_retained_bytes: int | None = None,
    ) -> None:
        if not isinstance(redactor, SecretRedactor):
            raise TypeError("SecretRedactionStream requires a SecretRedactor.")
        if max_retained_bytes is not None and (
            type(max_retained_bytes) is not int or max_retained_bytes <= 0
        ):
            raise ValueError("max_retained_bytes must be a positive integer or None.")
        self._patterns = redactor._ordered_byte_patterns()
        self._secret_patterns = redactor._secret_byte_patterns()
        self._marker_overlap_risk = any(
            _pattern_can_cross_marker(pattern) for pattern in self._secret_patterns
        )
        self._pattern_alphabet = frozenset().union(
            *(frozenset(pattern) for pattern in self._secret_patterns)
        )
        self._marker_run_collapse_threshold = _marker_run_collapse_threshold(self._secret_patterns)
        self._pending = bytearray()
        self._redacted_pending = _CompactRedactedSource()
        self._redacted_recheck_at = 0
        self._redacted_marker_fenced = False
        self._max_retained_bytes = max_retained_bytes
        self._finished = False
        self._aborted = False

    def feed(self, chunk: bytes) -> bytes:
        if type(chunk) is not bytes:
            raise TypeError("SecretRedactionStream.feed expects bytes.")
        if self._aborted:
            # Some remote SDKs may deliver a late callback after cancellation
            # cleanup. The capture is sealed, so ignore rather than retaining
            # newly delivered raw bytes.
            return b""
        if self._finished:
            raise RuntimeError("SecretRedactionStream is already finished.")
        output = bytearray()
        for offset in range(0, len(chunk), _STREAM_INPUT_CHUNK_BYTES):
            self._pending.extend(chunk[offset : offset + _STREAM_INPUT_CHUNK_BYTES])
            try:
                output.extend(
                    self._queue_redacted(
                        self._drain_input(final=False),
                        final=False,
                    )
                )
            except SecretRedactionCapacityError as exc:
                exc.released = bytes(output) + exc.released
                raise
        return bytes(output)

    def finish_complete(self) -> bytes:
        """Finish after proven EOF and release the final unmatched suffix."""

        if self._aborted or self._finished:
            return b""
        self._finished = True
        return self._queue_redacted(self._drain_input(final=True), final=True)

    def abort(self) -> bool:
        """Seal an incomplete stream and report whether raw bytes were discarded."""

        if self._finished:
            return False
        discarded = bool(self._pending or self._redacted_pending)
        self._pending.clear()
        self._redacted_pending.clear()
        self._aborted = True
        self._finished = True
        return discarded

    def _drain_input(self, *, final: bool) -> bytes:
        if not self._patterns:
            released = bytes(self._pending)
            self._pending.clear()
            return released

        output = bytearray()
        cursor = 0
        pending = self._pending
        while cursor < len(pending):
            selected: tuple[bytes, bool] | None = None
            must_wait = False
            for pattern, redact in self._patterns:
                available = len(pending) - cursor
                compared = min(available, len(pattern))
                if pending[cursor : cursor + compared] != pattern[:compared]:
                    continue
                if available < len(pattern):
                    if not final:
                        must_wait = True
                        break
                    continue
                selected = (pattern, redact)
                break
            if must_wait:
                break
            if selected is None:
                output.append(pending[cursor])
                cursor += 1
                continue
            pattern, redact = selected
            output.extend(REDACTED_SECRET.encode("utf-8") if redact else pattern)
            cursor += len(pattern)
        if cursor:
            del pending[:cursor]
        return bytes(output)

    def _queue_redacted(self, chunk: bytes, *, final: bool) -> bytes:
        shape_changed, has_hard_barrier = self._redacted_pending.append(
            chunk,
            pattern_alphabet=self._pattern_alphabet,
        )
        if final:
            compact_result = self._redacted_pending.stabilized_uniform_marker_run(
                collapse_threshold=self._marker_run_collapse_threshold,
            )
            if compact_result is not None:
                self._redacted_pending.clear()
                self._redacted_recheck_at = 0
                self._redacted_marker_fenced = False
                return compact_result
        if not final and self._redacted_marker_fenced and not has_hard_barrier:
            self._require_retained_capacity()
            return b""
        if has_hard_barrier:
            self._redacted_marker_fenced = False
        if (
            not final
            and self._marker_overlap_risk
            and self._redacted_pending.contains_marker
            and not has_hard_barrier
        ):
            self._redacted_marker_fenced = True
            self._require_retained_capacity()
            return b""
        if (
            not final
            and not shape_changed
            and len(self._redacted_pending) < self._redacted_recheck_at
        ):
            self._require_retained_capacity()
            return b""
        source = bytes(self._redacted_pending)
        pieces = _stabilize_redacted_pieces(
            source,
            secret_patterns=self._secret_patterns,
        )
        stabilized = b"".join(piece.value for piece in pieces)
        if final:
            released = stabilized
            self._redacted_pending.clear()
            self._redacted_recheck_at = 0
            self._redacted_marker_fenced = False
            return released

        # The first-pass source remains authoritative until every stabilized
        # piece derived from it is proven independent of future input. This
        # lets a later fragment recompute provisional marker-edge merges rather
        # than inheriting a chunk-boundary-dependent rewrite.
        source_cut = _future_sensitive_suffix_start(
            source,
            secret_patterns=self._secret_patterns,
        )
        output_cut = _future_sensitive_stabilized_suffix_start(
            stabilized,
            pieces=pieces,
            secret_patterns=self._secret_patterns,
        )
        released, source_release = _release_stable_pieces(
            pieces,
            output_cut=output_cut,
            source_cut=source_cut,
        )
        if source_release:
            self._redacted_pending.discard_prefix(source_release)
            self._redacted_recheck_at = 0
        else:
            retained_length = len(self._redacted_pending)
            self._redacted_recheck_at = max(
                retained_length + 1,
                retained_length * 2,
            )
        if self._marker_overlap_risk and self._redacted_pending.contains_marker:
            # Marker-crossing rewrites can expose another match arbitrarily far
            # left or right. Retain their exact compact source until EOF or a
            # byte outside both the marker and every registered pattern proves
            # a synchronization boundary.
            self._redacted_marker_fenced = True
        self._require_retained_capacity(released=released)
        return released

    def _require_retained_capacity(self, *, released: bytes = b"") -> None:
        """Fail closed if future-dependent source exceeded its declared bound."""

        compact_result = self._redacted_pending.stabilized_uniform_marker_run(
            collapse_threshold=self._marker_run_collapse_threshold,
        )
        redacted_retained = (
            self._redacted_pending.storage_bytes
            if compact_result is not None
            else len(self._redacted_pending)
        )
        if (
            self._max_retained_bytes is None
            or len(self._pending) + redacted_retained <= self._max_retained_bytes
        ):
            return
        self._pending.clear()
        self._redacted_pending.clear()
        self._aborted = True
        self._finished = True
        raise SecretRedactionCapacityError(released=released)


class SecretRedactionTail:
    """Retain a marker-safe bounded tail from an incrementally redacted source."""

    def __init__(self, redactor: SecretRedactor, *, max_bytes: int) -> None:
        if not isinstance(redactor, SecretRedactor):
            raise TypeError("redactor must be a SecretRedactor.")
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer.")
        self._redactor = redactor
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._stream = redactor.stream_bytes(
            max_retained_bytes=_bounded_stream_retention_bytes(
                redactor,
                max_bytes=max_bytes,
            )
        )
        self._max_bytes = max_bytes
        self._content = bytearray()
        self._sealed = False

    def feed(self, chunk: bytes) -> None:
        if self._sealed:
            return
        if type(chunk) is not bytes:
            raise TypeError("SecretRedactionTail.feed expects bytes.")
        for offset in range(0, len(chunk), _STREAM_INPUT_CHUNK_BYTES):
            normalized = self._decoder.decode(
                chunk[offset : offset + _STREAM_INPUT_CHUNK_BYTES],
                final=False,
            ).encode("utf-8")
            try:
                self._append_redacted(self._stream.feed(normalized))
            except SecretRedactionCapacityError as exc:
                self._append_redacted(exc.released)
                # An adversarial marker-overlap chain can make exact output
                # depend on an unbounded future suffix. Preserve only
                # already-proven output instead of retaining or publishing the
                # ambiguous source.
                self._sealed = True
                break

    def finish_complete(self) -> None:
        if self._sealed:
            return
        normalized = self._decoder.decode(b"", final=True).encode("utf-8")
        try:
            self._append_redacted(self._stream.feed(normalized))
            self._append_redacted(self._stream.finish_complete())
        except SecretRedactionCapacityError as exc:
            self._append_redacted(exc.released)
            # EOF can make the decoder release a final fragment after the last
            # ordinary feed. Apply the same fail-closed capacity policy.
            pass
        self._sealed = True

    def abort(self) -> None:
        if self._sealed:
            return
        self._stream.abort()
        self._sealed = True

    def text(self) -> str:
        content = bytes(self._content)
        marker = REDACTED_SECRET.encode("utf-8")
        cut = len(content) - self._max_bytes
        marker_start = content.rfind(marker, 0, max(0, cut) + len(marker))
        if (
            self._max_bytes >= len(marker)
            and marker_start >= 0
            and marker_start < cut < marker_start + len(marker)
        ):
            suffix = content[marker_start + len(marker) :]
            suffix_budget = self._max_bytes - len(marker)
            bounded = marker + (
                _bounded_redacted_tail(suffix, max_bytes=suffix_budget) if suffix_budget else b""
            )
            candidate = bounded.decode("utf-8", "ignore")
            if self._redactor.redact_text(candidate) != candidate:
                bounded = _bounded_redacted_tail(content, max_bytes=self._max_bytes)
        else:
            bounded = _bounded_redacted_tail(content, max_bytes=self._max_bytes)
        return bounded.decode("utf-8", "ignore")

    def _append_redacted(self, chunk: bytes) -> None:
        if not chunk:
            return
        self._content.extend(chunk)
        capture_limit = self._max_bytes + 2 * len(REDACTED_SECRET.encode("utf-8"))
        if len(self._content) > capture_limit:
            del self._content[: len(self._content) - capture_limit]


def _bounded_stream_retention_bytes(
    redactor: SecretRedactor,
    *,
    max_bytes: int,
) -> int:
    """Bound unresolved source proportionally to the public output contract."""

    overlap = redactor.max_secret_utf8_bytes + len(REDACTED_SECRET.encode("utf-8"))
    return max(
        _MIN_BOUNDED_STREAM_RETENTION_BYTES,
        max_bytes + 4 * overlap,
    )


def _bounded_redacted_head(value: bytes, *, max_bytes: int) -> bytes:
    """Return a UTF-8 prefix without splitting the public redaction marker."""

    if max_bytes <= 0:
        return b""
    if len(value) <= max_bytes:
        return value
    marker = REDACTED_SECRET.encode("utf-8")
    marker_start = value.rfind(marker, 0, max_bytes + len(marker))
    if marker_start >= 0 and marker_start < max_bytes < marker_start + len(marker):
        value = value[:marker_start]
    else:
        value = value[:max_bytes]
    return value.decode("utf-8", "ignore").encode("utf-8")


def _bounded_redacted_tail(value: bytes, *, max_bytes: int) -> bytes:
    """Return a UTF-8 suffix without splitting the public redaction marker."""

    if max_bytes <= 0:
        return b""
    if len(value) <= max_bytes:
        return value
    cut = len(value) - max_bytes
    marker = REDACTED_SECRET.encode("utf-8")
    marker_start = value.rfind(marker, 0, cut + len(marker))
    if marker_start >= 0 and marker_start < cut < marker_start + len(marker):
        value = value[marker_start + len(marker) :]
    else:
        value = value[cut:]
    return value.decode("utf-8", "ignore").encode("utf-8")


def _marker_safe_prefix_length(value: bytes, cut: int) -> int:
    """Move a proposed prefix boundary before an intersected public marker."""

    marker = REDACTED_SECRET.encode("utf-8")
    marker_start = value.rfind(marker, 0, cut + len(marker))
    if marker_start >= 0 and marker_start < cut < marker_start + len(marker):
        return marker_start
    return cut


def _redacted_output_holdback(
    value: bytes,
    *,
    secret_patterns: tuple[bytes, ...],
) -> int:
    """Return the suffix that any future redacted fragment could make unsafe."""

    if not value or not secret_patterns:
        return 0
    retained = 0
    for pattern in secret_patterns:
        maximum_prefix = min(len(pattern) - 1, len(value))
        for prefix_length in range(maximum_prefix, 0, -1):
            if value.endswith(pattern[:prefix_length]):
                retained = max(retained, prefix_length)
                break
    return retained


def _future_sensitive_suffix_start(
    value: bytes,
    *,
    secret_patterns: tuple[bytes, ...],
) -> int:
    """Return the earliest byte a future fragment could still affect."""

    cut = len(value)
    while cut:
        retained = _redacted_output_holdback(
            value[:cut],
            secret_patterns=secret_patterns,
        )
        if retained == 0:
            break
        next_cut = _marker_safe_prefix_length(value, cut - retained)
        if next_cut >= cut:
            raise AssertionError("Secret redaction suffix closure made no progress.")
        cut = next_cut
    return cut


def _future_sensitive_stabilized_suffix_start(
    value: bytes,
    *,
    pieces: list[_StabilizedPiece],
    secret_patterns: tuple[bytes, ...],
) -> int:
    """Close a future marker-producing match over the stabilized left edge."""

    cut = _future_sensitive_suffix_start(
        value,
        secret_patterns=secret_patterns,
    )
    marker = REDACTED_SECRET.encode("utf-8")
    if any(_pattern_can_cross_marker(pattern) for pattern in secret_patterns):
        pattern_alphabet = frozenset().union(*(frozenset(pattern) for pattern in secret_patterns))
        last_barrier = -1
        cursor = 0
        for marker_start, marker_end in _marker_spans(value, marker):
            for index in range(cursor, marker_start):
                if value[index] not in pattern_alphabet:
                    last_barrier = index
            cursor = marker_end
        for index in range(cursor, len(value)):
            if value[index] not in pattern_alphabet:
                last_barrier = index
        marker_start = value.find(marker, last_barrier + 1)
        if marker_start >= 0:
            cut = min(cut, marker_start)
    while cut:
        prefix = value[:cut]
        hypothetical = _stabilize_redacted_bytes(
            prefix + marker,
            secret_patterns=secret_patterns,
        )
        if hypothetical.startswith(prefix) and len(hypothetical) > len(prefix):
            break
        common = (
            len(prefix) - 1
            if hypothetical == prefix
            else _common_prefix_length(prefix, hypothetical)
        )
        next_cut = _marker_safe_prefix_length(value, common)
        next_cut = _stabilized_piece_safe_prefix_length(
            pieces,
            next_cut,
        )
        if next_cut >= cut:
            raise AssertionError("Secret redaction future-suffix closure made no progress.")
        cut = next_cut
    return cut


def _pattern_can_cross_marker(pattern: bytes) -> bool:
    """Return whether a match can contain marker and non-marker bytes."""

    marker = REDACTED_SECRET.encode("utf-8")
    for pattern_start in range(-len(pattern) + 1, len(marker)):
        pattern_end = pattern_start + len(pattern)
        overlap_start = max(0, pattern_start)
        overlap_end = min(len(marker), pattern_end)
        if overlap_start >= overlap_end:
            continue
        pattern_overlap_start = overlap_start - pattern_start
        pattern_overlap_end = overlap_end - pattern_start
        if marker[overlap_start:overlap_end] == pattern[
            pattern_overlap_start:pattern_overlap_end
        ] and (pattern_start < 0 or pattern_end > len(marker)):
            return True
    return False


def _marker_run_collapse_threshold(
    secret_patterns: tuple[bytes, ...],
) -> int | None:
    """Return the smallest marker run that one periodic match collapses."""

    marker = REDACTED_SECRET.encode("utf-8")
    threshold: int | None = None
    for pattern in secret_patterns:
        sample_repetitions = (len(pattern) + 2 * len(marker) - 2) // len(marker) + 1
        sample = marker * sample_repetitions
        for residue in range(len(marker)):
            if sample[residue : residue + len(pattern)] != pattern:
                continue
            touched_markers = (residue + len(pattern) + len(marker) - 1) // len(marker)
            if touched_markers < 2:
                continue
            threshold = touched_markers if threshold is None else min(threshold, touched_markers)
    return threshold


def _common_prefix_length(left: bytes, right: bytes) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


@dataclass(frozen=True, slots=True)
class _StabilizedPiece:
    value: bytes
    source_start: int
    source_end: int
    linear: bool


def _stabilized_piece_safe_prefix_length(
    pieces: list[_StabilizedPiece],
    cut: int,
) -> int:
    """Move a cut before an indivisible replacement piece."""

    offset = 0
    for piece in pieces:
        end = offset + len(piece.value)
        if offset < cut < end and not piece.linear:
            return offset
        if cut <= end:
            return cut
        offset = end
    return cut


def _release_stable_pieces(
    pieces: list[_StabilizedPiece],
    *,
    output_cut: int,
    source_cut: int,
) -> tuple[bytes, int]:
    """Release the stable output prefix and its exact first-pass ownership."""

    released = bytearray()
    output_offset = 0
    source_release = 0
    for piece in pieces:
        if piece.source_start != source_release:
            raise AssertionError("Stabilized redaction provenance is not contiguous.")
        output_remaining = output_cut - output_offset
        source_remaining = source_cut - piece.source_start
        if output_remaining <= 0 or source_remaining <= 0:
            break
        if len(piece.value) <= output_remaining and piece.source_end <= source_cut:
            released.extend(piece.value)
            output_offset += len(piece.value)
            source_release = piece.source_end
            continue
        if not piece.linear:
            break
        retained = min(len(piece.value), output_remaining, source_remaining)
        if retained <= 0:
            break
        released.extend(piece.value[:retained])
        source_release = piece.source_start + retained
        break
    return bytes(released), source_release


def _stabilize_redacted_bytes(
    value: bytes,
    *,
    secret_patterns: tuple[bytes, ...],
) -> bytes:
    """Collapse secrets reconstructed across otherwise-atomic marker edges.

    The first redaction pass can create a registered short secret where a
    literal prefix/suffix meets ``REDACTED_SECRET``, or between two adjacent
    markers. Matches wholly inside a public marker are intentionally ignored.
    Any match touching a marker consumes that complete marker, which guarantees
    progress and keeps the public token atomic.
    """

    return b"".join(
        piece.value
        for piece in _stabilize_redacted_pieces(
            value,
            secret_patterns=secret_patterns,
        )
    )


def _stabilize_redacted_pieces(
    value: bytes,
    *,
    secret_patterns: tuple[bytes, ...],
) -> list[_StabilizedPiece]:
    """Return canonical redaction plus first-pass byte provenance."""

    pieces = (
        [
            _StabilizedPiece(
                value=value,
                source_start=0,
                source_end=len(value),
                linear=True,
            )
        ]
        if value
        else []
    )
    return _stabilize_redacted_piece_sequence(
        pieces,
        secret_patterns=secret_patterns,
    )


def _source_redaction_pieces(
    value: bytes,
    *,
    ordered_patterns: tuple[tuple[bytes, bool], ...],
) -> list[_StabilizedPiece]:
    """Redact one complete source window while retaining raw byte ownership."""

    pieces: list[_StabilizedPiece] = []
    cursor = 0
    marker = REDACTED_SECRET.encode("utf-8")
    pattern = re.compile(b"|".join(re.escape(candidate) for candidate, _ in ordered_patterns))
    redacted_patterns = frozenset(candidate for candidate, redact in ordered_patterns if redact)
    for match in pattern.finditer(value):
        if match.start() > cursor:
            _append_stabilized_piece(
                pieces,
                _StabilizedPiece(
                    value=value[cursor : match.start()],
                    source_start=cursor,
                    source_end=match.start(),
                    linear=True,
                ),
            )
        matched = match.group()
        _append_stabilized_piece(
            pieces,
            _StabilizedPiece(
                value=marker if matched in redacted_patterns else matched,
                source_start=match.start(),
                source_end=match.end(),
                linear=False,
            ),
        )
        cursor = match.end()
    if cursor < len(value):
        _append_stabilized_piece(
            pieces,
            _StabilizedPiece(
                value=value[cursor:],
                source_start=cursor,
                source_end=len(value),
                linear=True,
            ),
        )
    return pieces


def _stabilize_redacted_piece_sequence(
    pieces: list[_StabilizedPiece],
    *,
    secret_patterns: tuple[bytes, ...],
) -> list[_StabilizedPiece]:
    """Canonicalize marker edges while preserving existing source provenance."""

    current = b"".join(piece.value for piece in pieces)
    marker = REDACTED_SECRET.encode("utf-8")
    if not secret_patterns or marker not in current:
        return pieces
    while True:
        marker_spans = _marker_spans(current, marker)
        marker_starts = tuple(start for start, _ in marker_spans)
        marker_ends = tuple(end for _, end in marker_spans)
        replacements: list[tuple[int, int]] = []
        for pattern in secret_patterns:
            start = current.find(pattern)
            while start >= 0:
                end = start + len(pattern)
                if not _inside_one_marker(
                    start,
                    end,
                    marker_starts=marker_starts,
                    marker_ends=marker_ends,
                ):
                    replacement_start = start
                    replacement_end = end
                    marker_index = bisect_right(marker_ends, replacement_start)
                    while (
                        marker_index < len(marker_starts)
                        and marker_starts[marker_index] < replacement_end
                    ):
                        replacement_start = min(
                            replacement_start,
                            marker_starts[marker_index],
                        )
                        replacement_end = max(
                            replacement_end,
                            marker_ends[marker_index],
                        )
                        marker_index += 1
                    replacements.append((replacement_start, replacement_end))
                start = current.find(pattern, start + 1)
        if not replacements:
            return pieces

        merged: list[tuple[int, int]] = []
        for start, end in sorted(replacements):
            if merged and start < merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        rebuilt = bytearray()
        rebuilt_pieces: list[_StabilizedPiece] = []
        piece_starts = _stabilized_piece_starts(pieces)
        cursor = 0
        for start, end in merged:
            rebuilt.extend(current[cursor:start])
            for piece in _slice_stabilized_pieces(
                pieces,
                piece_starts=piece_starts,
                start=cursor,
                end=start,
            ):
                _append_stabilized_piece(rebuilt_pieces, piece)
            consumed = _slice_stabilized_pieces(
                pieces,
                piece_starts=piece_starts,
                start=start,
                end=end,
            )
            if not consumed:
                raise AssertionError("Secret redaction replacement consumed no provenance.")
            rebuilt.extend(marker)
            _append_stabilized_piece(
                rebuilt_pieces,
                _StabilizedPiece(
                    value=marker,
                    source_start=consumed[0].source_start,
                    source_end=consumed[-1].source_end,
                    linear=False,
                ),
            )
            cursor = end
        rebuilt.extend(current[cursor:])
        for piece in _slice_stabilized_pieces(
            pieces,
            piece_starts=piece_starts,
            start=cursor,
            end=len(current),
        ):
            _append_stabilized_piece(rebuilt_pieces, piece)
        updated = bytes(rebuilt)
        if updated == current:
            raise AssertionError("Secret redaction stabilization made no progress.")
        current = updated
        pieces = rebuilt_pieces


def _inside_one_marker(
    start: int,
    end: int,
    *,
    marker_starts: tuple[int, ...],
    marker_ends: tuple[int, ...],
) -> bool:
    marker_index = bisect_right(marker_starts, start) - 1
    return marker_index >= 0 and end <= marker_ends[marker_index]


def _stabilized_piece_starts(pieces: list[_StabilizedPiece]) -> tuple[int, ...]:
    starts: list[int] = []
    offset = 0
    for piece in pieces:
        starts.append(offset)
        offset += len(piece.value)
    return tuple(starts)


def _slice_stabilized_pieces(
    pieces: list[_StabilizedPiece],
    *,
    piece_starts: tuple[int, ...],
    start: int,
    end: int,
) -> list[_StabilizedPiece]:
    if start >= end:
        return []
    piece_index = max(0, bisect_right(piece_starts, start) - 1)
    sliced: list[_StabilizedPiece] = []
    while piece_index < len(pieces):
        piece = pieces[piece_index]
        piece_start = piece_starts[piece_index]
        piece_end = piece_start + len(piece.value)
        if piece_start >= end:
            break
        local_start = max(start, piece_start) - piece_start
        local_end = min(end, piece_end) - piece_start
        if local_start < local_end:
            if not piece.linear and (local_start != 0 or local_end != len(piece.value)):
                raise AssertionError("Secret redaction split an atomic marker piece.")
            source_start = piece.source_start + local_start if piece.linear else piece.source_start
            source_end = piece.source_start + local_end if piece.linear else piece.source_end
            sliced.append(
                _StabilizedPiece(
                    value=piece.value[local_start:local_end],
                    source_start=source_start,
                    source_end=source_end,
                    linear=piece.linear,
                )
            )
        piece_index += 1
    return sliced


def _append_stabilized_piece(
    pieces: list[_StabilizedPiece],
    piece: _StabilizedPiece,
) -> None:
    if not piece.value:
        return
    if (
        pieces
        and pieces[-1].linear
        and piece.linear
        and pieces[-1].source_end == piece.source_start
    ):
        previous = pieces[-1]
        pieces[-1] = _StabilizedPiece(
            value=previous.value + piece.value,
            source_start=previous.source_start,
            source_end=piece.source_end,
            linear=True,
        )
        return
    pieces.append(piece)


def _marker_spans(value: bytes, marker: bytes) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    start = value.find(marker)
    while start >= 0:
        spans.append((start, start + len(marker)))
        start = value.find(marker, start + len(marker))
    return tuple(spans)


def _redaction_pattern(values: tuple[str, ...]) -> re.Pattern[str] | None:
    if not values:
        return None
    # A secret that starts with the public marker must win at the same position;
    # the marker must otherwise win over shorter secrets contained inside it.
    # ``re.sub`` does not rescan replacements, so one pass remains idempotent.
    marker_prefixed = [
        value for value in values if value != REDACTED_SECRET and value.startswith(REDACTED_SECRET)
    ]
    excluded = {*marker_prefixed, REDACTED_SECRET}
    remaining = [value for value in values if value not in excluded]
    alternatives = [
        *(re.escape(value) for value in marker_prefixed),
        re.escape(REDACTED_SECRET),
        *(re.escape(value) for value in remaining),
    ]
    return re.compile("|".join(alternatives))


def _secret_value(secret: str | SecretStr | ResolvedSecret) -> str:
    if type(secret) is str:
        return require_nonblank(secret, "secret")
    if type(secret) is SecretStr:
        return require_nonblank(secret.get_secret_value(), "secret")
    if type(secret) is ResolvedSecret:
        return require_nonblank(secret.value.get_secret_value(), "secret")
    raise TypeError("SecretRedactor secrets must be str, SecretStr, or ResolvedSecret.")


def _secret_items(
    secrets: str | SecretStr | ResolvedSecret | Sequence[str | SecretStr | ResolvedSecret] | None,
) -> tuple[str | SecretStr | ResolvedSecret, ...]:
    if secrets is None:
        return ()
    if type(secrets) is str:
        return (secrets,)
    if type(secrets) is SecretStr:
        return (secrets,)
    if type(secrets) is ResolvedSecret:
        return (secrets,)
    if not isinstance(secrets, Sequence):
        raise TypeError("SecretRedactor secrets must be a secret or a sequence of secrets.")
    items: list[str | SecretStr | ResolvedSecret] = []
    for secret in secrets:
        items.append(_as_secret(secret))
    return tuple(items)


def _as_secret(secret: object) -> str | SecretStr | ResolvedSecret:
    if type(secret) is str:
        return secret
    if type(secret) is SecretStr:
        return secret
    if type(secret) is ResolvedSecret:
        return secret
    raise TypeError("SecretRedactor secrets must be str, SecretStr, or ResolvedSecret.")
