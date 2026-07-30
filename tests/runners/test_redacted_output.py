from __future__ import annotations

from cayu.runners import ExecResult
from cayu.runners._redacted_output import RedactedOutputCapture, redact_completed_exec_result
from cayu.vaults import REDACTED_SECRET, SecretRedactor


def test_redacted_output_capture_counts_raw_bytes_before_replacement() -> None:
    secret = "capture-boundary-secret"
    capture = RedactedOutputCapture(redactor=SecretRedactor(secret), limit=64)

    for byte in f"before:{secret}:after".encode():
        capture.append(bytes([byte]))
    capture.finish_complete()

    assert capture.text() == f"before:{REDACTED_SECRET}:after"
    assert capture.total_bytes == len(f"before:{secret}:after".encode())
    assert capture.truncated is False


def test_redacted_output_capture_abandonment_drops_only_undecided_suffix() -> None:
    secret = "capture-abort-secret"
    capture = RedactedOutputCapture(redactor=SecretRedactor(secret), limit=128)
    capture.append(b"public:")
    capture.append(secret[:8].encode())

    capture.abort()
    capture.append(secret[8:].encode())

    assert capture.text() == "public:"
    assert capture.total_bytes == len(b"public:") + len(secret[:8].encode())
    assert capture.truncated is True


def test_redacted_output_capture_abandonment_always_reports_truncation() -> None:
    capture = RedactedOutputCapture(redactor=SecretRedactor("secret"), limit=128)
    capture.append(b"complete-safe-chunk")

    capture.abort()

    assert capture.text() == "complete-safe-chunk"
    assert capture.total_bytes == len(b"complete-safe-chunk")
    assert capture.truncated is True


def test_redacted_output_capture_does_not_reconstruct_secret_at_marker_edge() -> None:
    secrets = ["b密[", "c[[]b", "]cb]"]
    source = "b密c[[]bcéab]é密"
    redactor = SecretRedactor(secrets)
    capture = RedactedOutputCapture(redactor=redactor, limit=50)

    for byte in source.encode("utf-8"):
        capture.append(bytes([byte]))
    capture.finish_complete()
    output = capture.text()

    assert redactor.redact_text(output) == output
    assert not any(secret in output for secret in secrets)
    assert len(output.encode("utf-8")) <= 50


def test_redacted_output_capture_is_split_independent_for_marker_overlap_chain() -> None:
    secrets = ["[", "]密", "密["]
    source = "[[密密[密é]ba]密][b"
    redactor = SecretRedactor(secrets)
    expected = redactor.redact_text(source)

    for split in range(len(source.encode()) + 1):
        capture = RedactedOutputCapture(redactor=redactor, limit=256)
        encoded = source.encode()
        capture.append(encoded[:split])
        capture.append(encoded[split:])
        capture.finish_complete()

        assert capture.text() == expected
        assert capture.total_bytes == len(encoded)
        assert capture.truncated is False


def test_redacted_output_capture_keeps_resolved_prefix_when_aborted() -> None:
    capture = RedactedOutputCapture(
        redactor=SecretRedactor(["[", "]["]),
        limit=256,
    )

    capture.append(b"[")
    capture.append(b"[")
    capture.append(b"[")
    capture.append(b"x")
    capture.abort()

    assert capture.text() == f"{REDACTED_SECRET}x"
    assert capture.total_bytes == 4
    assert capture.truncated is True


def test_redacted_output_capture_bounds_an_ambiguous_literal_chain() -> None:
    capture = RedactedOutputCapture(
        redactor=SecretRedactor(["[", "]a"]),
        limit=32,
    )

    capture.append(b"public:" + b"[" + b"a" * 100_000)
    capture.finish_complete()

    assert capture.text() == "public:"
    assert capture.total_bytes == len(b"public:") + 100_001
    assert capture.truncated is True
    assert capture._stream._redacted_pending.storage_bytes == 0


def test_redacted_output_capture_bounds_an_ambiguous_marker_run() -> None:
    capture = RedactedOutputCapture(
        redactor=SecretRedactor(["[", "]["]),
        limit=32,
    )

    capture.append(b"[" * 100_000)
    capture.finish_complete()

    assert capture.text() == REDACTED_SECRET
    assert capture.total_bytes == 100_000
    assert capture.truncated is True
    assert capture._stream._redacted_pending.storage_bytes == 0


def test_redacted_output_capacity_keeps_the_proven_prefix_before_marker_run() -> None:
    capture = RedactedOutputCapture(
        redactor=SecretRedactor(["[", "]["]),
        limit=32,
    )

    capture.append(b"public:" + b"[" * 100_000)
    capture.finish_complete()

    assert capture.text() == f"public:{REDACTED_SECRET}"
    assert capture.total_bytes == len(b"public:") + 100_000
    assert capture.truncated is True


def test_compact_marker_stabilization_preserves_a_below_limit_result() -> None:
    capture = RedactedOutputCapture(
        redactor=SecretRedactor(["[", "]["]),
        limit=10_000,
    )

    capture.append(b"[" * 4_000)
    capture.finish_complete()

    assert capture.text() == REDACTED_SECRET
    assert capture.total_bytes == 4_000
    assert capture.truncated is False


def test_redacted_output_capacity_does_not_reject_determined_eof_output() -> None:
    capture = RedactedOutputCapture(
        redactor=SecretRedactor(["a", "a" * 10_000]),
        limit=32,
    )

    capture.append(b"a" * 9_999)
    capture.finish_complete()

    assert capture.text() == REDACTED_SECRET
    assert capture.total_bytes == 9_999
    assert capture.truncated is True


def test_custom_runner_fallback_omits_pretruncated_ambiguous_channel() -> None:
    result = ExecResult(
        stdout="public-prefix",
        stderr="complete secret-value diagnostic",
        stdout_truncated=True,
        stderr_truncated=False,
        stdout_bytes=1_000,
        stderr_bytes=32,
    )

    redacted = redact_completed_exec_result(
        result,
        redactor=SecretRedactor("secret-value"),
        output_limit_bytes=64,
        omit_pretruncated=True,
    )

    assert redacted.stdout == ""
    assert redacted.stdout_truncated is True
    assert redacted.stdout_bytes == 1_000
    assert redacted.stderr == f"complete {REDACTED_SECRET} diagnostic"
    assert redacted.stderr_bytes == 32


def test_redacted_output_capture_preserves_invalid_byte_replacement_within_bound() -> None:
    capture = RedactedOutputCapture(redactor=SecretRedactor(), limit=4)
    capture.append(b"a\xffb")
    capture.finish_complete()

    assert capture.text() == "a�"
    assert capture.total_bytes == 3
    assert capture.truncated is True


def test_redacted_output_capture_matches_secrets_synthesized_by_utf8_replacement() -> None:
    secret = "a\ufffdb"
    capture = RedactedOutputCapture(redactor=SecretRedactor(secret), limit=64)

    capture.append(b"a")
    capture.append(b"\xff")
    capture.append(b"b")
    capture.finish_complete()

    assert capture.text() == REDACTED_SECRET
    assert capture.total_bytes == 3
    assert capture.truncated is False


def test_redacted_output_capture_stops_redacting_after_bounded_prefix_is_final() -> None:
    capture = RedactedOutputCapture(
        redactor=SecretRedactor("registered-secret"),
        limit=8,
    )
    redacted_source_bytes = 0
    original_feed = capture._stream.feed

    def recording_feed(chunk: bytes) -> bytes:
        nonlocal redacted_source_bytes
        redacted_source_bytes += len(chunk)
        return original_feed(chunk)

    capture._stream.feed = recording_feed  # type: ignore[method-assign]
    first = b"x" * 4096
    discarded = b"registered-secret" * 100_000

    capture.append(first)
    redacted_after_prefix = redacted_source_bytes
    capture.append(discarded)
    capture.finish_complete()

    assert capture.text() == "x" * 8
    assert capture.total_bytes == len(first) + len(discarded)
    assert capture.truncated is True
    assert redacted_after_prefix <= 4096
    assert redacted_source_bytes == redacted_after_prefix
