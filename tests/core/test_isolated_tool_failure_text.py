"""Isolated tool failures carry the child's text, shown only after redaction."""

from __future__ import annotations

import io
import sys

from cayu.runtime import _isolated_tool_worker as worker
from cayu.runtime._isolated_tool_process import (
    IsolatedToolFailure,
    attach_isolated_failure_text,
    isolated_failure_text,
)
from cayu.runtime._isolated_tool_protocol import (
    MAX_ISOLATED_TOOL_ERROR_MESSAGE_BYTES,
    IsolatedToolChildErrorCode,
    IsolatedToolChildFailure,
    decode_isolated_tool_response,
    encode_isolated_tool_error,
)
from cayu.runtime._tool_execution import _isolated_failure_message
from cayu.vaults import SecretRedactor
from cayu.vaults.redaction import REDACTED_SECRET

_SHA = "sha256:" + "a" * 64


def test_protocol_round_trips_child_failure_text() -> None:
    encoded = encode_isolated_tool_error(
        request_sha256=_SHA,
        error_code=IsolatedToolChildErrorCode.CHILD_EXCEPTION,
        max_bytes=64 * 1024,
        error_message="ValueError: bad row 3",
        stderr_tail="Traceback ...\nValueError: bad row 3",
    )

    decoded = decode_isolated_tool_response(
        encoded, expected_request_sha256=_SHA, max_bytes=64 * 1024
    )

    assert decoded == IsolatedToolChildFailure(
        code=IsolatedToolChildErrorCode.CHILD_EXCEPTION,
        message="ValueError: bad row 3",
        stderr_tail="Traceback ...\nValueError: bad row 3",
    )


def test_protocol_drops_text_that_does_not_fit_instead_of_failing() -> None:
    encoded = encode_isolated_tool_error(
        request_sha256=_SHA,
        error_code=IsolatedToolChildErrorCode.CHILD_EXCEPTION,
        max_bytes=2048,
        error_message="short",
        stderr_tail="y" * 4000,
    )

    decoded = decode_isolated_tool_response(encoded, expected_request_sha256=_SHA, max_bytes=2048)

    assert decoded == IsolatedToolChildFailure(
        code=IsolatedToolChildErrorCode.CHILD_EXCEPTION, message="short", stderr_tail=None
    )


def test_child_omits_oversized_messages_instead_of_cutting_them() -> None:
    assert worker._failure_message(ValueError("bad input")) == "ValueError: bad input"
    huge = ValueError("x" * (MAX_ISOLATED_TOOL_ERROR_MESSAGE_BYTES + 1))
    assert worker._failure_message(huge) is None


def test_child_stderr_passes_through_and_keeps_a_tail(monkeypatch) -> None:
    target = io.StringIO()
    monkeypatch.setattr(sys, "stderr", target)

    with worker._capture_stderr_tail() as captured:
        print("x" * (worker._STDERR_TAIL_CHARS + 10), file=sys.stderr)
        print("last line", file=sys.stderr)

    assert sys.stderr is target
    assert target.getvalue().endswith("last line\n")
    assert len(captured.tail) == worker._STDERR_TAIL_CHARS
    assert captured.tail.endswith("last line\n")


def test_tool_message_redacts_child_text_before_bounding() -> None:
    secret = "workload-canary-ABCDEFGHIJKLMNOP"
    failure = IsolatedToolFailure("child_child_exception")
    attach_isolated_failure_text(
        failure,
        f"RuntimeError: token {secret}",
        "z" * 5000 + secret + "\nfinal error line",
    )

    message = _isolated_failure_message(failure, SecretRedactor([secret]))

    assert message.startswith("Isolated tool process execution failed.\n")
    assert f"RuntimeError: token {REDACTED_SECRET}" in message
    assert message.endswith("final error line")
    assert not any(secret[:size] in message for size in range(8, len(secret) + 1))
    # The failure itself keeps its fixed text.
    assert str(failure) == "Isolated tool execution failed: child_child_exception."
    assert secret not in repr(vars(failure))


def test_tool_message_without_child_text_is_the_fixed_headline() -> None:
    failure = IsolatedToolFailure("child_invalid_result")

    assert isolated_failure_text(failure) == (None, None)
    assert _isolated_failure_message(failure, SecretRedactor()) == (
        "Isolated tool process execution failed."
    )


def test_child_stderr_wrapper_delegates_the_real_stream(monkeypatch) -> None:
    class Stream(io.StringIO):
        def fileno(self) -> int:
            return 2

        def isatty(self) -> bool:
            return True

    target = Stream()
    monkeypatch.setattr(sys, "stderr", target)

    with worker._capture_stderr_tail():
        assert sys.stderr.fileno() == 2
        assert sys.stderr.isatty() is True
        sys.stderr.flush()


_WHITESPACE_SECRETS = (
    "CANARY-A\nCANARY-B",
    "CANARY-A\r\nCANARY-B",
    "CANARY-A  \t  CANARY-B",
    "  CANARY-LEADING",
    "CANARY-TRAILING\n",
)


def test_child_sends_its_text_unchanged_for_the_parent_to_redact() -> None:
    text = "line one\r\nline\x1btwo\t "
    assert worker._failure_message(RuntimeError(text)) == f"RuntimeError: {text}"
    assert worker._sendable_text(text) == text
    # The protocol cannot carry NUL or lone surrogates, so such text is omitted.
    assert worker._sendable_text("bad\x00text") is None
    assert worker._sendable_text("bad\ud800text") is None


def test_whitespace_secrets_are_redacted_before_display_normalization() -> None:
    for secret in _WHITESPACE_SECRETS:
        failure = IsolatedToolFailure("child_child_exception")
        message = worker._failure_message(RuntimeError(f"failed: {secret} end"))
        stderr = worker._sendable_text(f"Traceback:\n{secret}\nfinal\r\n")
        attach_isolated_failure_text(failure, message, stderr)

        shown = _isolated_failure_message(failure, SecretRedactor([secret]))

        assert "CANARY" not in shown, (secret, shown)
        assert REDACTED_SECRET in shown


def test_full_stderr_window_drops_a_secret_cut_at_its_start() -> None:
    secret = "workload-canary-" + "S" * 2000
    # The window starts mid-secret and redaction shrinks the rest, so without
    # dropping the head the fragment would land inside the displayed tail.
    window = (secret[1000:] + " " + secret * 7).ljust(worker._STDERR_TAIL_CHARS, " ")
    assert len(window) == worker._STDERR_TAIL_CHARS
    failure = IsolatedToolFailure("child_child_exception")
    attach_isolated_failure_text(failure, None, window)

    shown = _isolated_failure_message(failure, SecretRedactor([secret]))

    assert "SSSSSSSS" not in shown


def test_full_stderr_window_drops_a_nested_secret_cut_at_its_start() -> None:
    outer = "PREFIX_OUTER_INNER_SECRET_VALUE_1234_OUTER_SUFFIX_DATA_XYZ"
    inner = "INNER_SECRET_VALUE_1234"
    filler = "Z" * 400
    window = (outer[5:] + ("\n" + filler) * 50)[: worker._STDERR_TAIL_CHARS]
    window = window.ljust(worker._STDERR_TAIL_CHARS, "\n")
    failure = IsolatedToolFailure("child_child_exception")
    attach_isolated_failure_text(failure, None, window)

    shown = _isolated_failure_message(failure, SecretRedactor([outer, inner, filler]))

    assert "OUTER" not in shown and "SUFFIX" not in shown and "XYZ" not in shown


def test_repr_escaped_secrets_in_child_text_are_redacted() -> None:
    secret = "TOKEN\nPART2xyzabc"
    error = FileNotFoundError(2, "No such file", f"/run/{secret}")
    failure = IsolatedToolFailure("child_child_exception")
    attach_isolated_failure_text(
        failure,
        worker._failure_message(error),
        worker._sendable_text(f"Traceback:\nKeyError: {KeyError(secret)}\n"),
    )

    shown = _isolated_failure_message(failure, SecretRedactor([secret]))

    assert "PART2" not in shown
    assert "No such file" in shown


def test_isolated_failure_redacts_nested_and_encoded_secret_forms() -> None:
    import json

    cases = [
        ("TOKEN\nPART2xyzabcdef", lambda s: KeyError(repr(s))),
        ("crlf\r\nsecret-partZZ", lambda s: ValueError(repr(repr(s)))),
        ("c1\x85\x9fsecret-partZZ", lambda s: ValueError(repr(s.encode()))),
        ("astral\U0001f600secret-partZZ", lambda s: ValueError(ascii(s))),
        ('quote"ctl\x01secret-partZZ', lambda s: ValueError(json.dumps(s))),
        ("snow☃\nsecret-partZZ", lambda s: ValueError(json.dumps(s, ensure_ascii=False))),
    ]
    for secret, build in cases:
        error = build(secret)
        failure = IsolatedToolFailure("child_child_exception")
        attach_isolated_failure_text(
            failure,
            worker._failure_message(error),
            worker._sendable_text(f"Traceback:\n{type(error).__name__}: {error}\n"),
        )

        shown = _isolated_failure_message(failure, SecretRedactor([secret]))

        assert "secret-part" not in shown and "PART2" not in shown, (secret, shown)
