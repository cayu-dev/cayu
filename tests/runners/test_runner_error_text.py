"""Runner failures show the runner's own text, redacted before bounding."""

from __future__ import annotations

import pytest

from cayu.runners.base import RunnerExecutionError, runner_error_text, runner_execution_error
from cayu.runtime._tool_execution import _runner_failure_result
from cayu.tools.base import ToolEffect
from cayu.vaults import SecretRedactor
from cayu.vaults.redaction import REDACTED_SECRET


def test_runner_failure_shows_redacted_runner_text() -> None:
    secret = "workload-canary-0123456789"
    error = runner_execution_error(
        OSError(f"exec: python3: not found (token {secret})"), adapter="local"
    )

    # The detached error itself still carries only fixed text.
    assert str(error) == "Runner command execution failed."
    assert secret not in repr(vars(error))

    result, _controls = _runner_failure_result(
        error, effect=ToolEffect.NONE, redactor=SecretRedactor([secret])
    )

    assert result.content == (
        f"Runner command execution failed: exec: python3: not found (token {REDACTED_SECRET})"
    )
    assert result.structured["error"] == "runner_execution_failed"


def test_runner_failure_text_survives_re_detachment() -> None:
    first = runner_execution_error(ConnectionResetError("sandbox connection lost"), adapter="local")
    second = runner_execution_error(first, adapter="local")

    assert type(second) is RunnerExecutionError
    assert runner_error_text(second) == "sandbox connection lost"


def test_runner_failure_keeps_fixed_text_without_source_text() -> None:
    error = runner_execution_error(OSError(), adapter="local")
    oversized = runner_execution_error(OSError("x" * (64 * 1024 + 1)), adapter="local")

    for failure in (error, oversized):
        result, _controls = _runner_failure_result(
            failure, effect=ToolEffect.NONE, redactor=SecretRedactor()
        )
        assert result.content == "Runner command execution failed."


def test_runner_failure_text_is_bounded_after_redaction() -> None:
    from cayu.runtime._tool_results import _MAX_DIAGNOSTIC_UTF8_BYTES as bound

    secret = "workload-canary-ABCDEFGHIJKLMNOP"
    # The secret straddles the tool-failure bound, whatever that bound is.
    error = runner_execution_error(OSError("x" * (bound - 46) + secret), adapter="local")

    result, _controls = _runner_failure_result(
        error, effect=ToolEffect.NONE, redactor=SecretRedactor([secret])
    )

    assert not any(secret[:size] in result.content for size in range(8, len(secret) + 1))
    assert len(result.content.encode()) <= bound


class _Hooked:
    calls = 0

    def __str__(self) -> str:
        _Hooked.calls += 1
        return "hook ran"

    def __repr__(self) -> str:
        _Hooked.calls += 1
        return "hook ran"


class _Interrupting:
    def __str__(self) -> str:
        raise KeyboardInterrupt

    __repr__ = __str__


def _shown_text(error: BaseException) -> str:
    result, _controls = _runner_failure_result(
        runner_execution_error(error, adapter="local"),
        effect=ToolEffect.NONE,
        redactor=SecretRedactor(),
    )
    return result.content


def test_runner_failure_text_never_runs_argument_hooks() -> None:
    _Hooked.calls = 0

    assert _shown_text(RuntimeError(_Hooked())) == "Runner command execution failed."
    assert _shown_text(RuntimeError(_Hooked(), 1)) == "Runner command execution failed."
    assert _Hooked.calls == 0
    # A hook that would raise a non-Exception is never called either.
    assert _shown_text(RuntimeError(_Interrupting())) == "Runner command execution failed."


def test_runner_failure_text_omits_python_formatted_exceptions() -> None:
    import subprocess

    class Custom(RuntimeError):
        def __str__(self) -> str:
            return "custom text"

    called = subprocess.CalledProcessError(
        1, ["docker", "exec", "-e", "API_KEY=sk-live-canary", "box", "true"]
    )

    assert _shown_text(called) == "Runner command execution failed."
    assert _shown_text(Custom("x")) == "Runner command execution failed."


def test_runner_failure_text_keeps_os_error_details() -> None:
    error = FileNotFoundError(2, "No such file or directory", "python3")

    assert _shown_text(error) == (
        "Runner command execution failed: [Errno 2] No such file or directory: 'python3'"
    )
    error.filename = _Hooked()
    _Hooked.calls = 0
    assert _shown_text(error) == "Runner command execution failed."
    assert _Hooked.calls == 0


def test_runner_failure_text_omits_builtins_that_read_assignable_fields() -> None:
    _Hooked.calls = 0
    syntax = SyntaxError("bad syntax")
    syntax.msg = _Hooked()
    encode = UnicodeEncodeError("ascii", "é", 0, 1, "ordinal not in range")
    encode.reason = _Hooked()

    for error in (syntax, encode):
        assert _shown_text(error) == "Runner command execution failed."
    assert _Hooked.calls == 0
    assert _shown_text(KeyError("missing-key")) == (
        "Runner command execution failed: 'missing-key'"
    )


def test_runner_failure_text_keeps_import_errors_without_running_hooks() -> None:
    _Hooked.calls = 0
    missing = ModuleNotFoundError("No module named 'yaml'", name=_Hooked(), path=_Hooked())
    reassigned = ImportError("cannot import name 'x'")
    reassigned.msg = _Hooked()

    assert _shown_text(missing) == "Runner command execution failed: No module named 'yaml'"
    assert _shown_text(reassigned) == "Runner command execution failed: cannot import name 'x'"
    assert _Hooked.calls == 0


@pytest.mark.parametrize(
    "secret",
    [
        "first-canary-line\nsecond-canary-line",
        "first-canary\r\nsecond-canary",
        "canary  two \t spaces",
        "  leading-canary",
        "trailing-canary\n",
    ],
)
def test_runner_failure_redacts_whitespace_secrets_before_flattening(secret: str) -> None:
    error = runner_execution_error(OSError(f"failed: {secret} end"), adapter="local")

    result, _controls = _runner_failure_result(
        error, effect=ToolEffect.NONE, redactor=SecretRedactor([secret])
    )

    assert "canary" not in result.content
    assert result.content == f"Runner command execution failed: failed: {REDACTED_SECRET} end"


def test_runner_failure_redacts_a_secret_formed_by_flattening() -> None:
    secret = "joined canary"
    error = runner_execution_error(OSError("failed: joined\r\ncanary"), adapter="local")

    result, _controls = _runner_failure_result(
        error, effect=ToolEffect.NONE, redactor=SecretRedactor([secret])
    )

    assert "canary" not in result.content


@pytest.mark.parametrize(
    "secret",
    ["TOKEN\nPART2xyzabc", "it's\\secret-part", "tab\tsecret-part"],
)
def test_runner_failure_redacts_repr_escaped_secrets(secret: str) -> None:
    for error in (
        FileNotFoundError(2, "No such file", f"/run/{secret}"),
        OSError(2, "No such file", "/a", f"/b/{secret}"),
        KeyError(secret),
    ):
        result, _controls = _runner_failure_result(
            runner_execution_error(error, adapter="local"),
            effect=ToolEffect.NONE,
            redactor=SecretRedactor([secret]),
        )
        assert "secret-part" not in result.content and "PART2" not in result.content, (
            error,
            result.content,
        )


_ESCAPED_FORM_CASES = [
    ("TOKEN\nPART2xyzabcdef", lambda s: KeyError(repr(s))),
    ("tab\tsecret-partZZ", lambda s: KeyError(f"missing {s!r}")),
    ("crlf\r\nsecret-partZZ", lambda s: ValueError(repr(repr(s)))),
    ("back\\slash-secret-partZZ", lambda s: KeyError(repr(s))),
    ("c1\x85\x9fsecret-partZZ", lambda s: ValueError(repr(s.encode()))),
    ("lone\udc80surrogate-secret-partZZ", lambda s: ValueError(repr(s))),
    (
        "lone\udc80surrogate-secret-partZZ",
        lambda s: ValueError(repr(s.encode("utf-8", "surrogatepass"))),
    ),
    ("astral\U0001f600secret-partZZ", lambda s: ValueError(ascii(s))),
    ('quote"ctl\x01secret-partZZ', lambda s: ValueError(__import__("json").dumps(s))),
    ("snow☃\nsecret-partZZ", lambda s: ValueError(__import__("json").dumps(s, ensure_ascii=False))),
]


@pytest.mark.parametrize(("secret", "build"), _ESCAPED_FORM_CASES)
def test_runner_failure_redacts_nested_and_encoded_secret_forms(secret, build) -> None:
    result, _controls = _runner_failure_result(
        runner_execution_error(build(secret), adapter="local"),
        effect=ToolEffect.NONE,
        redactor=SecretRedactor([secret]),
    )

    assert "secret-part" not in result.content and "PART2" not in result.content, result.content
