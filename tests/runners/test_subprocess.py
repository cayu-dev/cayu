from __future__ import annotations

import asyncio
import io
import json
import subprocess
import sys
import threading
import time
from typing import Any, cast

import pytest
from pydantic import ValidationError
from tests.provider_traceback_assertions import is_cayu_source_filename

import cayu.runners._subprocess as subprocess_module
import cayu.runners.docker as docker_runner_module
import cayu.runners.local as local_runner_module
from cayu.runners import DockerRunner, ExecCommand, ExecResult, LocalRunner, Runner
from cayu.runners._secrets import DOCKER_ENV_FILE_MAX_LINE_BYTES
from cayu.runners._subprocess import (
    SubprocessCommand,
    copy_runner_env,
    remove_runner_env,
    run_subprocess,
    validate_output_limit,
    validate_stdin,
    validate_timeout,
)
from cayu.vaults import REDACTED_SECRET, SecretEnv, SecretRedactor, SecretRef, StaticVault

_ENV_NAME_CANARY = "runner-environment-name-canary"
_SECRET_NAME = "runner_secret_name"
_SECRET_VALUE = "runner-secret-value-canary"

_INVALID_ENV_NAMES = (
    pytest.param(None, id="not-string"),
    pytest.param("", id="empty"),
    pytest.param(" ", id="blank"),
    pytest.param(f" {_ENV_NAME_CANARY}", id="leading-whitespace"),
    pytest.param(f"{_ENV_NAME_CANARY} ", id="trailing-whitespace"),
    pytest.param(f"{_ENV_NAME_CANARY}\x00", id="nul"),
    pytest.param(f"{_ENV_NAME_CANARY}\ud800", id="surrogate"),
    pytest.param(f"{_ENV_NAME_CANARY}=value", id="equals"),
    pytest.param(f"{_ENV_NAME_CANARY}\nvalue", id="newline"),
    pytest.param(f"{_ENV_NAME_CANARY}\rvalue", id="carriage-return"),
)


def _assert_cayu_validation_traceback_does_not_retain(
    error: BaseException,
    secret: str,
) -> None:
    current = error.__traceback__
    while current is not None:
        frame = current.tb_frame
        if is_cayu_source_filename(frame.f_code.co_filename):
            assert all(secret not in repr(value) for value in frame.f_locals.values())
        current = current.tb_next


class _CountingVault(StaticVault):
    def __init__(self) -> None:
        super().__init__({_SECRET_NAME: _SECRET_VALUE})
        self.resolve_calls = 0

    async def resolve(self, ref, *, scope=None):
        self.resolve_calls += 1
        return await super().resolve(ref, scope=scope)


class _PortablePreflightRunner(Runner):
    async def exec(
        self,
        command: ExecCommand,
        **kwargs: Any,
    ) -> ExecResult:
        del command, kwargs
        return ExecResult()


def test_subprocess_command_accepts_exactly_one_command_shape() -> None:
    assert SubprocessCommand(argv=["python", "--version"]).argv == ["python", "--version"]
    assert SubprocessCommand(shell="echo ok").shell == "echo ok"

    with pytest.raises(ValueError, match="exactly one"):
        SubprocessCommand()

    with pytest.raises(ValueError, match="exactly one"):
        SubprocessCommand(argv=["echo"], shell="echo ok")


def test_run_subprocess_streams_binary_stdin_and_stdout_without_text_encoding() -> None:
    source = io.BytesIO(bytes(range(256)) * 32)
    target = io.BytesIO()

    result = asyncio.run(
        run_subprocess(
            SubprocessCommand(
                argv=[
                    sys.executable,
                    "-c",
                    "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read()[::-1])",
                ]
            ),
            stdin_stream=source,
            stdout_stream=target,
            stdout_limit_bytes=source.getbuffer().nbytes,
            output_limit_bytes=1024,
        )
    )

    assert result.exit_code == 0
    assert result.stdout == ""
    assert result.stdout_bytes == source.getbuffer().nbytes
    assert result.stdout_truncated is False
    assert target.getvalue() == (bytes(range(256)) * 32)[::-1]


def test_run_subprocess_bounds_binary_stdout_while_draining_child() -> None:
    target = io.BytesIO()

    result = asyncio.run(
        run_subprocess(
            SubprocessCommand(
                argv=[sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'x' * 65537)"]
            ),
            stdout_stream=target,
            stdout_limit_bytes=4096,
            output_limit_bytes=1024,
        )
    )

    assert result.exit_code == 0
    assert result.stdout_bytes == 65537
    assert result.stdout_truncated is True
    assert target.getvalue() == b"x" * 4096


def test_run_subprocess_rejects_paths_as_binary_streams(tmp_path) -> None:
    async def run() -> None:
        await run_subprocess(
            SubprocessCommand(argv=[sys.executable, "-c", "pass"]),
            stdin_stream=tmp_path / "host-secret",
        )

    with pytest.raises(TypeError, match="binary stdin"):
        asyncio.run(run())


def test_run_subprocess_settles_child_when_binary_stdin_reader_fails() -> None:
    class InvalidBinaryInput:
        def read(self, _size: int) -> str:
            return "not bytes"

    async def run() -> None:
        await asyncio.wait_for(
            run_subprocess(
                SubprocessCommand(
                    argv=[sys.executable, "-c", "import sys; sys.stdin.buffer.read()"]
                ),
                stdin_stream=InvalidBinaryInput(),  # type: ignore[arg-type]
            ),
            timeout=5,
        )

    with pytest.raises(TypeError, match="must return bytes"):
        asyncio.run(run())


def test_run_subprocess_rejects_empty_text_as_binary_stdin_eof() -> None:
    class InvalidBinaryInput:
        def read(self, _size: int) -> str:
            return ""

    with pytest.raises(TypeError, match="must return bytes"):
        asyncio.run(
            run_subprocess(
                SubprocessCommand(argv=[sys.executable, "-c", "pass"]),
                stdin_stream=InvalidBinaryInput(),  # type: ignore[arg-type]
            )
        )


def test_run_subprocess_completes_partial_binary_stdout_writes() -> None:
    class PartialBinaryOutput:
        def __init__(self) -> None:
            self.content = bytearray()

        def write(self, content: bytes) -> int:
            accepted = max(1, len(content) // 2)
            self.content.extend(content[:accepted])
            return accepted

    target = PartialBinaryOutput()
    result = asyncio.run(
        run_subprocess(
            SubprocessCommand(
                argv=[sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'x' * 257)"]
            ),
            stdout_stream=target,  # type: ignore[arg-type]
            stdout_limit_bytes=257,
        )
    )

    assert result.exit_code == 0
    assert result.stdout_bytes == 257
    assert target.content == b"x" * 257


def test_run_subprocess_settles_child_when_binary_stdout_writer_fails() -> None:
    class FailingBinaryOutput:
        def write(self, _content: bytes) -> int:
            raise OSError("test binary sink failed")

    async def run() -> None:
        await asyncio.wait_for(
            run_subprocess(
                SubprocessCommand(
                    argv=[
                        sys.executable,
                        "-c",
                        "import sys; sys.stdout.buffer.write(b'x' * (1 << 20))",
                    ]
                ),
                stdout_stream=FailingBinaryOutput(),  # type: ignore[arg-type]
            ),
            timeout=5,
        )

    with pytest.raises(OSError, match="test binary sink failed"):
        asyncio.run(run())


def test_run_subprocess_blocking_binary_input_keeps_event_loop_responsive_and_owned() -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingBinaryInput:
        def read(self, _size: int) -> bytes:
            started.set()
            if not release.wait(timeout=5):
                raise TimeoutError("test binary input was not released")
            return b""

    async def run() -> None:
        release_timer = threading.Timer(1, release.set)
        release_timer.start()
        started_at = time.monotonic()
        operation = asyncio.create_task(
            run_subprocess(
                SubprocessCommand(
                    argv=[sys.executable, "-c", "import sys; sys.stdin.buffer.read()"]
                ),
                stdin_stream=BlockingBinaryInput(),
            )
        )
        try:
            while not started.is_set():
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.05)
            assert time.monotonic() - started_at < 0.5

            operation.cancel("cancel blocked binary input")
            await asyncio.sleep(0.05)
            assert not operation.done()
            release.set()
            with pytest.raises(asyncio.CancelledError, match="cancel blocked binary input"):
                await operation
        finally:
            release.set()
            release_timer.cancel()
            release_timer.join(timeout=1)

    asyncio.run(run())


def test_run_subprocess_blocking_binary_output_keeps_event_loop_responsive_and_owned() -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingBinaryOutput:
        def write(self, content: bytes) -> int:
            started.set()
            if not release.wait(timeout=5):
                raise TimeoutError("test binary output was not released")
            return len(content)

    async def run() -> None:
        release_timer = threading.Timer(1, release.set)
        release_timer.start()
        started_at = time.monotonic()
        operation = asyncio.create_task(
            run_subprocess(
                SubprocessCommand(
                    argv=[sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'x')"]
                ),
                stdout_stream=BlockingBinaryOutput(),
            )
        )
        try:
            while not started.is_set():
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.05)
            assert time.monotonic() - started_at < 0.5

            operation.cancel("cancel blocked binary output")
            await asyncio.sleep(0.05)
            assert not operation.done()
            release.set()
            with pytest.raises(asyncio.CancelledError, match="cancel blocked binary output"):
                await operation
        finally:
            release.set()
            release_timer.cancel()
            release_timer.join(timeout=1)

    asyncio.run(run())


def test_local_runner_binary_stream_capability(tmp_path) -> None:
    source = io.BytesIO(b"\x00archive\xff")
    target = io.BytesIO()
    runner = LocalRunner(tmp_path)

    result = asyncio.run(
        runner.exec_stream(
            ExecCommand.process(
                sys.executable,
                "-c",
                "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())",
            ),
            stdin=source,
            stdout=target,
            stdout_limit_bytes=1024,
        )
    )

    assert result.exit_code == 0
    assert target.getvalue() == b"\x00archive\xff"


def test_local_runner_redacts_split_output_before_bounding(tmp_path) -> None:
    secret = "local-subprocess-boundary-secret"
    script = (
        "import os,sys,time; "
        "secret=os.environ['TOKEN']; "
        "sys.stdout.write('prefix:' + secret[:9]); sys.stdout.flush(); "
        "time.sleep(0.02); "
        "sys.stdout.write(secret[9:] + ':suffix'); sys.stdout.flush()"
    )

    result = asyncio.run(
        LocalRunner(tmp_path).exec_redacted(
            ExecCommand.process(sys.executable, "-c", script),
            redactor=SecretRedactor(secret),
            env={"TOKEN": secret},
            output_limit_bytes=128,
        )
    )

    assert result.stdout == f"prefix:{REDACTED_SECRET}:suffix"
    assert result.stdout_bytes == len(f"prefix:{secret}:suffix".encode())
    assert result.stdout_truncated is False


def test_subprocess_command_rejects_invalid_argv_and_shell() -> None:
    with pytest.raises(ValueError, match="empty"):
        SubprocessCommand(argv=[])

    with pytest.raises(ValueError, match="non-empty"):
        SubprocessCommand(argv=[" "])

    with pytest.raises(ValueError, match="non-empty"):
        SubprocessCommand(shell=" ")

    for invalid_text in ("invalid\x00command", "invalid\ud800command"):
        with pytest.raises(ValueError):
            SubprocessCommand(argv=[invalid_text])
        with pytest.raises(ValueError):
            SubprocessCommand(shell=invalid_text)
        with pytest.raises(ValueError):
            ExecCommand.process(invalid_text)
        with pytest.raises(ValueError):
            ExecCommand.bash(invalid_text)


def test_exec_command_validation_hides_secret_bearing_sibling_arguments() -> None:
    secret = "command-validation-secret-canary-ABCDEFGHIJKLMNOP"

    with pytest.raises(ValueError) as raised:
        ExecCommand.process(
            "curl",
            "-H",
            f"Authorization: Bearer {secret}",
            "invalid\x00argument",
        )

    rendered = f"{raised.value!s} {raised.value!r}"
    assert secret not in rendered
    assert "Authorization: Bearer" not in rendered
    assert type(raised.value) is ValidationError
    assert secret not in repr(raised.value.errors())
    assert secret not in raised.value.json()
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    _assert_cayu_validation_traceback_does_not_retain(raised.value, secret)


@pytest.mark.parametrize(
    "validation_entrance",
    ("model_validate", "model_validate_json", "model_validate_strings"),
)
def test_exec_command_model_validation_drops_rejected_input(
    validation_entrance: str,
) -> None:
    secret = "command-model-validation-secret-canary-ABCDEFGHIJKLMNOP"
    payload = {
        "kind": "process",
        "argv": [
            "curl",
            "-H",
            f"Authorization: Bearer {secret}",
            "invalid\x00argument",
        ],
    }

    with pytest.raises(ValueError) as raised:
        if validation_entrance == "model_validate":
            ExecCommand.model_validate(payload)
        elif validation_entrance == "model_validate_json":
            ExecCommand.model_validate_json(json.dumps(payload))
        else:
            ExecCommand.model_validate_strings(payload)

    rendered = f"{raised.value!s} {raised.value!r}"
    assert secret not in rendered
    assert "Authorization: Bearer" not in rendered
    assert type(raised.value) is ValidationError
    assert secret not in repr(raised.value.errors())
    assert secret not in raised.value.json()
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    _assert_cayu_validation_traceback_does_not_retain(raised.value, secret)


def test_exec_command_validation_preserves_safe_error_structure() -> None:
    with pytest.raises(ValidationError) as raised:
        ExecCommand.model_validate(
            {
                "kind": "unknown",
                "argv": "not-a-list",
                "unexpected": "value",
            }
        )

    errors = raised.value.errors()
    assert len(errors) == 3
    assert {(error["type"], error["loc"]) for error in errors} == {
        ("literal_error", ("kind",)),
        ("list_type", ("argv",)),
        ("extra_forbidden", ("invalid_input",)),
    }
    assert all(error.get("input") is None for error in errors)


@pytest.mark.parametrize("runner_kind", ("local", "docker"))
@pytest.mark.parametrize(
    "public_method",
    ("preflight_exec", "exec", "exec_redacted", "exec_system"),
)
@pytest.mark.parametrize("invalid_input", ("command", "environment"))
def test_direct_runner_preflight_drops_secret_bearing_command_traceback(
    runner_kind: str,
    public_method: str,
    invalid_input: str,
    tmp_path,
) -> None:
    secret = "direct-runner-command-secret-canary-ABCDEFGHIJKLMNOP"
    if runner_kind == "local":
        runner = LocalRunner(tmp_path)
    else:
        runner = DockerRunner("validation-probe", docker_path="/unreachable/docker")
    command = ExecCommand.process(
        "curl",
        "-H",
        f"Authorization: Bearer {secret}",
        "valid",
    )
    assert command.argv is not None
    kwargs: dict[str, Any] = {}
    if invalid_input == "command":
        command.argv[-1] = "invalid\x00argument"
    else:
        kwargs["env"] = {"VALUE": "invalid\x00value"}

    with pytest.raises(ValueError) as raised:
        if public_method == "preflight_exec":
            runner.preflight_exec(command, **kwargs)
        elif public_method == "exec":
            asyncio.run(runner.exec(command, **kwargs))
        elif public_method == "exec_redacted":
            asyncio.run(
                runner.exec_redacted(
                    command,
                    redactor=SecretRedactor(),
                    **kwargs,
                )
            )
        else:
            asyncio.run(runner.exec_system(command, **kwargs))

    assert secret not in str(raised.value)
    if isinstance(raised.value, ValidationError):
        assert secret not in repr(raised.value.errors())
    _assert_cayu_validation_traceback_does_not_retain(raised.value, secret)


@pytest.mark.parametrize("runner_kind", ("local", "docker"))
@pytest.mark.parametrize(
    "public_method",
    ("preflight_exec", "exec", "exec_redacted", "exec_system"),
)
@pytest.mark.parametrize(
    "secret_input",
    ("environment", "stdin", "cwd", "env_remove", "partial_environment"),
)
def test_direct_runner_preflight_drops_secret_bearing_sibling_inputs(
    runner_kind: str,
    public_method: str,
    secret_input: str,
    tmp_path,
) -> None:
    secret = "DIRECT_RUNNER_INPUT_SECRET_CANARY_ABCDEFGHIJKLMNOP"
    if runner_kind == "local":
        runner = LocalRunner(tmp_path)
    else:
        runner = DockerRunner("validation-probe", docker_path="/unreachable/docker")
    command = ExecCommand.process("valid")
    kwargs: dict[str, Any]
    if secret_input == "partial_environment":
        kwargs = {
            "env": {
                "TOKEN": secret,
                "INVALID": "invalid\x00value",
            }
        }
    elif secret_input in {"environment", "stdin", "env_remove"}:
        assert command.argv is not None
        command.argv[-1] = "invalid\x00argument"
        if secret_input == "environment":
            kwargs = {"env": {"TOKEN": secret}}
        elif secret_input == "stdin":
            kwargs = {"stdin": secret}
        else:
            kwargs = {"env_remove": (secret,)}
    else:
        assert command.argv is not None
        command.argv[-1] = "invalid\x00argument"
        (tmp_path / secret).mkdir()
        kwargs = {"cwd": secret}

    with pytest.raises(ValueError) as raised:
        if public_method == "preflight_exec":
            runner.preflight_exec(command, **kwargs)
        elif public_method == "exec":
            asyncio.run(runner.exec(command, **kwargs))
        elif public_method == "exec_redacted":
            asyncio.run(
                runner.exec_redacted(
                    command,
                    redactor=SecretRedactor(),
                    **kwargs,
                )
            )
        else:
            asyncio.run(runner.exec_system(command, **kwargs))

    assert secret not in str(raised.value)
    _assert_cayu_validation_traceback_does_not_retain(raised.value, secret)


def test_base_runner_direct_preflight_drops_secret_bearing_sibling_inputs() -> None:
    secret = "BASE_PREFLIGHT_SECRET_CANARY_ABCDEFGHIJKLMNOP"
    command = ExecCommand.process("valid", f"Authorization: Bearer {secret}")
    assert command.argv is not None
    command.argv[0] = "invalid\x00argument"

    with pytest.raises(ValueError) as raised:
        _PortablePreflightRunner().preflight_exec(
            command,
            env={"TOKEN": secret},
        )

    assert secret not in str(raised.value)
    _assert_cayu_validation_traceback_does_not_retain(raised.value, secret)


def test_runner_env_copy_can_inherit_or_isolate_parent_env(monkeypatch) -> None:
    monkeypatch.setenv("CAYU_PARENT_ENV", "visible")

    inherited = copy_runner_env({"CHILD": "set"}, inherit_env=True)
    assert inherited["CAYU_PARENT_ENV"] == "visible"
    assert inherited["CHILD"] == "set"

    isolated = copy_runner_env({"CHILD": "set"}, inherit_env=False)
    assert "CAYU_PARENT_ENV" not in isolated
    assert isolated == {"CHILD": "set"}

    multiline = "first line\nsecond line\rthird line"
    assert copy_runner_env({"MULTILINE": multiline}, inherit_env=False) == {"MULTILINE": multiline}


@pytest.mark.parametrize("public_method", ("preflight_exec", "exec", "exec_redacted"))
def test_local_runner_rejects_invalid_inherited_environment_before_secret_resolution(
    public_method: str,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(" PADDED_INHERITED_NAME ", "invalid")
    vault = _CountingVault()
    runner = LocalRunner(
        tmp_path,
        inherit_env=True,
        secret_env={"API_TOKEN": SecretRef(name=_SECRET_NAME)},
        secret_resolver=vault,
    )

    with pytest.raises(ValueError, match="env key"):
        if public_method == "preflight_exec":
            runner.preflight_exec(ExecCommand.process("true"))
        elif public_method == "exec":
            asyncio.run(runner.exec(ExecCommand.process("true")))
        else:
            asyncio.run(
                runner.exec_redacted(
                    ExecCommand.process("true"),
                    redactor=SecretRedactor(),
                )
            )

    assert vault.resolve_calls == 0


def test_local_runner_cwd_preserves_valid_surrounding_spaces(tmp_path) -> None:
    spaced = tmp_path / " spaced "
    spaced.mkdir()

    assert LocalRunner(tmp_path).resolve_cwd(" spaced ") == str(spaced.resolve())


def test_local_runner_root_preserves_valid_surrounding_spaces(tmp_path) -> None:
    spaced_root = tmp_path / " workspace "
    spaced_root.mkdir()
    runner = LocalRunner(spaced_root)

    assert runner.resolve_cwd() == str(spaced_root.resolve())
    result = asyncio.run(
        runner.exec(
            ExecCommand.process(
                sys.executable,
                "-c",
                "from pathlib import Path; print(Path.cwd().name)",
            )
        )
    )

    assert result.exit_code == 0
    assert result.stdout == " workspace \n"


@pytest.mark.parametrize("invalid_suffix", ("bad\x00root", "bad\ud800root"))
def test_local_runner_rejects_mutated_nonportable_root_before_secret_resolution(
    invalid_suffix: str,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vault = _CountingVault()
    runner = LocalRunner(
        tmp_path,
        secret_env={"API_TOKEN": SecretRef(name=_SECRET_NAME)},
        secret_resolver=vault,
    )
    runner.root = tmp_path / invalid_suffix
    spawn_calls = 0

    async def unexpected_spawn(*args: Any, **kwargs: Any) -> ExecResult:
        nonlocal spawn_calls
        del args, kwargs
        spawn_calls += 1
        return ExecResult()

    monkeypatch.setattr(local_runner_module, "run_subprocess", unexpected_spawn)

    with pytest.raises(ValueError, match="root"):
        asyncio.run(runner.exec(ExecCommand.process("true")))

    assert vault.resolve_calls == 0
    assert spawn_calls == 0


def test_run_subprocess_does_not_inherit_parent_env_by_default(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CAYU_PARENT_ENV", "hidden")

    isolated = asyncio.run(
        run_subprocess(
            SubprocessCommand(
                argv=[
                    sys.executable,
                    "-c",
                    "import os; print(os.environ.get('CAYU_PARENT_ENV', ''))",
                ]
            ),
            cwd=tmp_path,
        )
    )
    inherited = asyncio.run(
        run_subprocess(
            SubprocessCommand(
                argv=[
                    sys.executable,
                    "-c",
                    "import os; print(os.environ.get('CAYU_PARENT_ENV', ''))",
                ]
            ),
            cwd=tmp_path,
            env=copy_runner_env(None, inherit_env=True),
        )
    )

    assert isolated.stdout == "\n"
    assert inherited.stdout == "hidden\n"


def test_runner_env_copy_rejects_invalid_env() -> None:
    with pytest.raises(TypeError, match="dictionary"):
        copy_runner_env([], inherit_env=False)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", _INVALID_ENV_NAMES)
def test_runner_env_additions_and_removals_share_exact_name_validation(name) -> None:
    with pytest.raises(ValueError) as addition_error:
        copy_runner_env({name: "value"}, inherit_env=False)
    with pytest.raises(ValueError) as removal_error:
        remove_runner_env({"VALID": "value"}, (name,))

    rendered = (
        f"{addition_error.value!s} {addition_error.value!r} "
        f"{removal_error.value!s} {removal_error.value!r}"
    )
    assert _ENV_NAME_CANARY not in rendered


@pytest.mark.parametrize("invalid_remove", (None, "PATH", ["PATH"]))
def test_runner_env_removals_reject_invalid_container_types(invalid_remove: Any) -> None:
    with pytest.raises(TypeError, match="tuple"):
        remove_runner_env(
            {"PATH": "value"},
            cast("tuple[str, ...]", invalid_remove),
        )


@pytest.mark.parametrize("name", ("VALID_NAME", "ÜNICODE_NAME", "INNER SPACE"))
def test_runner_env_additions_and_removals_preserve_valid_names(name: str) -> None:
    assert copy_runner_env({name: "value"}, inherit_env=False) == {name: "value"}
    assert remove_runner_env({name: "value", "KEEP": "yes"}, (name, name)) == {"KEEP": "yes"}


@pytest.mark.parametrize("runner_kind", ("local", "docker"))
@pytest.mark.parametrize("public_method", ("exec", "exec_redacted"))
@pytest.mark.parametrize("invalid_argument", ("env", "env_remove"))
def test_runner_rejects_invalid_environment_before_resolving_secrets(
    runner_kind: str,
    public_method: str,
    invalid_argument: str,
    tmp_path,
) -> None:
    vault = _CountingVault()
    secret_env = [SecretEnv(name="API_TOKEN", ref=SecretRef(name=_SECRET_NAME))]
    runner = (
        LocalRunner(tmp_path, secret_env=secret_env, secret_resolver=vault)
        if runner_kind == "local"
        else DockerRunner(
            "validation-probe",
            docker_path="/unreachable/docker",
            secret_env=secret_env,
            secret_resolver=vault,
        )
    )
    invalid_name = f"{_ENV_NAME_CANARY}\x00"

    async def invoke() -> None:
        command = ExecCommand.process(
            sys.executable,
            "-c",
            "raise SystemExit(99)",
        )
        if public_method == "exec":
            if invalid_argument == "env":
                await runner.exec(command, env={invalid_name: "value"})
            else:
                await runner.exec(command, env_remove=(invalid_name,))
        elif invalid_argument == "env":
            await runner.exec_redacted(
                command,
                redactor=SecretRedactor(),
                env={invalid_name: "value"},
            )
        else:
            await runner.exec_redacted(
                command,
                redactor=SecretRedactor(),
                env_remove=(invalid_name,),
            )

    with pytest.raises(ValueError) as raised:
        asyncio.run(invoke())

    assert vault.resolve_calls == 0
    rendered = f"{raised.value!s} {raised.value!r}"
    assert _ENV_NAME_CANARY not in rendered
    assert _SECRET_NAME not in rendered
    assert _SECRET_VALUE not in rendered


@pytest.mark.parametrize("runner_kind", ("local", "docker"))
@pytest.mark.parametrize("public_method", ("exec", "exec_redacted"))
def test_runner_rejects_secret_env_collision_before_resolving_secrets(
    runner_kind: str,
    public_method: str,
    tmp_path,
) -> None:
    vault = _CountingVault()
    runner = (
        LocalRunner(
            tmp_path,
            secret_env={"API_TOKEN": SecretRef(name=_SECRET_NAME)},
            secret_resolver=vault,
        )
        if runner_kind == "local"
        else DockerRunner(
            "validation-probe",
            docker_path="/unreachable/docker",
            secret_env={"API_TOKEN": SecretRef(name=_SECRET_NAME)},
            secret_resolver=vault,
        )
    )
    command = ExecCommand.process(sys.executable, "-c", "raise SystemExit(99)")

    with pytest.raises(ValueError, match="collides with declared secret_env"):
        if public_method == "exec":
            asyncio.run(runner.exec(command, env={"API_TOKEN": "override"}))
        else:
            asyncio.run(
                runner.exec_redacted(
                    command,
                    redactor=SecretRedactor(),
                    env={"API_TOKEN": "override"},
                )
            )

    assert vault.resolve_calls == 0


def test_local_runner_rejects_safe_host_env_collision_before_resolving_secrets(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", "/usr/bin")
    vault = _CountingVault()
    runner = LocalRunner(
        tmp_path,
        secret_env={"PATH": SecretRef(name=_SECRET_NAME)},
        secret_resolver=vault,
    )

    with pytest.raises(ValueError, match="collides with declared secret_env"):
        asyncio.run(runner.exec(ExecCommand.process("true")))

    assert vault.resolve_calls == 0


@pytest.mark.parametrize("public_method", ("preflight_exec", "exec", "exec_redacted"))
def test_local_runner_rejects_invalid_safe_host_environment_before_resolving_secrets(
    public_method: str,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", "invalid\udcffvalue")
    vault = _CountingVault()
    runner = LocalRunner(
        tmp_path,
        secret_env={"API_TOKEN": SecretRef(name=_SECRET_NAME)},
        secret_resolver=vault,
    )

    with pytest.raises(ValueError, match="env value"):
        if public_method == "preflight_exec":
            runner.preflight_exec(ExecCommand.process("true"))
        elif public_method == "exec":
            asyncio.run(runner.exec(ExecCommand.process("true")))
        else:
            asyncio.run(
                runner.exec_redacted(
                    ExecCommand.process("true"),
                    redactor=SecretRedactor(),
                )
            )

    assert vault.resolve_calls == 0


@pytest.mark.parametrize("runner_kind", ("local", "docker"))
@pytest.mark.parametrize("public_method", ("exec", "exec_redacted"))
@pytest.mark.parametrize(
    ("invalid_kwargs", "error_type"),
    (
        ({"cwd": "/outside-runner-root"}, ValueError),
        ({"cwd": "invalid\x00cwd"}, ValueError),
        ({"cwd": "invalid\ud800cwd"}, ValueError),
        ({"timeout_s": 0}, ValueError),
        ({"stdin": b"invalid"}, TypeError),
        ({"stdin": "\ud800"}, ValueError),
        ({"env": {"VALUE": "invalid\x00value"}}, ValueError),
        ({"env": {"VALUE": "invalid\ud800value"}}, ValueError),
        ({"output_limit_bytes": 0}, ValueError),
    ),
)
def test_runner_rejects_invalid_command_inputs_before_resolving_secrets(
    runner_kind: str,
    public_method: str,
    invalid_kwargs: dict[str, Any],
    error_type: type[Exception],
    tmp_path,
) -> None:
    vault = _CountingVault()
    secret_env = [SecretEnv(name="API_TOKEN", ref=SecretRef(name=_SECRET_NAME))]
    runner = (
        LocalRunner(tmp_path, secret_env=secret_env, secret_resolver=vault)
        if runner_kind == "local"
        else DockerRunner(
            "validation-probe",
            docker_path="/unreachable/docker",
            secret_env=secret_env,
            secret_resolver=vault,
        )
    )
    command = ExecCommand.process(sys.executable, "-c", "raise SystemExit(99)")

    with pytest.raises(error_type):
        if public_method == "exec":
            asyncio.run(runner.exec(command, **invalid_kwargs))
        else:
            asyncio.run(
                runner.exec_redacted(
                    command,
                    redactor=SecretRedactor(),
                    **invalid_kwargs,
                )
            )

    assert vault.resolve_calls == 0


@pytest.mark.parametrize("public_method", ("exec", "exec_redacted"))
@pytest.mark.parametrize(
    "invalid_environment",
    (
        {"VALUE": "invalid\nvalue"},
        {"VALUE": "invalid\rvalue"},
        {"INNER SPACE": "value"},
        {"INNER\tTAB": "value"},
        {"#COMMENT": "value"},
        {"\ufeffBOM": "value"},
        {"VALUE": "x" * (DOCKER_ENV_FILE_MAX_LINE_BYTES - len("VALUE=") + 1)},
    ),
)
def test_docker_rejects_unrepresentable_env_file_before_resolving_secrets(
    public_method: str,
    invalid_environment: dict[str, str],
    tmp_path,
) -> None:
    vault = _CountingVault()
    runner = DockerRunner(
        "validation-probe",
        docker_path="/unreachable/docker",
        secret_env={"API_TOKEN": SecretRef(name=_SECRET_NAME)},
        secret_resolver=vault,
    )
    command = ExecCommand.process(sys.executable, "--version")

    with pytest.raises(ValueError) as raised:
        if public_method == "exec":
            asyncio.run(runner.exec(command, env=invalid_environment))
        else:
            asyncio.run(
                runner.exec_redacted(
                    command,
                    redactor=SecretRedactor(),
                    env=invalid_environment,
                )
            )

    assert vault.resolve_calls == 0
    rendered = f"{raised.value!s} {raised.value!r}"
    assert _SECRET_NAME not in rendered
    assert _SECRET_VALUE not in rendered


@pytest.mark.parametrize("public_method", ("exec", "exec_redacted"))
@pytest.mark.parametrize(
    "invalid_overlay",
    (
        {"VALUE": "invalid\x00value"},
        {"INNER SPACE": "value"},
    ),
)
def test_docker_revalidates_stored_overlay_before_resolving_secrets(
    public_method: str,
    invalid_overlay: dict[str, str],
) -> None:
    vault = _CountingVault()
    runner = DockerRunner(
        "validation-probe",
        docker_path="/unreachable/docker",
        secret_env={"API_TOKEN": SecretRef(name=_SECRET_NAME)},
        secret_resolver=vault,
    )
    runner.env_overlay = invalid_overlay
    command = ExecCommand.process(sys.executable, "--version")

    with pytest.raises(ValueError):
        if public_method == "exec":
            asyncio.run(runner.exec(command))
        else:
            asyncio.run(
                runner.exec_redacted(
                    command,
                    redactor=SecretRedactor(),
                )
            )

    assert vault.resolve_calls == 0


def test_docker_owns_overlay_snapshot_across_secret_resolution(monkeypatch) -> None:
    captured_environment: dict[str, str] = {}

    class _BlockingVault(_CountingVault):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def resolve(self, ref, *, scope=None):
            self.resolve_calls += 1
            self.started.set()
            await self.release.wait()
            return await StaticVault.resolve(self, ref, scope=scope)

    async def fake_run_subprocess(command, **kwargs):
        del kwargs
        assert command.argv is not None
        env_path = command.argv[command.argv.index("--env-file") + 1]
        with open(env_path, encoding="utf-8") as handle:
            for line in handle:
                key, value = line.rstrip("\n").split("=", 1)
                captured_environment[key] = value
        return ExecResult()

    monkeypatch.setattr(docker_runner_module, "run_subprocess", fake_run_subprocess)

    async def run() -> None:
        vault = _BlockingVault()
        runner = DockerRunner(
            "validation-probe",
            docker_path="/unreachable/docker",
            secret_env={"API_TOKEN": SecretRef(name=_SECRET_NAME)},
            secret_resolver=vault,
            env_overlay={"OVERLAY": "owned"},
        )
        operation = asyncio.create_task(runner.exec(ExecCommand.process("true")))
        await vault.started.wait()
        runner.env_overlay.clear()
        runner.env_overlay["OVERLAY"] = "mutated"
        runner.env_overlay["INVALID"] = "invalid\x00value"
        vault.release.set()
        await operation

        assert vault.resolve_calls == 1

    asyncio.run(run())

    assert captured_environment["OVERLAY"] == "owned"
    assert "INVALID" not in captured_environment


@pytest.mark.parametrize("runner_kind", ("local", "docker"))
@pytest.mark.parametrize("public_method", ("exec", "exec_redacted"))
@pytest.mark.parametrize("command_kind", ("process", "shell"))
@pytest.mark.parametrize("invalid_text", ("invalid\x00command", "invalid\ud800command"))
def test_runner_revalidates_mutated_command_text_before_resolving_secrets(
    runner_kind: str,
    public_method: str,
    command_kind: str,
    invalid_text: str,
    tmp_path,
) -> None:
    vault = _CountingVault()
    runner = (
        LocalRunner(
            tmp_path,
            secret_env={"API_TOKEN": SecretRef(name=_SECRET_NAME)},
            secret_resolver=vault,
        )
        if runner_kind == "local"
        else DockerRunner(
            "validation-probe",
            docker_path="/unreachable/docker",
            secret_env={"API_TOKEN": SecretRef(name=_SECRET_NAME)},
            secret_resolver=vault,
        )
    )
    command = (
        ExecCommand.process(sys.executable, "--version")
        if command_kind == "process"
        else ExecCommand.bash("true")
    )
    if command_kind == "process":
        assert command.argv is not None
        command.argv[-1] = invalid_text
    else:
        command.shell = invalid_text

    with pytest.raises(ValueError):
        if public_method == "exec":
            asyncio.run(runner.exec(command))
        else:
            asyncio.run(
                runner.exec_redacted(
                    command,
                    redactor=SecretRedactor(),
                )
            )

    assert vault.resolve_calls == 0


@pytest.mark.parametrize("runner_kind", ("local", "docker"))
@pytest.mark.parametrize(
    "invalid_name",
    (f"{_ENV_NAME_CANARY}=value", f"{_ENV_NAME_CANARY}\nvalue"),
)
def test_runner_rejects_invalid_declared_secret_environment_name(
    runner_kind: str,
    invalid_name: str,
    tmp_path,
) -> None:
    vault = _CountingVault()
    secret_env = {invalid_name: SecretRef(name=_SECRET_NAME)}

    with pytest.raises(ValueError) as raised:
        if runner_kind == "local":
            LocalRunner(tmp_path, secret_env=secret_env, secret_resolver=vault)
        else:
            DockerRunner(
                "validation-probe",
                docker_path="/unreachable/docker",
                secret_env=secret_env,
                secret_resolver=vault,
            )

    assert vault.resolve_calls == 0
    rendered = f"{raised.value!s} {raised.value!r}"
    assert _ENV_NAME_CANARY not in rendered
    assert _SECRET_NAME not in rendered
    assert _SECRET_VALUE not in rendered


@pytest.mark.parametrize("runner_kind", ("local", "docker"))
@pytest.mark.parametrize("public_method", ("exec", "exec_redacted"))
def test_runner_revalidates_mutated_stored_secret_environment_before_resolution(
    runner_kind: str,
    public_method: str,
    tmp_path,
) -> None:
    vault = _CountingVault()
    runner = (
        LocalRunner(
            tmp_path,
            secret_env={"API_TOKEN": SecretRef(name=_SECRET_NAME)},
            secret_resolver=vault,
        )
        if runner_kind == "local"
        else DockerRunner(
            "validation-probe",
            docker_path="/unreachable/docker",
            secret_env={"API_TOKEN": SecretRef(name=_SECRET_NAME)},
            secret_resolver=vault,
        )
    )
    invalid_name = f"{_ENV_NAME_CANARY}=value"
    runner.secret_env.clear()
    runner.secret_env[invalid_name] = SecretRef(name=_SECRET_NAME)
    command = ExecCommand.process(sys.executable, "-c", "raise SystemExit(99)")

    with pytest.raises(ValueError) as raised:
        if public_method == "exec":
            asyncio.run(runner.exec(command))
        else:
            asyncio.run(
                runner.exec_redacted(
                    command,
                    redactor=SecretRedactor(),
                )
            )

    assert vault.resolve_calls == 0
    rendered = f"{raised.value!s} {raised.value!r}"
    assert _ENV_NAME_CANARY not in rendered
    assert _SECRET_NAME not in rendered
    assert _SECRET_VALUE not in rendered


@pytest.mark.parametrize("public_method", ("exec", "exec_redacted"))
@pytest.mark.parametrize("mutated_after_construction", (False, True))
@pytest.mark.parametrize(
    "invalid_name",
    (f"#{_ENV_NAME_CANARY}", f"{_ENV_NAME_CANARY} SPACE"),
)
def test_docker_rejects_unrepresentable_declared_secret_name_before_resolution(
    public_method: str,
    mutated_after_construction: bool,
    invalid_name: str,
) -> None:
    vault = _CountingVault()
    runner = DockerRunner(
        "validation-probe",
        docker_path="/unreachable/docker",
        secret_env=(
            {"API_TOKEN": SecretRef(name=_SECRET_NAME)}
            if mutated_after_construction
            else {invalid_name: SecretRef(name=_SECRET_NAME)}
        ),
        secret_resolver=vault,
    )
    if mutated_after_construction:
        runner.secret_env.clear()
        runner.secret_env[invalid_name] = SecretRef(name=_SECRET_NAME)

    with pytest.raises(ValueError) as raised:
        if public_method == "exec":
            asyncio.run(runner.exec(ExecCommand.process("true")))
        else:
            asyncio.run(
                runner.exec_redacted(
                    ExecCommand.process("true"),
                    redactor=SecretRedactor(),
                )
            )

    assert vault.resolve_calls == 0
    rendered = f"{raised.value!s} {raised.value!r}"
    assert _ENV_NAME_CANARY not in rendered
    assert _SECRET_NAME not in rendered
    assert _SECRET_VALUE not in rendered


@pytest.mark.parametrize("runner_kind", ("local", "docker"))
def test_runner_stored_secret_mutation_preserves_raw_secret_opt_out(
    runner_kind: str,
    tmp_path,
) -> None:
    vault = _CountingVault()
    runner = (
        LocalRunner(
            tmp_path,
            secret_resolver=vault,
            allow_raw_secret_env=False,
        )
        if runner_kind == "local"
        else DockerRunner(
            "validation-probe",
            docker_path="/unreachable/docker",
            secret_resolver=vault,
            allow_raw_secret_env=False,
        )
    )
    runner.secret_env["API_TOKEN"] = SecretRef(name=_SECRET_NAME)

    with pytest.raises(ValueError, match="allow_raw_secret_env"):
        asyncio.run(runner.exec(ExecCommand.process(sys.executable, "--version")))

    assert vault.resolve_calls == 0


@pytest.mark.parametrize("runner_kind", ("local", "docker"))
@pytest.mark.parametrize(
    "invalid_name",
    (f"{_ENV_NAME_CANARY}=value", f"{_ENV_NAME_CANARY}\nvalue"),
)
def test_runner_rejects_duplicate_invalid_secret_environment_name_safely(
    runner_kind: str,
    invalid_name: str,
    tmp_path,
) -> None:
    vault = _CountingVault()
    secret_env = [
        SecretEnv(name=invalid_name, ref=SecretRef(name=_SECRET_NAME)),
        SecretEnv(name=invalid_name, ref=SecretRef(name=_SECRET_NAME)),
    ]

    with pytest.raises(ValueError) as raised:
        if runner_kind == "local":
            LocalRunner(tmp_path, secret_env=secret_env, secret_resolver=vault)
        else:
            DockerRunner(
                "validation-probe",
                docker_path="/unreachable/docker",
                secret_env=secret_env,
                secret_resolver=vault,
            )

    assert vault.resolve_calls == 0
    rendered = f"{raised.value!s} {raised.value!r}"
    assert _ENV_NAME_CANARY not in rendered
    assert _SECRET_NAME not in rendered
    assert _SECRET_VALUE not in rendered


@pytest.mark.parametrize("runner_kind", ("local", "docker"))
@pytest.mark.parametrize("public_method", ("exec", "exec_redacted"))
@pytest.mark.parametrize("invalid_remove", (None, "PATH", ["PATH"]))
def test_runner_rejects_invalid_env_remove_container_before_resolving_secrets(
    runner_kind: str,
    public_method: str,
    invalid_remove: Any,
    tmp_path,
) -> None:
    vault = _CountingVault()
    secret_env = [SecretEnv(name="API_TOKEN", ref=SecretRef(name=_SECRET_NAME))]
    runner = (
        LocalRunner(tmp_path, secret_env=secret_env, secret_resolver=vault)
        if runner_kind == "local"
        else DockerRunner(
            "validation-probe",
            docker_path="/unreachable/docker",
            secret_env=secret_env,
            secret_resolver=vault,
        )
    )
    env_remove = cast("tuple[str, ...]", invalid_remove)

    with pytest.raises(TypeError, match="tuple"):
        if public_method == "exec":
            asyncio.run(
                runner.exec(
                    ExecCommand.process(sys.executable, "-c", "raise SystemExit(99)"),
                    env_remove=env_remove,
                )
            )
        else:
            asyncio.run(
                runner.exec_redacted(
                    ExecCommand.process(sys.executable, "-c", "raise SystemExit(99)"),
                    redactor=SecretRedactor(),
                    env_remove=env_remove,
                )
            )

    assert vault.resolve_calls == 0


def test_local_runner_applies_valid_removal_after_secret_injection(tmp_path) -> None:
    vault = _CountingVault()
    runner = LocalRunner(
        tmp_path,
        secret_env={"API_TOKEN": SecretRef(name=_SECRET_NAME)},
        secret_resolver=vault,
    )

    result = asyncio.run(
        runner.exec(
            ExecCommand.process(
                sys.executable,
                "-c",
                "import os; print(os.environ.get('API_TOKEN', 'absent'))",
            ),
            env_remove=("API_TOKEN",),
        )
    )

    assert vault.resolve_calls == 1
    assert result.exit_code == 0
    assert result.stdout == "absent\n"


def test_local_runner_cancellation_does_not_retain_raw_environment(
    tmp_path,
) -> None:
    secret = "local-runner-traceback-environment-secret-canary"

    async def run() -> asyncio.CancelledError:
        task = asyncio.create_task(
            LocalRunner(tmp_path).exec(
                ExecCommand.process(
                    sys.executable,
                    "-c",
                    "import time; time.sleep(30)",
                ),
                env={"WORKLOAD_TOKEN": secret},
            )
        )
        await asyncio.sleep(0.1)
        task.cancel("operator cancelled")
        assert task.cancelling() == 1
        with pytest.raises(asyncio.CancelledError) as excinfo:
            await task
        assert task.cancelled()
        return excinfo.value

    error = asyncio.run(run())
    traceback = error.__traceback__
    while traceback is not None:
        if is_cayu_source_filename(traceback.tb_frame.f_code.co_filename):
            for name, value in traceback.tb_frame.f_locals.items():
                assert secret not in repr(value), (
                    traceback.tb_frame.f_code.co_filename,
                    traceback.tb_frame.f_code.co_name,
                    name,
                )
        traceback = traceback.tb_next

    with pytest.raises(ValueError, match="env key"):
        copy_runner_env({" ": "bad"}, inherit_env=False)

    with pytest.raises(ValueError, match="values"):
        copy_runner_env({"KEY": 1}, inherit_env=False)  # type: ignore[dict-item]


def test_runner_validation_helpers_reject_invalid_values() -> None:
    with pytest.raises(TypeError, match="timeout_s"):
        validate_timeout("1")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="greater than zero"):
        validate_timeout(0)

    with pytest.raises(TypeError, match="stdin"):
        validate_stdin(b"bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="surrogate"):
        validate_stdin("\ud800")

    with pytest.raises(TypeError, match="output_limit_bytes"):
        validate_output_limit("1")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="greater than zero"):
        validate_output_limit(0)


def test_run_subprocess_rejects_unencodable_stdin_before_spawning(monkeypatch) -> None:
    spawn_calls = 0

    def unexpected_spawn(*args, **kwargs):
        nonlocal spawn_calls
        del args, kwargs
        spawn_calls += 1
        raise AssertionError("invalid stdin must not create a subprocess")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", unexpected_spawn)

    with pytest.raises(ValueError, match="surrogate"):
        asyncio.run(
            run_subprocess(
                SubprocessCommand(argv=[sys.executable, "-c", "pass"]),
                stdin="\ud800",
            )
        )

    assert spawn_calls == 0


@pytest.mark.parametrize("command_kind", ("process", "shell"))
@pytest.mark.parametrize("invalid_text", ("invalid\x00command", "invalid\ud800command"))
def test_run_subprocess_revalidates_mutated_command_before_spawning(
    command_kind: str,
    invalid_text: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawn_calls = 0

    def unexpected_spawn(*args, **kwargs):
        nonlocal spawn_calls
        del args, kwargs
        spawn_calls += 1
        raise AssertionError("invalid command must not create a subprocess")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", unexpected_spawn)
    monkeypatch.setattr(asyncio, "create_subprocess_shell", unexpected_spawn)
    command = (
        SubprocessCommand(argv=[sys.executable, "--version"])
        if command_kind == "process"
        else SubprocessCommand(shell="true")
    )
    if command_kind == "process":
        assert command.argv is not None
        command.argv[-1] = invalid_text
    else:
        command.shell = invalid_text

    with pytest.raises(ValueError):
        asyncio.run(run_subprocess(command))

    assert spawn_calls == 0


def test_local_runner_uses_windows_environment_name_identity_before_secret_lookup(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        local_runner_module,
        "_local_environment_names_case_sensitive",
        lambda: False,
    )
    vault = _CountingVault()
    runner = LocalRunner(
        tmp_path,
        secret_env={"API_TOKEN": SecretRef(name=_SECRET_NAME)},
        secret_resolver=vault,
    )

    with pytest.raises(ValueError, match="collides with declared secret_env"):
        asyncio.run(
            runner.exec(
                ExecCommand.process(sys.executable, "-c", "pass"),
                env={"api_token": "override"},
            )
        )

    assert vault.resolve_calls == 0


def test_local_runner_uses_windows_environment_name_identity_for_removal(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        local_runner_module,
        "_local_environment_names_case_sensitive",
        lambda: False,
    )
    runner = LocalRunner(tmp_path)

    result = asyncio.run(
        runner.exec(
            ExecCommand.process(
                sys.executable,
                "-c",
                "import os; print('MixedCase' in os.environ)",
            ),
            env={"MixedCase": "value"},
            env_remove=("MIXEDCASE",),
        )
    )

    assert result.exit_code == 0
    assert result.stdout == "False\n"


def test_local_runner_rejects_windows_equivalent_explicit_duplicates_before_lookup(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        local_runner_module,
        "_local_environment_names_case_sensitive",
        lambda: False,
    )
    vault = _CountingVault()
    runner = LocalRunner(
        tmp_path,
        secret_env={"DECLARED_SECRET": SecretRef(name=_SECRET_NAME)},
        secret_resolver=vault,
    )

    with pytest.raises(ValueError, match="duplicate environment variable names"):
        asyncio.run(
            runner.exec(
                ExecCommand.process(sys.executable, "-c", "pass"),
                env={"EXPLICIT_NAME": "first", "explicit_name": "second"},
            )
        )

    assert vault.resolve_calls == 0


def test_local_runner_rejects_windows_equivalent_secret_duplicates_before_lookup(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        local_runner_module,
        "_local_environment_names_case_sensitive",
        lambda: False,
    )
    vault = _CountingVault()

    with pytest.raises(ValueError, match="duplicate environment variable names"):
        LocalRunner(
            tmp_path,
            secret_env={
                "API_TOKEN": SecretRef(name=_SECRET_NAME),
                "api_token": SecretRef(name=_SECRET_NAME),
            },
            secret_resolver=vault,
        )

    assert vault.resolve_calls == 0


def test_local_runner_windows_explicit_environment_replaces_inherited_equivalent(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        local_runner_module,
        "_local_environment_names_case_sensitive",
        lambda: False,
    )
    runner = LocalRunner(tmp_path, inherit_env=True)

    result = asyncio.run(
        runner.exec(
            ExecCommand.process(
                sys.executable,
                "-c",
                "import os; print(os.environ.get('PATH', '')); print(os.environ['Path'])",
            ),
            env={"Path": "explicit-path"},
        )
    )

    assert result.exit_code == 0
    assert result.stdout == "\nexplicit-path\n"


def test_run_subprocess_executes_process_and_bounds_output(tmp_path) -> None:
    result = asyncio.run(
        run_subprocess(
            SubprocessCommand(
                argv=[
                    sys.executable,
                    "-c",
                    "print('abcdef')",
                ]
            ),
            cwd=tmp_path,
            env={},
            output_limit_bytes=4,
        )
    )

    assert result.stdout == "abcd"
    assert result.stdout_truncated is True
    assert result.stdout_bytes == 7
    assert result.stderr_bytes == 0
    assert result.exit_code == 0


def test_run_subprocess_executes_shell_and_stdin(tmp_path) -> None:
    result = asyncio.run(
        run_subprocess(
            SubprocessCommand(shell="cat"),
            cwd=tmp_path,
            env={},
            stdin="hello",
        )
    )

    assert result.stdout == "hello"
    assert result.exit_code == 0


def test_run_subprocess_reports_missing_command(tmp_path) -> None:
    result = asyncio.run(
        run_subprocess(
            SubprocessCommand(argv=["cayu-command-that-does-not-exist"]),
            cwd=tmp_path,
            env={},
        )
    )

    assert result.exit_code == 127
    assert "Command not found" in result.stderr
    assert result.stdout_bytes == 0
    assert result.stderr_bytes == len(result.stderr.encode("utf-8"))


@pytest.mark.parametrize(
    ("spawn_error", "exit_code", "message"),
    (
        (FileNotFoundError(), 127, "Command not found"),
        (PermissionError(), 126, "Command not executable"),
    ),
)
def test_run_subprocess_spawn_diagnostic_uses_owned_command(
    monkeypatch,
    tmp_path,
    spawn_error: OSError,
    exit_code: int,
    message: str,
) -> None:
    spawn_started = asyncio.Event()
    release_spawn = asyncio.Event()

    async def fail_spawn(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        spawn_started.set()
        await release_spawn.wait()
        raise spawn_error

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fail_spawn)

    async def run() -> ExecResult:
        command = SubprocessCommand(argv=["owned-command"])
        task = asyncio.create_task(run_subprocess(command, cwd=tmp_path, env={}))
        await spawn_started.wait()
        assert command.argv is not None
        command.argv[0] = "mutated\ud800command"
        release_spawn.set()
        return await task

    result = asyncio.run(run())

    assert result.exit_code == exit_code
    assert result.stderr == f"{message}: owned-command"


def test_run_subprocess_times_out_and_returns_partial_output(tmp_path) -> None:
    result = asyncio.run(
        run_subprocess(
            SubprocessCommand(
                argv=[
                    sys.executable,
                    "-c",
                    "import time; print('before', flush=True); time.sleep(5)",
                ]
            ),
            cwd=tmp_path,
            env={},
            timeout_s=1,
        )
    )

    assert result.stdout == "before\n"
    assert result.timed_out is True
    assert result.exit_code != 0


def test_windows_taskkill_runs_off_event_loop_with_timeout(monkeypatch) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setenv("SYSTEMROOT", r"C:\Windows")
    monkeypatch.setenv("PATH", r"C:\Windows\System32")
    monkeypatch.setenv("OPENAI_API_KEY", "provider-canary-0123456789")
    monkeypatch.setenv("CAYU_HOME", r"C:\provider-auth-canary")

    def slow_taskkill(argv, *, capture_output, check, env, timeout):
        observed.update(
            argv=argv,
            capture_output=capture_output,
            check=check,
            env=env,
            timeout=timeout,
        )
        time.sleep(0.05)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess_module.subprocess, "run", slow_taskkill)

    async def run() -> tuple[bool, int]:
        task = asyncio.create_task(subprocess_module._taskkill_tree(123))
        ticks = 0
        while not task.done():
            ticks += 1
            await asyncio.sleep(0.005)
        return await task, ticks

    succeeded, ticks = asyncio.run(run())

    assert succeeded is True
    assert ticks > 1
    assert observed == {
        "argv": ["taskkill", "/F", "/T", "/PID", "123"],
        "capture_output": True,
        "check": False,
        "env": {
            "PATH": r"C:\Windows\System32",
            "SYSTEMROOT": r"C:\Windows",
        },
        "timeout": subprocess_module._TASKKILL_TIMEOUT_S,
    }


def test_windows_taskkill_timeout_falls_back_to_direct_kill(monkeypatch) -> None:
    def timed_out(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="taskkill", timeout=2)

    monkeypatch.setattr(subprocess_module.subprocess, "run", timed_out)
    assert asyncio.run(subprocess_module._taskkill_tree(123)) is False

    class FakeProcess:
        pid = 123

        def __init__(self) -> None:
            self.killed = False

        def kill(self) -> None:
            self.killed = True

    async def failed_tree_kill(pid: int) -> bool:
        assert pid == 123
        return False

    process = FakeProcess()
    monkeypatch.setattr(subprocess_module.os, "name", "nt")
    monkeypatch.setattr(subprocess_module, "_taskkill_tree", failed_tree_kill)

    asyncio.run(subprocess_module._kill_process(process, process_group=False))  # type: ignore[arg-type]

    assert process.killed is True


def test_windows_taskkill_timeout_includes_executor_queue_time(monkeypatch) -> None:
    worker_started = asyncio.Event()
    worker_cancelled = asyncio.Event()

    async def queued_to_thread(*_args, **_kwargs):
        worker_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            worker_cancelled.set()
            raise

    monkeypatch.setattr(subprocess_module.asyncio, "to_thread", queued_to_thread)
    monkeypatch.setattr(subprocess_module, "_TASKKILL_TIMEOUT_S", 0.01)

    async def run() -> bool:
        task = asyncio.create_task(subprocess_module._taskkill_tree(123))
        await worker_started.wait()
        return await asyncio.wait_for(task, timeout=0.2)

    assert asyncio.run(run()) is False
    assert worker_cancelled.is_set()


def test_timeout_kill_resists_repeated_cancellation_and_cleans_io_tasks(monkeypatch) -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.killed = False

        def kill(self) -> None:
            self.killed = True

    kill_started = asyncio.Event()
    release_kill = asyncio.Event()

    async def slow_kill(process, *, process_group):
        assert process_group is False
        kill_started.set()
        await release_kill.wait()
        process.kill()

    monkeypatch.setattr(subprocess_module, "_kill_process", slow_kill)

    async def run() -> tuple[FakeProcess, tuple[asyncio.Task, ...]]:
        blocker = asyncio.Event()
        io_tasks = tuple(asyncio.create_task(blocker.wait()) for _ in range(3))
        wait_task = asyncio.create_task(asyncio.sleep(0, result=0))
        process = FakeProcess()
        cleanup_task = asyncio.create_task(
            subprocess_module._kill_timed_out_process(
                process,  # type: ignore[arg-type]
                process_group=False,
                stdin_task=io_tasks[0],
                stdout_task=io_tasks[1],
                stderr_task=io_tasks[2],
                wait_task=wait_task,
            )
        )
        await kill_started.wait()
        cleanup_task.cancel()
        await asyncio.sleep(0)
        cleanup_task.cancel()
        release_kill.set()
        with pytest.raises(asyncio.CancelledError):
            await cleanup_task
        return process, (*io_tasks, wait_task)

    process, tasks = asyncio.run(run())

    assert process.killed is True
    assert all(task.done() for task in tasks)
    assert all(task.cancelled() for task in tasks[:3])


def test_cancelled_process_cleanup_resists_second_cancellation(monkeypatch) -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.killed = False

        def kill(self) -> None:
            self.killed = True

    kill_started = asyncio.Event()
    release_kill = asyncio.Event()

    async def slow_kill(process, *, process_group):
        assert process_group is False
        kill_started.set()
        await release_kill.wait()
        process.kill()

    monkeypatch.setattr(subprocess_module, "_kill_process", slow_kill)

    async def run() -> tuple[FakeProcess, tuple[asyncio.Task, ...]]:
        blocker = asyncio.Event()
        io_tasks = tuple(asyncio.create_task(blocker.wait()) for _ in range(3))
        wait_task = asyncio.create_task(asyncio.sleep(0, result=0))
        process = FakeProcess()

        async def operation() -> None:
            try:
                await blocker.wait()
            except asyncio.CancelledError:
                await subprocess_module._cleanup_cancelled_process(
                    process,  # type: ignore[arg-type]
                    process_group=False,
                    stdin_task=io_tasks[0],
                    stdout_task=io_tasks[1],
                    stderr_task=io_tasks[2],
                    wait_task=wait_task,
                )
                raise

        operation_task = asyncio.create_task(operation())
        await asyncio.sleep(0)
        operation_task.cancel()
        await kill_started.wait()
        operation_task.cancel()
        release_kill.set()
        with pytest.raises(asyncio.CancelledError):
            await operation_task
        return process, (*io_tasks, wait_task)

    process, tasks = asyncio.run(run())

    assert process.killed is True
    assert all(task.done() for task in tasks)
    assert all(task.cancelled() for task in tasks[:3])


def test_timeout_kill_failure_cleans_io_tasks_and_propagates(monkeypatch) -> None:
    async def failed_termination(process, *, process_group, wait_task):
        assert process_group is False
        raise RuntimeError("termination failed")

    monkeypatch.setattr(subprocess_module, "_kill_process_and_wait", failed_termination)

    async def run() -> tuple[asyncio.Task, ...]:
        blocker = asyncio.Event()
        tasks = tuple(asyncio.create_task(blocker.wait()) for _ in range(4))
        with pytest.raises(RuntimeError, match="termination failed"):
            await subprocess_module._kill_timed_out_process(
                object(),  # type: ignore[arg-type]
                process_group=False,
                stdin_task=tasks[0],
                stdout_task=tasks[1],
                stderr_task=tasks[2],
                wait_task=tasks[3],
            )
        return tasks

    tasks = asyncio.run(run())

    assert all(task.done() for task in tasks)
    assert all(task.cancelled() for task in tasks)


def test_timeout_kill_preserves_cancellation_when_termination_fails(monkeypatch) -> None:
    termination_started = asyncio.Event()
    release_termination = asyncio.Event()

    async def failed_termination(process, *, process_group, wait_task):
        assert process_group is False
        termination_started.set()
        await release_termination.wait()
        raise RuntimeError("termination failed")

    monkeypatch.setattr(subprocess_module, "_kill_process_and_wait", failed_termination)

    async def run() -> tuple[asyncio.CancelledError, tuple[asyncio.Task, ...]]:
        blocker = asyncio.Event()
        tasks = tuple(asyncio.create_task(blocker.wait()) for _ in range(4))
        cleanup_task = asyncio.create_task(
            subprocess_module._kill_timed_out_process(
                object(),  # type: ignore[arg-type]
                process_group=False,
                stdin_task=tasks[0],
                stdout_task=tasks[1],
                stderr_task=tasks[2],
                wait_task=tasks[3],
            )
        )
        await termination_started.wait()
        cleanup_task.cancel("caller cancelled")
        await asyncio.sleep(0)
        release_termination.set()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await cleanup_task
        return exc_info.value, tasks

    cancellation, tasks = asyncio.run(run())

    assert str(cancellation) == "caller cancelled"
    assert isinstance(cancellation.__cause__, RuntimeError)
    assert "termination failed" in "\n".join(cancellation.__notes__)
    assert all(task.done() for task in tasks)
    assert all(task.cancelled() for task in tasks)


def test_timeout_kill_preserves_cancellation_during_io_cleanup_after_termination_failure(
    monkeypatch,
) -> None:
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleanup_io_tasks = subprocess_module._cleanup_io_tasks

    async def failed_termination(process, *, process_group, wait_task):
        assert process_group is False
        raise RuntimeError("termination failed")

    async def slow_cleanup(*tasks):
        cleanup_started.set()
        await release_cleanup.wait()
        await cleanup_io_tasks(*tasks)

    monkeypatch.setattr(subprocess_module, "_kill_process_and_wait", failed_termination)
    monkeypatch.setattr(subprocess_module, "_cleanup_io_tasks", slow_cleanup)

    async def run() -> tuple[asyncio.CancelledError, tuple[asyncio.Task, ...]]:
        blocker = asyncio.Event()
        tasks = tuple(asyncio.create_task(blocker.wait()) for _ in range(4))
        kill_task = asyncio.create_task(
            subprocess_module._kill_timed_out_process(
                object(),  # type: ignore[arg-type]
                process_group=False,
                stdin_task=tasks[0],
                stdout_task=tasks[1],
                stderr_task=tasks[2],
                wait_task=tasks[3],
            )
        )
        await cleanup_started.wait()
        kill_task.cancel("caller cancelled during I/O cleanup")
        await asyncio.sleep(0)
        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await kill_task
        return exc_info.value, tasks

    cancellation, tasks = asyncio.run(run())

    assert str(cancellation) == "caller cancelled during I/O cleanup"
    assert isinstance(cancellation.__cause__, RuntimeError)
    assert "termination failed" in "\n".join(cancellation.__notes__)
    assert all(task.done() for task in tasks)
    assert all(task.cancelled() for task in tasks)


def test_cancelled_process_cleanup_falls_back_when_termination_fails(monkeypatch) -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.killed = False

        def kill(self) -> None:
            self.killed = True

    async def failed_termination(process, *, process_group, wait_task):
        assert process_group is False
        raise RuntimeError("termination failed")

    monkeypatch.setattr(subprocess_module, "_kill_process_and_wait", failed_termination)

    async def run() -> tuple[FakeProcess, tuple[asyncio.Task, ...]]:
        blocker = asyncio.Event()
        tasks = tuple(asyncio.create_task(blocker.wait()) for _ in range(4))
        process = FakeProcess()
        await subprocess_module._cleanup_cancelled_process(
            process,  # type: ignore[arg-type]
            process_group=False,
            stdin_task=tasks[0],
            stdout_task=tasks[1],
            stderr_task=tasks[2],
            wait_task=tasks[3],
        )
        return process, tasks

    process, tasks = asyncio.run(run())

    assert process.killed is True
    assert all(task.done() for task in tasks)
    assert all(task.cancelled() for task in tasks)


@pytest.mark.skipif(sys.platform == "win32", reason="posix session semantics")
def test_run_subprocess_bounded_drain_when_child_leaks_pipe(tmp_path) -> None:
    # The child spawns a detached (own-session) grandchild that inherits the
    # captured stdout pipe and outlives the kill, so the stdout read would never
    # see EOF. The bounded post-kill drain must still return promptly.
    child = (
        "import sys, subprocess, time\n"
        "subprocess.Popen(\n"
        "    [sys.executable, '-c', 'import time; time.sleep(30)'],\n"
        "    start_new_session=True,\n"
        ")\n"
        "print('parent', flush=True)\n"
        "time.sleep(30)\n"
    )
    started = time.monotonic()
    result = asyncio.run(
        run_subprocess(
            SubprocessCommand(argv=[sys.executable, "-c", child]),
            cwd=tmp_path,
            env={},
            timeout_s=1,
        )
    )
    elapsed = time.monotonic() - started

    assert result.timed_out is True
    assert "parent" in result.stdout
    # The grandchild sleeps 30s; without the bounded drain the gather would hang
    # that long. Timeout (1s) + drain bound (2s) plus margin must be well under.
    assert elapsed < 10
    assert result.stdout_truncated is True
