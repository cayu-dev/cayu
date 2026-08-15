"""Example: a custom `ModalRunner` that executes commands in a Modal sandbox.

This is a worked example for `docs/build-a-runner.md`, not maintained core — it
is intentionally NOT importable from `cayu`. It shows how to satisfy the
`cayu.runners.Runner` contract for any platform, using Modal Sandboxes as the
target.

NOT production-ready as written: it reads each command's full stdout/stderr into
memory before truncating (so `output_limit_bytes` does not bound memory), exposes
no `close()`/`__aexit__` lifecycle, and serves one command at a time. For the
complete contract — bounded streaming capture, command-vs-sandbox cleanup
policies, lifecycle — see the built-in runners under `src/cayu/runners/`.

The `modal` SDK is imported lazily (it is not a Cayu dependency) and is only
needed by `create()` / `main()`, so this file imports and type-checks cleanly
when `modal` is not installed. The Modal calls used here were checked against the
modal 1.5.x API and run end-to-end — `Sandbox.create(app=, image=)`,
`Sandbox.exec(*argv, env=, workdir=, timeout=)`, `ContainerProcess.wait` (returns
the exit code; `-1` when Modal kills the command at its `timeout=` deadline, with
partial output still readable) / `stdout`/`stderr`/`stdin`, `Sandbox.terminate`,
`App.lookup` — but re-verify against your installed Modal version.

Run (needs `pip install modal` and a configured Modal account):
    PYTHONPATH=src python examples/modal_runner.py
"""

from __future__ import annotations

import asyncio
import importlib
import posixpath
import traceback
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from cayu import (
    DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    CayuApp,
    Environment,
    EnvironmentSpec,
    ExecCommand,
    ExecResult,
    RunnerCancelledError,
)
from cayu.runners import Runner  # the base ABC is exported from cayu.runners, not top-level cayu

DEFAULT_MODAL_CWD = "/workspace"
# Best-effort cancellation cleanup is bounded so a hung terminate can't make
# cancellation itself hang (built-in runners bound cleanup the same way).
DEFAULT_MODAL_CANCEL_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class _PreparedModalExecRequest:
    """One fully owned request safe to carry across the Modal SDK await."""

    argv: tuple[str, ...]
    workdir: str
    environment_items: tuple[tuple[str, str], ...]
    timeout_s: int | None
    stdin: str | None
    output_limit_bytes: int | None


class ModalRunner(Runner):
    """Runs `ExecCommand`s inside a Modal sandbox.

    Like the built-in sandbox runners, this does NOT inherit the trusted host
    environment. Only the explicit `env` passed to `exec` reaches the sandbox,
    and secrets should be resolved at the environment/vault boundary before they
    get here (see the secret-injection section of docs/build-a-runner.md).
    """

    isolation = "modal"

    def __init__(self, sandbox: Any, *, default_cwd: str = DEFAULT_MODAL_CWD) -> None:
        if sandbox is None:
            raise TypeError("ModalRunner sandbox cannot be None.")
        self._sandbox = sandbox
        self.default_cwd = _validate_guest_root(default_cwd)

    @classmethod
    async def create(
        cls,
        *,
        app: Any,
        image: Any = None,
        default_cwd: str = DEFAULT_MODAL_CWD,
        modal_module: ModuleType | Any | None = None,
        **sandbox_options: Any,
    ) -> ModalRunner:
        """Provision a Modal sandbox and return a runner bound to it.

        `image` must be a `modal.Image` (not a string); when omitted we build the
        default `Image.debian_slim()`. The guest root (`default_cwd`) is created up
        front, because a base image has no `/workspace` and `exec(workdir=...)`
        would otherwise fail — the built-in sandbox runners verify their root too.
        """

        module = _modal_module(modal_module)
        resolved_image = module.Image.debian_slim() if image is None else image
        sandbox = await module.Sandbox.create.aio(app=app, image=resolved_image, **sandbox_options)
        # The sandbox is now provisioned (and billable). If guest-root setup fails or is
        # cancelled, tear it down rather than leaking it — the built-in sandbox runners
        # clean up the same way on a mid-setup failure (see e2b.py / docker.py).
        try:
            runner = cls(sandbox, default_cwd=default_cwd)
            await _ensure_guest_root(sandbox, runner.default_cwd)
        except BaseException:
            await _terminate_bounded(sandbox)
            raise
        return runner

    async def exec(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        env_remove: tuple[str, ...] = (),
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ) -> ExecResult:
        prepared = self._prepare_exec_request(
            command,
            cwd=cwd,
            env=env,
            env_remove=env_remove,
            timeout_s=timeout_s,
            stdin=stdin,
            output_limit_bytes=output_limit_bytes,
        )
        del command, cwd, env, env_remove, timeout_s, stdin, output_limit_bytes

        async def run() -> ExecResult:
            environment = dict(prepared.environment_items)
            # Modal's exec takes a plain env dict via `env=` (the guide also shows
            # `secrets=[Secret.from_dict(env)]`; both work). Modal's native `timeout=`
            # bounds the command server-side WITHOUT tearing down the sandbox: at the
            # deadline it kills the command and `wait()` returns exit_code -1, with the
            # stdout/stderr produced before the kill still readable. We rely on that
            # (no racing client-side timer) so partial output is preserved. For an
            # OS-pipe backend you would read stdout/stderr concurrently (asyncio.gather)
            # to avoid a full-pipe deadlock; Modal's streams are network-buffered, so
            # sequential is fine.
            #
            # SIMPLIFICATION: this reads ALL of stdout/stderr and truncates after the
            # fact, so it is NOT memory-safe for unbounded output. The Runner contract
            # expects `output_limit_bytes` to bound capture INSIDE the runner; a
            # production runner streams into a bounded buffer (see `_LimitedBytes` in
            # src/cayu/runners/microsandbox.py) so large output can't exhaust memory.
            process = await self._sandbox.exec.aio(
                *prepared.argv,
                workdir=prepared.workdir,
                env=environment,
                timeout=prepared.timeout_s,
            )
            if prepared.stdin is not None:
                process.stdin.write(prepared.stdin.encode("utf-8"))
                process.stdin.write_eof()
                await process.stdin.drain.aio()
            stdout, stdout_truncated = _truncate(
                await process.stdout.read.aio(),
                prepared.output_limit_bytes,
            )
            stderr, stderr_truncated = _truncate(
                await process.stderr.read.aio(),
                prepared.output_limit_bytes,
            )
            exit_code = await process.wait.aio()
            if prepared.timeout_s is not None and exit_code == -1:
                # Modal killed the command at its server-side `timeout=` deadline.
                # Report the runner's timeout contract (exit_code -9, matching the
                # built-in sandbox runners) and keep the partial output read above. A
                # single command timeout does NOT tear down the shared sandbox.
                return ExecResult(
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=-9,
                    timed_out=True,
                    stdout_truncated=stdout_truncated,
                    stderr_truncated=stderr_truncated,
                )
            # Never raise on a non-zero exit — surface it. A killed process can report
            # a non-int code, so coerce defensively rather than letting int(...) raise.
            return ExecResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=int(exit_code) if isinstance(exit_code, int) else -1,
                stdout_truncated=stdout_truncated,
                stderr_truncated=stderr_truncated,
            )

        try:
            return await run()
        except asyncio.CancelledError as exc:
            # Modal has no per-command kill (ContainerProcess has no terminate), so the
            # only way to stop a running command on cancellation is to tear down the
            # sandbox. Bound that cleanup so a hung terminate can't make cancellation
            # hang, and attach a cleanup diagnostic in the core shape (built-in runners
            # do this via cayu.runners._cleanup.cleanup_runner_command_with_diagnostic).
            terminated = await _terminate_bounded(self._sandbox)
            raise RunnerCancelledError(
                artifacts=[_cleanup_artifact(self.isolation, terminated)]
            ) from exc

    def preflight_exec(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        env_remove: tuple[str, ...] = (),
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ) -> None:
        """Validate the complete Modal request without SDK activity."""

        try:
            prepared = self._prepare_exec_request(
                command,
                cwd=cwd,
                env=env,
                env_remove=env_remove,
                timeout_s=timeout_s,
                stdin=stdin,
                output_limit_bytes=output_limit_bytes,
            )
        except BaseException as error:
            traceback.clear_frames(error.__traceback__)
            published_error = error
            del command, cwd, env, env_remove, timeout_s, stdin, output_limit_bytes
            del error, self
            published_error.__traceback__ = None
            raise published_error from None
        del prepared

    def _prepare_exec_request(
        self,
        command: ExecCommand,
        *,
        cwd: str | None,
        env: dict[str, str] | None,
        env_remove: tuple[str, ...],
        timeout_s: int | None,
        stdin: str | None,
        output_limit_bytes: int | None,
    ) -> _PreparedModalExecRequest:
        """Own and validate one complete request before Modal dispatch."""

        if type(command) is not ExecCommand:
            raise TypeError("ModalRunner command must be an ExecCommand.")
        self._ensure_exec_open()
        owned_command = ExecCommand.model_validate(
            command.model_dump(mode="python", warnings=False)
        )
        working_dir = self.resolve_cwd(cwd)
        environment = _copy_env(env)
        for key in _copy_env_remove(env_remove):
            environment.pop(key, None)
        argv = tuple(_command_argv(owned_command))
        timeout = _validate_timeout(timeout_s)
        standard_input = _validate_stdin(stdin)
        limit = _validate_output_limit(output_limit_bytes)
        return _PreparedModalExecRequest(
            argv=argv,
            workdir=working_dir,
            environment_items=tuple(environment.items()),
            timeout_s=timeout,
            stdin=standard_input,
            output_limit_bytes=limit,
        )


def _command_argv(command: ExecCommand) -> list[str]:
    """Translate an ExecCommand into argv. Modal exec takes argv natively, so
    process commands pass through; explicit shell scripts run under bash."""
    if command.kind == "process":
        if command.argv is None:
            raise ValueError("Process commands require argv.")
        return list(command.argv)
    if command.shell is None:
        raise ValueError("Shell commands require a script.")
    return ["bash", "-c", command.shell]


def _copy_env(env: dict[str, str] | None) -> dict[str, str]:
    """No host-env inheritance: only the explicit env reaches the sandbox.
    (The built-in runners use cayu.runners._subprocess.copy_runner_env.)"""
    if env is None:
        return {}
    if type(env) is not dict:
        raise TypeError("Runner env must be a dictionary.")
    copied: dict[str, str] = {}
    for key, value in env.items():
        key = _validate_env_name(key, "Runner env key")
        if type(value) is not str:
            raise ValueError("Runner env values must be strings.")
        value = _require_transport_text(value, "Runner env value", allow_nul=False)
        copied[key] = value
    return copied


def _copy_env_remove(keys: tuple[str, ...]) -> tuple[str, ...]:
    if type(keys) is not tuple:
        raise TypeError("Runner env_remove must be a tuple.")
    return tuple(_validate_env_name(key, "Runner env_remove entry") for key in keys)


def _validate_env_name(value: str, field_name: str) -> str:
    name = _require_transport_text(value, field_name, allow_nul=False)
    if not name.strip():
        raise ValueError(f"{field_name} must not be blank.")
    if name != name.strip():
        raise ValueError(f"{field_name} must not have surrounding whitespace.")
    if "=" in name:
        raise ValueError(f"{field_name} must not contain '='.")
    if "\n" in name or "\r" in name:
        raise ValueError(f"{field_name} must not contain line breaks.")
    return name


def _require_transport_text(value: str, field_name: str, *, allow_nul: bool) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a string.")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(f"{field_name} must not contain Unicode surrogate code points.") from None
    if not allow_nul and "\0" in value:
        raise ValueError(f"{field_name} must not contain NUL characters.")
    return value


def _truncate(data: bytes | str, limit: int | None) -> tuple[str, bool]:
    # Modal's text=True streams return str, but accept bytes too so a binary or
    # bytes-mode backend works; decode with errors="replace" so a split multibyte
    # char (or non-UTF-8 output) becomes a replacement char rather than crashing.
    raw = data if isinstance(data, bytes) else data.encode("utf-8")
    if limit is None or len(raw) <= limit:
        return raw.decode("utf-8", errors="replace"), False
    return raw[:limit].decode("utf-8", errors="replace"), True


def _validate_output_limit(value: int | None) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value <= 0:
        raise ValueError("output_limit_bytes must be a positive integer or None.")
    return value


def _validate_timeout(value: int | None) -> int | None:
    if value is None:
        return None
    if type(value) is not int:
        raise TypeError("timeout_s must be an integer.")
    if value <= 0:
        raise ValueError("timeout_s must be greater than zero.")
    return value


def _validate_stdin(value: str | None) -> str | None:
    if value is None:
        return None
    return _require_transport_text(value, "stdin", allow_nul=True)


def _validate_guest_root(path: str) -> str:
    path = _require_transport_text(path, "default_cwd", allow_nul=False)
    if not path.strip():
        raise ValueError("default_cwd must not be blank.")
    if path != path.strip():
        raise ValueError("default_cwd must not have surrounding whitespace.")
    if not posixpath.isabs(path):
        raise ValueError("ModalRunner default_cwd must be an absolute guest path.")
    return posixpath.normpath(path)


async def _ensure_guest_root(sandbox: Any, root: str) -> None:
    """Create the guest working directory before any command runs there."""
    process = await sandbox.exec.aio("mkdir", "-p", root)
    exit_code = await process.wait.aio()
    if isinstance(exit_code, int) and exit_code != 0:
        raise RuntimeError(f"Failed to create Modal guest root {root!r} (exit {exit_code}).")


async def _terminate(target: Any) -> bool:
    """Best-effort terminate; return only positive completion evidence."""
    if target is None:
        return False
    terminate = getattr(target, "terminate", None)
    if terminate is None:
        return False
    aio = getattr(terminate, "aio", None)
    result = await (aio() if aio is not None else asyncio.to_thread(terminate))
    return result is not False


async def _terminate_bounded(target: Any) -> bool:
    """Terminate within DEFAULT_MODAL_CANCEL_TIMEOUT_S; return whether it completed.

    Cancellation cleanup must not hang, so a slow/failed terminate is swallowed
    and reported as not-terminated in the cleanup diagnostic.
    """
    try:
        return await asyncio.wait_for(
            _terminate(target),
            timeout=DEFAULT_MODAL_CANCEL_TIMEOUT_S,
        )
    except Exception:
        return False


def _cleanup_artifact(adapter: str, terminated: bool) -> dict[str, Any]:
    """Cancellation cleanup diagnostic in the core `cayu.runner_cleanup.v1` shape.

    Mirrors the fields `cayu.runners._cleanup` emits ({type, adapter, action,
    status}) so downstream tooling parses it like a built-in runner's artifact.
    """
    return {
        "type": "cayu.runner_cleanup.v1",
        "adapter": adapter,
        "action": "kill_sandbox",
        "status": "completed" if terminated else "failed",
    }


def _modal_module(module: ModuleType | Any | None = None) -> ModuleType | Any:
    if module is not None:
        return module
    try:
        return importlib.import_module("modal")
    except ModuleNotFoundError as exc:
        if exc.name != "modal":
            raise
        raise RuntimeError(
            "ModalRunner requires the optional modal package. Install it with `pip install modal`."
        ) from exc


async def main() -> None:
    try:
        modal = _modal_module()
    except RuntimeError as exc:
        print(exc)
        return

    # App.lookup is a network call; use its async accessor inside an async context.
    app = await modal.App.lookup.aio("cayu-modal-runner-example", create_if_missing=True)
    # create() builds Image.debian_slim() by default and creates the guest root.
    runner = await ModalRunner.create(app=app)
    try:
        # Register the runner exactly like any built-in runner.
        cayu_app = CayuApp()
        cayu_app.register_environment(
            Environment(EnvironmentSpec(name="modal"), runner=runner),
            default=True,
        )

        result = await runner.exec(ExecCommand.process("echo", "hello from modal"))
        print("exit_code", result.exit_code)
        print("stdout", result.stdout.strip())

        failed = await runner.exec(ExecCommand.bash("ls /does-not-exist; exit 3"))
        print("failed.exit_code", failed.exit_code)
        print("failed.stderr", failed.stderr.strip())
    finally:
        await _terminate(runner._sandbox)


if __name__ == "__main__":
    asyncio.run(main())
