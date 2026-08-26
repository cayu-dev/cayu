"""Disposable child entrance for one process-isolated tool invocation."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import os
import signal
import sys
from collections.abc import Awaitable
from contextlib import suppress
from typing import Any

from cayu.core.execution_identity import (
    ExecutionProfileBehaviorIdentity,
    copy_execution_profile_behavior_identity,
)
from cayu.core.isolated_tools import MAX_ISOLATED_TOOL_MESSAGE_BYTES
from cayu.core.tools import ToolResult
from cayu.runtime._isolated_tool_protocol import (
    IsolatedToolChildErrorCode,
    IsolatedToolInvocationEnvelope,
    decode_isolated_tool_request,
    encode_isolated_tool_error,
    encode_isolated_tool_success,
    encode_isolated_tool_terminal_frame,
)

_REQUIRED_BASE_ENVIRONMENT = {
    "LC_ALL": "C",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUTF8": "1",
}


def _close_ambient_file_descriptors(*, result_fd: int) -> None:
    """Remove inherited handles before the isolated adapter acquires its own."""

    descriptor_names: list[str] | None = None
    for descriptor_directory in ("/proc/self/fd", "/dev/fd"):
        try:
            descriptor_names = os.listdir(descriptor_directory)
        except OSError:
            continue
        break
    if descriptor_names is not None:
        for name in descriptor_names:
            try:
                descriptor = int(name)
            except ValueError:
                continue
            if descriptor <= 2 or descriptor == result_fd:
                continue
            with suppress(OSError):
                os.close(descriptor)
        return

    try:
        maximum_descriptor = int(os.sysconf("SC_OPEN_MAX"))
    except (OSError, TypeError, ValueError):
        maximum_descriptor = 1 << 20
    os.closerange(3, result_fd)
    os.closerange(result_fd + 1, maximum_descriptor)


class _FactoryIdentityMismatch(Exception):
    pass


async def _await_result(awaitable: Awaitable[Any]) -> Any:
    return await awaitable


def _resolve_factory(envelope: IsolatedToolInvocationEnvelope) -> Any:
    module = importlib.import_module(envelope.factory.module)
    target: Any = module
    for segment in envelope.factory.qualname.split("."):
        target = getattr(target, segment)
    if not callable(target):
        raise TypeError
    if (
        getattr(target, "__module__", None) != envelope.factory.module
        or getattr(target, "__qualname__", None) != envelope.factory.qualname
    ):
        raise TypeError
    declared_identity = getattr(target, "execution_profile_identity", None)
    if not isinstance(declared_identity, ExecutionProfileBehaviorIdentity):
        raise _FactoryIdentityMismatch
    copied_identity = copy_execution_profile_behavior_identity(declared_identity)
    if copied_identity != envelope.factory.identity:
        raise _FactoryIdentityMismatch
    return target


def _execute(
    envelope: IsolatedToolInvocationEnvelope,
    *,
    result_fd: int,
) -> tuple[ToolResult | None, str | None]:
    # macOS injects this process-local locale hint even when ``env`` is supplied
    # explicitly. It is not adapter authority, so remove it before checking and
    # before application code can observe the child environment.
    os.environ.pop("__CF_USER_TEXT_ENCODING", None)
    expected_environment = {**_REQUIRED_BASE_ENVIRONMENT, **envelope.environment}
    if dict(os.environ) != expected_environment:
        return None, IsolatedToolChildErrorCode.ENVIRONMENT_INVALID.value
    _close_ambient_file_descriptors(result_fd=result_fd)
    try:
        factory = _resolve_factory(envelope)
    except _FactoryIdentityMismatch:
        return None, IsolatedToolChildErrorCode.FACTORY_IDENTITY_MISMATCH.value
    except BaseException:
        return None, IsolatedToolChildErrorCode.FACTORY_IMPORT_FAILED.value
    try:
        handler = factory(envelope.factory_config)
    except BaseException:
        return None, IsolatedToolChildErrorCode.FACTORY_CONSTRUCTION_FAILED.value
    if inspect.isawaitable(handler):
        return None, IsolatedToolChildErrorCode.FACTORY_INVALID.value
    try:
        run = handler.run
    except BaseException:
        return None, IsolatedToolChildErrorCode.FACTORY_INVALID.value
    if not callable(run):
        return None, IsolatedToolChildErrorCode.FACTORY_INVALID.value
    try:
        result = run(envelope.context, envelope.arguments)
        if inspect.isawaitable(result):
            result = asyncio.run(_await_result(result))
    except BaseException:
        return None, IsolatedToolChildErrorCode.CHILD_EXCEPTION.value
    if type(result) is not ToolResult:
        return None, IsolatedToolChildErrorCode.INVALID_RESULT.value
    return result, None


def _terminal_bytes(envelope: IsolatedToolInvocationEnvelope, *, result_fd: int) -> bytes:
    result, error_code = _execute(envelope, result_fd=result_fd)
    if error_code is not None:
        return encode_isolated_tool_error(
            request_sha256=envelope.request_sha256,
            error_code=IsolatedToolChildErrorCode(error_code),
            max_bytes=envelope.limits.max_response_bytes,
        )
    if result is None:  # pragma: no cover - execution tuple invariant
        raise AssertionError("Isolated tool execution produced no terminal value.")
    try:
        return encode_isolated_tool_success(
            request_sha256=envelope.request_sha256,
            result=result,
            max_bytes=envelope.limits.max_response_bytes,
        )
    except BaseException:
        return encode_isolated_tool_error(
            request_sha256=envelope.request_sha256,
            error_code=IsolatedToolChildErrorCode.INVALID_RESULT,
            max_bytes=envelope.limits.max_response_bytes,
        )


def _write_terminal(result_fd: int, terminal: bytes, *, max_bytes: int) -> None:
    framed = encode_isolated_tool_terminal_frame(terminal, max_bytes=max_bytes)
    with os.fdopen(result_fd, "wb", closefd=True) as result_stream:
        result_stream.write(framed)
        result_stream.flush()
    framed = b""


def _wait_for_parent_shutdown() -> None:
    while True:
        signal.pause()


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if os.name != "posix" or len(arguments) != 2 or arguments[0] != "--result-fd":
        return 64
    try:
        result_fd = int(arguments[1])
    except ValueError:
        return 64
    if result_fd < 3:
        return 64
    raw_request = sys.stdin.buffer.read(MAX_ISOLATED_TOOL_MESSAGE_BYTES + 1)
    try:
        envelope = decode_isolated_tool_request(raw_request)
    except BaseException:
        return 65
    finally:
        raw_request = b""
    try:
        terminal = _terminal_bytes(envelope, result_fd=result_fd)
    except BaseException:
        try:
            terminal = encode_isolated_tool_error(
                request_sha256=envelope.request_sha256,
                error_code=IsolatedToolChildErrorCode.INTERNAL_PROTOCOL_FAILURE,
                max_bytes=envelope.limits.max_response_bytes,
            )
        except BaseException:
            return 70
    try:
        _write_terminal(
            result_fd,
            terminal,
            max_bytes=envelope.limits.max_response_bytes,
        )
    except BaseException:
        return 74
    finally:
        terminal = b""
    _wait_for_parent_shutdown()
    return 0  # pragma: no cover - parent terminates the disposable worker


if __name__ == "__main__":  # pragma: no cover - exercised through real subprocess tests
    raise SystemExit(main())
