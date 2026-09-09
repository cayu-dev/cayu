"""Fixed public classifications for opaque runner failures."""

from __future__ import annotations

import errno as errno_module

from cayu._exception_state import set_exception_state


class SubprocessLaunchRefused(OSError):
    """An exec E2BIG refusal positively established at the local spawn boundary."""


_TRUSTED_RUNNER_ERROR_TYPE_NAMES = frozenset(
    {
        "ArithmeticError",
        "AssertionError",
        "AttributeError",
        "AuthenticationException",
        "BaseException",
        "BaseExceptionGroup",
        "BlockingIOError",
        "BrokenPipeError",
        "BuildException",
        "BufferError",
        "BytesWarning",
        "CancelledError",
        "ChildProcessError",
        "CloudHttpError",
        "CommandExitException",
        "ConnectionAbortedError",
        "ConnectionError",
        "ConnectionRefusedError",
        "ConnectionResetError",
        "DeprecationWarning",
        "E2BGuestHandoffError",
        "EOFError",
        "EncodingWarning",
        "Exception",
        "ExceptionGroup",
        "ExecFailedError",
        "ExecTimeoutError",
        "FileExistsError",
        "FileNotFoundException",
        "FileNotFoundError",
        "FileUploadException",
        "FilesystemError",
        "FloatingPointError",
        "FutureWarning",
        "GeneratorExit",
        "GitAuthException",
        "GitUpstreamException",
        "ImageInUseError",
        "ImageNotFoundError",
        "ImagePullFailedError",
        "ImportError",
        "ImportWarning",
        "IndentationError",
        "IndexError",
        "InvalidArgumentException",
        "InvalidConfigError",
        "InterruptedError",
        "IsADirectoryError",
        "IoError",
        "KeyError",
        "KeyboardInterrupt",
        "LambdaMicroVMEndpointUnauthorized",
        "LambdaMicroVMError",
        "LambdaMicroVMProtocolError",
        "LookupError",
        "MemoryError",
        "MetricsDisabledError",
        "MetricsUnavailableError",
        "MicrosandboxCleanupError",
        "MicrosandboxError",
        "MicrosandboxReconnectIdentityError",
        "MicrosandboxUnavailableError",
        "ModuleNotFoundError",
        "NameError",
        "NetworkPolicyError",
        "NotEnoughSpaceException",
        "NotADirectoryError",
        "NotFoundException",
        "NotImplementedError",
        "OSError",
        "SubprocessLaunchRefused",
        "OverflowError",
        "PathNotFoundError",
        "PendingDeprecationWarning",
        "PermissionError",
        "ProcessLookupError",
        "PythonFinalizationError",
        "RateLimitException",
        "RecursionError",
        "ReferenceError",
        "ResourceWarning",
        "RunnerExecutionError",
        "RunnerUnavailableError",
        "RuntimeError",
        "RuntimeWarning",
        "SandboxAlreadyExistsError",
        "SandboxException",
        "SandboxNotFoundError",
        "SandboxNotFoundException",
        "SandboxNotRunningError",
        "SandboxStillRunningError",
        "SecretViolationError",
        "StopAsyncIteration",
        "StopIteration",
        "SyntaxError",
        "SyntaxWarning",
        "SystemError",
        "SystemExit",
        "TabError",
        "TemplateException",
        "TimeoutException",
        "TimeoutError",
        "TlsError",
        "TypeError",
        "UnboundLocalError",
        "UnicodeDecodeError",
        "UnicodeEncodeError",
        "UnicodeError",
        "UnicodeTranslateError",
        "UnicodeWarning",
        "UnexpectedStatus",
        "UnsupportedError",
        "UnsupportedOperationError",
        "UserWarning",
        "ValueError",
        "VolumeException",
        "VolumeNotFoundError",
        "Warning",
        "ZeroDivisionError",
    }
)


def trusted_runner_exception_type_name(error: BaseException) -> str:
    """Return a fixed public classification for one opaque runner failure."""

    try:
        name = type.__getattribute__(type(error), "__name__")
    except BaseException:
        return "Exception"
    return trusted_runner_error_type_name(name) or "Exception"


def trusted_runner_error_type_name(value: object) -> str | None:
    """Accept only runtime-owned runner classifications, never extension text."""

    if type(value) is not str or value not in _TRUSTED_RUNNER_ERROR_TYPE_NAMES:
        return None
    return value


_RUNNER_FAILURE_PHASES = frozenset(
    {"launch", "transport", "stream_handling", "process_wait", "filesystem", "cleanup"}
)
_TRUSTED_OS_ERRORS = (
    SubprocessLaunchRefused,
    OSError,
    BlockingIOError,
    ChildProcessError,
    ConnectionError,
    BrokenPipeError,
    ConnectionAbortedError,
    ConnectionRefusedError,
    ConnectionResetError,
    FileExistsError,
    FileNotFoundError,
    InterruptedError,
    IsADirectoryError,
    NotADirectoryError,
    PermissionError,
    ProcessLookupError,
    TimeoutError,
)
_ERRNO_CODES = dict(errno_module.errorcode)


def safe_runner_failure_fields(errno: object, phase: object) -> dict[str, object]:
    """Normalize additive v1 fields without coercion or extension hooks."""

    number = errno if type(errno) is int and 0 < errno <= 2**31 - 1 else None
    return {
        "errno": number,
        "errno_code": _ERRNO_CODES.get(number) if number is not None else None,
        "execution_phase": (
            phase if type(phase) is str and phase in _RUNNER_FAILURE_PHASES else "unknown"
        ),
    }


def runner_failure_fields(error: BaseException) -> dict[str, object]:
    """Read errno only from exact trusted OS errors, never custom descriptors."""

    number = None
    if any(type(error) is candidate for candidate in _TRUSTED_OS_ERRORS):
        number = OSError.__dict__["errno"].__get__(error, OSError)
    namespace = BaseException.__dict__["__dict__"].__get__(error, BaseException)
    phase = None
    for key, value in dict.items(namespace):
        if type(key) is str and key == "_cayu_runner_execution_phase":
            phase = value
            break
    return safe_runner_failure_fields(number, phase)


def tag_runner_failure_phase(error: BaseException, phase: str) -> None:
    """Attach boundary knowledge without changing exception or settlement semantics."""

    safe = safe_runner_failure_fields(None, phase)
    set_exception_state(error, "_cayu_runner_execution_phase", safe["execution_phase"])
