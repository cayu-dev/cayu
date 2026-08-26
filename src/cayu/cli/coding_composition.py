"""Maintained explicit coding composition for ``cayu new``."""

from __future__ import annotations

from collections.abc import Callable

_COMMAND_PROBE_PY = r'''"""Project-owned bounded command probes for coding dependencies."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path


class BoundedCommandStartError(RuntimeError):
    """The owned command could not be started."""


class BoundedCommandTimeoutError(RuntimeError):
    """The owned command did not settle before its deadline."""


class BoundedCommandOutputOverflowError(RuntimeError):
    """The owned command exceeded an authoritative output bound."""


class BoundedCommandReadError(RuntimeError):
    """The owned command output could not be read authoritatively."""


@dataclass(frozen=True, slots=True)
class BoundedCommandResult:
    """Content-bounded result from one settled command."""

    returncode: int
    output: bytes
    output_truncated: bool


def run_bounded_command(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None,
    timeout_s: float,
    output_limit_bytes: int,
    capture_output: bool = True,
    reject_output_overflow: bool = False,
) -> BoundedCommandResult:
    """Run one command with bounded capture and platform-owned cleanup.

    POSIX commands run in a new session so the complete process group can be
    stopped and reaped. Windows uses a new process group and bounded best-effort
    cleanup of that group and the direct child; this is not a general Windows
    process-tree sandbox.
    """

    captured = bytearray()
    overflow = threading.Event()
    read_failed = threading.Event()
    stop_reader = threading.Event()
    closing_output = threading.Event()
    reader: threading.Thread | None = None
    timed_out = False
    rejected_overflow = False
    foreground_finished = False
    capture_unsettled = False
    try:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE if capture_output else subprocess.DEVNULL,
            stderr=subprocess.STDOUT if capture_output else subprocess.DEVNULL,
            start_new_session=os.name == "posix",
            creationflags=(
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                if os.name == "nt"
                else 0
            ),
            bufsize=0,
            text=False,
        )
    except OSError:
        raise BoundedCommandStartError from None

    try:
        if capture_output:
            if process.stdout is None:  # pragma: no cover - subprocess invariant
                raise BoundedCommandStartError
            if os.name == "posix":
                try:
                    os.set_blocking(process.stdout.fileno(), False)
                except OSError:
                    raise BoundedCommandReadError from None

            def drain_output() -> None:
                assert process.stdout is not None
                try:
                    with process.stdout:
                        while not stop_reader.is_set():
                            chunk = process.stdout.read(16 * 1024)
                            if chunk is None:
                                stop_reader.wait(0.01)
                                continue
                            if not chunk:
                                return
                            remaining = output_limit_bytes + 1 - len(captured)
                            if remaining > 0:
                                captured.extend(chunk[:remaining])
                            if len(captured) > output_limit_bytes:
                                overflow.set()
                except (OSError, ValueError):
                    if not closing_output.is_set():
                        read_failed.set()

            try:
                reader = threading.Thread(
                    target=drain_output,
                    name="coding-bounded-command-output",
                    daemon=True,
                )
                reader.start()
            except RuntimeError:
                raise BoundedCommandStartError from None

        deadline = time.monotonic() + timeout_s
        while process.poll() is None:
            if read_failed.is_set():
                break
            if reject_output_overflow and overflow.is_set():
                rejected_overflow = True
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            try:
                process.wait(timeout=min(remaining, 0.05))
            except subprocess.TimeoutExpired:
                continue
        foreground_finished = True
    finally:
        _terminate_owned_process_group(process)
        if reader is not None and reader.ident is not None:
            if (
                not foreground_finished
                or timed_out
                or rejected_overflow
                or read_failed.is_set()
            ):
                stop_reader.set()
            reader.join(timeout=1.0)
            if reader.is_alive():
                capture_unsettled = True
                stop_reader.set()
                reader.join(timeout=1.0)
            if reader.is_alive():
                assert process.stdout is not None
                closing_output.set()
                with suppress(OSError):
                    process.stdout.close()
                reader.join(timeout=1.0)

    if reject_output_overflow and overflow.is_set():
        rejected_overflow = True
    if read_failed.is_set():
        raise BoundedCommandReadError
    if timed_out:
        raise BoundedCommandTimeoutError
    if rejected_overflow:
        raise BoundedCommandOutputOverflowError
    if capture_unsettled or (reader is not None and reader.is_alive()):
        raise BoundedCommandTimeoutError
    if process.returncode is None:  # pragma: no cover - cleanup settles it
        raise BoundedCommandTimeoutError
    return BoundedCommandResult(
        returncode=process.returncode,
        output=bytes(captured[:output_limit_bytes]),
        output_truncated=overflow.is_set(),
    )


def _terminate_owned_process_group(process: subprocess.Popen[bytes]) -> None:
    """Stop the owned POSIX group or perform bounded platform cleanup."""

    if os.name == "posix":
        with suppress(OSError):
            os.killpg(process.pid, signal.SIGKILL)
    elif os.name == "nt":
        ctrl_break = getattr(signal, "CTRL_BREAK_EVENT", None)
        if ctrl_break is not None:
            with suppress(OSError, ValueError):
                os.kill(process.pid, ctrl_break)
        if process.poll() is None:
            with suppress(OSError):
                process.kill()
    elif process.poll() is None:  # pragma: no cover - supported platforms above
        with suppress(OSError):
            process.kill()
    if process.poll() is None:
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=1.0)
    if process.poll() is None:
        with suppress(OSError):
            process.kill()
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=1.0)
'''

_COMPOSITION_PY = r'''"""Explicit coding, knowledge, delegation, and human-input composition."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path

from cayu import (
    AgentSpec,
    AllRegisteredToolsExposurePolicy,
    ArtifactStore,
    BackgroundSubagentTaskRegistry,
    CayuApp,
    DeleteFileTool,
    DenyPatternRule,
    EditFileTool,
    Environment,
    EnvironmentSpec,
    ExecutionProfileBehaviorIdentity,
    GitChangesTool,
    KnowledgeAccessScope,
    KnowledgeStatus,
    KnowledgeStore,
    KnowledgeVisibility,
    ListArtifactsTool,
    ListFilesTool,
    ListKnowledgeTool,
    LocalArtifactStore,
    LocalRunner,
    LocalWorkspace,
    ModelProvider,
    ParameterConstrainedToolPolicy,
    ReadFileTool,
    ReadKnowledgeTool,
    RememberKnowledgeTool,
    RequiredAllowlistRule,
    RequiredFieldRule,
    RunLimits,
    SearchKnowledgeTool,
    SearchTextTool,
    SessionStore,
    SQLiteKnowledgeStore,
    SQLiteSessionStore,
    SQLiteTaskStore,
    SubagentExecutionMode,
    SubagentResultTool,
    SubagentSpec,
    SubagentTool,
    TaskStore,
    UserInputTool,
    WriteFileTool,
    public_authority_alias_codec_from_environment,
)
from command_probe import (
    BoundedCommandOutputOverflowError,
    BoundedCommandReadError,
    BoundedCommandResult,
    BoundedCommandStartError,
    BoundedCommandTimeoutError,
    run_bounded_command,
)

_PROJECT_ROOT = Path(__file__).resolve().parent
_STATE_ROOT = _PROJECT_ROOT / ".cayu" / "runtime"
_REVIEWER_ALIAS = "reviewer"
_PROTECTED_WORKSPACE_DIRECTORY_NAMES = (".cayu", ".git")
_SEARCH_EXCLUDED_DIRECTORIES = (
    ".cayu",
    ".git",
    ".hg",
    ".svn",
    ".cache",
    ".next",
    ".venv",
    "__pycache__",
    "build",
    "coverage",
    "dist",
    "node_modules",
    "target",
    "vendor",
    "venv",
)
_DENIED_PATH_PATTERNS = (
    r"^/",
    r"^[A-Za-z]:",
    r"(?:^|/)\.\.(?:/|$)",
    r"(?i)(?:^|[\\/])(?:\.cayu|\.git)(?:[\\/]|$)",
)
_COMMAND_TIMEOUT_S = 10.0
_COMMAND_OUTPUT_LIMIT_BYTES = 64 * 1024
_SAFE_LOCAL_ENV_KEYS = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "TZ",
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "COMSPEC",
    "PATHEXT",
    "TEMP",
    "TMP",
)
# These declarations authenticate editable application behavior across process
# reconstruction. Advance the named identity whenever the associated behavior
# changes; the generated README and AGENTS.md contain the complete mapping.
_SUBAGENT_RESULT_TOOL_IDENTITY = ExecutionProfileBehaviorIdentity(
    name="cayu.generated.coding.subagent_result",
    behavior_version="1",
    implementation_version="1",
)
_PRIMARY_TOOL_POLICY_IDENTITY = ExecutionProfileBehaviorIdentity(
    name="cayu.generated.coding.primary_tool_policy",
    behavior_version="1",
    implementation_version="1",
)


def _coding_environment_identity(
    *,
    root: Path,
    artifact_store: ArtifactStore,
    knowledge_store: KnowledgeStore,
    scope: KnowledgeAccessScope,
    generated_stores: bool,
) -> ExecutionProfileBehaviorIdentity | None:
    """Declare the generated environment/store wiring version."""

    if not generated_stores:
        return None
    if type(artifact_store) is not LocalArtifactStore:
        return None
    if type(knowledge_store) is not SQLiteKnowledgeStore:
        return None
    del root, scope
    return ExecutionProfileBehaviorIdentity(
        name="cayu.generated.coding.environment",
        behavior_version="2",
        implementation_version="1",
    )


def _subagent_tool_identity(
    reviewer_agent: AgentSpec,
    reviewer_identity: ExecutionProfileBehaviorIdentity | None,
    *,
    generated_session_store: bool,
) -> ExecutionProfileBehaviorIdentity | None:
    """Bind delegation settings to the reviewer behavior declaration."""

    if reviewer_identity is None or not generated_session_store:
        return None
    material = "\0".join(
        (
            reviewer_agent.name,
            reviewer_identity.model_dump_json(),
            _REVIEWER_ALIAS,
            SubagentExecutionMode.BACKGROUND.value,
            "max_steps=8",
            "result_max_chars=4000",
            "max_tool_calls=8",
            "max_elapsed_seconds=120",
        )
    ).encode("utf-8")
    return ExecutionProfileBehaviorIdentity(
        name="cayu.generated.coding.subagent",
        behavior_version="1",
        implementation_version=f"sha256:{sha256(material).hexdigest()}",
    )


def _command_environment() -> dict[str, str]:
    return {key: os.environ[key] for key in _SAFE_LOCAL_ENV_KEYS if key in os.environ}


def _git_command_environment(root: Path) -> dict[str, str]:
    environment = _command_environment()
    environment.update(
        {
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_CEILING_DIRECTORIES": str(root),
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_DISCOVERY_ACROSS_FILESYSTEM": "0",
            "GIT_OPTIONAL_LOCKS": "0",
            "LC_ALL": "C",
        }
    )
    return environment


def _execute_dependency_probe(
    argv: list[str],
    *,
    cwd: Path,
    environment: dict[str, str] | None = None,
    allowed_exit_codes: frozenset[int] = frozenset({0}),
    reject_output_overflow: bool = False,
    capture_output: bool = True,
) -> BoundedCommandResult:
    try:
        completed = run_bounded_command(
            argv,
            cwd=cwd,
            env=_command_environment() if environment is None else environment,
            timeout_s=_COMMAND_TIMEOUT_S,
            output_limit_bytes=_COMMAND_OUTPUT_LIMIT_BYTES,
            capture_output=capture_output,
            reject_output_overflow=reject_output_overflow,
        )
    except BoundedCommandStartError:
        raise RuntimeError(
            f"coding dependency {Path(argv[0]).name} could not start"
        ) from None
    except BoundedCommandTimeoutError:
        raise RuntimeError(
            f"coding dependency {Path(argv[0]).name} timed out"
        ) from None
    except BoundedCommandOutputOverflowError:
        raise RuntimeError(
            f"coding dependency {Path(argv[0]).name} produced excessive output"
        ) from None
    except BoundedCommandReadError:
        raise RuntimeError(
            f"coding dependency {Path(argv[0]).name} output could not be read"
        ) from None
    if completed.returncode not in allowed_exit_codes:
        raise RuntimeError(
            f"coding dependency {Path(argv[0]).name} is incompatible "
            f"(exit code {completed.returncode})"
        )
    return completed


def _run_dependency_probe(
    argv: list[str],
    *,
    cwd: Path,
    environment: dict[str, str] | None = None,
    allowed_exit_codes: frozenset[int] = frozenset({0}),
    reject_output_overflow: bool = False,
    capture_output: bool = True,
) -> str:
    completed = _execute_dependency_probe(
        argv,
        cwd=cwd,
        environment=environment,
        allowed_exit_codes=allowed_exit_codes,
        reject_output_overflow=reject_output_overflow,
        capture_output=capture_output,
    )
    return completed.output.decode("utf-8", errors="replace")


def _configured_command(command: str) -> str:
    executable = shutil.which(command)
    if executable is None:
        raise RuntimeError(f"coding composition requires command on PATH: {command}")
    return executable


def _portable_git_filter_name(value: str) -> bool:
    """Return whether a projected filter name can retain exact Git authority."""

    return (
        bool(value)
        and value.isascii()
        and value[0].isalnum()
        and value[-1].isalnum()
        and all(character.isalnum() or character in "._-" for character in value)
    )


def _configured_git_filter_names(output: str) -> tuple[str, ...]:
    names: set[str] = set()
    for key in output.split("\0"):
        if not key:
            continue
        # The exact query may return only executable filter keys. Treat any
        # transformed or incompatible record as ambiguous authority instead of
        # interpreting it as an unrelated key and silently leaving a filter on.
        if not key.startswith("filter."):
            raise RuntimeError(
                "coding workspace has a Git filter name that cannot be safely disabled"
            )
        driver, separator, field = key.removeprefix("filter.").rpartition(".")
        # This value is about to become command authority through ``git -c``.
        # Dependency output has crossed a UTF-8 projection boundary, so only a
        # complete expected key with a portable driver is exact enough to reissue.
        if (
            not separator
            or not _portable_git_filter_name(driver)
            or field not in {"clean", "smudge", "process"}
        ):
            raise RuntimeError(
                "coding workspace has a Git filter name that cannot be safely disabled"
            )
        names.add(driver)
    return tuple(sorted(names))


def _require_dependency_probe_evidence(
    output: str,
    *,
    command: str,
    required_fragments: tuple[str, ...],
) -> None:
    """Require positive, content-bound evidence from one dependency probe."""

    if any(fragment not in output for fragment in required_fragments):
        raise RuntimeError(f"coding dependency {command} semantic probe failed")


def _safe_git_probe_argv(
    git: str,
    *arguments: str,
    filter_names: tuple[str, ...] = (),
) -> list[str]:
    command = [
        git,
        "--no-pager",
        "-c",
        "core.fsmonitor=false",
        "-c",
        f"core.excludesFile={os.devnull}",
        "-c",
        f"core.attributesFile={os.devnull}",
    ]
    for driver in filter_names:
        command.extend(
            (
                "-c",
                f"filter.{driver}.clean=",
                "-c",
                f"filter.{driver}.smudge=",
                "-c",
                f"filter.{driver}.process=",
                "-c",
                f"filter.{driver}.required=false",
            )
        )
    command.extend(arguments)
    return command


def _verify_dependency_semantics(git: str, rg: str) -> None:
    """Exercise the exact Git and ripgrep dialects over known disposable data."""

    with tempfile.TemporaryDirectory(prefix="cayu-coding-dependency-probe-") as raw:
        probe = Path(raw)
        hooks = probe / "hooks"
        hooks.mkdir()
        git_environment = _git_command_environment(probe)
        _run_dependency_probe(
            _safe_git_probe_argv(
                git,
                "init",
                "-b",
                "main",
                f"--template={hooks}",
            ),
            cwd=probe,
            environment=git_environment,
        )
        (probe / "probe.txt").write_text(
            "cayu dependency baseline\n",
            encoding="utf-8",
        )
        _run_dependency_probe(
            _safe_git_probe_argv(git, "add", "--", "probe.txt"),
            cwd=probe,
            environment=git_environment,
        )
        (probe / "probe.txt").write_text(
            "cayu dependency changed\n",
            encoding="utf-8",
        )
        git_probes = (
            (
                ("status", "--porcelain=v1", "-z", "--untracked-files=normal", "--"),
                ("AM probe.txt\0",),
            ),
            (
                (
                    "diff",
                    "--no-color",
                    "--no-ext-diff",
                    "--no-textconv",
                    "--unified=3",
                    "--",
                ),
                ("-cayu dependency baseline", "+cayu dependency changed"),
            ),
            (
                (
                    "diff",
                    "--no-color",
                    "--no-ext-diff",
                    "--no-textconv",
                    "--cached",
                    "--unified=3",
                    "--",
                ),
                ("probe.txt", "+cayu dependency baseline"),
            ),
            (
                (
                    "diff",
                    "--no-color",
                    "--no-ext-diff",
                    "--no-textconv",
                    "--numstat",
                    "-z",
                    "--",
                ),
                ("1\t1\tprobe.txt\0",),
            ),
            (
                (
                    "diff",
                    "--no-color",
                    "--no-ext-diff",
                    "--no-textconv",
                    "--cached",
                    "--numstat",
                    "-z",
                    "--",
                ),
                ("1\t0\tprobe.txt\0",),
            ),
        )
        for arguments, required_fragments in git_probes:
            output = _run_dependency_probe(
                _safe_git_probe_argv(git, *arguments),
                cwd=probe,
                environment=git_environment,
            )
            _require_dependency_probe_evidence(
                output,
                command="git",
                required_fragments=required_fragments,
            )

        (probe / "search.txt").write_text(
            "cayu dependency needle\n",
            encoding="utf-8",
        )
        protected = probe / ".CaYu"
        protected.mkdir()
        (protected / "ignored.txt").write_text(
            "cayu dependency needle\n",
            encoding="utf-8",
        )
        command_environment = _command_environment()
        files_output = _run_dependency_probe(
            [
                rg,
                "--no-config",
                "--hidden",
                "--no-require-git",
                "--sort",
                "path",
                "--files",
                "--null",
                "--max-filesize",
                "1048576",
                "--iglob",
                "!.git",
                "--iglob",
                "!**/.git",
                "--iglob",
                "!.git/**",
                "--iglob",
                "!**/.git/**",
                "--iglob",
                "!.cayu",
                "--iglob",
                "!**/.cayu",
                "--iglob",
                "!.cayu/**",
                "--iglob",
                "!**/.cayu/**",
                "--",
                ".",
            ],
            cwd=probe,
            environment=command_environment,
        )
        _require_dependency_probe_evidence(
            files_output,
            command="rg",
            required_fragments=("probe.txt\0", "search.txt\0"),
        )
        if ".CaYu/ignored.txt" in files_output:
            raise RuntimeError("rg semantic probe failed")
        rg_probes = (
            (("--files-with-matches", "--null"), ("search.txt\0",)),
            (
                (
                    "--with-filename",
                    "--line-number",
                    "--null",
                    "--field-match-separator",
                    "|",
                    "--max-columns",
                    "1024",
                    "--max-columns-preview",
                ),
                ("search.txt", "cayu dependency needle"),
            ),
            (("--with-filename", "--count-matches", "--null"), ("search.txt", "1")),
        )
        for mode_arguments, required_fragments in rg_probes:
            output = _run_dependency_probe(
                [
                    rg,
                    "--no-config",
                    "--hidden",
                    "--no-require-git",
                    "--sort",
                    "path",
                    "--color",
                    "never",
                    "--max-filesize",
                    "1048576",
                    *mode_arguments,
                    "--ignore-case",
                    "--glob",
                    "search.txt",
                    "--iglob",
                    "!.git",
                    "--iglob",
                    "!**/.git",
                    "--iglob",
                    "!.git/**",
                    "--iglob",
                    "!**/.git/**",
                    "--iglob",
                    "!.cayu",
                    "--iglob",
                    "!**/.cayu",
                    "--iglob",
                    "!.cayu/**",
                    "--iglob",
                    "!**/.cayu/**",
                    "--",
                    "CAYU DEPENDENCY NEEDLE",
                    ".",
                ],
                cwd=probe,
                environment=command_environment,
            )
            _require_dependency_probe_evidence(
                output,
                command="rg",
                required_fragments=required_fragments,
            )
            if ".CaYu/ignored.txt" in output:
                raise RuntimeError("rg semantic probe failed")


def _verify_coding_dependencies(root: Path) -> None:
    git = _configured_command("git")
    rg = _configured_command("rg")
    git_environment = _git_command_environment(root)
    _run_dependency_probe([git, "--version"], cwd=root, environment=git_environment)
    _verify_dependency_semantics(git, rg)
    try:
        git_root = _run_dependency_probe(
            _safe_git_probe_argv(
                git,
                "rev-parse",
                "--show-toplevel",
            ),
            cwd=root,
            environment=git_environment,
        )
    except RuntimeError:
        raise RuntimeError(
            f"coding workspace must be a Git repository root: {root}"
        ) from None
    try:
        resolved_git_root = Path(git_root.strip()).resolve(strict=True)
    except (OSError, ValueError):
        raise RuntimeError(
            f"coding workspace must be a Git repository root: {root}"
        ) from None
    if resolved_git_root != root:
        raise RuntimeError(f"coding workspace must be a Git repository root: {root}")
    filter_result = _execute_dependency_probe(
        _safe_git_probe_argv(
            git,
            "config",
            "--includes",
            "--null",
            "--name-only",
            "--get-regexp",
            r"^filter\..*\.(clean|smudge|process)$",
        ),
        cwd=root,
        environment=git_environment,
        allowed_exit_codes=frozenset({0, 1}),
        reject_output_overflow=True,
    )
    filter_output = filter_result.output.decode("utf-8", errors="replace")
    filter_names = _configured_git_filter_names(filter_output)
    if (filter_result.returncode == 0 and not filter_names) or (
        filter_result.returncode == 1 and filter_names
    ):
        raise RuntimeError("coding dependency git filter inspection is inconsistent")
    _run_dependency_probe(
        _safe_git_probe_argv(
            git,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=normal",
            "--",
            filter_names=filter_names,
        ),
        cwd=root,
        environment=git_environment,
        capture_output=False,
    )
    for arguments in (
        (
            "diff",
            "--no-color",
            "--no-ext-diff",
            "--no-textconv",
            "--unified=3",
            "--",
        ),
        (
            "diff",
            "--no-color",
            "--no-ext-diff",
            "--no-textconv",
            "--cached",
            "--unified=3",
            "--",
        ),
        (
            "diff",
            "--no-color",
            "--no-ext-diff",
            "--no-textconv",
            "--numstat",
            "-z",
            "--",
        ),
        (
            "diff",
            "--no-color",
            "--no-ext-diff",
            "--no-textconv",
            "--cached",
            "--numstat",
            "-z",
            "--",
        ),
    ):
        _run_dependency_probe(
            _safe_git_probe_argv(
                git,
                *arguments,
                filter_names=filter_names,
            ),
            cwd=root,
            environment=git_environment,
            capture_output=False,
        )


def configured_workspace_root(override: str | os.PathLike[str] | None = None) -> Path:
    """Resolve and validate the trusted local Git workspace for this composition."""

    selected = override
    if selected is None:
        selected = os.environ.get("CAYU_WORKSPACE_ROOT", ".")
    candidate = Path(selected).expanduser()
    if not candidate.is_absolute():
        candidate = _PROJECT_ROOT / candidate
    try:
        root = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"coding workspace does not exist: {candidate}") from exc
    if not root.is_dir():
        raise RuntimeError(f"coding workspace is not a directory: {root}")
    if root == Path(root.anchor):
        raise RuntimeError("coding workspace cannot be a filesystem root")
    _verify_coding_dependencies(root)
    return root


def _knowledge_scope() -> KnowledgeAccessScope:
    return KnowledgeAccessScope(
        allowed_namespaces=["default"],
        allowed_statuses=[KnowledgeStatus.ACTIVE, KnowledgeStatus.PENDING],
    )


def _path_rules(*, required: bool) -> tuple:
    rules = []
    if required:
        rules.append(RequiredFieldRule("path"))
    rules.append(DenyPatternRule("path", patterns=_DENIED_PATH_PATTERNS))
    return tuple(rules)


def _primary_tool_policy() -> ParameterConstrainedToolPolicy:
    return ParameterConstrainedToolPolicy(
        {
            "search_text": (
                DenyPatternRule("path", patterns=_DENIED_PATH_PATTERNS),
                DenyPatternRule("glob", patterns=_DENIED_PATH_PATTERNS),
            ),
            "read_file": _path_rules(required=False),
            "write_file": _path_rules(required=True),
            "edit_file": _path_rules(required=True),
            "delete_file": _path_rules(required=True),
            "remember_knowledge": (RequiredFieldRule("text"),),
            "subagent": (
                RequiredAllowlistRule("agent", values=[_REVIEWER_ALIAS]),
                RequiredFieldRule("task"),
            ),
            "ask_user": (RequiredFieldRule("question"),),
        },
        execution_profile_identity=_PRIMARY_TOOL_POLICY_IDENTITY,
    )


def _require_coding_knowledge_scope(
    scope: KnowledgeAccessScope,
) -> KnowledgeAccessScope:
    if type(scope) is not KnowledgeAccessScope:
        raise RuntimeError("coding knowledge store returned an invalid access scope")
    scope = KnowledgeAccessScope(
        allowed_namespaces=list(scope.allowed_namespaces),
        allow_all_namespaces=scope.allow_all_namespaces,
        required_labels=dict(scope.required_labels),
        allowed_visibilities=list(scope.allowed_visibilities),
        allowed_source_types=(
            None
            if scope.allowed_source_types is None
            else list(scope.allowed_source_types)
        ),
        allowed_source_ids=(
            None if scope.allowed_source_ids is None else list(scope.allowed_source_ids)
        ),
        allowed_statuses=list(scope.allowed_statuses),
        include_expired=scope.include_expired,
    )
    problems: list[str] = []
    if not scope.allow_all_namespaces and "default" not in scope.allowed_namespaces:
        problems.append("namespace 'default'")
    if KnowledgeStatus.ACTIVE not in scope.allowed_statuses:
        problems.append("status 'active'")
    if KnowledgeStatus.PENDING not in scope.allowed_statuses:
        problems.append("status 'pending'")
    if KnowledgeVisibility.GLOBAL not in scope.allowed_visibilities:
        problems.append("visibility 'global'")
    if scope.required_labels:
        problems.append("unlabelled generated writes")
    if (
        scope.allowed_source_types is not None
        and "tool" not in scope.allowed_source_types
    ):
        problems.append("source type 'tool'")
    if scope.allowed_source_ids is not None:
        problems.append("dynamic session source IDs")
    if problems:
        raise RuntimeError(
            "coding knowledge store scope is incompatible with: " + ", ".join(problems)
        )
    return scope


def build_coding_app(
    *,
    primary_agent: AgentSpec,
    reviewer_agent: AgentSpec,
    reviewer_execution_profile_identity: ExecutionProfileBehaviorIdentity | None,
    configured_provider: Callable[[], ModelProvider],
    provider: ModelProvider | None = None,
    session_store: SessionStore | None = None,
    task_store: TaskStore | None = None,
    workspace_root: str | os.PathLike[str] | None = None,
    artifact_store: ArtifactStore | None = None,
    knowledge_store: KnowledgeStore | None = None,
) -> CayuApp:
    """Build one fresh, process-scoped coding composition."""

    LocalWorkspace.require_path_operations_supported()
    root = configured_workspace_root(workspace_root)
    scope = _knowledge_scope()
    generated_session_store = session_store is None
    selected_session_store = (
        session_store
        if session_store is not None
        else SQLiteSessionStore(
            _STATE_ROOT / "cayu.db",
            public_authority_alias_codec=public_authority_alias_codec_from_environment(),
        )
    )
    selected_task_store = (
        task_store
        if task_store is not None
        else SQLiteTaskStore(_STATE_ROOT / "cayu.db")
    )
    selected_knowledge_store = (
        knowledge_store
        if knowledge_store is not None
        else SQLiteKnowledgeStore(_STATE_ROOT / "cayu.db", access_scope=scope)
    )
    bound_scope = selected_knowledge_store.bound_access_scope()
    selected_scope = _require_coding_knowledge_scope(
        scope if bound_scope is None else bound_scope
    )
    selected_artifact_store = (
        artifact_store
        if artifact_store is not None
        else LocalArtifactStore(_STATE_ROOT / "artifacts", store_id="coding-artifacts")
    )
    environment_identity = _coding_environment_identity(
        root=root,
        artifact_store=selected_artifact_store,
        knowledge_store=selected_knowledge_store,
        scope=selected_scope,
        generated_stores=artifact_store is None and knowledge_store is None,
    )

    app = CayuApp(
        session_store=selected_session_store,
        task_store=selected_task_store,
        knowledge_store=selected_knowledge_store,
        knowledge_access_scope=selected_scope,
        knowledge_review_namespace="default",
    )
    selected_provider = provider if provider is not None else configured_provider()
    app.register_provider(selected_provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(
                name="coding",
                execution_profile_identity=environment_identity,
            ),
            workspace=LocalWorkspace(
                root,
                workspace_id="coding-workspace",
                excluded_directory_names=_PROTECTED_WORKSPACE_DIRECTORY_NAMES,
            ),
            runner=LocalRunner(root, inherit_env=False),
            artifact_store=selected_artifact_store,
            knowledge_store=selected_knowledge_store,
            knowledge_access_scope=selected_scope,
        ),
        default=True,
    )
    app.register_agent(reviewer_agent, tools=())

    background_registry = BackgroundSubagentTaskRegistry()
    tools = (
        ListFilesTool(),
        SearchTextTool(
            exclude_directories=_SEARCH_EXCLUDED_DIRECTORIES,
            protected_entry_names=_PROTECTED_WORKSPACE_DIRECTORY_NAMES,
        ),
        ReadFileTool(),
        WriteFileTool(),
        EditFileTool(),
        DeleteFileTool(),
        GitChangesTool(),
        ListArtifactsTool(),
        ListKnowledgeTool(),
        SearchKnowledgeTool(),
        ReadKnowledgeTool(),
        RememberKnowledgeTool(),
        SubagentTool(
            app,
            agents={
                _REVIEWER_ALIAS: SubagentSpec(
                    agent_name=reviewer_agent.name,
                    description="Review a bounded change and return concrete findings.",
                    mode=SubagentExecutionMode.BACKGROUND,
                    max_steps=8,
                    result_max_chars=4_000,
                    limits=RunLimits(
                        max_tool_calls=8,
                        max_elapsed_seconds=120,
                    ),
                )
            },
            background_registry=background_registry,
            execution_profile_identity=_subagent_tool_identity(
                reviewer_agent,
                reviewer_execution_profile_identity,
                generated_session_store=generated_session_store,
            ),
        ),
        SubagentResultTool(
            app.session_store,
            background_registry=background_registry,
            default_timeout_s=30,
            execution_profile_identity=(
                _SUBAGENT_RESULT_TOOL_IDENTITY if generated_session_store else None
            ),
        ),
        UserInputTool(),
    )
    app.register_agent(
        primary_agent,
        tools=tools,
        tool_exposure_policy=AllRegisteredToolsExposurePolicy(),
        tool_policy=_primary_tool_policy(),
    )
    return app
'''


_PRIMARY_AGENT_PY = '''"""Primary coding agent for __PROJECT_NAME__."""

from cayu import AgentSpec

from configuration import configured_model, configured_provider_name

AGENT = AgentSpec(
    name="__AGENT_NAME__",
    model=configured_model(),
    provider_name=configured_provider_name(),
    system_prompt="""You are the primary coding agent for this repository.

Work only through the registered, bounded tools. Inspect before editing, keep
changes inside the configured Git workspace, and use git_changes to review your
work. Durable knowledge writes are proposals pending review. Delegate focused
review tasks to the reviewer alias in the background and recover their result
with subagent_result. Use ask_user when a material choice cannot be inferred.
""",
)
'''


_REVIEWER_AGENT_PY = '''"""Bounded reviewer subagent for __PROJECT_NAME__."""

from configuration import configured_model, configured_provider_name

from cayu import AgentSpec, ExecutionProfileBehaviorIdentity

REVIEWER = AgentSpec(
    name="__REVIEWER_NAME__",
    model=configured_model(),
    provider_name=configured_provider_name(),
    system_prompt=(
        "Review only the delegated context. Return concise correctness, testing, "
        "and safety findings; do not modify files or delegate again."
    ),
)

# Advance this identity when the reviewer prompt, policy, tools, hooks, or other
# behavior changes. Its value also contributes to the primary subagent tool's
# reconstructed execution identity.
REVIEWER_EXECUTION_PROFILE_IDENTITY = ExecutionProfileBehaviorIdentity(
    name="__PROJECT_NAME__.coding_reviewer",
    behavior_version="1",
    implementation_version="1",
)
'''


_APP_BUILD = '''def build_app(
    *,
    provider: ModelProvider | None = None,
    session_store: SessionStore | None = None,
    task_store: TaskStore | None = None,
    workspace_root=None,
    artifact_store=None,
    knowledge_store=None,
) -> CayuApp:
    """Construct the explicit coding composition for one process."""

    return build_coding_app(
        primary_agent=_agent_for_provider_override(AGENT, provider),
        reviewer_agent=_agent_for_provider_override(REVIEWER, provider),
        reviewer_execution_profile_identity=REVIEWER_EXECUTION_PROFILE_IDENTITY,
        configured_provider=configured_provider,
        provider=provider,
        session_store=session_store,
        task_store=task_store,
        workspace_root=workspace_root,
        artifact_store=artifact_store,
        knowledge_store=knowledge_store,
    )
'''


_README_APPEND = """

## Maintained coding composition

This project opts in to Cayu's explicit coding starter. `composition.py` is the
ordinary editable assembly point: it registers bounded repository file tools,
Git review, local artifacts, durable SQLite knowledge, a background reviewer
subagent with result recovery, and human input. These are existing Cayu APIs;
there is no hidden agent kind, registry, permission grant, or post-start mutation.
The composition selects implementations only. `AllRegisteredToolsExposurePolicy`
separately controls which registered tools are model-visible, while the ordinary
tool policy, approval policy, and runtime gates independently authorize calls.
`command_probe.py` is project-owned standard-library support for bounded Git and
ripgrep compatibility checks; it does not depend on a private Cayu API or grant
tool authority.

The workspace defaults to this Git repository root. Override it with a path
relative to this project (or an absolute path) using `CAYU_WORKSPACE_ROOT`. The
selected path must already exist, must be a Git repository root, and cannot be a
filesystem root. Both `git` and `rg` must be on `PATH`. Repository-control
`.git` directories and runtime-private `.cayu` directories are excluded from
generic workspace file and search tools. Session, task, artifact, and knowledge
state is stored below that protected `.cayu` boundary; use the registered Git,
artifact, and knowledge tools at their authenticated boundaries instead.

`LocalWorkspace` and `LocalRunner` are trusted-host development adapters, not a
sandbox. This composition requires the POSIX descriptor-relative filesystem
primitives used by secure `LocalWorkspace` path operations and rejects an
unsupported host during generation or application construction. The runner does
not inherit the full or arbitrary ambient environment; with `inherit_env=False`
it forwards only Cayu's minimal operational allow-list
(including command-resolution, home, locale, and temporary-directory variables).
Path-addressed mutations are confined to the selected workspace by the workspace
adapter and explicit parameter policy. Knowledge writes remain pending until a
human reviews them.

Completed reviewer sessions and their results are durable and can be retrieved
after application reconstruction. Background reviewer execution itself belongs
to the current process: let it reach a terminal state before restarting, or
replace it with durable dispatch and a task-store worker when in-flight
cross-process recovery is required.

The generated application identities are manual version assertions, not source
hashes. When editing authorization or reconstructed behavior, advance the paired
identity in the same change:

- changes to `_DENIED_PATH_PATTERNS` or `_primary_tool_policy()` require advancing
  `_PRIMARY_TOOL_POLICY_IDENTITY`;
- changes to `SubagentResultTool` construction require advancing
  `_SUBAGENT_RESULT_TOOL_IDENTITY`;
- changes to `_PROTECTED_WORKSPACE_DIRECTORY_NAMES`,
  `_SEARCH_EXCLUDED_DIRECTORIES`, or other generated environment, workspace,
  runner, artifact, knowledge, or store wiring require advancing the identity returned by
  `_coding_environment_identity()`;
- changes to reviewer behavior require advancing
  `REVIEWER_EXECUTION_PROFILE_IDENTITY`; and
- changes to delegation limits or aliases must also update the material in
  `_subagent_tool_identity()`.

Failing to advance an application identity can allow a durable continuation to
resume under behavior different from the behavior that identity originally
authenticated. Advancing one intentionally makes older continuations fail
closed until the application explicitly adopts the new execution profile.

Because the project registers a primary agent and reviewer, live runs must select
the primary agent explicitly:

```bash
uv run python run.py --agent __AGENT_NAME__ --message "YOUR REQUEST"
```

Run the credential-free composition proof with:

```bash
uv run pytest -q tests/test_coding_composition.py
```
"""


_AGENTS_APPEND = """

## Maintained coding composition

Keep `composition.py` explicit. Do not replace it with an agent-type switch,
plugin registry, implicit permission grant, or runtime mutation. Preserve the
Git-root validation, `git`/`rg` compatibility preflight, minimal-environment local
runner, parameter policy, pending knowledge review, bounded background reviewer,
result tool, and human-input pause/resume contract.
Keep `.git` and runtime-private `.cayu` directories excluded at both the
workspace and search boundaries. Do not replace artifact or knowledge tools
with generic file access to their backing stores.
Keep `command_probe.py` project-owned and bounded; do not replace it with an
import from Cayu's private modules or an unbounded subprocess helper.

Treat the generated execution-profile identities as part of each editable
behavior's contract. Advance `_PRIMARY_TOOL_POLICY_IDENTITY` with deny-pattern or
primary-policy changes, `_SUBAGENT_RESULT_TOOL_IDENTITY` with result-tool changes,
the identity returned by `_coding_environment_identity()` with protected-directory,
environment, or store wiring changes, and `REVIEWER_EXECUTION_PROFILE_IDENTITY` with reviewer
behavior changes. Keep delegation aliases and limits represented in
`_subagent_tool_identity()`. An unchanged identity asserts unchanged behavior
across process reconstruction; changing it intentionally makes stale durable
continuations fail closed.

Run `uv run pytest -q tests/test_coding_composition.py` after composition changes.
Use `--agent __AGENT_NAME__` for live runs because the reviewer is also registered.
"""


_SMOKE_TEST_PY = r'''"""Credential-free smoke proof for the maintained coding composition."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
from collections.abc import AsyncIterator
from contextlib import suppress
from pathlib import Path

import command_probe
import composition
import pytest
import run as project_run
from app import build_app

from cayu import (
    AgentSpec,
    ArtifactScope,
    EventType,
    ExecutionProfileBehaviorIdentity,
    InMemoryKnowledgeStore,
    InMemorySessionStore,
    InMemoryTaskStore,
    KnowledgeAccessScope,
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeListQuery,
    KnowledgeStatus,
    LocalArtifactStore,
    Message,
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    RunRequest,
    SessionQuery,
    SessionStatus,
    ScriptedModelProvider,
    SQLiteKnowledgeStore,
    UserInputResponse,
)
from command_probe import BoundedCommandResult

_ARTIFACT_ID = "art_11111111111111111111111111111111"


class _CompositionProvider(ModelProvider):
    name = "composition-smoke"

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="cayu.generated.coding.smoke_provider",
            behavior_version="1",
            implementation_version="1",
        )

    def __init__(self) -> None:
        self.primary_step = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        if not request.tools:
            yield ModelStreamEvent.text_delta("Reviewer found no blocking issue.")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
            return

        calls = (
            ("list_files", {}),
            ("search_text", {"pattern": "gitdir", "path": ".", "mode": "files"}),
            ("read_file", {"path": ".cayu/runtime/private.txt"}),
            (
                "write_file",
                {
                    "path": ".git/unauthorized.txt",
                    "content": "must not be written\n",
                    "mode": "create",
                },
            ),
            ("read_file", {"path": "pyproject.toml"}),
            (
                "write_file",
                {
                    "path": "smoke_output.txt",
                    "content": "composition proof\n",
                    "mode": "create",
                },
            ),
            ("git_changes", {"mode": "diff", "scope": "unstaged"}),
            ("list_artifacts", {"scope": "session"}),
            ("read_file", {"artifact_id": _ARTIFACT_ID}),
            ("list_knowledge", {"include_entries": True}),
            ("search_knowledge", {"query": "stable"}),
            ("read_knowledge", {"entry_id": "coding-guide"}),
            (
                "remember_knowledge",
                {"text": "Review generated changes before committing."},
            ),
            (
                "subagent",
                {
                    "agent": "reviewer",
                    "task": (
                        "Review smoke_output.txt containing 'composition proof' and "
                        "report whether that bounded change has a blocking issue."
                    ),
                },
            ),
            ("ask_user", {"question": "Accept the deterministic smoke result?"}),
            ("subagent_result", {"all": True, "wait": True}),
        )
        if self.primary_step < len(calls):
            name, arguments = calls[self.primary_step]
            call_id = f"call_{self.primary_step}"
            self.primary_step += 1
            yield ModelStreamEvent.tool_call(id=call_id, name=name, arguments=arguments)
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        yield ModelStreamEvent.text_delta("Coding composition smoke complete.")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


def _git(*args: str, cwd: Path) -> str:
    git = composition._configured_command("git")
    return composition._run_dependency_probe(
        composition._safe_git_probe_argv(git, *args),
        cwd=cwd,
        environment=composition._git_command_environment(cwd),
    )


async def _run_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "pyproject.toml").write_text(
        '[project]\nname = "smoke-project"\nversion = "0.0.0"\n',
        encoding="utf-8",
    )
    _git("init", "-b", "main", cwd=workspace)
    _git("add", ".", cwd=workspace)
    _git(
        "-c",
        "user.name=Cayu Smoke",
        "-c",
        "user.email=smoke@cayu.local",
        "commit",
        "-m",
        "baseline",
        cwd=workspace,
    )
    ordinary_data = workspace / "data"
    ordinary_data.mkdir()
    ordinary_data_file = ordinary_data / "project.txt"
    ordinary_data_file.write_text(
        "ordinary project gitdir guidance remains model-visible\n",
        encoding="utf-8",
    )
    linked_worktree = workspace / "linked-worktree"
    linked_worktree.mkdir()
    (linked_worktree / ".GiT").write_text(
        "gitdir: ../.git\n",
        encoding="utf-8",
    )
    mixed_case_state = workspace / "nested-state" / ".CaYu"
    mixed_case_state.mkdir(parents=True)
    (mixed_case_state / "private-project.txt").write_text(
        "mixed-case private gitdir state must not be model-visible\n",
        encoding="utf-8",
    )
    git_head_before = (workspace / ".git" / "HEAD").read_bytes()

    scope = KnowledgeAccessScope(
        allowed_namespaces=["default"],
        allowed_statuses=[KnowledgeStatus.ACTIVE, KnowledgeStatus.PENDING],
    )
    private_state_root = workspace / ".cayu" / "runtime"
    monkeypatch.setattr(composition, "_STATE_ROOT", private_state_root)
    database = private_state_root / "cayu.db"
    knowledge_store = SQLiteKnowledgeStore(database, access_scope=scope)
    await knowledge_store.create_entry(
        KnowledgeEntry(id="coding-guide", text="Stable project guidance."),
        [
            KnowledgeChunk(
                id="coding-guide-0",
                entry_id="coding-guide",
                text="Stable project guidance.",
                chunk_index=0,
            )
        ],
    )
    artifact_root = private_state_root / "artifacts"
    artifact_store = LocalArtifactStore(artifact_root, store_id="smoke-artifacts")
    session_id = "coding-smoke"
    await artifact_store.put_bytes(
        b"artifact context\n",
        artifact_id=_ARTIFACT_ID,
        filename="context.txt",
        content_type="text/plain",
        scope=ArtifactScope.SESSION,
        session_id=session_id,
        agent_name="__AGENT_NAME__",
    )
    private_canary = private_state_root / "private.txt"
    private_canary.write_text(
        "private project state must not be model-visible\n",
        encoding="utf-8",
    )
    provider = _CompositionProvider()
    app = build_app(
        provider=provider,
        workspace_root=workspace,
    )

    pause_events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="__AGENT_NAME__",
                session_id=session_id,
                messages=[Message.text("user", "Exercise the maintained composition.")],
            )
        )
    ]
    awaiting = next(
        event
        for event in pause_events
        if event.type == EventType.SESSION_AWAITING_USER_INPUT
    )
    assert pause_events[-1].type == EventType.SESSION_INTERRUPTED
    for _ in range(100):
        children = (
            await app.session_store.list_sessions(
                SessionQuery(parent_session_id=session_id)
            )
        ).sessions
        if len(children) == 1 and children[0].status == SessionStatus.COMPLETED:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("background reviewer did not become terminal before reconstruction")

    reconstructed_provider = _CompositionProvider()
    reconstructed_provider.primary_step = provider.primary_step
    reconstructed = build_app(
        provider=reconstructed_provider,
        workspace_root=workspace,
    )
    resume_events = [
        event
        async for event in reconstructed.resolve_user_input(
            UserInputResponse(
                session_id=session_id,
                input_id=awaiting.payload["input_id"],
                answer="yes",
            )
        )
    ]

    events = [*pause_events, *resume_events]
    completed = [
        event for event in events if event.type == EventType.TOOL_CALL_COMPLETED
    ]
    blocked = [event for event in events if event.type == EventType.TOOL_CALL_BLOCKED]
    completed_tools = {event.tool_name for event in completed}
    assert {
        "list_files",
        "search_text",
        "read_file",
        "write_file",
        "git_changes",
        "list_artifacts",
        "list_knowledge",
        "search_knowledge",
        "read_knowledge",
        "remember_knowledge",
        "subagent",
        "subagent_result",
        "ask_user",
    } <= completed_tools
    assert [event.tool_name for event in blocked] == ["read_file", "write_file"]
    assert all(event.payload["result"]["is_error"] is False for event in completed)
    results: dict[str, list[dict]] = {}
    for event in completed:
        results.setdefault(event.tool_name, []).append(event.payload["result"])
    assert "pyproject.toml" in results["list_files"][0]["content"]
    assert ".git" not in results["list_files"][0]["content"]
    assert ".cayu" not in results["list_files"][0]["content"]
    assert "data/project.txt" in results["list_files"][0]["content"]
    assert "data/project.txt" in results["search_text"][0]["content"]
    assert "linked-worktree/.GiT" not in results["search_text"][0]["content"]
    assert "private-project.txt" not in results["search_text"][0]["content"]
    assert "private.txt" not in results["search_text"][0]["content"]
    assert any(
        result["content"] == "artifact context\n" for result in results["read_file"]
    )
    assert "smoke_output.txt" in results["git_changes"][0]["content"]
    assert _ARTIFACT_ID in {
        artifact["artifact_id"]
        for artifact in results["list_artifacts"][0]["structured"]["artifacts"]
    }
    assert "Stable project guidance." in results["list_knowledge"][0]["content"]
    assert "Stable project guidance." in results["search_knowledge"][0]["content"]
    assert "Stable project guidance." in results["read_knowledge"][0]["content"]
    remembered = results["remember_knowledge"][0]["structured"]
    assert remembered["status"] == "pending"
    assert remembered["written"] is True
    delegated = results["subagent"][0]["structured"]
    assert delegated["mode"] == "background"
    reviewer_result = results["subagent_result"][0]["structured"]
    assert reviewer_result["retrieval_status"] == "ready"
    assert len(reviewer_result["children"]) == 1
    reviewer = reviewer_result["children"][0]
    assert reviewer["status"] == "completed"
    assert reviewer["result_text"] == "Reviewer found no blocking issue."
    assert results["ask_user"][0]["content"] == "yes"
    assert resume_events[-1].type == EventType.SESSION_COMPLETED
    assert (workspace / "smoke_output.txt").read_text(
        encoding="utf-8"
    ) == "composition proof\n"
    assert ordinary_data_file.read_text(encoding="utf-8") == (
        "ordinary project gitdir guidance remains model-visible\n"
    )
    assert private_canary.read_text(encoding="utf-8") == (
        "private project state must not be model-visible\n"
    )
    assert not (workspace / ".git" / "unauthorized.txt").exists()
    assert (workspace / ".git" / "HEAD").read_bytes() == git_head_before
    assert private_state_root.is_dir()
    assert "smoke_output.txt" in _git("status", "--short", cwd=workspace)
    pending = await knowledge_store.list_entries(
        KnowledgeListQuery(statuses=[KnowledgeStatus.PENDING])
    )
    assert len(pending.entries) == 1


def test_coding_composition_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asyncio.run(_run_smoke(tmp_path, monkeypatch))


def test_documented_run_entrypoint_from_generated_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace = tmp_path / "entrypoint-workspace"
    workspace.mkdir()
    _git("init", "-b", "main", cwd=workspace)
    monkeypatch.setattr(composition, "_STATE_ROOT", workspace / ".cayu" / "runtime")
    provider = ScriptedModelProvider(
        [
            ModelStreamEvent.text_delta("Generated entrypoint complete."),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = build_app(provider=provider, workspace_root=workspace)
    build_calls = 0
    validated_agents: list[str] = []
    validate_run_configuration = project_run.validate_run_configuration

    def app_factory():
        nonlocal build_calls
        build_calls += 1
        return app

    def validate_run(app_to_validate, agent_name: str) -> None:
        validated_agents.append(agent_name)
        validate_run_configuration(app_to_validate, agent_name)

    monkeypatch.setattr(project_run, "build_app", app_factory)
    monkeypatch.setattr(project_run, "validate_run_configuration", validate_run)

    result = project_run.main(
        [
            "--agent",
            "__AGENT_NAME__",
            "--message",
            "Run the coding agent.",
        ]
    )

    assert result == 0
    assert build_calls == 1
    assert validated_agents == ["__AGENT_NAME__"]
    assert len(provider.requests) == 1
    assert provider.requests[0].messages[-1].content[0].text == (
        "Run the coding agent."
    )
    captured = capsys.readouterr()
    assert captured.out == "Generated entrypoint complete.\n"
    assert captured.err == ""


def test_coding_workspace_contract_is_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="filesystem root"):
        composition.configured_workspace_root(Path(Path.cwd().anchor))
    non_repo = tmp_path / "not-a-repository"
    non_repo.mkdir()
    with pytest.raises(RuntimeError, match="Git repository root"):
        composition.configured_workspace_root(non_repo)

    relative_repo = tmp_path / "relative-workspace"
    relative_repo.mkdir()
    _git("init", "-b", "main", cwd=relative_repo)
    monkeypatch.setattr(composition, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("CAYU_WORKSPACE_ROOT", "relative-workspace")
    assert composition.configured_workspace_root() == relative_repo.resolve()

    real_which = composition.shutil.which
    monkeypatch.setattr(
        composition.shutil,
        "which",
        lambda command: "/missing/rg" if command == "rg" else real_which(command),
    )
    with pytest.raises(RuntimeError, match="rg could not start"):
        composition.configured_workspace_root(relative_repo)


def test_coding_app_construction_rejects_unsupported_workspace_operations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "unsupported-workspace"
    workspace.mkdir()
    _git("init", "-b", "main", cwd=workspace)
    private_state_root = workspace / ".cayu" / "runtime"
    monkeypatch.setattr(composition, "_STATE_ROOT", private_state_root)

    def unsupported() -> None:
        raise RuntimeError(
            "LocalWorkspace requires POSIX descriptor-relative filesystem primitives."
        )

    monkeypatch.setattr(
        composition.LocalWorkspace,
        "require_path_operations_supported",
        staticmethod(unsupported),
    )
    provider = _CompositionProvider()

    with pytest.raises(RuntimeError, match="POSIX descriptor-relative"):
        build_app(
            provider=provider,
            session_store=InMemorySessionStore(),
            task_store=InMemoryTaskStore(),
            workspace_root=workspace,
        )

    assert provider.primary_step == 0
    assert not private_state_root.exists()


@pytest.mark.parametrize(
    ("command", "unsupported_flag"),
    [
        ("git", "--cached"),
        ("git", "--numstat"),
        ("rg", "--files-with-matches"),
        ("rg", "--count-matches"),
        ("rg", "--glob"),
        ("rg", "--ignore-case"),
    ],
)
def test_coding_workspace_rejects_incompatible_runtime_command_dialects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    unsupported_flag: str,
) -> None:
    repository = tmp_path / "incompatible-command-repository"
    repository.mkdir()
    _git("init", "-b", "main", cwd=repository)
    original = composition.run_bounded_command

    def incompatible(argv, **kwargs):
        if Path(argv[0]).name == command and unsupported_flag in argv:
            return BoundedCommandResult(
                returncode=7,
                output=b"",
                output_truncated=False,
            )
        return original(argv, **kwargs)

    monkeypatch.setattr(composition, "run_bounded_command", incompatible)
    with pytest.raises(RuntimeError, match=rf"{command} is incompatible"):
        composition.configured_workspace_root(repository)


@pytest.mark.parametrize(
    ("command", "semantic_argument"),
    [
        ("git", "status"),
        ("rg", "--files"),
    ],
)
def test_coding_workspace_rejects_false_success_dependency_shims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    semantic_argument: str,
) -> None:
    repository = tmp_path / f"false-success-{command}-repository"
    repository.mkdir()
    _git("init", "-b", "main", cwd=repository)
    original = composition.run_bounded_command

    def false_success(argv, **kwargs):
        if Path(argv[0]).name == command and semantic_argument in argv:
            return BoundedCommandResult(
                returncode=0,
                output=b"",
                output_truncated=False,
            )
        return original(argv, **kwargs)

    monkeypatch.setattr(composition, "run_bounded_command", false_success)
    with pytest.raises(RuntimeError, match=rf"{command} semantic probe failed"):
        composition.configured_workspace_root(repository)


@pytest.mark.skipif(os.name == "nt", reason="POSIX filter-execution regression")
def test_coding_dependency_probe_disables_repository_filters(tmp_path: Path) -> None:
    repository = tmp_path / "filter-repository"
    repository.mkdir()
    _git("init", "-b", "main", cwd=repository)
    (repository / ".gitattributes").write_text(
        "*.txt filter=hostile\n", encoding="utf-8"
    )
    (repository / "tracked.txt").write_text("before\n", encoding="utf-8")
    _git("add", ".", cwd=repository)
    _git(
        "-c",
        "user.name=Cayu Smoke",
        "-c",
        "user.email=smoke@cayu.local",
        "commit",
        "-m",
        "baseline",
        cwd=repository,
    )
    marker = tmp_path / "filter-ran"
    filter_command = tmp_path / "hostile-filter"
    filter_command.write_text(
        f"#!/bin/sh\ntouch {marker}\ncat\n",
        encoding="utf-8",
    )
    filter_command.chmod(0o755)
    included_config = tmp_path / "included-filter-config"
    included_config.write_text(
        (f'[filter "hostile"]\n\tclean = {filter_command}\n\trequired = true\n'),
        encoding="utf-8",
    )
    _git("config", "include.path", str(included_config), cwd=repository)
    (repository / "tracked.txt").write_text("after\n", encoding="utf-8")

    assert composition.configured_workspace_root(repository) == repository.resolve()
    assert not marker.exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX filter-execution regression")
def test_coding_dependency_probe_disables_worktree_filters(tmp_path: Path) -> None:
    repository = tmp_path / "worktree-filter-repository"
    repository.mkdir()
    _git("init", "-b", "main", cwd=repository)
    (repository / ".gitattributes").write_text(
        "*.txt filter=hostile\n", encoding="utf-8"
    )
    (repository / "tracked.txt").write_text("before\n", encoding="utf-8")
    _git("add", ".", cwd=repository)
    _git(
        "-c",
        "user.name=Cayu Smoke",
        "-c",
        "user.email=smoke@cayu.local",
        "commit",
        "-m",
        "baseline",
        cwd=repository,
    )
    marker = tmp_path / "worktree-filter-ran"
    filter_command = tmp_path / "hostile-worktree-filter"
    filter_command.write_text(
        f"#!/bin/sh\ntouch {marker}\ncat\n",
        encoding="utf-8",
    )
    filter_command.chmod(0o755)
    _git("config", "extensions.worktreeConfig", "true", cwd=repository)
    _git(
        "config",
        "--worktree",
        "filter.hostile.clean",
        str(filter_command),
        cwd=repository,
    )
    _git(
        "config",
        "--worktree",
        "filter.hostile.required",
        "true",
        cwd=repository,
    )
    (repository / "tracked.txt").write_text("after\n", encoding="utf-8")

    assert composition.configured_workspace_root(repository) == repository.resolve()
    assert not marker.exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX filter-execution regression")
def test_coding_dependency_probe_rejects_ambiguous_filter_names(tmp_path: Path) -> None:
    repository = tmp_path / "ambiguous-filter-repository"
    repository.mkdir()
    _git("init", "-b", "main", cwd=repository)
    (repository / ".gitattributes").write_text(
        "*.txt filter=hostile=driver\n", encoding="utf-8"
    )
    (repository / "tracked.txt").write_text("before\n", encoding="utf-8")
    _git("add", ".", cwd=repository)
    _git(
        "-c",
        "user.name=Cayu Smoke",
        "-c",
        "user.email=smoke@cayu.local",
        "commit",
        "-m",
        "baseline",
        cwd=repository,
    )
    marker = tmp_path / "ambiguous-filter-ran"
    filter_command = tmp_path / "hostile-ambiguous-filter"
    filter_command.write_text(
        f"#!/bin/sh\ntouch {marker}\ncat\n",
        encoding="utf-8",
    )
    filter_command.chmod(0o755)
    _git(
        "config",
        "filter.hostile=driver.clean",
        str(filter_command),
        cwd=repository,
    )
    _git(
        "config",
        "filter.hostile=driver.required",
        "true",
        cwd=repository,
    )
    (repository / "tracked.txt").write_text("after\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="cannot be safely disabled"):
        composition.configured_workspace_root(repository)
    assert not marker.exists()


@pytest.mark.parametrize(
    "projected_name",
    [
        "hostile\ufffdname",
        "[REDACTED_SECRET]",
        "hostile [REDACTED_SECRET] name",
        "hostile/name",
        ".hostile",
        "hostile.",
    ],
)
def test_coding_dependency_probe_rejects_nonportable_filter_authority(
    projected_name: str,
) -> None:
    with pytest.raises(RuntimeError, match="cannot be safely disabled"):
        composition._configured_git_filter_names(f"filter.{projected_name}.clean\0")


@pytest.mark.parametrize(
    "record",
    [
        "[REDACTED_SECRET].hostile.clean",
        "filter.hostile.[REDACTED_SECRET]",
        "unrelated.hostile.clean",
        "filter.hostile.required",
    ],
)
def test_coding_dependency_probe_rejects_malformed_filter_records(
    record: str,
) -> None:
    with pytest.raises(RuntimeError, match="cannot be safely disabled"):
        composition._configured_git_filter_names(f"{record}\0")


def test_coding_workspace_rejects_invalid_utf8_filter_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "invalid-filter-repository"
    repository.mkdir()
    _git("init", "-b", "main", cwd=repository)
    original = composition.run_bounded_command

    def invalid_filter_output(argv, **kwargs):
        if "--get-regexp" in argv:
            return BoundedCommandResult(
                returncode=0,
                output=b"filter.hostile\xffname.clean\0",
                output_truncated=False,
            )
        return original(argv, **kwargs)

    monkeypatch.setattr(composition, "run_bounded_command", invalid_filter_output)
    with pytest.raises(RuntimeError, match="cannot be safely disabled"):
        composition.configured_workspace_root(repository)


def test_coding_workspace_rejects_inconsistent_filter_query_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "inconsistent-filter-repository"
    repository.mkdir()
    _git("init", "-b", "main", cwd=repository)
    original = composition.run_bounded_command

    def inconsistent_filter_status(argv, **kwargs):
        if "--get-regexp" in argv:
            return BoundedCommandResult(
                returncode=0,
                output=b"",
                output_truncated=False,
            )
        return original(argv, **kwargs)

    monkeypatch.setattr(
        composition,
        "run_bounded_command",
        inconsistent_filter_status,
    )
    with pytest.raises(RuntimeError, match="filter inspection is inconsistent"):
        composition.configured_workspace_root(repository)


@pytest.mark.parametrize(
    ("scope", "detail"),
    [
        (
            KnowledgeAccessScope(
                allowed_namespaces=["team"],
                allowed_statuses=[KnowledgeStatus.ACTIVE, KnowledgeStatus.PENDING],
            ),
            "namespace 'default'",
        ),
        (
            KnowledgeAccessScope(
                allowed_namespaces=["default"],
                allowed_statuses=[KnowledgeStatus.ACTIVE],
            ),
            "status 'pending'",
        ),
    ],
)
def test_coding_composition_rejects_incompatible_knowledge_scope(
    tmp_path: Path,
    scope: KnowledgeAccessScope,
    detail: str,
) -> None:
    workspace = tmp_path / "scope-workspace"
    workspace.mkdir()
    _git("init", "-b", "main", cwd=workspace)
    store = InMemoryKnowledgeStore(access_scope=scope)
    with pytest.raises(RuntimeError, match=detail):
        build_app(
            provider=_CompositionProvider(),
            session_store=InMemorySessionStore(),
            task_store=InMemoryTaskStore(),
            workspace_root=workspace,
            knowledge_store=store,
        )


def test_coding_execution_identities_fail_closed_and_bind_reviewer_version(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "identity-workspace"
    workspace.mkdir()
    _git("init", "-b", "main", cwd=workspace)
    scope = KnowledgeAccessScope(
        allowed_namespaces=["default"],
        allowed_statuses=[KnowledgeStatus.ACTIVE, KnowledgeStatus.PENDING],
    )
    custom_clock_store = SQLiteKnowledgeStore(
        tmp_path / "custom-clock.db",
        access_scope=scope,
        clock=lambda: pytest.fail("construction must not invoke the custom clock"),
    )
    app = build_app(
        provider=_CompositionProvider(),
        session_store=InMemorySessionStore(),
        task_store=InMemoryTaskStore(),
        workspace_root=workspace,
        artifact_store=LocalArtifactStore(tmp_path / "identity-artifacts"),
        knowledge_store=custom_clock_store,
    )
    assert app._environments["coding"].spec.execution_profile_identity is None

    reviewer = AgentSpec(name="reviewer", model="model")
    first = composition._subagent_tool_identity(
        reviewer,
        ExecutionProfileBehaviorIdentity(
            name="application.reviewer",
            behavior_version="1",
            implementation_version="1",
        ),
        generated_session_store=True,
    )
    second = composition._subagent_tool_identity(
        reviewer,
        ExecutionProfileBehaviorIdentity(
            name="application.reviewer",
            behavior_version="2",
            implementation_version="1",
        ),
        generated_session_store=True,
    )
    assert first is not None
    assert second is not None
    assert first != second
    assert (
        composition._subagent_tool_identity(
            reviewer,
            None,
            generated_session_store=True,
        )
        is None
    )
    assert (
        composition._subagent_tool_identity(
            reviewer,
            ExecutionProfileBehaviorIdentity(
                name="application.reviewer",
                behavior_version="1",
                implementation_version="1",
            ),
            generated_session_store=False,
        )
        is None
    )
    primary = app._agents["__AGENT_NAME__"]
    assert primary.tools["subagent"].tool.spec.execution_profile_identity is None
    assert primary.tools["subagent_result"].tool.spec.execution_profile_identity is None


def test_coding_dependency_diagnostics_do_not_publish_command_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canary = "PRIVATE_DEPENDENCY_OUTPUT"

    def incompatible(argv, **kwargs):
        del argv, kwargs
        return BoundedCommandResult(
            returncode=7,
            output=canary.encode(),
            output_truncated=False,
        )

    monkeypatch.setattr(composition, "run_bounded_command", incompatible)
    with pytest.raises(RuntimeError) as raised:
        composition._run_dependency_probe(["rg", "--version"], cwd=tmp_path)
    assert "exit code 7" in str(raised.value)
    assert canary not in str(raised.value)


def test_coding_dependency_filter_inspection_rejects_truncated_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def excessive(argv, **kwargs):
        del argv, kwargs
        raise composition.BoundedCommandOutputOverflowError

    monkeypatch.setattr(composition, "run_bounded_command", excessive)
    with pytest.raises(RuntimeError, match="excessive output"):
        composition._run_dependency_probe(
            ["git", "config"],
            cwd=tmp_path,
            reject_output_overflow=True,
        )


def test_coding_dependency_probe_fails_closed_on_output_read_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unreadable(argv, **kwargs):
        del argv, kwargs
        raise composition.BoundedCommandReadError

    monkeypatch.setattr(composition, "run_bounded_command", unreadable)
    with pytest.raises(RuntimeError, match="output could not be read"):
        composition._run_dependency_probe(["git", "config"], cwd=tmp_path)


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX detached process")
def test_coding_dependency_probe_has_a_bounded_detached_pipe_failure(
    tmp_path: Path,
) -> None:
    child_pid_file = tmp_path / "detached-child-pid"
    descendant = "import time; time.sleep(30)"
    parent = (
        "import pathlib, subprocess, sys; "
        f"child = subprocess.Popen([sys.executable, '-c', {descendant!r}], "
        "stdout=sys.stdout, stderr=sys.stderr, start_new_session=True); "
        f"pathlib.Path({str(child_pid_file)!r}).write_text(str(child.pid), "
        "encoding='utf-8')"
    )
    started = time.monotonic()
    try:
        with pytest.raises(command_probe.BoundedCommandTimeoutError):
            command_probe.run_bounded_command(
                [sys.executable, "-c", parent],
                cwd=tmp_path,
                env=os.environ.copy(),
                timeout_s=5,
                output_limit_bytes=128,
            )
    finally:
        if child_pid_file.exists():
            with suppress(ProcessLookupError):
                os.kill(int(child_pid_file.read_text(encoding="utf-8")), signal.SIGKILL)

    assert time.monotonic() - started < 3
    assert not any(
        thread.name == "coding-bounded-command-output" and thread.is_alive()
        for thread in command_probe.threading.enumerate()
    )
'''


def _coding_app_source(source: str) -> str:
    for unused_import in (
        "    AlwaysRequireApprovalToolPolicy,\n",
        "    SQLiteSessionStore,\n",
        "    SQLiteTaskStore,\n",
        "    public_authority_alias_codec_from_environment,\n",
    ):
        source = source.replace(unused_import, "", 1)
    import_marker = "from agents.agent import AGENT\n"
    source = source.replace(
        import_marker,
        import_marker
        + (
            "from agents.reviewer import (\n"
            "    REVIEWER,\n"
            "    REVIEWER_EXECUTION_PROFILE_IDENTITY,\n"
            ")\n"
        )
        + "from composition import build_coding_app\n",
        1,
    )
    start = source.index("def build_app(")
    end_marker = "\n    return app\n"
    end = source.index(end_marker, start) + len(end_marker)
    return source[:start] + _APP_BUILD + source[end:]


def coding_project_files(
    *,
    files: dict[str, str],
    render: Callable[[str], str],
) -> dict[str, str]:
    """Return the explicit overlay for the opt-in coding composition."""

    return {
        ".gitignore": ".cayu/\n" + files[".gitignore"],
        "app.py": _coding_app_source(files["app.py"]),
        "command_probe.py": render(_COMMAND_PROBE_PY),
        "composition.py": render(_COMPOSITION_PY),
        "agents/agent.py": render(_PRIMARY_AGENT_PY),
        "agents/reviewer.py": render(_REVIEWER_AGENT_PY),
        "tests/test_coding_composition.py": render(_SMOKE_TEST_PY),
        "README.md": files["README.md"] + render(_README_APPEND),
        "AGENTS.md": files["AGENTS.md"] + render(_AGENTS_APPEND),
    }
