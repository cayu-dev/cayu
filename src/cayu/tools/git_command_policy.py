from __future__ import annotations

import posixpath
import re
from collections.abc import Iterable
from dataclasses import dataclass

from cayu._validation import require_clean_nonblank
from cayu.core.tools import ToolContext
from cayu.runners.base import is_same_or_child
from cayu.tools.command_policy import ProcessCommandPolicy
from cayu.tools.commands import (
    CommandPolicy,
    CommandPolicyDecision,
    CommandPolicyResult,
    CommandRequest,
)

_FS_MONITOR_DISABLED = "core.fsmonitor=false"
_LOG_SIGNATURE_DISABLED = "log.showSignature=false"
_SAFE_GLOBAL_CONFIG = frozenset(
    {
        _FS_MONITOR_DISABLED,
        _LOG_SIGNATURE_DISABLED,
        "core.hooksPath=/dev/null",
        "commit.gpgSign=false",
    }
)
_SAFE_GIT_ENV_NAMES = frozenset({"LANG", "LC_ALL", "LC_CTYPE", "TZ"})
_STATUS_OPTIONS = frozenset(
    {
        "--short",
        "--porcelain",
        "--porcelain=v1",
        "--branch",
        "--untracked-files=no",
        "--untracked-files=normal",
        "--untracked-files=all",
    }
)
_LS_FILES_OPTIONS = frozenset(
    {
        "--cached",
        "--modified",
        "--deleted",
        "--others",
        "--exclude-standard",
    }
)
_DIFF_OPTIONS = frozenset(
    {
        "--cached",
        "--staged",
        "--stat",
        "--name-only",
        "--name-status",
        "--patch",
        "--exit-code",
        "--quiet",
    }
)
_SIMPLE_REVISION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*(?:[~^][0-9]*)*\Z")
_MAX_LOG_COUNT = 1000
_MAX_PATH_BYTES = 4096
_MAX_PATH_COUNT = 256
DEFAULT_MAX_COMMIT_MESSAGE_BYTES = 4096


@dataclass(frozen=True)
class _GitInvocation:
    subcommand: str
    args: tuple[str, ...]
    configs: frozenset[str]
    no_pager: bool


class GitCommandPolicy(CommandPolicy):
    """Authorize a bounded local Git workflow after general process checks."""

    def __init__(
        self,
        *,
        process_policy: ProcessCommandPolicy,
        git_executables: Iterable[str] = ("git",),
        allowed_repositories: Iterable[str],
        max_commit_message_bytes: int = DEFAULT_MAX_COMMIT_MESSAGE_BYTES,
    ) -> None:
        if not isinstance(process_policy, ProcessCommandPolicy):
            raise TypeError("process_policy must be a ProcessCommandPolicy.")
        self._process_policy = process_policy
        self._git_executables = _validated_strings(
            git_executables,
            field_name="git_executables",
        )
        if not self._git_executables:
            raise ValueError("git_executables must contain at least one executable.")
        self._allowed_repositories = _validated_repository_roots(allowed_repositories)
        if not self._allowed_repositories:
            raise ValueError("allowed_repositories must contain at least one root.")
        self._max_commit_message_bytes = _positive_int(
            max_commit_message_bytes,
            field_name="max_commit_message_bytes",
        )

    async def evaluate(
        self,
        ctx: ToolContext,
        request: CommandRequest,
    ) -> CommandPolicyResult:
        process_result = await self._process_policy.evaluate(ctx, request)
        if process_result.decision is not CommandPolicyDecision.ALLOW:
            return process_result

        command = request.command
        argv = command.argv
        if command.kind != "process" or not argv or argv[0] not in self._git_executables:
            return _deny("Executable is not a configured Git executable.")

        if request.env and any(name not in _SAFE_GIT_ENV_NAMES for name in request.env):
            return _deny("Git environment is not allowed by the Git policy.")
        if request.stdin is not None:
            return _deny("Git stdin is not allowed by the Git policy.")

        parsed = self._parse_invocation(argv, request.canonical_cwd)
        if isinstance(parsed, CommandPolicyResult):
            return parsed
        if not parsed.no_pager or _FS_MONITOR_DISABLED not in parsed.configs:
            return _deny("Git invocation is missing required safety controls.")
        return _evaluate_subcommand(parsed, self._max_commit_message_bytes)

    def _parse_invocation(
        self,
        argv: list[str],
        canonical_cwd: str | None,
    ) -> _GitInvocation | CommandPolicyResult:
        repository = _containing_root(canonical_cwd, self._allowed_repositories)
        if repository is None:
            return _deny("Git working directory is not an allowed repository.")
        if canonical_cwd is None:  # narrowed by _containing_root; retained for type checkers
            return _deny("Git working directory is not an allowed repository.")
        effective_cwd = canonical_cwd
        configs: set[str] = set()
        no_pager = False
        index = 1
        while index < len(argv):
            item = argv[index]
            if item == "--no-pager":
                no_pager = True
                index += 1
                continue
            if item == "-C":
                if index + 1 >= len(argv):
                    return _deny("Git global option is missing its operand.")
                operand = argv[index + 1]
                index += 2
                effective_cwd = _resolve_git_cwd(effective_cwd, operand, repository)
                if effective_cwd is None:
                    return _deny("Git working-directory change is not allowed.")
                continue
            if item == "-c":
                if index + 1 >= len(argv):
                    return _deny("Git global option is missing its operand.")
                config = argv[index + 1]
                index += 2
                if config not in _SAFE_GLOBAL_CONFIG:
                    return _deny("Git global option is not allowed.")
                configs.add(config)
                continue
            if item.startswith("-"):
                return _deny("Git global option is not allowed.")
            break

        if index >= len(argv):
            return _deny("Git subcommand is not supported by the Git policy.")
        subcommand = argv[index]
        if subcommand.startswith("-"):
            return _deny("Git subcommand is not supported by the Git policy.")
        return _GitInvocation(
            subcommand=subcommand,
            args=tuple(argv[index + 1 :]),
            configs=frozenset(configs),
            no_pager=no_pager,
        )


def _evaluate_subcommand(
    invocation: _GitInvocation,
    max_commit_message_bytes: int,
) -> CommandPolicyResult:
    if invocation.subcommand == "commit":
        return _evaluate_commit(invocation, max_commit_message_bytes)
    required_configs = {_FS_MONITOR_DISABLED}
    if invocation.subcommand in {"log", "show"}:
        required_configs.add(_LOG_SIGNATURE_DISABLED)
    if invocation.configs != frozenset(required_configs):
        return _deny("Git global option is not allowed.")
    evaluators = {
        "add": _evaluate_add,
        "branch": _evaluate_branch,
        "cat-file": _evaluate_cat_file,
        "diff": _evaluate_diff,
        "log": _evaluate_log,
        "ls-files": _evaluate_ls_files,
        "rev-parse": _evaluate_rev_parse,
        "show": _evaluate_show,
        "status": _evaluate_status,
    }
    evaluator = evaluators.get(invocation.subcommand)
    if evaluator is None:
        return _deny("Git subcommand is not supported by the Git policy.")
    return evaluator(invocation.args)


def _evaluate_status(args: tuple[str, ...]) -> CommandPolicyResult:
    options, paths = _split_paths(args)
    if options is None or any(option not in _STATUS_OPTIONS for option in options):
        return _deny("Git option is not allowed for this subcommand.")
    return _paths_result(paths)


def _evaluate_ls_files(args: tuple[str, ...]) -> CommandPolicyResult:
    options, paths = _split_paths(args)
    if options is None or any(option not in _LS_FILES_OPTIONS for option in options):
        return _deny("Git option is not allowed for this subcommand.")
    return _paths_result(paths)


def _evaluate_rev_parse(args: tuple[str, ...]) -> CommandPolicyResult:
    if args in {
        ("--show-toplevel",),
        ("--show-prefix",),
        ("--is-inside-work-tree",),
        ("--git-dir",),
    }:
        return _allow()
    if len(args) == 2 and args[0] == "--verify" and _is_revision(args[1]):
        return _allow()
    return _deny("Git option is not allowed for this subcommand.")


def _evaluate_branch(args: tuple[str, ...]) -> CommandPolicyResult:
    if args == ("--show-current",):
        return _allow()
    return _deny("Git option is not allowed for this subcommand.")


def _evaluate_cat_file(args: tuple[str, ...]) -> CommandPolicyResult:
    if len(args) == 2 and args[0] in {"-e", "-t", "-s", "-p"} and _is_object_spec(args[1]):
        return _allow()
    return _deny("Git option is not allowed for this subcommand.")


def _evaluate_log(args: tuple[str, ...]) -> CommandPolicyResult:
    before_paths, paths = _split_paths(args)
    if before_paths is None:
        return _deny("Git option is not allowed for this subcommand.")
    revisions: list[str] = []
    oneline = False
    for item in before_paths:
        if item == "--oneline" and not oneline:
            oneline = True
            continue
        if item in {"--no-decorate", "--decorate=short"}:
            continue
        if item.startswith("--max-count="):
            count = item.removeprefix("--max-count=")
            if not count.isdigit() or not 1 <= int(count) <= _MAX_LOG_COUNT:
                return _deny("Git option is not allowed for this subcommand.")
            continue
        if item.startswith("-") or not _is_revision(item):
            return _deny("Git option is not allowed for this subcommand.")
        revisions.append(item)
    if not oneline or len(revisions) > 2:
        return _deny("Git option is not allowed for this subcommand.")
    return _paths_result(paths)


def _evaluate_show(args: tuple[str, ...]) -> CommandPolicyResult:
    before_paths, paths = _split_paths(args)
    if before_paths is None:
        return _deny("Git option is not allowed for this subcommand.")
    required = {"--format=medium", "--no-ext-diff", "--no-textconv"}
    remaining = [item for item in before_paths if item not in required]
    if not required.issubset(before_paths) or len(remaining) != 1:
        return _deny("Git option is not allowed for this subcommand.")
    if not _is_object_spec(remaining[0]):
        return _deny("Git option is not allowed for this subcommand.")
    return _paths_result(paths)


def _evaluate_diff(args: tuple[str, ...]) -> CommandPolicyResult:
    before_paths, paths = _split_paths(args)
    if before_paths is None:
        return _deny("Git option is not allowed for this subcommand.")
    required = {"--no-ext-diff", "--no-textconv"}
    if not required.issubset(before_paths):
        return _deny("Git option is not allowed for this subcommand.")
    revisions: list[str] = []
    for item in before_paths:
        if item in required or item in _DIFF_OPTIONS:
            continue
        if item.startswith("--unified="):
            lines = item.removeprefix("--unified=")
            if lines.isdigit() and 0 <= int(lines) <= 20:
                continue
        if item.startswith("-") or not _is_revision(item):
            return _deny("Git option is not allowed for this subcommand.")
        revisions.append(item)
    if len(revisions) > 2:
        return _deny("Git option is not allowed for this subcommand.")
    return _paths_result(paths)


def _evaluate_add(args: tuple[str, ...]) -> CommandPolicyResult:
    if not args or args[0] != "--" or len(args) == 1:
        return _deny("Git option is not allowed for this subcommand.")
    return _paths_result(args[1:])


def _evaluate_commit(
    invocation: _GitInvocation,
    max_commit_message_bytes: int,
) -> CommandPolicyResult:
    required_configs = frozenset(
        {
            _FS_MONITOR_DISABLED,
            "core.hooksPath=/dev/null",
            "commit.gpgSign=false",
        }
    )
    if invocation.configs != required_configs:
        return _deny("Git commit shape is not allowed by the Git policy.")
    no_verify = False
    no_gpg_sign = False
    message: str | None = None
    index = 0
    while index < len(invocation.args):
        item = invocation.args[index]
        if item == "--no-verify" and not no_verify:
            no_verify = True
            index += 1
            continue
        if item == "--no-gpg-sign" and not no_gpg_sign:
            no_gpg_sign = True
            index += 1
            continue
        if item in {"-m", "--message"} and message is None:
            if index + 1 >= len(invocation.args):
                return _deny("Git commit shape is not allowed by the Git policy.")
            message = invocation.args[index + 1]
            index += 2
            continue
        return _deny("Git commit shape is not allowed by the Git policy.")
    if not no_verify or not no_gpg_sign or message is None or "\0" in message:
        return _deny("Git commit shape is not allowed by the Git policy.")
    if len(message.encode("utf-8")) > max_commit_message_bytes:
        return _deny("Git commit message exceeds the Git policy byte limit.")
    return _allow()


def _split_paths(
    args: tuple[str, ...],
) -> tuple[tuple[str, ...] | None, tuple[str, ...]]:
    if args.count("--") > 1:
        return None, ()
    if "--" not in args:
        return args, ()
    index = args.index("--")
    return args[:index], args[index + 1 :]


def _paths_result(paths: tuple[str, ...]) -> CommandPolicyResult:
    if len(paths) > _MAX_PATH_COUNT or any(not _is_safe_path(path) for path in paths):
        return _deny("Git path is not allowed by the Git policy.")
    return _allow()


def _is_safe_path(path: str) -> bool:
    if (
        not path
        or "\0" in path
        or len(path.encode("utf-8")) > _MAX_PATH_BYTES
        or posixpath.isabs(path)
        or posixpath.normpath(path) != path
        or path.startswith(":")
        or any(character in path for character in "*?[")
    ):
        return False
    components = path.split("/")
    return not any(
        component in {"", ".", ".."} or component.casefold() == ".git" for component in components
    )


def _is_object_spec(value: str) -> bool:
    if ":" not in value:
        return _is_revision(value)
    revision, path = value.split(":", 1)
    return _is_revision(revision) and _is_safe_path(path)


def _is_revision(value: str) -> bool:
    if not value or len(value.encode("utf-8")) > 256 or "\0" in value:
        return False
    if "..." in value:
        parts = value.split("...")
    elif ".." in value:
        parts = value.split("..")
    else:
        parts = [value]
    return len(parts) in {1, 2} and all(
        _SIMPLE_REVISION_PATTERN.fullmatch(part) is not None for part in parts
    )


def _validated_strings(values: Iterable[str], *, field_name: str) -> frozenset[str]:
    if isinstance(values, str | bytes):
        raise TypeError(f"{field_name} must be an iterable of strings, not a string.")
    result: set[str] = set()
    for value in values:
        value = require_clean_nonblank(value, f"{field_name} item")
        if "\0" in value:
            raise ValueError(f"{field_name} entries cannot contain NUL bytes.")
        result.add(value)
    return frozenset(result)


def _validated_repository_roots(values: Iterable[str]) -> frozenset[str]:
    roots = _validated_strings(values, field_name="allowed_repositories")
    for root in roots:
        if not posixpath.isabs(root):
            raise ValueError("allowed_repositories entries must be absolute POSIX paths.")
        if posixpath.normpath(root) != root:
            raise ValueError("allowed_repositories entries must be normalized POSIX paths.")
    return roots


def _containing_root(path: str | None, roots: frozenset[str]) -> str | None:
    if path is None or not posixpath.isabs(path) or posixpath.normpath(path) != path:
        return None
    matches = [root for root in roots if is_same_or_child(path, root)]
    return max(matches, key=len, default=None)


def _resolve_git_cwd(current: str, operand: str, repository: str) -> str | None:
    if not operand or "\0" in operand:
        return None
    component_text = operand[1:] if posixpath.isabs(operand) else operand
    components = component_text.split("/")
    if any(component in {"", ".", ".."} for component in components):
        return None
    if posixpath.isabs(operand):
        resolved = posixpath.normpath(operand)
    else:
        resolved = posixpath.normpath(posixpath.join(current, operand))
    if not is_same_or_child(resolved, repository):
        return None
    return resolved


def _positive_int(value: int, *, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")
    return value


def _allow() -> CommandPolicyResult:
    return CommandPolicyResult(decision=CommandPolicyDecision.ALLOW)


def _deny(reason: str) -> CommandPolicyResult:
    return CommandPolicyResult(
        decision=CommandPolicyDecision.DENY,
        reason=reason,
    )
