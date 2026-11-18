"""Command executor for the Kubernetes-mode sidecar's agent container.

Run as the agent container's PID 1 (``python3.11 -m lambda_microvm_sidecar.executor``),
as the agent's own non-root user. It spawns, feeds, drains, signals and confirms
the supervisor's commands, and holds nothing else: owner fencing, command
records, output limits and every credential stay in the supervisor container.
See ``executor_protocol`` for the transport.

Agent commands run as the same user, so the executor keeps them out of its own
PID namespace: at start it creates one user, mount and PID namespace (the agent
namespace) and runs every command in it through a small launcher of its own
interpreter (``NAMESPACE_ENTER``), so it needs no binary from the agent's image. No agent process
can signal the executor or see it in ``/proc``. If the agent namespace's init
dies, every agent process dies with it; the executor creates a new one and
draws a new boot ID, so the host sees the loss. A new supervisor session
replaces it the same way, so nothing an earlier supervisor started survives.
Interactive commands also get a PID namespace of their own inside it, whose
teardown can be confirmed.

Requests (``op``), each with the supervisor's ``session``:

- ``hello``: ``pid_namespace`` (whether interactive commands get a PID namespace
  of their own) and ``protocol_version``.
- ``resolve_cwd`` ``cwd``: the resolved directory, which must lie in the workspace.
- ``spawn`` ``command_id argv env cwd process_tree``: start the command with
  exactly ``env``, in a new session; repeating it returns the same process.
- ``read`` ``command_id stream offset max_bytes wait_s``: output of ``stream``
  (``stdout`` or ``stderr``) from ``offset``, which acknowledges every earlier byte.
- ``write`` ``command_id offset data_base64 eof``: stdin, idempotent by ``offset``;
  answers once the command has taken the chunk.
- ``wait`` ``command_id wait_s``: ``returncode``, or null while it runs.
- ``stop_group``, ``stop_tree`` ``command_id``: stop the process group, or the
  PID namespace when the command has one.
- ``tree_stopped`` ``command_id timeout_s``: whether the whole tree is gone.
- ``release`` ``command_id``: forget the command.

Every answer carries ``boot_id``, drawn at start and again whenever the agent
namespace is replaced, and a refusal carries ``error`` (``kind`` and ``message``).
"""

from __future__ import annotations

import base64
import contextlib
import ctypes
import hashlib
import json
import os
import select
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from .executor_protocol import (
    DEFAULT_EXECUTOR_SOCKET,
    EXECUTOR_SOCKET_ENV,
    MAX_CHUNK_BYTES,
    MAX_WAIT_SECONDS,
    PROTOCOL_VERSION,
    FrameError,
    listen,
    peer_uid,
    recv_frame,
    send_frame,
)
from .supervisor import (
    LAUNCHER_ENVIRONMENT,
    PID_NAMESPACE_INIT,
    CommandConflictError,
    CommandRequestError,
    LocalProcesses,
    _boolean,
    _child_pids,
    _command_id,
    _CommandRecord,
    _drain_interactive,
    _InteractiveStreams,
    _nonblank_string,
    _proc_stat,
    _process_tree_stopped,
    _stop_interactive,
    _stop_process_group,
    _string,
)

SUPERVISOR_UID_ENV = "CAYU_EXECUTOR_SUPERVISOR_UID"
# Output held for the supervisor per stream. Beyond it the executor stops
# draining the pipe, so a supervisor that applies backpressure stalls the command.
STREAM_WINDOW_BYTES = 4 * 1024 * 1024
_CONNECTION_TIMEOUT_SECONDS = 30.0
_NAMESPACE_START_TIMEOUT_SECONDS = 10.0
_INIT_START_TIMEOUT_SECONDS = 5.0
_PR_SET_DUMPABLE = 4
# The agent namespace: user, mount and PID namespaces whose user maps only the
# executor's own UID and GID. Its init mounts a /proc of its own, reports
# "ready" (or the error) on stdout, and reaps orphans; it dies with its launcher.
NAMESPACE_HOLDER = """
import ctypes, os, signal, sys
libc = ctypes.CDLL(None, use_errno=True)
libc.mount.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_ulong, ctypes.c_void_p)
libc.prctl.argtypes = (ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong)
def check(result, what):
    if result != 0:
        raise OSError(ctypes.get_errno(), what)
try:
    uid, gid = os.getuid(), os.getgid()
    check(libc.unshare(0x10000000 | 0x20000000 | 0x00020000), "unshare")
    for name, value in (("uid_map", f"{uid} {uid} 1"), ("setgroups", "deny"), ("gid_map", f"{gid} {gid} 1")):
        with open(f"/proc/self/{name}", "w") as file:
            file.write(value)
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGCHLD})
    child = os.fork()
except OSError as exc:
    print(f"error: {exc}", flush=True)
    sys.exit(1)
if child == 0:
    try:
        check(libc.prctl(1, signal.SIGKILL, 0, 0, 0), "prctl")
        check(libc.mount(b"none", b"/", None, 0x4000 | 0x40000, None), "mount --make-rprivate /")
        check(libc.mount(b"proc", b"/proc", b"proc", 0x2 | 0x4 | 0x8, None), "mount /proc")
    except OSError as exc:
        print(f"error: {exc}", flush=True)
        os._exit(1)
    print("ready", flush=True)
    os.dup2(os.open(os.devnull, os.O_WRONLY), 1)
    while True:
        signal.sigwait({signal.SIGCHLD})
        try:
            while os.waitpid(-1, os.WNOHANG)[0]:
                pass
        except ChildProcessError:
            pass
os.waitpid(child, 0)
sys.exit(1)
"""
# Starts an interactive command in new user and PID namespaces, as
# ``unshare --user --map-user=<uid> --map-group=<gid> --pid --fork --kill-child``
# does with util-linux 2.38 or later (the image has an older one). The user
# namespace maps only the executor's own UID and GID, so the command keeps its
# identity; it exists because only a user namespace's owner may create a PID
# namespace without privileges. The launcher's child is the namespace init,
# which it kills on the launcher's death and whose status it exits with.
NAMESPACE_LAUNCHER = """
import ctypes, os, signal, sys
libc = ctypes.CDLL(None, use_errno=True)
uid, gid = os.getuid(), os.getgid()
if libc.unshare(0x10000000 | 0x20000000) != 0:
    raise OSError(ctypes.get_errno(), "unshare failed")
for name, value in (("uid_map", f"{uid} {uid} 1"), ("setgroups", "deny"), ("gid_map", f"{gid} {gid} 1")):
    with open(f"/proc/self/{name}", "w") as file:
        file.write(value)
init_source, argv = sys.argv[1], sys.argv[2:]
child = os.fork()
if child == 0:
    try:
        libc.prctl(1, signal.SIGKILL)
        os.execv(sys.executable, [sys.executable, "-I", "-S", "-c", init_source, *argv])
    finally:
        os._exit(127)
_, raw = os.waitpid(child, 0)
if os.WIFSIGNALED(raw):
    try:
        signal.signal(os.WTERMSIG(raw), signal.SIG_DFL)
    except (OSError, ValueError):
        pass  # SIGKILL and SIGSTOP keep their default action.
    os.kill(os.getpid(), os.WTERMSIG(raw))
sys.exit(os.waitstatus_to_exitcode(raw))
"""
# Runs a command in the agent namespace as ``nsenter --target=INIT --user --mount --pid
# --preserve-credentials --wd=CWD -- env -i -- ENV... ARGV`` would, with the interpreter alone:
# ``INIT CWD COUNT NAME=VALUE... ARGV...``. It opens the init's namespaces before entering any
# (the user namespace changes what it may open), enters the user, mount and PID namespaces in
# that order, and forks, as the PID namespace applies only to children. The child changes to the
# working directory only inside the mount namespace (busybox ``nsenter --wd`` changes it before,
# which leaves ``getcwd`` failing) and executes the command with exactly its environment; the
# launcher exits with the command's status or dies of its signal. Exit statuses follow nsenter and
# env: 1 when the namespace or directory cannot be entered, 127 for a missing program, 126 for
# one that cannot run.
NAMESPACE_ENTER = """
import ctypes, errno, os, signal, sys
libc = ctypes.CDLL(None, use_errno=True)
init, cwd, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
env = dict(item.split("=", 1) for item in sys.argv[4:4 + count])
argv = sys.argv[4 + count:]
try:
    namespaces = [(name, os.open(f"/proc/{init}/ns/{name}", os.O_RDONLY)) for name in ("user", "mnt", "pid")]
    for name, descriptor in namespaces:
        if libc.setns(descriptor, 0) != 0:
            raise OSError(ctypes.get_errno(), f"setns {name}")
        os.close(descriptor)
    child = os.fork()
except OSError as exc:
    print(f"cayu: cannot enter the agent namespace: {exc}", file=sys.stderr, flush=True)
    sys.exit(1)
if child == 0:
    try:
        os.chdir(cwd)
    except OSError as exc:
        print(f"cayu: cannot change directory to {cwd}: {exc.strerror}", file=sys.stderr, flush=True)
        os._exit(1)
    # The interpreter ignores these; a command starts with their default actions, as from env.
    for name in ("SIGPIPE", "SIGXFSZ"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), signal.SIG_DFL)
    # As execvp(3): a program without a slash is looked up in the command's PATH, and a file the
    # kernel cannot execute (a script without a shebang) runs with /bin/sh.
    if "/" in argv[0]:
        paths = [argv[0]]
    else:
        paths = [os.path.join(directory, argv[0]) for directory in os.get_exec_path(env)]
    # The search goes on past the errors execvp skips, and reports a denied file over a missing one.
    skipped = (errno.EACCES, errno.ENOENT, errno.ESTALE, errno.ENOTDIR, errno.ENODEV, errno.ETIMEDOUT)
    error = denied = None
    for path in paths:
        try:
            try:
                os.execve(path, argv, env)
            except OSError as exc:
                if exc.errno != errno.ENOEXEC:
                    raise
                os.execve("/bin/sh", ["/bin/sh", path, *argv[1:]], env)
        except OSError as exc:
            error = exc
            if exc.errno == errno.EACCES:
                denied = exc
            elif exc.errno not in skipped:
                break
    else:
        error = denied or error
    print(f"{argv[0]}: {error.strerror}", file=sys.stderr, flush=True)
    os._exit(127 if error.errno == errno.ENOENT else 126)
_, raw = os.waitpid(child, 0)
if os.WIFSIGNALED(raw):
    try:
        signal.signal(os.WTERMSIG(raw), signal.SIG_DFL)
    except (OSError, ValueError):
        pass  # SIGKILL and SIGSTOP keep their default action.
    os.kill(os.getpid(), os.WTERMSIG(raw))
sys.exit(os.waitstatus_to_exitcode(raw))
"""
# What the PID-namespace probe prints. It reports on stdout, not by exit status,
# which the executor's orphan reaper may collect first.
_PROBE_MARKER = "cayu-pid-namespace"
_STREAMS = ("stdout", "stderr")
_PROCESS_TREES = ("pid_namespace", "process_group")


class AgentNamespace:
    """The agent namespace every command runs in; see the module docstring."""

    def __init__(self, interpreter: str) -> None:
        self.interpreter = interpreter
        self.process = subprocess.Popen(
            [interpreter, "-I", "-S", "-c", NAMESPACE_HOLDER],
            env={},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            status = _read_line(self.process.stdout, timeout_s=_NAMESPACE_START_TIMEOUT_SECONDS)
            if status != "ready":
                raise RuntimeError(f"the agent namespace was not created: {status or 'no answer'}")
            children = _child_pids(self.process.pid)
            if len(children) != 1:
                raise RuntimeError("the agent namespace's init was not found")
            self.init_pid = children[0]
            # Interactive commands get a PID namespace of their own where nesting works.
            probe = [interpreter, "-I", "-S", "-c", f"print({_PROBE_MARKER!r})"]
            self.isolates_process_trees = self._prints_marker(
                [interpreter, "-I", "-S", "-c", NAMESPACE_LAUNCHER, PID_NAMESPACE_INIT, *probe]
            )
        except BaseException:
            self.process.kill()
            self.process.wait()
            raise

    def command(self, argv: list[str], env: dict[str, str], *, cwd: str) -> list[str]:
        """The launcher argv that runs ``argv`` in the namespace with exactly ``env``.

        The launcher runs in isolated mode (``-I -S``) with ``LAUNCHER_ENVIRONMENT``,
        so nothing from the command's environment (such as ``LD_PRELOAD`` or
        ``PYTHONPATH``) runs where it could reach the executor; it applies the
        environment only when it executes the command inside.
        """
        for key in env:
            if "=" in key:
                raise CommandRequestError("env key must not contain '='")
        return [
            self.interpreter,
            "-I",
            "-S",
            "-c",
            NAMESPACE_ENTER,
            str(self.init_pid),
            cwd,
            str(len(env)),
            *(f"{key}={value}" for key, value in env.items()),
            *argv,
        ]

    def _prints_marker(self, argv: list[str]) -> bool:
        try:
            probe = subprocess.run(
                self.command(argv, {}, cwd="/"),
                env=dict(LAUNCHER_ENVIRONMENT),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=_NAMESPACE_START_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return probe.stdout.decode("utf-8", errors="replace").strip() == _PROBE_MARKER


def _read_line(pipe: Any, *, timeout_s: float) -> str:
    """The first line ``pipe`` carries, or what arrived before EOF or the timeout."""
    descriptor = pipe.fileno()
    data = bytearray()
    deadline = time.monotonic() + timeout_s
    while b"\n" not in data:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([descriptor], [], [], remaining)[0]:
            break
        chunk = os.read(descriptor, 4096)
        if not chunk:
            break
        data.extend(chunk)
    pipe.close()
    return data.decode("utf-8", errors="replace").split("\n", 1)[0].strip()


@dataclass
class _Command:
    record: _CommandRecord
    process: subprocess.Popen[bytes]
    # Stdin is fed through the stdout streams' writer; stderr only drains.
    streams: dict[str, _InteractiveStreams] = field(default_factory=dict)


class Executor:
    """Run commands for one supervisor session, keyed by command ID."""

    def __init__(
        self,
        *,
        root: str | Path,
        interpreter: str = sys.executable,
        isolate: bool = True,
        stream_window_bytes: int = STREAM_WINDOW_BYTES,
    ) -> None:
        """``isolate=False`` runs commands in the executor's own namespaces, for tests only."""
        self.boot_id = str(uuid.uuid4())
        self.root = Path(root).resolve()
        self.interpreter = interpreter
        self._namespace = AgentNamespace(interpreter) if isolate else None
        self.stream_window_bytes = stream_window_bytes
        self._local = LocalProcesses()
        self._commands: dict[str, _Command] = {}
        self._session: str | None = None
        self._closed = False
        self._lock = threading.Lock()
        # Serializes spawns, so a retried spawn waits for the first and finds it.
        self._spawn_lock = threading.Lock()
        # Held while a new supervisor session retires the previous one's commands.
        self._session_lock = threading.Lock()
        # Serializes agent-namespace replacement, so a dead namespace is replaced once.
        self._namespace_lock = threading.Lock()
        if self._namespace is not None:
            threading.Thread(
                target=self._renew_namespace, name="cayu-executor-namespace", daemon=True
            ).start()

    @property
    def pid_namespace(self) -> bool:
        """Whether interactive commands get a PID namespace of their own."""
        namespace = self._namespace
        return namespace is not None and namespace.isolates_process_trees

    def _renew_namespace(self) -> None:
        """Replace an agent namespace whose init died, and draw a new boot ID.

        Every agent process died with it; the new boot ID tells the host so.
        """
        while True:
            with self._lock:
                namespace = cast("AgentNamespace", self._namespace)
            namespace.process.wait()
            if self._closed:
                return
            self._replace_namespace(namespace, "the agent namespace ended")

    def _replace_namespace(self, retired: AgentNamespace, reason: str) -> None:
        """Put a new agent namespace and boot ID in place of ``retired``, once."""
        with self._namespace_lock:
            with self._lock:
                if self._closed or self._namespace is not retired:
                    return
            print(f"cayu executor: {reason}; creating a new agent namespace", file=sys.stderr)
            try:
                renewed = AgentNamespace(self.interpreter)
            except BaseException as exc:
                print(f"cayu executor: {exc}", file=sys.stderr, flush=True)
                os._exit(1)
            with self._lock:
                self._namespace = renewed
                self.boot_id = str(uuid.uuid4())

    def close(self) -> None:
        """Stop every command and the agent namespace, which is not renewed."""
        with self._lock:
            self._closed = True
            commands = list(self._commands.values())
            self._commands.clear()
            namespace = self._namespace
        for command in commands:
            _stop(command)
        if namespace is not None:
            namespace.process.kill()
            namespace.process.wait()

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        """Answer one request; every answer carries the executor's current boot ID.

        The boot ID is read after the request is handled, so an answer about a
        command in a namespace that has since been replaced reports the loss.
        """
        response = self._dispatch(request)
        with self._lock:
            return {**response, "boot_id": self.boot_id}

    def _dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        try:
            self._enter_session(_nonblank_string(request.get("session"), "session"))
            operation = _OPERATIONS.get(cast("str", request.get("op")))
            if operation is None:
                raise CommandRequestError("unknown executor operation")
            return operation(self, request)
        except CommandRequestError as exc:
            return _error("invalid_request", str(exc))
        except CommandConflictError as exc:
            return _error("conflict", str(exc))
        except LookupError as exc:
            return _error("unknown_command", str(exc))
        except OSError as exc:
            return {
                "error": {
                    "kind": "os_error",
                    "errno": exc.errno,
                    "message": exc.strerror or str(exc),
                    "filename": None if exc.filename is None else str(exc.filename),
                }
            }

    def owns(self, pid: int) -> bool:
        """Whether ``pid`` is a command's own process, which its ``Popen`` reaps.

        Waits for a spawn in progress, whose process may have exited before it
        was registered.
        """
        with self._spawn_lock, self._lock:
            commands = list(self._commands.values())
            namespace = self._namespace
        if namespace is not None and namespace.process.pid == pid:
            namespace.process.poll()
            return True
        for command in commands:
            if command.process.pid == pid:
                command.process.poll()
                return True
        return False

    def _enter_session(self, session: str) -> None:
        """Begin a supervisor session, retiring everything an earlier one started.

        A new supervisor knows none of the earlier session's processes, so none
        may outlive it unseen: its commands are stopped, and the agent namespace
        is replaced, which also ends detached descendants and pruned commands.
        Spawns wait until the new namespace is in place.
        """
        with self._session_lock:
            with self._lock:
                if session == self._session:
                    return
                previous = self._session
            with self._spawn_lock:
                with self._lock:
                    stale = list(self._commands.values())
                    self._commands.clear()
                    self._session = session
                    namespace = self._namespace
                for command in stale:
                    _stop(command)
                if previous is not None and namespace is not None:
                    namespace.process.kill()
                    namespace.process.wait()
                    self._replace_namespace(namespace, "a new supervisor session began")

    def _command(self, request: dict[str, Any]) -> _Command:
        identifier = _command_id(request.get("command_id"))
        with self._lock:
            command = self._commands.get(identifier)
        if command is None:
            raise LookupError(f"the executor has no command {identifier!r}")
        return command

    def _hello(self, request: dict[str, Any]) -> dict[str, Any]:
        return {
            "pid_namespace": self.pid_namespace,
            "protocol_version": PROTOCOL_VERSION,
        }

    def _resolve_cwd(self, request: dict[str, Any]) -> dict[str, Any]:
        cwd = _nonblank_string(request.get("cwd"), "cwd")
        return {"cwd": self._local.resolve_cwd(cwd, self.root)}

    def _spawn(self, request: dict[str, Any]) -> dict[str, Any]:
        identifier = _command_id(request.get("command_id"))
        raw_argv = request.get("argv")
        if type(raw_argv) is not list or not raw_argv:
            raise CommandRequestError("argv must be a non-empty list")
        argv = [_nonblank_string(value, "argv") for value in raw_argv]
        raw_env = request.get("env")
        if type(raw_env) is not dict:
            raise CommandRequestError("env must be an object")
        env = {
            _nonblank_string(key, "env key"): _string(value, "env value")
            for key, value in raw_env.items()
        }
        process_tree = request.get("process_tree")
        if process_tree not in _PROCESS_TREES:
            raise CommandRequestError("process_tree must be pid_namespace or process_group")
        if process_tree == "pid_namespace" and not self.pid_namespace:
            raise CommandRequestError("this executor cannot create PID namespaces")
        cwd = self._local.resolve_cwd(_nonblank_string(request.get("cwd"), "cwd"), self.root)
        fingerprint = hashlib.sha256(
            json.dumps([argv, env, cwd, process_tree], separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        with self._spawn_lock:
            with self._lock:
                existing = self._commands.get(identifier)
            if existing is not None:
                if existing.record.payload_fingerprint != fingerprint:
                    raise CommandConflictError(
                        f"Command id {identifier!r} was already spawned with another command"
                    )
                return _spawned(existing)
            record = _CommandRecord(command_id=identifier, payload_fingerprint=fingerprint)
            record.process_tree = cast("str", process_tree)
            if process_tree == "pid_namespace":
                argv = [
                    self.interpreter,
                    "-I",
                    "-S",
                    "-c",
                    NAMESPACE_LAUNCHER,
                    PID_NAMESPACE_INIT,
                    *argv,
                ]
            with self._lock:
                namespace = self._namespace
            # Only the given environment: nothing of the executor's own reaches the command.
            if namespace is not None:
                argv, env = namespace.command(argv, env, cwd=cwd), dict(LAUNCHER_ENVIRONMENT)
            process = cast(
                "subprocess.Popen[bytes]", self._local.spawn(record, argv, cwd=cwd, env=env)
            )
            record.process = process
            if process_tree == "pid_namespace":
                record.namespace_init = _await_nested_init(process)
            command = _Command(record=record, process=process)
            for name, pipe in (("stdout", process.stdout), ("stderr", process.stderr)):
                streams = _InteractiveStreams(self.stream_window_bytes)
                command.streams[name] = streams
                threading.Thread(
                    target=_drain,
                    args=(pipe, streams),
                    name=f"cayu-executor-{name}-{identifier}",
                    daemon=True,
                ).start()
            command.streams["stdout"].attach(process.stdin)
            with self._lock:
                self._commands[identifier] = command
            return _spawned(command)

    def _read(self, request: dict[str, Any]) -> dict[str, Any]:
        command = self._command(request)
        stream = request.get("stream")
        if stream not in _STREAMS:
            raise CommandRequestError("stream must be stdout or stderr")
        offset = _offset(request.get("offset"))
        max_bytes = request.get("max_bytes")
        if type(max_bytes) is not int or not 0 < max_bytes <= MAX_CHUNK_BYTES:
            raise CommandRequestError(f"max_bytes must be an integer from 1 to {MAX_CHUNK_BYTES}")
        data, eof = command.streams[cast("str", stream)].read(
            offset, max_bytes=max_bytes, wait_s=_wait_seconds(request.get("wait_s", 0))
        )
        return {"data_base64": base64.b64encode(data).decode("ascii"), "eof": eof}

    def _write(self, request: dict[str, Any]) -> dict[str, Any]:
        command = self._command(request)
        raw = request.get("data_base64", "")
        if type(raw) is not str:
            raise CommandRequestError("data_base64 must be a string")
        try:
            data = base64.b64decode(raw, validate=True)
        except ValueError as exc:
            raise CommandRequestError("data_base64 must be valid base64") from exc
        if len(data) > MAX_CHUNK_BYTES:
            raise CommandRequestError(f"a stdin chunk exceeds {MAX_CHUNK_BYTES} bytes")
        return command.streams["stdout"].write(
            _offset(request.get("offset")), data, eof=_boolean(request.get("eof", False), "eof")
        )

    def _wait(self, request: dict[str, Any]) -> dict[str, Any]:
        command = self._command(request)
        with contextlib.suppress(subprocess.TimeoutExpired):
            command.process.wait(timeout=_wait_seconds(request.get("wait_s", 0)))
        return {"returncode": command.process.returncode}

    def _stop_group(self, request: dict[str, Any]) -> dict[str, Any]:
        _stop_process_group(self._command(request).process)
        return {}

    def _stop_tree(self, request: dict[str, Any]) -> dict[str, Any]:
        _stop_tree(self._command(request))
        return {}

    def _tree_stopped(self, request: dict[str, Any]) -> dict[str, Any]:
        command = self._command(request)
        timeout_s = _wait_seconds(request.get("timeout_s", 0))
        return {"stopped": _process_tree_stopped(command.record, timeout_s=timeout_s)}

    def _release(self, request: dict[str, Any]) -> dict[str, Any]:
        identifier = _command_id(request.get("command_id"))
        with self._lock:
            command = self._commands.pop(identifier, None)
        if command is not None:
            _stop(command)
        return {}


_OPERATIONS: dict[str, Callable[[Executor, dict[str, Any]], dict[str, Any]]] = {
    "hello": Executor._hello,
    "resolve_cwd": Executor._resolve_cwd,
    "spawn": Executor._spawn,
    "read": Executor._read,
    "write": Executor._write,
    "wait": Executor._wait,
    "stop_group": Executor._stop_group,
    "stop_tree": Executor._stop_tree,
    "tree_stopped": Executor._tree_stopped,
    "release": Executor._release,
}


def _spawned(command: _Command) -> dict[str, Any]:
    return {"pid": command.process.pid, "process_tree": command.record.process_tree}


def _error(kind: str, message: str) -> dict[str, Any]:
    return {"error": {"kind": kind, "message": message}}


def _offset(value: object) -> int:
    if type(value) is not int or value < 0:
        raise CommandRequestError("offset must be a non-negative integer")
    return value


def _wait_seconds(value: object) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not 0 <= value <= MAX_WAIT_SECONDS
    ):
        raise CommandRequestError(f"waits must be from 0 to {MAX_WAIT_SECONDS:g} seconds")
    return float(value)


def _drain(pipe: Any, streams: _InteractiveStreams) -> None:
    _drain_interactive(pipe, streams)
    streams.finish_stdout()


def _stop(command: _Command) -> None:
    """Stop a command nobody will read again and drop its output."""
    for streams in command.streams.values():
        streams.abandon()
    if command.process.poll() is None:
        if command.record.process_tree == "pid_namespace":
            _stop_tree(command)
        else:
            _stop_process_group(command.process)
    with contextlib.suppress(OSError, ValueError):
        if command.process.stdin is not None:
            command.process.stdin.close()


def _stop_tree(command: _Command) -> None:
    record = command.record
    if record.namespace_init is None and command.process.poll() is None:
        record.namespace_init = _await_nested_init(command.process, timeout_s=0.5)
    _stop_interactive(record, command.process)


def _await_nested_init(
    process: subprocess.Popen[bytes], *, timeout_s: float = _INIT_START_TIMEOUT_SECONDS
) -> tuple[int, str] | None:
    """Find an interactive command's namespace init, with its start time.

    The namespace launcher forks the PID-namespace launcher into the agent
    namespace, and that forks the init, so the init is the spawned process's
    grandchild.
    """
    deadline = time.monotonic() + timeout_s
    while process.poll() is None:
        for launcher in _child_pids(process.pid):
            for init in _child_pids(launcher):
                stat = _proc_stat(init)
                if stat is not None:
                    return init, stat[2]
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.005)
    return None


class ExecutorServer:
    """Serve one executor on a listening socket, one request per connection."""

    def __init__(
        self,
        executor: Executor,
        listener: socket.socket,
        *,
        supervisor_uid: int,
        peer_uid: Callable[[socket.socket], int] = peer_uid,
    ) -> None:
        self.executor = executor
        self._listener = listener
        self.supervisor_uid = supervisor_uid
        self._peer_uid = peer_uid

    def serve_forever(self) -> None:
        while True:
            try:
                connection, _address = self._listener.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(connection,), daemon=True).start()

    def start(self) -> threading.Thread:
        thread = threading.Thread(target=self.serve_forever, name="cayu-executor", daemon=True)
        thread.start()
        return thread

    def close(self) -> None:
        with contextlib.suppress(OSError):
            self._listener.shutdown(socket.SHUT_RDWR)
        self._listener.close()

    def _serve(self, connection: socket.socket) -> None:
        with connection:
            try:
                uid = self._peer_uid(connection)
            except OSError:
                return
            if uid != self.supervisor_uid:
                # Agent commands share this network namespace but never its supervisor UID.
                print(f"cayu executor refused a connection from uid {uid}", file=sys.stderr)
                return
            connection.settimeout(_CONNECTION_TIMEOUT_SECONDS)
            try:
                request = recv_frame(connection)
            except (OSError, FrameError):
                return
            response = self.executor.handle(request)
            with contextlib.suppress(OSError):
                send_frame(connection, response)


def _reap_orphans(executor: Executor) -> None:
    """Reap every exited child that no command's ``Popen`` owns, as PID 1 must.

    ``WNOWAIT`` looks at a zombie without reaping it, so a command's own exit
    status still reaches its ``Popen``.
    """
    while True:
        try:
            exited = os.waitid(os.P_ALL, 0, os.WEXITED | os.WNOWAIT)
        except ChildProcessError:
            time.sleep(0.2)
            continue
        if exited is None:
            continue
        if executor.owns(exited.si_pid):
            time.sleep(0.01)
            continue
        with contextlib.suppress(ChildProcessError):
            os.waitpid(exited.si_pid, 0)


def _set_not_dumpable() -> None:
    # Agent commands run as this UID; without this they could read its memory.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_DUMPABLE, 0, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_DUMPABLE) failed")


def main() -> None:
    # Bound first, before any command can run, so nothing else can take the name.
    listener = listen(os.environ.get(EXECUTOR_SOCKET_ENV, DEFAULT_EXECUTOR_SOCKET))
    _set_not_dumpable()
    raw_supervisor_uid = os.environ.get(SUPERVISOR_UID_ENV, "")
    if not raw_supervisor_uid.isdigit():
        raise SystemExit(f"{SUPERVISOR_UID_ENV} must name the supervisor container's UID")
    supervisor_uid = int(raw_supervisor_uid)
    if os.geteuid() == 0:
        raise SystemExit("the command executor must not run as root")
    if supervisor_uid == os.geteuid():
        raise SystemExit(f"{SUPERVISOR_UID_ENV} must differ from the executor's own UID")
    executor = Executor(root=os.environ.get("CAYU_MICROVM_WORKSPACE_ROOT", "/workspace"))
    threading.Thread(target=_reap_orphans, args=(executor,), daemon=True).start()
    print(
        f"cayu executor {executor.boot_id} serving uid {supervisor_uid}"
        f" (pid_namespace={str(executor.pid_namespace).lower()})",
        file=sys.stderr,
        flush=True,
    )
    ExecutorServer(executor, listener, supervisor_uid=supervisor_uid).serve_forever()


if __name__ == "__main__":
    main()
