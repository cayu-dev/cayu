"""Real Linux enforcement checks for the daemon-free vendor launcher."""

from __future__ import annotations

import asyncio
import ctypes
import json
import os
import socket
import sys
from pathlib import Path

import pytest

from cayu.external_agents import linux_agent_launch

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="requires Linux Landlock ABI 6")


@pytest.fixture(autouse=True)
def _landlock_available():
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.syscall(444, 0, 0, 1) < 6:
        pytest.skip("kernel does not support Landlock ABI 6")


def _cli(path: Path, source: str) -> Path:
    path.write_text(f"#!{sys.executable}\n{source}")
    path.chmod(0o755)
    return path


def test_linux_launch_confines_files_network_environment_and_processes(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    outside = tmp_path / "secret"
    outside.write_text("host-secret")
    (checkout / "escape").symlink_to(outside)
    monkeypatch.setenv("HOST_SECRET", "host-secret-env")
    source = """
import errno, json, os, socket, subprocess, sys
from pathlib import Path
result = {}
def denied(name, operation):
    try:
        operation()
    except PermissionError:
        result[name] = True
    else:
        result[name] = False
denied("read", lambda: Path(sys.argv[1]).read_text())
denied("write", lambda: Path(sys.argv[1]).write_text("changed"))
denied("truncate", lambda: os.truncate(sys.argv[1], 0))
denied("symlink", lambda: Path("escape").read_text())
denied("setsid", os.setsid)
denied("setpgid", lambda: os.setpgid(0, 0))
denied("signal_parent", lambda: os.kill(os.getppid(), 15))
denied("udp", lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM))
denied("unix", lambda: socket.socket(socket.AF_UNIX))
for family in (socket.AF_INET, socket.AF_INET6):
    for protocol in (132, 262):  # SCTP and MPTCP must fail even if kernel-enabled.
        denied(f"protocol_{family}_{protocol}",
               lambda: socket.socket(family, socket.SOCK_STREAM, protocol))
def connect():
    with socket.socket() as s:
        s.connect(("127.0.0.1", int(sys.argv[2])))
denied("tcp", connect)
Path("output").write_text("ok")
Path(os.environ["HOME"], "cache").write_text("ok")
result["env"] = dict(os.environ)
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
result["child"] = child.pid
print(json.dumps(result), flush=True)
for line in sys.stdin:
    print(line.strip().upper(), flush=True)
"""
    cli = _cli(tmp_path / "vendor", source)

    async def scenario():
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            async with linux_agent_launch(
                cli=str(cli), cwd=checkout, read_only_paths=[sys.base_prefix, sys.prefix]
            ) as launch:
                process = await asyncio.create_subprocess_exec(
                    launch.executable,
                    str(outside),
                    str(port),
                    cwd=launch.cwd,
                    env=dict(launch.environment),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                assert process.stdout is not None and process.stdin is not None
                line = await asyncio.wait_for(process.stdout.readline(), 10)
                if not line:
                    _, error = await process.communicate()
                    pytest.fail(error.decode())
                result = json.loads(line)
                for name in (
                    "read",
                    "write",
                    "truncate",
                    "symlink",
                    "setsid",
                    "setpgid",
                    "signal_parent",
                    "udp",
                    "unix",
                    "tcp",
                ):
                    assert result[name], name
                for family in (socket.AF_INET, socket.AF_INET6):
                    for protocol in (132, 262):
                        assert result[f"protocol_{family}_{protocol}"]
                assert "HOST_SECRET" not in result["env"]
                assert result["env"]["HOME"] != os.environ.get("HOME")
                process.stdin.write(b"hello\n")
                await process.stdin.drain()
                assert await asyncio.wait_for(process.stdout.readline(), 5) == b"HELLO\n"
                await launch.close()
                await asyncio.wait_for(process.wait(), 10)
            child = Path(f"/proc/{result['child']}/stat")
            assert not child.exists() or child.read_text().split(") ", 1)[1][0] == "Z"
        assert outside.read_text() == "host-secret"
        assert (checkout / "output").read_text() == "ok"

    asyncio.run(scenario())


def test_linux_launch_allows_only_explicit_tcp_ports(tmp_path):
    cli = _cli(
        tmp_path / "vendor",
        """
import socket, sys
with socket.create_connection(("127.0.0.1", int(sys.argv[1]))) as s:
    s.sendall(b"gateway")
""",
    )
    checkout = tmp_path / "checkout"
    checkout.mkdir()

    async def scenario():
        received = asyncio.get_running_loop().create_future()

        async def accept(reader, writer):
            received.set_result(await reader.read(7))
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(accept, "127.0.0.1", 0)
        try:
            port = server.sockets[0].getsockname()[1]
            async with linux_agent_launch(
                cli=str(cli),
                cwd=checkout,
                allowed_tcp_ports=[port],
                read_only_paths=[sys.base_prefix, sys.prefix],
            ) as launch:
                process = await asyncio.create_subprocess_exec(
                    launch.executable,
                    str(port),
                    env=dict(launch.environment),
                )
                assert await asyncio.wait_for(process.wait(), 10) == 0
                assert await asyncio.wait_for(received, 5) == b"gateway"
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


def test_guest_pythonpath_cannot_run_before_confinement(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("original")
    (hooks / "sitecustomize.py").write_text(
        "import os\nfrom pathlib import Path\n"
        f"try:\n    Path({str(outside)!r}).write_text('escaped')\n"
        "except PermissionError:\n    Path(os.environ['HOME'], 'hook-confined').touch()\n"
    )
    cli = _cli(
        tmp_path / "vendor",
        "import os\nfrom pathlib import Path\n"
        "print(Path(os.environ['HOME'], 'hook-confined').exists(), flush=True)\n",
    )

    async def scenario():
        async with linux_agent_launch(
            cli=str(cli),
            cwd=checkout,
            read_only_paths=[hooks, sys.base_prefix, sys.prefix],
            environment={"PYTHONPATH": str(hooks), "MODEL_TOKEN": "scoped-test-token"},
        ) as launch:
            assert "PYTHONPATH" not in launch.environment
            assert "MODEL_TOKEN" not in launch.environment
            assert "scoped-test-token" in launch.secret_values
            process = await asyncio.create_subprocess_exec(
                launch.executable,
                env=dict(launch.environment),
                stdout=asyncio.subprocess.PIPE,
            )
            output, _ = await asyncio.wait_for(process.communicate(), 10)
            assert process.returncode == 0
            assert output.strip() == b"True"
        assert outside.read_text() == "original"

    asyncio.run(scenario())


def test_linux_launch_denies_metadata_changes_even_on_read_only_descriptors(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    targets = [tmp_path / "outside", tmp_path / "read-only", checkout / "writable"]
    for target in targets:
        target.write_text("unchanged")
    before = [target.stat() for target in targets]
    cli = _cli(
        tmp_path / "vendor",
        """
import array, ctypes, fcntl, os, sys
from pathlib import Path

def denied(operation):
    try:
        operation()
    except PermissionError:
        return
    raise AssertionError("metadata change was allowed")

libc = ctypes.CDLL(None, use_errno=True)
def file_setattr(path):
    # Linux 6.17's path-based inode flags bypass both open and legacy ioctls.
    attributes = (ctypes.c_uint64 * 3)()
    attributes[0] = 0x80  # FS_XFLAG_NODUMP
    result = libc.syscall(469, -100, os.fsencode(path), ctypes.byref(attributes), 24, 0)
    if result < 0:
        raise OSError(ctypes.get_errno(), "file_setattr")

for path in sys.argv[1:]:
    denied(lambda: file_setattr(path))
    denied(lambda: os.chmod(path, 0))
    denied(lambda: os.chmod(path, 0, dir_fd=-100))
    denied(lambda: os.chown(path, os.getuid(), os.getgid()))
    denied(lambda: os.chown(path, os.getuid(), os.getgid(), follow_symlinks=False))
    denied(lambda: os.utime(path, ns=(0, 0)))
    denied(lambda: os.setxattr(path, "user.cayu-test", b"changed"))
    denied(lambda: os.removexattr(path, "user.cayu-test"))
for path in sys.argv[2:]:
    fd = os.open(path, os.O_RDONLY)
    try:
        denied(lambda: os.fchmod(fd, 0))
        denied(lambda: os.fchown(fd, os.getuid(), os.getgid()))
        denied(lambda: os.utime(fd, ns=(0, 0)))
        denied(lambda: os.setxattr(fd, "user.cayu-test", b"changed"))
        denied(lambda: os.removexattr(fd, "user.cayu-test"))
        # FS_IOC_SETFLAGS: a read-only descriptor must not allow filesystem mutation.
        denied(lambda: fcntl.ioctl(fd, 0x40086602, array.array("l", [0])))
    finally:
        os.close(fd)
Path("new-file").write_text("normal writes still work")
""",
    )

    async def scenario():
        async with linux_agent_launch(
            cli=str(cli),
            cwd=checkout,
            read_only_paths=[targets[1], sys.base_prefix, sys.prefix],
        ) as launch:
            process = await asyncio.create_subprocess_exec(
                launch.executable,
                *(str(target) for target in targets),
                env=dict(launch.environment),
                stderr=asyncio.subprocess.PIPE,
            )
            _, error = await asyncio.wait_for(process.communicate(), 10)
            assert process.returncode == 0, error.decode()

    asyncio.run(scenario())
    for target, original in zip(targets, before, strict=True):
        after = target.stat()
        assert (after.st_mode, after.st_uid, after.st_gid, after.st_mtime_ns) == (
            original.st_mode,
            original.st_uid,
            original.st_gid,
            original.st_mtime_ns,
        )
        assert target.read_text() == "unchanged"
    assert (checkout / "new-file").read_text() == "normal writes still work"
