"""Python standard-library guest control program, executed on the agent lane.

This file is packaged as source and sent over Runner.exec. It has no Cayu or
third-party imports. Payloads, process logs and private paths are never echoed.
"""

import base64
import hashlib
import json
import os
import pathlib
import platform
import stat
import subprocess
import sys
import tarfile


def regular_files(root, limit, process=False):
    found = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        if process and pathlib.Path(directory) == root:
            # Generated restart shell scripts (including their symlink) are
            # never restoration inputs. Retain only images and mapped support.
            dirs[:] = [name for name in dirs if name.endswith("_files")]
            files = [name for name in files if name.endswith(".dmtcp")]
        for name in dirs:
            path = pathlib.Path(directory, name)
            if path.is_symlink():
                raise ValueError("symlink")
            found.append(path)
            if len(found) > limit:
                raise ValueError("file limit")
        for name in files:
            path = pathlib.Path(directory, name)
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError("special file")
            found.append(path)
            if len(found) > limit:
                raise ValueError("file limit")
    return sorted(found)


def inventory(root, request, process=False):
    """Check every deterministic packing limit without writing anything."""
    files = regular_files(root, request["max_files"], process=process)
    # A lower bound on the archive (end-of-archive blocks, one header and
    # block-padded content per entry), so this never rejects what pack accepts.
    estimate = 1024
    for path in files:
        info = path.lstat()
        if not os.access(path, os.R_OK | (os.X_OK if path.is_dir() else 0)):
            raise ValueError("unreadable")
        estimate += 512 + -(-info.st_size // 512) * 512
    if estimate > request["max_artifact_bytes"]:
        raise ValueError("byte limit")
    return files


def pack(root, destination, request, process=False):
    total = 0
    files = regular_files(root, request["max_files"], process=process)
    with tarfile.open(destination, "w", format=tarfile.PAX_FORMAT) as archive:
        for path in files:
            info = path.lstat()
            total += info.st_size
            if total > request["max_artifact_bytes"]:
                raise ValueError("byte limit")
            entry = tarfile.TarInfo(path.relative_to(root).as_posix())
            entry.size = info.st_size
            entry.mode = info.st_mode & 0o777
            if path.is_dir():
                entry.type = tarfile.DIRTYPE
                entry.size = 0
                archive.addfile(entry)
            else:
                with path.open("rb") as source:
                    archive.addfile(entry, source)
            after = path.lstat()
            if (
                after.st_ino,
                after.st_dev,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ) != (info.st_ino, info.st_dev, info.st_size, info.st_mtime_ns, info.st_ctime_ns):
                raise ValueError("file changed")
    if destination.stat().st_size > request["max_artifact_bytes"]:
        raise ValueError("archive limit")
    return {
        "size": destination.stat().st_size,
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }


def unpack(archive, root, request):
    if root.is_symlink() or any(root.iterdir()):
        raise ValueError("target not empty")
    with tarfile.open(archive, "r:") as source:
        seen = set()
        total = 0
        directory_modes = []
        for entry in source:
            parts = pathlib.PurePosixPath(entry.name).parts
            if (
                not (entry.isfile() or entry.isdir())
                or not parts
                or entry.name.startswith("/")
                or any(part in (".", "..") for part in parts)
                or entry.name in seen
                or entry.mode & ~0o777
                or entry.size < 0
                or (entry.isdir() and entry.size != 0)
            ):
                raise ValueError("unsafe archive")
            seen.add(entry.name)
            total += entry.size
            if len(seen) > request["max_files"] or total > request["max_artifact_bytes"]:
                raise ValueError("archive limit")
            target = root.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            for parent in target.parents:
                if parent == root:
                    break
                if parent.is_symlink():
                    raise ValueError("unsafe parent")
            if entry.isdir():
                target.mkdir(mode=0o700, exist_ok=True)
                directory_modes.append((target, entry.mode))
                continue
            data = source.extractfile(entry)
            if data is None:
                raise ValueError("missing file")
            descriptor = os.open(
                target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, entry.mode & 0o777
            )
            with os.fdopen(descriptor, "wb") as destination:
                while block := data.read(1024 * 1024):
                    destination.write(block)
                destination.flush()
                os.fchmod(destination.fileno(), entry.mode)
                os.fsync(destination.fileno())
        for target, mode in reversed(directory_modes):
            target.chmod(mode)


def execute(request):
    root = pathlib.Path(request["root"])
    workspace = pathlib.Path(request["workspace"])
    prefix = pathlib.Path(request["prefix"])
    operation = request["operation"]
    port = str(request["port"])
    images = root / "images"
    if operation == "inspect":
        if os.getuid() == 0:
            raise ValueError("root workload")
        version = subprocess.check_output(
            [str(prefix / "bin/dmtcp_launch"), "--version"], stderr=subprocess.DEVNULL
        ).decode()
        if "4.2.0" not in version:
            raise ValueError("unsupported engine")
        paths = [
            prefix / "bin" / name for name in ("dmtcp_launch", "dmtcp_restart", "dmtcp_command")
        ]
        paths += sorted((prefix / "lib/dmtcp").glob("*.so"))
        barrier = prefix / "lib/dmtcp/libcayu_snapshot.so"
        if barrier not in paths:
            raise ValueError("missing barrier")
        status = pathlib.Path("/proc/self/status").read_text()
        fields = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
        return {
            "kernel": platform.release(),
            "architecture": platform.machine(),
            "libc": platform.libc_ver(),
            "uid": os.getuid(),
            "gid": os.getgid(),
            "capabilities": fields.get("CapEff", "").strip(),
            "no_new_privileges": fields.get("NoNewPrivs", "").strip(),
            "seccomp": fields.get("Seccomp", "").strip(),
            "tools": {
                str(path.relative_to(prefix)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in paths
            },
        }
    if operation == "launch":
        root.mkdir(mode=0o700)
        images.mkdir(mode=0o700)
        if workspace.is_symlink() or not workspace.is_dir():
            raise ValueError("workspace")
        environment = dict(os.environ)
        environment["CAYU_SNAPSHOT_ROOT"] = str(root)
        environment["DMTCP_CHECKPOINT_INTERVAL"] = "0"
        environment["DMTCP_GZIP"] = "0"
        args = [
            str(prefix / "bin/dmtcp_launch"),
            "--new-coordinator",
            "--coord-port",
            port,
            "--ckptdir",
            str(images),
            "--with-plugin",
            str(prefix / "lib/dmtcp/libcayu_snapshot.so"),
            *request["argv"],
        ]
        process = subprocess.Popen(
            args,
            cwd=workspace,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        (root / "pid").write_text(str(process.pid))
        return {"launched": True}
    if operation == "prepare_restore":
        if workspace.is_symlink() or not workspace.is_dir() or any(workspace.iterdir()):
            raise ValueError("restore requires empty workspace")
        root.mkdir(mode=0o700)
        images.mkdir(mode=0o700)
        (root / "restore.operation").write_text(request["id"])
        return {"prepared": True}
    if root.is_symlink() or not root.is_dir():
        raise ValueError("missing managed workload")
    if operation in ("preflight", "checkpoint"):
        # Require exactly one process and no live non-local TCP connections.
        result = subprocess.check_output(
            [str(prefix / "bin/dmtcp_command"), "--coord-port", port, "--status"],
            stderr=subprocess.DEVNULL,
        ).decode()
        if "NUM_PEERS=1" not in result or "RUNNING=yes" not in result:
            raise ValueError("unsupported process group")
        for table in ("/proc/net/tcp", "/proc/net/tcp6"):
            for row in pathlib.Path(table).read_text().splitlines()[1:]:
                columns = row.split()
                remote = columns[2].split(":")[0]
                if columns[3] == "01" and remote not in (
                    "0100007F",
                    "00000000000000000000000001000000",
                ):
                    raise ValueError("external connection")
        # The barrier holds the workload until release. Reject a workspace the
        # pack step would refuse before freezing anything. Checkpoint repeats
        # these checks in case the workload changed after preflight.
        inventory(workspace, request)
        if operation == "preflight":
            return {"eligible": True}
        for name in ("capture.ready", "capture.release"):
            (root / name).unlink(missing_ok=True)
        (root / "capture.operation").write_text(request["id"])
        # A unique-ckpt directory and stale images are excluded by the inventory
        # below; this slice deliberately supports one process only.
        subprocess.run(
            [str(prefix / "bin/dmtcp_command"), "--coord-port", port, "--checkpoint"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return {"submitted": True}
    if operation == "ready":
        kind = request["kind"]
        marker = root / (kind + ".ready")
        receipt = root / (kind + ".operation")
        complete = receipt.read_text() == request["id"] and marker.is_file()
        if kind == "capture":
            found = list(images.glob("*.dmtcp"))
            complete = complete and len(found) == 1 and not list(images.rglob("*.temp"))
        return {"ready": complete}
    if operation == "pack":
        if (
            (root / "capture.operation").read_text() != request["id"]
            or not (root / "capture.ready").is_file()
            or (root / "capture.release").exists()
        ):
            raise ValueError("capture is not held")
        return {
            role: pack(path, root / (role + ".tar"), request, process=role == "process")
            for role, path in (("process", images), ("workspace", workspace))
        }
    if operation == "read":
        path = root / (request["role"] + ".tar")
        with path.open("rb") as source:
            source.seek(request["offset"])
            return {"bytes": base64.b64encode(source.read(request["length"])).decode()}
    if operation == "write":
        path = root / (request["role"] + ".tar")
        if path.is_symlink():
            raise ValueError("unsafe transfer")
        data = base64.b64decode(request["bytes"], validate=True)
        if request["offset"] == 0:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        else:
            descriptor = os.open(path, os.O_WRONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "r+b") as target:
            if target.seek(0, 2) != request["offset"]:
                raise ValueError("transfer offset")
            target.write(data)
            target.flush()
            os.fsync(target.fileno())
        return {"written": len(data)}
    if operation == "restart":
        if (root / "restore.operation").read_text() != request["id"]:
            raise ValueError("restore identity")
        for role, target in (("process", images), ("workspace", workspace)):
            path = root / (role + ".tar")
            if hashlib.sha256(path.read_bytes()).hexdigest() != request["hashes"][role]:
                raise ValueError("transfer integrity")
            unpack(path, target, request)
        found = sorted(images.glob("*.dmtcp"))
        if len(found) != 1:
            raise ValueError("unsupported process set")
        subprocess.Popen(
            [
                str(prefix / "bin/dmtcp_restart"),
                "--new-coordinator",
                "--coord-port",
                port,
                str(found[0]),
            ],
            cwd=workspace,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        return {"submitted": True}
    if operation == "release":
        kind = request["kind"]
        if (root / (kind + ".operation")).read_text() != request["id"] or not (
            root / (kind + ".ready")
        ).is_file():
            raise ValueError("release identity")
        descriptor = os.open(
            root / (kind + ".release"), os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600
        )
        os.fsync(descriptor)
        os.close(descriptor)
        return {"released": True}
    raise ValueError("unsupported operation")


if __name__ == "__main__":
    try:
        request = json.loads(sys.stdin.read())
        print(json.dumps(execute(request)))
    except Exception as error:
        # Neither raw exceptions nor the request can cross the public boundary.
        print(
            json.dumps(
                {
                    "error": "snapshot_guest_operation_failed",
                    "errno": error.errno if isinstance(error, OSError) else None,
                }
            )
        )
        sys.exit(1)
