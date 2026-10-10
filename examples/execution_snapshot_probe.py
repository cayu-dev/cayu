"""Docker/CRIU selected-process proof and AWS Lambda MicroVM capability probe.

This is a disposable substrate investigation, not a production snapshot adapter.
Only report.json is publishable. Private journals, CRIU memory images, resource
identities and fixture contents remain under the mode-0700 state directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = "/workspace/cayu_snapshot_probe"
FILES = ("payload.bin", "action.json", "counter.json")
LABEL = "cayu.snapshot-probe"


class FixtureAcknowledgementLost(Exception):
    """The external fixture committed, but its caller receives no success receipt."""


def external_fixture(directory: Path) -> None:
    with sqlite3.connect(directory / "external.db") as database:
        database.execute("CREATE TABLE effects (call TEXT PRIMARY KEY, count INTEGER)")
        database.execute("INSERT INTO effects VALUES ('external-mutation',1)")
    # The transaction is outside the captured environment and already committed.
    raise FixtureAcknowledgementLost


def write_private_bytes(path: Path, content: bytes) -> None:
    temporary = path.with_suffix(".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def write_private(path: Path, value: dict[str, Any]) -> None:
    write_private_bytes(path, json.dumps(value, indent=2, sort_keys=True).encode())


def sync_tree(path: Path) -> None:
    for item in sorted(path.rglob("*"), reverse=True):
        descriptor = os.open(item, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def verify_restore(
    expected: dict[str, Any], observed: dict[str, Any], digests: dict[str, str], disk: int
) -> None:
    if not expected["quiescent"] or not observed["quiescent"]:
        raise RuntimeError("restore point requires a quiescent writer")
    if expected["token"] != observed["token"]:
        raise RuntimeError("memory-only state was not restored")
    if expected["counter"] != observed["counter"] or disk != observed["counter"]:
        raise RuntimeError("filesystem and process state do not share a restore boundary")
    if expected["digests"] != digests:
        raise RuntimeError("restored fixture integrity mismatch")


def continuation(completed: dict[str, Any], effect: dict[str, Any]) -> dict[str, str]:
    if completed != {"call": "fixture-write", "count": 1}:
        raise RuntimeError("completed action requires reconciliation")
    return {
        "fixture-write": "reuse_recorded_result",
        "external-mutation": "reconcile" if effect["state"] == "outcome_unknown" else "stop",
    }


def docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=60, check=False
    )
    if check and result.returncode:
        if log_directory := os.environ.get("CAYU_EXECUTION_SNAPSHOT_LOG_DIR"):
            Path(log_directory, "command-error.log").write_text(result.stderr)
        raise RuntimeError("Docker probe operation failed; inspect private controller logs")
    return result


def command(container: str, source: str) -> str:
    return docker("exec", container, "sh", "-c", source).stdout


def observe(container: str, *, quiesce: bool = False, resume: bool = False) -> dict[str, Any]:
    path = "/quiesce" if quiesce else "/resume" if resume else "/state"
    method = "POST" if quiesce or resume else "GET"
    source = (
        "import urllib.request; "
        f"r=urllib.request.Request('http://127.0.0.1:8087{path}',method='{method}'); "
        "print(urllib.request.urlopen(r,timeout=2).read().decode())"
    )
    return json.loads(command(container, "python3 -c " + shlex.quote(source)))


def create(state: dict[str, Any], operation: str) -> str:
    name = f"cayu-snapshot-{state['run']}-{operation}"
    docker(
        "run",
        "--name",
        name,
        "--label",
        f"{LABEL}={state['run']}",
        "--label",
        f"{LABEL}.operation={operation}",
        "--network",
        "none",
        "--privileged",
        "--memory",
        "512m",
        "--cpus",
        "2",
        "-d",
        state["image"],
        "sleep",
        "300",
    )
    return name


def scoped(state: dict[str, Any], operation: str | None = None) -> list[str]:
    args = ["ps", "-aq", "--filter", f"label={LABEL}={state['run']}"]
    if operation:
        args.extend(["--filter", f"label={LABEL}.operation={operation}"])
    return docker(*args).stdout.split()


def cleanup(directory: Path) -> None:
    state = load(directory / "state.json")
    identifiers = scoped(state)
    if identifiers:
        docker("rm", "-f", *identifiers)
    if scoped(state):
        raise RuntimeError("probe resource cleanup was not confirmed")


def tree_hashes(path: Path) -> dict[str, str]:
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError("snapshot artifact is unavailable")
    result = {}
    for item in sorted(path.rglob("*")):
        if item.is_symlink() or not (item.is_dir() or item.is_file()):
            raise RuntimeError("unsupported snapshot artifact entry")
        if item.is_file():
            result[str(item.relative_to(path))] = hashlib.sha256(item.read_bytes()).hexdigest()
    return result


def verify_artifact(path: Path, expected: dict[str, str]) -> None:
    if not expected or tree_hashes(path) != expected:
        raise RuntimeError("snapshot artifact integrity mismatch")


def verify_compatibility(expected: dict[str, str], observed: dict[str, str]) -> None:
    if expected != observed:
        raise RuntimeError("snapshot compatibility mismatch")


def compatibility(container: str, image: str) -> dict[str, str]:
    return {
        "image": image,
        "kernel": command(container, "uname -r").strip(),
        "architecture": command(container, "uname -m").strip(),
        "criu": command(container, "criu --version").strip(),
    }


def file_digests(container: str) -> dict[str, str]:
    source = (
        "import hashlib,json; from pathlib import Path; "
        f"r=Path({ROOT!r}); "
        f"print(json.dumps({{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in {FILES!r}}}))"
    )
    return json.loads(command(container, "python3 -c " + shlex.quote(source)))


def copy_files(container: str, destination: Path) -> None:
    destination.mkdir(mode=0o700)
    for name in FILES:
        docker("cp", f"{container}:{ROOT}/{name}", str(destination / name))


def restore_files(container: str, source: Path) -> None:
    command(container, f"mkdir -p {ROOT}")
    for name in FILES:
        docker("cp", str(source / name), f"{container}:{ROOT}/{name}")


def worker(directory: Path, operation: str, crash: str | None) -> None:
    state = load(directory / "state.json")
    journal = directory / f"{operation}.json"
    intent = {"state": "intent", "operation": operation, "active": False}
    write_private(journal, intent)
    if crash == "before-submit":
        os._exit(77)
    if operation.startswith("capture"):
        artifact = directory / operation
        artifact.mkdir(mode=0o700)
        guest_directory = f"/checkpoint-{operation}"
        command(state["source"], f"mkdir {guest_directory}")
        pid = int(command(state["source"], f"cat {ROOT}/guest.pid"))
        command(
            state["source"],
            f"criu dump --tree {pid} --images-dir {guest_directory} "
            f"--leave-running --log-file dump.log -v4 || {{ cat {guest_directory}/dump.log >&2; exit 1; }}",
        )
        docker("cp", f"{state['source']}:{guest_directory}", str(artifact / "process"))
        copy_files(state["source"], artifact / "workspace")
        if crash == "after-accept":
            os._exit(77)
        sync_tree(artifact)
        write_private(journal, {**intent, "state": "published", "hashes": tree_hashes(artifact)})
        return
    artifact = directory / "capture"
    try:
        verify_artifact(artifact, state["hashes"])
    except RuntimeError:
        write_private(journal, {**intent, "state": "rejected"})
        raise
    restored = create(state, operation)
    if crash == "after-accept":
        os._exit(77)
    write_private(journal, {**intent, "state": "verifying", "container": restored})
    verify_compatibility(state["compatibility"], compatibility(restored, state["image"]))
    restore_files(restored, artifact / "workspace")
    docker("cp", str(artifact / "process"), f"{restored}:/checkpoint")
    # The guest owns its session. The new controller also gets an independent
    # session; saved PID/session identities must not collide with its shell.
    command(
        restored,
        "setsid criu restore --images-dir /checkpoint --restore-detached "
        "--manage-cgroups=ignore --log-file restore.log",
    )
    observed = observe(restored)
    disk = json.loads(command(restored, f"cat {ROOT}/counter.json"))["counter"]
    verify_restore(state["expected"], observed, file_digests(restored), disk)
    completed = json.loads(command(restored, f"cat {ROOT}/action.json"))
    plan = continuation(completed, state["external_effect"])
    if plan["external-mutation"] != "reconcile":
        raise RuntimeError("unsafe continuation plan")
    write_private(
        journal,
        {
            **intent,
            "state": "verified",
            "active": True,
            "plan": plan,
            "container": restored,
        },
    )


def run_worker(
    directory: Path, operation: str, crash: str | None = None, *, reject: bool = False
) -> None:
    args = [sys.executable, __file__, "--state-dir", str(directory), "--worker", operation]
    if crash:
        args.extend(["--crash", crash])
    result = subprocess.run(args, capture_output=True, timeout=90, check=False)
    # Private logs include host/container identities and must never be published.
    (directory / f"{operation}.log").write_bytes(result.stdout + result.stderr)
    expected = 77 if crash else 1 if reject else 0
    if result.returncode != expected:
        raise RuntimeError(f"fresh controller failed during {operation}; exit={result.returncode}")


def probe_docker(directory: Path, image: str) -> dict[str, Any]:
    started = time.monotonic()
    image_id = json.loads(docker("image", "inspect", image).stdout)[0]["Id"]
    state: dict[str, Any] = {"run": uuid.uuid4().hex[:12], "image": image_id}
    write_private(directory / "state.json", state)
    report: dict[str, Any] = {
        "backend": "docker-criu",
        "fidelity": "selected-process",
        "checks": {},
    }
    checks = report["checks"]
    try:
        source = create(state, "source")
        state["source"] = source
        write_private(directory / "state.json", state)
        command(source, f"mkdir -p {ROOT}")
        docker(
            "cp",
            str(Path(__file__).with_name("_execution_snapshot_guest.py")),
            f"{source}:{ROOT}/guest.py",
        )
        command(
            source,
            f"setsid python3 {ROOT}/guest.py >/dev/null 2>&1 </dev/null & echo $! >{ROOT}/guest.pid",
        )
        for _ in range(40):
            try:
                before = observe(source)
                break
            except RuntimeError:
                time.sleep(0.05)
        else:
            raise RuntimeError("guest fixture did not become ready")
        time.sleep(0.1)
        paused = observe(source, quiesce=True)
        checks["background_writer_advanced"] = paused["counter"] > before["counter"]
        if not checks["background_writer_advanced"]:
            raise RuntimeError("background writer was not exercised")
        state["expected"] = {**paused, "digests": file_digests(source)}
        state["compatibility"] = compatibility(source, image_id)
        try:
            external_fixture(directory)
        except FixtureAcknowledgementLost:
            state["external_effect"] = {"state": "outcome_unknown"}
        write_private(directory / "state.json", state)

        # Native Engine checkpoint support and guest process CRIU are distinct.
        native = docker("checkpoint", "create", "--leave-running", source, "native", check=False)
        report["docker_engine_checkpoint"] = (
            "captured"
            if native.returncode == 0
            else "experimental_disabled"
            if "experimental features enabled" in native.stderr
            else "unqualified_failure"
        )
        for crash in ("before-submit", "after-accept"):
            operation = f"capture-{crash}"
            run_worker(directory, operation, crash)
            checks[f"{operation}_unpublished"] = (
                load(directory / f"{operation}.json")["state"] == "intent"
            )
            artifact = directory / operation
            checks[f"{operation}_artifact_discovery"] = (
                artifact.is_dir() if crash == "after-accept" else not artifact.exists()
            )
            if artifact.exists():
                shutil.rmtree(artifact)
        capture_started = time.monotonic()
        run_worker(directory, "capture")
        report["capture_and_private_copy_s"] = time.monotonic() - capture_started
        state["hashes"] = load(directory / "capture.json")["hashes"]
        files_started = time.monotonic()
        copy_files(source, directory / "files-only")
        report["files_only_capture_s"] = time.monotonic() - files_started
        report["files_only_bytes"] = sum(
            p.stat().st_size for p in (directory / "files-only").iterdir()
        )
        report["process_checkpoint_bytes"] = sum(
            p.stat().st_size for p in (directory / "capture/process").iterdir() if p.is_file()
        )
        checks["published_snapshot_integrity"] = bool(state["hashes"])
        # Alter both memory and disk after the checkpoint and remove the source.
        observe(source, resume=True)
        time.sleep(0.1)
        checks["source_advanced_after_capture"] = observe(source)["counter"] > paused["counter"]
        command(source, f"printf changed >{ROOT}/payload.bin")
        docker("rm", "-f", source)
        checks["source_allocation_removed"] = not scoped(state, "source")
        write_private(directory / "state.json", state)
        for crash in ("before-submit", "after-accept"):
            operation = f"restore-{crash}"
            run_worker(directory, operation, crash)
            checks[f"{operation}_inactive"] = (
                load(directory / f"{operation}.json")["active"] is False
            )
            discovered = scoped(state, operation)
            checks[f"{operation}_allocation_discovery"] = bool(discovered) == (
                crash == "after-accept"
            )
            if discovered:
                docker("rm", "-f", *discovered)
        restore_started = time.monotonic()
        run_worker(directory, "restore")
        report["fresh_controller_restore_and_verify_s"] = time.monotonic() - restore_started
        checks["memory_files_and_boundary_verified"] = load(directory / "restore.json")["active"]
        run_worker(directory, "restore-again")
        checks["immutable_snapshot_reusable"] = load(directory / "restore-again.json")["active"]
        checks["completed_action_reused"] = (
            load(directory / "restore.json")["plan"]["fixture-write"] == "reuse_recorded_result"
        )
        with sqlite3.connect(directory / "external.db") as database:
            checks["external_unknown_effect_not_repeated"] = database.execute(
                "SELECT count FROM effects"
            ).fetchall() == [(1,)]
        files_restore_started = time.monotonic()
        baseline = create(state, "files-only")
        restore_files(baseline, directory / "files-only")
        checks["files_only_integrity"] = file_digests(baseline) == state["expected"]["digests"]
        refusal = command(
            baseline,
            "python3 -c 'import socket; s=socket.socket(); print(s.connect_ex((\"127.0.0.1\",8087))); s.close()'",
        ).strip()
        checks["files_only_process_absent"] = refusal == "111"
        report["files_only_restore_and_verify_s"] = time.monotonic() - files_restore_started

        # Corruption is rejected before allocation or process activation.
        image_path = directory / "capture/process/inventory.img"
        original = image_path.read_bytes()
        image_path.write_bytes(b"corrupt" + original)
        run_worker(directory, "restore-corrupt", reject=True)
        checks["corruption_rejected_before_allocation"] = load(directory / "restore-corrupt.json")[
            "state"
        ] == "rejected" and not scoped(state, "restore-corrupt")
        image_path.write_bytes(original)
        shutil.rmtree(directory / "capture")
        run_worker(directory, "restore-deleted", reject=True)
        checks["deleted_artifact_restore_refused"] = (
            load(directory / "restore-deleted.json")["state"] == "rejected"
        )
        report["environment"] = state["compatibility"]
    finally:
        cleanup(directory)
        checks["scoped_cleanup_confirmed"] = not scoped(state)
    if not all(checks.values()):
        raise RuntimeError("a substrate proof check failed")
    report["total_s"] = time.monotonic() - started
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--image", default="cayu-execution-snapshot-criu:probe")
    parser.add_argument("--worker")
    parser.add_argument("--crash", choices=["before-submit", "after-accept"])
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()
    directory = args.state_dir.resolve()
    if args.worker:
        worker(directory, args.worker, args.crash)
        return
    if args.cleanup:
        cleanup(directory)
        return
    directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    os.umask(0o077)
    os.environ["CAYU_EXECUTION_SNAPSHOT_LOG_DIR"] = str(directory)
    report = probe_docker(directory, args.image)
    report["probe_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["guest_sha256"] = hashlib.sha256(
        Path(__file__).with_name("_execution_snapshot_guest.py").read_bytes()
    ).hexdigest()
    write_private(directory / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
