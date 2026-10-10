"""Disposable DMTCP process-memory capture/restore proof on staging Lambda MicroVM.

This is a trusted synthetic-process experiment, not a production adapter.
Only report.json may be published. Journals and checkpoint contents stay private.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import shlex
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from examples.aws.lambda_microvm_execution_snapshot_probe import cleanup, client, run_options
from examples.execution_snapshot_probe import (
    FILES,
    ROOT,
    FixtureAcknowledgementLost,
    continuation,
    external_fixture,
    load,
    sync_tree,
    tree_hashes,
    verify_artifact,
    verify_compatibility,
    verify_restore,
    write_private,
    write_private_bytes,
)

from cayu import ExecCommand, LambdaMicroVMRunner

SOURCE_SHA256 = "043410566fd7c09f21e0ec485cf72e9481722629ea877650142b7505e0d7f2d2"
PREFIX = "/opt/cayu-dmtcp"
CHECKPOINT = "/tmp/cayu-dmtcp-checkpoint"
CHUNK = 1024 * 1024
TRANSFER_LIMIT = 64 * CHUNK


def checkpoint_ready(inventory: dict[str, list[str]], status: str) -> bool:
    fields = dict(line.strip().split("=", 1) for line in status.splitlines() if "=" in line)
    return (
        len(inventory["images"]) == 1
        and not inventory["temporary"]
        and fields.get("NUM_PEERS") == "1"
        and fields.get("RUNNING") == "yes"
    )


async def command(
    runner: LambdaMicroVMRunner, source: str, *, timeout: int = 120, stdin: str | None = None
) -> str:
    result = await runner.exec_system(
        ExecCommand.bash(source), stdin=stdin, timeout_s=timeout, output_limit_bytes=2 * CHUNK
    )
    if (
        result.exit_code != 0
        or result.timed_out
        or result.stdout_truncated
        or result.stderr_truncated
    ):
        raise RuntimeError(
            f"guest DMTCP command failed: exit={result.exit_code}, timed_out={result.timed_out}"
        )
    return result.stdout


async def spawn(runner: LambdaMicroVMRunner, arguments: list[str], log: str) -> None:
    program = (
        "import subprocess; "
        f"f=open({log!r},'wb'); "
        f"p=subprocess.Popen({arguments!r},stdin=subprocess.DEVNULL,stdout=f,stderr=f,"
        "start_new_session=True,close_fds=True); print(p.pid)"
    )
    await command(runner, "python3 -c " + shlex.quote(program))


async def download(runner: LambdaMicroVMRunner, source: str, destination: Path) -> None:
    size = int(await command(runner, "stat -c %s " + shlex.quote(source)))
    if not 0 < size <= TRANSFER_LIMIT:
        raise RuntimeError("probe transfer exceeds its declared bound")
    content = bytearray()
    for offset in range(0, size, CHUNK):
        program = (
            "import base64; "
            f"f=open({source!r},'rb'); f.seek({offset}); "
            f"print(base64.b64encode(f.read({CHUNK})).decode())"
        )
        block = base64.b64decode(
            (await command(runner, "python3 -c " + shlex.quote(program))).strip(), validate=True
        )
        if len(block) != min(CHUNK, size - offset):
            raise RuntimeError("probe transfer was incomplete")
        content.extend(block)
    expected = (await command(runner, "sha256sum " + shlex.quote(source))).split()[0]
    if hashlib.sha256(content).hexdigest() != expected:
        raise RuntimeError("downloaded probe artifact integrity mismatch")
    write_private_bytes(destination, bytes(content))


async def upload(runner: LambdaMicroVMRunner, destination: str, content: bytes) -> None:
    if not 0 < len(content) <= TRANSFER_LIMIT:
        raise RuntimeError("probe transfer exceeds its declared bound")
    await command(runner, "mkdir -p " + shlex.quote(str(Path(destination).parent)))
    for offset in range(0, len(content), CHUNK):
        program = (
            "import base64,os,sys; "
            f"f=open({destination!r},{'wb' if offset == 0 else 'ab'!r}); "
            "f.write(base64.b64decode(sys.stdin.buffer.read(),validate=True)); "
            f"f.close(); os.chmod({destination!r},0o600)"
        )
        await command(
            runner,
            "python3 -c " + shlex.quote(program),
            stdin=base64.b64encode(content[offset : offset + CHUNK]).decode(),
        )
    expected = hashlib.sha256(content).hexdigest()
    if (await command(runner, "sha256sum " + shlex.quote(destination))).split()[0] != expected:
        raise RuntimeError("uploaded probe artifact integrity mismatch")


async def observe(
    runner: LambdaMicroVMRunner, *, quiesce: bool = False, resume: bool = False
) -> dict[str, Any]:
    path = "/quiesce" if quiesce else "/resume" if resume else "/state"
    method = "POST" if quiesce or resume else "GET"
    program = (
        "import urllib.request; "
        f"r=urllib.request.Request('http://127.0.0.1:8087{path}',method={method!r}); "
        "print(urllib.request.urlopen(r,timeout=2).read().decode())"
    )
    return json.loads(await command(runner, "python3 -c " + shlex.quote(program)))


async def ready(runner: LambdaMicroVMRunner) -> dict[str, Any]:
    for _ in range(40):
        try:
            return await observe(runner)
        except RuntimeError:
            await asyncio.sleep(0.1)
    raise RuntimeError("DMTCP fixture did not become ready; inspect private guest logs")


async def file_digests(runner: LambdaMicroVMRunner) -> dict[str, str]:
    program = (
        "import hashlib,json; from pathlib import Path; "
        f"r=Path({ROOT!r}); "
        f"print(json.dumps({{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in {FILES!r}}}))"
    )
    return json.loads(await command(runner, "python3 -c " + shlex.quote(program)))


async def compatibility(runner: LambdaMicroVMRunner, state: dict[str, Any]) -> dict[str, str]:
    return {
        "image": state["image"],
        "image_version": state["image_version"],
        "kernel": (await command(runner, "uname -r")).strip(),
        "architecture": (await command(runner, "uname -m")).strip(),
        "glibc": (await command(runner, "rpm -q glibc")).strip(),
        "runtime_libraries": (await command(runner, "rpm -q libatomic libstdc++")).strip(),
        "dmtcp": (await command(runner, f"{PREFIX}/bin/dmtcp_launch --version")).strip(),
    }


async def worker(directory: Path, operation: str) -> None:
    state = load(directory / "state.json")
    artifact = directory / "artifact"
    if operation != "capture":
        verify_artifact(artifact, state["hashes"])
    owned = directory / operation
    owned.mkdir(mode=0o700)
    allocation = {
        **{k: state[k] for k in ("profile", "region", "image", "image_version")},
        "install_criu": True,  # Shared allocation helper: dependency-install internet egress.
        "submitted": False,
        "client_token": "cayu-dmtcp-" + uuid.uuid4().hex,
        "submission_deadline": time.time() + 480,
    }
    write_private(owned / "state.json", allocation)
    write_private(owned / "operation.json", {"state": "intent", "active": False})
    runner = None
    try:
        control = client(allocation)
        allocation["submitted"] = True
        write_private(owned / "state.json", allocation)
        allocation["allocation"] = control.run_microvm(**run_options(allocation))["microvmId"]
        write_private(owned / "state.json", allocation)
        runner = await LambdaMicroVMRunner.from_existing(
            allocation["allocation"],
            profile_name=state["profile"],
            region_name=state["region"],
            close_action="none",
        )
        await command(
            runner,
            "dnf install -y libatomic libstdc++ tar gzip >/tmp/cayu-dmtcp-runtime-install.log 2>&1",
        )
        if operation == "capture":
            if state.get("prebuilt"):
                await upload(
                    runner, "/tmp/cayu-dmtcp-tools.tar.gz", (artifact / "tools.tar.gz").read_bytes()
                )
                await command(
                    runner,
                    f"mkdir -p {PREFIX} && tar -xzf /tmp/cayu-dmtcp-tools.tar.gz -C {PREFIX}",
                )
            else:
                await command(
                    runner,
                    "dnf install -y gcc gcc-c++ make tar gzip >/tmp/cayu-dmtcp-install.log 2>&1",
                )
                await command(
                    runner,
                    "curl -fsSL https://github.com/dmtcp/dmtcp/archive/refs/tags/v4.2.0.tar.gz "
                    "-o /tmp/dmtcp.tar.gz && "
                    f"echo '{SOURCE_SHA256}  /tmp/dmtcp.tar.gz' | sha256sum -c - && "
                    "tar -xzf /tmp/dmtcp.tar.gz -C /tmp && cd /tmp/dmtcp-4.2.0 && "
                    f"CFLAGS='-O2' CXXFLAGS='-O2' ./configure --prefix={PREFIX} "
                    ">/tmp/cayu-dmtcp-configure.log 2>&1 && "
                    "make -j2 >/tmp/cayu-dmtcp-build.log 2>&1 && "
                    "make install >/tmp/cayu-dmtcp-install-build.log 2>&1",
                    timeout=240,
                )
                artifact.mkdir(mode=0o700)
                await command(
                    runner, f"tar --dereference -czf /tmp/cayu-dmtcp-tools.tar.gz -C {PREFIX} ."
                )
                await download(runner, "/tmp/cayu-dmtcp-tools.tar.gz", artifact / "tools.tar.gz")
            await upload(
                runner,
                f"{ROOT}/guest.py",
                Path(__file__).parents[1].joinpath("_execution_snapshot_guest.py").read_bytes(),
            )
            await command(runner, f"mkdir -p {CHECKPOINT}")
            await spawn(
                runner,
                [
                    f"{PREFIX}/bin/dmtcp_launch",
                    "--new-coordinator",
                    "--coord-port",
                    "7779",
                    "--ckptdir",
                    CHECKPOINT,
                    "python3",
                    f"{ROOT}/guest.py",
                ],
                "/tmp/cayu-dmtcp-guest.log",
            )
            before = await ready(runner)
            await asyncio.sleep(0.1)
            paused_at = time.monotonic()
            expected = await observe(runner, quiesce=True)
            expected["digests"] = await file_digests(runner)
            if expected["counter"] <= before["counter"]:
                raise RuntimeError("background writer was not exercised")
            state["compatibility"] = await compatibility(runner, state)
            state["expected"] = expected
            write_private(directory / "state.json", state)
            try:
                external_fixture(directory)
            except FixtureAcknowledgementLost:
                state["external_effect"] = {"state": "outcome_unknown"}
            capture_started = time.monotonic()
            await command(
                runner,
                f"{PREFIX}/bin/dmtcp_command --coord-port 7779 --bcheckpoint >/tmp/cayu-dmtcp-command.log 2>&1",
            )
            files = []
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                inventory = json.loads(
                    await command(
                        runner,
                        "python3 -c "
                        + shlex.quote(
                            f"import json; from pathlib import Path; r=Path({CHECKPOINT!r}); "
                            "print(json.dumps({'images':[str(p) for p in r.rglob('*.dmtcp')],"
                            "'temporary':[str(p) for p in r.rglob('*.temp')]}))"
                        ),
                    )
                )
                status = await command(
                    runner, f"{PREFIX}/bin/dmtcp_command --coord-port 7779 --status"
                )
                if checkpoint_ready(inventory, status):
                    files = inventory["images"]
                    break
                await asyncio.sleep(0.1)
            if not files:
                status = await command(
                    runner, f"{PREFIX}/bin/dmtcp_command --coord-port 7779 --status"
                )
                write_private_bytes(owned / "status.log", status.encode())
                inventory = await command(
                    runner,
                    f"find {CHECKPOINT} {ROOT} /tmp -maxdepth 3 -name '*ckpt*' -o -name '*.dmtcp*'",
                )
                write_private_bytes(owned / "inventory.log", inventory.encode())
                raise RuntimeError("DMTCP did not produce a process checkpoint")
            images = artifact / "process"
            images.mkdir(mode=0o700)
            program = (
                "import json; from pathlib import Path; "
                f"r=Path({CHECKPOINT!r}); names={files!r}; "
                "support={Path(p).stem+'_files' for p in names}; "
                "entries=[p for p in r.rglob('*') if p.parts[len(r.parts)] in support "
                "or str(p) in names]; "
                "assert not any(p.is_symlink() for p in entries); "
                "files=[p for p in entries if p.is_file()]; "
                "assert len(files)<=128 and sum(p.stat().st_size for p in files)<=128*1024*1024; "
                "print(json.dumps([str(p.relative_to(r)) for p in files]))"
            )
            captured_files = json.loads(await command(runner, "python3 -c " + shlex.quote(program)))
            for relative in captured_files:
                destination = images / relative
                destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                await download(runner, f"{CHECKPOINT}/{relative}", destination)
            workspace = artifact / "workspace"
            workspace.mkdir(mode=0o700)
            for name in FILES:
                await download(runner, f"{ROOT}/{name}", workspace / name)
            state["capture_and_copy_s"] = time.monotonic() - capture_started
            await observe(runner, resume=True)
            state["quiescence_to_resume_s"] = time.monotonic() - paused_at
            await asyncio.sleep(0.1)
            if (await observe(runner))["counter"] <= expected["counter"]:
                raise RuntimeError("source did not advance after capture")
            await command(runner, f"printf changed >{ROOT}/payload.bin")
            sync_tree(artifact)
            state["hashes"] = tree_hashes(artifact)
            write_private(directory / "state.json", state)
            write_private(owned / "operation.json", {"state": "published", "active": False})
        else:
            await upload(
                runner, "/tmp/cayu-dmtcp-tools.tar.gz", (artifact / "tools.tar.gz").read_bytes()
            )
            await command(
                runner, f"mkdir -p {PREFIX} && tar -xzf /tmp/cayu-dmtcp-tools.tar.gz -C {PREFIX}"
            )
            verify_compatibility(state["compatibility"], await compatibility(runner, state))
            for name in FILES:
                await upload(runner, f"{ROOT}/{name}", (artifact / "workspace" / name).read_bytes())
            images = sorted((artifact / "process").glob("*.dmtcp"))
            for image in sorted((artifact / "process").rglob("*")):
                if image.is_file():
                    relative = image.relative_to(artifact / "process")
                    await upload(runner, f"{CHECKPOINT}/{relative}", image.read_bytes())
            restore_started = time.monotonic()
            await spawn(
                runner,
                [
                    f"{PREFIX}/bin/dmtcp_restart",
                    "--new-coordinator",
                    "--coord-port",
                    "7779",
                    *[f"{CHECKPOINT}/{image.name}" for image in images],
                ],
                "/tmp/cayu-dmtcp-restore.log",
            )
            observed = await ready(runner)
            disk = json.loads(await command(runner, f"cat {ROOT}/counter.json"))["counter"]
            verify_restore(state["expected"], observed, await file_digests(runner), disk)
            completed = json.loads(await command(runner, f"cat {ROOT}/action.json"))
            plan = continuation(completed, state["external_effect"])
            write_private(
                owned / "operation.json",
                {
                    "state": "verified",
                    "active": True,
                    "plan": plan,
                    "restore_and_verify_s": time.monotonic() - restore_started,
                },
            )
    except BaseException:
        if runner:
            try:
                output = await command(runner, "cat /tmp/dmtcp-4.2.0/config.log")
                write_private_bytes(owned / "config.log", output.encode())
            except Exception:
                pass
            for name in (
                "guest",
                "restore",
                "configure",
                "build",
                "install",
                "install-build",
                "command",
            ):
                try:
                    output = await command(runner, f"cat /tmp/cayu-dmtcp-{name}.log")
                    write_private_bytes(owned / f"{name}.log", output.encode())
                except Exception:
                    pass
        raise
    finally:
        try:
            if runner:
                await runner.close()
        finally:
            cleanup(owned)
        write_private(owned / "cleanup.json", {"confirmed": True})


def run_worker(directory: Path, operation: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.aws.lambda_microvm_dmtcp_snapshot_probe",
            "--state-dir",
            str(directory),
            "--worker",
            operation,
        ],
        capture_output=True,
        timeout=540,
        check=False,
    )
    write_private_bytes(directory / f"{operation}.log", result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"DMTCP {operation} failed; inspect private journals")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--profile")
    parser.add_argument("--region")
    parser.add_argument("--image")
    parser.add_argument("--image-version")
    parser.add_argument("--tools-archive", type=Path)
    parser.add_argument("--worker", choices=("capture", "restore", "restore-again"))
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    directory = args.state_dir.resolve()
    if args.worker:
        asyncio.run(worker(directory, args.worker))
        return
    if args.cleanup:
        for path in directory.glob("*/state.json"):
            cleanup(path.parent)
        return
    if not all((args.profile, args.region, args.image, args.image_version)):
        parser.error("provide --profile, --region, --image, and --image-version")
    prebuilt = args.tools_archive.read_bytes() if args.tools_archive else None
    if prebuilt and len(prebuilt) > TRANSFER_LIMIT:
        parser.error("tool archive exceeds the transfer bound")
    directory.mkdir(mode=0o700)
    if prebuilt:
        (directory / "artifact").mkdir(mode=0o700)
        write_private_bytes(directory / "artifact/tools.tar.gz", prebuilt)
    write_private(
        directory / "state.json",
        {
            "profile": args.profile,
            "region": args.region,
            "image": args.image,
            "image_version": args.image_version,
            "prebuilt": bool(prebuilt),
        },
    )
    started = time.monotonic()
    try:
        for operation in ("capture", "restore", "restore-again"):
            run_worker(directory, operation)
        state = load(directory / "state.json")
        with sqlite3.connect(directory / "external.db") as database:
            external_count = database.execute("SELECT count FROM effects").fetchall()
        report = {
            "backend": "aws-lambda-microvm-dmtcp",
            "fidelity": "selected-process",
            "checks": {
                "source_termination_confirmed": load(directory / "capture/cleanup.json")[
                    "confirmed"
                ],
                "fresh_controller_memory_and_files_restored": load(
                    directory / "restore/operation.json"
                )["active"],
                "same_artifact_restored_twice": load(directory / "restore-again/operation.json")[
                    "active"
                ],
                "scoped_cleanup_confirmed": all(
                    load(p)["confirmed"] for p in directory.glob("*/cleanup.json")
                ),
                "completed_action_reused": load(directory / "restore/operation.json")["plan"][
                    "fixture-write"
                ]
                == "reuse_recorded_result",
                "external_unknown_effect_not_repeated": external_count == [(1,)]
                and load(directory / "restore/operation.json")["plan"]["external-mutation"]
                == "reconcile",
            },
            "environment": {k: v for k, v in state["compatibility"].items() if k != "image"},
            "capture_and_copy_s": state["capture_and_copy_s"],
            "quiescence_to_resume_s": state["quiescence_to_resume_s"],
            "fresh_controller_restore_and_verify_s": load(directory / "restore/operation.json")[
                "restore_and_verify_s"
            ],
            "process_checkpoint_bytes": sum(
                p.stat().st_size for p in (directory / "artifact/process").rglob("*") if p.is_file()
            ),
            "files_bytes": sum(
                p.stat().st_size for p in (directory / "artifact/workspace").iterdir()
            ),
            "tool_archive_bytes": (directory / "artifact/tools.tar.gz").stat().st_size,
            "configured_source_archive_sha256": SOURCE_SHA256,
            "built_in_probe": not state["prebuilt"],
            "tool_archive_sha256": hashlib.sha256(
                (directory / "artifact/tools.tar.gz").read_bytes()
            ).hexdigest(),
            "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "guest_sha256": hashlib.sha256(
                Path(__file__).parents[1].joinpath("_execution_snapshot_guest.py").read_bytes()
            ).hexdigest(),
            "total_s": time.monotonic() - started,
        }
        if not all(report["checks"].values()):
            raise RuntimeError("DMTCP probe verification failed")
        write_private(directory / "report.json", report)
        print(json.dumps(report, indent=2))
    finally:
        for path in directory.glob("*/state.json"):
            cleanup(path.parent)


if __name__ == "__main__":
    main()
