"""Use Cayu's execution-snapshot APIs through fresh controller processes.

Docker: build examples/execution_snapshot_dmtcp/Dockerfile, then run this module
with --state-dir PRIVATE_DIRECTORY. AWS: supply a private --aws-config JSON with
profile, region, image, image_version, and optionally tools_archive for disposable
image qualification. Bootstrap modifies only the newly owned allocations.
Journals and snapshot stores stay private; only the bounded report is shareable.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from examples.aws.lambda_microvm_dmtcp_snapshot_probe import upload
from examples.aws.lambda_microvm_execution_snapshot_probe import cleanup, client, run_options
from examples.execution_snapshot_probe import load, write_private

from cayu import (
    CayuApp,
    DockerRunner,
    Environment,
    EnvironmentSpec,
    ExecCommand,
    LambdaMicroVMRunner,
    LocalArtifactStore,
)
from cayu.environments.dmtcp_snapshots import DmtcpExecutionSnapshotAdapter
from cayu.environments.snapshot_lifecycle import ensure_execution_snapshot_binding
from cayu.environments.snapshots import ExecutionSnapshotPolicy
from cayu.runners.docker_workload import (
    DockerImageIdentity,
    DockerTmpfsMount,
    DockerWorkloadRestrictions,
)
from cayu.sessions.base import RunRequest
from cayu.sessions.records import SessionIdentity
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.workspaces import RunnerWorkspace


async def system(runner, command, stdin=None):
    result = await runner.exec_system(command, stdin=stdin, timeout_s=120)
    if result.exit_code or result.timed_out:
        raise RuntimeError(
            "Disposable snapshot image setup failed; inspect private guest diagnostics."
        )
    return result.stdout


async def bootstrap(runner, archive):
    """Qualification only. Production images bake the same tools and plugin in."""
    await system(
        runner,
        ExecCommand.process("dnf", "install", "-y", "gcc", "libatomic", "libstdc++", "tar", "gzip"),
    )
    content = Path(archive).read_bytes()
    await upload(runner, "/tmp/cayu-snapshot-tools.tar.gz", content)
    await system(
        runner,
        ExecCommand.bash(
            "mkdir -p /opt/cayu-dmtcp && tar -xzf /tmp/cayu-snapshot-tools.tar.gz -C /opt/cayu-dmtcp && mkdir -p /opt/cayu-dmtcp/include/dmtcp && cp /opt/cayu-dmtcp/include/version.h /opt/cayu-dmtcp/include/dmtcp/version.h"
        ),
    )
    await upload(
        runner,
        "/tmp/cayu-snapshot-barrier.c",
        Path("src/cayu/environments/_snapshot_barrier.c").read_bytes(),
    )
    await system(
        runner,
        ExecCommand.process(
            "gcc",
            "-shared",
            "-fPIC",
            "-O2",
            "-I/opt/cayu-dmtcp/include",
            "/tmp/cayu-snapshot-barrier.c",
            "-o",
            "/opt/cayu-dmtcp/lib/dmtcp/libcayu_snapshot.so",
        ),
    )
    await system(
        runner, ExecCommand.process("chmod", "755", "/opt/cayu-dmtcp/lib/dmtcp/libcayu_snapshot.so")
    )


async def allocate(directory, configuration):
    if configuration["backend"] == "docker":
        image = configuration["image"]
        digest = subprocess.check_output(
            ["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True
        ).strip()
        return await DockerRunner.create(
            "cayu-snapshot-feature-" + uuid.uuid4().hex[:12],
            image=image,
            image_identity=DockerImageIdentity(reference=image, content_digest=digest),
            workload_restrictions=DockerWorkloadRestrictions(
                tmpfs=(
                    DockerTmpfsMount(
                        target="/tmp", size_bytes=256 * 1024**2, mode=0o1777, noexec=False
                    ),
                    DockerTmpfsMount(
                        target="/workspace", size_bytes=256 * 1024**2, mode=0o750, noexec=False
                    ),
                )
            ),
            network="none",
            replace=False,
            credential_mode="trusted_tool",
            allow_raw_secret_env=False,
            cancellation_cleanup="sandbox",
            timeout_cleanup="sandbox",
        )
    directory.mkdir(mode=0o700)
    state: dict[str, Any] = {
        **configuration,
        "install_criu": bool(configuration.get("tools_archive")),
        "submitted": True,
        "client_token": "cayu-snapshot-feature-" + uuid.uuid4().hex,
        "submission_deadline": time.time() + 480,
    }
    write_private(directory / "state.json", state)
    state["allocation"] = client(state).run_microvm(**run_options(state))["microvmId"]
    write_private(directory / "state.json", state)
    runner = await LambdaMicroVMRunner.from_existing(
        state["allocation"],
        profile_name=state["profile"],
        region_name=state["region"],
        close_action="none",
    )
    if configuration.get("tools_archive"):
        await bootstrap(runner, configuration["tools_archive"])
    return runner


async def observe(runner, quiesce=False):
    program = (
        "import urllib.request; r=urllib.request.Request('http://127.0.0.1:8087/"
        + ("quiesce',method='POST')" if quiesce else "state')")
        + "; print(urllib.request.urlopen(r,timeout=1).read().decode())"
    )
    for _ in range(40):
        result = await runner.exec(ExecCommand.process("python3", "-c", program))
        if result.exit_code == 0:
            value = json.loads(result.stdout)
            return {
                "memory_sha256": hashlib.sha256(value["token"].encode()).hexdigest(),
                "counter": value["counter"],
                "quiescent": value["quiescent"],
            }
        await asyncio.sleep(0.1)
    raise RuntimeError("Restored workload did not become available.")


async def worker(directory, phase):
    configuration = load(directory / "configuration.json")
    owned = directory / (phase + "-allocation")
    runner = None
    store = SQLiteSessionStore(directory / "sessions.db")
    try:
        runner = await allocate(owned, configuration)
        owned_workspace = "/workspace/cayu_snapshot_probe"
        prepared = await runner.exec(
            ExecCommand.process(
                "python3",
                "-I",
                "-c",
                "from pathlib import Path; Path('/workspace/cayu_snapshot_probe').mkdir(mode=0o700)",
            )
        )
        if prepared.exit_code:
            raise RuntimeError("Disposable workload workspace was not empty and exclusively owned.")
        adapter = await DmtcpExecutionSnapshotAdapter.create(
            runner,
            workload_id="cayu-snapshot-feature",
            workspace_path=owned_workspace,
        )
        workspace = RunnerWorkspace(
            runner, cwd="cayu_snapshot_probe", workspace_id="snapshot-feature"
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="sandbox"),
                runner=runner,
                workspace=workspace,
                execution_snapshot_adapter=adapter,
            )
        )
        registered = app.get_environment("sandbox")
        artifacts = LocalArtifactStore(directory / "private-snapshots")
        if phase == "capture":
            await workspace.write_bytes(
                "guest.py", Path("examples/_execution_snapshot_guest.py").read_bytes()
            )
            await adapter.launch(ExecCommand.process("python3", owned_workspace + "/guest.py"))
            await observe(runner)
            await asyncio.sleep(0.1)
            expected = await observe(runner, quiesce=True)
            session = await store.create(
                RunRequest(
                    agent_name="snapshot-example", session_id="snapshot-feature", messages=[]
                ),
                identity=SessionIdentity(provider_name="example", model="example"),
            )
            await store.checkpoint(
                session.id,
                {
                    "completed_tool_result": {"call_id": "fixture-write", "count": 1},
                    "external_effect": {"state": "outcome_unknown"},
                },
            )
            started = time.monotonic()
            record = await app.capture_execution_snapshot(
                session.id,
                "sandbox",
                snapshot_store=artifacts,
                expected_run_epoch=session.run_epoch,
                expected_generation=registered.binding_generation_id,
                idempotency_key="capture",
                policy=ExecutionSnapshotPolicy(timeout_seconds=180),
            )
            write_private(
                directory / "capture.json",
                {
                    "snapshot_id": record.id,
                    "source_generation": registered.binding_generation_id,
                    "expected": expected,
                    "bytes": record.total_bytes,
                    "capture_seconds": time.monotonic() - started,
                },
            )
        else:
            source = load(directory / "capture.json")
            generation = (
                source["source_generation"]
                if phase == "restore"
                else load(directory / "restore.json")["target_generation"]
            )
            session = await store.load("snapshot-feature")
            assert session is not None
            started = time.monotonic()
            # Fresh allocation is blocked before restoration is durably settled.
            try:
                await ensure_execution_snapshot_binding(store, session, registered)
            except Exception:
                pass
            else:
                raise RuntimeError("Fresh allocation was exposed before restore.")
            await app.restore_execution_snapshot(
                session.id,
                "sandbox",
                source["snapshot_id"],
                snapshot_store=artifacts,
                expected_run_epoch=session.run_epoch,
                expected_generation=generation,
                idempotency_key=phase,
                policy=ExecutionSnapshotPolicy(timeout_seconds=180),
            )
            await ensure_execution_snapshot_binding(store, session, registered)
            observed = await observe(runner)
            if observed != source["expected"]:
                raise RuntimeError("Process memory did not match the captured workload.")
            disk = json.loads((await workspace.read_bytes("counter.json")).content)
            action = json.loads((await workspace.read_bytes("action.json")).content)
            checkpoint = await store.load_checkpoint(session.id)
            assert checkpoint is not None
            if (
                disk["counter"] != observed["counter"]
                or action["count"] != 1
                or checkpoint["completed_tool_result"]["count"] != 1
                or checkpoint["external_effect"]["state"] != "outcome_unknown"
            ):
                raise RuntimeError("Restored files/controller/effects are inconsistent.")
            write_private(
                directory / (phase + ".json"),
                {
                    "verified": True,
                    "target_generation": registered.binding_generation_id,
                    "restore_seconds": time.monotonic() - started,
                },
            )
    finally:
        try:
            if runner is not None:
                await runner.close()
        finally:
            if configuration["backend"] == "aws" and owned.exists():
                cleanup(owned)
            await store.close()
    write_private(directory / (phase + "-cleanup.json"), {"confirmed": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--aws-config", type=Path)
    parser.add_argument("--image", default="cayu-snapshot-dmtcp:local")
    parser.add_argument("--worker", choices=("capture", "restore", "restore-again"))
    args = parser.parse_args()
    os.umask(0o077)
    directory = args.state_dir.resolve()
    if args.worker:
        asyncio.run(worker(directory, args.worker))
        return
    directory.mkdir(mode=0o700)
    configuration = (
        {"backend": "docker", "image": args.image}
        if args.aws_config is None
        else {**load(args.aws_config), "backend": "aws"}
    )
    write_private(directory / "configuration.json", configuration)
    for phase in ("capture", "restore", "restore-again"):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "examples.execution_snapshot_dmtcp_live",
                "--state-dir",
                str(directory),
                "--worker",
                phase,
            ],
            capture_output=True,
            timeout=540,
            check=False,
        )
        (directory / (phase + ".log")).write_bytes(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError("Snapshot feature worker failed; inspect private journals.")
    report = {
        "backend": configuration["backend"],
        "fidelity": "selected_processes",
        "agent_uid": 1000,
        "fresh_cayu_process_restore": load(directory / "restore.json")["verified"],
        "same_artifact_restored_twice": load(directory / "restore-again.json")["verified"],
        "all_owned_allocations_disposed": all(
            load(directory / (phase + "-cleanup.json"))["confirmed"]
            for phase in ("capture", "restore", "restore-again")
        ),
        "bytes": load(directory / "capture.json")["bytes"],
        "capture_seconds": load(directory / "capture.json")["capture_seconds"],
        "restore_seconds": load(directory / "restore.json")["restore_seconds"],
    }
    write_private(directory / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
