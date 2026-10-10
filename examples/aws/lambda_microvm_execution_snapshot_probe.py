"""Probe guest-CRIU prerequisites on a disposable staging Lambda MicroVM.

This is a negative-capability investigation, not an AWS snapshot adapter.
The current staging kernel refuses kcmp; capture cannot qualify restoration.
--build-criu builds pinned CRIU 4.2 inside the isolated guest. No existing
image/allocation is changed. Only report.json may be published.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import importlib
import json
import os
import shlex
import time
import uuid
from pathlib import Path
from typing import Any

from examples.execution_snapshot_probe import ROOT, load, write_private, write_private_bytes

from cayu import ExecCommand, LambdaMicroVMRunner

COMMAND_LIMIT = 24 * 1024 * 1024


def client(state: dict[str, Any]) -> Any:
    boto3 = importlib.import_module("boto3")
    Config = importlib.import_module("botocore.config").Config

    return boto3.Session(profile_name=state["profile"], region_name=state["region"]).client(
        "lambda-microvms", config=Config(retries={"max_attempts": 0}, read_timeout=30)
    )


def run_options(state: dict[str, Any]) -> dict[str, Any]:
    region = state["region"]
    return {
        "imageIdentifier": state["image"],
        "imageVersion": state["image_version"],
        "clientToken": state["client_token"],
        "maximumDurationInSeconds": 600,
        "ingressNetworkConnectors": [
            f"arn:aws:lambda:{region}:aws:network-connector:aws-network-connector:ALL_INGRESS"
        ],
        "egressNetworkConnectors": [
            f"arn:aws:lambda:{region}:aws:network-connector:aws-network-connector:INTERNET_EGRESS"
        ]
        if state["install_criu"]
        else [],
    }


async def command(runner: LambdaMicroVMRunner, source: str, *, stdin: str | None = None) -> str:
    result = await runner.exec_system(
        ExecCommand.bash(source), stdin=stdin, timeout_s=120, output_limit_bytes=COMMAND_LIMIT
    )
    if result.exit_code or result.timed_out or result.stdout_truncated or result.stderr_truncated:
        raise RuntimeError("guest probe command failed")
    return result.stdout


async def write_guest(
    runner: LambdaMicroVMRunner, path: str, content: bytes, *, executable: bool = False
) -> None:
    source = (
        "import base64,sys; from pathlib import Path; "
        f"p=Path({path!r}); p.parent.mkdir(parents=True,exist_ok=True); "
        "p.write_bytes(base64.b64decode(sys.stdin.buffer.read(),validate=True)); "
        f"p.chmod({0o700 if executable else 0o600})"
    )
    encoded = base64.b64encode(content).decode()
    if len(encoded) > COMMAND_LIMIT:
        raise RuntimeError("probe input exceeds its transfer bound")
    await command(runner, "python3 -c " + shlex.quote(source), stdin=encoded)


async def capture_archive(runner: LambdaMicroVMRunner, source: str) -> bytes:
    source_code = (
        "import base64,sys; from pathlib import Path; "
        f"p=Path({source!r}); "
        f"assert p.stat().st_size<={COMMAND_LIMIT // 2}; "
        "print(base64.b64encode(p.read_bytes()).decode())"
    )
    return base64.b64decode(
        (await command(runner, "python3 -c " + shlex.quote(source_code))).strip(), validate=True
    )


async def build_criu(runner: LambdaMicroVMRunner, directory: Path, state: dict[str, Any]) -> None:
    if not state["install_criu"]:
        raise RuntimeError("building CRIU requires --install-criu on the disposable guest")
    await command(
        runner,
        "dnf install -y criu gcc make perl pkgconf protobuf-compiler "
        "protobuf-c-devel protobuf-devel git libuuid-devel protobuf-c-compiler "
        "libnl3-devel libcap-devel libaio-devel libnet-devel gnutls-devel tar gzip python3-PyYAML "
        ">/tmp/cayu-criu-build-install.log 2>&1",
    )
    await command(
        runner,
        "curl -fsSL https://github.com/checkpoint-restore/criu/archive/refs/tags/v4.2.tar.gz "
        "-o /tmp/criu.tar.gz && echo '0c6e51af878e63df7391e6dffbbe5f0ced429bc9f1e5a603020bfd2503065c39 "
        " /tmp/criu.tar.gz' | sha256sum -c - && tar -xzf /tmp/criu.tar.gz -C /tmp "
        "&& make -C /tmp/criu-4.2 -j2 criu CONFIG_NFTABLES=n >/tmp/cayu-criu-build.log 2>&1",
    )
    binary = await capture_archive(runner, "/tmp/criu-4.2/criu/criu")
    path = directory / "criu-binary"
    write_private_bytes(path, binary)
    state["criu_binary"] = str(path)
    state["criu_sha256"] = hashlib.sha256(binary).hexdigest()
    write_private(directory / "state.json", state)


async def prepare_criu(runner: LambdaMicroVMRunner, state: dict[str, Any]) -> str:
    if state["install_criu"]:
        await command(runner, "dnf install -y criu gnutls >/tmp/cayu-criu-install.log 2>&1")
    if state["criu_binary"]:
        binary = Path(state["criu_binary"]).read_bytes()
        if hashlib.sha256(binary).hexdigest() != state["criu_sha256"]:
            raise RuntimeError("probe CRIU binary changed")
        await write_guest(runner, "/tmp/cayu-criu", binary, executable=True)
        return "/tmp/cayu-criu"
    return "criu"


def cleanup(directory: Path) -> None:
    state = load(directory / "state.json")
    if not state["submitted"]:
        return
    control = client(state)
    identifier = state.get("allocation")
    if identifier is None:
        if time.time() > state["submission_deadline"]:
            raise RuntimeError("unknown allocation needs explicit token reconciliation")
        identifier = control.run_microvm(**run_options(state))["microvmId"]
        state["allocation"] = identifier
        write_private(directory / "state.json", state)
    control.terminate_microvm(microvmIdentifier=identifier)
    for _ in range(60):
        if control.get_microvm(microvmIdentifier=identifier)["state"] == "TERMINATED":
            return
        time.sleep(0.5)
    raise RuntimeError("AWS probe termination not confirmed")


async def probe(directory: Path, state: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    write_private(directory / "state.json", state)
    report: dict[str, Any] = {
        "backend": "aws-lambda-microvm",
        "qualified": False,
        "region": state["region"],
        "maximum_allocation_duration_s": 600,
    }
    runner = None
    try:
        control = client(state)
        version = control.get_microvm_image_version(
            imageIdentifier=state["image"], imageVersion=state["image_version"]
        )
        report["image_configuration"] = {
            key: version[key]
            for key in ("additionalOsCapabilities", "cpuConfigurations", "resources")
            if key in version
        }
        state["submitted"] = True
        write_private(directory / "state.json", state)
        state["allocation"] = control.run_microvm(**run_options(state))["microvmId"]
        write_private(directory / "state.json", state)
        runner = await LambdaMicroVMRunner.from_existing(
            state["allocation"],
            profile_name=state["profile"],
            region_name=state["region"],
            close_action="none",
        )
        if state["build_criu"]:
            await build_criu(runner, directory, state)
        criu = await prepare_criu(runner, state)
        source = """import ctypes,errno,json,os,platform
libc=ctypes.CDLL(None,use_errno=True)
result=libc.syscall(272,os.getpid(),os.getpid(),1,0,0) if platform.machine()=="aarch64" else None
number=ctypes.get_errno()
print(json.dumps({
    "kernel":platform.release(),"architecture":platform.machine(),"uid":os.getuid(),
    "kcmp_self_result":result,"kcmp_errno":errno.errorcode.get(number),
    "process_status":[x.strip() for x in open("/proc/self/status") if x.startswith(
        ("CapEff:","NoNewPrivs:","Seccomp:"))]
}))
"""
        report["environment"] = json.loads(
            await command(runner, "python3 -c " + shlex.quote(source))
        )
        report["environment"]["criu"] = (
            await command(runner, shlex.quote(criu) + " --version")
        ).strip()
        report["environment"]["packages"] = (
            await command(runner, "rpm -q criu gnutls glibc python3.11")
        ).strip()
        report["criu_binary_sha256"] = state.get("criu_sha256")
        guest = Path(__file__).parents[1] / "_execution_snapshot_guest.py"
        await write_guest(runner, f"{ROOT}/guest.py", guest.read_bytes())
        await command(
            runner,
            f"setsid python3 {ROOT}/guest.py >/dev/null 2>&1 </dev/null & echo $! >{ROOT}/guest.pid",
        )
        observe_source = """import json,urllib.request
request=urllib.request.Request("http://127.0.0.1:8087/quiesce",method="POST")
print(urllib.request.urlopen(request,timeout=2).read().decode())
"""
        for _ in range(40):
            try:
                expected = json.loads(
                    await command(runner, "python3 -c " + shlex.quote(observe_source))
                )
                break
            except RuntimeError:
                await asyncio.sleep(0.05)
        else:
            raise RuntimeError("fixture did not become ready")
        write_private(
            directory / "capture.json",
            {"state": "intent", "published": False, "active": False, "expected": expected},
        )
        pid = int(await command(runner, f"cat {ROOT}/guest.pid"))
        await command(runner, "mkdir /tmp/checkpoint")
        capture_started = time.monotonic()
        result = await runner.exec_system(
            ExecCommand.bash(
                f"{shlex.quote(criu)} dump --tree {pid} --images-dir /tmp/checkpoint "
                "--leave-running --log-file dump.log -v4"
            ),
            timeout_s=60,
        )
        report["capture_attempt_s"] = time.monotonic() - capture_started
        log = await command(runner, "cat /tmp/checkpoint/dump.log")
        write_private_bytes(directory / "criu-dump.log", log.encode())
        report["capture_exit_code"] = result.exit_code
        if result.exit_code and "kcmp failed:" in log and "Function not implemented" in log:
            report["status"] = "blocked_kcmp_ENOSYS"
        elif result.exit_code:
            report["status"] = "unqualified_capture_failure"
        else:
            report["status"] = "capture_only_restore_unqualified"
        write_private(
            directory / "capture.json",
            {
                "state": "rejected" if result.exit_code else "capture_only",
                "published": False,
                "active": False,
            },
        )
        report["partial_capture_unpublished"] = True
    finally:
        try:
            if runner:
                await runner.close()
        finally:
            cleanup(directory)
        report["scoped_cleanup_confirmed"] = True
    report["total_s"] = time.monotonic() - started
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--profile")
    parser.add_argument("--region")
    parser.add_argument("--image")
    parser.add_argument("--image-version")
    parser.add_argument("--install-criu", action="store_true")
    parser.add_argument("--build-criu", action="store_true")
    parser.add_argument("--criu-binary", type=Path)
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()
    directory = args.state_dir.resolve()
    if args.cleanup:
        cleanup(directory)
        return
    if not all((args.profile, args.region, args.image, args.image_version)):
        parser.error("provide --profile, --region, --image, and --image-version")
    if args.build_criu and (not args.install_criu or args.criu_binary):
        parser.error("--build-criu requires --install-criu and excludes --criu-binary")
    binary = args.criu_binary.resolve() if args.criu_binary else None
    content = binary.read_bytes() if binary else None
    directory.mkdir(mode=0o700, exist_ok=False)
    os.umask(0o077)
    state = {
        "profile": args.profile,
        "region": args.region,
        "image": args.image,
        "image_version": args.image_version,
        "install_criu": args.install_criu,
        "build_criu": args.build_criu,
        "criu_binary": str(binary) if binary else None,
        "criu_sha256": hashlib.sha256(content).hexdigest() if content else None,
        "client_token": "cayu-snapshot-" + uuid.uuid4().hex,
        "submitted": False,
        "submission_deadline": time.time() + 480,
    }
    report = asyncio.run(probe(directory, state))
    report["probe_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_private(directory / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
