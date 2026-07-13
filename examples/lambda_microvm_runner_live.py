"""Verified AWS Lambda MicroVM runner, workspace, cancellation, and lifecycle contract."""

from __future__ import annotations

import asyncio
import json
import os

from _live_checks import require, require_exec_success
from _runner_conformance import verify_bounded_output_drain
from cayu import ExecCommand, LambdaMicroVMRunner, RunnerWorkspace

EVIDENCE_PREFIX = "CAYU_NIGHTLY_EVIDENCE="


async def main() -> None:
    if os.environ.get("CAYU_LAMBDA_MICROVM_LIVE") != "1":
        raise SystemExit("Set CAYU_LAMBDA_MICROVM_LIVE=1 to run this live contract.")
    image = os.environ.get("CAYU_LAMBDA_MICROVM_IMAGE")
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if not image:
        raise SystemExit("Set CAYU_LAMBDA_MICROVM_IMAGE to a built MicroVM image ARN.")
    if not region:
        raise SystemExit("Set AWS_REGION or AWS_DEFAULT_REGION.")
    ingress = os.environ.get(
        "CAYU_LAMBDA_MICROVM_INGRESS_CONNECTOR",
        f"arn:aws:lambda:{region}:aws:network-connector:aws-network-connector:ALL_INGRESS",
    )
    egress = [
        value.strip()
        for value in os.environ.get("CAYU_LAMBDA_MICROVM_EGRESS_CONNECTORS", "").split(",")
        if value.strip()
    ]

    runner: LambdaMicroVMRunner | None = None
    try:
        runner = await LambdaMicroVMRunner.create(
            image,
            region_name=region,
            ingress_network_connectors=[ingress],
            egress_network_connectors=egress or None,
            idle_policy={
                "autoResumeEnabled": True,
                "maxIdleDurationSeconds": 900,
                "suspendedDurationSeconds": 300,
            },
            maximum_duration_in_seconds=1800,
            close_action="terminate",
        )
        workspace = RunnerWorkspace(runner)

        process = await runner.exec(ExecCommand.process("python3", "-c", "print('process-ok')"))
        require_exec_success(process, stdout="process-ok\n", label="process form")
        shell = await runner.exec(ExecCommand.bash("printf shell-ok"))
        require_exec_success(shell, stdout="shell-ok", label="shell form")

        os.environ["CAYU_HOST_SECRET_SHOULD_NOT_LEAK"] = "must-not-appear"
        env_result = await runner.exec(
            ExecCommand.bash(
                'printf "%s/%s" "${VISIBLE:-missing}" "${CAYU_HOST_SECRET_SHOULD_NOT_LEAK:-absent}"'
            ),
            env={"VISIBLE": "yes"},
        )
        require_exec_success(env_result, stdout="yes/absent", label="environment isolation")

        await verify_bounded_output_drain(runner, adapter="lambda-microvm")

        timed_out = await runner.exec(ExecCommand.bash("sleep 30"), timeout_s=1)
        require(timed_out.timed_out, "sidecar did not report command timeout")
        require(
            any(
                artifact.get("action") == "kill_command" and artifact.get("status") == "completed"
                for artifact in timed_out.artifacts
            ),
            "timeout did not confirm command cleanup",
        )

        cancelled = asyncio.create_task(runner.exec(ExecCommand.bash("sleep 30")))
        await asyncio.sleep(0.5)
        cancelled.cancel()
        try:
            await cancelled
            raise RuntimeError("cancelled command unexpectedly completed")
        except asyncio.CancelledError as exc:
            artifacts = getattr(exc, "artifacts", [])
            require(
                any(
                    artifact.get("action") == "kill_command"
                    and artifact.get("status") == "completed"
                    for artifact in artifacts
                ),
                "cancellation did not confirm command cleanup",
            )

        reusable = await runner.exec(ExecCommand.bash("printf cleanup-reusable"))
        require_exec_success(
            reusable,
            stdout="cleanup-reusable",
            label="post-cleanup reuse",
        )

        await workspace.write_bytes("live/state.txt", b"preserved")
        read = await workspace.read_bytes("live/state.txt")
        require(read.content == b"preserved", "workspace read/write failed")
        listed = await workspace.list("**/*.txt")
        require("live/state.txt" in listed.paths, "workspace list omitted written file")

        await runner.suspend()
        await runner.resume()
        read_after_resume = await workspace.read_bytes("live/state.txt")
        require(read_after_resume.content == b"preserved", "resume did not preserve disk state")
        await workspace.delete("live/state.txt")

        evidence = {
            "adapter": "lambda-microvm",
            "microvm_id": runner.microvm_id,
            "region": region,
            "process": "verified",
            "shell": "verified",
            "environment_isolation": "verified",
            "bounded_output": "verified",
            "timeout_cleanup": "verified",
            "cancellation_cleanup": "verified",
            "post_cleanup_reuse": "verified",
            "workspace": "verified",
            "suspend_resume": "verified",
        }
        print(f"{EVIDENCE_PREFIX}{json.dumps(evidence, sort_keys=True)}")
    finally:
        if runner is not None:
            await runner.close()


if __name__ == "__main__":
    asyncio.run(main())
