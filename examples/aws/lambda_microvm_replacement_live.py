"""Live contract: restore a workspace checkpoint into a replacement Lambda MicroVM.

A MicroVM owned through its sidecar owner claim writes a workspace, Cayu
captures a pinned checkpoint under that exclusive writer isolation, and the
MicroVM is terminated. The replacement is pinned to its predecessor's exact
image version, disposal of the predecessor is proven from the control plane,
and the checkpoint is restored and verified before any other command runs.

This exercises the runtime's replacement mechanics against real AWS. The full
factory path additionally needs the integrated private egress proxy and is
covered by deterministic tests.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import tempfile
import time
import uuid
from pathlib import Path

from cayu import LambdaMicroVMRunner
from cayu.artifacts import LocalArtifactStore
from cayu.egress import VirtualEgressAllocationPreparation, VirtualEgressAllocationReap
from cayu.egress.aws_lambda_microvm_adapter import LambdaMicroVMEgressAdapter, _client_token
from cayu.egress.proxy_exposure import VpcTaskProxyExposure
from cayu.runners.aws_lambda_microvm import terminate_microvm_confirmed
from cayu.workspaces import RunnerWorkspace
from cayu.workspaces.checkpoints import (
    WorkspaceCheckpointPolicy,
    capture_workspace_checkpoint,
    restore_workspace_checkpoint,
    workspace_checkpoint_revision,
)

EVIDENCE_PREFIX = "CAYU_NIGHTLY_EVIDENCE="
_FILES = {
    "README.md": b"replacement live check\n",
    "src/app.py": b"print('restored')\n",
    "scripts/run.sh": b"#!/bin/sh\necho ok\n",
}


def _configuration() -> dict[str, str]:
    image = os.environ.get("CAYU_LAMBDA_MICROVM_IMAGE", "")
    connector = os.environ.get("CAYU_LAMBDA_MICROVM_EGRESS_CONNECTOR", "")
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or ""
    if not image.startswith("arn:") or not connector.startswith("arn:") or not region:
        raise SystemExit(
            "Set CAYU_LAMBDA_MICROVM_IMAGE, CAYU_LAMBDA_MICROVM_EGRESS_CONNECTOR, and AWS_REGION."
        )
    return {"image": image, "connector": connector, "region": region}


async def _allocate(
    adapter: LambdaMicroVMEgressAdapter,
    config: dict[str, str],
    *,
    predecessor: dict[str, str] | None,
) -> tuple[LambdaMicroVMRunner, dict[str, object]]:
    allocation_id = f"ealloc_{uuid.uuid4().hex}"
    metadata = await adapter.prepare_allocation_metadata(
        VirtualEgressAllocationPreparation(
            allocation_id=allocation_id,
            session_id="live-replacement",
            environment_name="sandbox",
            image=config["image"],
            predecessor_identity=predecessor,
        )
    )
    generation = adapter.allocation_adapter_generation
    assert generation is not None
    try:
        runner = await LambdaMicroVMRunner.create(
            metadata["image_arn"],
            region_name=config["region"],
            image_version=metadata["image_version"],
            ingress_network_connectors=adapter.ingress_network_connectors,
            egress_network_connectors=[config["connector"]],
            maximum_duration_in_seconds=metadata["maximum_duration_s"],
            client_token=_client_token(generation, allocation_id),
            close_action="terminate",
        )
    except BaseException as error:
        # A token-created MicroVM is not disposed of by a constructor that never
        # claimed it; the allocation reap is the authoritative cleanup path.
        try:
            await adapter.reap_allocation(
                VirtualEgressAllocationReap(
                    allocation_id=allocation_id,
                    session_id="live-replacement",
                    environment_name="sandbox",
                    image=config["image"],
                    allocation_metadata=metadata,
                )
            )
        except Exception as reap_error:
            error.add_note(f"Allocation reap after failed create also failed: {reap_error!r}")
        raise
    return runner, metadata


def _identity(runner: LambdaMicroVMRunner, region: str) -> dict[str, str]:
    assert runner.image_identifier is not None and runner.image_version is not None
    return {
        "microvm_id": runner.microvm_id,
        "endpoint": runner.endpoint,
        "region": region,
        "image_identifier": runner.image_identifier,
        "image_version": runner.image_version,
        "session_id": "live-replacement",
        "environment_name": "sandbox",
    }


async def main() -> None:
    if os.environ.get("CAYU_LAMBDA_MICROVM_REPLACEMENT_LIVE") != "1":
        raise SystemExit("Set CAYU_LAMBDA_MICROVM_REPLACEMENT_LIVE=1 to run this contract.")
    config = _configuration()
    started = time.monotonic()
    adapter = LambdaMicroVMEgressAdapter(
        region_name=config["region"],
        egress_network_connector_arn=config["connector"],
        exposure=VpcTaskProxyExposure("10.0.0.10"),
        runner_options={"maximum_duration_in_seconds": 900},
    )
    policy = WorkspaceCheckpointPolicy(allocation_replacement="restore")
    created: list[LambdaMicroVMRunner] = []
    with tempfile.TemporaryDirectory(prefix="cayu-replacement-live-") as directory:
        store = LocalArtifactStore(Path(directory) / "artifacts")
        try:
            original, original_metadata = await _allocate(adapter, config, predecessor=None)
            created.append(original)
            workspace = RunnerWorkspace(original, workspace_id="sandbox")
            for path, content in _FILES.items():
                await workspace.create_bytes(path, content)
            original_isolation = adapter.observe_writer_isolation(original)
            if original_isolation.mechanism != "lambda-microvm-owner-fence":
                raise RuntimeError("Original MicroVM did not report owner-fence isolation.")
            _manifest_id, manifest = await capture_workspace_checkpoint(
                workspace,
                store,
                policy=policy,
                environment_name="sandbox",
                owner="live-replacement",
                isolation=lambda: adapter.observe_writer_isolation(original),
            )
            predecessor = _identity(original, config["region"])

            # AWS retires the MicroVM (here: explicitly) and its disk with it.
            await original.close()
            if not await adapter.is_allocation_disposed(predecessor):
                raise RuntimeError("Terminated predecessor was not proven disposed.")

            replacement, replacement_metadata = await _allocate(
                adapter, config, predecessor=predecessor
            )
            created.append(replacement)
            if replacement_metadata["image_version"] != original_metadata["image_version"]:
                raise RuntimeError("Replacement was not pinned to the predecessor image version.")
            restored = RunnerWorkspace(replacement, workspace_id="sandbox")
            if await workspace_checkpoint_revision(restored, policy=policy) == manifest.revision:
                raise RuntimeError("Replacement unexpectedly started with the predecessor disk.")
            await restore_workspace_checkpoint(
                restored,
                store,
                manifest,
                policy=policy,
                isolation=lambda: adapter.observe_writer_isolation(replacement),
            )
            if await workspace_checkpoint_revision(restored, policy=policy) != manifest.revision:
                raise RuntimeError("Restored workspace revision does not match the checkpoint.")
            for path, content in _FILES.items():
                if (await restored.read_bytes(path)).content != content:
                    raise RuntimeError(f"Restored file {path} differs.")
            replacement_isolation = adapter.observe_writer_isolation(replacement)
            if replacement_isolation.generation == original_isolation.generation:
                raise RuntimeError("Replacement reused the predecessor's owner generation.")
            await replacement.close()
        finally:
            for runner in created:
                if not runner.is_closed:
                    with contextlib.suppress(Exception):
                        await runner.close()
    # Census: every MicroVM this contract created must be terminated.
    import boto3  # ty: ignore[unresolved-import]

    client = boto3.client("lambda-microvms", region_name=config["region"])
    for runner in created:
        await terminate_microvm_confirmed(client, runner.microvm_id, timeout_s=60)
    print(
        EVIDENCE_PREFIX
        + json.dumps(
            {
                "adapter": "aws-lambda-microvm-replacement",
                "region": config["region"],
                "checkpoint_files": len(manifest.files),
                "owner_fence_isolation": "verified",
                "predecessor_disposal_proof": "verified",
                "replacement_image_version_pinned": "verified",
                "replacement_started_without_predecessor_disk": "verified",
                "checkpoint_restored_revision": "verified",
                "microvms_created": len(created),
                "unaccounted_microvms": 0,
                "seconds": round(time.monotonic() - started, 1),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
