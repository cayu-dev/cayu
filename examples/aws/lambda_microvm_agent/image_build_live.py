"""Build and boot the integrated Lambda MicroVM image in a throwaway AWS account."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import boto3  # ty: ignore[unresolved-import]
from examples.aws.lambda_microvm_agent.package_microvm import package

from cayu import ExecCommand, LambdaMicroVMRunner

EVIDENCE_PREFIX = "CAYU_NIGHTLY_EVIDENCE="
_IMAGE_TIMEOUT_SECONDS = 15 * 60


async def main() -> None:
    if os.environ.get("CAYU_AWS_MICROVM_IMAGE_BUILD_LIVE") != "1":
        raise SystemExit("Set CAYU_AWS_MICROVM_IMAGE_BUILD_LIVE=1 to run this contract.")
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if not region:
        raise SystemExit("Set AWS_REGION or AWS_DEFAULT_REGION.")

    session = boto3.Session(region_name=region)
    account = session.client("sts").get_caller_identity()["Account"]
    suffix = uuid.uuid4().hex[:12]
    bucket = f"cayu-microvm-build-{account}-{suffix}".lower()
    object_key = "cayu-microvm.zip"
    role_name = f"cayu-microvm-build-{suffix}"
    policy_name = "read-build-artifact"
    image_name = f"cayu-integrated-{suffix}"
    s3 = session.client("s3")
    iam = session.client("iam")
    microvms = session.client("lambda-microvms")
    image_identifier: str | None = None
    role_created = False
    bucket_created = False
    runner: LambdaMicroVMRunner | None = None
    started = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="cayu-microvm-live-") as directory:
        archive = package(Path(directory) / "cayu-microvm.zip")
        try:
            await asyncio.to_thread(s3.create_bucket, **_create_bucket_options(bucket, region))
            bucket_created = True
            await asyncio.to_thread(
                s3.put_bucket_encryption,
                Bucket=bucket,
                ServerSideEncryptionConfiguration={
                    "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]
                },
            )
            await asyncio.to_thread(s3.upload_file, str(archive), bucket, object_key)

            role = await asyncio.to_thread(
                iam.create_role,
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(
                    {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Principal": {"Service": "lambda.amazonaws.com"},
                                "Action": ["sts:AssumeRole", "sts:TagSession"],
                            }
                        ],
                    }
                ),
            )
            role_created = True
            role_arn = role["Role"]["Arn"]
            await asyncio.to_thread(
                iam.put_role_policy,
                RoleName=role_name,
                PolicyName=policy_name,
                PolicyDocument=json.dumps(
                    {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Action": ["s3:GetObject"],
                                "Resource": f"arn:aws:s3:::{bucket}/{object_key}",
                            }
                        ],
                    }
                ),
            )

            created = await _create_image_after_role_propagates(
                microvms,
                name=image_name,
                role_arn=role_arn,
                artifact_uri=f"s3://{bucket}/{object_key}",
                region=region,
            )
            image_identifier = created["imageArn"]
            image_version = created.get("imageVersion")
            await _wait_for_image(
                microvms,
                image_identifier=image_identifier,
                image_version=image_version,
            )

            runner = await LambdaMicroVMRunner.create(
                image_identifier,
                region_name=region,
                client=microvms,
                ingress_network_connectors=[
                    f"arn:aws:lambda:{region}:aws:network-connector:"
                    "aws-network-connector:ALL_INGRESS"
                ],
                close_action="terminate",
                maximum_duration_in_seconds=900,
            )
            result = await runner.exec(
                ExecCommand.process(
                    "python3",
                    "-c",
                    (
                        "import pathlib; "
                        "roots=('/sbin','/usr/sbin','/usr/bin'); "
                        "exists=lambda name: any((pathlib.Path(root)/name).is_file() "
                        "for root in roots); "
                        "print('efs=' + str(exists('mount.efs'))); "
                        "print('s3files=' + str(exists('mount.s3files')))"
                    ),
                ),
                timeout_s=30,
            )
            if result.exit_code != 0 or result.timed_out:
                raise RuntimeError(f"Built MicroVM command failed: {result.stderr}")
            if result.stdout.splitlines() != ["efs=True", "s3files=True"]:
                raise RuntimeError(f"Built image omitted a workspace mount helper: {result.stdout}")
            microvm_id = runner.microvm_id
            await runner.close()
            await _wait_for_microvm_termination(microvms, microvm_id)
            runner = None

            await _delete_image(microvms, image_identifier)
            image_identifier = None
            await _delete_role(iam, role_name, policy_name)
            role_created = False
            await _delete_bucket(s3, bucket)
            bucket_created = False

            print(
                EVIDENCE_PREFIX
                + json.dumps(
                    {
                        "adapter": "aws-lambda-microvm-image-build",
                        "cleanup": "verified",
                        "efs_mount_helper": "verified",
                        "image_build": "verified",
                        "image_boot": "verified",
                        "region": region,
                        "s3_files_mount_helper": "verified",
                        "seconds": round(time.monotonic() - started, 1),
                    },
                    sort_keys=True,
                )
            )
        finally:
            if runner is not None:
                microvm_id = runner.microvm_id
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await runner.close()
                with contextlib.suppress(Exception):
                    await _wait_for_microvm_termination(microvms, microvm_id)
            if image_identifier is not None:
                with contextlib.suppress(Exception):
                    await _delete_image(microvms, image_identifier)
            if role_created:
                with contextlib.suppress(Exception):
                    await _delete_role(iam, role_name, policy_name)
            if bucket_created:
                with contextlib.suppress(Exception):
                    await _delete_bucket(s3, bucket)


async def _create_image_after_role_propagates(
    client: Any,
    *,
    name: str,
    role_arn: str,
    artifact_uri: str,
    region: str,
) -> dict[str, Any]:
    deadline = time.monotonic() + 120
    while True:
        try:
            return await asyncio.to_thread(
                client.create_microvm_image,
                name=name,
                description="Live build of the Cayu integrated AWS example",
                baseImageArn=f"arn:aws:lambda:{region}:aws:microvm-image:al2023-1",
                buildRoleArn=role_arn,
                codeArtifact={"uri": artifact_uri},
                cpuConfigurations=[{"architecture": "ARM_64"}],
                resources=[{"minimumMemoryInMiB": 1024}],
                additionalOsCapabilities=["ALL"],
                hooks={
                    "port": 8080,
                    "microvmHooks": {
                        "run": "ENABLED",
                        "resume": "ENABLED",
                        "suspend": "ENABLED",
                        "terminate": "ENABLED",
                    },
                    "microvmImageHooks": {"ready": "ENABLED"},
                },
            )
        except client.exceptions.InvalidParameterValueException:
            if time.monotonic() >= deadline:
                raise
            await asyncio.sleep(5)


async def _wait_for_image(
    client: Any,
    *,
    image_identifier: str,
    image_version: str | None,
) -> None:
    deadline = time.monotonic() + _IMAGE_TIMEOUT_SECONDS
    while True:
        image = await asyncio.to_thread(
            client.get_microvm_image,
            imageIdentifier=image_identifier,
        )
        state = image.get("state")
        if state == "CREATED":
            return
        if state == "CREATE_FAILED":
            version = image_version or image.get("latestFailedImageVersion")
            reason = await _image_failure_reason(client, image_identifier, version)
            raise RuntimeError(f"Lambda MicroVM image build failed: {reason}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Lambda MicroVM image build remained in state {state}.")
        await asyncio.sleep(10)


async def _image_failure_reason(
    client: Any,
    image_identifier: str,
    image_version: str | None,
) -> str:
    if not image_version:
        return "AWS did not report a failed image version."
    builds = await asyncio.to_thread(
        client.list_microvm_image_builds,
        imageIdentifier=image_identifier,
        imageVersion=image_version,
    )
    reasons: list[str] = []
    for item in builds.get("items", []):
        build = await asyncio.to_thread(
            client.get_microvm_image_build,
            imageIdentifier=image_identifier,
            imageVersion=image_version,
            buildId=item["buildId"],
        )
        reasons.append(str(build.get("stateReason") or build.get("buildState")))
    return "; ".join(reasons) or "AWS did not report a build reason."


async def _delete_image(client: Any, image_identifier: str) -> None:
    deadline = time.monotonic() + 180
    while True:
        try:
            await asyncio.to_thread(
                client.delete_microvm_image,
                imageIdentifier=image_identifier,
            )
            break
        except client.exceptions.ResourceNotFoundException:
            return
        except client.exceptions.ConflictException:
            if time.monotonic() >= deadline:
                raise
            await asyncio.sleep(5)
        except client.exceptions.ValidationException as exc:
            if "running microvms" not in str(exc).lower() or time.monotonic() >= deadline:
                raise
            await asyncio.sleep(5)
    while True:
        try:
            image = await asyncio.to_thread(
                client.get_microvm_image,
                imageIdentifier=image_identifier,
            )
        except client.exceptions.ResourceNotFoundException:
            return
        if image.get("state") == "DELETED":
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("Lambda MicroVM image deletion did not complete.")
        await asyncio.sleep(5)


async def _wait_for_microvm_termination(client: Any, microvm_id: str) -> None:
    deadline = time.monotonic() + 180
    while True:
        try:
            microvm = await asyncio.to_thread(
                client.get_microvm,
                microvmIdentifier=microvm_id,
            )
        except client.exceptions.ResourceNotFoundException:
            return
        if microvm.get("state") == "TERMINATED":
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("Lambda MicroVM termination did not complete.")
        await asyncio.sleep(5)


async def _delete_role(client: Any, role_name: str, policy_name: str) -> None:
    await asyncio.to_thread(
        client.delete_role_policy,
        RoleName=role_name,
        PolicyName=policy_name,
    )
    await asyncio.to_thread(client.delete_role, RoleName=role_name)


async def _delete_bucket(client: Any, bucket: str) -> None:
    objects = await asyncio.to_thread(client.list_objects_v2, Bucket=bucket)
    for item in objects.get("Contents", []):
        await asyncio.to_thread(client.delete_object, Bucket=bucket, Key=item["Key"])
    await asyncio.to_thread(client.delete_bucket, Bucket=bucket)


def _create_bucket_options(bucket: str, region: str) -> dict[str, Any]:
    options: dict[str, Any] = {"Bucket": bucket}
    if region != "us-east-1":
        options["CreateBucketConfiguration"] = {"LocationConstraint": region}
    return options


if __name__ == "__main__":
    asyncio.run(main())
