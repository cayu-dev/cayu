"""Live contract for S3ArtifactStore and SecretsManagerVault."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import uuid
from typing import Any

import boto3  # ty: ignore[unresolved-import]

from cayu import ArtifactScope, S3ArtifactStore, SecretsManagerVault

EVIDENCE_PREFIX = "CAYU_NIGHTLY_EVIDENCE="


async def main() -> None:
    if os.environ.get("CAYU_AWS_DURABLE_SERVICES_LIVE") != "1":
        raise SystemExit("Set CAYU_AWS_DURABLE_SERVICES_LIVE=1 to run this contract.")
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if not region:
        raise SystemExit("Set AWS_REGION or AWS_DEFAULT_REGION.")

    session = boto3.Session(region_name=region)
    account = session.client("sts").get_caller_identity()["Account"]
    suffix = uuid.uuid4().hex[:12]
    bucket: str | None = f"cayu-durable-{account}-{suffix}".lower()
    secret_name = f"cayu-durable-{suffix}"
    secret_value = f"live-contract-{uuid.uuid4().hex}"
    s3 = session.client("s3")
    secrets = session.client("secretsmanager")
    secret_arn: str | None = None

    try:
        create_options: dict[str, Any] = {"Bucket": bucket}
        if region != "us-east-1":
            create_options["CreateBucketConfiguration"] = {"LocationConstraint": region}
        await asyncio.to_thread(s3.create_bucket, **create_options)
        await asyncio.to_thread(
            s3.put_bucket_encryption,
            Bucket=bucket,
            ServerSideEncryptionConfiguration={
                "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]
            },
        )
        created = await asyncio.to_thread(
            secrets.create_secret,
            Name=secret_name,
            SecretString=secret_value,
        )
        secret_arn = created["ARN"]

        vault = SecretsManagerVault(
            {"service_token": secret_arn},
            client=secrets,
        )
        ref = await vault.get("service_token", scope={"session_id": "live"})
        resolved = await vault.resolve(ref, scope={"session_id": "live"})
        if resolved.value.get_secret_value() != secret_value:
            raise RuntimeError("Secrets Manager value did not round-trip.")

        store = S3ArtifactStore(bucket, client=s3)
        metadata = await store.put_bytes(
            b"durable-artifact",
            filename="evidence.txt",
            content_type="text/plain",
            scope=ArtifactScope.SESSION,
            session_id="live",
        )
        read = await store.read_bytes(metadata.id, max_bytes=7)
        if read.content != b"durable" or not read.truncated:
            raise RuntimeError("S3 ranged artifact read contract failed.")
        listed = await store.list(scope=ArtifactScope.SESSION, session_id="live")
        if [item.id for item in listed.artifacts] != [metadata.id]:
            raise RuntimeError("S3 committed artifact listing contract failed.")
        await store.delete(metadata.id)

        await _cleanup(secrets, s3, secret_arn=secret_arn, bucket=bucket)
        secret_arn = None
        bucket = None

        print(
            EVIDENCE_PREFIX
            + json.dumps(
                {
                    "adapter": "aws-durable-services",
                    "region": region,
                    "s3_artifacts": "verified",
                    "secrets_manager_vault": "verified",
                    "ranged_read": "verified",
                    "cleanup": "verified",
                },
                sort_keys=True,
            )
        )
    finally:
        with contextlib.suppress(Exception):
            await _cleanup(secrets, s3, secret_arn=secret_arn, bucket=bucket)


async def _cleanup(
    secrets: Any,
    s3: Any,
    *,
    secret_arn: str | None,
    bucket: str | None,
) -> None:
    if secret_arn is not None:
        await asyncio.to_thread(
            secrets.delete_secret,
            SecretId=secret_arn,
            ForceDeleteWithoutRecovery=True,
        )
    if bucket is None:
        return
    objects = await asyncio.to_thread(
        s3.list_objects_v2,
        Bucket=bucket,
    )
    for item in objects.get("Contents", []):
        await asyncio.to_thread(
            s3.delete_object,
            Bucket=bucket,
            Key=item["Key"],
        )
    await asyncio.to_thread(
        s3.delete_bucket,
        Bucket=bucket,
    )


if __name__ == "__main__":
    asyncio.run(main())
