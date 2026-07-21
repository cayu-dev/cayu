"""Safely tear down the integrated AWS example and its versioned data buckets."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

_BUCKET_LOGICAL_IDS = ("ArtifactBucket", "WorkspaceBucket")
_MAX_PURGE_PASSES = 10
_S3_DELETE_LIMIT = 1_000


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Delete the integrated AWS example stack and its bucket data."
    )
    parser.add_argument("--stack-name", required=True)
    parser.add_argument("--region", help="AWS region; defaults to the configured SDK region.")
    parser.add_argument(
        "--purge-data",
        action="store_true",
        help="Irreversibly delete every object version and delete marker in stack buckets.",
    )
    args = parser.parse_args(argv)
    if not args.purge_data:
        parser.error("--purge-data is required because teardown irreversibly deletes bucket data")

    import boto3  # ty: ignore[unresolved-import]

    session_options = {"region_name": args.region} if args.region else {}
    session = boto3.Session(**session_options)
    cloudformation = session.client("cloudformation")
    s3 = session.client("s3")
    stack_id, stack_deleted = _resolve_stack(cloudformation, stack_name=args.stack_name)
    buckets = _stack_bucket_names(cloudformation, stack_name=stack_id)
    if stack_deleted:
        print(f"Stack already deleted and writers quiesced: {args.stack_name}")
    else:
        cloudformation.delete_stack(StackName=stack_id)
        cloudformation.get_waiter("stack_delete_complete").wait(StackName=stack_id)
        print(f"Deleted stack and quiesced writers: {args.stack_name}")
    for bucket in buckets:
        if _purge_and_delete_bucket(s3, bucket=bucket):
            print(f"Purged and deleted retained bucket: {bucket}")
        else:
            print(f"Retained bucket already absent: {bucket}")


def _resolve_stack(cloudformation: Any, *, stack_name: str) -> tuple[str, bool]:
    try:
        response = cloudformation.describe_stacks(StackName=stack_name)
    except Exception as exc:
        if _aws_error_code(exc) != "ValidationError":
            raise
        deleted_stack_id = _latest_deleted_stack_id(cloudformation, stack_name=stack_name)
        if deleted_stack_id is None:
            raise
        return deleted_stack_id, True

    stacks = response.get("Stacks", [])
    if len(stacks) != 1 or not isinstance(stacks[0].get("StackId"), str):
        raise RuntimeError(f"CloudFormation returned no unique stack for {stack_name}.")
    stack = stacks[0]
    return stack["StackId"], stack.get("StackStatus") == "DELETE_COMPLETE"


def _latest_deleted_stack_id(cloudformation: Any, *, stack_name: str) -> str | None:
    latest: tuple[Any, str] | None = None
    paginator = cloudformation.get_paginator("list_stacks")
    for page in paginator.paginate(StackStatusFilter=["DELETE_COMPLETE"]):
        for stack in page.get("StackSummaries", []):
            stack_id = stack.get("StackId")
            if not isinstance(stack_id, str) or (
                stack.get("StackName") != stack_name and stack_id != stack_name
            ):
                continue
            deletion_time = stack.get("DeletionTime")
            if latest is None or (
                deletion_time is not None and (latest[0] is None or deletion_time > latest[0])
            ):
                latest = (deletion_time, stack_id)
    return latest[1] if latest is not None else None


def _stack_bucket_names(cloudformation: Any, *, stack_name: str) -> list[str]:
    found: dict[str, str] = {}
    paginator = cloudformation.get_paginator("list_stack_resources")
    for page in paginator.paginate(StackName=stack_name):
        for resource in page.get("StackResourceSummaries", []):
            logical_id = resource.get("LogicalResourceId")
            physical_id = resource.get("PhysicalResourceId")
            if (
                logical_id in _BUCKET_LOGICAL_IDS
                and resource.get("ResourceType") == "AWS::S3::Bucket"
                and resource.get("ResourceStatus") != "DELETE_COMPLETE"
                and isinstance(physical_id, str)
            ):
                found[logical_id] = physical_id
    return [found[logical_id] for logical_id in _BUCKET_LOGICAL_IDS if logical_id in found]


def _purge_and_delete_bucket(s3: Any, *, bucket: str) -> bool:
    for _pass in range(_MAX_PURGE_PASSES):
        try:
            found_versions = _delete_listed_versions(s3, bucket=bucket)
        except Exception as exc:
            if _aws_error_code(exc) == "NoSuchBucket":
                return False
            raise
        if found_versions:
            continue
        try:
            s3.delete_bucket(Bucket=bucket)
        except Exception as exc:
            if _aws_error_code(exc) == "NoSuchBucket":
                return False
            raise
        return True
    raise RuntimeError(
        f"S3 bucket {bucket} did not become empty after {_MAX_PURGE_PASSES} purge passes."
    )


def _delete_listed_versions(s3: Any, *, bucket: str) -> bool:
    found_objects = False
    paginator = s3.get_paginator("list_object_versions")
    for page in paginator.paginate(Bucket=bucket):
        objects = [
            {"Key": item["Key"], "VersionId": item["VersionId"]}
            for collection in (page.get("Versions", []), page.get("DeleteMarkers", []))
            for item in collection
        ]
        found_objects = found_objects or bool(objects)
        for start in range(0, len(objects), _S3_DELETE_LIMIT):
            response = s3.delete_objects(
                Bucket=bucket,
                Delete={"Objects": objects[start : start + _S3_DELETE_LIMIT], "Quiet": True},
            )
            errors = response.get("Errors", [])
            if errors:
                raise RuntimeError(f"S3 rejected version deletion for {bucket}: {errors}")
    return found_objects


def _aws_error_code(exc: Exception) -> str | None:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return None
    error = response.get("Error")
    if not isinstance(error, dict):
        return None
    code = error.get("Code")
    return code if isinstance(code, str) else None


if __name__ == "__main__":
    main()
