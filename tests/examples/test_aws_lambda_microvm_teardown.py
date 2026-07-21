from __future__ import annotations

import sys
import types
from datetime import UTC, datetime
from typing import Any

import pytest


class _AwsServiceError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


def _install_fake_boto3(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cloudformation: Any,
    s3: Any,
    expected_session_options: dict[str, Any] | None = None,
) -> None:
    clients = {"cloudformation": cloudformation, "s3": s3}

    class _Session:
        def __init__(self, **kwargs: Any) -> None:
            if expected_session_options is not None:
                assert kwargs == expected_session_options

        def client(self, service: str) -> Any:
            return clients[service]

    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.Session = _Session  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)


def test_teardown_requires_explicit_data_purge(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from examples.aws.lambda_microvm_agent import teardown

    with pytest.raises(SystemExit, match="2"):
        teardown.main(["--stack-name", "cayu-aws-agent"])

    assert "--purge-data is required" in capsys.readouterr().err


def test_teardown_quiesces_stack_then_rechecks_late_versions_before_deleting_buckets(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from examples.aws.lambda_microvm_agent import teardown

    stack_id = (
        "arn:aws:cloudformation:us-east-1:123456789012:stack/"
        "cayu-aws-agent/00000000-0000-0000-0000-000000000000"
    )
    events: list[tuple[str, str]] = []
    delete_calls: list[dict[str, Any]] = []
    version_passes = {
        "artifact-bucket": [
            [
                {
                    "Versions": [{"Key": "artifact", "VersionId": "version-1"}],
                    "DeleteMarkers": [{"Key": "artifact", "VersionId": "marker-1"}],
                }
            ],
            [{"Versions": [{"Key": "late-write", "VersionId": "version-2"}]}],
            [{}],
        ],
        "workspace-bucket": [
            [{"Versions": [{"Key": "workspace", "VersionId": "version-3"}]}],
            [{}],
        ],
    }

    class _Paginator:
        def __init__(self, operation: str) -> None:
            self.operation = operation

        def paginate(self, **kwargs: Any) -> list[dict[str, Any]]:
            if self.operation == "list_stack_resources":
                assert kwargs == {"StackName": stack_id}
                return [
                    {
                        "StackResourceSummaries": [
                            {
                                "LogicalResourceId": "ArtifactBucket",
                                "PhysicalResourceId": "artifact-bucket",
                                "ResourceType": "AWS::S3::Bucket",
                            },
                            {
                                "LogicalResourceId": "WorkspaceBucket",
                                "PhysicalResourceId": "workspace-bucket",
                                "ResourceType": "AWS::S3::Bucket",
                            },
                            {
                                "LogicalResourceId": "ControlService",
                                "PhysicalResourceId": "service-arn",
                                "ResourceType": "AWS::ECS::Service",
                            },
                        ]
                    }
                ]
            bucket = kwargs["Bucket"]
            events.append(("list_object_versions", bucket))
            return version_passes[bucket].pop(0)

    class _Waiter:
        def wait(self, **kwargs: Any) -> None:
            assert kwargs == {"StackName": stack_id}
            events.append(("wait", stack_id))

    class _CloudFormation:
        def describe_stacks(self, **kwargs: Any) -> dict[str, Any]:
            assert kwargs == {"StackName": "cayu-aws-agent"}
            return {
                "Stacks": [
                    {
                        "StackName": "cayu-aws-agent",
                        "StackId": stack_id,
                        "StackStatus": "CREATE_COMPLETE",
                    }
                ]
            }

        def get_paginator(self, operation: str) -> _Paginator:
            assert operation == "list_stack_resources"
            return _Paginator(operation)

        def delete_stack(self, **kwargs: Any) -> None:
            assert kwargs == {"StackName": stack_id}
            events.append(("delete_stack", stack_id))

        def get_waiter(self, waiter: str) -> _Waiter:
            assert waiter == "stack_delete_complete"
            return _Waiter()

    class _S3:
        def get_paginator(self, operation: str) -> _Paginator:
            assert operation == "list_object_versions"
            return _Paginator(operation)

        def delete_objects(self, **kwargs: Any) -> dict[str, Any]:
            delete_calls.append(kwargs)
            events.append(("delete_objects", kwargs["Bucket"]))
            return {}

        def delete_bucket(self, **kwargs: Any) -> None:
            events.append(("delete_bucket", kwargs["Bucket"]))

    _install_fake_boto3(
        monkeypatch,
        cloudformation=_CloudFormation(),
        s3=_S3(),
        expected_session_options={"region_name": "us-east-1"},
    )

    teardown.main(
        [
            "--stack-name",
            "cayu-aws-agent",
            "--region",
            "us-east-1",
            "--purge-data",
        ]
    )

    deleted = {
        (call["Bucket"], item["Key"], item["VersionId"])
        for call in delete_calls
        for item in call["Delete"]["Objects"]
    }
    assert deleted == {
        ("artifact-bucket", "artifact", "version-1"),
        ("artifact-bucket", "artifact", "marker-1"),
        ("artifact-bucket", "late-write", "version-2"),
        ("workspace-bucket", "workspace", "version-3"),
    }
    assert all(call["Delete"]["Quiet"] is True for call in delete_calls)
    wait_index = events.index(("wait", stack_id))
    first_bucket_list_index = next(
        index for index, event in enumerate(events) if event[0] == "list_object_versions"
    )
    assert wait_index < first_bucket_list_index
    assert events[:2] == [
        ("delete_stack", stack_id),
        ("wait", stack_id),
    ]
    assert events[-1] == ("delete_bucket", "workspace-bucket")
    assert capsys.readouterr().out.splitlines() == [
        "Deleted stack and quiesced writers: cayu-aws-agent",
        "Purged and deleted retained bucket: artifact-bucket",
        "Purged and deleted retained bucket: workspace-bucket",
    ]


def test_teardown_preserves_s3_version_deletion_errors_after_quiescing_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from examples.aws.lambda_microvm_agent import teardown

    stack_id = "arn:aws:cloudformation:us-east-1:123456789012:stack/cayu-aws-agent/id"
    events: list[str] = []

    class _Paginator:
        def __init__(self, operation: str) -> None:
            self.operation = operation

        def paginate(self, **kwargs: Any) -> list[dict[str, Any]]:
            if self.operation == "list_stack_resources":
                return [
                    {
                        "StackResourceSummaries": [
                            {
                                "LogicalResourceId": "ArtifactBucket",
                                "PhysicalResourceId": "artifact-bucket",
                                "ResourceType": "AWS::S3::Bucket",
                            }
                        ]
                    }
                ]
            return [{"Versions": [{"Key": "artifact", "VersionId": "version-1"}]}]

    class _CloudFormation:
        def describe_stacks(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "Stacks": [
                    {
                        "StackName": "cayu-aws-agent",
                        "StackId": stack_id,
                        "StackStatus": "CREATE_COMPLETE",
                    }
                ]
            }

        def get_paginator(self, operation: str) -> _Paginator:
            return _Paginator(operation)

        def delete_stack(self, **kwargs: Any) -> None:
            events.append("delete_stack")

        def get_waiter(self, waiter: str) -> _Waiter:
            return _Waiter()

    class _Waiter:
        def wait(self, **kwargs: Any) -> None:
            events.append("wait")

    class _S3:
        def get_paginator(self, operation: str) -> _Paginator:
            return _Paginator(operation)

        def delete_objects(self, **kwargs: Any) -> dict[str, Any]:
            return {"Errors": [{"Key": "artifact", "Code": "AccessDenied"}]}

        def delete_bucket(self, **kwargs: Any) -> None:
            pytest.fail("delete_bucket called after S3 rejected a version deletion")

    _install_fake_boto3(monkeypatch, cloudformation=_CloudFormation(), s3=_S3())

    with pytest.raises(RuntimeError, match="AccessDenied"):
        teardown.main(["--stack-name", "cayu-aws-agent", "--purge-data"])

    assert events == ["delete_stack", "wait"]


def test_teardown_treats_confirmed_missing_bucket_as_already_purged_on_retry(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from examples.aws.lambda_microvm_agent import teardown

    stack_id = "arn:aws:cloudformation:us-east-1:123456789012:stack/cayu-aws-agent/id"
    listed_buckets: list[str] = []

    class _Paginator:
        def __init__(self, operation: str) -> None:
            self.operation = operation

        def paginate(self, **kwargs: Any) -> list[dict[str, Any]]:
            if self.operation == "list_stack_resources":
                return [
                    {
                        "StackResourceSummaries": [
                            {
                                "LogicalResourceId": "ArtifactBucket",
                                "PhysicalResourceId": "missing-artifact-bucket",
                                "ResourceType": "AWS::S3::Bucket",
                                "ResourceStatus": "DELETE_FAILED",
                            },
                            {
                                "LogicalResourceId": "WorkspaceBucket",
                                "PhysicalResourceId": "deleted-workspace-bucket",
                                "ResourceType": "AWS::S3::Bucket",
                                "ResourceStatus": "DELETE_COMPLETE",
                            },
                        ]
                    }
                ]
            bucket = kwargs["Bucket"]
            listed_buckets.append(bucket)
            raise _AwsServiceError("NoSuchBucket")

    class _Waiter:
        def wait(self, **kwargs: Any) -> None:
            pass

    class _CloudFormation:
        def describe_stacks(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "Stacks": [
                    {
                        "StackName": "cayu-aws-agent",
                        "StackId": stack_id,
                        "StackStatus": "DELETE_FAILED",
                    }
                ]
            }

        def get_paginator(self, operation: str) -> _Paginator:
            return _Paginator(operation)

        def delete_stack(self, **kwargs: Any) -> None:
            pass

        def get_waiter(self, waiter: str) -> _Waiter:
            return _Waiter()

    class _S3:
        def get_paginator(self, operation: str) -> _Paginator:
            return _Paginator(operation)

        def delete_bucket(self, **kwargs: Any) -> None:
            pytest.fail("delete_bucket called for a confirmed missing bucket")

    _install_fake_boto3(monkeypatch, cloudformation=_CloudFormation(), s3=_S3())

    teardown.main(["--stack-name", "cayu-aws-agent", "--purge-data"])

    assert listed_buckets == ["missing-artifact-bucket"]
    assert capsys.readouterr().out.splitlines() == [
        "Deleted stack and quiesced writers: cayu-aws-agent",
        "Retained bucket already absent: missing-artifact-bucket",
    ]


def test_teardown_resumes_by_name_after_stack_deletion_and_partial_bucket_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from examples.aws.lambda_microvm_agent import teardown

    stack_id = (
        "arn:aws:cloudformation:us-east-1:123456789012:stack/"
        "cayu-aws-agent/00000000-0000-0000-0000-000000000000"
    )
    stack_deleted = False
    deleted_buckets: set[str] = set()
    workspace_listing_failed = False
    delete_stack_calls = 0

    class _Paginator:
        def __init__(self, operation: str) -> None:
            self.operation = operation

        def paginate(self, **kwargs: Any) -> list[dict[str, Any]]:
            nonlocal workspace_listing_failed
            if self.operation == "list_stack_resources":
                assert kwargs == {"StackName": stack_id}
                return [
                    {
                        "StackResourceSummaries": [
                            {
                                "LogicalResourceId": "ArtifactBucket",
                                "PhysicalResourceId": "artifact-bucket",
                                "ResourceType": "AWS::S3::Bucket",
                                "ResourceStatus": "DELETE_SKIPPED",
                            },
                            {
                                "LogicalResourceId": "WorkspaceBucket",
                                "PhysicalResourceId": "workspace-bucket",
                                "ResourceType": "AWS::S3::Bucket",
                                "ResourceStatus": "DELETE_SKIPPED",
                            },
                        ]
                    }
                ]
            if self.operation == "list_stacks":
                assert kwargs == {"StackStatusFilter": ["DELETE_COMPLETE"]}
                return [
                    {
                        "StackSummaries": [
                            {
                                "StackName": "cayu-aws-agent",
                                "StackId": "arn:aws:cloudformation:us-east-1:123456789012:stack/old/id",
                                "StackStatus": "DELETE_COMPLETE",
                                "DeletionTime": datetime(2026, 1, 1, tzinfo=UTC),
                            },
                            {
                                "StackName": "cayu-aws-agent",
                                "StackId": stack_id,
                                "StackStatus": "DELETE_COMPLETE",
                                "DeletionTime": datetime(2026, 7, 20, tzinfo=UTC),
                            },
                        ]
                    }
                ]
            bucket = kwargs["Bucket"]
            if bucket in deleted_buckets:
                raise _AwsServiceError("NoSuchBucket")
            if bucket == "workspace-bucket" and not workspace_listing_failed:
                workspace_listing_failed = True
                raise _AwsServiceError("AccessDenied")
            return [{}]

    class _Waiter:
        def wait(self, **kwargs: Any) -> None:
            assert kwargs == {"StackName": stack_id}

    class _CloudFormation:
        def describe_stacks(self, **kwargs: Any) -> dict[str, Any]:
            assert kwargs == {"StackName": "cayu-aws-agent"}
            if stack_deleted:
                raise _AwsServiceError("ValidationError")
            return {
                "Stacks": [
                    {
                        "StackName": "cayu-aws-agent",
                        "StackId": stack_id,
                        "StackStatus": "CREATE_COMPLETE",
                    }
                ]
            }

        def get_paginator(self, operation: str) -> _Paginator:
            return _Paginator(operation)

        def delete_stack(self, **kwargs: Any) -> None:
            nonlocal delete_stack_calls, stack_deleted
            assert kwargs == {"StackName": stack_id}
            delete_stack_calls += 1
            stack_deleted = True

        def get_waiter(self, waiter: str) -> _Waiter:
            assert waiter == "stack_delete_complete"
            return _Waiter()

    class _S3:
        def get_paginator(self, operation: str) -> _Paginator:
            assert operation == "list_object_versions"
            return _Paginator(operation)

        def delete_bucket(self, **kwargs: Any) -> None:
            deleted_buckets.add(kwargs["Bucket"])

    _install_fake_boto3(monkeypatch, cloudformation=_CloudFormation(), s3=_S3())
    argv = ["--stack-name", "cayu-aws-agent", "--purge-data"]

    with pytest.raises(_AwsServiceError, match="AccessDenied"):
        teardown.main(argv)
    teardown.main(argv)

    assert delete_stack_calls == 1
    assert deleted_buckets == {"artifact-bucket", "workspace-bucket"}
    assert capsys.readouterr().out.splitlines() == [
        "Deleted stack and quiesced writers: cayu-aws-agent",
        "Purged and deleted retained bucket: artifact-bucket",
        "Stack already deleted and writers quiesced: cayu-aws-agent",
        "Retained bucket already absent: artifact-bucket",
        "Purged and deleted retained bucket: workspace-bucket",
    ]
