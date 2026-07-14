"""ECS/Fargate entrypoint for the trusted Cayu control plane."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Literal, cast

from examples.aws.lambda_microvm_agent.runtime import (
    AwsAgentRuntimeConfig,
    build_app,
    build_runtime,
)

from cayu import PostgresSessionStore, PostgresTaskStore
from cayu.server import BasicAuth, create_server
from cayu.storage.migrations import SchemaMode


def create_application() -> Any:
    config = _config_from_environment()
    conninfo = _required_env("CAYU_DATABASE_URL")
    runtime = build_runtime(config)
    cayu_app = build_app(
        runtime,
        session_store=PostgresSessionStore(conninfo, schema_mode=SchemaMode.MIGRATE),
        task_store=PostgresTaskStore(conninfo, schema_mode=SchemaMode.MIGRATE),
    )
    auth = BasicAuth(
        username=os.environ.get("CAYU_SERVER_USERNAME", "admin"),
        password=_required_env("CAYU_SERVER_PASSWORD"),
    )
    return create_server(cayu_app, auth=auth, expose_docs=False)


def _config_from_environment() -> AwsAgentRuntimeConfig:
    backend = os.environ.get("CAYU_WORKSPACE_BACKEND", "efs")
    if backend not in {"efs", "s3files"}:
        raise RuntimeError("CAYU_WORKSPACE_BACKEND must be efs or s3files.")
    return AwsAgentRuntimeConfig(
        region=_required_env("AWS_REGION"),
        bedrock_model=_required_env("CAYU_BEDROCK_MODEL"),
        microvm_image=_required_env("CAYU_LAMBDA_MICROVM_IMAGE"),
        microvm_egress_connector=_required_env("CAYU_LAMBDA_MICROVM_EGRESS_CONNECTOR"),
        microvm_execution_role_arn=_required_env("CAYU_LAMBDA_MICROVM_EXECUTION_ROLE"),
        task_private_ipv4=_task_private_ipv4(),
        artifact_bucket=_required_env("CAYU_ARTIFACT_BUCKET"),
        service_secret_id=_required_env("CAYU_INTERNAL_SERVICE_SECRET"),
        receiver_origin=os.environ.get(
            "CAYU_INTERNAL_SERVICE_ORIGIN", "http://receiver.service.local:8080"
        ),
        workspace_backend=cast("Literal['efs', 's3files']", backend),
        efs_file_system_id=os.environ.get("CAYU_EFS_FILE_SYSTEM_ID"),
        efs_access_point_id=os.environ.get("CAYU_EFS_ACCESS_POINT_ID"),
        efs_mount_target_ip=os.environ.get("CAYU_EFS_MOUNT_TARGET_IP"),
        s3files_file_system_id=os.environ.get("CAYU_S3FILES_FILE_SYSTEM_ID"),
        s3files_access_point_id=os.environ.get("CAYU_S3FILES_ACCESS_POINT_ID"),
        s3files_mount_target_ip=os.environ.get("CAYU_S3FILES_MOUNT_TARGET_IP"),
        s3files_availability_zone_id=os.environ.get("CAYU_S3FILES_AZ_ID"),
    )


def _task_private_ipv4() -> str:
    explicit = os.environ.get("CAYU_TASK_PRIVATE_IPV4")
    if explicit:
        return explicit
    metadata_uri = _required_env("ECS_CONTAINER_METADATA_URI_V4")
    with urllib.request.urlopen(f"{metadata_uri}/task", timeout=2) as response:
        metadata = json.load(response)
    for container in metadata.get("Containers", []):
        for network in container.get("Networks", []):
            addresses = network.get("IPv4Addresses", [])
            if addresses:
                return str(addresses[0])
    raise RuntimeError("ECS task metadata did not contain a private IPv4 address.")


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise RuntimeError(f"Set {name}.")
    return value.strip()


app = create_application()
