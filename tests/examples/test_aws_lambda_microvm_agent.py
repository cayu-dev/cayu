from __future__ import annotations

import asyncio
import re
import zipfile
from pathlib import Path
from typing import Any

from examples.aws.lambda_microvm_agent.package_microvm import package
from examples.aws.lambda_microvm_agent.runtime import (
    AwsAgentRuntimeConfig,
    InternalActionTool,
    build_runtime,
)

from cayu import ExecResult, ToolContext
from cayu.artifacts import S3ArtifactStore
from cayu.environments import EFSAccessPointBinding, S3FilesAccessPointBinding
from cayu.vaults import SecretsManagerVault


def _cloudformation_resource_body(template: str, resource: str) -> str:
    match = re.search(
        rf"(?ms)^  {resource}:\n(?P<body>.*?)(?=^  [A-Za-z][A-Za-z0-9]*:\n|\Z)",
        template,
    )
    assert match is not None
    return match.group("body")


def _config(**overrides: Any) -> AwsAgentRuntimeConfig:
    values: dict[str, Any] = {
        "region": "us-east-1",
        "bedrock_model": "model-id",
        "microvm_image": "image-arn",
        "microvm_egress_connector": "connector-arn",
        "microvm_execution_role_arn": "execution-role-arn",
        "task_private_ipv4": "10.0.1.20",
        "artifact_bucket": "artifact-bucket",
        "service_secret_id": "secret-arn",
        "receiver_origin": "http://receiver.service.local:8080",
        "efs_file_system_id": "fs-123",
        "efs_access_point_id": "fsap-123",
        "efs_mount_target_ip": "10.0.2.30",
    }
    values.update(overrides)
    return AwsAgentRuntimeConfig(**values)


def test_build_runtime_composes_efs_artifacts_vault_and_private_egress() -> None:
    runtime = build_runtime(
        _config(),
        lambda_client=object(),
        s3_client=object(),
        secrets_manager_client=object(),
    )

    assert isinstance(runtime.artifact_store, S3ArtifactStore)
    assert isinstance(runtime.vault, SecretsManagerVault)
    assert isinstance(runtime.workspace_binding, EFSAccessPointBinding)
    assert runtime.environment_factory is not None
    assert runtime.egress_adapter.exposure.task_ipv4 == "10.0.1.20"
    assert runtime.egress_adapter.probe_metadata is False
    assert runtime.internal_service_host == "receiver.internal"


def test_build_runtime_can_opt_into_s3_files_workspace() -> None:
    runtime = build_runtime(
        _config(
            workspace_backend="s3files",
            s3files_file_system_id="s3fs-123",
            s3files_access_point_id="s3fsap-123",
            s3files_mount_target_ip="10.0.2.31",
            s3files_availability_zone_id="use1-az1",
        ),
        lambda_client=object(),
        s3_client=object(),
        secrets_manager_client=object(),
    )

    assert isinstance(runtime.workspace_binding, S3FilesAccessPointBinding)


class _Runner:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def exec(self, command: Any, **kwargs: Any) -> ExecResult:
        self.calls.append({"command": command, **kwargs})
        return ExecResult(stdout='{"accepted":true}', exit_code=0)


class _Artifacts:
    id = "artifacts"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def put_bytes(self, content: bytes, *, filename: str, **kwargs: Any) -> Any:
        self.calls.append({"content": content, "filename": filename, **kwargs})

        class _Artifact:
            id = "art_123"

        return _Artifact()

    async def read_bytes(self, artifact_id: str, *, max_bytes: int | None = None) -> Any:
        raise NotImplementedError

    async def list(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def delete(self, artifact_id: str) -> None:
        raise NotImplementedError


def test_internal_action_tool_uses_microvm_virtual_token_and_persists_artifact() -> None:
    runner = _Runner()
    artifacts = _Artifacts()
    tool = InternalActionTool(logical_host="receiver.internal")
    context = ToolContext(
        session_id="sess-1",
        agent_name="aws-agent",
        environment_name="aws",
        runner=runner,
        artifact_store=artifacts,
    )

    result = asyncio.run(tool.run(context, {"action": "reindex"}))

    command = runner.calls[0]["command"]
    assert "https://receiver.internal/v1/actions" in command.argv[2]
    assert "INTERNAL_SERVICE_TOKEN" in command.argv[2]
    assert runner.calls[0].get("env") is None
    assert artifacts.calls[0]["content"] == b'{"accepted":true}'
    assert artifacts.calls[0]["session_id"] == "sess-1"
    assert result.structured == {"accepted": True, "artifact_id": "art_123"}


def test_microvm_package_contains_build_context(tmp_path) -> None:
    archive = package(tmp_path / "microvm.zip")

    with zipfile.ZipFile(archive) as bundle:
        assert set(bundle.namelist()) == {
            "Dockerfile",
            "entrypoint.sh",
            "__init__.py",
            "app.py",
            "supervisor.py",
            "requirements.txt",
        }
        dockerfile = bundle.read("Dockerfile").decode("utf-8")
        assert "COPY requirements.txt /opt/cayu/requirements.txt" in dockerfile
        assert "COPY __init__.py app.py supervisor.py" in dockerfile
        assert "COPY entrypoint.sh /opt/cayu/entrypoint.sh" in dockerfile
        assert "COPY examples/" not in dockerfile


def test_cloudformation_makes_s3_files_opt_in_and_uses_https() -> None:
    template = Path("examples/aws/lambda_microvm_agent/infra.yaml").read_text(encoding="utf-8")

    assert "UseS3Files: !Equals [!Ref WorkspaceBackend, s3files]" in template
    for resource in (
        "WorkspaceBucket",
        "S3FilesSyncRole",
        "S3FilesFileSystem",
        "S3FilesMountTarget",
        "S3FilesAccessPoint",
        "S3FilesPolicy",
    ):
        section = _cloudformation_resource_body(template, resource)
        assert "Condition: UseS3Files" in section
    assert "CertificateArn:" in template
    assert "Port: 443" in template
    assert "Protocol: HTTPS" in template
    assert 'Value: !Sub "https://${LoadBalancer.DNSName}"' in template
    assert 'BaseImageVersion: "0"' in template
    assert "EgressNetworkConnectors: []" in template
    assert "Value: !Ref MicrovmNetworkConnector" in template
    assert "aws-network-connector:ALL_INGRESS" in template
    for hook in ("Run", "Resume", "Suspend", "Terminate"):
        assert f"{hook}: ENABLED" in template
    assert "Ready: ENABLED" in template


def test_cloudformation_keeps_control_api_outside_microvm_proxy_port_range() -> None:
    template = Path("examples/aws/lambda_microvm_agent/infra.yaml").read_text(encoding="utf-8")

    control_ingress = _cloudformation_resource_body(template, "ControlFromLoadBalancer")
    control_port_match = re.search(r"FromPort: (\d+)", control_ingress)
    assert control_port_match is not None
    control_port = int(control_port_match.group(1))
    microvm_rules = (
        _cloudformation_resource_body(template, "ControlFromMicrovm"),
        _cloudformation_resource_body(template, "MicrovmToControl"),
    )
    for rule in microvm_rules:
        proxy_from, proxy_to = (
            int(value) for value in re.findall(r"(?:FromPort|ToPort): (\d+)", rule)
        )
        assert not proxy_from <= control_port <= proxy_to

    dockerfile = Path("examples/aws/lambda_microvm_agent/Dockerfile").read_text(encoding="utf-8")
    assert f"EXPOSE {control_port}" in dockerfile
    assert f'"--port", "{control_port}"' in dockerfile
