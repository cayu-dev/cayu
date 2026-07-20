from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import re
import sys
import types
import zipfile
from pathlib import Path
from typing import Any

import pytest
from examples.aws.lambda_microvm_agent import metadata_isolation_live, metadata_isolation_task
from examples.aws.lambda_microvm_agent.package_microvm import package
from examples.aws.lambda_microvm_agent.runtime import (
    AwsAgentRuntimeConfig,
    InternalActionTool,
    build_app,
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
    assert runtime.egress_adapter.metadata_isolation == "required"
    assert runtime.internal_service_host == "receiver.internal"


def test_metadata_isolation_control_uses_deployed_microvm_execution_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _SecretsClient:
        def get_secret_value(self, **kwargs: Any) -> dict[str, str]:
            assert kwargs["SecretId"] == "secret-arn"
            return {"SecretString": "vault-canary"}

    lambda_client = object()

    def client(service: str, **kwargs: Any) -> Any:
        assert kwargs == {"region_name": "us-east-1"}
        if service == "lambda-microvms":
            return lambda_client
        if service == "secretsmanager":
            return _SecretsClient()
        raise AssertionError(f"unexpected AWS client: {service}")

    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    monkeypatch.setenv("CAYU_AWS_METADATA_ISOLATION_LIVE", "1")
    monkeypatch.setenv("CAYU_AWS_METADATA_ISOLATION_RUN_ID", "run-1")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("CAYU_LAMBDA_MICROVM_IMAGE", "image-arn")
    monkeypatch.setenv("CAYU_LAMBDA_MICROVM_EGRESS_CONNECTOR", "connector-arn")
    monkeypatch.setenv("CAYU_LAMBDA_MICROVM_EXECUTION_ROLE", "execution-role-arn")
    monkeypatch.setenv("CAYU_INTERNAL_SERVICE_SECRET", "secret-arn")
    monkeypatch.setattr(metadata_isolation_task, "_task_private_ipv4", lambda: "10.0.1.20")
    captured: dict[str, Any] = {}

    async def verified_boundary_probe(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"run_id": "run-1"}

    monkeypatch.setattr(
        metadata_isolation_task,
        "_verified_boundary_probe",
        verified_boundary_probe,
    )

    asyncio.run(metadata_isolation_task.main())

    adapter = captured["adapter"]
    assert adapter.execution_role_arn == "execution-role-arn"
    assert captured["lambda_client"] is lambda_client


def _complete_metadata_isolation_evidence() -> dict[str, Any]:
    return {
        "adapter": "aws-lambda-microvm",
        "agent_capabilities": "empty",
        "agent_identity": "uid-gid-1000",
        "agent_network_namespace": "verified",
        "agent_no_new_privs": "verified",
        "agent_routes": "relay-only",
        "aws_credential_material": "absent",
        "cleanup": "verified",
        "credential_paths_checked": 7,
        "direct_public_egress": "denied",
        "execution_role": "configured",
        "filesystem_files_inspected": 3,
        "metadata_credentials": "absent",
        "metadata_endpoint": "denied",
        "metadata_isolation": "verified",
        "processes_inspected": 2,
        "proof_source": "real-aws",
        "proxy_reachability": "verified",
        "required_metadata_isolation": "verified",
        "revocation": "verified",
        "runtime_capability_schema": "cayu.egress_capabilities.v1",
        "run_id": "metadata-isolation-abcdef123456",
        "schema": "cayu.aws_lambda_microvm_metadata_isolation.v1",
        "scoped_request": "verified",
        "sidecar": "verified",
        "sidecar_api": "denied",
        "vault_canary": "absent",
        "virtual_credential": "verified",
        "workspace_release": "verified",
    }


def _valid_audit_observations(**overrides: Any) -> dict[str, Any]:
    observations: dict[str, Any] = {
        "aws_credentials_present": False,
        "candidate_fingerprint_overflow": False,
        "candidate_fingerprints": ["0" * 64],
        "cap_ambient": 0,
        "cap_bounding": 0,
        "cap_effective": 0,
        "cap_inheritable": 0,
        "credential_paths_checked": 7,
        "direct_public_reachable": False,
        "effective_gid": 1000,
        "effective_uid": 1000,
        "filesystem_files_inspected": 3,
        "init_network_namespace": "net:[root]",
        "init_network_namespace_access": "readable",
        "metadata_credentials_present": False,
        "metadata_network_reachable": False,
        "network_namespace": "net:[agent]",
        "network_routes": ["192.0.2.0/30"],
        "no_new_privs": True,
        "processes_inspected": 2,
        "proxy_status": 202,
        "sidecar_api_reachable": False,
        "unexpected_virtual_credentials": False,
        "virtual_credential_count": 1,
        "virtual_credential_present": True,
    }
    observations.update(overrides)
    return observations


def _run_metadata_isolation_launcher(
    monkeypatch: pytest.MonkeyPatch,
    evidence_records: list[dict[str, Any]],
) -> None:
    outputs = {
        "ClusterArn": "cluster-arn",
        "ControlTaskDefinitionArn": "task-definition-arn",
        "ControlSecurityGroupId": "sg-1",
        "PrivateSubnetId": "subnet-1",
        "ControlLogGroupName": "log-group",
    }

    class _CloudFormation:
        def describe_stacks(self, **kwargs: Any) -> dict[str, Any]:
            assert kwargs == {"StackName": "stack-name"}
            return {
                "Stacks": [
                    {
                        "Outputs": [
                            {"OutputKey": key, "OutputValue": value}
                            for key, value in outputs.items()
                        ]
                    }
                ]
            }

    class _ECS:
        def run_task(self, **kwargs: Any) -> dict[str, Any]:
            assert kwargs["cluster"] == "cluster-arn"
            return {"failures": [], "tasks": [{"taskArn": "task-arn"}]}

        def describe_tasks(self, **kwargs: Any) -> dict[str, Any]:
            assert kwargs == {"cluster": "cluster-arn", "tasks": ["task-arn"]}
            return {"tasks": [{"lastStatus": "STOPPED", "containers": [{"exitCode": 0}]}]}

    class _Logs:
        def filter_log_events(self, **kwargs: Any) -> dict[str, Any]:
            assert kwargs["logGroupName"] == "log-group"
            return {
                "events": [
                    {"message": metadata_isolation_task.EVIDENCE_PREFIX + json.dumps(record)}
                    for record in evidence_records
                ]
            }

    clients = {
        "cloudformation": _CloudFormation(),
        "ecs": _ECS(),
        "logs": _Logs(),
    }

    class _Session:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs == {"region_name": "us-east-1"}

        def client(self, service: str) -> Any:
            return clients[service]

    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.Session = _Session  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    monkeypatch.setattr(
        metadata_isolation_live.uuid,
        "uuid4",
        lambda: types.SimpleNamespace(hex="abcdef1234567890"),
    )
    monkeypatch.setenv("CAYU_AWS_METADATA_ISOLATION_LIVE", "1")
    monkeypatch.setenv("CAYU_AWS_METADATA_ISOLATION_STACK", "stack-name")
    monkeypatch.setenv("AWS_REGION", "us-east-1")

    asyncio.run(metadata_isolation_live.main())


def test_metadata_isolation_launcher_rejects_incomplete_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _complete_metadata_isolation_evidence()
    del evidence["cleanup"]

    with pytest.raises(RuntimeError, match="cleanup"):
        _run_metadata_isolation_launcher(monkeypatch, [evidence])


@pytest.mark.parametrize("field", sorted(_complete_metadata_isolation_evidence()))
def test_metadata_isolation_launcher_rejects_each_invalid_evidence_value(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    evidence = _complete_metadata_isolation_evidence()
    evidence[field] = "invalid"

    with pytest.raises(RuntimeError, match=field):
        _run_metadata_isolation_launcher(monkeypatch, [evidence])


def test_metadata_isolation_launcher_rejects_unexpected_evidence_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _complete_metadata_isolation_evidence()
    evidence["untrusted_claim"] = "verified"

    with pytest.raises(RuntimeError, match="untrusted_claim"):
        _run_metadata_isolation_launcher(monkeypatch, [evidence])


def test_metadata_isolation_launcher_rejects_duplicate_evidence_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _complete_metadata_isolation_evidence()

    with pytest.raises(RuntimeError, match="exactly one evidence record"):
        _run_metadata_isolation_launcher(monkeypatch, [evidence, evidence])


def test_metadata_isolation_launcher_accepts_exact_versioned_evidence(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    evidence = _complete_metadata_isolation_evidence()

    _run_metadata_isolation_launcher(monkeypatch, [evidence])

    assert capsys.readouterr().out.strip() == (
        metadata_isolation_task.EVIDENCE_PREFIX + json.dumps(evidence, sort_keys=True)
    )


def test_metadata_isolation_launcher_paginates_cloudwatch_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _complete_metadata_isolation_evidence()

    class _Logs:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def filter_log_events(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                assert "nextToken" not in kwargs
                return {"events": [{"message": "task started"}], "nextToken": "page-2"}
            assert kwargs["nextToken"] == "page-2"
            return {
                "events": [
                    {"message": metadata_isolation_task.EVIDENCE_PREFIX + json.dumps(evidence)}
                ]
            }

    logs = _Logs()
    monkeypatch.setattr(metadata_isolation_live.time, "sleep", lambda _seconds: None)

    messages = metadata_isolation_live._evidence_messages(
        logs,
        log_group="log-group",
        run_id=evidence["run_id"],
        start_time_ms=123,
    )

    assert len(logs.calls) == 2
    assert any(metadata_isolation_task.EVIDENCE_PREFIX in message for message in messages)


def test_build_app_declares_required_metadata_isolation() -> None:
    runtime = build_runtime(
        _config(),
        lambda_client=object(),
        s3_client=object(),
        secrets_manager_client=object(),
    )
    app = build_app(runtime, bedrock_client=object())

    registration = app.list_environment_registrations()[0]

    assert "egress_capabilities" not in registration.spec.metadata
    assert registration.spec.metadata["egress_configuration"] == {
        "metadata_isolation_mode": "required",
    }
    assert registration.environment.artifact_store is runtime.artifact_store


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
        entrypoint = bundle.read("entrypoint.sh").decode("utf-8")
        assert "COPY requirements.txt /opt/cayu/requirements.txt" in dockerfile
        assert "COPY __init__.py app.py supervisor.py" in dockerfile
        assert "COPY entrypoint.sh /opt/cayu/entrypoint.sh" in dockerfile
        assert "COPY examples/" not in dockerfile
        assert "iptables-nft" in dockerfile
        assert "useradd --uid 1000" in dockerfile
        assert 'ip netns add "$CAYU_MICROVM_AGENT_NETNS"' in entrypoint
        assert "--dport 18080 -j ACCEPT" in entrypoint
        assert "-i cayu-root -j REJECT" in entrypoint


def test_guest_audit_treats_unreadable_root_paths_as_absent() -> None:
    script = Path("examples/aws/lambda_microvm_agent/metadata_isolation_guest.py").read_text(
        encoding="utf-8"
    )

    root_probe = script.split("filesystem_files = []", 1)[1].split("filesystem_sources = []", 1)[0]
    assert root_probe.count("except OSError:") == 2
    assert '"/root/.aws"' in root_probe


def test_guest_audit_records_unreadable_init_network_namespace() -> None:
    script = Path("examples/aws/lambda_microvm_agent/metadata_isolation_guest.py").read_text(
        encoding="utf-8"
    )

    namespace_probe = script.split("def read_init_network_namespace", 1)[1].split(
        "def process_status", 1
    )[0]
    assert "except PermissionError:" in namespace_probe
    assert 'return None, "permission-denied"' in namespace_probe


def test_live_boundary_validation_accepts_denied_init_namespace_read() -> None:
    observations = _valid_audit_observations(
        init_network_namespace=None,
        init_network_namespace_access="permission-denied",
    )

    metadata_isolation_task._assert_audit_observations(
        observations,
        trusted_values=[],
        fingerprint_key=b"k" * 32,
    )


@pytest.mark.parametrize("access", ["missing", "os-error", None])
def test_live_boundary_validation_rejects_unverified_init_namespace(access: str | None) -> None:
    observations = _valid_audit_observations(
        init_network_namespace=None,
        init_network_namespace_access=access,
    )

    with pytest.raises(RuntimeError, match="network_namespace"):
        metadata_isolation_task._assert_audit_observations(
            observations,
            trusted_values=[],
            fingerprint_key=b"k" * 32,
        )


def test_live_boundary_validation_rejects_agent_access_to_trusted_sidecar() -> None:
    observations = _valid_audit_observations(sidecar_api_reachable=True)

    with pytest.raises(RuntimeError, match="sidecar_api_reachable"):
        metadata_isolation_task._assert_audit_observations(
            observations,
            trusted_values=[],
            fingerprint_key=b"k" * 32,
        )


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("effective_uid", 0),
        ("effective_gid", 0),
        ("cap_effective", 1),
        ("cap_inheritable", 1),
        ("cap_ambient", 1),
        ("cap_bounding", 1),
        ("no_new_privs", False),
        ("network_namespace", "net:[root]"),
        ("network_routes", ["0.0.0.0/0", "192.0.2.0/30"]),
    ],
)
def test_live_boundary_validation_rejects_invalid_agent_process_boundary(
    field: str,
    invalid: Any,
) -> None:
    observations = _valid_audit_observations(**{field: invalid})

    with pytest.raises(RuntimeError, match=field):
        metadata_isolation_task._assert_audit_observations(
            observations,
            trusted_values=[],
            fingerprint_key=b"k" * 32,
        )


def test_live_boundary_validation_compares_secret_fingerprints_only_in_trusted_control() -> None:
    fingerprint_key = b"k" * 32
    secret = "guessable-password"
    secret_fingerprint = hmac.new(
        fingerprint_key,
        secret.encode(),
        hashlib.sha256,
    ).hexdigest()
    observations = _valid_audit_observations(candidate_fingerprints=[secret_fingerprint])

    with pytest.raises(RuntimeError, match="trusted_secret_fingerprints"):
        metadata_isolation_task._assert_audit_observations(
            observations,
            trusted_values=[secret],
            fingerprint_key=fingerprint_key,
        )


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
    for output in (
        "ClusterArn",
        "ControlTaskDefinitionArn",
        "ControlSecurityGroupId",
        "PrivateSubnetId",
        "ControlLogGroupName",
    ):
        assert f"  {output}:" in template
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
