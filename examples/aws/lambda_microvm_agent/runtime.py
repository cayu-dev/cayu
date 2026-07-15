"""Production composition for the AWS Lambda MicroVM agent example."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from cayu import (
    AgentSpec,
    ArtifactScope,
    BedrockProvider,
    CayuApp,
    EFSAccessPointBinding,
    EnvironmentSpec,
    ExecCommand,
    S3ArtifactStore,
    S3FilesAccessPointBinding,
    SecretRef,
    SecretsManagerVault,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
    VirtualCredentialSpec,
    VirtualEgressEnvironmentFactory,
)
from cayu.egress import HttpEgressPolicy, HttpxUpstream
from cayu.egress.aws_lambda_microvm_adapter import LambdaMicroVMEgressAdapter
from cayu.egress.proxy_exposure import VpcTaskProxyExposure
from cayu.environments import WorkspaceBinding

INTERNAL_SERVICE_HOST = "receiver.internal"
INTERNAL_SERVICE_TOKEN = "INTERNAL_SERVICE_TOKEN"
INTERNAL_SERVICE_POLICY = "internal-actions"


@dataclass(frozen=True)
class AwsAgentRuntimeConfig:
    region: str
    bedrock_model: str
    microvm_image: str
    microvm_egress_connector: str
    microvm_execution_role_arn: str
    task_private_ipv4: str
    artifact_bucket: str
    service_secret_id: str
    receiver_origin: str
    workspace_backend: Literal["efs", "s3files"] = "efs"
    efs_file_system_id: str | None = None
    efs_access_point_id: str | None = None
    efs_mount_target_ip: str | None = None
    s3files_file_system_id: str | None = None
    s3files_access_point_id: str | None = None
    s3files_mount_target_ip: str | None = None
    s3files_availability_zone_id: str | None = None
    artifact_prefix: str = "cayu/artifacts"
    internal_service_host: str = INTERNAL_SERVICE_HOST


@dataclass(frozen=True)
class AwsAgentRuntime:
    config: AwsAgentRuntimeConfig
    environment_factory: VirtualEgressEnvironmentFactory
    egress_adapter: LambdaMicroVMEgressAdapter
    workspace_binding: WorkspaceBinding
    artifact_store: S3ArtifactStore
    vault: SecretsManagerVault
    internal_service_host: str


def build_runtime(
    config: AwsAgentRuntimeConfig,
    *,
    lambda_client: Any | None = None,
    s3_client: Any | None = None,
    secrets_manager_client: Any | None = None,
) -> AwsAgentRuntime:
    """Compose AWS services without ever injecting a real secret into the guest."""
    workspace_binding = _workspace_binding(config)
    artifact_store = S3ArtifactStore(
        config.artifact_bucket,
        prefix=config.artifact_prefix,
        region_name=None if s3_client is not None else config.region,
        client=s3_client,
    )
    vault = SecretsManagerVault(
        {"internal_service_token": config.service_secret_id},
        region_name=None if secrets_manager_client is not None else config.region,
        client=secrets_manager_client,
    )
    adapter = LambdaMicroVMEgressAdapter(
        region_name=config.region,
        egress_network_connector_arn=config.microvm_egress_connector,
        exposure=VpcTaskProxyExposure(config.task_private_ipv4),
        execution_role_arn=config.microvm_execution_role_arn,
        metadata_isolation="required",
        client=lambda_client,
    )
    factory = VirtualEgressEnvironmentFactory(
        resolver=vault,
        policies={
            INTERNAL_SERVICE_POLICY: HttpEgressPolicy(
                name=INTERNAL_SERVICE_POLICY,
                allowed_hosts=[config.internal_service_host],
                allowed_endpoints=[("POST", "/v1/actions")],
            )
        },
        credentials=[
            VirtualCredentialSpec(
                env_name=INTERNAL_SERVICE_TOKEN,
                secret=SecretRef(name="internal_service_token"),
                destination=config.internal_service_host,
                policy_name=INTERNAL_SERVICE_POLICY,
                credential_kind="opaque_bearer",
            )
        ],
        image=config.microvm_image,
        adapter=adapter,
        inner_binding=workspace_binding,
        artifact_store=artifact_store,
        upstream=HttpxUpstream(routes={config.internal_service_host: config.receiver_origin}),
        require_test_mode_credentials=False,
    )
    return AwsAgentRuntime(
        config=config,
        environment_factory=factory,
        egress_adapter=adapter,
        workspace_binding=workspace_binding,
        artifact_store=artifact_store,
        vault=vault,
        internal_service_host=config.internal_service_host,
    )


def build_app(
    runtime: AwsAgentRuntime,
    *,
    session_store: Any | None = None,
    task_store: Any | None = None,
    bedrock_client: Any | None = None,
) -> CayuApp:
    """Build a role-authenticated Bedrock agent on the integrated environment."""
    app = CayuApp(session_store=session_store, task_store=task_store)
    app.register_provider(
        BedrockProvider(
            region_name=None if bedrock_client is not None else runtime.config.region,
            client=bedrock_client,
        ),
        default=True,
    )
    app.register_environment_factory(
        EnvironmentSpec(
            name="aws-lambda-microvm",
            metadata={
                "kind": "lambda-microvm",
                "workspace_backend": runtime.config.workspace_backend,
                "artifact_backend": "s3",
                "vault_backend": "secrets-manager",
                "egress_configuration": runtime.egress_adapter.configuration_metadata(),
            },
        ),
        runtime.environment_factory,
        default=True,
    )
    app.register_agent(
        AgentSpec(
            name="aws-agent",
            provider_name="bedrock",
            model=runtime.config.bedrock_model,
            system_prompt=(
                "You operate in an isolated AWS Lambda MicroVM. Use request_internal_action "
                "when the user asks the trusted internal service to perform an action. The "
                "tool persists the response in both the durable workspace and S3 artifacts."
            ),
        ),
        tools=[InternalActionTool(logical_host=runtime.internal_service_host)],
    )
    return app


class InternalActionTool(Tool):
    """Send one policy-shaped request from the MicroVM through virtual egress."""

    spec = ToolSpec(
        name="request_internal_action",
        description="Request one named action from the trusted internal service.",
        parallel_safe=False,
        input_schema={
            "type": "object",
            "properties": {"action": {"type": "string", "minLength": 1, "maxLength": 128}},
            "required": ["action"],
            "additionalProperties": False,
        },
    )

    def __init__(self, *, logical_host: str) -> None:
        super().__init__()
        if not re.fullmatch(r"[a-z0-9.-]+", logical_host):
            raise ValueError("logical_host must be a lowercase DNS hostname.")
        self.logical_host = logical_host

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        if ctx.runner is None:
            return ToolResult(content="The environment has no runner.", is_error=True)
        if ctx.artifact_store is None:
            return ToolResult(content="The environment has no artifact store.", is_error=True)
        action = args.get("action")
        if not isinstance(action, str) or not action.strip():
            return ToolResult(content="action must be a nonblank string.", is_error=True)

        output_name = _safe_output_name(ctx.idempotency_key or ctx.session_id)
        script = _internal_action_script(self.logical_host)
        result = await ctx.runner.exec(
            ExecCommand.process("python3", "-c", script, action.strip(), output_name),
            timeout_s=30,
        )
        if result.exit_code != 0 or result.timed_out:
            detail = (result.stderr or result.stdout).strip()[:500]
            return ToolResult(
                content=f"Internal action request failed: {detail}",
                is_error=True,
            )

        response_bytes = result.stdout.encode("utf-8")
        artifact = await ctx.artifact_store.put_bytes(
            response_bytes,
            filename=f"internal-actions/{output_name}.json",
            content_type="application/json",
            scope=ArtifactScope.SESSION,
            session_id=ctx.session_id,
            agent_name=ctx.agent_name,
            environment_name=ctx.environment_name,
            metadata={"logical_destination": self.logical_host, "action": action.strip()},
        )
        try:
            decoded = json.loads(result.stdout)
        except json.JSONDecodeError:
            decoded = {"response": result.stdout}
        structured = decoded if isinstance(decoded, dict) else {"response": decoded}
        structured["artifact_id"] = artifact.id
        return ToolResult(
            content=f"Internal action accepted; response artifact {artifact.id}.",
            structured=structured,
            artifacts=[{"artifact_id": artifact.id}],
        )


def _workspace_binding(config: AwsAgentRuntimeConfig) -> WorkspaceBinding:
    if config.workspace_backend == "efs":
        return EFSAccessPointBinding(
            file_system_id=_required(config.efs_file_system_id, "efs_file_system_id"),
            access_point_id=_required(config.efs_access_point_id, "efs_access_point_id"),
            mount_target_ip=_required(config.efs_mount_target_ip, "efs_mount_target_ip"),
            workspace_id="aws:efs:agent-workspace",
        )
    if config.workspace_backend == "s3files":
        return S3FilesAccessPointBinding(
            file_system_id=_required(config.s3files_file_system_id, "s3files_file_system_id"),
            access_point_id=_required(config.s3files_access_point_id, "s3files_access_point_id"),
            mount_target_ip=_required(config.s3files_mount_target_ip, "s3files_mount_target_ip"),
            availability_zone_id=_required(
                config.s3files_availability_zone_id,
                "s3files_availability_zone_id",
            ),
            region_name=config.region,
            workspace_id="aws:s3files:agent-workspace",
        )
    raise ValueError(f"Unsupported workspace_backend: {config.workspace_backend}")


def _internal_action_script(logical_host: str) -> str:
    return f"""
import json
import os
import pathlib
import sys
import urllib.request

action = sys.argv[1]
output_name = sys.argv[2]
payload = json.dumps({{"action": action}}).encode("utf-8")
request = urllib.request.Request(
    "https://{logical_host}/v1/actions",
    data=payload,
    headers={{
        "Authorization": "Bearer " + os.environ["{INTERNAL_SERVICE_TOKEN}"],
        "Content-Type": "application/json",
    }},
    method="POST",
)
with urllib.request.urlopen(request, timeout=20) as response:
    body = response.read()
path = pathlib.Path("/workspace/actions") / (output_name + ".json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_bytes(body)
sys.stdout.write(body.decode("utf-8"))
""".strip()


def _required(value: str | None, name: str) -> str:
    if value is None or not value.strip():
        raise ValueError(f"{name} is required for the selected workspace backend.")
    return value.strip()


def _safe_output_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip(".-")
    return (normalized or "action")[:128]
