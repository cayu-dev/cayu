"""Verify Lambda MicroVM metadata-isolation capability against a deployed stack."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import hmac
import json
import os
import secrets
import tempfile
import time
import urllib.request
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from examples.aws.lambda_microvm_agent.runtime import (
    INTERNAL_SERVICE_HOST,
    INTERNAL_SERVICE_POLICY,
    INTERNAL_SERVICE_TOKEN,
)

from cayu import (
    BoundWorkspace,
    ExecCommand,
    LocalWorkspace,
    RunnerWorkspace,
    SecretRef,
    SecretsManagerVault,
    SyncBinding,
)
from cayu.egress import (
    EgressBinding,
    HttpEgressPolicy,
    HttpxUpstream,
    TransparentEgressBroker,
    VirtualCredentialGrant,
    VirtualCredentialRegistry,
    VirtualEgressRunnerRequest,
)
from cayu.egress.aws_lambda_microvm_adapter import (
    LambdaMicroVMEgressAdapter,
    LambdaMicroVMMetadataIsolationMode,
)
from cayu.egress.proxy_exposure import VpcTaskProxyExposure

EVIDENCE_PREFIX = "CAYU_NIGHTLY_EVIDENCE="
EVIDENCE_SCHEMA = "cayu.aws_lambda_microvm_metadata_isolation.v1"
REQUIRED_EVIDENCE_VALUES = {
    "adapter": "aws-lambda-microvm",
    "agent_capabilities": "empty",
    "agent_identity": "uid-gid-1000",
    "agent_network_namespace": "verified",
    "agent_no_new_privs": "verified",
    "agent_routes": "relay-only",
    "aws_credential_material": "absent",
    "cleanup": "verified",
    "direct_public_egress": "denied",
    "execution_role": "configured",
    "metadata_credentials": "absent",
    "metadata_endpoint": "denied",
    "metadata_isolation": "verified",
    "proof_source": "real-aws",
    "proxy_reachability": "verified",
    "required_metadata_isolation": "verified",
    "revocation": "verified",
    "runtime_capability_schema": "cayu.egress_capabilities.v1",
    "schema": EVIDENCE_SCHEMA,
    "scoped_request": "verified",
    "sidecar": "verified",
    "sidecar_api": "denied",
    "vault_canary": "absent",
    "virtual_credential": "verified",
    "workspace_release": "verified",
}
REQUIRED_EVIDENCE_POSITIVE_INTEGERS = frozenset(
    {
        "credential_paths_checked",
        "filesystem_files_inspected",
        "processes_inspected",
    }
)
_CHECK_FLAG = "CAYU_AWS_METADATA_ISOLATION_LIVE"
_RUN_ID_ENV = "CAYU_AWS_METADATA_ISOLATION_RUN_ID"


@dataclass(frozen=True)
class _PreparedVirtualCredentialBoundary:
    session_id: str
    registry: VirtualCredentialRegistry
    grant: VirtualCredentialGrant
    binding: EgressBinding
    ca_path: Path


async def main() -> None:
    if os.environ.get(_CHECK_FLAG) != "1":
        raise SystemExit(f"Set {_CHECK_FLAG}=1 to run this contract.")
    await _run_in_control_task()


async def _run_in_control_task() -> None:
    import boto3  # ty: ignore[unresolved-import]

    region = _required_region()
    run_id = _required_env(_RUN_ID_ENV)
    image = _required_env("CAYU_LAMBDA_MICROVM_IMAGE")
    connector = _required_env("CAYU_LAMBDA_MICROVM_EGRESS_CONNECTOR")
    execution_role_arn = _required_env("CAYU_LAMBDA_MICROVM_EXECUTION_ROLE")
    secret_id = _required_env("CAYU_INTERNAL_SERVICE_SECRET")
    receiver_origin = os.environ.get(
        "CAYU_INTERNAL_SERVICE_ORIGIN", "http://receiver.service.local:8080"
    )
    lambda_client = boto3.client("lambda-microvms", region_name=region)
    secrets_client = boto3.client("secretsmanager", region_name=region)
    vault = SecretsManagerVault(
        {"internal_service_token": secret_id},
        client=secrets_client,
    )
    resolved = await vault.resolve(SecretRef(name="internal_service_token"))
    trusted_values = [resolved.value.get_secret_value()]
    for name in ("CAYU_SERVER_PASSWORD", "CAYU_DATABASE_URL"):
        value = os.environ.get(name)
        if value:
            trusted_values.append(value)
    exposure = VpcTaskProxyExposure(_task_private_ipv4())
    required = _adapter(
        metadata_isolation="required",
        region=region,
        connector=connector,
        execution_role_arn=execution_role_arn,
        exposure=exposure,
        lambda_client=lambda_client,
    )
    evidence = await _verified_boundary_probe(
        adapter=required,
        image=image,
        vault=vault,
        receiver_origin=receiver_origin,
        trusted_values=trusted_values,
        lambda_client=lambda_client,
        run_id=run_id,
    )
    evidence["required_metadata_isolation"] = "verified"
    print(EVIDENCE_PREFIX + json.dumps(evidence, sort_keys=True), flush=True)


async def _verified_boundary_probe(
    *,
    adapter: LambdaMicroVMEgressAdapter,
    image: str,
    vault: SecretsManagerVault,
    receiver_origin: str,
    trusted_values: list[str],
    lambda_client: Any,
    run_id: str,
) -> dict[str, Any]:
    session_id = f"{run_id}-required"
    fingerprint_key = secrets.token_bytes(32)
    encoded_fingerprint_key = base64.b64encode(fingerprint_key).decode("ascii")
    runner = None
    workspace_binding: SyncBinding | None = None
    bound_workspace: BoundWorkspace | None = None
    microvm_id: str | None = None
    cleanup = "failed"
    async with _prepare_virtual_credential_boundary(
        adapter=adapter,
        vault=vault,
        receiver_origin=receiver_origin,
        session_id=session_id,
    ) as boundary:
        try:
            runner = await adapter.create_runner(
                _runner_request(
                    session_id=session_id,
                    image=image,
                    binding=boundary.binding,
                    virtual_token=boundary.grant.presented_value,
                    ca_path=boundary.ca_path,
                )
            )
            microvm_id = adapter.reconnect_metadata(runner)["microvm_id"]
            capability_evidence = adapter.capability_evidence(runner)
            direct_public_egress = capability_evidence.claim_for("direct_public_egress")
            metadata_isolation = capability_evidence.claim_for("metadata_isolation")
            proxy_reachability = capability_evidence.claim_for("proxy_reachability")
            if (
                direct_public_egress is None
                or metadata_isolation is None
                or proxy_reachability is None
            ):
                raise RuntimeError("Lambda MicroVM capability evidence is incomplete.")

            source_root = boundary.ca_path.parent / "trusted-workspace"
            source_root.mkdir()
            source_workspace = LocalWorkspace(source_root)
            await source_workspace.write_bytes("seed.txt", b"metadata-isolation-live")
            target_workspace = RunnerWorkspace(runner)
            workspace_binding = SyncBinding(
                target_workspace=target_workspace,
                path="/workspace",
                max_files=32,
                max_total_bytes=1024 * 1024,
                max_archive_bytes=2 * 1024 * 1024,
            )
            bound_workspace = await workspace_binding.bind(
                source_workspace,
                runner,
                session_id=session_id,
                agent_name="metadata-isolation-live",
                environment_name="lambda-microvm",
            )

            audit = await runner.exec(
                ExecCommand.process(
                    "python3",
                    "-c",
                    _guest_audit_source(),
                    encoded_fingerprint_key,
                    "audit",
                ),
                timeout_s=60,
            )
            observations = _guest_evidence(audit, "audit")
            _assert_audit_observations(
                observations,
                trusted_values=trusted_values,
                fingerprint_key=fingerprint_key,
            )
            await target_workspace.write_bytes("boundary-result.json", b'{"scoped_request":202}')

            await boundary.registry.revoke_and_wait(boundary.grant.presented_value)
            revoked = await runner.exec(
                ExecCommand.process(
                    "python3",
                    "-c",
                    _guest_audit_source(),
                    encoded_fingerprint_key,
                    "revoked",
                ),
                timeout_s=30,
            )
            revoked_observations = _guest_evidence(revoked, "revocation")
            if revoked_observations.get("proxy_status") != 403:
                raise RuntimeError(
                    "Revoked virtual credential did not receive the expected HTTP 403."
                )

            await workspace_binding.finalize(bound_workspace, outcome="completed")
            bound_workspace = None
            synced = await source_workspace.read_bytes("boundary-result.json")
            if synced.content != b'{"scoped_request":202}':
                raise RuntimeError("Workspace result did not sync before MicroVM cleanup.")

            await adapter.finalize_runner(runner, outcome="completed")
            runner = None
            await boundary.binding.close()
            await _wait_for_termination(lambda_client, microvm_id)
            cleanup = "verified"
            return {
                "adapter": "aws-lambda-microvm",
                "agent_capabilities": "empty",
                "agent_identity": "uid-gid-1000",
                "agent_network_namespace": "verified",
                "agent_no_new_privs": "verified",
                "agent_routes": "relay-only",
                "aws_credential_material": "absent",
                "cleanup": cleanup,
                "credential_paths_checked": observations["credential_paths_checked"],
                "direct_public_egress": direct_public_egress.observation,
                "execution_role": "configured",
                "filesystem_files_inspected": observations["filesystem_files_inspected"],
                "metadata_credentials": (
                    "present" if observations["metadata_credentials_present"] else "absent"
                ),
                "metadata_endpoint": (
                    "reachable" if observations["metadata_network_reachable"] else "denied"
                ),
                "metadata_isolation": metadata_isolation.state,
                "processes_inspected": observations["processes_inspected"],
                "proof_source": "real-aws",
                "proxy_reachability": proxy_reachability.state,
                "revocation": "verified",
                "runtime_capability_schema": capability_evidence.schema_version,
                "run_id": run_id,
                "schema": EVIDENCE_SCHEMA,
                "scoped_request": "verified",
                "sidecar": "verified",
                "sidecar_api": "denied",
                "vault_canary": "absent",
                "virtual_credential": "verified",
                "workspace_release": "verified",
            }
        finally:
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await boundary.registry.revoke_and_wait(boundary.grant.presented_value)
            if workspace_binding is not None and bound_workspace is not None:
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await workspace_binding.finalize(bound_workspace, outcome="failed")
            if runner is not None:
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await adapter.finalize_runner(runner, outcome="failed")
            if cleanup != "verified" and microvm_id is not None:
                with contextlib.suppress(Exception):
                    await _wait_for_termination(lambda_client, microvm_id)


@contextlib.asynccontextmanager
async def _prepare_virtual_credential_boundary(
    *,
    adapter: LambdaMicroVMEgressAdapter,
    vault: SecretsManagerVault,
    receiver_origin: str,
    session_id: str,
) -> AsyncIterator[_PreparedVirtualCredentialBoundary]:
    registry = VirtualCredentialRegistry()
    grant = registry.mint(
        session_id=session_id,
        env_name=INTERNAL_SERVICE_TOKEN,
        secret=SecretRef(name="internal_service_token"),
        destination=INTERNAL_SERVICE_HOST,
        credential_kind="opaque_bearer",
        policy_name=INTERNAL_SERVICE_POLICY,
    )
    broker = TransparentEgressBroker(
        registry=registry,
        resolver=vault,
        policies={
            INTERNAL_SERVICE_POLICY: HttpEgressPolicy(
                name=INTERNAL_SERVICE_POLICY,
                allowed_hosts=[INTERNAL_SERVICE_HOST],
                allowed_endpoints=[("POST", "/v1/actions")],
            )
        },
        upstream=HttpxUpstream(routes={INTERNAL_SERVICE_HOST: receiver_origin}),
        require_test_mode_credentials=False,
    )
    binding = await adapter.prepare(session_id=session_id, grants=[grant], broker=broker)
    try:
        with tempfile.TemporaryDirectory(prefix="cayu-aws-metadata-isolation-") as directory:
            ca_path = Path(directory) / "ca.pem"
            ca_path.write_bytes(binding.ca_cert_pem or b"")
            yield _PreparedVirtualCredentialBoundary(
                session_id=session_id,
                registry=registry,
                grant=grant,
                binding=binding,
                ca_path=ca_path,
            )
    finally:
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await registry.revoke_and_wait(grant.presented_value)
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await binding.close()


def _adapter(
    *,
    metadata_isolation: LambdaMicroVMMetadataIsolationMode,
    region: str,
    connector: str,
    execution_role_arn: str,
    exposure: VpcTaskProxyExposure,
    lambda_client: Any,
) -> LambdaMicroVMEgressAdapter:
    return LambdaMicroVMEgressAdapter(
        region_name=region,
        egress_network_connector_arn=connector,
        execution_role_arn=execution_role_arn,
        exposure=exposure,
        client=lambda_client,
        metadata_isolation=metadata_isolation,
    )


def _runner_request(
    *,
    session_id: str,
    image: str,
    binding: Any,
    virtual_token: str,
    ca_path: Path,
) -> VirtualEgressRunnerRequest:
    return VirtualEgressRunnerRequest(
        name=session_id,
        runner_kind="lambda-microvm",
        image=image,
        binding=binding,
        env_overlay={**binding.env, INTERNAL_SERVICE_TOKEN: virtual_token},
        ca_cert_host_path=str(ca_path),
        guest_ca_path="/etc/cayu/ca.pem",
        setup_commands=(),
        egress_destinations=(INTERNAL_SERVICE_HOST,),
        session_id=session_id,
    )


def _secret_fingerprint(value: str, *, key: bytes) -> str:
    return hmac.new(key, value.encode("utf-8"), hashlib.sha256).hexdigest()


def _guest_evidence(result: Any, phase: str) -> dict[str, Any]:
    if result.exit_code != 0 or result.timed_out:
        detail = (result.stderr or result.stdout).strip()[:1000]
        raise RuntimeError(f"Guest {phase} probe failed: {detail}")
    try:
        decoded = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Guest {phase} probe returned invalid JSON.") from exc
    if type(decoded) is not dict:
        raise RuntimeError(f"Guest {phase} probe must return an object.")
    return decoded


def _assert_audit_observations(
    observations: dict[str, Any],
    *,
    trusted_values: list[str],
    fingerprint_key: bytes,
) -> None:
    expected = {
        "direct_public_reachable": False,
        "aws_credentials_present": False,
        "candidate_fingerprint_overflow": False,
        "cap_ambient": 0,
        "cap_bounding": 0,
        "cap_effective": 0,
        "cap_inheritable": 0,
        "effective_gid": 1000,
        "effective_uid": 1000,
        "metadata_credentials_present": False,
        "metadata_network_reachable": False,
        "network_routes": ["192.0.2.0/30"],
        "no_new_privs": True,
        "sidecar_api_reachable": False,
        "unexpected_virtual_credentials": False,
        "virtual_credential_count": 1,
        "virtual_credential_present": True,
        "proxy_status": 202,
    }
    mismatches = {
        key: {"expected": value, "actual": observations.get(key)}
        for key, value in expected.items()
        if observations.get(key) != value
    }
    network_namespace = observations.get("network_namespace")
    init_network_namespace = observations.get("init_network_namespace")
    init_network_namespace_access = observations.get("init_network_namespace_access")
    init_namespace_verified = (
        init_network_namespace_access == "readable"
        and type(init_network_namespace) is str
        and init_network_namespace.startswith("net:[")
        and network_namespace != init_network_namespace
    ) or (init_network_namespace_access == "permission-denied" and init_network_namespace is None)
    if (
        type(network_namespace) is not str
        or not network_namespace.startswith("net:[")
        or not init_namespace_verified
    ):
        mismatches["network_namespace"] = {
            "expected": (
                "agent namespace distinct from readable init namespace, or init namespace "
                "read denied"
            ),
            "actual": {
                "agent": network_namespace,
                "init": init_network_namespace,
                "init_access": init_network_namespace_access,
            },
        }
    raw_fingerprints = observations.get("candidate_fingerprints")
    fingerprints: set[str] = set()
    if (
        type(raw_fingerprints) is not list
        or not raw_fingerprints
        or len(raw_fingerprints) > 8192
        or any(type(value) is not str or len(value) != 64 for value in raw_fingerprints)
    ):
        mismatches["candidate_fingerprints"] = {
            "expected": "one to 8192 SHA-256 HMAC fingerprints",
            "actual": type(raw_fingerprints).__name__,
        }
    else:
        try:
            for value in raw_fingerprints:
                bytes.fromhex(value)
                fingerprints.add(value)
        except ValueError:
            mismatches["candidate_fingerprints"] = {
                "expected": "lowercase hexadecimal SHA-256 HMAC fingerprints",
                "actual": "invalid encoding",
            }
        if len(fingerprints) != len(raw_fingerprints):
            mismatches["candidate_fingerprints"] = {
                "expected": "unique SHA-256 HMAC fingerprints",
                "actual": "duplicates",
            }
    trusted_fingerprints = {
        _secret_fingerprint(value, key=fingerprint_key) for value in trusted_values
    }
    if fingerprints.intersection(trusted_fingerprints):
        mismatches["trusted_secret_fingerprints"] = {
            "expected": "no observed trusted-secret fingerprint",
            "actual": "one or more matches",
        }
    for key in ("credential_paths_checked", "filesystem_files_inspected", "processes_inspected"):
        value = observations.get(key)
        if type(value) is not int or value <= 0:
            mismatches[key] = {"expected": "positive integer", "actual": value}
    if mismatches:
        raise RuntimeError(
            "Guest isolation observations did not satisfy the live contract: "
            + json.dumps(mismatches, sort_keys=True)
        )


async def _wait_for_termination(client: Any, microvm_id: str) -> None:
    deadline = time.monotonic() + 90
    while True:
        response = await asyncio.to_thread(
            client.get_microvm,
            microvmIdentifier=microvm_id,
        )
        if response.get("state") == "TERMINATED":
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("Lambda MicroVM did not reach TERMINATED during live cleanup.")
        await asyncio.sleep(2)


def _required_region() -> str:
    value = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if value is None or not value.strip():
        raise RuntimeError("Set AWS_REGION or AWS_DEFAULT_REGION.")
    return value.strip()


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise RuntimeError(f"Set {name}.")
    return value.strip()


def _task_private_ipv4() -> str:
    metadata_uri = _required_env("ECS_CONTAINER_METADATA_URI_V4")
    with urllib.request.urlopen(f"{metadata_uri}/task", timeout=2) as response:
        metadata = json.load(response)
    for container in metadata.get("Containers", []):
        for network in container.get("Networks", []):
            addresses = network.get("IPv4Addresses", [])
            if addresses:
                return str(addresses[0])
    raise RuntimeError("ECS task metadata did not contain a private IPv4 address.")


def _guest_audit_source() -> str:
    return Path(__file__).with_name("metadata_isolation_guest.py").read_text()


if __name__ == "__main__":
    asyncio.run(main())
