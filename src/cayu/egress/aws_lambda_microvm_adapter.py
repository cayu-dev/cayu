from __future__ import annotations

import asyncio
import contextlib
import weakref
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypedDict

from cayu.egress._remote_adapter import (
    DEFAULT_PROXY_SERVER_FACTORY,
    ProxyServerFactory,
    prepare_exposed_proxy_binding,
    run_enforcement_preflight,
    run_setup_commands,
)
from cayu.egress.adapter import (
    EgressBinding,
    SandboxEgressAdapter,
    VirtualEgressRunnerRequest,
)
from cayu.egress.broker import TransparentEgressBroker
from cayu.egress.errors import UnsupportedEgressError
from cayu.egress.grants import VirtualCredentialGrant
from cayu.egress.proxy_exposure import ProxyExposure, VpcTaskProxyExposure
from cayu.runners import (
    ExecCommand,
    LambdaMicroVMProtocolError,
    LambdaMicroVMRunner,
    Runner,
)
from cayu.runners.aws_lambda_microvm import LambdaMicroVMEndpointTransport

DEFAULT_PREFLIGHT_TIMEOUT_SECONDS = 30


class _ReconnectIdentity(TypedDict):
    microvm_id: str
    endpoint: str
    region: str
    image_identifier: str
    image_version: str | None


class LambdaMicroVMEgressAdapter(SandboxEgressAdapter):
    """Run virtual-egress sandboxes in AWS Lambda MicroVMs.

    The trusted Cayu control plane and CONNECT proxy live in a private
    ECS/Fargate task. The MicroVM receives only virtual credentials, a private
    proxy URL, and an egress connector that can reach the task. A startup
    preflight proves that the broker is reachable while direct internet and
    metadata paths remain blocked. ``probe_metadata=False`` is available for
    Lambda configurations whose managed ingress shares the link-local metadata
    endpoint; callers using it must keep the execution role narrowly scoped.
    """

    runner_kind = "lambda-microvm"

    def __init__(
        self,
        *,
        region_name: str,
        egress_network_connector_arn: str,
        exposure: ProxyExposure,
        ingress_network_connectors: Sequence[str] | None = None,
        execution_role_arn: str | None = None,
        profile_name: str | None = None,
        endpoint_url: str | None = None,
        client: Any | None = None,
        endpoint_transport_factory: Callable[[], LambdaMicroVMEndpointTransport] | None = None,
        bind_host: str = "0.0.0.0",
        loop: asyncio.AbstractEventLoop | None = None,
        proxy_server_factory: ProxyServerFactory = DEFAULT_PROXY_SERVER_FACTORY,
        preflight_timeout_s: int = DEFAULT_PREFLIGHT_TIMEOUT_SECONDS,
        probe_metadata: bool = True,
        runner_options: Mapping[str, Any] | None = None,
    ) -> None:
        if not region_name.strip():
            raise ValueError("region_name must be nonblank.")
        if not egress_network_connector_arn.strip():
            raise ValueError("egress_network_connector_arn must be nonblank.")
        if preflight_timeout_s <= 0:
            raise ValueError("preflight_timeout_s must be positive.")
        if type(probe_metadata) is not bool:
            raise TypeError("probe_metadata must be a bool.")
        self.region_name = region_name
        self.egress_network_connector_arn = egress_network_connector_arn
        self.exposure = exposure
        self.ingress_network_connectors = list(
            ingress_network_connectors
            if ingress_network_connectors is not None
            else [_all_ingress_connector_arn(region_name)]
        )
        self.execution_role_arn = execution_role_arn
        self.profile_name = profile_name
        self.endpoint_url = endpoint_url
        self.client = client
        self.endpoint_transport_factory = endpoint_transport_factory
        self.bind_host = bind_host
        self.loop = loop
        self.proxy_server_factory = proxy_server_factory
        self.preflight_timeout_s = preflight_timeout_s
        self.probe_metadata = probe_metadata
        self.runner_options = dict(runner_options or {})
        self._runner_session_ids: weakref.WeakKeyDictionary[Runner, str] = (
            weakref.WeakKeyDictionary()
        )
        reserved = {
            "region_name",
            "profile_name",
            "endpoint_url",
            "client",
            "ingress_network_connectors",
            "egress_network_connectors",
            "execution_role_arn",
            "close_action",
            "endpoint_transport",
            "env_overlay",
        }
        overlap = reserved.intersection(self.runner_options)
        if overlap:
            raise ValueError(
                "runner_options cannot override adapter-owned options: "
                + ", ".join(sorted(overlap))
            )

    async def prepare(
        self,
        *,
        session_id: str,
        grants: Sequence[VirtualCredentialGrant],
        broker: TransparentEgressBroker,
    ) -> EgressBinding:
        return await prepare_exposed_proxy_binding(
            runner_kind=self.runner_kind,
            broker=broker,
            grants=grants,
            exposure=self.exposure,
            bind_host=self.bind_host,
            loop=self.loop,
            proxy_server_factory=self.proxy_server_factory,
        )

    async def create_runner(self, request: VirtualEgressRunnerRequest) -> Runner:
        if request.runner_kind != self.runner_kind:
            raise UnsupportedEgressError(
                f"Lambda MicroVM adapter cannot create runner kind {request.runner_kind!r}."
            )
        endpoint = request.binding.proxy_endpoint
        if endpoint is None:
            raise UnsupportedEgressError(
                "Lambda MicroVM virtual egress requires an HTTP proxy endpoint."
            )
        try:
            VpcTaskProxyExposure(endpoint.host)
        except ValueError as exc:
            raise UnsupportedEgressError(
                "Lambda MicroVM virtual egress requires a private IPv4 proxy endpoint."
            ) from exc

        common_options: dict[str, Any] = {
            "close_action": "none",
            "env_overlay": dict(request.env_overlay),
            **self.runner_options,
        }
        if self.profile_name is not None:
            common_options["profile_name"] = self.profile_name
        if self.endpoint_url is not None:
            common_options["endpoint_url"] = self.endpoint_url
        if self.client is not None:
            common_options["client"] = self.client
        if self.endpoint_transport_factory is not None:
            common_options["endpoint_transport"] = self.endpoint_transport_factory()

        reconnect = _reconnect_identity(request)
        if reconnect is None:
            created_new = True
            create_options: dict[str, Any] = {
                "region_name": self.region_name,
                "ingress_network_connectors": self.ingress_network_connectors,
                "egress_network_connectors": [self.egress_network_connector_arn],
                **common_options,
            }
            if self.execution_role_arn is not None:
                create_options["execution_role_arn"] = self.execution_role_arn
            runner = await LambdaMicroVMRunner.create(request.image, **create_options)
        else:
            created_new = False
            runner = await LambdaMicroVMRunner.from_existing(
                reconnect["microvm_id"],
                region_name=reconnect["region"],
                **common_options,
            )
            mismatch = _identity_mismatch(runner, reconnect)
            if mismatch is not None:
                await runner.close()
                raise LambdaMicroVMProtocolError(
                    f"Lambda MicroVM reconnect {mismatch} changed from durable metadata."
                )
        if request.session_id is not None:
            self._runner_session_ids[runner] = request.session_id
        try:
            await _install_ca(runner, request)
            await run_enforcement_preflight(
                runner,
                request,
                timeout_s=self.preflight_timeout_s,
                probe_metadata=self.probe_metadata,
            )
            await run_setup_commands(runner, request)
        except BaseException:
            if created_new:
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await runner.terminate()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await runner.close()
            raise
        return runner

    def reconnect_metadata(self, runner: Runner) -> dict[str, Any]:
        if not isinstance(runner, LambdaMicroVMRunner):
            raise TypeError("Lambda MicroVM adapter received a different runner type.")
        metadata = {
            "microvm_id": runner.microvm_id,
            "endpoint": runner.endpoint,
            "region": runner.region_name or self.region_name,
            "image_identifier": runner.image_identifier,
            "image_version": runner.image_version,
        }
        session_id = self._runner_session_ids.get(runner)
        if session_id is not None:
            metadata["session_id"] = session_id
        return metadata

    async def finalize_runner(self, runner: Runner, *, outcome: str | None) -> None:
        if not isinstance(runner, LambdaMicroVMRunner):
            raise TypeError("Lambda MicroVM adapter received a different runner type.")
        if outcome == "interrupted":
            await runner.suspend()
        else:
            await runner.terminate()
        await runner.close()


async def _install_ca(
    runner: LambdaMicroVMRunner,
    request: VirtualEgressRunnerRequest,
) -> None:
    certificate = request.binding.ca_cert_pem
    if not certificate:
        raise UnsupportedEgressError(
            "Lambda MicroVM egress binding did not provide a session CA certificate."
        )
    script = (
        "import os, pathlib, sys; "
        "path = pathlib.Path(sys.argv[1]); "
        "path.parent.mkdir(parents=True, exist_ok=True); "
        "path.write_bytes(sys.stdin.buffer.read()); "
        "os.chmod(path, 0o644)"
    )
    result = await runner.exec(
        ExecCommand.process("python3", "-c", script, request.guest_ca_path),
        stdin=certificate.decode("utf-8"),
        timeout_s=30,
    )
    if result.exit_code != 0 or result.timed_out:
        detail = (result.stderr or result.stdout).strip()[:500]
        raise UnsupportedEgressError(
            f"Lambda MicroVM failed to install its virtual-egress CA: {detail}"
        )


def _all_ingress_connector_arn(region_name: str) -> str:
    return f"arn:aws:lambda:{region_name}:aws:network-connector:aws-network-connector:ALL_INGRESS"


def _reconnect_identity(request: VirtualEgressRunnerRequest) -> _ReconnectIdentity | None:
    # Fork checkpoints can inherit the parent's metadata. Only metadata stamped
    # for this session may reattach a child; after interruption, the child's own
    # record has that stamp and can safely resume.
    if not request.reconnect_metadata:
        return None
    metadata = request.reconnect_metadata
    owner = metadata.get("session_id")
    if request.parent_session_id is not None and owner != request.session_id:
        return None
    if owner is not None and request.session_id is not None and owner != request.session_id:
        raise ValueError("Lambda MicroVM reconnect metadata belongs to another session.")
    return {
        "microvm_id": _required_reconnect_string(metadata, "microvm_id"),
        "endpoint": _required_reconnect_string(metadata, "endpoint"),
        "region": _required_reconnect_string(metadata, "region"),
        "image_identifier": _required_reconnect_string(metadata, "image_identifier"),
        "image_version": _optional_reconnect_string(metadata, "image_version"),
    }


def _required_reconnect_string(metadata: Mapping[str, Any], key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Lambda MicroVM reconnect metadata requires nonblank {key}.")
    return value.strip()


def _optional_reconnect_string(metadata: Mapping[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Lambda MicroVM reconnect metadata {key} must be nonblank or null.")
    return value.strip()


def _identity_mismatch(
    runner: LambdaMicroVMRunner,
    reconnect: _ReconnectIdentity,
) -> str | None:
    if runner.endpoint != reconnect["endpoint"]:
        return "endpoint"
    if runner.image_identifier != reconnect["image_identifier"]:
        return "image_identifier"
    expected_version = reconnect["image_version"]
    if expected_version is not None and runner.image_version != expected_version:
        return "image_version"
    return None
