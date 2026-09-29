from __future__ import annotations

import asyncio
import contextlib
import email.utils
import time
import weakref
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime
from hashlib import sha256
from typing import Any, Literal, TypedDict

from cayu.egress._remote_adapter import (
    DEFAULT_PROXY_SERVER_FACTORY,
    ProxyServerFactory,
    prepare_exposed_proxy_binding,
    run_enforcement_preflight,
    run_setup_commands,
)
from cayu.egress.adapter import (
    EgressBinding,
    RunnerFinalizationResult,
    SandboxEgressAdapter,
    VirtualEgressAllocationPreparation,
    VirtualEgressAllocationReap,
    VirtualEgressRunnerRequest,
    _virtual_egress_execution_capability_evidence,
)
from cayu.egress.broker import TransparentEgressBroker
from cayu.egress.capabilities import EgressCapabilityClaim, EgressCapabilityEvidence
from cayu.egress.errors import (
    InvalidEgressReconnectMetadataError,
    UnsupportedEgressCapabilityError,
    UnsupportedEgressError,
)
from cayu.egress.grants import VirtualCredentialGrant
from cayu.egress.proxy_exposure import ProxyExposure, VpcTaskProxyExposure
from cayu.environments.admission import ExecutionCapabilityEvidence
from cayu.runners import (
    ExecCommand,
    LambdaMicroVMProtocolError,
    LambdaMicroVMRunner,
    Runner,
)
from cayu.runners.aws_lambda_microvm import (
    LAMBDA_SIGV4_REQUEST_VALIDITY_SECONDS,
    LambdaMicroVMEndpointTransport,
    LambdaMicroVMError,
    _control_client,
    lambda_microvm_run_options,
    read_microvm_state,
    run_microvm_with_client_token,
    terminate_microvm_confirmed,
)
from cayu.workspaces.revisions import (
    WorkspaceWriterIsolationEvidence,
    WorkspaceWriterIsolationStatus,
)

DEFAULT_PREFLIGHT_TIMEOUT_SECONDS = 30
#: Longest time after intent preparation that Cayu submits or replays one
#: allocation's RunMicrovm client token. It must stay below AWS's token
#: retention, which is undocumented; a replay after retention could create a
#: second MicroVM. Live replays still returned the original MicroVM after more
#: than one hour, the configurable maximum.
DEFAULT_CLIENT_TOKEN_REPLAY_WINDOW_SECONDS = 900
#: AWS's maximum MicroVM lifetime; recoverable allocations always pin a bound.
LAMBDA_MICROVM_MAXIMUM_DURATION_SECONDS = 28_800
#: Allowance for AWS's maximum-duration enforcement to finish terminating.
_LIFETIME_EXPIRY_MARGIN_SECONDS = 300
_REAP_TERMINATION_TIMEOUT_SECONDS = 8.0
_ALLOCATION_METADATA_VERSION = 1
_ALLOCATION_METADATA_KEYS = frozenset(
    {
        "version",
        "image_arn",
        "image_version",
        "maximum_duration_s",
        "prepared_at_s",
        "replay_window_s",
    }
)
_RECONNECT_IDENTITY_KEYS = frozenset(
    {
        "microvm_id",
        "endpoint",
        "region",
        "image_identifier",
        "image_version",
        "session_id",
        "environment_name",
    }
)
_METADATA_ISOLATION_UNVERIFIED_REASON = "guest_process_boundary_unverified"
_wall_clock: Callable[[], float] = time.time

LambdaMicroVMMetadataIsolationMode = Literal["required", "unverified"]


class LambdaMicroVMAllocationRecoveryError(LambdaMicroVMError):
    """A recoverable allocation cannot be adopted or disposed without guessing.

    The durable intent remains owned and retryable. Cayu never answers this
    error by allocating a replacement MicroVM.
    """


class _AllocationMetadata(TypedDict):
    version: int
    image_arn: str
    image_version: str
    maximum_duration_s: int
    prepared_at_s: int
    replay_window_s: int


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
    metadata paths remain blocked. The first-party image puts ordinary commands
    in a route-less network namespace with only a narrow relay to the private
    Cayu proxy, while the root sidecar retains managed ingress. Custom images may
    explicitly set ``metadata_isolation="unverified"``, but that mode never
    reports the capability as verified.
    """

    runner_kind = "lambda-microvm"
    process_external_allocation = True
    allocation_provider = "aws-lambda-microvm"
    allocation_adapter_generation = "virtual-egress-client-token-v1"
    supports_reconnect = True
    supports_allocation_fingerprint = True

    def execution_admission_evidence_for(self, requirements):
        return self._execution_admission_executable_declaration(requirements)

    def execution_capability_evidence(
        self,
        runner: Runner | None = None,
    ) -> ExecutionCapabilityEvidence:
        if runner is not None and not isinstance(runner, LambdaMicroVMRunner):
            raise TypeError("Lambda MicroVM adapter received a different runner type.")
        privilege_posture: Literal["available", "live_verified"] = (
            "live_verified"
            if runner is not None and runner in self._runner_privilege_verified
            else "available"
        )
        return _virtual_egress_execution_capability_evidence(
            runner_kind=self.runner_kind,
            runner_ready=runner is not None,
            preflight_observed_at=(
                self._runner_preflight_observations.get(runner) if runner is not None else None
            ),
            untrusted_isolation=True,
            credential_non_possession_posture=(
                "unverified" if self.metadata_isolation == "unverified" else "available"
            ),
            guest_privilege=privilege_posture,
            unprivileged_guest=privilege_posture,
            host_filesystem_isolation=True,
            reconnect=self.supports_reconnect,
            network_unverified=self.metadata_isolation == "unverified",
            cancellation_confirmed=(
                getattr(runner, "cancellation_cleanup", None)
                if runner is not None
                else self.runner_options.get("cancellation_cleanup", "command")
            )
            != "none",
        )

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
        metadata_isolation: LambdaMicroVMMetadataIsolationMode = "required",
        runner_options: Mapping[str, Any] | None = None,
        client_token_replay_window_s: int = DEFAULT_CLIENT_TOKEN_REPLAY_WINDOW_SECONDS,
    ) -> None:
        if (
            type(client_token_replay_window_s) is not int
            or not 0 < client_token_replay_window_s <= 3_600
        ):
            raise ValueError(
                "client_token_replay_window_s must be an integer between 1 and 3600 seconds."
            )
        if not region_name.strip():
            raise ValueError("region_name must be nonblank.")
        if not egress_network_connector_arn.strip():
            raise ValueError("egress_network_connector_arn must be nonblank.")
        if preflight_timeout_s <= 0:
            raise ValueError("preflight_timeout_s must be positive.")
        if type(metadata_isolation) is not str:
            raise TypeError("metadata_isolation must be 'required' or 'unverified'.")
        if metadata_isolation not in {"required", "unverified"}:
            raise ValueError("metadata_isolation must be 'required' or 'unverified'.")
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
        self.metadata_isolation = metadata_isolation
        self.runner_options = dict(runner_options or {})
        self.client_token_replay_window_s = client_token_replay_window_s
        self._runner_session_ids: weakref.WeakKeyDictionary[Runner, str] = (
            weakref.WeakKeyDictionary()
        )
        self._runner_capabilities: weakref.WeakKeyDictionary[
            Runner,
            EgressCapabilityEvidence,
        ] = weakref.WeakKeyDictionary()
        self._runner_preflight_observations: weakref.WeakKeyDictionary[
            Runner,
            datetime,
        ] = weakref.WeakKeyDictionary()
        self._runner_environment_names: weakref.WeakKeyDictionary[Runner, str] = (
            weakref.WeakKeyDictionary()
        )
        self._runner_privilege_verified: weakref.WeakSet[Runner] = weakref.WeakSet()
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
            "client_token",
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
            session_id=session_id,
            broker=broker,
            grants=grants,
            exposure=self.exposure,
            bind_host=self.bind_host,
            loop=self.loop,
            proxy_server_factory=self.proxy_server_factory,
        )

    async def prepare_reconnect(
        self,
        *,
        session_id: str,
        environment_name: str,
        grants: Sequence[VirtualCredentialGrant],
        broker: TransparentEgressBroker,
        reconnect_metadata: Mapping[str, Any],
    ) -> EgressBinding:
        """Build fresh proxy, CA, and grant authority for the same MicroVM.

        Only durable non-secret identity and trusted adapter configuration are
        inputs. Earlier virtual credentials, proxy endpoints, and CA material
        belonged to a broker that was revoked at interruption or died with its
        worker, so none of it is honored by the new binding.
        """

        identity = self.validate_reconnect_metadata(reconnect_metadata)
        if identity["session_id"] != session_id:
            raise InvalidEgressReconnectMetadataError(
                "Lambda MicroVM reconnect identity belongs to a different session."
            )
        if identity["environment_name"] != environment_name:
            raise InvalidEgressReconnectMetadataError(
                "Lambda MicroVM reconnect identity belongs to a different environment."
            )
        if identity["region"] != self.region_name:
            raise InvalidEgressReconnectMetadataError(
                "Lambda MicroVM reconnect identity names a different region."
            )
        return await self.prepare(session_id=session_id, grants=grants, broker=broker)

    async def create_runner(self, request: VirtualEgressRunnerRequest) -> Runner:
        common_options = self._common_runner_options(request)
        reconnect = _reconnect_identity(request)
        if reconnect is None:
            create_options: dict[str, Any] = {
                "region_name": self.region_name,
                "ingress_network_connectors": self.ingress_network_connectors,
                "egress_network_connectors": [self.egress_network_connector_arn],
                **common_options,
            }
            if self.execution_role_arn is not None:
                create_options["execution_role_arn"] = self.execution_role_arn
            return await self._admit_runner(
                request,
                LambdaMicroVMRunner.create(request.image, **create_options),
                owns_allocation=True,
            )
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
        return await self._admit_runner(request, _ready(runner), owns_allocation=False)

    async def prepare_allocation_metadata(
        self,
        request: VirtualEgressAllocationPreparation,
    ) -> dict[str, Any]:
        """Pin the exact image, lifetime bound, and replay window before dispatch.

        A replacement pins its predecessor's exact image version so the session
        continues on a compatible execution identity even if a newer version
        of the configured image has since become active.
        """

        image_arn, image_version = await self._resolve_exact_image(request.image)
        predecessor = request.predecessor_identity
        if predecessor is not None:
            predecessor = self.validate_reconnect_metadata(predecessor)
            if predecessor["image_identifier"] != image_arn:
                raise UnsupportedEgressError(
                    "Lambda MicroVM replacement image differs from its predecessor's image."
                )
            if predecessor["region"] != self.region_name:
                raise UnsupportedEgressError(
                    "Lambda MicroVM replacement region differs from its predecessor's region."
                )
            image_version = predecessor["image_version"]
        maximum_duration = self.runner_options.get(
            "maximum_duration_in_seconds", LAMBDA_MICROVM_MAXIMUM_DURATION_SECONDS
        )
        if (
            type(maximum_duration) is not int
            or not 0 < maximum_duration <= LAMBDA_MICROVM_MAXIMUM_DURATION_SECONDS
        ):
            raise ValueError(
                "Recoverable Lambda MicroVM allocation requires maximum_duration_in_seconds "
                f"between 1 and {LAMBDA_MICROVM_MAXIMUM_DURATION_SECONDS}."
            )
        metadata: _AllocationMetadata = {
            "version": _ALLOCATION_METADATA_VERSION,
            "image_arn": image_arn,
            "image_version": image_version,
            "maximum_duration_s": maximum_duration,
            # Floor so every elapsed-time decision errs toward "later".
            "prepared_at_s": int(_wall_clock()),
            "replay_window_s": self.client_token_replay_window_s,
        }
        return dict(metadata)

    async def create_or_recover_runner(
        self,
        request: VirtualEgressRunnerRequest,
        *,
        allow_create: bool,
    ) -> Runner:
        """Submit or replay the intent's client token; never allocate a replacement.

        The first dispatch and every recovery submit identical RunMicrovm
        parameters under a token derived from the durable allocation id, so AWS
        returns the original MicroVM instead of creating another. Submissions
        stop once the pinned replay window has elapsed because AWS token
        retention is bounded.
        """

        if type(allow_create) is not bool:
            raise TypeError("Lambda MicroVM recoverable creation requires an exact decision.")
        if request.runner_kind != self.runner_kind:
            raise UnsupportedEgressError(
                f"Lambda MicroVM adapter cannot create runner kind {request.runner_kind!r}."
            )
        allocation_id = request.allocation_id
        if type(allocation_id) is not str:
            raise TypeError("Lambda MicroVM recoverable creation requires a durable allocation id.")
        if _reconnect_identity(request) is not None:
            raise LambdaMicroVMAllocationRecoveryError(
                "Recoverable Lambda MicroVM creation cannot consume reconnect metadata."
            )
        metadata = _validated_allocation_metadata(request.allocation_metadata)
        self._require_replay_window_open(metadata, action="create" if allow_create else "recover")
        common_options = self._common_runner_options(request)
        common_options.pop("image_version", None)
        common_options.pop("maximum_duration_in_seconds", None)
        create_options: dict[str, Any] = {
            "region_name": self.region_name,
            "ingress_network_connectors": self.ingress_network_connectors,
            "egress_network_connectors": [self.egress_network_connector_arn],
            "image_version": metadata["image_version"],
            "maximum_duration_in_seconds": metadata["maximum_duration_s"],
            "client_token": _client_token(self.allocation_adapter_generation, allocation_id),
            "client_token_not_after": _replay_deadline(metadata),
            **common_options,
        }
        if self.execution_role_arn is not None:
            create_options["execution_role_arn"] = self.execution_role_arn
        return await self._admit_runner(
            request,
            LambdaMicroVMRunner.create(metadata["image_arn"], **create_options),
            owns_allocation=True,
        )

    async def reap_allocation(self, request: VirtualEgressAllocationReap) -> None:
        """Terminate the token-owned MicroVM, or prove the platform already did.

        Within the pinned replay window the client token resolves the exact
        allocation (creating it only if the original submission never arrived)
        and the MicroVM is terminated with positive readback. No submission for
        the token is sent after the replay deadline, and AWS rejects a SigV4
        request more than 15 minutes after its signing time, so once AWS's own
        clock passes the lifetime bound its maximum-duration enforcement is the
        termination proof. Between those bounds, or when AWS's time cannot be
        read, ownership stays pending.
        """

        if type(request) is not VirtualEgressAllocationReap:
            raise TypeError("Lambda MicroVM cleanup requires VirtualEgressAllocationReap.")
        metadata = _validated_allocation_metadata(request.allocation_metadata)
        identity = request.acknowledged_identity or {}
        microvm_id = identity.get("microvm_id")
        now = _wall_clock()
        control_client, owns_client = _control_client(
            client=self.client,
            region_name=self.region_name,
            profile_name=self.profile_name,
            endpoint_url=self.endpoint_url,
        )
        try:
            if microvm_id is None:
                if now <= _replay_deadline(metadata):
                    response = await run_microvm_with_client_token(
                        control_client,
                        self._run_options(metadata),
                        client_token=_client_token(
                            self.allocation_adapter_generation, request.allocation_id
                        ),
                        submit_not_after=_replay_deadline(metadata),
                    )
                    microvm_id = response.get("microvmId")
                    if type(microvm_id) is not str or not microvm_id.strip():
                        raise LambdaMicroVMProtocolError("run_microvm response omitted microvmId.")
                else:
                    # The local clock only decides whether asking is worthwhile;
                    # AWS's clock is the proof.
                    server_now = (
                        await _aws_server_time(control_client, metadata["image_arn"])
                        if now >= _lifetime_deadline(metadata)
                        else None
                    )
                    if server_now is not None and server_now >= _lifetime_deadline(metadata):
                        return
                    raise LambdaMicroVMAllocationRecoveryError(
                        "Lambda MicroVM client-token replay window has elapsed and AWS has not "
                        "confirmed that the allocation's maximum lifetime has; cleanup remains "
                        f"pending until {_lifetime_deadline(metadata)} (epoch seconds, AWS time)."
                    )
            if type(microvm_id) is not str:
                raise LambdaMicroVMProtocolError("Acknowledged Lambda MicroVM id is invalid.")
            await terminate_microvm_confirmed(
                control_client,
                microvm_id,
                timeout_s=_REAP_TERMINATION_TIMEOUT_SECONDS,
            )
        finally:
            if owns_client:
                close = getattr(control_client, "close", None)
                if callable(close):
                    await asyncio.to_thread(close)

    async def _resolve_exact_image(self, image: str) -> tuple[str, str]:
        configured_version = self.runner_options.get("image_version")
        if configured_version is not None and (
            type(configured_version) is not str or not configured_version.strip()
        ):
            raise ValueError("Lambda MicroVM image_version must be a nonblank string.")
        if image.startswith("arn:") and configured_version is not None:
            return image, configured_version
        control_client, owns_client = _control_client(
            client=self.client,
            region_name=self.region_name,
            profile_name=self.profile_name,
            endpoint_url=self.endpoint_url,
        )
        try:
            response = await asyncio.to_thread(
                control_client.get_microvm_image, imageIdentifier=image
            )
        finally:
            if owns_client:
                close = getattr(control_client, "close", None)
                if callable(close):
                    await asyncio.to_thread(close)
        if not isinstance(response, Mapping):
            raise LambdaMicroVMProtocolError("get_microvm_image response must be an object.")
        image_arn = response.get("imageArn")
        version = configured_version or response.get("latestActiveImageVersion")
        if type(image_arn) is not str or not image_arn.startswith("arn:"):
            raise LambdaMicroVMProtocolError("get_microvm_image omitted a valid imageArn.")
        if type(version) is not str or not version.strip():
            raise UnsupportedEgressError(
                "Lambda MicroVM image has no active version to pin for recoverable allocation."
            )
        return image_arn, version

    def _run_options(self, metadata: _AllocationMetadata) -> dict[str, Any]:
        options = self.runner_options
        return lambda_microvm_run_options(
            metadata["image_arn"],
            image_version=metadata["image_version"],
            execution_role_arn=self.execution_role_arn,
            ingress_network_connectors=self.ingress_network_connectors,
            egress_network_connectors=[self.egress_network_connector_arn],
            idle_policy=options.get("idle_policy"),
            maximum_duration_in_seconds=metadata["maximum_duration_s"],
            run_hook_payload=options.get("run_hook_payload"),
        )

    def _require_replay_window_open(self, metadata: _AllocationMetadata, *, action: str) -> None:
        if _wall_clock() > _replay_deadline(metadata):
            raise LambdaMicroVMAllocationRecoveryError(
                f"Cannot {action} Lambda MicroVM allocation: its client-token replay window "
                "has elapsed, so another submission could create a duplicate MicroVM."
            )

    def _common_runner_options(self, request: VirtualEgressRunnerRequest) -> dict[str, Any]:
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
        return common_options

    async def _admit_runner(
        self,
        request: VirtualEgressRunnerRequest,
        allocation: Awaitable[LambdaMicroVMRunner],
        *,
        owns_allocation: bool,
    ) -> Runner:
        runner = await allocation
        if request.session_id is not None:
            self._runner_session_ids[runner] = request.session_id
        if request.environment_name is not None:
            self._runner_environment_names[runner] = request.environment_name
        try:
            await _install_ca(runner, request)
            await run_setup_commands(runner, request)
            preflight_observed_at = await run_enforcement_preflight(
                runner,
                request,
                timeout_s=self.preflight_timeout_s,
                probe_metadata=self.metadata_isolation == "required",
                metadata_isolation_reason=(
                    "guest-initiated requests reached the link-local metadata endpoint; "
                    "the agent network-namespace boundary was absent or ineffective"
                ),
                metadata_isolation_remediation=(
                    "use the first-party image with its route-less agent network namespace and "
                    "fixed-port relay, or an equivalent topology that denies guest link-local "
                    "metadata while preserving managed ingress; otherwise explicitly select "
                    "metadata_isolation='unverified' and do not treat the environment as fully "
                    "verified"
                ),
            )
            if self.metadata_isolation == "required":
                await _verify_agent_privilege_boundary(runner, timeout_s=self.preflight_timeout_s)
                self._runner_privilege_verified.add(runner)
            self._runner_preflight_observations[runner] = preflight_observed_at
        except BaseException:
            if owns_allocation:
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await runner.terminate()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await runner.close()
            raise
        metadata_claim = (
            EgressCapabilityClaim(
                capability="metadata_isolation",
                state="verified",
                proof_source="agent_preflight",
                observation="denied",
            )
            if self.metadata_isolation == "required"
            else EgressCapabilityClaim(
                capability="metadata_isolation",
                state="unverified",
                proof_source="operator_opt_out",
                observation="not_probed",
                reason_code=_METADATA_ISOLATION_UNVERIFIED_REASON,
                remediation_code="supply_enforceable_guest_boundary",
            )
        )
        self._runner_capabilities[runner] = EgressCapabilityEvidence(
            adapter=self.runner_kind,
            claims=(
                EgressCapabilityClaim(
                    capability="proxy_reachability",
                    state="verified",
                    proof_source="agent_preflight",
                    observation="reachable",
                ),
                EgressCapabilityClaim(
                    capability="direct_public_egress",
                    state="verified",
                    proof_source="agent_preflight",
                    observation="denied",
                ),
                metadata_claim,
            ),
        )
        return runner

    def capability_evidence(self, runner: Runner) -> EgressCapabilityEvidence:
        if not isinstance(runner, LambdaMicroVMRunner):
            raise TypeError("Lambda MicroVM adapter received a different runner type.")
        capabilities = self._runner_capabilities.get(runner)
        if capabilities is None:
            raise ValueError("Lambda MicroVM runner has not passed its egress preflight.")
        return capabilities

    def configuration_metadata(self) -> dict[str, str]:
        """Describe configured metadata isolation without claiming runtime proof."""
        if self.metadata_isolation == "required":
            return {"metadata_isolation_mode": "required"}
        return {
            "metadata_isolation_mode": "unverified",
            "metadata_isolation_reason": _METADATA_ISOLATION_UNVERIFIED_REASON,
        }

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
        environment_name = self._runner_environment_names.get(runner)
        if environment_name is not None:
            metadata["environment_name"] = environment_name
        return metadata

    def validate_reconnect_metadata(
        self,
        reconnect_metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Allowlist the exact non-secret identity of one owned MicroVM."""

        if not isinstance(reconnect_metadata, Mapping):
            raise InvalidEgressReconnectMetadataError(
                "Lambda MicroVM reconnect identity must be an object."
            )
        if set(reconnect_metadata) != _RECONNECT_IDENTITY_KEYS:
            raise InvalidEgressReconnectMetadataError(
                "Lambda MicroVM reconnect identity has an invalid schema."
            )
        identity: dict[str, Any] = {}
        for key in sorted(_RECONNECT_IDENTITY_KEYS):
            value = reconnect_metadata[key]
            if type(value) is not str or not value.strip() or value != value.strip():
                raise InvalidEgressReconnectMetadataError(
                    f"Lambda MicroVM reconnect identity requires a clean {key}."
                )
            identity[key] = value
        if not identity["image_identifier"].startswith("arn:"):
            raise InvalidEgressReconnectMetadataError(
                "Lambda MicroVM reconnect identity must name an image ARN."
            )
        return identity

    def observe_writer_isolation(self, runner: Runner) -> WorkspaceWriterIsolationEvidence:
        """Exclusive while this runner holds the MicroVM's current owner claim.

        The sidecar rejects or cancels every other owner's commands, so only
        this runner can write the MicroVM filesystem while its claim is current.
        """

        if not isinstance(runner, LambdaMicroVMRunner):
            raise TypeError("Lambda MicroVM adapter received a different runner type.")
        generation = runner.owner_fence_generation
        if generation is None:
            return WorkspaceWriterIsolationEvidence()
        return WorkspaceWriterIsolationEvidence(
            status=WorkspaceWriterIsolationStatus.EXCLUSIVE,
            mechanism="lambda-microvm-owner-fence",
            generation=generation,
            detail_code=None,
        )

    async def egress_environment_fingerprint(self, runner: Runner) -> str:
        if not isinstance(runner, LambdaMicroVMRunner):
            raise TypeError("Lambda MicroVM egress identity requires a LambdaMicroVMRunner.")
        return _lambda_environment_fingerprint(runner.microvm_id)

    async def is_allocation_disposed(self, reconnect_metadata: Mapping[str, Any]) -> bool:
        """Prove exact disposal from the control plane without mutating anything.

        MicroVM identifiers are never reused, so a not-found response for an
        identity Cayu durably acknowledged means AWS has retired the record.
        Every other state, and any read failure, preserves reconnect.
        """

        identity = self.validate_reconnect_metadata(reconnect_metadata)
        control_client, owns_client = _control_client(
            client=self.client,
            region_name=identity["region"],
            profile_name=self.profile_name,
            endpoint_url=self.endpoint_url,
        )
        try:
            state = await read_microvm_state(control_client, identity["microvm_id"])
        finally:
            if owns_client:
                close = getattr(control_client, "close", None)
                if callable(close):
                    await asyncio.to_thread(close)
        return state in {"TERMINATED", None}

    async def finalize_runner(
        self,
        runner: Runner,
        *,
        outcome: str | None,
    ) -> RunnerFinalizationResult:
        if not isinstance(runner, LambdaMicroVMRunner):
            raise TypeError("Lambda MicroVM adapter received a different runner type.")
        preserve = outcome == "interrupted"
        runner.close_action = "suspend" if preserve else "terminate"
        # close() returns only after the control plane reports SUSPENDED or
        # TERMINATED, which is the quiescence and preservation proof.
        await runner.close()
        return RunnerFinalizationResult(
            workspace_mutations_quiescent=True,
            allocation_preserved=preserve,
        )


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
    result = await runner.exec_system(
        ExecCommand.process("python3", "-c", script, request.guest_ca_path),
        stdin=certificate.decode("utf-8"),
        timeout_s=30,
    )
    if result.exit_code != 0 or result.timed_out:
        detail = (result.stderr or result.stdout).strip()[:500]
        raise UnsupportedEgressError(
            f"Lambda MicroVM failed to install its virtual-egress CA: {detail}"
        )


async def _ready(runner: LambdaMicroVMRunner) -> LambdaMicroVMRunner:
    return runner


# Exit codes are the probe's whole protocol so that no guest-controlled text is
# interpreted by the control plane.
_PRIVILEGE_PROBE_FAILURES = {
    10: "agent commands run with a root user or group id",
    11: "agent commands can gain privileges (no_new_privs is not set)",
    12: "agent commands retain Linux capabilities",
    13: "agent command environment contains AWS credential variables",
    14: "agent commands can read an AWS credentials file",
}
_PRIVILEGE_PROBE_SCRIPT = """
import os, sys
fields = {}
with open('/proc/self/status') as status:
    for line in status:
        key, _, value = line.partition(':')
        fields[key] = value.strip()
ids = [int(part) for part in fields['Uid'].split() + fields['Gid'].split()]
if 0 in ids:
    sys.exit(10)
if fields.get('NoNewPrivs') != '1':
    sys.exit(11)
if any(int(fields.get(name, '0'), 16) for name in ('CapEff', 'CapPrm', 'CapAmb')):
    sys.exit(12)
markers = ('ACCESS_KEY', 'SECRET', 'SESSION_TOKEN', 'CONTAINER_CREDENTIALS', 'WEB_IDENTITY')
if any(key.startswith('AWS_') and any(m in key for m in markers) for key in os.environ):
    sys.exit(13)
for path in ('/root/.aws/credentials', os.path.expanduser('~/.aws/credentials')):
    try:
        open(path).close()
    except OSError:
        continue
    sys.exit(14)
"""


async def _verify_agent_privilege_boundary(runner: Runner, *, timeout_s: int) -> None:
    """Prove the agent command profile is unprivileged and holds no AWS credentials."""

    result = await runner.exec(
        ExecCommand.process("python3", "-c", _PRIVILEGE_PROBE_SCRIPT),
        timeout_s=timeout_s,
    )
    if result.timed_out or result.exit_code != 0:
        reason = _PRIVILEGE_PROBE_FAILURES.get(
            result.exit_code, "the agent privilege probe did not complete"
        )
        raise UnsupportedEgressCapabilityError(
            runner_kind="lambda-microvm",
            capability="guest_privilege_containment",
            reason=reason,
            remediation=(
                "use the first-party image, which runs agent commands as UID/GID 1000 with "
                "no capabilities and no_new_privs, or select metadata_isolation='unverified' "
                "and do not treat the environment as fully verified"
            ),
        )


def _lambda_environment_fingerprint(microvm_id: str) -> str:
    return sha256(f"lambda-microvm\0{microvm_id}".encode()).hexdigest()


def _client_token(adapter_generation: str, allocation_id: str) -> str:
    # Derived rather than stored: the durable intent never carries submission
    # authority, and every recovery worker reconstructs the identical token.
    digest = sha256(
        f"cayu.lambda-microvm.run\0{adapter_generation}\0{allocation_id}".encode()
    ).hexdigest()
    return f"cayu-{digest}"


def _validated_allocation_metadata(metadata: Mapping[str, Any]) -> _AllocationMetadata:
    if not isinstance(metadata, Mapping) or set(metadata) != _ALLOCATION_METADATA_KEYS:
        raise LambdaMicroVMAllocationRecoveryError(
            "Lambda MicroVM allocation intent is missing its pinned provider inputs."
        )
    if metadata["version"] != _ALLOCATION_METADATA_VERSION:
        raise LambdaMicroVMAllocationRecoveryError(
            "Lambda MicroVM allocation intent has an unsupported metadata version."
        )
    for key in ("image_arn", "image_version"):
        value = metadata[key]
        if type(value) is not str or not value.strip():
            raise LambdaMicroVMAllocationRecoveryError(
                f"Lambda MicroVM allocation intent has an invalid {key}."
            )
    if not metadata["image_arn"].startswith("arn:"):
        raise LambdaMicroVMAllocationRecoveryError(
            "Lambda MicroVM allocation intent must pin an image ARN."
        )
    for key, upper in (
        ("maximum_duration_s", LAMBDA_MICROVM_MAXIMUM_DURATION_SECONDS),
        ("replay_window_s", 3_600),
        ("prepared_at_s", None),
    ):
        value = metadata[key]
        if type(value) is not int or value <= 0 or (upper is not None and value > upper):
            raise LambdaMicroVMAllocationRecoveryError(
                f"Lambda MicroVM allocation intent has an invalid {key}."
            )
    return {
        "version": metadata["version"],
        "image_arn": metadata["image_arn"],
        "image_version": metadata["image_version"],
        "maximum_duration_s": metadata["maximum_duration_s"],
        "prepared_at_s": metadata["prepared_at_s"],
        "replay_window_s": metadata["replay_window_s"],
    }


def _replay_deadline(metadata: _AllocationMetadata) -> int:
    return metadata["prepared_at_s"] + metadata["replay_window_s"]


def _lifetime_deadline(metadata: _AllocationMetadata) -> int:
    # No request for the token is signed after the replay deadline, and AWS
    # accepts a SigV4 request for at most 15 minutes after signing, so any
    # token-owned MicroVM started by then; maximum-duration enforcement has
    # terminated it after this instant in AWS's time.
    return (
        _replay_deadline(metadata)
        + LAMBDA_SIGV4_REQUEST_VALIDITY_SECONDS
        + metadata["maximum_duration_s"]
        + _LIFETIME_EXPIRY_MARGIN_SECONDS
    )


async def _aws_server_time(client: Any, image_arn: str) -> float | None:
    """Return AWS's clock from a read response's ``Date`` header, or ``None``.

    An error response from AWS carries the same authoritative header.
    """

    try:
        response: Any = await asyncio.to_thread(client.get_microvm_image, imageIdentifier=image_arn)
    except Exception as exc:
        response = getattr(exc, "response", None)
    metadata = response.get("ResponseMetadata") if isinstance(response, Mapping) else None
    headers = metadata.get("HTTPHeaders") if isinstance(metadata, Mapping) else None
    value = headers.get("date") if isinstance(headers, Mapping) else None
    if type(value) is not str:
        return None
    try:
        parsed = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.timestamp()


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
