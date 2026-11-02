"""AWS Lambda MicroVM environments allocated through Cayu's recoverable-allocation seam.

``LambdaMicroVMEnvironmentFactory`` gives a session (or every session of a
shared allocation scope) one Lambda MicroVM whose disk is the workspace. It
composes ``LambdaMicroVMRunner``, ``RunnerWorkspace`` and
``LambdaMicroVMLifecycleBinding``; each stays usable on its own.

Creation is crash-safe. The factory declares an allocation scope, so Cayu
always calls :meth:`LambdaMicroVMEnvironmentFactory.create_recoverable` with a
durable allocation intent instead of :meth:`create`. Before any provider call
the intent pins the exact image version, a lifetime bound and a client-token
replay window. ``RunMicrovm`` then runs under a client token derived from the
intent's allocation id, so a create whose acknowledgement was lost, or that
two recovery workers repeat at once, resolves to the one MicroVM AWS created
for that token. The intent never holds a credential: the token is recomputed
from the allocation id, and the reconnect metadata is the MicroVM's identity
(id, endpoint, region, image and version) plus the owning session and
environment.

RECONNECT attaches to that same MicroVM with fresh access: the runner mints a
new endpoint auth token and claims the sidecar's owner fence, which rejects
every command and lifecycle call from the runner that owned it before. When the
request carries an ``ownership_epoch`` the claim is epoch-fenced too, so an
owner that lost the allocation can't take it back with a late attach.
Disposal is proven from the control plane's terminal readback: termination
waits for ``TERMINATED``, and :meth:`is_allocation_disposed` reports True only
for ``TERMINATED`` or an identifier AWS no longer knows.

This factory doesn't mediate the MicroVM's network. It has whatever network
the configured connectors give it. For brokered, audited egress with virtual
credentials use ``SandboxCodingEnvironmentFactory`` with
``LambdaMicroVMEgressAdapter``, which shares the same allocation protocol.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping
from hashlib import sha256
from typing import Any

from cayu._validation import canonical_durable_json_bytes
from cayu.environments._finalization_disposal import checkpoint_finalization_disposal
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.environments.bindings import BoundWorkspace, NativeBinding, WorkspaceBinding
from cayu.environments.factory import (
    EnvironmentAllocationContext,
    EnvironmentAllocationScope,
    EnvironmentAllocationState,
    EnvironmentAllocationUnsupportedError,
    EnvironmentFactory,
    EnvironmentFactoryOperation,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
)
from cayu.runners._lambda_microvm_allocation import (
    DEFAULT_CLIENT_TOKEN_REPLAY_WINDOW_SECONDS,
    LAMBDA_MICROVM_MAXIMUM_DURATION_SECONDS,
    LambdaMicroVMAllocationRecoveryError,
    client_token,
    close_control_client,
    pin_allocation_metadata,
    reap_token_owned_microvm,
    replay_deadline,
    require_maximum_duration,
    require_replay_window,
    require_replay_window_open,
    resolve_exact_image,
    validated_allocation_metadata,
)
from cayu.runners.aws_lambda_microvm import (
    DEFAULT_LAMBDA_MICROVM_LIFECYCLE_TIMEOUT_SECONDS,
    DEFAULT_LAMBDA_MICROVM_POLL_INTERVAL_SECONDS,
    LambdaMicroVMEndpointTransport,
    LambdaMicroVMProtocolError,
    LambdaMicroVMRunner,
    _control_client,
    lambda_microvm_run_options,
    read_microvm_state,
    terminate_microvm_confirmed,
)
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.workspaces import RunnerWorkspace
from cayu.workspaces.revisions import (
    WorkspaceWriterIsolationEvidence,
    WorkspaceWriterIsolationStatus,
)

LAMBDA_MICROVM_ALLOCATION_PROVIDER = "aws-lambda-microvm"
LAMBDA_MICROVM_ENVIRONMENT_ADAPTER_GENERATION = "environment-client-token-v1"
_RECONNECT_KEYS = frozenset(
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
_RECOVERY_STATE_KIND = "lambda-microvm-native"
# Options the factory owns: the allocation intent pins the image version and
# lifetime, and the factory chooses the client, transport and close action.
_RESERVED_OPTIONS = frozenset(
    {
        "client",
        "endpoint_transport",
        "close_action",
        "region_name",
        "profile_name",
        "endpoint_url",
        "client_token",
        "client_token_not_after",
        "owner_epoch",
    }
)
# Create options the allocation intent pins or that change RunMicrovm parameters.
_RUN_OPTION_NAMES = frozenset(
    {
        "image_version",
        "execution_role_arn",
        "ingress_network_connectors",
        "egress_network_connectors",
        "idle_policy",
        "maximum_duration_in_seconds",
        "run_hook_payload",
    }
)
_wall_clock: Callable[[], float] = time.time


class LambdaMicroVMLifecycleBinding(WorkspaceBinding):
    """Native binding that maps terminal session outcomes to MicroVM lifecycle actions.

    The MicroVM's disk is the workspace, so binding copies nothing. Finalizing
    with ``interrupted`` suspends the MicroVM, so a resumed session reconnects
    to it; any other outcome terminates it. Both return only after the
    control plane reports ``SUSPENDED`` or ``TERMINATED``.

    With ``durable_disposal=True`` (``LambdaMicroVMEnvironmentFactory`` sets it)
    termination goes through the runtime's durable finalization: a completed
    session's obligation to terminate is recorded before finalize runs, and the
    exact MicroVM is recorded before the terminate call, so recovery finishes
    a teardown a lost worker left through the factory's
    ``recover_finalization_disposal``. That needs a factory to recover through,
    so the default, for an environment registered directly with this binding,
    terminates in finalize only.
    """

    def __init__(self, runner: LambdaMicroVMRunner, *, durable_disposal: bool = False) -> None:
        if not isinstance(runner, LambdaMicroVMRunner):
            raise TypeError("LambdaMicroVMLifecycleBinding requires a LambdaMicroVMRunner")
        if type(durable_disposal) is not bool:
            raise TypeError("durable_disposal must be a boolean.")
        self.runner = runner
        self.durable_disposal = durable_disposal
        self._native = NativeBinding(default_path=runner.default_cwd)
        self._finalized = False
        self._finalize_lock = asyncio.Lock()

    async def bind(
        self,
        workspace,
        runner,
        *,
        session_id: str,
        agent_name: str | None = None,
        environment_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BoundWorkspace:
        return await self._native.bind(
            workspace,
            runner,
            session_id=session_id,
            agent_name=agent_name,
            environment_name=environment_name,
            metadata=metadata,
        )

    async def finalize(
        self,
        bound: BoundWorkspace,
        *,
        outcome: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        await self._native.finalize(bound, outcome=outcome, metadata=metadata)
        async with self._finalize_lock:
            if self._finalized:
                return None
            terminate = outcome != "interrupted"
            if terminate and self.durable_disposal:
                # Hand the exact disposal authority to the runtime before the
                # irreversible call, so a worker that dies from here on leaves
                # recovery something to finish (recover_finalization_disposal).
                await checkpoint_finalization_disposal(
                    lambda_microvm_disposal_state(self.runner.microvm_id)
                )
            self.runner.close_action = "terminate" if terminate else "suspend"
            await self.runner.close()
            self._finalized = True
        return None

    def _completion_requires_successful_finalization(self, bound: BoundWorkspace) -> bool:
        # The binding owns the MicroVM, so a completed session isn't done until
        # it is terminated: use the runtime's durable finalization path, which
        # records the obligation before finalize and lets recovery finish it.
        super()._completion_requires_successful_finalization(bound)
        return self.durable_disposal

    def observe_writer_isolation(self, bound: BoundWorkspace) -> WorkspaceWriterIsolationEvidence:
        """Exclusive while this runner holds the MicroVM's current owner claim."""

        if type(bound) is not BoundWorkspace:
            raise TypeError("Workspace writer-isolation observation requires a BoundWorkspace.")
        generation = self.runner.owner_fence_generation
        if generation is None:
            return WorkspaceWriterIsolationEvidence()
        return WorkspaceWriterIsolationEvidence(
            status=WorkspaceWriterIsolationStatus.EXCLUSIVE,
            mechanism="lambda-microvm-owner-fence",
            generation=generation,
            detail_code=None,
        )

    def _completion_finalization_recovery_state(
        self, bound: BoundWorkspace
    ) -> dict[str, Any] | None:
        if type(bound) is not BoundWorkspace:
            raise TypeError("WorkspaceBinding recovery query requires a BoundWorkspace.")
        # Nothing was copied in, so rebuilding the bound workspace only needs
        # to know which MicroVM and directory it names.
        return {
            "kind": _RECOVERY_STATE_KIND,
            "microvm_id": self.runner.microvm_id,
            "path": self.runner.default_cwd,
        }

    async def _recover_completion_finalization(
        self,
        workspace,
        runner,
        *,
        session_id: str,
        agent_name: str | None,
        environment_name: str | None,
        recovery_state: dict[str, Any],
    ) -> BoundWorkspace:
        if (
            type(recovery_state) is not dict
            or recovery_state.get("kind") != _RECOVERY_STATE_KIND
            or recovery_state.get("microvm_id") != self.runner.microvm_id
            or recovery_state.get("path") != self.runner.default_cwd
        ):
            raise RuntimeError("Lambda MicroVM binding recovery state names another MicroVM.")
        return await self.bind(
            workspace,
            runner,
            session_id=session_id,
            agent_name=agent_name,
            environment_name=environment_name,
        )


class LambdaMicroVMEnvironmentFactory(EnvironmentFactory):
    """Allocate one Lambda MicroVM per allocation, recoverably, and reattach on resume.

    ``create_options`` are passed to ``LambdaMicroVMRunner.create`` (network
    connectors, execution role, idle policy, ``maximum_duration_in_seconds``
    and so on); ``runner_options`` are passed to both ``create`` and
    ``from_existing`` (working directory, timeouts). The intent pins
    ``maximum_duration_in_seconds`` (AWS's 8-hour maximum by default) and the
    image version (``image_version``, or the image's latest active version).

    ``fence_owner_epochs`` passes the request's ``ownership_epoch`` to the
    sidecar's owner claim. The first-party sidecar supports it; set it False
    only for an image whose sidecar doesn't advertise ``owner_epoch``.

    Forks always allocate a fresh MicroVM even when inherited checkpoint state
    contains the parent's reconnect metadata.
    """

    def __init__(
        self,
        *,
        image_identifier: str,
        region_name: str,
        client: Any | None = None,
        profile_name: str | None = None,
        endpoint_url: str | None = None,
        endpoint_transport_factory: Callable[[], LambdaMicroVMEndpointTransport] | None = None,
        create_options: Mapping[str, Any] | None = None,
        runner_options: Mapping[str, Any] | None = None,
        client_token_replay_window_s: int = DEFAULT_CLIENT_TOKEN_REPLAY_WINDOW_SECONDS,
        fence_owner_epochs: bool = True,
    ) -> None:
        if type(image_identifier) is not str or not image_identifier.strip():
            raise ValueError("image_identifier must be nonblank.")
        if type(region_name) is not str or not region_name.strip():
            raise ValueError("region_name must be nonblank.")
        if client is not None and (profile_name is not None or endpoint_url is not None):
            raise ValueError(
                "An injected client cannot be combined with profile_name or endpoint_url."
            )
        if type(fence_owner_epochs) is not bool:
            raise TypeError("fence_owner_epochs must be a boolean.")
        self.image_identifier = image_identifier
        self.region_name = region_name
        self.client = client
        self.profile_name = profile_name
        self.endpoint_url = endpoint_url
        self.endpoint_transport_factory = endpoint_transport_factory
        self.create_options = dict(create_options or {})
        self.runner_options = dict(runner_options or {})
        self.client_token_replay_window_s = require_replay_window(client_token_replay_window_s)
        self.fence_owner_epochs = fence_owner_epochs
        conflicts = sorted(
            _RESERVED_OPTIONS.intersection(self.runner_options)
            | _RESERVED_OPTIONS.intersection(self.create_options)
        )
        if conflicts:
            raise ValueError(f"factory options cannot override: {', '.join(conflicts)}")
        overlap = sorted(self.create_options.keys() & self.runner_options.keys())
        if overlap:
            raise ValueError(f"create_options and runner_options overlap: {', '.join(overlap)}")
        run_options_in_runner = sorted(_RUN_OPTION_NAMES.intersection(self.runner_options))
        if run_options_in_runner:
            raise ValueError(
                "Pass RunMicrovm parameters in create_options, not runner_options: "
                + ", ".join(run_options_in_runner)
            )
        require_maximum_duration(
            self.create_options.get(
                "maximum_duration_in_seconds", LAMBDA_MICROVM_MAXIMUM_DURATION_SECONDS
            )
        )
        configuration = canonical_durable_json_bytes(
            {
                "image_identifier": image_identifier,
                "region_name": region_name,
                "create_options": self.create_options,
                "runner_options": self.runner_options,
                "client_token_replay_window_s": self.client_token_replay_window_s,
            },
            "lambda_microvm_configuration",
        )
        self._profile_identity = ExecutionProfileBehaviorIdentity(
            name="cayu.lambda_microvm_environment",
            behavior_version="1",
            implementation_version=sha256(configuration).hexdigest(),
        )

    @property
    def secret_resolution_scope(self):
        """Static: no session's request changes what an allocation is granted.

        The factory resolves no secrets and returns no vault, credential proxy,
        MCP servers or stores, so every session of a shared allocation scope
        gets exactly what the first one did.
        """

        return "static"

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        """Identity of what every allocation is created from."""

        return self._profile_identity

    @property
    def allocation_scope_identity(self) -> EnvironmentAllocationScope:
        return EnvironmentAllocationScope(
            provider=LAMBDA_MICROVM_ALLOCATION_PROVIDER,
            adapter_generation=LAMBDA_MICROVM_ENVIRONMENT_ADAPTER_GENERATION,
        )

    def allocation_scope(
        self,
        request: EnvironmentFactoryRequest,
    ) -> EnvironmentAllocationScope | None:
        if request.operation is EnvironmentFactoryOperation.RECONNECT:
            return None
        return self.allocation_scope_identity

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        if not isinstance(request, EnvironmentFactoryRequest):
            raise TypeError("LambdaMicroVMEnvironmentFactory requires EnvironmentFactoryRequest")
        if request.operation is not EnvironmentFactoryOperation.RECONNECT:
            raise EnvironmentAllocationUnsupportedError(
                "A Lambda MicroVM is created only through create_recoverable, under a "
                "durable allocation intent; CayuApp does this for every session."
            )
        identity = self._reconnect_identity(request)
        runner = await LambdaMicroVMRunner.from_existing(
            identity["microvm_id"],
            region_name=identity["region"],
            **self._attach_options(request),
            **self.runner_options,
        )
        mismatch = _identity_mismatch(runner, identity)
        if mismatch is not None:
            await runner.close()
            raise LambdaMicroVMProtocolError(
                f"Lambda MicroVM reconnect {mismatch} changed from durable metadata."
            )
        return self._result(request, runner)

    async def create_recoverable(
        self,
        request: EnvironmentFactoryRequest,
        allocation: EnvironmentAllocationContext,
    ) -> EnvironmentFactoryResult:
        """Submit or replay the intent's client token; never allocate a replacement."""

        if not isinstance(request, EnvironmentFactoryRequest):
            raise TypeError("LambdaMicroVMEnvironmentFactory requires EnvironmentFactoryRequest")
        if not isinstance(allocation, EnvironmentAllocationContext):
            raise TypeError("Recoverable Lambda MicroVM creation requires an allocation context.")
        if request.operation is not EnvironmentFactoryOperation.CREATE:
            raise ValueError("Recoverable Lambda MicroVM creation requires a CREATE request.")
        self._require_intent(request, allocation)
        if allocation.state is EnvironmentAllocationState.UNPREPARED:
            metadata = await self._pin_metadata(request)
            await allocation.prepare(
                {"allocation_marker": allocation.intent.allocation_id, "lambda": dict(metadata)}
            )
        metadata = _intent_metadata(allocation)
        allow_create = allocation.state is EnvironmentAllocationState.PREPARED
        require_replay_window_open(
            metadata, action="create" if allow_create else "recover", now=_wall_clock()
        )
        if allow_create:
            await allocation.mark_dispatched()
        if allocation.state not in {
            EnvironmentAllocationState.DISPATCHED,
            EnvironmentAllocationState.ACKNOWLEDGED,
        }:
            raise RuntimeError("Recoverable Lambda MicroVM allocation is not dispatchable.")
        options = {
            key: value for key, value in self.create_options.items() if key != "image_version"
        }
        options["maximum_duration_in_seconds"] = metadata["maximum_duration_s"]
        runner = await LambdaMicroVMRunner.create(
            metadata["image_arn"],
            region_name=self.region_name,
            image_version=metadata["image_version"],
            client_token=client_token(
                allocation.intent.adapter_generation, allocation.intent.allocation_id
            ),
            client_token_not_after=replay_deadline(metadata),
            **self._attach_options(request),
            **options,
            **self.runner_options,
        )
        try:
            reconnect_metadata = self._reconnect_metadata(request, runner)
            if allocation.state is EnvironmentAllocationState.DISPATCHED:
                await allocation.acknowledge(reconnect_metadata)
            elif allocation.acknowledged_reconnect_metadata != reconnect_metadata:
                raise LambdaMicroVMAllocationRecoveryError(
                    "Recovered Lambda MicroVM allocation changed its durable identity."
                )
        except BaseException:
            # The durable intent owns the MicroVM, not this attempt: a retry
            # adopts it by replaying the token, and cleanup terminates it.
            # Terminating here could dispose of an allocation a newer owner of
            # the intent has adopted.
            runner.close_action = "none"
            await asyncio.shield(_close_quietly(runner))
            raise
        return self._result(request, runner)

    async def reap_allocation(
        self,
        request: EnvironmentFactoryRequest,
        allocation: EnvironmentAllocationContext,
    ) -> None:
        """Terminate the intent's MicroVM, or prove AWS already did, without creating guest work."""

        if not isinstance(allocation, EnvironmentAllocationContext):
            raise TypeError("Lambda MicroVM cleanup requires an allocation context.")
        intent = allocation.intent
        if (
            request.operation is not EnvironmentFactoryOperation.CREATE
            or intent.session_id != request.session_id
            or intent.environment_name != request.environment_name
            or intent.scope != self.allocation_scope(request)
        ):
            raise ValueError("Lambda MicroVM cleanup allocation authority changed.")
        if allocation.state is EnvironmentAllocationState.REAPED:
            return
        if allocation.state is EnvironmentAllocationState.UNPREPARED:
            raise RuntimeError("Lambda MicroVM cleanup found an unprepared allocation.")
        if allocation.state is EnvironmentAllocationState.PREPARED:
            # Fencing a never-dispatched intent atomically precludes dispatch.
            await allocation.mark_reaping()
        if not allocation.dispatch_precluded:
            if intent.provider_metadata.get("allocation_marker") != intent.allocation_id:
                raise ValueError("Lambda MicroVM cleanup allocation marker changed.")
            metadata = _intent_metadata(allocation)
            acknowledged = allocation.acknowledged_reconnect_metadata
            microvm_id = None if acknowledged is None else acknowledged.get("microvm_id")
            if not await allocation.mark_reaping():
                # Publication won the fence; the published owner disposes it.
                return
            control_client, owns_client = self._control_client(self.region_name)
            try:
                await reap_token_owned_microvm(
                    control_client,
                    run_options=self._run_options(metadata),
                    token=client_token(intent.adapter_generation, intent.allocation_id),
                    metadata=metadata,
                    acknowledged_microvm_id=microvm_id,
                    now=_wall_clock(),
                )
            finally:
                await close_control_client(control_client, owns_client)
        await allocation.mark_reaped()

    async def is_allocation_disposed(self, request: EnvironmentFactoryRequest) -> bool:
        """Prove disposal from the control plane without mutating anything.

        MicroVM identifiers are never reused, so a not-found response for an
        identity Cayu durably acknowledged means AWS has retired it. Every
        other state, and any read failure, preserves reconnect.
        """

        if not request.reconnect_metadata:
            return False
        identity = _validated_reconnect_metadata(request.reconnect_metadata)
        control_client, owns_client = self._control_client(identity["region"])
        try:
            state = await read_microvm_state(control_client, identity["microvm_id"])
        finally:
            await close_control_client(control_client, owns_client)
        return state in {"TERMINATED", None}

    async def recover_finalization_disposal(
        self,
        request: EnvironmentFactoryRequest,
        state: dict[str, Any],
    ) -> None:
        """Finish a termination the binding recorded but a lost worker may not have done.

        Cleanup only: no reattach, no guest work, no replacement. It validates
        the exact reconnect identity and disposal state, then terminates that
        MicroVM through the control plane with positive readback, which is a
        no-op for one that is already terminated or gone.
        """

        if request.operation is not EnvironmentFactoryOperation.RECONNECT:
            raise ValueError("Lambda MicroVM disposal recovery requires reconnect authority.")
        identity = self._reconnect_identity(request)
        if state != lambda_microvm_disposal_state(identity["microvm_id"]):
            raise ValueError("Lambda MicroVM disposal state conflicts with its exact allocation.")
        control_client, owns_client = self._control_client(identity["region"])
        try:
            await terminate_microvm_confirmed(
                control_client,
                identity["microvm_id"],
                timeout_s=self.runner_options.get(
                    "lifecycle_timeout_s", DEFAULT_LAMBDA_MICROVM_LIFECYCLE_TIMEOUT_SECONDS
                ),
                poll_interval_s=self.runner_options.get(
                    "poll_interval_s", DEFAULT_LAMBDA_MICROVM_POLL_INTERVAL_SECONDS
                ),
            )
        finally:
            await close_control_client(control_client, owns_client)

    # Internals ------------------------------------------------------------

    def _require_intent(
        self, request: EnvironmentFactoryRequest, allocation: EnvironmentAllocationContext
    ) -> None:
        intent = allocation.intent
        if (
            intent.session_id != request.session_id
            or intent.environment_name != request.environment_name
            or intent.scope != self.allocation_scope_identity
        ):
            raise ValueError("Lambda MicroVM allocation intent belongs to another allocation.")

    async def _pin_metadata(self, request: EnvironmentFactoryRequest):
        control_client, owns_client = self._control_client(self.region_name)
        try:
            image_arn, image_version = await resolve_exact_image(
                control_client,
                self.image_identifier,
                configured_version=self.create_options.get("image_version"),
            )
        finally:
            await close_control_client(control_client, owns_client)
        if request.replacement_predecessor:
            # A replacement continues the session on the same execution
            # identity, even if a newer image version has become active.
            predecessor = _validated_reconnect_metadata(request.replacement_predecessor)
            if predecessor["image_identifier"] != image_arn:
                raise ValueError(
                    "Lambda MicroVM replacement image differs from its predecessor's image."
                )
            if predecessor["region"] != self.region_name:
                raise ValueError(
                    "Lambda MicroVM replacement region differs from its predecessor's region."
                )
            image_version = predecessor["image_version"]
        return pin_allocation_metadata(
            image_arn=image_arn,
            image_version=image_version,
            maximum_duration_s=self.create_options.get(
                "maximum_duration_in_seconds", LAMBDA_MICROVM_MAXIMUM_DURATION_SECONDS
            ),
            replay_window_s=self.client_token_replay_window_s,
            now=_wall_clock(),
        )

    def _run_options(self, metadata) -> dict[str, Any]:
        options = self.create_options
        return lambda_microvm_run_options(
            metadata["image_arn"],
            image_version=metadata["image_version"],
            execution_role_arn=options.get("execution_role_arn"),
            ingress_network_connectors=options.get("ingress_network_connectors"),
            egress_network_connectors=options.get("egress_network_connectors"),
            idle_policy=options.get("idle_policy"),
            maximum_duration_in_seconds=metadata["maximum_duration_s"],
            run_hook_payload=options.get("run_hook_payload"),
        )

    def _attach_options(self, request: EnvironmentFactoryRequest) -> dict[str, Any]:
        options: dict[str, Any] = {"close_action": "none"}
        if self.client is not None:
            options["client"] = self.client
        if self.profile_name is not None:
            options["profile_name"] = self.profile_name
        if self.endpoint_url is not None:
            options["endpoint_url"] = self.endpoint_url
        if self.endpoint_transport_factory is not None:
            options["endpoint_transport"] = self.endpoint_transport_factory()
        if self.fence_owner_epochs and request.ownership_epoch is not None:
            options["owner_epoch"] = request.ownership_epoch
        return options

    def _control_client(self, region_name: str) -> tuple[Any, bool]:
        return _control_client(
            client=self.client,
            region_name=region_name,
            profile_name=self.profile_name,
            endpoint_url=self.endpoint_url,
        )

    def _reconnect_identity(self, request: EnvironmentFactoryRequest) -> dict[str, str]:
        if not request.reconnect_metadata:
            raise ValueError("MicroVM reconnect requires durable reconnect metadata")
        identity = _validated_reconnect_metadata(request.reconnect_metadata)
        # A fork's checkpoint can inherit its parent's metadata; only metadata
        # stamped for this session and environment may reattach.
        if identity["session_id"] != request.session_id:
            raise ValueError("Lambda MicroVM reconnect metadata belongs to another session.")
        if identity["environment_name"] != request.environment_name:
            raise ValueError("Lambda MicroVM reconnect metadata belongs to another environment.")
        return identity

    def _reconnect_metadata(
        self, request: EnvironmentFactoryRequest, runner: LambdaMicroVMRunner
    ) -> dict[str, str]:
        return _validated_reconnect_metadata(
            {
                "microvm_id": runner.microvm_id,
                "endpoint": runner.endpoint,
                "region": runner.region_name or self.region_name,
                "image_identifier": runner.image_identifier or self.image_identifier,
                "image_version": runner.image_version,
                "session_id": request.session_id,
                "environment_name": request.environment_name,
            }
        )

    def _result(
        self,
        request: EnvironmentFactoryRequest,
        runner: LambdaMicroVMRunner,
    ) -> EnvironmentFactoryResult:
        reconnect_metadata = self._reconnect_metadata(request, runner)
        workspace = RunnerWorkspace(
            runner,
            workspace_id=f"lambda-microvm:{runner.microvm_id}:{runner.default_cwd}",
        )

        async def release(action: EnvironmentFactoryReleaseAction) -> None:
            # Before a binding adopts the MicroVM: discard terminates it with
            # positive readback; preserve only lets go of this process's handle.
            runner.close_action = (
                "terminate" if action is EnvironmentFactoryReleaseAction.DISCARD else "none"
            )
            await runner.close()

        return EnvironmentFactoryResult(
            environment=Environment(
                EnvironmentSpec(
                    name=request.environment_name,
                    metadata={"kind": "lambda-microvm", "region": reconnect_metadata["region"]},
                    execution_profile_identity=self.execution_profile_identity,
                ),
                workspace=workspace,
                runner=runner,
                binding=LambdaMicroVMLifecycleBinding(runner, durable_disposal=True),
            ),
            metadata={"kind": "lambda-microvm", "microvm_id": runner.microvm_id},
            reconnect_metadata=reconnect_metadata,
            release=release,
        )


def lambda_microvm_disposal_state(microvm_id: str) -> dict[str, Any]:
    """The disposal authority the binding records before terminating ``microvm_id``."""

    return {"version": 1, "kind": "lambda_microvm_disposal", "microvm_id": microvm_id}


def _intent_metadata(allocation: EnvironmentAllocationContext):
    return validated_allocation_metadata(allocation.intent.provider_metadata.get("lambda") or {})


def _validated_reconnect_metadata(metadata: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(metadata, Mapping) or set(metadata) != _RECONNECT_KEYS:
        raise ValueError("Lambda MicroVM reconnect metadata has an invalid schema.")
    identity: dict[str, str] = {}
    for key in sorted(_RECONNECT_KEYS):
        value = metadata[key]
        if type(value) is not str or not value.strip() or value != value.strip():
            raise ValueError(f"Lambda MicroVM reconnect metadata requires a clean {key}.")
        identity[key] = value
    if not identity["image_identifier"].startswith("arn:"):
        raise ValueError("Lambda MicroVM reconnect metadata must name an image ARN.")
    return identity


def _identity_mismatch(runner: LambdaMicroVMRunner, identity: Mapping[str, str]) -> str | None:
    if runner.endpoint != identity["endpoint"]:
        return "endpoint"
    if runner.image_identifier != identity["image_identifier"]:
        return "image_identifier"
    if runner.image_version != identity["image_version"]:
        return "image_version"
    return None


async def _close_quietly(runner: LambdaMicroVMRunner) -> None:
    try:
        await runner.close()
    except Exception:
        return
