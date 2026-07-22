"""Runtime wiring for virtual egress credentials.

Turns the egress library into a first-class, session-lifecycle-managed mode: a
``VirtualEgressEnvironmentFactory`` mints per-session grants, stands up the
broker + an adapter-enforced runner, and emits audit events; teardown (revoke +
remove runtime network resources + stop proxy) runs from the workspace binding's
``finalize`` hook that the runtime already calls at session end.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import inspect
import os
import re
import secrets
import shutil
import tempfile
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from cayu._task_wait import (
    await_shielded_task_outcome,
    consume_pending_task_cancellation,
)
from cayu._validation import copy_json_value
from cayu.artifacts import ArtifactStore
from cayu.core.events import Event, EventType
from cayu.egress import (
    ApprovedEgressDestination,
    CredentialKind,
    CredentialMode,
    EgressAdapterRegistry,
    EgressBinding,
    EgressCapabilityEvidence,
    EgressDecision,
    EgressPolicy,
    EgressUpstream,
    InvalidEgressReconnectMetadataError,
    SandboxEgressAdapter,
    TransparentEgressBroker,
    UnsupportedEgressAdapter,
    UnsupportedEgressError,
    UnsupportedEgressReconnectError,
    VirtualCredentialGrant,
    VirtualCredentialRegistry,
    VirtualEgressRunnerRequest,
)
from cayu.egress.adapter import (
    DEFAULT_EGRESS_TEARDOWN_TIMEOUT_SECONDS,
    _await_bounded_cleanup_task,
)
from cayu.egress.credential_kinds import validate_credential_kind
from cayu.egress.destinations import validate_approved_destinations
from cayu.environments.admission import (
    ExecutionAdmissionCandidate,
    evaluate_execution_admission,
)
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.environments.bindings import (
    BoundWorkspace,
    NativeBinding,
    NoWorkspaceBinding,
    WorkspaceBinding,
    WorkspaceSnapshot,
)
from cayu.environments.factory import (
    EnvironmentFactory,
    EnvironmentFactoryOperation,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
)
from cayu.runners.base import (
    DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ExecCommand,
    ExecResult,
    Runner,
    RunnerWorkspaceCapabilityT,
)
from cayu.runtime._binding_cleanup import (
    BindingFinalizeFailure,
    binding_finalize_explicit_cancellation,
    binding_finalize_fatal_signal,
    record_binding_finalize_failures,
)
from cayu.vaults import SecretRedactor, SecretRef, SecretResolver
from cayu.workspaces import RunnerBoundWorkspace, Workspace

EventEmitter = Callable[[Event], Awaitable[Event]]
VirtualEgressWorkspaceFactory = Callable[[Runner], Workspace | Awaitable[Workspace]]

DEFAULT_SANDBOX_IMAGE = "python:3.12-slim"
VIRTUAL_EGRESS_RECONNECT_VERSION = 1
VIRTUAL_EGRESS_EVENT_TYPES = (
    EventType.CREDENTIAL_MODE_SELECTED,
    EventType.EGRESS_GRANT_MINTED,
    EventType.EGRESS_REQUEST_AUTHORIZED,
    EventType.EGRESS_REQUEST_DENIED,
    EventType.EGRESS_GRANT_REVOKED,
)


@dataclass(frozen=True)
class VirtualCredentialSpec:
    """Declares one virtual credential the runner workload should present."""

    env_name: str
    secret: SecretRef
    destination: str
    policy_name: str
    credential_kind: CredentialKind = "stripe_bearer"
    ttl_seconds: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "credential_kind", validate_credential_kind(self.credential_kind))


_RECONNECT_COMMON_FIELDS = {
    "version",
    "runner_kind",
    "session_id",
    "environment_name",
    "capability",
}
_SUPPORTED_RECONNECT_FIELDS = _RECONNECT_COMMON_FIELDS | {"identity"}
_UNSUPPORTED_RECONNECT_FIELDS = _RECONNECT_COMMON_FIELDS | {"reason"}
_REPLAYABLE_AUTHORITY_KEY_PARTS = {
    "auth",
    "authorization",
    "bearer",
    "cookie",
    "credential",
    "credentials",
    "passwd",
    "password",
    "secret",
    "secrets",
    "token",
    "tokens",
}
_REPLAYABLE_AUTHORITY_COMPACT_KEYS = {
    "accesstoken",
    "apikey",
    "caprivatekey",
    "presentedvalue",
    "privatekey",
    "proxyauthorization",
}


def _parse_reconnect_metadata(
    request: EnvironmentFactoryRequest,
    *,
    runner_kind: str,
) -> dict[str, Any] | None:
    metadata = request.reconnect_metadata
    if request.operation is EnvironmentFactoryOperation.RECONNECT and not metadata:
        raise InvalidEgressReconnectMetadataError(
            "Virtual-egress reconnect requires durable reconnect metadata; refusing to "
            "create a replacement environment during recovery."
        )
    if request.operation is EnvironmentFactoryOperation.CREATE and not metadata:
        return None
    fields = set(metadata)
    capability = metadata.get("capability")
    if capability == "supported":
        expected_fields = _SUPPORTED_RECONNECT_FIELDS
    elif capability == "unsupported":
        expected_fields = _UNSUPPORTED_RECONNECT_FIELDS
    else:
        raise InvalidEgressReconnectMetadataError(
            "Virtual-egress reconnect metadata capability must be supported or unsupported."
        )
    missing = expected_fields - fields
    unexpected = fields - expected_fields
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing {', '.join(sorted(missing))}")
        if unexpected:
            details.append(f"unexpected {', '.join(sorted(unexpected))}")
        raise InvalidEgressReconnectMetadataError(
            f"Virtual-egress reconnect metadata has an invalid schema ({'; '.join(details)})."
        )
    version = metadata["version"]
    if type(version) is not int or version != VIRTUAL_EGRESS_RECONNECT_VERSION:
        raise InvalidEgressReconnectMetadataError(
            "Virtual-egress reconnect metadata version is unsupported; "
            "the application must explicitly rebuild the environment."
        )
    if metadata["runner_kind"] != runner_kind:
        raise InvalidEgressReconnectMetadataError(
            "Virtual-egress reconnect metadata belongs to a different runner kind."
        )
    if metadata["environment_name"] != request.environment_name:
        raise InvalidEgressReconnectMetadataError(
            "Virtual-egress reconnect metadata belongs to a different environment."
        )
    identity: dict[str, Any] | None = None
    if capability == "supported":
        candidate_identity = metadata["identity"]
        if not isinstance(candidate_identity, dict) or not candidate_identity:
            raise InvalidEgressReconnectMetadataError(
                "Virtual-egress reconnect metadata identity must be a non-empty object."
            )
        _reject_replayable_authority(candidate_identity)
        identity = candidate_identity
    else:
        candidate_reason = metadata["reason"]
        if not isinstance(candidate_reason, str) or not candidate_reason.strip():
            raise InvalidEgressReconnectMetadataError(
                "Virtual-egress unsupported reconnect metadata requires a nonblank reason."
            )
    owner_session_id = metadata["session_id"]
    if request.operation is EnvironmentFactoryOperation.CREATE:
        if request.parent_session_id is not None and owner_session_id == request.parent_session_id:
            return None
        raise InvalidEgressReconnectMetadataError(
            "Virtual-egress create operations cannot attach reconnect metadata; use an "
            "explicit reconnect operation."
        )
    if owner_session_id != request.session_id:
        raise InvalidEgressReconnectMetadataError(
            "Virtual-egress reconnect metadata belongs to a different session."
        )
    if capability == "unsupported":
        raise UnsupportedEgressReconnectError(
            f"Runner {runner_kind!r} does not support virtual-egress reconnect. "
            "The application must explicitly rebuild the environment."
        )
    assert identity is not None
    return copy_json_value(identity, "reconnect_metadata.identity")


def _build_reconnect_metadata(
    request: EnvironmentFactoryRequest,
    *,
    runner_kind: str,
    identity: dict[str, Any],
    supported: bool,
) -> dict[str, Any]:
    if not isinstance(identity, dict):
        raise TypeError("Egress adapter reconnect metadata must be a dictionary.")
    common = {
        "version": VIRTUAL_EGRESS_RECONNECT_VERSION,
        "runner_kind": runner_kind,
        "session_id": request.session_id,
        "environment_name": request.environment_name,
    }
    if not supported:
        return {
            **common,
            "capability": "unsupported",
            "reason": (
                f"Runner {runner_kind!r} does not support virtual-egress reconnect. "
                "The application must explicitly rebuild the environment."
            ),
        }
    if not identity:
        raise InvalidEgressReconnectMetadataError(
            f"Runner {runner_kind!r} declared reconnect support without durable identity."
        )
    copied_identity = copy_json_value(identity, "adapter reconnect metadata")
    _reject_replayable_authority(copied_identity)
    return {
        **common,
        "capability": "supported",
        "identity": copied_identity,
    }


def _reject_replayable_authority(value: Any, *, path: str = "identity") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key.strip())
            parts = tuple(part for part in re.split(r"[^A-Za-z0-9]+", separated.lower()) if part)
            compact = "".join(parts)
            if any(part in _REPLAYABLE_AUTHORITY_KEY_PARTS for part in parts) or any(
                marker in compact for marker in _REPLAYABLE_AUTHORITY_COMPACT_KEYS
            ):
                raise InvalidEgressReconnectMetadataError(
                    f"Virtual-egress reconnect metadata cannot contain replayable authority at "
                    f"{path}.{key}."
                )
            _reject_replayable_authority(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_replayable_authority(item, path=f"{path}[{index}]")


class VirtualEgressEnvironmentFactory(EnvironmentFactory):
    """Per-session environment factory that enforces virtual egress.

    ``create`` mints grants, builds a broker (wired to emit audit events),
    prepares the explicitly selected egress adapter, and returns an
    ``Environment`` whose runner is on the enforced network and whose binding
    tears everything down at session end. Omitted and unsupported selections
    fail at construction before per-session resources exist.

    Scope: virtual egress governs the **runner process** credential — the value
    the executed app can read. It does not govern MCP servers: ``McpServerSpec``
    ``secret_env``/``secret_headers`` are resolved *host-side* (into the MCP
    server subprocess or the host HTTP client), never injected into the runner,
    so they sit at the ``trusted_tool`` boundary and are outside this factory.
    """

    def __init__(
        self,
        *,
        policies: Mapping[str, EgressPolicy],
        credentials: Sequence[VirtualCredentialSpec] = (),
        approved_destinations: Sequence[ApprovedEgressDestination] = (),
        resolver: SecretResolver | None = None,
        image: str = DEFAULT_SANDBOX_IMAGE,
        setup_commands: Sequence[str] = (),
        adapter: SandboxEgressAdapter | None = None,
        adapter_registry: EgressAdapterRegistry | None = None,
        runner_kind: str | None = None,
        inner_binding: WorkspaceBinding | None = None,
        workspace_factory: VirtualEgressWorkspaceFactory | None = None,
        artifact_store: ArtifactStore | None = None,
        event_emitter: EventEmitter | None = None,
        upstream: EgressUpstream | None = None,
        require_test_mode_credentials: bool = True,
    ) -> None:
        if not credentials and not approved_destinations:
            raise ValueError(
                "VirtualEgressEnvironmentFactory requires at least one credential or "
                "approved destination."
            )
        if credentials and resolver is None:
            raise ValueError("Virtual-egress credentials require a secret resolver.")
        if adapter is not None and adapter_registry is not None:
            raise ValueError("Pass either adapter or adapter_registry, not both.")
        if adapter is not None:
            if isinstance(adapter, UnsupportedEgressAdapter):
                raise UnsupportedEgressError(
                    f"Runner {adapter.runner_kind!r} has no enforcing egress adapter."
                )
            if runner_kind is not None and adapter.runner_kind != runner_kind:
                raise ValueError(
                    f"Explicit adapter runner kind {adapter.runner_kind!r} does not match "
                    f"runner_kind {runner_kind!r}."
                )
            selected_runner_kind = adapter.runner_kind
        else:
            if runner_kind is None:
                raise ValueError(
                    "VirtualEgressEnvironmentFactory requires an explicit adapter or runner_kind."
                )
            selected_runner_kind = runner_kind
            if adapter_registry is None and selected_runner_kind != "docker":
                raise UnsupportedEgressError(
                    f"Runner {selected_runner_kind!r} has no built-in enforcing egress "
                    "adapter; pass an explicit adapter or adapter_registry."
                )
            if adapter_registry is not None:
                selected_adapter = adapter_registry.resolve(selected_runner_kind)
                if isinstance(selected_adapter, UnsupportedEgressAdapter):
                    raise UnsupportedEgressError(
                        f"Runner {selected_runner_kind!r} has no registered enforcing "
                        "egress adapter; refusing to fall back to Docker."
                    )
        duplicate_env_names = _duplicate_env_names(credentials)
        if duplicate_env_names:
            raise ValueError(
                "Virtual credential env_name values must be unique: "
                + ", ".join(duplicate_env_names)
            )
        self._resolver = resolver
        self._policies = dict(policies)
        self._credentials = tuple(credentials)
        self._approved_destinations = validate_approved_destinations(
            approved_destinations,
        )
        self._image = image
        self._setup_commands = tuple(setup_commands)
        self._adapter = adapter
        self._adapter_registry = adapter_registry
        self._runner_kind = selected_runner_kind
        if workspace_factory is not None and not callable(workspace_factory):
            raise TypeError("workspace_factory must be callable or None.")
        self._workspace_factory = workspace_factory
        self._inner_binding = inner_binding or (
            NativeBinding() if workspace_factory is not None else NoWorkspaceBinding()
        )
        if artifact_store is not None and not isinstance(artifact_store, ArtifactStore):
            raise TypeError("artifact_store must be an ArtifactStore.")
        self._artifact_store = artifact_store
        self._emitter = event_emitter
        self._upstream = upstream
        self._require_test_mode = require_test_mode_credentials

    def execution_admission_candidate(
        self,
        request: EnvironmentFactoryRequest,
    ) -> ExecutionAdmissionCandidate:
        """Publish adapter-owned declarations without creating provider resources."""

        del request
        adapter = self._adapter or self._resolve_adapter(asyncio.get_running_loop())
        return ExecutionAdmissionCandidate(
            candidate=adapter.runner_kind,
            evidence=adapter.execution_capability_evidence(),
        )

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        loop = asyncio.get_running_loop()
        adapter = self._adapter or self._resolve_adapter(loop)
        runner_kind = adapter.runner_kind
        admission_evidence = adapter.execution_capability_evidence()
        evaluate_execution_admission(
            candidate=runner_kind,
            requirements=request.execution_requirements,
            evidence=admission_evidence,
            stage="pre_create",
        ).require_admitted()
        raw_configuration = adapter.configuration_metadata()
        if type(raw_configuration) is not dict:
            raise TypeError("Egress adapter configuration_metadata must return a dict.")
        configuration_metadata = copy_json_value(
            raw_configuration,
            "egress_configuration",
        )
        reconnect_identity = _parse_reconnect_metadata(
            request,
            runner_kind=runner_kind,
        )
        if reconnect_identity is not None:
            reconnect_identity = adapter.validate_reconnect_metadata(reconnect_identity)
        registry = VirtualCredentialRegistry()
        grants = [
            registry.mint(
                session_id=request.session_id,
                env_name=spec.env_name,
                secret=spec.secret,
                destination=spec.destination,
                credential_kind=spec.credential_kind,
                policy_name=spec.policy_name,
                ttl_seconds=spec.ttl_seconds,
            )
            for spec in self._credentials
        ]

        audit = _EgressAuditBridge(
            loop=loop,
            emitter=self._emitter,
            session_id=request.session_id,
            agent_name=request.agent_name,
            environment_name=request.environment_name,
        )
        broker = TransparentEgressBroker(
            registry=registry,
            resolver=self._resolver,
            policies=self._policies,
            approved_destinations=self._approved_destinations,
            upstream=self._upstream,
            audit=audit,
            require_test_mode_credentials=self._require_test_mode,
        )
        authority_revoker = _EgressAuthorityRevoker(
            grants=grants,
            broker=broker,
        )

        # From here on, adapter.prepare may return resources (proxy thread +
        # docker network + sidecar) before workspace binding/finalization is
        # guaranteed. Guard the whole handoff so the factory owns cleanup until
        # the returned Environment owns it.
        binding: EgressBinding | None = None
        ca_dir: str | None = None
        runner: Runner | None = None
        managed_runner: _EgressManagedRunner | None = None
        workspace: Workspace | None = None
        capability_metadata: dict[str, Any]
        try:
            if reconnect_identity is None:
                binding = await adapter.prepare(
                    session_id=request.session_id,
                    grants=grants,
                    broker=broker,
                )
            else:
                binding = await adapter.prepare_reconnect(
                    session_id=request.session_id,
                    environment_name=request.environment_name,
                    grants=grants,
                    broker=broker,
                    reconnect_metadata=reconnect_identity,
                )
            authority_revoker.teardown_timeout_s = binding.teardown_timeout_s
            ca_dir = tempfile.mkdtemp(prefix="cayu-egress-ca-")
            ca_host = os.path.join(ca_dir, "ca.pem")
            with open(ca_host, "wb") as handle:
                handle.write(binding.ca_cert_pem or b"")

            env_overlay = {**binding.env, **{g.env_name: g.presented_value for g in grants}}
            guest_ca_path = _required_binding_field(binding, "guest_ca_path")
            runner_request = VirtualEgressRunnerRequest(
                name=f"cayu-egress-sandbox-{secrets.token_hex(4)}",
                runner_kind=runner_kind,
                image=self._image,
                binding=binding,
                env_overlay=env_overlay,
                ca_cert_host_path=ca_host,
                guest_ca_path=guest_ca_path,
                setup_commands=self._setup_commands,
                egress_destinations=_ordered_destinations(
                    grants,
                    self._approved_destinations,
                ),
                session_id=request.session_id,
                environment_name=request.environment_name,
                parent_session_id=request.parent_session_id,
                reconnect_metadata=reconnect_identity or {},
            )
            runner = await adapter.create_runner(runner_request)
            runtime_admission_evidence = adapter.execution_capability_evidence(runner)
            runtime_admission = evaluate_execution_admission(
                candidate=runner_kind,
                requirements=request.execution_requirements,
                evidence=runtime_admission_evidence,
                stage="pre_exposure",
            ).require_admitted()
            if runtime_admission.evidence is None:
                raise RuntimeError("Admitted execution evidence is missing.")
            execution_capability_metadata = runtime_admission.evidence.to_metadata()
            if adapter.supports_reconnect:
                adapter_reconnect_metadata = adapter.validate_reconnect_metadata(
                    adapter.reconnect_metadata(runner)
                )
            else:
                adapter_reconnect_metadata = {}
            reconnect_metadata = _build_reconnect_metadata(
                request,
                runner_kind=runner_kind,
                identity=adapter_reconnect_metadata,
                supported=adapter.supports_reconnect,
            )
            evidence = adapter.capability_evidence(runner)
            if not isinstance(evidence, EgressCapabilityEvidence):
                raise TypeError(
                    "Egress adapter capability_evidence must return EgressCapabilityEvidence."
                )
            capability_metadata = evidence.to_metadata()

            managed_runner = _EgressManagedRunner(
                runner=runner,
                adapter=adapter,
                egress_binding=binding,
                ca_dir=ca_dir,
                authority_revoker=authority_revoker,
                audit=audit,
            )
            runner = None
            binding = None
            ca_dir = None
            workspace = await self._create_workspace(managed_runner)
            teardown_binding = _EgressTeardownBinding(
                inner=self._inner_binding,
                runner=managed_runner,
                grants=grants,
                emitter=self._emitter,
                audit=audit,
                session_id=request.session_id,
                agent_name=request.agent_name,
                environment_name=request.environment_name,
            )

            await self._emit_grant_events(request, grants, runner_kind=runner_kind)
            final_admission_candidate = managed_runner.execution_admission_candidate()
            if final_admission_candidate is None:
                raise RuntimeError("Managed egress runner omitted execution admission evidence.")
            final_admission = evaluate_execution_admission(
                candidate=runner_kind,
                requirements=request.execution_requirements,
                evidence=final_admission_candidate.evidence,
                stage="pre_exposure",
            ).require_admitted()
            if final_admission.evidence is None:
                raise RuntimeError("Admitted execution evidence is missing.")
            execution_capability_metadata = final_admission.evidence.to_metadata()
        except BaseException as original:
            cleanup_errors: list[tuple[str, BaseException]] = []
            original_cancellation = binding_finalize_explicit_cancellation(original)
            rollback_cancellation = original_cancellation
            deadline = asyncio.get_running_loop().time() + (
                binding.teardown_timeout_s
                if binding is not None
                else DEFAULT_EGRESS_TEARDOWN_TIMEOUT_SECONDS
            )
            if managed_runner is not None:
                try:
                    await managed_runner.finalize(
                        outcome="interrupted" if reconnect_identity is not None else None
                    )
                except asyncio.CancelledError as cancellation:
                    rollback_cancellation = cancellation
                except BaseException as cleanup_error:
                    cleanup_errors.append(("managed runner", cleanup_error))
            else:
                revocation_complete = False
                try:
                    if await authority_revoker.revoke(
                        timeout_s=_remaining_before_deadline(
                            deadline,
                            "Virtual-egress rollback timed out before grant revocation.",
                        )
                    ):
                        rollback_cancellation = asyncio.CancelledError()
                    revocation_complete = True
                except asyncio.CancelledError as cancellation:
                    rollback_cancellation = cancellation
                    revocation_complete = True
                except BaseException as cleanup_error:
                    cleanup_errors.append(("grant revocation", cleanup_error))
                if runner is not None and revocation_complete:
                    try:
                        if await _await_rollback_phase(
                            lambda: adapter.finalize_runner(
                                runner,
                                outcome=(
                                    "interrupted" if reconnect_identity is not None else "failed"
                                ),
                            ),
                            deadline=deadline,
                            phase="runner",
                        ):
                            rollback_cancellation = asyncio.CancelledError()
                    except asyncio.CancelledError as cancellation:
                        rollback_cancellation = cancellation
                    except BaseException as cleanup_error:
                        cleanup_errors.append(("runner", cleanup_error))
                if binding is not None and revocation_complete:
                    try:
                        if await _await_rollback_phase(
                            binding.close,
                            deadline=deadline,
                            phase="binding",
                        ):
                            rollback_cancellation = asyncio.CancelledError()
                    except asyncio.CancelledError as cancellation:
                        rollback_cancellation = cancellation
                    except BaseException as cleanup_error:
                        cleanup_errors.append(("binding", cleanup_error))
            if ca_dir is not None:
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    shutil.rmtree(ca_dir, ignore_errors=True)
            if cleanup_errors:
                details = "; ".join(
                    f"{phase}: {type(error).__name__}" for phase, error in cleanup_errors
                )
                original.add_note(f"Virtual-egress rollback incomplete: {details}.")
                cleanup_cancellation = next(
                    (
                        cancellation
                        for _, cleanup_error in cleanup_errors
                        if (cancellation := binding_finalize_explicit_cancellation(cleanup_error))
                        is not None
                    ),
                    None,
                )
                rollback_cancellation = rollback_cancellation or cleanup_cancellation
            if rollback_cancellation is not None and (
                cleanup_errors or original_cancellation is None
            ):
                failures = [original, *(error for _, error in cleanup_errors)]
                if (
                    binding_finalize_explicit_cancellation(
                        BaseExceptionGroup("Virtual-egress rollback failures.", failures)
                    )
                    is None
                ):
                    failures.append(rollback_cancellation)
                raise BaseExceptionGroup(
                    "Virtual-egress creation rollback failed after cancellation.",
                    failures,
                ) from rollback_cancellation
            raise
        environment_metadata: dict[str, Any] = {
            "kind": runner_kind,
            "credential_mode": CredentialMode.VIRTUAL_EGRESS.value,
        }
        result_metadata: dict[str, Any] = {}
        environment_metadata["egress_capabilities"] = capability_metadata
        result_metadata["egress_capabilities"] = capability_metadata
        execution_requirements_metadata = request.execution_requirements.model_dump(mode="json")
        environment_metadata["execution_requirements"] = execution_requirements_metadata
        result_metadata["execution_requirements"] = execution_requirements_metadata
        environment_metadata["execution_capabilities"] = execution_capability_metadata
        result_metadata["execution_capabilities"] = execution_capability_metadata
        if configuration_metadata:
            environment_metadata["egress_configuration"] = configuration_metadata
            result_metadata["egress_configuration"] = configuration_metadata
        spec = EnvironmentSpec(name=request.environment_name, metadata=environment_metadata)
        environment = Environment(
            spec,
            workspace=workspace,
            artifact_store=self._artifact_store,
            runner=managed_runner,
            binding=teardown_binding,
        )

        async def release(action: EnvironmentFactoryReleaseAction) -> None:
            await teardown_binding.release_unbound(
                outcome=(
                    "interrupted" if action is EnvironmentFactoryReleaseAction.PRESERVE else None
                )
            )

        return EnvironmentFactoryResult(
            environment=environment,
            metadata=result_metadata,
            reconnect_metadata=reconnect_metadata,
            release=release,
        )

    async def _create_workspace(self, runner: Runner) -> Workspace | None:
        if self._workspace_factory is None:
            return None
        created = self._workspace_factory(runner)
        if inspect.isawaitable(created):
            created = await created
        if not isinstance(created, Workspace):
            raise TypeError("workspace_factory must return a Workspace.")
        if isinstance(self._inner_binding, NativeBinding):
            if not isinstance(created, RunnerBoundWorkspace):
                raise TypeError(
                    "A NativeBinding virtual-egress workspace must implement "
                    "RunnerBoundWorkspace. Use an explicit non-native inner_binding for an "
                    "external workspace."
                )
            if not created.is_bound_to_runner(runner):
                raise ValueError(
                    "A NativeBinding virtual-egress workspace must be bound to the managed "
                    "runner passed to workspace_factory."
                )
            runner_key = runner.resource_key
            workspace_runner_key = created.bound_runner_resource_key
            if runner_key is None or workspace_runner_key is None:
                raise ValueError(
                    "A NativeBinding virtual-egress runner and workspace must expose stable "
                    "resource identity."
                )
            if workspace_runner_key != runner_key:
                raise ValueError(
                    "A NativeBinding virtual-egress workspace targets a different runner "
                    "resource than the managed runner."
                )
        return created

    def _resolve_adapter(self, loop: asyncio.AbstractEventLoop):
        if self._adapter_registry is not None:
            return self._adapter_registry.resolve(self._runner_kind)

        # Lazy import so `import cayu` never requires the [egress] extra.
        from cayu.egress.docker_adapter import DockerEgressAdapter

        registry = EgressAdapterRegistry()
        registry.register(DockerEgressAdapter(loop=loop))
        return registry.resolve(self._runner_kind)

    async def _emit_grant_events(
        self,
        request: EnvironmentFactoryRequest,
        grants: Sequence[VirtualCredentialGrant],
        *,
        runner_kind: str,
    ) -> None:
        if self._emitter is None:
            return
        with contextlib.suppress(Exception):
            await self._emitter(
                Event(
                    type=EventType.CREDENTIAL_MODE_SELECTED,
                    session_id=request.session_id,
                    agent_name=request.agent_name,
                    environment_name=request.environment_name,
                    payload={
                        "credential_mode": CredentialMode.VIRTUAL_EGRESS.value,
                        "runner_kind": runner_kind,
                        "grant_count": len(grants),
                        "approved_destination_count": len(self._approved_destinations),
                    },
                )
            )
            for grant in grants:
                await self._emitter(
                    Event(
                        type=EventType.EGRESS_GRANT_MINTED,
                        session_id=request.session_id,
                        agent_name=request.agent_name,
                        environment_name=request.environment_name,
                        payload=_grant_payload(grant),
                    )
                )


class _EgressAuthorityRevoker:
    """Disables a session's credentialed and credentialless egress authority."""

    def __init__(
        self,
        *,
        grants: Sequence[VirtualCredentialGrant],
        broker: TransparentEgressBroker,
    ) -> None:
        self._presented_values = tuple(grant.presented_value for grant in grants)
        self._broker = broker
        self._revoked = False
        self._drained = False
        self._task: asyncio.Task[None] | None = None
        self.teardown_timeout_s = DEFAULT_EGRESS_TEARDOWN_TIMEOUT_SECONDS

    async def revoke(self, *, timeout_s: float | None = None) -> bool:
        if self._drained:
            return False
        if self._task is None:
            self._task = asyncio.create_task(self._revoke_and_wait())
        task = self._task
        effective_timeout_s = self.teardown_timeout_s if timeout_s is None else timeout_s
        try:
            cancelled = await _await_cleanup_task(
                task,
                timeout_s=effective_timeout_s,
                timeout_message=(
                    "Virtual-egress grant revocation did not complete within "
                    f"{effective_timeout_s:g} seconds."
                ),
            )
        except BaseException:
            if task.done() and self._task is task:
                self._task = None
            raise
        self._drained = True
        return cancelled

    async def _revoke_and_wait(self) -> None:
        if not self._revoked:
            await self._broker.revoke_authority_and_wait(self._presented_values)
            self._revoked = True


async def _await_cleanup_task(
    task: asyncio.Task[None],
    *,
    timeout_s: float | None = None,
    timeout_message: str | None = None,
    cancellation: asyncio.CancelledError | None = None,
) -> bool:
    """Wait for cleanup to finish even if the awaiting task is cancelled."""
    if timeout_s is not None:
        return await _await_bounded_cleanup_task(
            task,
            timeout_s=timeout_s,
            timeout_message=timeout_message or "Cleanup timed out.",
            cancellation=cancellation,
        )
    task_outcome = await await_shielded_task_outcome(
        task,
        cancellation=cancellation,
    )
    if task_outcome.error is not None:
        if isinstance(task_outcome.error, asyncio.CancelledError):
            if task_outcome.cancellation is not None:
                raise task_outcome.cancellation from task_outcome.error
            raise task_outcome.error
        if task_outcome.cancellation is not None:
            raise BaseExceptionGroup(
                "Cleanup failed after caller cancellation.",
                [task_outcome.cancellation, task_outcome.error],
            ) from task_outcome.error
        raise task_outcome.error
    return task_outcome.cancellation is not None


async def _await_cleanup(awaitable: Awaitable[None]) -> bool:
    async def _run() -> None:
        await awaitable

    return await _await_cleanup_task(asyncio.create_task(_run()))


def _split_cleanup_cancellation(
    error: BaseExceptionGroup,
) -> tuple[asyncio.CancelledError | None, Exception | None]:
    """Separate one explicit cancellation from ordinary cleanup failures."""

    cancellation = binding_finalize_explicit_cancellation(error)
    _, ordinary_group = error.split(asyncio.CancelledError)
    if ordinary_group is None:
        return cancellation, None
    ordinary_error: BaseException = ordinary_group
    while isinstance(ordinary_error, BaseExceptionGroup) and len(ordinary_error.exceptions) == 1:
        ordinary_error = ordinary_error.exceptions[0]
    if not isinstance(ordinary_error, Exception):
        raise error
    return cancellation, ordinary_error


def _contains_timeout(error: BaseException) -> bool:
    """Return whether a cleanup error contains a timeout at any nesting level."""

    if isinstance(error, TimeoutError):
        return True
    if isinstance(error, BaseExceptionGroup):
        return any(_contains_timeout(child) for child in error.exceptions)
    return False


def _append_prior_cleanup_cancellation(
    error: BaseException,
    cancellation: asyncio.CancelledError | None,
) -> BaseException:
    """Retain cancellation completed by an earlier cleanup phase."""

    if cancellation is None or binding_finalize_explicit_cancellation(error) is not None:
        return error
    return BaseExceptionGroup(
        "Virtual-egress cleanup timed out after caller cancellation.",
        [cancellation, error],
    )


def _remaining_before_deadline(deadline: float, timeout_message: str) -> float:
    remaining = deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        raise TimeoutError(timeout_message)
    return remaining


async def _await_rollback_phase(
    close: Callable[[], Awaitable[None]],
    *,
    deadline: float,
    phase: str,
) -> bool:
    timeout_message = f"Virtual-egress {phase} rollback timed out."
    remaining = _remaining_before_deadline(deadline, timeout_message)

    async def run_close() -> None:
        await close()

    task = asyncio.create_task(run_close())
    return await _await_bounded_cleanup_task(
        task,
        timeout_s=remaining,
        timeout_message=timeout_message,
    )


class _EgressManagedRunner(Runner):
    """Runner wrapper that also owns pre-bind egress resources.

    Workspace binding finalization remains the normal session-end cleanup path,
    but a caller that closes the runner before binding/finalization still tears
    down the egress proxy/network/grants and CA material.
    """

    def __init__(
        self,
        *,
        runner: Runner,
        adapter: SandboxEgressAdapter,
        egress_binding: EgressBinding,
        ca_dir: str,
        authority_revoker: _EgressAuthorityRevoker,
        audit: _EgressAuditBridge | None = None,
    ) -> None:
        self._runner = runner
        self._adapter = adapter
        self._egress_binding = egress_binding
        self._ca_dir = ca_dir
        self._authority_revoker = authority_revoker
        self._audit = audit
        self._teardown_timeout_s = egress_binding.teardown_timeout_s
        self._runner_close_task: asyncio.Task[None] | None = None
        self._runner_close_action: str | None = None
        self._completed_runner_action: str | None = None
        self._binding_close_task: asyncio.Task[None] | None = None
        self._audit_drain_task: asyncio.Task[None] | None = None
        self._finalize_lock = asyncio.Lock()
        self._requested_runner_action: str | None = None
        self._requested_terminal_outcome: str | None = None
        self._binding_release_started = False
        self.isolation = runner.isolation
        self.default_cwd = runner.default_cwd
        self.system_execution_mode = runner.system_execution_mode

    @property
    def resource_key(self) -> tuple[object, ...] | None:
        return self._runner.resource_key

    def execution_admission_candidate(self) -> ExecutionAdmissionCandidate:
        """Return the adapter's evidence for the exact managed runner."""

        return ExecutionAdmissionCandidate(
            candidate=self._adapter.runner_kind,
            evidence=self._adapter.execution_capability_evidence(self._runner),
        )

    @property
    def closed(self) -> bool:
        """Report whether managed finalization completed."""

        return self._closed

    def workspace_capability(
        self,
        capability_type: type[RunnerWorkspaceCapabilityT],
    ) -> RunnerWorkspaceCapabilityT | None:
        """Delegate only the explicit, lifecycle-free workspace capability."""

        return self._runner.workspace_capability(capability_type)

    async def exec(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ) -> ExecResult:
        self._ensure_exec_open()
        return await self._runner.exec(
            command,
            cwd=cwd,
            env=env,
            timeout_s=timeout_s,
            stdin=stdin,
            output_limit_bytes=output_limit_bytes,
        )

    async def exec_system(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ) -> ExecResult:
        self._ensure_exec_open()
        return await self._runner.exec_system(
            command,
            cwd=cwd,
            env=env,
            timeout_s=timeout_s,
            stdin=stdin,
            output_limit_bytes=output_limit_bytes,
        )

    def reopen_exec(self) -> None:
        """Reopen both wrapper and inner execution after out-of-band verification."""

        if self._closed:
            super().reopen_exec()
        self._runner.reopen_exec()
        super().reopen_exec()

    async def close(self) -> None:
        await self.finalize(outcome=None)

    async def revoke_authority(self) -> bool:
        return await self._authority_revoker.revoke()

    async def finalize(self, *, outcome: str | None) -> None:
        requested_action = "detach" if outcome == "interrupted" else "remove"
        # Register an escalation before the first await so an in-flight detach
        # coordinator sees it before releasing the ownership claim.
        if not self._binding_release_started:
            self._register_runner_action(requested_action, outcome=outcome)
        async with self._finalize_lock:
            self._register_runner_action(requested_action, outcome=outcome)
            if self._closed and self._completed_runner_action == self._requested_runner_action:
                return
            if self._closed:
                self._closed = False
            await self._finalize_serialized()

    async def _finalize_serialized(self) -> None:
        deadline = asyncio.get_running_loop().time() + self._teardown_timeout_s
        cancellation = (
            asyncio.CancelledError()
            if await self._authority_revoker.revoke(
                timeout_s=self._remaining_teardown_time(deadline)
            )
            else None
        )
        # Do not release enforcement resources unless revocation completed.
        # A revocation error leaves this runner open for a truthful retry.
        errors: list[tuple[str, BaseException]] = []

        def record_timeout(phase: str, failure: BaseException) -> None:
            nonlocal cancellation
            if not isinstance(failure, BaseExceptionGroup):
                errors.append((phase, failure))
                return
            phase_cancellation, timeout_error = _split_cleanup_cancellation(failure)
            cancellation = cancellation or phase_cancellation
            if (
                phase_cancellation is not None
                and isinstance(timeout_error, TimeoutError)
                and len(failure.exceptions) == 2
            ):
                errors.append((phase, failure))
                return
            errors.append((phase, timeout_error or failure))

        try:
            runner_outcome = (
                "interrupted"
                if self._requested_runner_action == "detach"
                else self._requested_terminal_outcome
            )
            if await self._await_runner_close(outcome=runner_outcome, deadline=deadline):
                cancellation = cancellation or asyncio.CancelledError()
            self._completed_runner_action = self._runner_close_action
            # A terminal caller may have registered while detach was pending.
            if (
                self._completed_runner_action == "detach"
                and self._requested_runner_action == "remove"
            ):
                if await self._await_runner_close(
                    outcome=self._requested_terminal_outcome,
                    deadline=deadline,
                ):
                    cancellation = cancellation or asyncio.CancelledError()
                self._completed_runner_action = self._runner_close_action
        except TimeoutError as exc:
            timeout_failure = _append_prior_cleanup_cancellation(exc, cancellation)
            record_timeout("runner", timeout_failure)
        except BaseExceptionGroup as exc:
            fatal_signal = binding_finalize_fatal_signal(exc)
            if fatal_signal is not None:
                raise
            if exc.subgroup(TimeoutError) is not None:
                timeout_failure = _append_prior_cleanup_cancellation(exc, cancellation)
                record_timeout("runner", timeout_failure)
            else:
                phase_cancellation, cleanup_error = _split_cleanup_cancellation(exc)
                cancellation = cancellation or phase_cancellation
                if cleanup_error is not None:
                    errors.append(("runner", cleanup_error))
        except Exception as exc:
            errors.append(("runner", exc))
        # The binding owns the provider ownership claim. Never release it while
        # runner finalization is incomplete: another process could otherwise
        # attach to a sandbox that is still executable under this owner.
        if not errors:
            self._binding_release_started = True
            try:
                if await self._await_close_phase(
                    "_binding_close_task",
                    self._egress_binding.close,
                    deadline=deadline,
                    phase="binding",
                ):
                    cancellation = cancellation or asyncio.CancelledError()
            except TimeoutError as exc:
                timeout_failure = _append_prior_cleanup_cancellation(exc, cancellation)
                record_timeout("binding", timeout_failure)
            except BaseExceptionGroup as exc:
                fatal_signal = binding_finalize_fatal_signal(exc)
                if fatal_signal is not None:
                    raise
                if exc.subgroup(TimeoutError) is not None:
                    timeout_failure = _append_prior_cleanup_cancellation(exc, cancellation)
                    record_timeout("binding", timeout_failure)
                else:
                    phase_cancellation, cleanup_error = _split_cleanup_cancellation(exc)
                    cancellation = cancellation or phase_cancellation
                    if cleanup_error is not None:
                        errors.append(("binding", cleanup_error))
            except Exception as exc:
                errors.append(("binding", exc))
        if self._audit is not None and not any(_contains_timeout(error) for _, error in errors):
            try:
                if await self._await_close_phase(
                    "_audit_drain_task",
                    self._audit.drain,
                    deadline=deadline,
                    phase="audit",
                ):
                    cancellation = cancellation or asyncio.CancelledError()
            except TimeoutError as exc:
                timeout_failure = _append_prior_cleanup_cancellation(exc, cancellation)
                record_timeout("audit", timeout_failure)
            except BaseExceptionGroup as exc:
                fatal_signal = binding_finalize_fatal_signal(exc)
                if fatal_signal is not None:
                    raise
                if exc.subgroup(TimeoutError) is not None:
                    timeout_failure = _append_prior_cleanup_cancellation(exc, cancellation)
                    record_timeout("audit", timeout_failure)
                else:
                    phase_cancellation, cleanup_error = _split_cleanup_cancellation(exc)
                    cancellation = cancellation or phase_cancellation
                    if cleanup_error is not None:
                        errors.append(("audit", cleanup_error))
            except Exception as exc:
                errors.append(("audit", exc))
        if errors:
            if len(errors) == 1 and cancellation is None:
                only_error = errors[0][1]
                details = "; ".join(
                    f"{phase}: {type(error).__name__}: {error}" for phase, error in errors
                )
                if isinstance(only_error, TimeoutError):
                    raise only_error
                cleanup_error = RuntimeError(
                    f"Virtual-egress resource cleanup incomplete: {details}"
                )
                raise cleanup_error from only_error
            if (
                len(errors) == 1
                and cancellation is not None
                and isinstance(errors[0][1], BaseExceptionGroup)
                and binding_finalize_explicit_cancellation(errors[0][1]) is not None
            ):
                raise errors[0][1]
            if (
                cancellation is not None
                and len(errors) == 1
                and isinstance(errors[0][1], TimeoutError)
            ):
                raise BaseExceptionGroup(
                    "Virtual-egress cleanup failed after caller cancellation.",
                    [cancellation, errors[0][1]],
                ) from errors[0][1]
            failures = [error for _, error in errors]
            failure_tree = BaseExceptionGroup(
                "Virtual-egress resource cleanup phases failed.",
                failures,
            )
            details = "; ".join(
                f"{phase}: {type(error).__name__}: {error}" for phase, error in errors
            )
            cleanup_error = RuntimeError(f"Virtual-egress resource cleanup incomplete: {details}")
            cleanup_error.__cause__ = failure_tree
            if cancellation is not None:
                raise BaseExceptionGroup(
                    "Virtual-egress cleanup failed after caller cancellation.",
                    [cancellation, cleanup_error],
                ) from cleanup_error
            raise cleanup_error from failure_tree
        with contextlib.suppress(Exception):
            shutil.rmtree(self._ca_dir, ignore_errors=True)
        self._closed = True
        if cancellation is not None:
            raise cancellation

    def _register_runner_action(self, action: str, *, outcome: str | None) -> None:
        if action == "remove":
            self._requested_runner_action = "remove"
            self._requested_terminal_outcome = outcome
        elif self._requested_runner_action is None:
            self._requested_runner_action = "detach"

    async def _await_runner_close(self, *, outcome: str | None, deadline: float) -> bool:
        action = "detach" if outcome == "interrupted" else "remove"
        cancelled = False
        current = self._runner_close_task
        if current is not None and self._runner_close_action != action:
            # Never overlap provider lifecycle calls. Finish the in-flight
            # action before escalating detach -> remove, and never downgrade a
            # completed terminal removal back to detach.
            if self._runner_close_action == "remove":
                action = "remove"
            else:
                try:
                    cancelled = await _await_bounded_cleanup_task(
                        current,
                        timeout_s=self._remaining_teardown_time(deadline),
                        timeout_message=(
                            "Virtual-egress runner cleanup did not complete within "
                            f"{self._teardown_timeout_s:g} seconds."
                        ),
                    )
                except BaseException:
                    if current.done() and self._runner_close_task is current:
                        self._runner_close_task = None
                        self._runner_close_action = None
                    raise
                self._runner_close_task = None
                self._runner_close_action = None
        if self._runner_close_task is None:

            async def close_runner() -> None:
                effective_outcome = "interrupted" if action == "detach" else outcome
                await self._adapter.finalize_runner(self._runner, outcome=effective_outcome)

            self._runner_close_task = asyncio.create_task(close_runner())
            self._runner_close_action = action
        task = self._runner_close_task
        try:
            return (
                await _await_bounded_cleanup_task(
                    task,
                    timeout_s=self._remaining_teardown_time(deadline),
                    timeout_message=(
                        "Virtual-egress runner cleanup did not complete within "
                        f"{self._teardown_timeout_s:g} seconds."
                    ),
                )
                or cancelled
            )
        except BaseException:
            if task.done() and self._runner_close_task is task:
                self._runner_close_task = None
                self._runner_close_action = None
            raise

    def _remaining_teardown_time(self, deadline: float) -> float:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError(
                "Virtual-egress teardown did not complete within "
                f"{self._teardown_timeout_s:g} seconds."
            )
        return remaining

    async def _await_close_phase(
        self,
        task_field: str,
        close: Callable[[], Awaitable[None]],
        *,
        deadline: float,
        phase: str,
    ) -> bool:
        task = getattr(self, task_field)
        if task is None:

            async def run_close() -> None:
                await close()

            task = asyncio.create_task(run_close())
            setattr(self, task_field, task)
        try:
            return await _await_bounded_cleanup_task(
                task,
                timeout_s=self._remaining_teardown_time(deadline),
                timeout_message=(
                    f"Virtual-egress {phase} cleanup did not complete within "
                    f"{self._teardown_timeout_s:g} seconds."
                ),
            )
        except BaseException:
            if task.done() and getattr(self, task_field) is task:
                setattr(self, task_field, None)
            raise

    def resolve_cwd(self, cwd: str | None = None) -> str:
        return self._runner.resolve_cwd(cwd)


class _EgressAuditBridge:
    """Turns secret-free ``EgressDecision`` records into runtime events.

    Called synchronously from inside ``broker.handle_request`` (which runs on the
    app loop), so it schedules the async emit onto that loop without blocking.
    """

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        emitter: EventEmitter | None,
        session_id: str,
        agent_name: str,
        environment_name: str,
    ) -> None:
        self._loop = loop
        self._emitter = emitter
        self._session_id = session_id
        self._agent_name = agent_name
        self._environment_name = environment_name
        self._pending: set[concurrent.futures.Future[Event]] = set()

    def __call__(self, decision: EgressDecision) -> None:
        if self._emitter is None:
            return
        event = Event(
            type=EventType.EGRESS_REQUEST_AUTHORIZED
            if decision.allowed
            else EventType.EGRESS_REQUEST_DENIED,
            session_id=self._session_id,
            agent_name=self._agent_name,
            environment_name=self._environment_name,
            payload={
                "allowed": decision.allowed,
                "status_code": decision.status_code,
                "destination": decision.destination,
                "method": decision.method,
                "path": decision.path,
                "grant_id": decision.grant_id,
                "policy_name": decision.policy_name,
                "reason": decision.reason,
                "authorization_kind": decision.authorization_kind,
            },
        )
        emitter = self._emitter

        async def _emit() -> Event:
            return await emitter(event)

        coro = _emit()
        try:
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        except RuntimeError:
            # Loop stopped (e.g. a late request during teardown): close the
            # coroutine so it isn't left un-awaited.
            coro.close()
            return
        self._pending.add(future)
        future.add_done_callback(self._pending.discard)

    async def drain(self) -> None:
        while self._pending:
            await asyncio.gather(
                *(asyncio.wrap_future(future) for future in tuple(self._pending)),
                return_exceptions=True,
            )


class _EgressTeardownBinding(WorkspaceBinding):
    """Wraps an inner binding and runs egress teardown at session end."""

    def __init__(
        self,
        *,
        inner: WorkspaceBinding,
        runner: _EgressManagedRunner,
        grants: Sequence[VirtualCredentialGrant],
        emitter: EventEmitter | None,
        audit: _EgressAuditBridge | None,
        session_id: str,
        agent_name: str,
        environment_name: str,
    ) -> None:
        self._inner = inner
        self._runner = runner
        self._grants = tuple(grants)
        self._finalize_redactor = SecretRedactor(
            tuple(grant.presented_value for grant in self._grants)
        )
        self._emitter = emitter
        self._audit = audit
        self._session_id = session_id
        self._agent_name = agent_name
        self._environment_name = environment_name
        self._revocation_emit_lock = asyncio.Lock()
        self._revocation_emission_attempted_grant_ids: set[str] = set()

    async def bind(
        self,
        workspace: Any,
        runner: Runner | None,
        *,
        session_id: str,
        agent_name: str | None = None,
        environment_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BoundWorkspace:
        # Until bind returns successfully, the EnvironmentFactoryResult remains
        # unadopted and its release callback owns factory-created resources. The
        # inner binding remains responsible only for rolling back state that it
        # created while attempting this bind.
        return await self._inner.bind(
            workspace,
            runner,
            session_id=session_id,
            agent_name=agent_name,
            environment_name=environment_name,
            metadata=metadata,
        )

    async def release_unbound(self, *, outcome: str | None) -> None:
        """Release a pre-adoption environment with explicit preserve semantics."""

        cleanup_error: BaseException | None = None
        try:
            await self._close_resources(outcome=outcome)
        except Exception as initial_error:
            # Provider cleanup is designed to converge idempotently. Retry one
            # incomplete attempt inside the factory result's outer timeout so a
            # transient detach/remove failure does not leak an unadopted result.
            try:
                await self._close_resources(outcome=outcome)
            except BaseException as retry_error:
                retry_error.add_note(
                    "Virtual-egress factory release retry followed "
                    f"{type(initial_error).__name__}: {initial_error}."
                )
                cleanup_error = retry_error
        except BaseException as exc:
            cleanup_error = exc
        await self._drain_audit()
        if cleanup_error is not None:
            raise cleanup_error
        await self._emit_revoked()

    async def finalize(
        self,
        bound: BoundWorkspace,
        *,
        outcome: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkspaceSnapshot | None:
        snapshot: WorkspaceSnapshot | None = None
        inner_error: BaseException | None = None
        revoke_cancelled = False
        # Disable guest-side authority before workspace commands run. If this
        # fails, leave both the workspace and ownership claim untouched so a
        # truthful retry can resume from the same safety boundary.
        try:
            revoke_cancelled = await self._runner.revoke_authority()
        except BaseException as exc:
            record_binding_finalize_failures(
                exc,
                (
                    BindingFinalizeFailure(
                        phase="managed_resource_cleanup",
                        error=exc,
                    ),
                ),
                supplemental_redactor=self._finalize_redactor,
            )
            raise
        try:
            snapshot = await self._inner.finalize(bound, outcome=outcome, metadata=metadata)
        except BaseException as exc:
            inner_error = exc
            if binding_finalize_explicit_cancellation(exc) is not None:
                # This boundary already owns the workspace-side cancellation.
                # Normalize its task state before nested managed cleanup so the
                # shared waiter cannot report the same request a second time.
                consume_pending_task_cancellation()
        cleanup_error: BaseException | None = None
        try:
            # Workspace sync/unmount must finish before an interrupted MicroVM
            # is suspended or a terminal MicroVM is terminated.
            await self._close_resources(outcome=outcome)
        except BaseException as exc:
            if cleanup_error is None:
                cleanup_error = exc
        diagnostic_cancellation: asyncio.CancelledError | None = None
        diagnostic_fatal: BaseException | None = None
        try:
            if await _await_cleanup(self._drain_audit()):
                diagnostic_cancellation = asyncio.CancelledError()
        except asyncio.CancelledError as cancellation:
            diagnostic_cancellation = cancellation
        except BaseExceptionGroup as diagnostic_error:
            fatal_signal = binding_finalize_fatal_signal(diagnostic_error)
            if fatal_signal is not None:
                diagnostic_fatal = fatal_signal
            else:
                diagnostic_cancellation = binding_finalize_explicit_cancellation(diagnostic_error)
        except (KeyboardInterrupt, SystemExit, GeneratorExit) as fatal_signal:
            diagnostic_fatal = fatal_signal
        if diagnostic_fatal is None:
            try:
                if await _await_cleanup(self._emit_revoked()) and diagnostic_cancellation is None:
                    diagnostic_cancellation = asyncio.CancelledError()
            except asyncio.CancelledError as cancellation:
                if diagnostic_cancellation is None:
                    diagnostic_cancellation = cancellation
            except BaseExceptionGroup as diagnostic_error:
                fatal_signal = binding_finalize_fatal_signal(diagnostic_error)
                if fatal_signal is not None:
                    diagnostic_fatal = fatal_signal
                elif diagnostic_cancellation is None:
                    diagnostic_cancellation = binding_finalize_explicit_cancellation(
                        diagnostic_error
                    )
            except (KeyboardInterrupt, SystemExit, GeneratorExit) as fatal_signal:
                diagnostic_fatal = fatal_signal
        failures: list[BindingFinalizeFailure] = []
        if revoke_cancelled:
            failures.append(
                BindingFinalizeFailure(
                    phase="cancellation",
                    error=asyncio.CancelledError(),
                )
            )
        if inner_error is not None:
            failures.append(
                BindingFinalizeFailure(
                    phase="workspace_finalize",
                    error=inner_error,
                )
            )
        if cleanup_error is not None:
            failures.append(
                BindingFinalizeFailure(
                    phase="managed_resource_cleanup",
                    error=cleanup_error,
                )
            )
        if diagnostic_cancellation is not None:
            failures.append(
                BindingFinalizeFailure(
                    phase="cancellation",
                    error=diagnostic_cancellation,
                )
            )
        if diagnostic_fatal is not None:
            if not failures:
                raise diagnostic_fatal
            finalization_error = BaseExceptionGroup(
                "Virtual-egress finalization failed during diagnostics.",
                [*(failure.error for failure in failures), diagnostic_fatal],
            )
            record_binding_finalize_failures(
                finalization_error,
                tuple(failures),
                supplemental_redactor=self._finalize_redactor,
            )
            raise finalization_error from diagnostic_fatal
        if failures:
            if len(failures) == 1:
                finalization_error = failures[0].error
            else:
                finalization_error = BaseExceptionGroup(
                    "Virtual-egress finalization reported multiple failures.",
                    [failure.error for failure in failures],
                )
            record_binding_finalize_failures(
                finalization_error,
                tuple(failures),
                supplemental_redactor=self._finalize_redactor,
            )
            if len(failures) == 1:
                raise finalization_error
            raise finalization_error from failures[-1].error
        return snapshot

    async def _close_resources(
        self,
        *,
        outcome: str | None = None,
    ) -> None:
        cancelled = await _await_cleanup(self._runner.finalize(outcome=outcome))
        if cancelled:
            raise asyncio.CancelledError()

    async def _drain_audit(self) -> None:
        if self._audit is None:
            return
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await self._audit.drain()

    async def _emit_revoked(self) -> None:
        async with self._revocation_emit_lock:
            if all(
                grant.grant_id in self._revocation_emission_attempted_grant_ids
                for grant in self._grants
            ):
                return
            if self._emitter is None:
                self._revocation_emission_attempted_grant_ids.update(
                    grant.grant_id for grant in self._grants
                )
                return
            for grant in self._grants:
                if grant.grant_id in self._revocation_emission_attempted_grant_ids:
                    continue
                # Revocation events are best-effort and at-most-once. Mark the
                # attempt before awaiting the emitter so a committed delivery
                # with a lost acknowledgement is not duplicated by a retry.
                self._revocation_emission_attempted_grant_ids.add(grant.grant_id)
                with contextlib.suppress(Exception):
                    await self._emitter(
                        Event(
                            type=EventType.EGRESS_GRANT_REVOKED,
                            session_id=self._session_id,
                            agent_name=self._agent_name,
                            environment_name=self._environment_name,
                            payload=_grant_payload(grant),
                        )
                    )


def _grant_payload(grant: VirtualCredentialGrant) -> dict[str, Any]:
    return {
        "grant_id": grant.grant_id,
        "destination": grant.destination,
        "credential_kind": grant.credential_kind,
        "policy_name": grant.policy_name,
        "env_name": grant.env_name,
    }


def _required_binding_field(binding: EgressBinding, field_name: str) -> str:
    value = getattr(binding, field_name)
    if not isinstance(value, str) or not value:
        raise UnsupportedEgressError(
            f"Egress adapter did not return {field_name}; refusing to start "
            "a virtual-egress runner with an incomplete adapter binding."
        )
    return value


def _duplicate_env_names(credentials: Sequence[VirtualCredentialSpec]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for credential in credentials:
        if credential.env_name in seen:
            duplicates.add(credential.env_name)
        seen.add(credential.env_name)
    return tuple(sorted(duplicates))


def _ordered_destinations(
    grants: Sequence[VirtualCredentialGrant],
    approved_destinations: Sequence[ApprovedEgressDestination],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            [
                *(grant.destination for grant in grants),
                *(destination.destination for destination in approved_destinations),
            ]
        )
    )
