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
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from cayu._exception_groups import (
    add_exception_note_safely,
    exception_group_children,
    exception_tree_contains,
    set_exception_cause,
)
from cayu._task_wait import (
    ShieldedTaskOutcome,
    await_shielded_task_outcome,
    consume_pending_task_cancellation,
    unexpected_child_cancellation_error,
)
from cayu._validation import copy_json_value, require_clean_nonblank
from cayu._workspace_mutation import detached_workspace_mutation_process_signal
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
    RunnerFinalizationResult,
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
from cayu.egress.destinations import normalize_egress_hostname, validate_approved_destinations
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
    _EnvironmentLifecycleBindAttempt,
    copy_workspace_snapshot,
)
from cayu.environments.factory import (
    EnvironmentAllocationScope,
    EnvironmentAllocationUnsupportedError,
    EnvironmentFactory,
    EnvironmentFactoryOperation,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    attach_environment_factory_cleanup_settlement_task,
    combine_environment_factory_cleanup_settlement_tasks,
    environment_factory_cleanup_settlement_task,
    environment_factory_cleanup_settlement_tasks,
)
from cayu.runners._subprocess import (
    copy_runner_env,
    validate_output_limit,
    validate_runner_env_remove,
    validate_stdin,
    validate_timeout,
)
from cayu.runners.base import (
    DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ExecCommand,
    ExecResult,
    Runner,
    RunnerWorkspaceCapabilityT,
    _clean_runner_preflight,
    _clear_preflight_traceback_frames,
    copy_exec_command,
    runner_pending_command_settlement_cancellation_safe,
    runner_workspace_mutation_settlement,
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
_RollbackResultT = TypeVar("_RollbackResultT")

DEFAULT_SANDBOX_IMAGE = "python:3.12-slim"
VIRTUAL_EGRESS_RECONNECT_VERSION = 1
_FACTORY_CLEANUP_RETRY_INITIAL_BACKOFF_SECONDS = 0.05
_FACTORY_CLEANUP_RETRY_MAX_BACKOFF_SECONDS = 1.0
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
        object.__setattr__(
            self,
            "destination",
            normalize_egress_hostname(self.destination, field_name="destination"),
        )
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

    def allocation_scope(
        self,
        request: EnvironmentFactoryRequest,
    ) -> EnvironmentAllocationScope | None:
        adapter = self._adapter or self._resolve_adapter(asyncio.get_running_loop())
        self._require_remote_allocation_supported(request, adapter)
        return None

    @staticmethod
    def _require_remote_allocation_supported(
        request: EnvironmentFactoryRequest,
        adapter: SandboxEgressAdapter,
    ) -> None:
        if request.operation is not EnvironmentFactoryOperation.CREATE:
            return
        process_external_allocation = adapter.process_external_allocation
        if type(process_external_allocation) is not bool:
            raise EnvironmentAllocationUnsupportedError(
                f"Runner {adapter.runner_kind!r} must explicitly classify whether "
                "creation allocates a process-external provider resource."
            )
        if process_external_allocation:
            raise EnvironmentAllocationUnsupportedError(
                f"Runner {adapter.runner_kind!r} cannot allocate safely because its "
                "provider adapter does not support durable create-or-lookup recovery."
            )

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        loop = asyncio.get_running_loop()
        adapter = self._adapter or self._resolve_adapter(loop)
        runner_kind = adapter.runner_kind
        self._require_remote_allocation_supported(request, adapter)
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
                output_redactor=SecretRedactor(tuple(grant.presented_value for grant in grants)),
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
            factory_settlement_tasks: list[asyncio.Task[None]] = []
            if (
                factory_settlement_task := environment_factory_cleanup_settlement_task(original)
            ) is not None:
                factory_settlement_tasks.append(factory_settlement_task)
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
                    if not managed_runner.closed:
                        factory_settlement_tasks.append(
                            managed_runner.defer_finalization_settlement(
                                outcome=("interrupted" if reconnect_identity is not None else None),
                                prerequisite_task=(
                                    environment_factory_cleanup_settlement_task(cleanup_error)
                                ),
                            )
                        )
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
                        if (
                            cleanup_settlement_task := environment_factory_cleanup_settlement_task(
                                cleanup_error
                            )
                        ) is not None:
                            factory_settlement_tasks.append(cleanup_settlement_task)
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
                        if (
                            cleanup_settlement_task := environment_factory_cleanup_settlement_task(
                                cleanup_error
                            )
                        ) is not None:
                            factory_settlement_tasks.append(cleanup_settlement_task)
                        cleanup_errors.append(("binding", cleanup_error))
            if ca_dir is not None:
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    shutil.rmtree(ca_dir, ignore_errors=True)
            if cleanup_errors:
                details = "; ".join(
                    f"{phase}: {type(error).__name__}" for phase, error in cleanup_errors
                )
                add_exception_note_safely(
                    original,
                    f"Virtual-egress rollback incomplete: {details}.",
                )
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
            factory_settlement_task = _combine_environment_factory_settlement_tasks(
                factory_settlement_tasks
            )
            if factory_settlement_task is not None:
                attach_environment_factory_cleanup_settlement_task(
                    original,
                    factory_settlement_task,
                )
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
                rollback_error = BaseExceptionGroup(
                    "Virtual-egress creation rollback failed after cancellation.",
                    failures,
                )
                if factory_settlement_task is not None:
                    attach_environment_factory_cleanup_settlement_task(
                        rollback_error,
                        factory_settlement_task,
                    )
                raise rollback_error from rollback_cancellation
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
    ordinary_failures: list[Exception] = []
    pending: list[BaseException] = [error]
    while pending:
        candidate = pending.pop()
        if isinstance(candidate, asyncio.CancelledError):
            continue
        if isinstance(candidate, BaseExceptionGroup):
            children = exception_group_children(candidate)
            if children is None:
                raise error
            pending.extend(reversed(children))
            continue
        if not isinstance(candidate, Exception):
            raise error
        ordinary_failures.append(candidate)
    if not ordinary_failures:
        return cancellation, None
    if len(ordinary_failures) == 1:
        return cancellation, ordinary_failures[0]
    return (
        cancellation,
        ExceptionGroup(
            "Virtual-egress cleanup had concurrent failures.",
            ordinary_failures,
        ),
    )


def _contains_timeout(error: BaseException) -> bool:
    """Return whether a cleanup error contains a timeout at any nesting level."""

    return exception_tree_contains(error, TimeoutError)


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
    close: Callable[[], Awaitable[_RollbackResultT]],
    *,
    deadline: float,
    phase: str,
) -> bool:
    timeout_message = f"Virtual-egress {phase} rollback timed out."
    remaining = _remaining_before_deadline(deadline, timeout_message)

    async def run_close() -> None:
        # Rollback owns completion and cancellation, not the provider-specific
        # success payload. Typed results are interpreted at their owning boundary.
        await close()

    task = asyncio.create_task(run_close())
    try:
        return await _await_bounded_cleanup_task(
            task,
            timeout_s=remaining,
            timeout_message=timeout_message,
        )
    except TimeoutError as error:
        if not task.done():
            attach_environment_factory_cleanup_settlement_task(error, task)
        raise


def _combine_environment_factory_settlement_tasks(
    tasks: Sequence[asyncio.Task[None]],
) -> asyncio.Task[None] | None:
    """Return one exact owner for every still-relevant factory cleanup task."""

    return combine_environment_factory_cleanup_settlement_tasks(
        tasks,
        task_name="cayu-virtual-egress-factory-cleanup-settlement",
        failure_message="Virtual-egress factory cleanup settlement failed.",
    )


def _environment_factory_settlement_tasks(
    errors: Sequence[BaseException],
) -> tuple[asyncio.Task[None], ...]:
    """Collect authenticated cleanup owners carried by phase failures."""

    return tuple(
        dict.fromkeys(
            task for error in errors for task in environment_factory_cleanup_settlement_tasks(error)
        )
    )


_EGRESS_WORKSPACE_SYNC_OWNER: ContextVar[tuple[int, str] | None] = ContextVar(
    "cayu_egress_workspace_sync_owner",
    default=None,
)


def _workspace_dispatch_settlement_kind(
    *,
    result: ExecResult | None,
    error: BaseException | None,
) -> Literal["complete", "runner_quiescent", "deferred", "uncertain"]:
    """Classify whether one returned command can still mutate its workspace."""
    return runner_workspace_mutation_settlement(result=result, error=error)


def _validated_runner_dispatch_kwargs(
    runner: Runner,
    cwd: str | None,
    env: dict[str, str] | None,
    env_remove: tuple[str, ...],
    timeout_s: int | None,
    stdin: str | None,
    output_limit_bytes: int | None,
) -> dict[str, Any]:
    """Own validated command inputs before managed dispatch admission."""

    validated_cwd = runner.resolve_cwd(cwd)
    validated_env = None if env is None else copy_runner_env(env, inherit_env=False)
    validated_env_remove = validate_runner_env_remove(env_remove)
    kwargs: dict[str, Any] = {
        "cwd": validated_cwd,
        "env": validated_env,
        "timeout_s": validate_timeout(timeout_s),
        "stdin": validate_stdin(stdin),
        "output_limit_bytes": validate_output_limit(output_limit_bytes),
    }
    if validated_env_remove:
        kwargs["env_remove"] = validated_env_remove
    return kwargs


class _EgressManagedRunner(Runner):
    """Runner wrapper that also owns pre-bind egress resources.

    Workspace binding finalization remains the normal session-end cleanup path,
    but a caller that closes the runner before binding/finalization still tears
    down the egress proxy/network/grants and CA material.
    """

    pending_command_settlement_cancellation_safe = True

    def __init__(
        self,
        *,
        runner: Runner,
        adapter: SandboxEgressAdapter,
        egress_binding: EgressBinding,
        ca_dir: str,
        authority_revoker: _EgressAuthorityRevoker,
        output_redactor: SecretRedactor,
        audit: _EgressAuditBridge | None = None,
    ) -> None:
        if not isinstance(output_redactor, SecretRedactor):
            raise TypeError("output_redactor must be a SecretRedactor.")
        self._runner = runner
        self._adapter = adapter
        self._egress_binding = egress_binding
        self._ca_dir = ca_dir
        self._authority_revoker = authority_revoker
        self._output_redactor = output_redactor
        self._audit = audit
        self._teardown_timeout_s = egress_binding.teardown_timeout_s
        self._runner_close_task: asyncio.Task[RunnerFinalizationResult] | None = None
        self._runner_close_action: str | None = None
        self._completed_runner_action: str | None = None
        self._completed_runner_result: RunnerFinalizationResult | None = None
        self._binding_close_task: asyncio.Task[None] | None = None
        self._audit_drain_task: asyncio.Task[None] | None = None
        self._factory_cleanup_settlement_task: asyncio.Task[None] | None = None
        self._factory_cleanup_prerequisite_tasks: set[asyncio.Task[None]] = set()
        self._settled_factory_cleanup_prerequisite_tasks: set[asyncio.Task[None]] = set()
        self._finalize_lock = asyncio.Lock()
        self._requested_runner_action: str | None = None
        self._requested_terminal_outcome: str | None = None
        self._require_workspace_mutations_quiescent = False
        self._binding_release_started = False
        self._finalization_started = False
        self._binding_admissions = 0
        self._binding_admissions_drained = asyncio.Event()
        self._binding_admissions_drained.set()
        self._workspace_binding_owners: set[str] = set()
        self._workspace_binding_owners_drained = asyncio.Event()
        self._workspace_binding_owners_drained.set()
        self._workspace_dispatch_gate_owner: str | None = None
        self._workspace_sync_access_owner: str | None = None
        self._next_workspace_dispatch_id = 0
        self._active_workspace_dispatches: set[int] = set()
        self._uncertain_workspace_dispatches: dict[int, BaseException] = {}
        self._workspace_dispatch_settlement_tasks: set[asyncio.Task[None]] = set()
        self._active_workspace_dispatches_drained = asyncio.Event()
        self._active_workspace_dispatches_drained.set()
        self._workspace_target_unavailable_after_dispatch = False
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

    @property
    def workspace_mutations_quiescent(self) -> bool:
        result = self._completed_runner_result
        return self._closed and result is not None and result.workspace_mutations_quiescent

    @property
    def workspace_target_unavailable_after_dispatch(self) -> bool:
        """Whether command cleanup positively terminated the target sandbox."""

        return self._workspace_target_unavailable_after_dispatch

    async def await_pending_command_settlement(self) -> bool:
        """Join the wrapper-owned dispatch settlement without starting it twice."""

        await self._active_workspace_dispatches_drained.wait()
        process_signal = self._claim_workspace_dispatch_process_signal()
        if process_signal is not None:
            raise process_signal from None
        return not self._active_workspace_dispatches and not self._uncertain_workspace_dispatches

    def _claim_workspace_dispatch_process_signal(self) -> BaseException | None:
        """Claim sanitized deferred-settlement control evidence exactly once."""

        process_signals: list[BaseException] = []
        for dispatch_id, uncertainty in tuple(self._uncertain_workspace_dispatches.items()):
            process_signal = detached_workspace_mutation_process_signal(uncertainty)
            if process_signal is None:
                continue
            self._uncertain_workspace_dispatches[dispatch_id] = RuntimeError(
                "Managed runner deferred command settlement remains uncertain."
            )
            process_signals.append(process_signal)
        if len(process_signals) == 1:
            return process_signals[0]
        if process_signals:
            return BaseExceptionGroup(
                "Managed runner deferred command settlements carried process-control failures.",
                process_signals,
            )
        return None

    def workspace_capability(
        self,
        capability_type: type[RunnerWorkspaceCapabilityT],
    ) -> RunnerWorkspaceCapabilityT | None:
        """Delegate only the explicit, lifecycle-free workspace capability."""

        return self._runner.workspace_capability(capability_type)

    @_clean_runner_preflight
    def preflight_exec(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        env_remove: tuple[str, ...] = (),
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ) -> None:
        """Validate the selected backend before managed dispatch admission."""

        self._ensure_exec_open()
        try:
            owned_command = copy_exec_command(command)
            kwargs = _validated_runner_dispatch_kwargs(
                self._runner,
                cwd,
                env,
                env_remove,
                timeout_s,
                stdin,
                output_limit_bytes,
            )
            self._runner.preflight_exec(owned_command, **kwargs)
        except BaseException as error:
            _clear_preflight_traceback_frames(error)
            raise
        finally:
            del command

    async def exec(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        env_remove: tuple[str, ...] = (),
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ) -> ExecResult:
        self._ensure_exec_open()
        try:
            owned_command = copy_exec_command(command)
            kwargs = _validated_runner_dispatch_kwargs(
                self._runner,
                cwd,
                env,
                env_remove,
                timeout_s,
                stdin,
                output_limit_bytes,
            )
            self._runner.preflight_exec(owned_command, **kwargs)
        except BaseException as error:
            _clear_preflight_traceback_frames(error)
            owned_command = None
            kwargs = {}
            cwd = None
            env = None
            env_remove = ()
            stdin = None
            raise
        finally:
            del command
        dispatch_id = self._begin_workspace_dispatch()
        try:
            result = await self._runner.exec(owned_command, **kwargs)
        except BaseException as exc:
            self._finish_workspace_dispatch(dispatch_id=dispatch_id, error=exc)
            raise
        self._finish_workspace_dispatch(dispatch_id=dispatch_id, result=result)
        return result

    async def exec_redacted(
        self,
        command: ExecCommand,
        *,
        redactor: SecretRedactor,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        env_remove: tuple[str, ...] = (),
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ) -> ExecResult:
        self._ensure_exec_open()
        try:
            owned_command = copy_exec_command(command)
            kwargs = _validated_runner_dispatch_kwargs(
                self._runner,
                cwd,
                env,
                env_remove,
                timeout_s,
                stdin,
                output_limit_bytes,
            )
            self._runner.preflight_exec(owned_command, **kwargs)
        except BaseException as error:
            _clear_preflight_traceback_frames(error)
            owned_command = None
            kwargs = {}
            cwd = None
            env = None
            env_remove = ()
            stdin = None
            del redactor
            raise
        finally:
            del command
        kwargs["redactor"] = self._output_redactor.merged_with(redactor)
        dispatch_id = self._begin_workspace_dispatch()
        try:
            result = await self._runner.exec_redacted(owned_command, **kwargs)
        except BaseException as exc:
            self._finish_workspace_dispatch(dispatch_id=dispatch_id, error=exc)
            raise
        self._finish_workspace_dispatch(dispatch_id=dispatch_id, result=result)
        return result

    async def exec_system(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        env_remove: tuple[str, ...] = (),
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ) -> ExecResult:
        self._ensure_exec_open()
        try:
            owned_command = copy_exec_command(command)
            kwargs = _validated_runner_dispatch_kwargs(
                self._runner,
                cwd,
                env,
                env_remove,
                timeout_s,
                stdin,
                output_limit_bytes,
            )
            self._runner.preflight_exec(owned_command, **kwargs)
        except BaseException as error:
            _clear_preflight_traceback_frames(error)
            owned_command = None
            kwargs = {}
            cwd = None
            env = None
            env_remove = ()
            stdin = None
            raise
        finally:
            del command
        dispatch_id = self._begin_workspace_dispatch()
        try:
            result = await self._runner.exec_system(owned_command, **kwargs)
        except BaseException as exc:
            self._finish_workspace_dispatch(dispatch_id=dispatch_id, error=exc)
            raise
        self._finish_workspace_dispatch(dispatch_id=dispatch_id, result=result)
        return result

    def _begin_workspace_dispatch(self) -> int:
        """Admit one workload call or authenticate the retained sync owner."""

        self._ensure_exec_open()
        gate_owner = self._workspace_dispatch_gate_owner
        workspace_sync = (
            gate_owner is not None
            and self._workspace_sync_access_owner == gate_owner
            and _EGRESS_WORKSPACE_SYNC_OWNER.get() == (id(self), gate_owner)
        )
        if gate_owner is not None and not workspace_sync:
            raise RuntimeError(
                "Managed runner workspace dispatch is closed for binding finalization."
            )
        self._next_workspace_dispatch_id += 1
        dispatch_id = self._next_workspace_dispatch_id
        self._active_workspace_dispatches.add(dispatch_id)
        self._active_workspace_dispatches_drained.clear()
        return dispatch_id

    def _finish_workspace_dispatch(
        self,
        *,
        dispatch_id: int,
        result: ExecResult | None = None,
        error: BaseException | None = None,
    ) -> None:
        if dispatch_id not in self._active_workspace_dispatches:
            raise RuntimeError("Managed runner workspace dispatch owner is not active.")
        settlement = _workspace_dispatch_settlement_kind(result=result, error=error)
        if settlement in {"complete", "runner_quiescent"}:
            if settlement == "runner_quiescent":
                self._workspace_target_unavailable_after_dispatch = True
            self._retire_workspace_dispatch(dispatch_id)
            return
        if settlement == "uncertain":
            self._retire_workspace_dispatch(
                dispatch_id,
                error=RuntimeError(
                    "Managed runner command completion did not prove workspace mutation quiescence."
                ),
            )
            return

        if not runner_pending_command_settlement_cancellation_safe(self._runner):
            self._retire_workspace_dispatch(
                dispatch_id,
                error=RuntimeError(
                    "Managed runner deferred command settlement is not cancellation-safe."
                ),
            )
            return

        async def settle_deferred_dispatch() -> None:
            settlement_error: BaseException | None = None
            try:
                settled = await self._runner.await_pending_command_settlement()
                if type(settled) is not bool:
                    raise TypeError("Runner await_pending_command_settlement must return a bool.")
                if not settled:
                    settlement_error = RuntimeError(
                        "Managed runner deferred command cleanup did not prove "
                        "workspace mutation quiescence."
                    )
            except asyncio.CancelledError as cancellation:
                settlement_error = unexpected_child_cancellation_error(
                    cancellation,
                    operation="Managed runner deferred command settlement",
                )
            except BaseException as exc:
                settlement_error = detached_workspace_mutation_process_signal(exc) or RuntimeError(
                    "Managed runner deferred command settlement failed."
                )
                del exc
            self._retire_workspace_dispatch(
                dispatch_id,
                error=settlement_error,
            )

        settlement_task = asyncio.create_task(
            settle_deferred_dispatch(),
            name=f"cayu-egress-command-settlement-{dispatch_id}",
        )
        self._workspace_dispatch_settlement_tasks.add(settlement_task)
        settlement_task.add_done_callback(self._workspace_dispatch_settlement_tasks.discard)

    def _retire_workspace_dispatch(
        self,
        dispatch_id: int,
        *,
        error: BaseException | None = None,
    ) -> None:
        if dispatch_id not in self._active_workspace_dispatches:
            raise RuntimeError("Managed runner workspace dispatch owner is not active.")
        self._active_workspace_dispatches.remove(dispatch_id)
        if error is not None:
            self._uncertain_workspace_dispatches[dispatch_id] = error
        if not self._active_workspace_dispatches:
            self._active_workspace_dispatches_drained.set()

    def fence_workspace_dispatch(self, owner_key: str) -> None:
        """Close workload admission while retaining one binding's sync access."""

        owner = require_clean_nonblank(owner_key, "owner_key")
        if owner not in self._workspace_binding_owners:
            raise RuntimeError("Managed runner workspace sync owner is not active.")
        gate_owner = self._workspace_dispatch_gate_owner
        if gate_owner is None:
            self._workspace_dispatch_gate_owner = owner
        elif gate_owner != owner:
            raise RuntimeError("Managed runner workspace dispatch is fenced by another owner.")

    async def prepare_workspace_sync(
        self,
        owner_key: str,
    ) -> asyncio.CancelledError | None:
        """Drain calls admitted before ``owner_key`` performs its authoritative sync."""

        self.fence_workspace_dispatch(owner_key)
        drain_task = asyncio.create_task(
            self._active_workspace_dispatches_drained.wait(),
            name="cayu-egress-workspace-dispatch-drain",
        )
        drain_outcome = await await_shielded_task_outcome(
            drain_task,
            timeout_s=self._teardown_timeout_s,
        )
        if drain_outcome.timed_out:
            drain_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await drain_task
            timeout_error = TimeoutError(
                "Managed runner workspace dispatch did not quiesce within "
                f"{self._teardown_timeout_s:g} seconds."
            )
            if drain_outcome.cancellation is not None:
                raise BaseExceptionGroup(
                    "Managed runner workspace dispatch drain timed out after caller cancellation.",
                    [drain_outcome.cancellation, timeout_error],
                ) from timeout_error
            raise timeout_error
        if drain_outcome.error is not None:
            if drain_outcome.cancellation is not None:
                raise BaseExceptionGroup(
                    "Managed runner workspace dispatch drain failed after caller cancellation.",
                    [drain_outcome.cancellation, drain_outcome.error],
                ) from drain_outcome.error
            raise drain_outcome.error
        process_signal = self._claim_workspace_dispatch_process_signal()
        if process_signal is not None:
            if drain_outcome.cancellation is not None:
                raise BaseExceptionGroup(
                    "Managed runner command settlement carried a process-control "
                    "failure after caller cancellation.",
                    [drain_outcome.cancellation, process_signal],
                ) from process_signal
            raise process_signal from None
        if self._uncertain_workspace_dispatches:
            uncertainties = tuple(self._uncertain_workspace_dispatches.values())
            if len(uncertainties) == 1:
                settlement_error = uncertainties[0]
            else:
                settlement_error = BaseExceptionGroup(
                    "Managed runner command settlements remain uncertain.",
                    list(uncertainties),
                )
            if drain_outcome.cancellation is not None:
                raise BaseExceptionGroup(
                    "Managed runner command settlement remained uncertain after "
                    "caller cancellation.",
                    [drain_outcome.cancellation, settlement_error],
                ) from settlement_error
            raise settlement_error
        return drain_outcome.cancellation

    @contextlib.contextmanager
    def workspace_sync_access(self, owner_key: str) -> Iterator[None]:
        """Allow only one authenticated binding to use the fenced runner."""

        owner = require_clean_nonblank(owner_key, "owner_key")
        if (
            owner not in self._workspace_binding_owners
            or self._workspace_dispatch_gate_owner != owner
            or not self._active_workspace_dispatches_drained.is_set()
        ):
            raise RuntimeError("Managed runner workspace sync access is not authorized.")
        if self._workspace_sync_access_owner is not None:
            raise RuntimeError("Managed runner workspace sync access is already active.")
        self._workspace_sync_access_owner = owner
        token = _EGRESS_WORKSPACE_SYNC_OWNER.set((id(self), owner))
        try:
            yield
        finally:
            _EGRESS_WORKSPACE_SYNC_OWNER.reset(token)
            self._workspace_sync_access_owner = None

    def reopen_exec(self) -> None:
        """Reopen both wrapper and inner execution after out-of-band verification."""

        if self._active_workspace_dispatches:
            raise RuntimeError(
                "Managed runner has active command settlements and cannot be reopened."
            )
        if self._closed:
            super().reopen_exec()
        self._runner.reopen_exec()
        self._uncertain_workspace_dispatches.clear()
        super().reopen_exec()

    async def close(self) -> None:
        await self.finalize(outcome=None)

    async def revoke_authority(self) -> bool:
        return await self._authority_revoker.revoke()

    async def finalize(self, *, outcome: str | None) -> None:
        await self._finalize(
            outcome=outcome,
            require_workspace_mutations_quiescent=False,
        )

    def defer_finalization_settlement(
        self,
        *,
        outcome: str | None,
        prerequisite_task: asyncio.Task[None] | None = None,
    ) -> asyncio.Task[None]:
        """Own incomplete finalization until every retained phase settles."""

        if prerequisite_task is not None:
            if not isinstance(prerequisite_task, asyncio.Task):
                raise TypeError("Factory cleanup prerequisite must be an asyncio Task.")
            if prerequisite_task is self._factory_cleanup_settlement_task:
                raise ValueError("Factory cleanup settlement cannot depend on itself.")
            if prerequisite_task not in self._settled_factory_cleanup_prerequisite_tasks:
                self._factory_cleanup_prerequisite_tasks.add(prerequisite_task)
        self._register_runner_action(
            "detach" if outcome == "interrupted" else "remove",
            outcome=outcome,
        )
        current = self._factory_cleanup_settlement_task
        if current is not None:
            if not current.done():
                return current
            # A later release can escalate a completed detach into terminal
            # removal. Its cleanup needs a new owner rather than the successful
            # task for the earlier action.
            with contextlib.suppress(BaseException):
                current.result()
            self._factory_cleanup_settlement_task = None

        async def settle() -> None:
            retry_delay = _FACTORY_CLEANUP_RETRY_INITIAL_BACKOFF_SECONDS
            while True:

                async def run_finalization_attempt() -> BaseException | None:
                    try:
                        await self.finalize(outcome=outcome)
                    except BaseException as attempt_error:
                        # Complete the child task normally so a provider's
                        # child-only CancelledError cannot be mistaken for
                        # cancellation of this settlement owner.
                        return attempt_error
                    return None

                finalization_task = asyncio.create_task(
                    run_finalization_attempt(),
                    name="cayu-virtual-egress-factory-finalization-attempt",
                )
                finalization = await await_shielded_task_outcome(finalization_task)
                error = (
                    finalization.error if finalization.error is not None else finalization.result
                )
                if error is not None:
                    self._factory_cleanup_prerequisite_tasks.update(
                        prerequisite
                        for prerequisite in _environment_factory_settlement_tasks((error,))
                        if prerequisite not in self._settled_factory_cleanup_prerequisite_tasks
                    )
                if finalization.cancellation is not None:
                    if error is None:
                        raise finalization.cancellation
                    fatal_signal = binding_finalize_fatal_signal(error)
                    if fatal_signal is not None:
                        raise BaseExceptionGroup(
                            "Virtual-egress cleanup encountered a fatal signal "
                            "after settlement cancellation.",
                            [finalization.cancellation, error],
                        ) from error
                    raise finalization.cancellation from error
                if error is None:
                    if (
                        self._finalization_request_completed()
                        and not self._factory_cleanup_prerequisite_tasks
                    ):
                        return
                    continue
                if (
                    self._finalization_request_completed()
                    and not self._factory_cleanup_prerequisite_tasks
                ):
                    return
                fatal_signal = binding_finalize_fatal_signal(error)
                if fatal_signal is not None:
                    raise fatal_signal from error
                # Resume the exact retained phase tasks after every
                # recoverable provider failure, including child-only
                # cancellation. The bounded exponential delay prevents an
                # immediate failure from busy-looping while preserving the
                # same ownership fence.
                await asyncio.sleep(retry_delay)
                retry_delay = min(
                    retry_delay * 2,
                    _FACTORY_CLEANUP_RETRY_MAX_BACKOFF_SECONDS,
                )

        current = asyncio.create_task(
            settle(),
            name="cayu-virtual-egress-factory-finalization-settlement",
        )
        self._factory_cleanup_settlement_task = current
        return current

    async def _drain_factory_cleanup_prerequisites(self, *, deadline: float) -> None:
        """Reach terminal predecessor ownership before another finalization."""

        while self._factory_cleanup_prerequisite_tasks:
            prerequisite = next(iter(self._factory_cleanup_prerequisite_tasks))
            remaining = deadline - asyncio.get_running_loop().time()
            prerequisite_outcome = (
                ShieldedTaskOutcome[None](timed_out=True)
                if remaining <= 0
                else await await_shielded_task_outcome(
                    prerequisite,
                    timeout_s=remaining,
                )
            )
            if prerequisite_outcome.timed_out:
                timeout_error = TimeoutError(
                    "Virtual-egress predecessor cleanup did not complete within "
                    f"{self._teardown_timeout_s:g} seconds."
                )
                attach_environment_factory_cleanup_settlement_task(
                    timeout_error,
                    prerequisite,
                )
                if prerequisite_outcome.cancellation is not None:
                    cancellation_group = BaseExceptionGroup(
                        "Virtual-egress predecessor cleanup timed out after caller cancellation.",
                        [prerequisite_outcome.cancellation, timeout_error],
                    )
                    attach_environment_factory_cleanup_settlement_task(
                        cancellation_group,
                        prerequisite,
                    )
                    raise cancellation_group from timeout_error
                raise timeout_error
            prerequisite_error = prerequisite_outcome.error
            nested_prerequisites = (
                ()
                if prerequisite_error is None
                else _environment_factory_settlement_tasks((prerequisite_error,))
            )
            self._factory_cleanup_prerequisite_tasks.discard(prerequisite)
            self._settled_factory_cleanup_prerequisite_tasks.add(prerequisite)
            self._factory_cleanup_prerequisite_tasks.update(
                nested
                for nested in nested_prerequisites
                if nested not in self._settled_factory_cleanup_prerequisite_tasks
            )
            if prerequisite_outcome.cancellation is not None:
                if prerequisite_error is None:
                    raise prerequisite_outcome.cancellation
                fatal_signal = binding_finalize_fatal_signal(prerequisite_error)
                if fatal_signal is not None:
                    raise BaseExceptionGroup(
                        "Virtual-egress predecessor cleanup encountered a fatal "
                        "signal after caller cancellation.",
                        [prerequisite_outcome.cancellation, prerequisite_error],
                    ) from prerequisite_error
                raise prerequisite_outcome.cancellation from prerequisite_error
            if prerequisite_error is None:
                continue
            fatal_signal = binding_finalize_fatal_signal(prerequisite_error)
            if fatal_signal is not None:
                raise fatal_signal from prerequisite_error

    def _finalization_request_completed(self) -> bool:
        return (
            self._closed
            and self._completed_runner_action == self._requested_runner_action
            and (
                not self._require_workspace_mutations_quiescent
                or self.workspace_mutations_quiescent
            )
        )

    async def finalize_for_binding(
        self,
        *,
        outcome: str | None,
        require_workspace_mutations_quiescent: bool,
        workspace_owner_key: str | None = None,
    ) -> None:
        if type(require_workspace_mutations_quiescent) is not bool:
            raise TypeError("Managed runner quiescence requirement must be a bool.")
        if require_workspace_mutations_quiescent:
            if workspace_owner_key is None:
                if self._workspace_binding_owners:
                    raise ValueError(
                        "Managed runner binding finalization requires its active owner key."
                    )
            else:
                self._retire_workspace_binding_owner(workspace_owner_key)
        elif workspace_owner_key is not None:
            raise ValueError(
                "Managed runner binding finalization owner requires mutation quiescence."
            )
        if (
            require_workspace_mutations_quiescent
            and not self._require_workspace_mutations_quiescent
        ):
            self.arm_workspace_mutation_quiescence()
        # Preserve the ordinary dispatch seam for wrappers and diagnostics.
        # The quiescence requirement is already monotonic before this await.
        await self.finalize(outcome=outcome)

    def arm_workspace_mutation_quiescence(self) -> None:
        """Monotonically require quiescence before managed claim release."""

        if self._binding_release_started:
            raise RuntimeError(
                "Managed runner claim release started before workspace ownership "
                "could require mutation quiescence."
            )
        self._require_workspace_mutations_quiescent = True

    def begin_workspace_binding(self) -> None:
        """Fence runner finalization before a binding can mutate its workspace."""

        if self._finalization_started or self._binding_release_started:
            raise RuntimeError(
                "Managed runner finalization started before workspace binding admission."
            )
        if self._binding_admissions:
            raise RuntimeError(
                "Managed runner already has a workspace binding admission in progress."
            )
        if self._workspace_binding_owners:
            raise RuntimeError("Managed runner already has an active stateful workspace binding.")
        self._binding_admissions += 1
        self._binding_admissions_drained.clear()

    def finish_workspace_binding(
        self,
        *,
        require_mutation_quiescence: bool,
        workspace_owner_key: str | None = None,
    ) -> None:
        """Publish a binding's quiescence requirement and retire its admission."""

        if type(require_mutation_quiescence) is not bool:
            raise TypeError("Managed runner binding quiescence requirement must be a bool.")
        if self._binding_admissions <= 0:
            raise RuntimeError("Managed runner workspace binding admission is not active.")
        if require_mutation_quiescence:
            if workspace_owner_key is None:
                raise ValueError("Managed runner workspace binding requires its owner key.")
            owner = require_clean_nonblank(workspace_owner_key, "workspace_owner_key")
            if owner in self._workspace_binding_owners:
                raise RuntimeError("Managed runner received a duplicate workspace owner.")
            self._workspace_binding_owners.add(owner)
            self._workspace_binding_owners_drained.clear()
            self._require_workspace_mutations_quiescent = True
            if self._requested_runner_action == "detach":
                self._requested_runner_action = "quiesce"
        elif workspace_owner_key is not None:
            raise ValueError("Managed runner workspace owner requires mutation quiescence.")
        self._binding_admissions -= 1
        if self._binding_admissions == 0:
            self._binding_admissions_drained.set()

    def _retire_workspace_binding_owner(self, owner_key: str) -> None:
        owner = require_clean_nonblank(owner_key, "workspace_owner_key")
        if owner not in self._workspace_binding_owners:
            raise RuntimeError("Managed runner workspace binding owner is not active.")
        self._workspace_binding_owners.remove(owner)
        if not self._workspace_binding_owners:
            self._workspace_binding_owners_drained.set()

    def owns_workspace_binding(self, owner_key: str) -> bool:
        owner = require_clean_nonblank(owner_key, "workspace_owner_key")
        return owner in self._workspace_binding_owners

    def _close_workspace_dispatch_gate(self) -> None:
        if self._workspace_dispatch_gate_owner is not None:
            return
        if len(self._workspace_binding_owners) > 1:
            raise RuntimeError("Managed runner has multiple active workspace owners.")
        self._workspace_dispatch_gate_owner = next(
            iter(self._workspace_binding_owners),
            "runner-finalization",
        )

    async def _finalize(
        self,
        *,
        outcome: str | None,
        require_workspace_mutations_quiescent: bool,
    ) -> None:
        if type(require_workspace_mutations_quiescent) is not bool:
            raise TypeError("Managed runner quiescence requirement must be a bool.")
        self._finalization_started = True
        deadline = asyncio.get_running_loop().time() + self._teardown_timeout_s
        require_quiescence = (
            self._require_workspace_mutations_quiescent or require_workspace_mutations_quiescent
        )
        requested_action = (
            "quiesce"
            if outcome == "interrupted" and require_quiescence
            else ("detach" if outcome == "interrupted" else "remove")
        )
        # Register an escalation before the first await so an in-flight detach
        # coordinator sees it before releasing the ownership claim.
        if not self._binding_release_started:
            self._register_runner_action(requested_action, outcome=outcome)
            if require_workspace_mutations_quiescent:
                self._require_workspace_mutations_quiescent = True
        try:
            async with asyncio.timeout_at(deadline):
                await self._binding_admissions_drained.wait()
                # A bind admitted before finalization may still need the
                # runner and publishes its exact owner when this event drains.
                # Close dispatch immediately afterward, before the next await.
                self._close_workspace_dispatch_gate()
                await self._active_workspace_dispatches_drained.wait()
                await self._workspace_binding_owners_drained.wait()
                await self._finalize_lock.acquire()
        except TimeoutError as exc:
            raise TimeoutError(
                "Virtual-egress runner finalization did not acquire its binding "
                f"lifecycle boundary within {self._teardown_timeout_s:g} seconds."
            ) from exc
        try:
            require_quiescence = (
                self._require_workspace_mutations_quiescent or require_workspace_mutations_quiescent
            )
            requested_action = (
                "quiesce"
                if outcome == "interrupted" and require_quiescence
                else ("detach" if outcome == "interrupted" else "remove")
            )
            self._register_runner_action(requested_action, outcome=outcome)
            if require_workspace_mutations_quiescent:
                self._require_workspace_mutations_quiescent = True
            # Every entry point shares this lock, so an authenticated owner
            # published by a prior failed finalization is terminal before any
            # later provider lifecycle call begins.
            await self._drain_factory_cleanup_prerequisites(deadline=deadline)
            if self._finalization_request_completed():
                return
            if self._closed:
                self._closed = False
            try:
                await self._finalize_serialized(deadline=deadline)
            except BaseException as error:
                # Publish predecessor ownership before releasing the lock.
                # This closes the gap in which another direct finalizer could
                # otherwise enter before the background settlement sees it.
                self._factory_cleanup_prerequisite_tasks.update(
                    prerequisite
                    for prerequisite in _environment_factory_settlement_tasks((error,))
                    if prerequisite not in self._settled_factory_cleanup_prerequisite_tasks
                )
                raise
        finally:
            self._finalize_lock.release()

    async def _finalize_serialized(self, *, deadline: float) -> None:
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
                and len(exception_group_children(failure) or ()) == 2
            ):
                errors.append((phase, failure))
                return
            errors.append((phase, timeout_error or failure))

        try:
            runner_outcome = (
                "interrupted"
                if self._requested_runner_action in {"detach", "quiesce"}
                else self._requested_terminal_outcome
            )
            runner_cancelled, runner_result = await self._await_runner_close(
                outcome=runner_outcome,
                deadline=deadline,
            )
            if runner_cancelled:
                cancellation = cancellation or asyncio.CancelledError()
            self._completed_runner_action = self._runner_close_action
            self._completed_runner_result = runner_result
            # A terminal caller may have registered while a preserving action
            # was pending.
            if (
                self._completed_runner_action in {"detach", "quiesce"}
                and self._requested_runner_action == "remove"
            ):
                runner_cancelled, runner_result = await self._await_runner_close(
                    outcome=self._requested_terminal_outcome,
                    deadline=deadline,
                )
                if runner_cancelled:
                    cancellation = cancellation or asyncio.CancelledError()
                self._completed_runner_action = self._runner_close_action
                self._completed_runner_result = runner_result
            if (
                self._require_workspace_mutations_quiescent
                and not runner_result.workspace_mutations_quiescent
            ):
                errors.append(
                    (
                        "runner",
                        RuntimeError(
                            "Managed runner cleanup did not prove workspace mutation quiescence."
                        ),
                    )
                )
        except TimeoutError as exc:
            timeout_failure = _append_prior_cleanup_cancellation(exc, cancellation)
            record_timeout("runner", timeout_failure)
        except BaseExceptionGroup as exc:
            fatal_signal = binding_finalize_fatal_signal(exc)
            if fatal_signal is not None:
                raise
            if exception_tree_contains(exc, TimeoutError):
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
                if exception_tree_contains(exc, TimeoutError):
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
                if exception_tree_contains(exc, TimeoutError):
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
            factory_settlement_task = _combine_environment_factory_settlement_tasks(
                _environment_factory_settlement_tasks(tuple(error for _, error in errors))
            )
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
                if factory_settlement_task is not None:
                    attach_environment_factory_cleanup_settlement_task(
                        cleanup_error,
                        factory_settlement_task,
                    )
                raise cleanup_error from only_error
            if (
                len(errors) == 1
                and cancellation is not None
                and isinstance(errors[0][1], BaseExceptionGroup)
                and binding_finalize_explicit_cancellation(errors[0][1]) is not None
            ):
                if factory_settlement_task is not None:
                    attach_environment_factory_cleanup_settlement_task(
                        errors[0][1],
                        factory_settlement_task,
                    )
                raise errors[0][1]
            if (
                cancellation is not None
                and len(errors) == 1
                and isinstance(errors[0][1], TimeoutError)
            ):
                cleanup_group = BaseExceptionGroup(
                    "Virtual-egress cleanup failed after caller cancellation.",
                    [cancellation, errors[0][1]],
                )
                if factory_settlement_task is not None:
                    attach_environment_factory_cleanup_settlement_task(
                        cleanup_group,
                        factory_settlement_task,
                    )
                raise cleanup_group from errors[0][1]
            failures = [error for _, error in errors]
            failure_tree = BaseExceptionGroup(
                "Virtual-egress resource cleanup phases failed.",
                failures,
            )
            details = "; ".join(
                f"{phase}: {type(error).__name__}: {error}" for phase, error in errors
            )
            cleanup_error = RuntimeError(f"Virtual-egress resource cleanup incomplete: {details}")
            set_exception_cause(cleanup_error, failure_tree)
            if factory_settlement_task is not None:
                attach_environment_factory_cleanup_settlement_task(
                    cleanup_error,
                    factory_settlement_task,
                )
            if cancellation is not None:
                cleanup_group = BaseExceptionGroup(
                    "Virtual-egress cleanup failed after caller cancellation.",
                    [cancellation, cleanup_error],
                )
                if factory_settlement_task is not None:
                    attach_environment_factory_cleanup_settlement_task(
                        cleanup_group,
                        factory_settlement_task,
                    )
                raise cleanup_group from cleanup_error
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
        elif action == "quiesce" and self._requested_runner_action != "remove":
            self._requested_runner_action = "quiesce"
        elif self._requested_runner_action is None:
            self._requested_runner_action = "detach"

    async def _await_runner_close(
        self,
        *,
        outcome: str | None,
        deadline: float,
    ) -> tuple[bool, RunnerFinalizationResult]:
        action = self._requested_runner_action or (
            "detach" if outcome == "interrupted" else "remove"
        )
        if action not in {"detach", "quiesce", "remove"}:
            raise AssertionError(f"Unsupported managed runner action: {action}")
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

            async def close_runner() -> RunnerFinalizationResult:
                effective_outcome = "interrupted" if action in {"detach", "quiesce"} else outcome
                if action == "quiesce":
                    result = await self._adapter.finalize_runner_for_binding(
                        self._runner,
                        outcome=effective_outcome,
                    )
                else:
                    result = await self._adapter.finalize_runner(
                        self._runner,
                        outcome=effective_outcome,
                    )
                if result is None:
                    return RunnerFinalizationResult(workspace_mutations_quiescent=False)
                if type(result) is not RunnerFinalizationResult:
                    raise TypeError(
                        "SandboxEgressAdapter.finalize_runner must return RunnerFinalizationResult."
                    )
                if (
                    action == "quiesce"
                    and self._adapter.supports_reconnect
                    and not result.allocation_preserved
                ):
                    raise RuntimeError(
                        "Reconnectable egress adapter quiesced an interrupted runner "
                        "without preserving its allocation."
                    )
                return result

            self._runner_close_task = asyncio.create_task(close_runner())
            self._runner_close_action = action
        task = self._runner_close_task
        try:
            task_cancelled = await _await_bounded_cleanup_task(
                task,
                timeout_s=self._remaining_teardown_time(deadline),
                timeout_message=(
                    "Virtual-egress runner cleanup did not complete within "
                    f"{self._teardown_timeout_s:g} seconds."
                ),
            )
            return task_cancelled or cancelled, task.result()
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


@dataclass
class _EgressBoundFinalizationState:
    snapshot: WorkspaceSnapshot | None = None
    authority_revoked: bool = False
    workspace_pass_succeeded: bool = False
    workspace_frozen: bool = False
    cleanup_quiescent: bool = False
    diagnostics_completed: bool = False
    inner_retired: bool = False
    terminal_workspace_error: BaseException | None = None


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
        self._unbound_release_settlement_task: asyncio.Task[None] | None = None
        self._unbound_release_managed_settlement_task: asyncio.Task[None] | None = None
        self._finalize_lock = asyncio.Lock()
        self._bound_finalization_states: dict[str, _EgressBoundFinalizationState] = {}
        # A composite owner may need this wrapper's inner binding to remain
        # reserved after our runner becomes quiescent. The parent later retires
        # that exact owner through ``abandon`` once its own runner is closed.
        self._defer_inner_release = False

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
        return await self._bind(
            workspace,
            runner,
            session_id=session_id,
            agent_name=agent_name,
            environment_name=environment_name,
            metadata=metadata,
        )

    async def _bind_for_environment_lifecycle(
        self,
        workspace: Any,
        runner: Runner | None,
        *,
        session_id: str,
        agent_name: str | None = None,
        environment_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        _attempt: _EnvironmentLifecycleBindAttempt | None = None,
    ) -> BoundWorkspace:
        outer_attempt = _attempt or _EnvironmentLifecycleBindAttempt()
        self._runner.begin_workspace_binding()
        admission_active = True
        inner_attempt = _EnvironmentLifecycleBindAttempt()
        try:
            bound = await self._inner._bind_for_environment_lifecycle(
                workspace,
                runner,
                session_id=session_id,
                agent_name=agent_name,
                environment_name=environment_name,
                metadata=metadata,
                _attempt=inner_attempt,
            )
            requires_mutation_quiescence = self._inner._requires_mutation_quiescence(bound)
            workspace_owner_key = bound.state_key or f"bound:{id(bound)}"
            self._runner.finish_workspace_binding(
                require_mutation_quiescence=requires_mutation_quiescence,
                workspace_owner_key=(workspace_owner_key if requires_mutation_quiescence else None),
            )
            admission_active = False
            outer_attempt.retain(
                inner_attempt.release_failed_reservations,
            )
            return bound
        except BaseException:
            outer_attempt.retain(
                inner_attempt.release_failed_reservations,
            )
            raise
        finally:
            if admission_active:
                self._runner.finish_workspace_binding(
                    require_mutation_quiescence=False,
                )

    async def _bind(
        self,
        workspace: Any,
        runner: Runner | None,
        *,
        session_id: str,
        agent_name: str | None,
        environment_name: str | None,
        metadata: dict[str, Any] | None,
    ) -> BoundWorkspace:
        # Until bind returns successfully, the EnvironmentFactoryResult remains
        # unadopted and its release callback owns factory-created resources. The
        # inner binding remains responsible only for rolling back state that it
        # created while attempting this bind.
        self._runner.begin_workspace_binding()
        admission_active = True
        try:
            bound = await self._inner.bind(
                workspace,
                runner,
                session_id=session_id,
                agent_name=agent_name,
                environment_name=environment_name,
                metadata=metadata,
            )
            requires_mutation_quiescence = self._inner._requires_mutation_quiescence(bound)
            # Publish the requirement before the bound workspace is exposed.
            # A concurrent finalizer waits for this admission to drain, so it
            # cannot detach/release the runner between inner bind success and
            # the quiescence decision.
            workspace_owner_key = bound.state_key or f"bound:{id(bound)}"
            self._runner.finish_workspace_binding(
                require_mutation_quiescence=requires_mutation_quiescence,
                workspace_owner_key=(workspace_owner_key if requires_mutation_quiescence else None),
            )
            admission_active = False
            return bound
        finally:
            if admission_active:
                self._runner.finish_workspace_binding(
                    require_mutation_quiescence=False,
                )

    async def release_unbound(self, *, outcome: str | None) -> None:
        """Release a pre-adoption environment with explicit preserve semantics."""

        cleanup_error: BaseException | None = None
        try:
            await self._close_resources(outcome=outcome)
        except Exception as initial_error:
            initial_prerequisites = _environment_factory_settlement_tasks((initial_error,))
            if initial_prerequisites:
                # The provider authenticated an exact owner for work that may
                # still be mutating resources. Retain it before retrying so a
                # second finalization cannot overlap the predecessor.
                cleanup_error = initial_error
            else:
                # Provider cleanup is designed to converge idempotently. Retry
                # one incomplete attempt inside the factory result's outer
                # timeout so an ordinary transient failure does not leak an
                # unadopted result.
                try:
                    await self._close_resources(outcome=outcome)
                except BaseException as retry_error:
                    add_exception_note_safely(
                        retry_error,
                        "Virtual-egress factory release retry followed "
                        f"{type(initial_error).__name__}.",
                    )
                    cleanup_error = retry_error
        except BaseException as exc:
            cleanup_error = exc
        await self._drain_audit()
        if cleanup_error is not None:
            if binding_finalize_fatal_signal(cleanup_error) is None:
                prerequisite_task = _combine_environment_factory_settlement_tasks(
                    _environment_factory_settlement_tasks((cleanup_error,))
                )
                settlement_task = self._defer_unbound_release_settlement(
                    outcome=outcome,
                    prerequisite_task=prerequisite_task,
                )
                # Keep the provider's original handoff intact. Replacing it on
                # a reused exception can make the managed settlement adopt its
                # own wrapper task and deadlock.
                release_error = BaseExceptionGroup(
                    "Virtual-egress unbound resource cleanup remains incomplete.",
                    [cleanup_error],
                )
                attach_environment_factory_cleanup_settlement_task(
                    release_error,
                    settlement_task,
                )
                raise release_error from cleanup_error
            raise cleanup_error
        await self._emit_revoked()

    def _defer_unbound_release_settlement(
        self,
        *,
        outcome: str | None,
        prerequisite_task: asyncio.Task[None] | None,
    ) -> asyncio.Task[None]:
        """Retain unadopted resources and evidence until cleanup converges."""

        managed_settlement = self._runner.defer_finalization_settlement(
            outcome=outcome,
            prerequisite_task=prerequisite_task,
        )
        current = self._unbound_release_settlement_task
        if current is not None:
            if (
                not current.done()
                and self._unbound_release_managed_settlement_task is managed_settlement
            ):
                return current
            if current.done():
                with contextlib.suppress(BaseException):
                    current.result()
                current = None
                self._unbound_release_settlement_task = None
                self._unbound_release_managed_settlement_task = None
        prerequisites = _combine_environment_factory_settlement_tasks(
            tuple(task for task in (current, managed_settlement) if task is not None)
        )
        if prerequisites is None:
            raise AssertionError("Unbound release settlement omitted its cleanup owner.")

        async def settle() -> None:
            await prerequisites
            await self._drain_audit()
            await self._emit_revoked()

        current = asyncio.create_task(
            settle(),
            name="cayu-virtual-egress-unbound-release-settlement",
        )
        self._unbound_release_settlement_task = current
        self._unbound_release_managed_settlement_task = managed_settlement
        return current

    async def finalize(
        self,
        bound: BoundWorkspace,
        *,
        outcome: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkspaceSnapshot | None:
        async with self._finalize_lock:
            return await self._finalize_locked(
                bound,
                outcome=outcome,
                metadata=metadata,
            )

    async def _finalize_locked(
        self,
        bound: BoundWorkspace,
        *,
        outcome: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkspaceSnapshot | None:
        state_key = bound.state_key or f"bound:{id(bound)}"
        finalization_state = self._bound_finalization_states.setdefault(
            state_key,
            _EgressBoundFinalizationState(),
        )
        snapshot: WorkspaceSnapshot | None = None
        inner_error: BaseException | None = None
        revoke_cancelled = False
        dispatch_cancellation: asyncio.CancelledError | None = None
        requires_mutation_quiescence = self._inner._requires_mutation_quiescence(bound)
        runner_bound_target = isinstance(
            bound.workspace, RunnerBoundWorkspace
        ) and bound.workspace.is_bound_to_runner(self._runner)
        workspace_owner_active = (
            requires_mutation_quiescence and self._runner.owns_workspace_binding(state_key)
        )
        if (
            workspace_owner_active
            and not finalization_state.workspace_frozen
            and not finalization_state.inner_retired
        ):
            try:
                # Close workload admission before the first await. The exact
                # binding owner remains authorized to synchronize after calls
                # admitted before this boundary have drained.
                self._runner.fence_workspace_dispatch(state_key)
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
        # Disable guest-side authority before workspace commands run. If this
        # fails, leave both the workspace and ownership claim untouched so a
        # truthful retry can resume from the same safety boundary.
        if not finalization_state.authority_revoked:
            try:
                revoke_cancelled = await self._runner.revoke_authority()
                finalization_state.authority_revoked = True
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
        if (
            workspace_owner_active
            and not finalization_state.workspace_frozen
            and not finalization_state.inner_retired
        ):
            try:
                dispatch_cancellation = await self._runner.prepare_workspace_sync(state_key)
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
        if finalization_state.terminal_workspace_error is not None:
            inner_error = finalization_state.terminal_workspace_error
            snapshot = copy_workspace_snapshot(finalization_state.snapshot)
        elif finalization_state.workspace_frozen or finalization_state.inner_retired:
            snapshot = copy_workspace_snapshot(finalization_state.snapshot)
        else:
            try:
                # Workspace finalization precedes managed-runner teardown, but a
                # successful sync must not make a fixed target reusable while the
                # old guest can still execute. Stateful inner bindings retain
                # exact-owner retry state until this wrapper observes quiescence.
                self._inner._defer_finalize_release(bound)
                if requires_mutation_quiescence and runner_bound_target:
                    with self._runner.workspace_sync_access(state_key):
                        snapshot = await self._inner.finalize(
                            bound,
                            outcome=outcome,
                            metadata=metadata,
                        )
                else:
                    snapshot = await self._inner.finalize(
                        bound,
                        outcome=outcome,
                        metadata=metadata,
                    )
                finalization_state.snapshot = copy_workspace_snapshot(snapshot)
                finalization_state.workspace_pass_succeeded = True
                if requires_mutation_quiescence and runner_bound_target:
                    # No managed workload dispatch can enter after the gate,
                    # so a successful pass is authoritative for later cleanup
                    # and diagnostic retries even if the runner becomes closed.
                    finalization_state.workspace_frozen = True
            except BaseException as exc:
                finalization_state.workspace_pass_succeeded = False
                inner_error = exc
                if (
                    requires_mutation_quiescence
                    and runner_bound_target
                    and self._runner.workspace_target_unavailable_after_dispatch
                ):
                    # A command-level sandbox kill is positive quiescence but
                    # permanently removes the readable sync target. Preserve
                    # the authoritative sync failure while cleanup retires the
                    # now-unusable exact generation.
                    finalization_state.terminal_workspace_error = exc
                if binding_finalize_explicit_cancellation(exc) is not None:
                    # This boundary already owns the workspace-side cancellation.
                    # Normalize its task state before nested managed cleanup so the
                    # shared waiter cannot report the same request a second time.
                    consume_pending_task_cancellation()
        cleanup_error: BaseException | None = None
        if (
            inner_error is None
            or not requires_mutation_quiescence
            or finalization_state.terminal_workspace_error is not None
        ) and not finalization_state.cleanup_quiescent:
            try:
                # Workspace sync/unmount must finish before an interrupted MicroVM
                # is suspended or a terminal MicroVM is terminated.
                await self._close_resources(
                    outcome=outcome,
                    require_workspace_mutations_quiescent=requires_mutation_quiescence,
                    workspace_owner_key=(state_key if workspace_owner_active else None),
                )
                if not self._runner.is_closed:
                    raise RuntimeError(
                        "Managed runner remained open after successful finalization."
                    )
                if requires_mutation_quiescence and not self._runner.workspace_mutations_quiescent:
                    raise RuntimeError(
                        "Managed runner closed without proving workspace mutation quiescence."
                    )
                finalization_state.cleanup_quiescent = True
            except BaseException as exc:
                cleanup_error = exc
        if (
            finalization_state.terminal_workspace_error is not None
            and finalization_state.cleanup_quiescent
            and not finalization_state.inner_retired
        ):
            try:
                if self._inner.abandon(bound) is False:
                    raise RuntimeError(
                        "Inner workspace binding retained ownership after its "
                        "runner-backed target became unavailable."
                    )
                finalization_state.inner_retired = True
            except BaseException as exc:
                cleanup_error = (
                    exc
                    if cleanup_error is None
                    else BaseExceptionGroup(
                        "Virtual-egress cleanup and target retirement both failed.",
                        [cleanup_error, exc],
                    )
                )
        diagnostic_cancellation: asyncio.CancelledError | None = None
        diagnostic_fatal: BaseException | None = None
        if not finalization_state.diagnostics_completed:
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
                    diagnostic_cancellation = binding_finalize_explicit_cancellation(
                        diagnostic_error
                    )
            except (KeyboardInterrupt, SystemExit, GeneratorExit) as fatal_signal:
                diagnostic_fatal = fatal_signal
            if diagnostic_fatal is None:
                try:
                    if (
                        await _await_cleanup(self._emit_revoked())
                        and diagnostic_cancellation is None
                    ):
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
            if diagnostic_cancellation is None and diagnostic_fatal is None:
                finalization_state.diagnostics_completed = True
        if (
            inner_error is None
            and cleanup_error is None
            and not revoke_cancelled
            and dispatch_cancellation is None
            and diagnostic_cancellation is None
            and diagnostic_fatal is None
            and not self._defer_inner_release
            and not finalization_state.inner_retired
        ):
            try:
                if self._inner.abandon(bound) is False:
                    raise RuntimeError(
                        "Inner workspace binding retained ownership after managed cleanup."
                    )
                finalization_state.inner_retired = True
            except BaseException as exc:
                cleanup_error = exc
        failures: list[BindingFinalizeFailure] = []
        if revoke_cancelled:
            failures.append(
                BindingFinalizeFailure(
                    phase="cancellation",
                    error=asyncio.CancelledError(),
                )
            )
        if dispatch_cancellation is not None:
            failures.append(
                BindingFinalizeFailure(
                    phase="cancellation",
                    error=dispatch_cancellation,
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
        if finalization_state.inner_retired:
            self._bound_finalization_states.pop(state_key, None)
        return copy_workspace_snapshot(finalization_state.snapshot)

    def _defer_finalize_release(self, bound: BoundWorkspace) -> None:
        """Propagate a parent ownership fence through nested egress wrappers."""

        self._inner._defer_finalize_release(bound)
        self._defer_inner_release = True

    def _requires_mutation_quiescence(self, bound: BoundWorkspace) -> bool:
        return self._inner._requires_mutation_quiescence(bound)

    def abandon(self, bound: BoundWorkspace) -> bool:
        """Release inner retry ownership only after managed execution is quiescent."""

        # A revoke or managed-cleanup failure can leave the guest executable.
        # Releasing the workspace generation then would let that old guest race
        # a new owner. Keep the inner reservation fail-closed until runner
        # finalization positively reaches its terminal boundary.
        if not self._runner.is_closed:
            return False
        if (
            self._inner._requires_mutation_quiescence(bound)
            and not self._runner.workspace_mutations_quiescent
        ):
            return False
        state_key = bound.state_key or f"bound:{id(bound)}"
        finalization_state = self._bound_finalization_states.get(state_key)
        released = (
            True
            if finalization_state is not None and finalization_state.inner_retired
            else self._inner.abandon(bound)
        )
        if released is not False:
            self._defer_inner_release = False
            self._bound_finalization_states.pop(state_key, None)
        return released

    async def _close_resources(
        self,
        *,
        outcome: str | None = None,
        require_workspace_mutations_quiescent: bool = False,
        workspace_owner_key: str | None = None,
    ) -> None:
        cancelled = await _await_cleanup(
            self._runner.finalize_for_binding(
                outcome=outcome,
                require_workspace_mutations_quiescent=(require_workspace_mutations_quiescent),
                workspace_owner_key=workspace_owner_key,
            )
        )
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
