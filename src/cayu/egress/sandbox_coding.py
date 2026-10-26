"""Coding environments in an isolated cloud sandbox, behind one provider-neutral seam.

``SandboxCodingEnvironmentFactory`` gives each session its own sandbox from the
selected ``SandboxEgressAdapter``, copies a durable source workspace into it,
runs every file, search, check and command tool inside the sandbox, and
copies changes back when the session ends. The adapter is the only
provider-specific part: AWS Lambda MicroVM (``LambdaMicroVMEgressAdapter``) is
the first implementation, and another cloud's sandbox plugs in through its own
adapter without changing callers.

The factory is a preset over existing parts and adds no execution path of its
own: ``VirtualEgressEnvironmentFactory`` owns allocation, egress, admission,
recovery and teardown, ``SyncBinding`` owns copy-in and copy-back, and
``RunnerWorkspace`` exposes the sandbox filesystem to tools. Each part stays
usable on its own.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from cayu.artifacts import ArtifactStore
from cayu.egress.adapter import DEFAULT_SETUP_COMMAND_TIMEOUT_SECONDS, SandboxEgressAdapter
from cayu.egress.broker import (
    DEFAULT_EGRESS_MAX_ACTIVE_UPSTREAM_OPERATIONS,
    DEFAULT_EGRESS_UPSTREAM_MAX_RESPONSE_BYTES,
    DEFAULT_EGRESS_UPSTREAM_TOTAL_TIMEOUT_S,
)
from cayu.egress.destinations import ApprovedEgressDestination
from cayu.egress.policy import EgressPolicy
from cayu.egress.runtime import (
    DEFAULT_SANDBOX_IMAGE,
    EventEmitter,
    VirtualCredentialSpec,
    VirtualEgressEnvironmentFactory,
    VirtualEgressKnowledgeAccessScopeFactory,
)
from cayu.environments._sync_staging import SyncBindingStagingCapacity
from cayu.environments.bindings import (
    MAX_WORKSPACE_TRANSFER_ARCHIVE_BYTES,
    MAX_WORKSPACE_TRANSFER_FILE_BYTES,
    MAX_WORKSPACE_TRANSFER_FILES,
    MAX_WORKSPACE_TRANSFER_TOTAL_BYTES,
    SyncBinding,
    SyncBindingContext,
    SyncTargetWorkspacePlan,
)
from cayu.knowledge.base import KnowledgeStore
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.vaults import SecretResolver
from cayu.workspaces import LocalWorkspace, RunnerWorkspace, WorkspaceManifestLimits

__all__ = ["SandboxCodingEnvironmentFactory", "SandboxWorkspaceTransferLimits"]

# Directories that never cross between the source and the sandbox, in either direction.
_PROTECTED_DIRECTORY_NAMES = (".cayu", ".git", ".runtime")


class SandboxWorkspaceTransferLimits(BaseModel):
    """Finite copy-in and copy-back limits for one coding sandbox.

    Transfers spool to disk and reserve staging capacity from the files they
    measure, so these are ceilings, not reservations. A single file is held in
    memory while it is copied, which the staging capacity bounds.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    max_files: int = Field(default=100_000, ge=1, le=MAX_WORKSPACE_TRANSFER_FILES)
    max_file_bytes: int = Field(
        default=256 * 1024 * 1024,
        ge=1,
        le=MAX_WORKSPACE_TRANSFER_FILE_BYTES,
    )
    max_total_bytes: int = Field(
        default=4 * 1024 * 1024 * 1024,
        ge=1,
        le=MAX_WORKSPACE_TRANSFER_TOTAL_BYTES,
    )
    max_archive_bytes: int = Field(
        default=8 * 1024 * 1024 * 1024,
        ge=1024,
        le=MAX_WORKSPACE_TRANSFER_ARCHIVE_BYTES,
    )

    def manifest_limits(self) -> WorkspaceManifestLimits:
        """The bounds a command's workspace manifest is captured within."""

        return WorkspaceManifestLimits(
            max_paths=self.max_files,
            max_file_bytes=self.max_file_bytes,
            max_total_bytes=self.max_total_bytes,
        )


class SandboxCodingEnvironmentFactory(VirtualEgressEnvironmentFactory):
    """Run coding tools in a per-session cloud sandbox over a durable source workspace.

    ``source_workspace`` stays outside the sandbox and is the only durable copy:
    it is copied into the sandbox's working directory (the runner's default
    working directory, configured on the adapter) before the first tool call and
    receives changes back according to ``sync_back``. Sandbox filesystems are
    reconstructible execution state, never canonical. Commands and checks run in
    that same working directory, so a named check sees the files the agent
    edited. The sandbox copy excludes everything the source workspace excludes,
    ``sandbox_excluded_directory_names``, and the protected ``.git``, ``.cayu``
    and ``.runtime`` directories, which never cross in either direction: the
    sandbox gets the working tree, never the host's git config or hooks.
    Copy-back is revision-aware: a source file changed outside the sandbox after
    copy-in fails finalization with a conflict instead of being overwritten.

    All network traffic goes through the adapter's managed egress proxy, which
    decides every request and, with ``event_emitter``, records each decision on
    the session's event stream. Choose one explicit network mode:

    - **Allowlist** (``approved_destinations`` and/or ``credentials``): only the
      declared hosts are reachable, under their policies.
    - **Open internet** (``open_internet_policy`` naming an
      ``OpenInternetEgressPolicy`` in ``policies``): any public HTTPS host is
      reachable with every method and the sandbox's own headers, for package
      registries, git hosts and other public services. The environment holds
      no credentials: it refuses credentials, a resolver, approved destinations
      and other policies. Private, loopback, link-local and metadata addresses
      stay unreachable. It requires ``event_emitter``, so every request the
      sandbox makes is recorded.
    - **Public research** (``public_web_policy``): bodyless GET/HEAD to any
      public host with fixed headers, as on ``VirtualEgressEnvironmentFactory``.

    There is no network-free mode; one of them must be configured. Tools are
    admitted through the adapter's live executable evidence, so a tool whose
    executables are missing from the image is refused before the model sees it.

    One source workspace belongs to one active session at a time; a concurrent
    session over the same source is refused rather than sharing a writable
    sandbox.

    Each of ``setup_commands`` may run for ``setup_command_timeout_s`` (default
    300 s); provider timeouts are configured on the adapter.

    ``max_active_upstream_operations``, ``upstream_max_response_bytes`` and
    ``upstream_total_timeout_s`` set the egress broker's limits, as on
    ``VirtualEgressEnvironmentFactory``.
    """

    def __init__(
        self,
        *,
        adapter: SandboxEgressAdapter,
        source_workspace: LocalWorkspace,
        policies: Mapping[str, EgressPolicy],
        image: str = DEFAULT_SANDBOX_IMAGE,
        credentials: Sequence[VirtualCredentialSpec] = (),
        approved_destinations: Sequence[ApprovedEgressDestination] = (),
        public_web_policy: str | None = None,
        open_internet_policy: str | None = None,
        resolver: SecretResolver | None = None,
        setup_commands: Sequence[str] = (),
        setup_command_timeout_s: int = DEFAULT_SETUP_COMMAND_TIMEOUT_SECONDS,
        transfer_limits: SandboxWorkspaceTransferLimits | None = None,
        sync_back: Literal["always", "on_success", "never"] = "on_success",
        sandbox_excluded_directory_names: Sequence[str] = (),
        staging_capacity: SyncBindingStagingCapacity | None = None,
        artifact_store: ArtifactStore | None = None,
        knowledge_store: KnowledgeStore | None = None,
        knowledge_access_scope_factory: VirtualEgressKnowledgeAccessScopeFactory | None = None,
        event_emitter: EventEmitter | None = None,
        execution_profile_identity: ExecutionProfileBehaviorIdentity | None = None,
        max_active_upstream_operations: int = DEFAULT_EGRESS_MAX_ACTIVE_UPSTREAM_OPERATIONS,
        upstream_max_response_bytes: int = DEFAULT_EGRESS_UPSTREAM_MAX_RESPONSE_BYTES,
        upstream_total_timeout_s: float = DEFAULT_EGRESS_UPSTREAM_TOTAL_TIMEOUT_S,
    ) -> None:
        if not isinstance(source_workspace, LocalWorkspace):
            raise TypeError(
                "source_workspace must be a LocalWorkspace: durable storage outside the "
                "sandbox whose protected paths can be excluded before traversal."
            )
        if transfer_limits is None:
            transfer_limits = SandboxWorkspaceTransferLimits()
        elif type(transfer_limits) is not SandboxWorkspaceTransferLimits:
            raise TypeError("transfer_limits must be SandboxWorkspaceTransferLimits.")
        if open_internet_policy is not None and event_emitter is None:
            raise ValueError(
                "Open internet egress requires an event_emitter so every sandbox request "
                "is recorded."
            )
        # One projection serves both directions. Protected directories never cross:
        # the sandbox must not write .git config or hooks (or Cayu runtime state)
        # into the host checkout. Everything the source excludes stays out of the
        # sandbox copy too, so copy-back never tries to publish a path the source
        # refuses (a .pyc under an excluded __pycache__, say).
        names = {
            name.casefold(): name
            for name in (
                *source_workspace.excluded_directory_names,
                *sandbox_excluded_directory_names,
                *_PROTECTED_DIRECTORY_NAMES,
            )
        }
        excluded = tuple(names[key] for key in sorted(names))
        excluded_patterns = source_workspace.excluded_path_patterns
        protected_source = LocalWorkspace(
            source_workspace.root,
            workspace_id=source_workspace.id,
            excluded_directory_names=excluded,
            excluded_path_patterns=excluded_patterns,
        )

        def target_plan(context: SyncBindingContext) -> SyncTargetWorkspacePlan:
            if context.runner is None or context.session_id is None:
                raise RuntimeError("A sandbox coding target needs the session's managed runner.")
            return SyncTargetWorkspacePlan(
                workspace=RunnerWorkspace(
                    context.runner,
                    workspace_id=f"sandbox-coding:{context.session_id}",
                    excluded_directory_names=excluded,
                    excluded_path_patterns=excluded_patterns,
                    manifest_limits=transfer_limits.manifest_limits(),
                )
            )

        self.source_workspace = source_workspace
        self.transfer_limits = transfer_limits
        super().__init__(
            policies=policies,
            credentials=credentials,
            approved_destinations=approved_destinations,
            public_web_policy=public_web_policy,
            open_internet_policy=open_internet_policy,
            resolver=resolver,
            image=image,
            setup_commands=setup_commands,
            setup_command_timeout_s=setup_command_timeout_s,
            adapter=adapter,
            inner_binding=SyncBinding(
                target_workspace_plan_factory=target_plan,
                max_files=transfer_limits.max_files,
                max_file_bytes=transfer_limits.max_file_bytes,
                max_total_bytes=transfer_limits.max_total_bytes,
                max_archive_bytes=transfer_limits.max_archive_bytes,
                clean_target="always",
                sync_back=sync_back,
                delete_missing=True,
                # A source file changed outside the sandbox after copy-in is a
                # conflict at copy-back, never silently overwritten.
                source_conflict_policy="require_revision",
                preserve_git_modes=True,
                staging_capacity=staging_capacity,
            ),
            workspace_factory=lambda _runner: protected_source,
            artifact_store=artifact_store,
            knowledge_store=knowledge_store,
            knowledge_access_scope_factory=knowledge_access_scope_factory,
            event_emitter=event_emitter,
            execution_profile_identity=execution_profile_identity,
            max_active_upstream_operations=max_active_upstream_operations,
            upstream_max_response_bytes=upstream_max_response_bytes,
            upstream_total_timeout_s=upstream_total_timeout_s,
        )
