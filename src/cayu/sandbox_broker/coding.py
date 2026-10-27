"""Coding environments in a sandbox brokered by Cayu Cloud.

``BrokeredSandboxCodingEnvironmentFactory`` is the sibling of
``SandboxCodingEnvironmentFactory`` for Agents that must not hold
cloud-provider permissions. Each session asks the broker for its own sandbox
from a catalog image, copies the durable source workspace into it, runs every
file, search, check and command tool through the broker, copies changes back
when the session ends, and destroys the sandbox. The Agent holds only its
broker credential.

It composes the same parts as the direct factory: ``SyncBinding`` for
copy-in and revision-aware copy-back, and a ``RunnerWorkspace`` (here
``BrokeredSandboxWorkspace``) over the sandbox. Like Docker's coding binding,
it commits each copy-in to an ephemeral Git baseline in the sandbox. The broker
client, lease keeper and runner are each usable on their own.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Literal

from cayu._validation import canonical_durable_json_bytes
from cayu.egress._remote_adapter import setup_command_failure_message
from cayu.egress.adapter import validate_setup_command_timeout
from cayu.egress.sandbox_coding import SandboxWorkspaceTransferLimits
from cayu.environments import (
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryOperation,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    EnvironmentSpec,
)
from cayu.environments._ephemeral_git import (
    initialize_ephemeral_git_baseline,
    sync_bound_tree_size,
)
from cayu.environments._sync_staging import SyncBindingStagingCapacity
from cayu.environments.admission import (
    ExecutionAdmissionCandidate,
    ExecutionCapabilityEvidence,
    ExecutionExecutableEvidence,
    ExecutionRequirements,
    ExecutionToolRequirementEvidence,
)
from cayu.environments.bindings import (
    BoundWorkspace,
    SyncBinding,
    SyncBindingContext,
    SyncTargetWorkspacePlan,
    SyncTargetWorkspaceProvisioner,
    WorkspaceSnapshot,
)
from cayu.environments.coding_toolchains import (
    CodingToolchainError,
    CodingToolchainProfile,
    owned_coding_toolchain_profile,
    run_coding_toolchain_admission_probes,
    verify_local_coding_toolchain_dependencies,
)
from cayu.runners import ExecCommand
from cayu.runners.base import Runner
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sandbox_broker.client import (
    SandboxBrokerClient,
    SandboxBrokerError,
    SandboxEgressMode,
    SandboxRecord,
    validate_egress,
    validate_labels,
    validate_lease_seconds,
)
from cayu.sandbox_broker.lease import SandboxLeaseKeeper
from cayu.sandbox_broker.runner import (
    BROKERED_SANDBOX_CANDIDATE,
    MAX_TRANSFER_FILE_BYTES,
    STAGED_FILE_OVERHEAD_BYTES,
    BrokeredSandboxRunner,
    BrokeredSandboxWorkspace,
    StagedCopiesMode,
    sandbox_capability_claims,
    validate_staged_copies_mode,
)
from cayu.workspaces import LocalWorkspace, RunnerWorkspace, Workspace

if TYPE_CHECKING:
    from cayu.knowledge.base import KnowledgeStore
    from cayu.knowledge.scopes import KnowledgeAccessScope

__all__ = [
    "BROKERED_SANDBOX_EGRESS_ENV",
    "BROKERED_SANDBOX_EGRESS_HOSTS_ENV",
    "BROKERED_SANDBOX_IMAGE_ENV",
    "BROKERED_SANDBOX_LEASE_ENV",
    "BrokeredCodingBackendOptions",
    "BrokeredSandboxCodingBinding",
    "BrokeredSandboxCodingEnvironmentFactory",
    "brokered_coding_backend",
]

#: Catalog image for coding sandboxes (default ``coding-python``).
BROKERED_SANDBOX_IMAGE_ENV = "CAYU_CLOUD_SANDBOX_IMAGE"
#: ``none``, ``public_web`` or ``allowlist``; required by ``from_env``, since there is no
#: default network.
BROKERED_SANDBOX_EGRESS_ENV = "CAYU_CLOUD_SANDBOX_EGRESS"
#: With ``allowlist`` egress, and only with it: the hosts the sandbox may reach, separated by
#: commas or whitespace (``pypi.org,files.pythonhosted.org,*.githubusercontent.com``).
BROKERED_SANDBOX_EGRESS_HOSTS_ENV = "CAYU_CLOUD_SANDBOX_EGRESS_HOSTS"
#: Lease length in seconds, 60 to 3600 (default 900).
BROKERED_SANDBOX_LEASE_ENV = "CAYU_CLOUD_SANDBOX_LEASE_SECONDS"

_LOGGER = logging.getLogger(__name__)
DEFAULT_IMAGE = "coding-python"
DEFAULT_LEASE_SECONDS = 900
# Cloud fails a sandbox still provisioning after five minutes.
DEFAULT_READY_TIMEOUT_SECONDS = 330.0
#: How long each setup command may run, unless the factory sets otherwise.
DEFAULT_BROKERED_SETUP_COMMAND_TIMEOUT_SECONDS = 600
_PROTECTED_DIRECTORY_NAMES = (".cayu", ".git", ".runtime")
_FACTORY_LABEL_KEYS = frozenset({"cayu.environment", "cayu.session"})
_MAX_USER_LABELS = 10 - len(_FACTORY_LABEL_KEYS)
_TOOLCHAIN_WORKSPACE = "/workspace"


class BrokeredSandboxCodingBinding(SyncBinding):
    """``SyncBinding`` that destroys the brokered sandbox once finalization succeeds.

    With ``git_baseline`` (the default), each copy-in is committed to an
    ephemeral local Git repository in the sandbox, as Docker's coding binding
    does, so ``git_changes`` and the guest's own ``git`` see what the session
    changed. The repository is made in the target's directory and replaces
    any ``.git`` there, so with it the binding requires, before copying
    anything, a ``LocalWorkspace`` source and a ``RunnerWorkspace`` target on
    the bound sandbox runner, both excluding ``.git``. The source's repository
    never enters the sandbox, and the sandbox's never reaches the source.

    When copy-back fails, for example on a source conflict, the binding keeps
    the sandbox for a retry but stops renewing its lease, so the broker
    destroys it when the lease lapses if no retry comes.
    """

    def __init__(
        self,
        *,
        git_baseline: bool = True,
        workspace_setup_commands: Sequence[str] = (),
        setup_command_timeout_s: int = DEFAULT_BROKERED_SETUP_COMMAND_TIMEOUT_SECONDS,
        **options: Any,
    ) -> None:
        if type(git_baseline) is not bool:
            raise TypeError("git_baseline must be a bool.")
        self._setup_command_timeout_s = validate_setup_command_timeout(setup_command_timeout_s)
        super().__init__(**options)
        self._git_baseline = git_baseline
        self._workspace_setup_commands = _validated_commands(
            workspace_setup_commands, field_name="workspace_setup_commands"
        )

    async def bind(
        self,
        workspace: Workspace | None,
        runner: Runner | None,
        *,
        session_id: str,
        agent_name: str | None = None,
        environment_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BoundWorkspace:
        if self._git_baseline and not (
            isinstance(workspace, LocalWorkspace) and ".git" in workspace.excluded_directory_names
        ):
            raise ValueError(
                "A brokered coding Git baseline requires a source LocalWorkspace "
                "that excludes .git."
            )
        bound = await super().bind(
            workspace,
            runner,
            session_id=session_id,
            agent_name=agent_name,
            environment_name=environment_name,
            metadata=metadata,
        )
        if not self._git_baseline and not self._workspace_setup_commands:
            return bound
        try:
            target = bound.workspace
            if not isinstance(target, RunnerWorkspace) or not isinstance(
                bound.runner, BrokeredSandboxRunner
            ):
                raise RuntimeError("Brokered coding binding lost its sandbox target.")
            if self._git_baseline:
                tree_bytes, tree_files = sync_bound_tree_size(bound.metadata)
                await initialize_ephemeral_git_baseline(
                    bound.runner,
                    cwd=target.runner_cwd,
                    tree_bytes=tree_bytes,
                    tree_files=tree_files,
                )
            for command in self._workspace_setup_commands:
                result = await bound.runner.exec(
                    ExecCommand.bash(command),
                    cwd=target.runner_cwd,
                    timeout_s=self._setup_command_timeout_s,
                )
                if result.exit_code != 0 or result.timed_out or result.cancelled:
                    raise RuntimeError(
                        setup_command_failure_message(
                            "Brokered sandbox workspace setup command",
                            result,
                            timeout_s=self._setup_command_timeout_s,
                        )
                    )
        except BaseException:
            self.abandon(bound)
            raise
        return bound

    async def _target_workspace(
        self, context: SyncBindingContext
    ) -> tuple[Workspace, SyncTargetWorkspaceProvisioner | None]:
        target, provision = await super()._target_workspace(context)
        if self._git_baseline and not (
            isinstance(context.runner, BrokeredSandboxRunner)
            and isinstance(target, RunnerWorkspace)
            and target.is_bound_to_runner(context.runner)
            and ".git" in target.excluded_directory_names
        ):
            raise ValueError(
                "A brokered coding Git baseline requires a target RunnerWorkspace "
                "on the bound sandbox runner that excludes .git."
            )
        return target, provision

    async def finalize(
        self,
        bound: BoundWorkspace,
        *,
        outcome: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkspaceSnapshot | None:
        runner = bound.runner
        if not isinstance(runner, BrokeredSandboxRunner):
            raise ValueError("Brokered coding cleanup requires its BrokeredSandboxRunner.")
        try:
            snapshot = await super().finalize(bound, outcome=outcome, metadata=metadata)
        except BaseException:
            if runner.lease_keeper is not None:
                with contextlib.suppress(Exception):
                    await runner.lease_keeper.stop()
            raise
        try:
            await runner.close()
        except Exception as error:
            # Publication succeeded and renewal has stopped, so the broker
            # destroys the sandbox when its lease lapses.
            _LOGGER.warning(
                "Destroying brokered sandbox %s failed (%s); its lease will lapse.",
                runner.sandbox_id,
                type(error).__name__,
            )
        return snapshot

    def _completion_requires_successful_finalization(self, bound: BoundWorkspace) -> bool:
        # The sandbox belongs to this binding even when nothing is published.
        super()._completion_requires_successful_finalization(bound)
        return True

    async def _recover_completion_finalization(
        self,
        workspace: Workspace | None,
        runner: Runner | None,
        *,
        session_id: str,
        agent_name: str | None,
        environment_name: str | None,
        recovery_state: dict[str, Any],
    ) -> BoundWorkspace:
        """Refuse completion recovery: the bound sandbox can't be reattached.

        Recovery would run against a fresh, empty sandbox and publish its
        emptiness back to the source as deletions. The source keeps its last
        published state instead.
        """

        del workspace, runner, session_id, agent_name, environment_name, recovery_state
        raise RuntimeError(
            "Brokered sandboxes can't be reattached, so completion finalization can't be "
            "recovered; the source workspace keeps its last published state."
        )


class BrokeredSandboxCodingEnvironmentFactory(EnvironmentFactory):
    """Run coding tools in a per-session sandbox brokered by Cayu Cloud.

    ``source_workspace`` stays outside the sandbox and is the only durable
    copy, exactly as with ``SandboxCodingEnvironmentFactory``: it is copied
    into ``/workspace`` before the first tool call and receives changes back
    according to ``sync_back``. The sandbox copy excludes what the source
    excludes, ``sandbox_excluded_directory_names``, and the protected
    ``.git``, ``.cayu`` and ``.runtime`` directories. Copy-back is
    revision-aware: a source file changed outside the sandbox after copy-in
    fails finalization with a conflict.

    With ``git_baseline`` (the default) the sandbox's copy is committed to an
    ephemeral local Git repository right after copy-in, as with
    ``DockerCodingEnvironmentFactory``, so ``git_changes`` works in the
    sandbox. The image then needs ``git``; ``git_baseline=False`` drops the
    baseline for an image without it.

    The network is the broker's: ``egress="none"`` gives the sandbox no
    outbound network, ``egress="public_web"`` the open internet through Cloud's
    logging proxy, and ``egress="allowlist"`` with ``egress_hosts`` only the
    HTTPS hosts listed (1 to 50 DNS names, each exact or ``*.name``), through
    the same proxy. No credential ever enters the sandbox, and the proxy adds
    none. The choice is explicit; there is no default.

    ``none`` and ``allowlist`` sandboxes can be admitted for
    ``ExecutionRequirements.untrusted()`` (``network_access="deny_by_default"``)
    when their record carries the guarantee items behind it; ``public_web``
    can't. No mode is ``network_access="brokered_egress"``, which needs
    Runtime's own egress broker in the network path.

    Each session creates one sandbox under a fresh idempotency key and keeps
    its lease renewed at a third of ``lease_seconds``. The sandbox is destroyed
    when the session's binding finalizes, or by the factory's release on a
    failed setup. If the Agent process dies, nothing renews the lease and the
    broker destroys the sandbox within ``lease_seconds``. Each turn of a
    session gets a new sandbox, copied in from the source workspace; completion
    recovery after a crash is refused, so it never runs against an empty
    replacement.

    ``setup_commands`` run in the fresh sandbox before the workspace is copied
    in. ``workspace_setup_commands`` run after copy-in (and after the Git
    baseline) in ``/workspace``, for example to install the project's locked
    dependencies; anything they write under ``/workspace`` is published back
    like any other change, so write environments elsewhere (for example
    ``UV_PROJECT_ENVIRONMENT=/tmp/venv``). Each setup command may run for
    ``setup_command_timeout_s`` (default 600 s); a failure ends with a
    masked tail of the command's output.

    ``staged_copies`` chooses when the workspace moves through the sandbox's
    staged copies (see ``BrokeredSandboxWorkspace``): ``"auto"`` (the default)
    where needed, falling back to split archives when a staged copy-in fails;
    ``"never"``; or ``"always"``, which refuses a sandbox without them. Without
    ``transfer_limits`` the largest file follows each sandbox: what its staged
    copies carry where it offers them, otherwise one broker archive.

    ``toolchain_profile`` binds every sandbox to one neutral
    ``CodingToolchainProfile``, so ``run_check`` and ``run_command`` with that
    same profile work here as they do on Docker. It requires ``egress="none"``
    (profiles prohibit the network, an allowlist included) and no setup
    commands (profiles prohibit installing packages at run time), and refuses
    a ``DockerCodingToolchainProfile``:
    pass ``profile.coding_toolchain()``. Before allocation the profile's
    dependency inputs are verified in the source workspace. After allocation
    the factory requires the sandbox image to report the profile's
    ``trusted_build_context_sha256`` when the profile sets one, checks the
    read-only support paths exist (it can't prove they are read-only), runs the
    profile's admission probes and probes its executables live. The runner's
    evidence then names the profile, which the tools require.

    ``knowledge_store`` and ``knowledge_access_scope`` go to every session's
    environment, where ``search_knowledge`` and the other knowledge tools find
    them; the scope defaults to the one bound to the store. Without a store the
    sessions have no knowledge. Neither enters the sandbox: the tools run in
    the Agent process.
    """

    @property
    def secret_resolution_scope(self) -> Literal["static"]:
        return "static"

    def __init__(
        self,
        *,
        broker: SandboxBrokerClient,
        source_workspace: LocalWorkspace,
        egress: SandboxEgressMode,
        egress_hosts: Sequence[str] | None = None,
        image: str = DEFAULT_IMAGE,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
        labels: Mapping[str, str] | None = None,
        setup_commands: Sequence[str] = (),
        transfer_limits: SandboxWorkspaceTransferLimits | None = None,
        sync_back: Literal["always", "on_success", "never"] = "on_success",
        sandbox_excluded_directory_names: Sequence[str] = (),
        staging_capacity: SyncBindingStagingCapacity | None = None,
        ready_timeout_s: float = DEFAULT_READY_TIMEOUT_SECONDS,
        renew_every_s: float | None = None,
        git_baseline: bool = True,
        workspace_setup_commands: Sequence[str] = (),
        toolchain_profile: CodingToolchainProfile | None = None,
        setup_command_timeout_s: int = DEFAULT_BROKERED_SETUP_COMMAND_TIMEOUT_SECONDS,
        staged_copies: StagedCopiesMode = "auto",
        knowledge_store: KnowledgeStore | None = None,
        knowledge_access_scope: KnowledgeAccessScope | None = None,
    ) -> None:
        if not isinstance(broker, SandboxBrokerClient):
            raise TypeError("broker must be a SandboxBrokerClient.")
        # The concrete environment's validation, once, before any allocation.
        knowledge = Environment(
            EnvironmentSpec(name="coding"),
            knowledge_store=knowledge_store,
            knowledge_access_scope=knowledge_access_scope,
        )
        self._knowledge_store = knowledge.knowledge_store
        self._knowledge_access_scope = knowledge.knowledge_access_scope
        if not isinstance(source_workspace, LocalWorkspace):
            raise TypeError(
                "source_workspace must be a LocalWorkspace: durable storage outside the "
                "sandbox whose protected paths can be excluded before traversal."
            )
        egress, owned_hosts = validate_egress(egress, egress_hosts)
        if type(image) is not str or not image.strip():
            raise ValueError("image must name a sandbox catalog entry.")
        owned_staged_copies = validate_staged_copies_mode(staged_copies)
        # Without explicit limits, the largest file follows what each sandbox
        # can carry: its staged copies where it offers them, else one archive.
        self._max_file_bytes_follow_sandbox = transfer_limits is None
        if transfer_limits is None:
            transfer_limits = SandboxWorkspaceTransferLimits(max_file_bytes=MAX_TRANSFER_FILE_BYTES)
        elif type(transfer_limits) is not SandboxWorkspaceTransferLimits:
            raise TypeError("transfer_limits must be SandboxWorkspaceTransferLimits.")
        elif (
            owned_staged_copies == "never"
            and transfer_limits.max_file_bytes > MAX_TRANSFER_FILE_BYTES
        ):
            raise ValueError(
                f"transfer_limits.max_file_bytes={transfer_limits.max_file_bytes} needs staged "
                f"copies, which staged_copies='never' turns off; one broker archive carries "
                f"files of at most {MAX_TRANSFER_FILE_BYTES} bytes."
            )
        if type(ready_timeout_s) not in {int, float} or ready_timeout_s <= 0:
            raise ValueError("ready_timeout_s must be positive.")
        if type(git_baseline) is not bool:
            raise TypeError("git_baseline must be a bool.")
        owned_setup = _validated_commands(setup_commands, field_name="setup_commands")
        owned_workspace_setup = _validated_commands(
            workspace_setup_commands, field_name="workspace_setup_commands"
        )
        owned_profile = _brokered_toolchain_profile(
            toolchain_profile,
            egress=egress,
            setup_commands=owned_setup,
            workspace_setup_commands=owned_workspace_setup,
        )
        self.broker = broker
        self.source_workspace = source_workspace
        self.egress: SandboxEgressMode = egress
        #: The canonical hosts of an ``allowlist`` egress; ``None`` for the other modes.
        self.egress_hosts: tuple[str, ...] | None = owned_hosts
        self.image = image
        self.lease_seconds = validate_lease_seconds(lease_seconds)
        self.labels = validate_labels(labels or {})
        reserved = set(self.labels).intersection(_FACTORY_LABEL_KEYS)
        if reserved:
            raise ValueError(
                "Sandbox labels cayu.environment and cayu.session are set by the factory."
            )
        if len(self.labels) > _MAX_USER_LABELS:
            raise ValueError(
                f"The factory adds {len(_FACTORY_LABEL_KEYS)} labels of its own, so it accepts "
                f"at most {_MAX_USER_LABELS}."
            )
        self.setup_commands = owned_setup
        self.workspace_setup_commands = owned_workspace_setup
        self.toolchain_profile = owned_profile
        self.transfer_limits = transfer_limits
        self.sync_back = sync_back
        self.staging_capacity = staging_capacity
        self.ready_timeout_s = float(ready_timeout_s)
        self.renew_every_s = renew_every_s
        self.git_baseline = git_baseline
        self.setup_command_timeout_s = validate_setup_command_timeout(setup_command_timeout_s)
        self.staged_copies: StagedCopiesMode = owned_staged_copies
        names = {
            name.casefold(): name
            for name in (
                *source_workspace.excluded_directory_names,
                *sandbox_excluded_directory_names,
                *_PROTECTED_DIRECTORY_NAMES,
            )
        }
        self._excluded = tuple(names[key] for key in sorted(names))
        self._excluded_patterns = source_workspace.excluded_path_patterns
        self._protected_source = LocalWorkspace(
            source_workspace.root,
            workspace_id=source_workspace.id,
            excluded_directory_names=self._excluded,
            excluded_path_patterns=self._excluded_patterns,
        )
        self._profile_identity = ExecutionProfileBehaviorIdentity(
            name="cayu.brokered_sandbox_coding_environment",
            behavior_version="1",
            implementation_version=_configuration_fingerprint(self),
        )

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        """Identity of what every session's sandbox is built from; it never reconnects."""

        return self._profile_identity

    @property
    def configured_knowledge_store(self) -> KnowledgeStore | None:
        """The knowledge store every session's environment carries, for the knowledge tools."""

        return self._knowledge_store

    @property
    def configured_knowledge_access_scope(self) -> KnowledgeAccessScope | None:
        """A copy of the access scope that goes with ``configured_knowledge_store``."""

        from cayu.knowledge.scopes import copy_knowledge_access_scope

        scope = self._knowledge_access_scope
        return None if scope is None else copy_knowledge_access_scope(scope)

    @classmethod
    def from_env(
        cls,
        *,
        source_workspace: LocalWorkspace,
        environ: Mapping[str, str] | None = None,
        **options: Any,
    ) -> BrokeredSandboxCodingEnvironmentFactory:
        """Build from ``CAYU_CLOUD_*`` variables (see ``docs/environment-factories.md``)."""

        env = os.environ if environ is None else environ
        egress = env.get(BROKERED_SANDBOX_EGRESS_ENV, "").strip()
        modes: dict[str, SandboxEgressMode] = {
            "none": "none",
            "public_web": "public_web",
            "allowlist": "allowlist",
        }
        mode = modes.get(egress)
        if mode is None:
            raise ValueError(
                f"{BROKERED_SANDBOX_EGRESS_ENV} must be set to 'none', 'public_web' or 'allowlist'."
            )
        hosts = _egress_hosts_from_env(env)
        if mode == "allowlist" and hosts is None:
            raise ValueError(
                f"{BROKERED_SANDBOX_EGRESS_ENV}=allowlist needs {BROKERED_SANDBOX_EGRESS_HOSTS_ENV}."
            )
        if mode != "allowlist" and hosts is not None:
            raise ValueError(
                f"{BROKERED_SANDBOX_EGRESS_HOSTS_ENV} is only for "
                f"{BROKERED_SANDBOX_EGRESS_ENV}=allowlist."
            )
        lease_text = env.get(BROKERED_SANDBOX_LEASE_ENV, "").strip()
        try:
            lease_seconds = int(lease_text) if lease_text else DEFAULT_LEASE_SECONDS
        except ValueError:
            raise ValueError(f"{BROKERED_SANDBOX_LEASE_ENV} must be an integer.") from None
        return cls(
            broker=SandboxBrokerClient.from_env(env),
            source_workspace=source_workspace,
            egress=mode,
            egress_hosts=hosts,
            image=env.get(BROKERED_SANDBOX_IMAGE_ENV, "").strip() or DEFAULT_IMAGE,
            lease_seconds=lease_seconds,
            **options,
        )

    def construction_admission_candidate(self) -> ExecutionAdmissionCandidate:
        return self._declared_candidate(ExecutionRequirements.trusted())

    def execution_admission_candidate(
        self, request: EnvironmentFactoryRequest
    ) -> ExecutionAdmissionCandidate:
        """Declare, before allocation, what every brokered sandbox must then prove.

        The final admission uses the sandbox record's ``guarantees`` and live
        executable probes, so an untrusted workload is refused on a sandbox
        whose record lacks a required guarantee item.
        """

        if not isinstance(request, EnvironmentFactoryRequest):
            raise TypeError("Brokered admission requires an EnvironmentFactoryRequest.")
        return self._declared_candidate(request.execution_requirements)

    def _declared_candidate(
        self, requirements: ExecutionRequirements
    ) -> ExecutionAdmissionCandidate:
        probes = {probe.executable: probe.fingerprint for probe in requirements.executable_probes()}
        fingerprint = (
            "sha256:"
            + sha256(
                "\0".join(
                    (
                        "brokered-sandbox-configuration",
                        self.broker.settings.api_url,
                        self.broker.settings.application_id,
                        self.image,
                        self.egress,
                        # Only allowlist names hosts; the other modes keep their fingerprint.
                        *(() if self.egress_hosts is None else self.egress_hosts),
                    )
                ).encode()
            ).hexdigest()
        )
        return ExecutionAdmissionCandidate(
            candidate=BROKERED_SANDBOX_CANDIDATE,
            evidence=ExecutionCapabilityEvidence(
                subject=BROKERED_SANDBOX_CANDIDATE,
                claims=sandbox_capability_claims(None, egress_mode=self.egress, declared=True),
                environment_fingerprint=fingerprint,
                toolchain_profile_fingerprint=(
                    None if self.toolchain_profile is None else self.toolchain_profile.fingerprint
                ),
                tool_requirements=ExecutionToolRequirementEvidence(
                    environment_fingerprint=fingerprint,
                    executables=tuple(
                        ExecutionExecutableEvidence(
                            executable=name,
                            state="declared",
                            requirement_fingerprint=probes.get(name),
                        )
                        for name in requirements.executable_names()
                    ),
                ),
            ),
        )

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        if (
            request.operation is EnvironmentFactoryOperation.RECONNECT
            and request.reconnect_metadata
        ):
            # This factory never publishes reconnect metadata, so it can't be ours.
            raise RuntimeError(
                "Brokered sandboxes do not reconnect; the session must start a new sandbox."
            )
        # A RECONNECT without metadata is a follow-up turn: it gets a fresh
        # sandbox, which binding fills from the source. Completion recovery
        # against that sandbox is refused by the binding.
        profile = self.toolchain_profile
        if profile is not None:
            # Like Docker: a stale source fails before any sandbox is allocated.
            verify_local_coding_toolchain_dependencies(profile, self.source_workspace.root)
        sandbox = await self._allocate(request)
        keeper: SandboxLeaseKeeper | None = None
        runner: BrokeredSandboxRunner | None = None
        try:
            keeper = SandboxLeaseKeeper(
                self.broker,
                sandbox,
                lease_seconds=self.lease_seconds,
                renew_every_s=self.renew_every_s,
            ).start()
            runner = BrokeredSandboxRunner(
                self.broker,
                sandbox,
                lease_keeper=keeper,
                toolchain_profile_fingerprint=None if profile is None else profile.fingerprint,
                toolchain_executables=() if profile is None else profile.required_executables,
            )
            self._require_transfer_capacity(sandbox)
            if profile is not None:
                await _admit_toolchain(runner, profile)
            for command in self.setup_commands:
                result = await runner.exec(
                    ExecCommand.bash(command), timeout_s=self.setup_command_timeout_s
                )
                if result.exit_code != 0 or result.timed_out:
                    raise RuntimeError(
                        setup_command_failure_message(
                            "Brokered sandbox setup command",
                            result,
                            timeout_s=self.setup_command_timeout_s,
                        )
                    )
            environment = Environment(
                EnvironmentSpec(name=request.environment_name),
                workspace=self._protected_source,
                runner=runner,
                binding=self._binding(sandbox),
                knowledge_store=self._knowledge_store,
                knowledge_access_scope=self._knowledge_access_scope,
            )
        except BaseException as original:
            await self._discard(sandbox, keeper, runner, original)
            raise

        owned_runner = runner

        async def release(action: EnvironmentFactoryReleaseAction) -> None:
            # A brokered sandbox cannot be reattached, so every release destroys it.
            del action
            await owned_runner.close()

        metadata: dict[str, Any] = {
            "sandbox_id": sandbox.id,
            "sandbox_image": sandbox.image,
            "sandbox_egress": sandbox.egress.mode,
        }
        if sandbox.image_identity is not None:
            metadata["sandbox_image_fingerprint"] = sandbox.image_identity.fingerprint
        if profile is not None:
            metadata.update(profile.evidence())
        return EnvironmentFactoryResult(
            environment=environment,
            metadata=metadata,
            release=release,
        )

    def _require_transfer_capacity(self, sandbox: SandboxRecord) -> None:
        """Refuse a sandbox that can't move the largest file the limits allow.

        One archive carries a file of up to ``MAX_TRANSFER_FILE_BYTES``; a larger
        one moves only as a staged copy, which the install must offer and size.
        With ``staged_copies="always"`` the sandbox must offer staged copies.
        """

        staged = sandbox.staged_copies
        if self.staged_copies == "always" and staged is None:
            raise RuntimeError(
                "staged_copies='always', but this sandbox broker offers no staged copies."
            )
        limits = self.transfer_limits
        if self._max_file_bytes_follow_sandbox or limits.max_file_bytes <= MAX_TRANSFER_FILE_BYTES:
            return
        needed = limits.max_file_bytes + STAGED_FILE_OVERHEAD_BYTES
        if staged is None or staged.max_bytes < needed:
            raise RuntimeError(
                f"transfer_limits.max_file_bytes={limits.max_file_bytes} needs staged copies of "
                f"at least {needed} bytes, which this sandbox broker doesn't offer; one broker "
                f"archive carries files of at most {MAX_TRANSFER_FILE_BYTES} bytes."
            )

    async def _allocate(self, request: EnvironmentFactoryRequest) -> SandboxRecord:
        scope = sha256(f"{request.session_id}\0{request.environment_name}".encode()).hexdigest()[
            :32
        ]
        # Fresh per allocation: the key only makes this create's retries idempotent.
        idempotency_key = f"cayu-{scope}-{secrets.token_hex(8)}"
        labels = {
            **self.labels,
            "cayu.environment": request.environment_name[:200],
            "cayu.session": scope,
        }
        sandbox = await self.broker.create(
            idempotency_key=idempotency_key,
            image=self.image,
            egress_mode=self.egress,
            egress_hosts=self.egress_hosts,
            lease_seconds=self.lease_seconds,
            labels=labels,
        )
        try:
            return await self.broker.wait_until_ready(sandbox.id, timeout_s=self.ready_timeout_s)
        except BaseException as original:
            await self._discard(sandbox, None, None, original)
            raise

    async def _discard(
        self,
        sandbox: SandboxRecord,
        keeper: SandboxLeaseKeeper | None,
        runner: BrokeredSandboxRunner | None,
        original: BaseException,
    ) -> None:
        try:
            if runner is not None:
                await runner.close()
            else:
                if keeper is not None:
                    await keeper.stop()
                await self.broker.destroy(sandbox.id)
        except (Exception, asyncio.CancelledError) as cleanup_error:
            original.add_note(
                f"Destroying sandbox {sandbox.id} also failed ({type(cleanup_error).__name__}); "
                "the broker destroys it when its lease lapses."
            )

    def _max_file_bytes(self, sandbox: SandboxRecord) -> int:
        """The largest file one session moves through this sandbox."""

        limits = self.transfer_limits
        staged = sandbox.staged_copies
        if (
            not self._max_file_bytes_follow_sandbox
            or staged is None
            or self.staged_copies == "never"
        ):
            return limits.max_file_bytes
        # A file moves alone as one staged copy, beside its tar framing.
        return max(
            limits.max_file_bytes,
            min(staged.max_bytes - STAGED_FILE_OVERHEAD_BYTES, limits.max_total_bytes),
        )

    def _binding(self, sandbox: SandboxRecord) -> BrokeredSandboxCodingBinding:
        limits = self.transfer_limits
        excluded = self._excluded
        excluded_patterns = self._excluded_patterns
        staged_copies = self.staged_copies

        def target_plan(context: SyncBindingContext) -> SyncTargetWorkspacePlan:
            runner = context.runner
            if not isinstance(runner, BrokeredSandboxRunner) or context.session_id is None:
                raise RuntimeError("A brokered coding target needs the session's sandbox runner.")
            return SyncTargetWorkspacePlan(
                workspace=BrokeredSandboxWorkspace(
                    runner,
                    staged_copies=staged_copies,
                    workspace_id=f"brokered-sandbox:{context.session_id}",
                    excluded_directory_names=excluded,
                    excluded_path_patterns=excluded_patterns,
                )
            )

        return BrokeredSandboxCodingBinding(
            target_workspace_plan_factory=target_plan,
            max_files=limits.max_files,
            max_file_bytes=self._max_file_bytes(sandbox),
            max_total_bytes=limits.max_total_bytes,
            max_archive_bytes=limits.max_archive_bytes,
            clean_target="always",
            sync_back=self.sync_back,
            delete_missing=True,
            source_conflict_policy="require_revision",
            preserve_git_modes=True,
            staging_capacity=self.staging_capacity,
            git_baseline=self.git_baseline,
            workspace_setup_commands=self.workspace_setup_commands,
            setup_command_timeout_s=self.setup_command_timeout_s,
        )


def _egress_hosts_from_env(env: Mapping[str, str]) -> tuple[str, ...] | None:
    hosts = tuple(env.get(BROKERED_SANDBOX_EGRESS_HOSTS_ENV, "").replace(",", " ").split())
    return hosts or None


def _validated_commands(commands: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    if isinstance(commands, str | bytes):
        raise TypeError(f"{field_name} must be a sequence of shell commands, not one string.")
    owned = tuple(commands)
    if any(type(command) is not str or not command.strip() for command in owned):
        raise ValueError(f"{field_name} entries must be nonblank shell commands.")
    return owned


def _brokered_toolchain_profile(
    profile: object,
    *,
    egress: str,
    setup_commands: tuple[str, ...],
    workspace_setup_commands: tuple[str, ...],
) -> CodingToolchainProfile | None:
    if profile is None:
        return None
    from cayu.environments.docker_toolchains import DockerCodingToolchainProfile

    if type(profile) is DockerCodingToolchainProfile:
        raise TypeError(
            "A brokered sandbox can't verify a Docker image; pass the neutral profile "
            "(profile.coding_toolchain()) instead of a DockerCodingToolchainProfile."
        )
    owned = owned_coding_toolchain_profile(profile)
    if egress != "none":
        # An allowlist is still network: a profile's checks must not depend on any host.
        raise ValueError(
            "A toolchain profile prohibits the network, so a brokered sandbox with one needs "
            "egress='none'."
        )
    if setup_commands or workspace_setup_commands:
        raise ValueError(
            "A toolchain profile prohibits installing packages at run time, so a brokered "
            "sandbox with one takes no setup_commands or workspace_setup_commands; build "
            "the dependencies into the sandbox image instead."
        )
    if owned.workspace_path != _TOOLCHAIN_WORKSPACE:
        raise ValueError(
            f"A brokered sandbox's workspace is {_TOOLCHAIN_WORKSPACE}; the toolchain "
            f"profile names {owned.workspace_path}."
        )
    return owned


async def _admit_toolchain(runner: BrokeredSandboxRunner, profile: CodingToolchainProfile) -> None:
    """Bind one allocated sandbox to the profile before anything is copied in."""

    expected_context = profile.trusted_build_context_sha256
    identity = runner.sandbox.image_identity
    if expected_context is not None and (
        identity is None or identity.trusted_build_context_sha256 != expected_context
    ):
        raise CodingToolchainError(
            "toolchain_image_unverified",
            "The brokered sandbox image does not report the trusted build context this "
            "toolchain profile was built from. Catalog images don't carry an application's "
            "dependencies; use an image the broker built from the same coding image inputs "
            "(for example the Release's own image).",
        )
    await run_coding_toolchain_admission_probes(runner, profile)
    try:
        await runner.refresh_execution_admission()
    except SandboxBrokerError:
        raise CodingToolchainError(
            "toolchain_executables_unverified",
            "The brokered sandbox's toolchain executables could not be probed.",
        ) from None
    evidence = runner.execution_admission_candidate().evidence.tool_requirements
    unavailable = tuple(
        executable
        for executable in profile.required_executables
        if evidence is None
        or (claim := evidence.executable_for(executable)) is None
        or claim.state != "live_verified"
    )
    if unavailable:
        raise CodingToolchainError(
            "toolchain_executables_unavailable",
            f"{len(unavailable)} of the toolchain profile's {len(profile.required_executables)} "
            "executables are missing in the brokered sandbox image.",
            paths=unavailable,
        )


def _configuration_fingerprint(factory: BrokeredSandboxCodingEnvironmentFactory) -> str:
    limits = factory.transfer_limits
    material = {
        "schema": "cayu.brokered_sandbox_coding_environment.v1",
        "image": factory.image,
        "egress": factory.egress,
        "sync_back": factory.sync_back,
        "git_baseline": factory.git_baseline,
        "staged_copies": factory.staged_copies,
        "transfer_limits": {
            "max_files": limits.max_files,
            # Without explicit limits it follows each sandbox's staged copies.
            "max_file_bytes": (
                "sandbox" if factory._max_file_bytes_follow_sandbox else limits.max_file_bytes
            ),
            "max_total_bytes": limits.max_total_bytes,
            "max_archive_bytes": limits.max_archive_bytes,
        },
        "excluded_directory_names": list(factory._excluded),
        "excluded_path_patterns": list(factory._excluded_patterns),
        "setup_commands_sha256": sha256(
            "\0".join(factory.setup_commands).encode("utf-8")
        ).hexdigest(),
        "workspace_setup_commands_sha256": sha256(
            "\0".join(factory.workspace_setup_commands).encode("utf-8")
        ).hexdigest(),
        "toolchain_profile_fingerprint": (
            None if factory.toolchain_profile is None else factory.toolchain_profile.fingerprint
        ),
    }
    if factory.egress_hosts is not None:
        # Only allowlist names hosts; the other modes keep their earlier fingerprint.
        material["egress_hosts"] = list(factory.egress_hosts)
    return (
        "sha256:"
        + sha256(
            canonical_durable_json_bytes(material, "brokered_sandbox_coding_configuration")
        ).hexdigest()
    )


@dataclass(frozen=True)
class BrokeredCodingBackendOptions:
    """How the ``cayu-cloud`` coding backend asks Cayu Cloud's broker for sandboxes.

    Sandboxes always run with ``egress="none"``: toolchain profiles prohibit
    the network, so an allowlist isn't offered here. ``image`` names the
    catalog entry, or the image a broker builds for the application (for
    example ``"release"``). ``broker`` reuses
    an existing client; otherwise one is made from the ``CAYU_CLOUD_*``
    variables and shared by every workspace the backend serves.
    """

    image: str = DEFAULT_IMAGE
    lease_seconds: int = DEFAULT_LEASE_SECONDS
    transfer_limits: SandboxWorkspaceTransferLimits | None = None
    broker: SandboxBrokerClient | None = None
    labels: Mapping[str, str] | None = None
    git_baseline: bool = True
    staged_copies: StagedCopiesMode = "auto"

    def __post_init__(self) -> None:
        if type(self.image) is not str or not self.image.strip():
            raise ValueError("image must name a sandbox catalog entry.")
        validate_lease_seconds(self.lease_seconds)
        if self.broker is not None and not isinstance(self.broker, SandboxBrokerClient):
            raise TypeError("broker must be a SandboxBrokerClient or None.")
        if type(self.git_baseline) is not bool:
            raise TypeError("git_baseline must be a bool.")
        validate_staged_copies_mode(self.staged_copies)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> BrokeredCodingBackendOptions:
        """Read ``CAYU_CLOUD_SANDBOX_IMAGE`` and ``CAYU_CLOUD_SANDBOX_LEASE_SECONDS``.

        ``CAYU_CLOUD_SANDBOX_EGRESS``, when set, must be ``none``, and
        ``CAYU_CLOUD_SANDBOX_EGRESS_HOSTS`` must be unset: the toolchain prohibits
        the network, an allowlist included.
        """

        env = os.environ if environ is None else environ
        egress = env.get(BROKERED_SANDBOX_EGRESS_ENV, "").strip()
        if egress not in ("", "none"):
            raise ValueError(
                f"{BROKERED_SANDBOX_EGRESS_ENV}={egress} conflicts with the coding toolchain, "
                "which prohibits the network; set it to 'none' or leave it unset."
            )
        if _egress_hosts_from_env(env) is not None:
            raise ValueError(
                f"{BROKERED_SANDBOX_EGRESS_HOSTS_ENV} conflicts with the coding toolchain, "
                "which prohibits the network; leave it unset."
            )
        lease_text = env.get(BROKERED_SANDBOX_LEASE_ENV, "").strip()
        try:
            lease_seconds = int(lease_text) if lease_text else DEFAULT_LEASE_SECONDS
        except ValueError:
            raise ValueError(f"{BROKERED_SANDBOX_LEASE_ENV} must be an integer.") from None
        return cls(
            image=env.get(BROKERED_SANDBOX_IMAGE_ENV, "").strip() or DEFAULT_IMAGE,
            lease_seconds=lease_seconds,
        )


def brokered_coding_backend(context: Any) -> Any:
    """Build the ``cayu-cloud`` coding backend from a ``CodingBackendContext``."""

    from cayu.environments.coding_backends import (
        CLOUD_CODING_BACKEND,
        CodingBackend,
        CodingBackendContext,
    )

    if type(context) is not CodingBackendContext:
        raise TypeError("context must be a CodingBackendContext.")
    options = context.options
    if options is None:
        options = BrokeredCodingBackendOptions.from_env(context.environ)
    if type(options) is not BrokeredCodingBackendOptions:
        raise TypeError("options['cayu-cloud'] must be BrokeredCodingBackendOptions or omitted.")
    profile = context.toolchain_profile
    # Refuse a profile the broker can't host before any workspace is registered.
    _brokered_toolchain_profile(
        profile, egress="none", setup_commands=(), workspace_setup_commands=()
    )
    # One client for every workspace of this process: it holds the broker's connections.
    broker = options.broker or SandboxBrokerClient.from_env(context.environ)

    def build_factory(
        source_workspace: LocalWorkspace,
        *,
        artifact_store: Any = None,
        knowledge_store: KnowledgeStore | None = None,
        knowledge_access_scope: KnowledgeAccessScope | None = None,
    ) -> BrokeredSandboxCodingEnvironmentFactory:
        # The artifact store is registered with the app; sandboxes never hold it.
        del artifact_store
        return BrokeredSandboxCodingEnvironmentFactory(
            broker=broker,
            source_workspace=source_workspace,
            egress="none",
            image=options.image,
            lease_seconds=options.lease_seconds,
            labels=options.labels,
            transfer_limits=options.transfer_limits,
            git_baseline=options.git_baseline,
            toolchain_profile=profile,
            staged_copies=options.staged_copies,
            knowledge_store=knowledge_store,
            knowledge_access_scope=knowledge_access_scope,
        )

    return CodingBackend(
        name=CLOUD_CODING_BACKEND,
        candidate=BROKERED_SANDBOX_CANDIDATE,
        toolchain_profile=profile,
        build_factory=build_factory,
        metadata={
            "execution_kind": "trusted_repository_brokered_sandbox",
            "sandbox_image": options.image,
        },
    )
