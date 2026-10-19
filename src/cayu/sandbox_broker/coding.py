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
``BrokeredSandboxWorkspace``) over the sandbox. The broker client, lease
keeper and runner are each usable on their own.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import secrets
from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import Any, Literal

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
    WorkspaceSnapshot,
)
from cayu.runners import ExecCommand
from cayu.runners.base import Runner
from cayu.sandbox_broker.client import (
    SandboxBrokerClient,
    SandboxEgressMode,
    SandboxRecord,
    validate_labels,
    validate_lease_seconds,
)
from cayu.sandbox_broker.lease import SandboxLeaseKeeper
from cayu.sandbox_broker.runner import (
    BROKERED_SANDBOX_CANDIDATE,
    MAX_TRANSFER_FILE_BYTES,
    BrokeredSandboxRunner,
    BrokeredSandboxWorkspace,
    sandbox_capability_claims,
)
from cayu.workspaces import LocalWorkspace, Workspace

__all__ = [
    "BROKERED_SANDBOX_EGRESS_ENV",
    "BROKERED_SANDBOX_IMAGE_ENV",
    "BROKERED_SANDBOX_LEASE_ENV",
    "BrokeredSandboxCodingBinding",
    "BrokeredSandboxCodingEnvironmentFactory",
]

#: Catalog image for coding sandboxes (default ``coding-python``).
BROKERED_SANDBOX_IMAGE_ENV = "CAYU_CLOUD_SANDBOX_IMAGE"
#: ``none`` or ``public_web``; required by ``from_env``, since there is no default network.
BROKERED_SANDBOX_EGRESS_ENV = "CAYU_CLOUD_SANDBOX_EGRESS"
#: Lease length in seconds, 60 to 3600 (default 900).
BROKERED_SANDBOX_LEASE_ENV = "CAYU_CLOUD_SANDBOX_LEASE_SECONDS"

_LOGGER = logging.getLogger(__name__)
DEFAULT_IMAGE = "coding-python"
DEFAULT_LEASE_SECONDS = 900
# Cloud fails a sandbox still provisioning after five minutes.
DEFAULT_READY_TIMEOUT_SECONDS = 330.0
_PROTECTED_DIRECTORY_NAMES = (".cayu", ".git", ".runtime")
_FACTORY_LABEL_KEYS = frozenset({"cayu.environment", "cayu.session"})
_MAX_USER_LABELS = 10 - len(_FACTORY_LABEL_KEYS)
_SETUP_COMMAND_TIMEOUT_SECONDS = 600


class BrokeredSandboxCodingBinding(SyncBinding):
    """``SyncBinding`` that destroys the brokered sandbox once finalization succeeds.

    When copy-back fails, for example on a source conflict, the binding keeps
    the sandbox for a retry but stops renewing its lease, so the broker
    destroys it when the lease lapses if no retry comes.
    """

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

    The network is the broker's: ``egress="none"`` gives the sandbox no
    outbound network, ``egress="public_web"`` the open internet through Cloud's
    logging proxy. No credential ever enters the sandbox. The choice is
    explicit; there is no default.

    Each session creates one sandbox under a fresh idempotency key and keeps
    its lease renewed at a third of ``lease_seconds``. The sandbox is destroyed
    when the session's binding finalizes, or by the factory's release on a
    failed setup. If the Agent process dies, nothing renews the lease and the
    broker destroys the sandbox within ``lease_seconds``. Each turn of a
    session gets a new sandbox, copied in from the source workspace; completion
    recovery after a crash is refused, so it never runs against an empty
    replacement.
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
    ) -> None:
        if not isinstance(broker, SandboxBrokerClient):
            raise TypeError("broker must be a SandboxBrokerClient.")
        if not isinstance(source_workspace, LocalWorkspace):
            raise TypeError(
                "source_workspace must be a LocalWorkspace: durable storage outside the "
                "sandbox whose protected paths can be excluded before traversal."
            )
        if egress not in ("none", "public_web"):
            raise ValueError("egress must be 'none' or 'public_web'.")
        if type(image) is not str or not image.strip():
            raise ValueError("image must name a sandbox catalog entry.")
        if transfer_limits is None:
            transfer_limits = SandboxWorkspaceTransferLimits(max_file_bytes=MAX_TRANSFER_FILE_BYTES)
        elif type(transfer_limits) is not SandboxWorkspaceTransferLimits:
            raise TypeError("transfer_limits must be SandboxWorkspaceTransferLimits.")
        if transfer_limits.max_file_bytes > MAX_TRANSFER_FILE_BYTES:
            # Larger trees are split across archives, but one file must fit in one.
            raise ValueError(
                "transfer_limits.max_file_bytes cannot exceed "
                f"{MAX_TRANSFER_FILE_BYTES} bytes, the largest file one broker archive carries."
            )
        if type(ready_timeout_s) not in {int, float} or ready_timeout_s <= 0:
            raise ValueError("ready_timeout_s must be positive.")
        self.broker = broker
        self.source_workspace = source_workspace
        self.egress: SandboxEgressMode = egress
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
        self.setup_commands = tuple(setup_commands)
        self.transfer_limits = transfer_limits
        self.sync_back = sync_back
        self.staging_capacity = staging_capacity
        self.ready_timeout_s = float(ready_timeout_s)
        self.renew_every_s = renew_every_s
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
        if egress not in ("none", "public_web"):
            raise ValueError(
                f"{BROKERED_SANDBOX_EGRESS_ENV} must be set to 'none' or 'public_web'."
            )
        lease_text = env.get(BROKERED_SANDBOX_LEASE_ENV, "").strip()
        try:
            lease_seconds = int(lease_text) if lease_text else DEFAULT_LEASE_SECONDS
        except ValueError:
            raise ValueError(f"{BROKERED_SANDBOX_LEASE_ENV} must be an integer.") from None
        return cls(
            broker=SandboxBrokerClient.from_env(env),
            source_workspace=source_workspace,
            egress="none" if egress == "none" else "public_web",
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
            runner = BrokeredSandboxRunner(self.broker, sandbox, lease_keeper=keeper)
            for command in self.setup_commands:
                result = await runner.exec(
                    ExecCommand.bash(command), timeout_s=_SETUP_COMMAND_TIMEOUT_SECONDS
                )
                if result.exit_code != 0 or result.timed_out:
                    raise RuntimeError(
                        f"Brokered sandbox setup command failed with exit code {result.exit_code}."
                    )
            environment = Environment(
                EnvironmentSpec(name=request.environment_name),
                workspace=self._protected_source,
                runner=runner,
                binding=self._binding(),
            )
        except BaseException as original:
            await self._discard(sandbox, keeper, runner, original)
            raise

        owned_runner = runner

        async def release(action: EnvironmentFactoryReleaseAction) -> None:
            # A brokered sandbox cannot be reattached, so every release destroys it.
            del action
            await owned_runner.close()

        return EnvironmentFactoryResult(
            environment=environment,
            metadata={
                "sandbox_id": sandbox.id,
                "sandbox_image": sandbox.image,
                "sandbox_egress": sandbox.egress.mode,
            },
            release=release,
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

    def _binding(self) -> BrokeredSandboxCodingBinding:
        limits = self.transfer_limits
        excluded = self._excluded
        excluded_patterns = self._excluded_patterns

        def target_plan(context: SyncBindingContext) -> SyncTargetWorkspacePlan:
            runner = context.runner
            if not isinstance(runner, BrokeredSandboxRunner) or context.session_id is None:
                raise RuntimeError("A brokered coding target needs the session's sandbox runner.")
            return SyncTargetWorkspacePlan(
                workspace=BrokeredSandboxWorkspace(
                    runner,
                    workspace_id=f"brokered-sandbox:{context.session_id}",
                    excluded_directory_names=excluded,
                    excluded_path_patterns=excluded_patterns,
                )
            )

        return BrokeredSandboxCodingBinding(
            target_workspace_plan_factory=target_plan,
            max_files=limits.max_files,
            max_file_bytes=limits.max_file_bytes,
            max_total_bytes=limits.max_total_bytes,
            max_archive_bytes=limits.max_archive_bytes,
            clean_target="always",
            sync_back=self.sync_back,
            delete_missing=True,
            source_conflict_policy="require_revision",
            preserve_git_modes=True,
            staging_capacity=self.staging_capacity,
        )
