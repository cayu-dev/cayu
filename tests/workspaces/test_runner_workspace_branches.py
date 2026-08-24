from __future__ import annotations

import asyncio
import hashlib
import json
import stat
import sys
from pathlib import Path

import pytest
from tests.workspaces.branch_conformance import (
    verify_atomic_publication,
    verify_bound_rollback_and_cleanup,
    verify_branch_isolation_and_net_changes,
    verify_conflict_is_all_or_none,
)

from cayu.runners import (
    ExecResult,
    LocalRunner,
    RemoteWorkspaceBranchCapability,
    attach_cancellation_artifacts,
)
from cayu.workspaces import (
    LocalWorkspace,
    RunnerWorkspace,
    WorkspaceBranchAuthority,
    WorkspaceBranchBindingAuthority,
    WorkspaceBranchBindingAuthorityClaim,
    WorkspaceBranchBindingAuthorityClaimScope,
    WorkspaceBranchBindingAuthorityProvider,
    WorkspaceBranchBindingAuthorityRegistry,
    WorkspaceBranchClosedError,
    WorkspaceBranchFencedError,
    WorkspaceBranchLifecycleStatus,
    WorkspaceBranchLimits,
    WorkspaceBranchOperationConflict,
    WorkspaceBranchOutcomeStatus,
    WorkspaceBranchPublicationRequest,
    WorkspaceBranchRecoveryRequest,
    WorkspaceBranchRecoveryStrength,
    WorkspaceBranchRequest,
    WorkspaceBranchResourceExhaustedError,
    WorkspaceBranchRetentionStrength,
    WorkspaceBranchRollbackRequest,
)
from cayu.workspaces.revisions import (
    WorkspaceRevisionObservationLimits,
    observe_deterministic_workspace,
)


class _DeterministicRemoteBranchCapability(RemoteWorkspaceBranchCapability):
    def __init__(self, runner: _BranchRunner) -> None:
        self._runner = runner

    @property
    def resource_key(self) -> tuple[object, ...]:
        return ("deterministic-remote", str(self._runner.root))

    @property
    def allocation_fingerprint(self) -> str:
        encoded = str(self._runner.root).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()


class _BranchRunner(LocalRunner):
    isolation = "deterministic-remote"

    def workspace_capability(self, capability_type):
        if capability_type is RemoteWorkspaceBranchCapability:
            return _DeterministicRemoteBranchCapability(self)
        return super().workspace_capability(capability_type)


class _UnprovenRunnerWorkspace(RunnerWorkspace):
    pass


class _DurableTestAuthorityProvider(WorkspaceBranchBindingAuthorityRegistry):
    @property
    def claim_scope(self) -> WorkspaceBranchBindingAuthorityClaimScope:
        return WorkspaceBranchBindingAuthorityClaimScope.DURABLE


class _BindingOnlyDurableAuthorityProvider:
    def __init__(self, authority: WorkspaceBranchBindingAuthority) -> None:
        self._registry = WorkspaceBranchBindingAuthorityRegistry(authority)

    @property
    def claim_scope(self) -> WorkspaceBranchBindingAuthorityClaimScope:
        return WorkspaceBranchBindingAuthorityClaimScope.DURABLE

    def __call__(self) -> WorkspaceBranchBindingAuthority:
        return self._registry()

    def claim(
        self,
        expected: WorkspaceBranchBindingAuthority,
    ) -> WorkspaceBranchBindingAuthorityClaim:
        return self._registry.claim(expected)


class _FailOnceReleaseClaim:
    def __init__(self, inner: WorkspaceBranchBindingAuthorityClaim) -> None:
        self._inner = inner
        self._attempts = 0

    def release(self) -> None:
        self._attempts += 1
        if self._attempts == 1:
            raise RuntimeError("injected claim release failure")
        self._inner.release()


class _FailOnceReleaseAuthorityProvider(_DurableTestAuthorityProvider):
    def __init__(self, authority: WorkspaceBranchBindingAuthority) -> None:
        super().__init__(authority)
        self._wrap_next_claim = True

    def claim(
        self,
        expected: WorkspaceBranchBindingAuthority,
    ) -> WorkspaceBranchBindingAuthorityClaim:
        claim = super().claim(expected)
        return self._wrap_claim(claim)

    def claim_operation(
        self,
        expected: WorkspaceBranchAuthority,
    ) -> WorkspaceBranchBindingAuthorityClaim:
        claim = super().claim_operation(expected)
        return self._wrap_claim(claim)

    def _wrap_claim(
        self,
        claim: WorkspaceBranchBindingAuthorityClaim,
    ) -> WorkspaceBranchBindingAuthorityClaim:
        if not self._wrap_next_claim:
            return claim
        self._wrap_next_claim = False
        return _FailOnceReleaseClaim(claim)


class _TrackingReleaseClaim:
    def __init__(self, inner: WorkspaceBranchBindingAuthorityClaim) -> None:
        self._inner = inner
        self.release_calls = 0

    def release(self) -> None:
        self.release_calls += 1
        self._inner.release()


class _TrackNextReleaseAuthorityProvider(_DurableTestAuthorityProvider):
    def __init__(self, authority: WorkspaceBranchBindingAuthority) -> None:
        super().__init__(authority)
        self.track_next_claim = False
        self.tracked_claim: _TrackingReleaseClaim | None = None

    def claim_operation(
        self,
        expected: WorkspaceBranchAuthority,
    ) -> WorkspaceBranchBindingAuthorityClaim:
        claim = super().claim_operation(expected)
        if not self.track_next_claim:
            return claim
        self.track_next_claim = False
        tracked = _TrackingReleaseClaim(claim)
        self.tracked_claim = tracked
        return tracked


class _SecretBearingWrongType:
    def __init__(self, canary: str) -> None:
        self._canary = canary

    def __repr__(self) -> str:
        return self._canary

    def __str__(self) -> str:
        return self._canary


def _workspace(
    root: Path,
    *,
    resolver: WorkspaceBranchBindingAuthorityProvider | None = None,
) -> RunnerWorkspace:
    return RunnerWorkspace(
        _BranchRunner(root, inherit_env=False),
        workspace_id="remote-source",
        python_executable=sys.executable,
        enable_workspace_branches=True,
        branch_authority_resolver=resolver,
    )


def _durable_workspace(
    root: Path,
    *,
    identity: str,
) -> tuple[RunnerWorkspace, WorkspaceBranchAuthority]:
    binding = WorkspaceBranchBindingAuthority(
        environment_name="remote-env",
        binding_generation=f"{identity}-generation",
        binding_identity=f"{identity}-allocation",
    )
    authority = WorkspaceBranchAuthority(
        session_id=f"{identity}-session",
        expected_run_epoch=5,
        environment_name=binding.environment_name,
        binding_generation=binding.binding_generation,
        binding_identity=binding.binding_identity,
        creating_authority=f"{identity}-worker",
        resource_policy="remote-branch-defaults",
    )
    return _workspace(root, resolver=_DurableTestAuthorityProvider(binding)), authority


async def _request(
    workspace: RunnerWorkspace,
    *,
    limits: WorkspaceBranchLimits | None = None,
    authority: WorkspaceBranchAuthority | None = None,
    branch_id: str | None = None,
    idempotency_key: str | None = None,
) -> WorkspaceBranchRequest:
    resolver = workspace._branch_authority_resolver
    if authority is not None and isinstance(
        resolver,
        WorkspaceBranchBindingAuthorityRegistry,
    ):
        resolver.authorize_operation(authority)
    observation = await observe_deterministic_workspace(
        workspace,
        observer="runner-branch-tests",
        limits=WorkspaceRevisionObservationLimits(),
    )
    return WorkspaceBranchRequest(
        baseline=observation,
        limits=limits or WorkspaceBranchLimits(),
        authority=authority,
        branch_id=branch_id,
        idempotency_key=idempotency_key,
    )


def _populated_root(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir(parents=True)
    (root / "original.txt").write_bytes(b"original")
    (root / "deleted.txt").write_bytes(b"delete-me")
    return root


def _private_branch_directory(root: Path) -> Path:
    matches = tuple(
        path for path in root.parent.glob(".cayu-workspace-branch-*.stage") if path.is_dir()
    )
    assert len(matches) == 1
    return matches[0]


def test_workspace_branch_capabilities_default_unsupported_and_are_instance_explicit(
    tmp_path: Path,
) -> None:
    local = LocalWorkspace(tmp_path)
    unsupported = RunnerWorkspace(
        LocalRunner(tmp_path, inherit_env=False),
        python_executable=sys.executable,
    )
    supported = _workspace(tmp_path)
    unproven = _UnprovenRunnerWorkspace(
        _BranchRunner(tmp_path, inherit_env=False),
        python_executable=sys.executable,
        enable_workspace_branches=True,
    )
    process_scoped = _workspace(
        tmp_path,
        resolver=WorkspaceBranchBindingAuthorityRegistry(
            WorkspaceBranchBindingAuthority(
                environment_name="remote-env",
                binding_generation="generation-1",
                binding_identity="allocation-binding-1",
            )
        ),
    )
    binding_only = _workspace(
        tmp_path,
        resolver=_BindingOnlyDurableAuthorityProvider(
            WorkspaceBranchBindingAuthority(
                environment_name="remote-env",
                binding_generation="generation-binding-only",
                binding_identity="allocation-binding-only",
            )
        ),
    )

    assert local.branch_capabilities().isolation is True
    assert unsupported.branch_capabilities().isolation is False
    assert supported.branch_capabilities().isolation is True
    assert unproven.branch_capabilities().isolation is False
    assert supported.branch_capabilities().recovery is WorkspaceBranchRecoveryStrength.PROCESS_LOCAL
    assert (
        supported.branch_capabilities().retention is WorkspaceBranchRetentionStrength.PROCESS_LOCAL
    )
    assert (
        process_scoped.branch_capabilities().recovery
        is WorkspaceBranchRecoveryStrength.PROCESS_LOCAL
    )
    assert (
        binding_only.branch_capabilities().recovery is WorkspaceBranchRecoveryStrength.PROCESS_LOCAL
    )

    async def unproven_scenario() -> None:
        request = await _request(unproven)
        result = await unproven.create_branch(request)
        assert result.status is WorkspaceBranchOutcomeStatus.UNSUPPORTED

    asyncio.run(unproven_scenario())


def test_process_local_binding_registry_cannot_authorize_durable_remote_branch(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        root = _populated_root(tmp_path)
        binding = WorkspaceBranchBindingAuthority(
            environment_name="remote-env",
            binding_generation="generation-1",
            binding_identity="allocation-binding-1",
        )
        source = _workspace(
            root,
            resolver=WorkspaceBranchBindingAuthorityRegistry(binding),
        )
        authority = WorkspaceBranchAuthority(
            session_id="session-process-local",
            expected_run_epoch=1,
            environment_name=binding.environment_name,
            binding_generation=binding.binding_generation,
            binding_identity=binding.binding_identity,
            creating_authority="worker-1",
            resource_policy="remote-branch-defaults",
        )

        with pytest.raises(RuntimeError, match="cross-process invocation authority claims"):
            await source.create_branch(
                await _request(
                    source,
                    authority=authority,
                    branch_id="must-not-be-durable",
                    idempotency_key="create-1",
                )
            )

    asyncio.run(scenario())


def test_binding_only_durable_provider_cannot_authorize_remote_recovery(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        root = _populated_root(tmp_path)
        binding = WorkspaceBranchBindingAuthority(
            environment_name="remote-env",
            binding_generation="binding-only-generation",
            binding_identity="binding-only-allocation",
        )
        source = _workspace(
            root,
            resolver=_BindingOnlyDurableAuthorityProvider(binding),
        )
        authority = WorkspaceBranchAuthority(
            session_id="binding-only-session",
            expected_run_epoch=1,
            environment_name=binding.environment_name,
            binding_generation=binding.binding_generation,
            binding_identity=binding.binding_identity,
            creating_authority="binding-only-worker",
            resource_policy="remote-branch-defaults",
        )

        with pytest.raises(RuntimeError, match="cross-process invocation authority claims"):
            await source.create_branch(
                await _request(
                    source,
                    authority=authority,
                    branch_id="binding-only-branch",
                    idempotency_key="binding-only-create",
                )
            )

    asyncio.run(scenario())


def test_runner_workspace_branches_share_local_conformance(tmp_path: Path) -> None:
    async def scenario() -> None:
        source = _workspace(_populated_root(tmp_path))
        request = await _request(source)
        first_result = await source.create_branch(request)
        second_result = await source.create_branch(request)
        assert first_result.status is WorkspaceBranchOutcomeStatus.CREATED
        assert second_result.status is WorkspaceBranchOutcomeStatus.CREATED
        assert first_result.branch is not None
        assert second_result.branch is not None
        await verify_branch_isolation_and_net_changes(
            source,
            first_result.branch,
            second_result.branch,
        )
        await first_result.branch.rollback()
        await second_result.branch.rollback()

        publication_result = await source.create_branch(await _request(source))
        assert publication_result.branch is not None
        assert publication_result.evidence.baseline_revision is not None
        await verify_atomic_publication(
            source,
            publication_result.branch,
            publication_result.evidence.baseline_revision,
        )

        await source.write_bytes("original.txt", b"original")
        await source.write_bytes("deleted.txt", b"delete-me")
        conflict_result = await source.create_branch(await _request(source))
        assert conflict_result.branch is not None
        assert conflict_result.evidence.baseline_revision is not None
        await verify_conflict_is_all_or_none(
            source,
            conflict_result.branch,
            conflict_result.evidence.baseline_revision,
        )
        await conflict_result.branch.rollback()

    asyncio.run(scenario())


def test_bounded_conflict_response_replays_without_transfer_fencing(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = _populated_root(tmp_path)
        binding = WorkspaceBranchBindingAuthority(
            environment_name="remote-env",
            binding_generation="bounded-generation",
            binding_identity="bounded-allocation",
        )
        resolver = _DurableTestAuthorityProvider(binding)
        authority = WorkspaceBranchAuthority(
            session_id="bounded-session",
            expected_run_epoch=3,
            environment_name=binding.environment_name,
            binding_generation=binding.binding_generation,
            binding_identity=binding.binding_identity,
            creating_authority="bounded-worker",
            resource_policy="remote-branch-defaults",
        )
        source = _workspace(root, resolver=resolver)
        created = await source.create_branch(
            await _request(
                source,
                limits=WorkspaceBranchLimits(max_evidence_bytes=48 * 1024),
                authority=authority,
                branch_id="bounded-conflict",
                idempotency_key="bounded-create",
            )
        )
        assert created.branch is not None
        component = "d" * 180
        names = [f"{component}/{component}/f{index:03d}.txt" for index in range(70)]
        for index, name in enumerate(names):
            await created.branch.create_bytes(name, f"branch-{index}".encode())
        changes = await created.branch.changes()

        for name in names:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"source-drift")

        request = WorkspaceBranchPublicationRequest(
            branch_id=created.branch.branch_id,
            baseline_revision=changes.baseline_revision,
            change_set_digest=changes.digest,
            idempotency_key="bounded-publish",
            expected_run_epoch=authority.expected_run_epoch,
            binding_generation=binding.binding_generation,
        )
        first = await created.branch.publish(request)
        second = await created.branch.publish(request)

        assert first.status is WorkspaceBranchOutcomeStatus.CONFLICTED
        assert second == first
        assert len(first.conflicts) == len(names)
        record = json.loads(
            (_private_branch_directory(root) / "record.json").read_text(encoding="utf-8")
        )
        assert record["publication_attempts"] == 1

    asyncio.run(scenario())


def test_create_replay_and_publication_share_one_lock_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.workspaces import _runner_branch

    marker = "# CAYU_TEST_AFTER_OPERATION_BRANCH_LOCK"
    assert marker in _runner_branch.RUNNER_WORKSPACE_BRANCH_PROGRAM

    async def scenario() -> None:
        root = _populated_root(tmp_path)
        binding = WorkspaceBranchBindingAuthority(
            environment_name="remote-env",
            binding_generation="lock-order-generation",
            binding_identity="lock-order-allocation",
        )
        resolver = _DurableTestAuthorityProvider(binding)
        authority = WorkspaceBranchAuthority(
            session_id="lock-order-session",
            expected_run_epoch=17,
            environment_name=binding.environment_name,
            binding_generation=binding.binding_generation,
            binding_identity=binding.binding_identity,
            creating_authority="lock-order-worker",
            resource_policy="remote-branch-defaults",
        )
        source = RunnerWorkspace(
            _BranchRunner(root, inherit_env=False),
            workspace_id="remote-source",
            python_executable=sys.executable,
            enable_workspace_branches=True,
            branch_operation_timeout_s=2,
            branch_authority_resolver=resolver,
        )
        request = await _request(
            source,
            authority=authority,
            branch_id="lock-order-branch",
            idempotency_key="lock-order-create",
        )
        created = await source.create_branch(request)
        assert created.branch is not None
        await created.branch.write_bytes("original.txt", b"published")
        changes = await created.branch.changes()

        entered = tmp_path / "publish-branch-lock-entered"
        release = tmp_path / "release-publish-branch-lock"
        replacement = (
            f'open({str(entered)!r}, "w").close()\n'
            f"        while not os.path.exists({str(release)!r}):\n"
            "            time.sleep(0.01)"
        )
        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            _runner_branch.RUNNER_WORKSPACE_BRANCH_PROGRAM.replace(marker, replacement),
        )
        publication_task = asyncio.create_task(
            created.branch.publish(
                WorkspaceBranchPublicationRequest(
                    branch_id=created.branch.branch_id,
                    baseline_revision=changes.baseline_revision,
                    change_set_digest=changes.digest,
                    idempotency_key="lock-order-publish",
                    expected_run_epoch=authority.expected_run_epoch,
                    binding_generation=binding.binding_generation,
                )
            )
        )
        for _ in range(200):
            if entered.exists():
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("publication did not acquire its branch lock")

        replay_task = asyncio.create_task(source.create_branch(request))
        await asyncio.sleep(0.1)
        release.write_text("release", encoding="utf-8")

        publication = await publication_task
        assert publication.status is WorkspaceBranchOutcomeStatus.COMMITTED
        with pytest.raises(WorkspaceBranchOperationConflict):
            await replay_task

    asyncio.run(scenario())


def test_durable_publication_key_stays_bound_after_conflict_and_mutation(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        root = _populated_root(tmp_path)
        binding = WorkspaceBranchBindingAuthority(
            environment_name="remote-env",
            binding_generation="ledger-generation",
            binding_identity="ledger-allocation",
        )
        resolver = _DurableTestAuthorityProvider(binding)
        authority = WorkspaceBranchAuthority(
            session_id="ledger-session",
            expected_run_epoch=5,
            environment_name=binding.environment_name,
            binding_generation=binding.binding_generation,
            binding_identity=binding.binding_identity,
            creating_authority="ledger-worker",
            resource_policy="remote-branch-defaults",
        )
        source = _workspace(root, resolver=resolver)
        created = await source.create_branch(
            await _request(
                source,
                authority=authority,
                branch_id="ledger-branch",
                idempotency_key="ledger-create",
            )
        )
        assert created.branch is not None
        await created.branch.write_bytes("original.txt", b"first-branch-value")
        first_changes = await created.branch.changes()
        await source.write_bytes("original.txt", b"source-drift")
        first_request = WorkspaceBranchPublicationRequest(
            branch_id=created.branch.branch_id,
            baseline_revision=first_changes.baseline_revision,
            change_set_digest=first_changes.digest,
            idempotency_key="ledger-publish",
            expected_run_epoch=authority.expected_run_epoch,
            binding_generation=binding.binding_generation,
        )
        conflicted = await created.branch.publish(first_request)
        assert conflicted.status is WorkspaceBranchOutcomeStatus.CONFLICTED

        await source.write_bytes("original.txt", b"original")
        await created.branch.write_bytes("original.txt", b"second-branch-value")
        second_changes = await created.branch.changes()
        reused = WorkspaceBranchPublicationRequest(
            branch_id=created.branch.branch_id,
            baseline_revision=second_changes.baseline_revision,
            change_set_digest=second_changes.digest,
            idempotency_key="ledger-publish",
            expected_run_epoch=authority.expected_run_epoch,
            binding_generation=binding.binding_generation,
        )
        with pytest.raises(WorkspaceBranchOperationConflict, match="identity_reused"):
            await created.branch.publish(reused)
        assert (root / "original.txt").read_bytes() == b"original"

        committed = await created.branch.publish(
            reused.model_copy(update={"idempotency_key": "ledger-publish-2"})
        )
        assert committed.status is WorkspaceBranchOutcomeStatus.COMMITTED
        assert (root / "original.txt").read_bytes() == b"second-branch-value"

    asyncio.run(scenario())


def test_publication_preserves_source_mode_and_file_to_directory_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.workspaces import _runner_branch

    marker = "# CAYU_TEST_AFTER_PUBLICATION_APPLY"
    original_program = _runner_branch.RUNNER_WORKSPACE_BRANCH_PROGRAM
    assert marker in original_program

    async def scenario() -> None:
        root = tmp_path / "workspace"
        root.mkdir()
        node = root / "node"
        node.write_bytes(b"baseline-file")
        node.chmod(0o755)
        source = _workspace(root)
        created = await source.create_branch(await _request(source))
        assert created.branch is not None
        await created.branch.delete("node")
        await created.branch.create_bytes("node/nested/child.txt", b"child")
        changes = await created.branch.changes()

        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program.replace(marker, 'raise OSError("post-apply failure")'),
        )
        failed = await created.branch.publish(
            WorkspaceBranchPublicationRequest(
                branch_id=created.branch.branch_id,
                baseline_revision=changes.baseline_revision,
                change_set_digest=changes.digest,
            )
        )
        assert failed.status is WorkspaceBranchOutcomeStatus.FAILED
        assert node.is_file()
        assert node.read_bytes() == b"baseline-file"
        assert stat.S_IMODE(node.stat().st_mode) == 0o755

        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program,
        )
        replacement = await source.create_branch(await _request(source))
        assert replacement.branch is not None
        await replacement.branch.write_bytes("node", b"published-file")
        replacement_changes = await replacement.branch.changes()
        committed = await replacement.branch.publish(
            WorkspaceBranchPublicationRequest(
                branch_id=replacement.branch.branch_id,
                baseline_revision=replacement_changes.baseline_revision,
                change_set_digest=replacement_changes.digest,
            )
        )
        assert committed.status is WorkspaceBranchOutcomeStatus.COMMITTED
        assert node.read_bytes() == b"published-file"
        assert stat.S_IMODE(node.stat().st_mode) == 0o755

        transition = await source.create_branch(await _request(source))
        assert transition.branch is not None
        await transition.branch.delete("node")
        await transition.branch.create_bytes("node/nested/child.txt", b"child")
        transition_changes = await transition.branch.changes()
        transitioned = await transition.branch.publish(
            WorkspaceBranchPublicationRequest(
                branch_id=transition.branch.branch_id,
                baseline_revision=transition_changes.baseline_revision,
                change_set_digest=transition_changes.digest,
            )
        )
        assert transitioned.status is WorkspaceBranchOutcomeStatus.COMMITTED
        assert (node / "nested" / "child.txt").read_bytes() == b"child"

    asyncio.run(scenario())


def test_publication_rolls_back_a_source_write_that_commits_then_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.workspaces import _runner_branch

    marker = "# CAYU_TEST_AFTER_SOURCE_WRITE_MUTATION"
    original_program = _runner_branch.RUNNER_WORKSPACE_BRANCH_PROGRAM
    assert marker in original_program

    async def scenario() -> None:
        root = tmp_path / "commit-then-raise-workspace"
        root.mkdir()
        source = _workspace(root)
        created = await source.create_branch(await _request(source))
        assert created.branch is not None
        await created.branch.create_bytes("nested/new.txt", b"new")
        changes = await created.branch.changes()

        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program.replace(marker, 'raise OSError("post-write failure")'),
        )
        failed = await created.branch.publish(
            WorkspaceBranchPublicationRequest(
                branch_id=created.branch.branch_id,
                baseline_revision=changes.baseline_revision,
                change_set_digest=changes.digest,
            )
        )

        assert failed.status is WorkspaceBranchOutcomeStatus.FAILED
        assert not (root / "nested").exists()

    asyncio.run(scenario())


def test_publication_rollback_does_not_overwrite_external_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.workspaces import _runner_branch

    marker = "# CAYU_TEST_AFTER_PUBLICATION_PROGRESS"
    original_program = _runner_branch.RUNNER_WORKSPACE_BRANCH_PROGRAM
    assert marker in original_program
    monkeypatch.setattr(
        _runner_branch,
        "RUNNER_WORKSPACE_BRANCH_PROGRAM",
        original_program.replace(
            marker,
            'external_fd = os.open(change["path"], os.O_WRONLY | os.O_TRUNC, dir_fd=root_fd)\n'
            "            try:\n"
            '                write_all(external_fd, b"external-writer")\n'
            "            finally:\n"
            "                os.close(external_fd)\n"
            '            raise OSError("post-apply failure")',
        ),
    )

    async def scenario() -> None:
        root = _populated_root(tmp_path)
        source = _workspace(root)
        created = await source.create_branch(await _request(source))
        assert created.branch is not None
        await created.branch.write_bytes("original.txt", b"branch-value")
        changes = await created.branch.changes()
        result = await created.branch.publish(
            WorkspaceBranchPublicationRequest(
                branch_id=created.branch.branch_id,
                baseline_revision=changes.baseline_revision,
                change_set_digest=changes.digest,
            )
        )
        assert result.status is WorkspaceBranchOutcomeStatus.AMBIGUOUS
        assert (root / "original.txt").read_bytes() == b"external-writer"

    asyncio.run(scenario())


def test_remote_detail_codes_are_fixed_before_public_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canary = "provider-private-secret-canary"

    async def scenario() -> None:
        root = _populated_root(tmp_path)
        runner = _BranchRunner(root, inherit_env=False)
        source = RunnerWorkspace(
            runner,
            workspace_id="remote-source",
            python_executable=sys.executable,
            enable_workspace_branches=True,
        )
        request = await _request(source)

        async def forged_exec(command, **kwargs):
            del command, kwargs
            return ExecResult(
                stdout=json.dumps(
                    {
                        "ok": False,
                        "error_type": "resource_exhausted",
                        "detail_code": canary,
                    }
                ),
                exit_code=1,
            )

        monkeypatch.setattr(runner, "_exec", forged_exec)
        result = await source.create_branch(request)
        assert result.status is WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED
        assert result.evidence.detail_code == "workspace_branch_remote_failure"
        assert canary not in repr(result)

    asyncio.run(scenario())


def test_runner_branch_rejects_a_baseline_from_another_workspace_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        root = _populated_root(tmp_path)
        original = _workspace(root)
        request = await _request(original)
        other_runner = _BranchRunner(root, inherit_env=False)
        other = RunnerWorkspace(
            other_runner,
            workspace_id="different-remote-source",
            python_executable=sys.executable,
            enable_workspace_branches=True,
        )

        async def unexpected_exec(command, **kwargs):
            del command, kwargs
            pytest.fail("foreign baseline reached remote branch dispatch")

        monkeypatch.setattr(other_runner, "_exec", unexpected_exec)
        with pytest.raises(ValueError, match="different workspace"):
            await other.create_branch(request)

    asyncio.run(scenario())


def test_runner_branch_rejects_fixed_identity_that_cannot_fit_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        root = _populated_root(tmp_path)
        runner = _BranchRunner(root, inherit_env=False)
        source = RunnerWorkspace(
            runner,
            workspace_id="workspace-" + "w" * 60_000,
            python_executable=sys.executable,
            enable_workspace_branches=True,
        )
        request = await _request(
            source,
            limits=WorkspaceBranchLimits(max_evidence_bytes=1024),
        )

        async def unexpected_exec(command, **kwargs):
            del command, kwargs
            pytest.fail("unrepresentable authority reached remote branch dispatch")

        monkeypatch.setattr(runner, "_exec", unexpected_exec)
        result = await source.create_branch(request)
        assert result.status is WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED
        assert result.branch is None
        assert result.evidence.detail_code in {
            "change_evidence_limit_exceeded",
            "result_evidence_limit_exceeded",
        }
        assert result.evidence.source.workspace_id.startswith("sha256:")
        assert len(result.evidence.model_dump_json().encode("utf-8")) <= 1024

    asyncio.run(scenario())


def test_runner_branch_fences_provider_controlled_read_mutation_and_list_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        source = _workspace(_populated_root(tmp_path))
        created = await source.create_branch(await _request(source))
        second = await source.create_branch(await _request(source))
        third = await source.create_branch(await _request(source))
        assert created.branch is not None
        assert second.branch is not None
        assert third.branch is not None

        responses = iter(
            (
                {
                    "ok": True,
                    "content_base64": "b3JpZ2luYWw=",
                    "total_bytes": 8,
                    "revision": "sha256:" + "0" * 64,
                    "sha256": "0" * 64,
                },
                {
                    "ok": True,
                    "mutation": {
                        "operation": "create",
                        "before": None,
                        "after": {"sha256": "0" * 64, "bytes": 5},
                    },
                },
                {"ok": True, "paths": [], "total_count": 100_001},
            )
        )

        async def forged_exec(command, **kwargs):
            del command, kwargs
            return ExecResult(stdout=json.dumps(next(responses)), exit_code=0)

        monkeypatch.setattr(source._runner, "_exec", forged_exec)
        with pytest.raises(WorkspaceBranchFencedError, match="read identity"):
            await created.branch.read_bytes("original.txt")
        assert created.branch.lifecycle_status is WorkspaceBranchLifecycleStatus.FENCED

        with pytest.raises(WorkspaceBranchFencedError, match="create evidence"):
            await second.branch.create_bytes("new.txt", b"value")
        assert second.branch.lifecycle_status is WorkspaceBranchLifecycleStatus.FENCED

        with pytest.raises(WorkspaceBranchFencedError, match="list is invalid"):
            await third.branch.list()
        assert third.branch.lifecycle_status is WorkspaceBranchLifecycleStatus.FENCED

    asyncio.run(scenario())


def test_runner_branch_fences_contradictory_mutation_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        source = _workspace(_populated_root(tmp_path))
        branches = []
        for _ in range(5):
            created = await source.create_branch(await _request(source))
            assert created.branch is not None
            branches.append(created.branch)

        original = {"sha256": hashlib.sha256(b"original").hexdigest(), "bytes": 8}
        value = {"sha256": hashlib.sha256(b"value").hexdigest(), "bytes": 5}
        forged = iter(
            (
                {"ok": True, "mutation": {"operation": "replace", "before": None, "after": value}},
                {"ok": True, "mutation": {"operation": "delete", "before": None, "after": None}},
                {
                    "ok": True,
                    "mutation": {"operation": "create", "before": original, "after": value},
                },
                {"ok": True, "mutation": {"operation": "replace", "before": value, "after": value}},
                {"ok": True, "mutation": {"operation": "delete", "before": value, "after": value}},
            )
        )

        async def forged_exec(command, **kwargs):
            del command, kwargs
            return ExecResult(stdout=json.dumps(next(forged)), exit_code=0)

        monkeypatch.setattr(source._runner, "_exec", forged_exec)
        with pytest.raises(WorkspaceBranchFencedError, match="write evidence|invalid mutation"):
            await branches[0].write_bytes("original.txt", b"value")
        with pytest.raises(WorkspaceBranchFencedError, match="delete evidence|invalid mutation"):
            await branches[1].delete("original.txt")
        with pytest.raises(WorkspaceBranchFencedError, match="create evidence|invalid mutation"):
            await branches[2].create_bytes("new.txt", b"value")
        with pytest.raises(WorkspaceBranchFencedError, match="replace evidence|invalid mutation"):
            await branches[3].replace_bytes(
                "original.txt",
                b"value",
                expected_revision="sha256:" + original["sha256"],
            )
        with pytest.raises(WorkspaceBranchFencedError, match="delete evidence|invalid mutation"):
            await branches[4].delete_if_revision(
                "original.txt",
                expected_revision="sha256:" + original["sha256"],
            )
        assert all(
            branch.lifecycle_status is WorkspaceBranchLifecycleStatus.FENCED for branch in branches
        )

    asyncio.run(scenario())


def test_runner_branch_rejects_provider_controlled_terminal_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canary = "provider-terminal-digest-secret"

    async def scenario() -> None:
        process_local = _workspace(_populated_root(tmp_path / "rollback"))
        created = await process_local.create_branch(await _request(process_local))
        assert created.branch is not None

        async def forged_rollback(command, **kwargs):
            del command, kwargs
            return ExecResult(
                stdout=json.dumps({"ok": True, "status": "rolled_back", "digest": canary}),
                exit_code=0,
            )

        monkeypatch.setattr(process_local._runner, "_exec", forged_rollback)
        with pytest.raises(WorkspaceBranchFencedError, match="change-set digest") as raised:
            await created.branch.rollback()
        assert canary not in repr(raised.value)
        assert created.branch.lifecycle_status is WorkspaceBranchLifecycleStatus.FENCED

        binding = WorkspaceBranchBindingAuthority(
            environment_name="remote-env",
            binding_generation="terminal-generation",
            binding_identity="terminal-binding",
        )
        resolver = _DurableTestAuthorityProvider(binding)
        authority = WorkspaceBranchAuthority(
            session_id="terminal-session",
            expected_run_epoch=41,
            environment_name=binding.environment_name,
            binding_generation=binding.binding_generation,
            binding_identity=binding.binding_identity,
            creating_authority="terminal-worker",
            resource_policy="remote-branch-defaults",
        )
        resolver.authorize_operation(authority)
        recovered_source = _workspace(
            _populated_root(tmp_path / "recovery"),
            resolver=resolver,
        )
        response = {
            "ok": True,
            "state": "rolled_back",
            "branch_id": "terminal-branch",
            "source": {
                "workspace_id": recovered_source.id,
                "observer": "runner-branch-tests",
            },
            "baseline_revision": "sha256:" + "1" * 64,
            "limits": WorkspaceBranchLimits().model_dump(mode="json"),
            "authority": authority.model_dump(mode="json"),
            "detail_code": "workspace_branch_rolled_back",
            "digest": canary,
        }

        async def forged_recovery(command, **kwargs):
            del command, kwargs
            return ExecResult(stdout=json.dumps(response), exit_code=0)

        monkeypatch.setattr(recovered_source._runner, "_exec", forged_recovery)
        with pytest.raises(WorkspaceBranchFencedError, match="change-set digest") as recovered:
            await recovered_source.recover_branch(
                WorkspaceBranchRecoveryRequest(
                    branch_id="terminal-branch",
                    session_id=authority.session_id,
                    expected_run_epoch=authority.expected_run_epoch,
                    binding_generation=binding.binding_generation,
                    binding_identity=binding.binding_identity,
                    recovery_id="terminal-recovery",
                )
            )
        assert canary not in repr(recovered.value)

    asyncio.run(scenario())


def test_runner_workspace_branch_bounds_and_rollback(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = tmp_path / "workspace"
        root.mkdir()
        source = _workspace(root)
        created = await source.create_branch(
            await _request(source, limits=WorkspaceBranchLimits(max_file_bytes=3))
        )
        assert created.branch is not None
        await verify_bound_rollback_and_cleanup(source, created.branch)

    asyncio.run(scenario())


def test_remote_branch_rejects_oversized_mutation_inputs_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        root = tmp_path / "workspace"
        root.mkdir()
        source = _workspace(root)
        created = await source.create_branch(
            await _request(
                source,
                limits=WorkspaceBranchLimits(max_file_bytes=3, max_path_bytes=8),
            )
        )
        assert created.branch is not None

        async def unexpected_exec(command, **kwargs):
            del command, kwargs
            pytest.fail("oversized branch mutation reached runner dispatch")

        monkeypatch.setattr(source._runner, "_exec", unexpected_exec)
        with pytest.raises(
            WorkspaceBranchResourceExhaustedError,
            match="file_byte_limit_exceeded",
        ):
            await created.branch.write_bytes("ok.txt", b"four")
        with pytest.raises(
            WorkspaceBranchResourceExhaustedError,
            match="path_byte_limit_exceeded",
        ):
            await created.branch.create_bytes("path-too-long.txt", b"ok")

    asyncio.run(scenario())


def test_remote_branch_counts_logical_directories_against_path_limit(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = tmp_path / "workspace"
        root.mkdir()
        source = _workspace(root)
        created = await source.create_branch(
            await _request(source, limits=WorkspaceBranchLimits(max_paths=2))
        )
        assert created.branch is not None

        with pytest.raises(
            WorkspaceBranchResourceExhaustedError,
            match="path_count_limit_exceeded",
        ):
            await created.branch.write_bytes("one/two/file.txt", b"value")

        assert (await created.branch.list()).paths == ()

    asyncio.run(scenario())


def test_remote_branch_preserves_baseline_directory_shape_after_child_deletion(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        root = tmp_path / "workspace"
        (root / "node").mkdir(parents=True)
        (root / "node" / "first.txt").write_bytes(b"first")
        (root / "node" / "second.txt").write_bytes(b"second")
        source = _workspace(root)
        created = await source.create_branch(await _request(source))
        assert created.branch is not None
        await created.branch.delete("node/first.txt")
        await created.branch.delete("node/second.txt")

        with pytest.raises(IsADirectoryError):
            await created.branch.create_bytes("node", b"replacement")

        changes = await created.branch.changes()
        assert [(change.path, change.operation) for change in changes.changes] == [
            ("node/first.txt", "deleted"),
            ("node/second.txt", "deleted"),
        ]
        assert (root / "node").is_dir()

    asyncio.run(scenario())


def test_remote_branch_prunes_transient_overlay_directories(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = tmp_path / "workspace"
        root.mkdir()
        source = _workspace(root)
        created = await source.create_branch(await _request(source))
        assert created.branch is not None

        await created.branch.create_bytes("node/nested.txt", b"temporary")
        await created.branch.delete("node/nested.txt")
        assert (await created.branch.changes()).changes == ()

        await created.branch.create_bytes("node", b"replacement-file")
        assert (await created.branch.read_bytes("node")).content == b"replacement-file"
        assert [
            (change.path, change.operation) for change in (await created.branch.changes()).changes
        ] == [("node", "created")]

    asyncio.run(scenario())


def test_remote_branch_fences_an_unacknowledged_private_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.workspaces import _runner_branch

    marker = "# CAYU_TEST_AFTER_PRIVATE_MUTATION"
    original_program = _runner_branch.RUNNER_WORKSPACE_BRANCH_PROGRAM
    assert marker in original_program

    async def scenario() -> None:
        root = tmp_path / "workspace"
        root.mkdir()
        source = _workspace(root)
        created = await source.create_branch(await _request(source))
        assert created.branch is not None
        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program.replace(marker, 'raise OSError("injected mutation failure")'),
        )

        with pytest.raises(WorkspaceBranchFencedError, match="operation failed"):
            await created.branch.write_bytes("new.txt", b"unacknowledged")

        assert created.branch.lifecycle_status is WorkspaceBranchLifecycleStatus.FENCED
        assert not (root / "new.txt").exists()

    asyncio.run(scenario())


def test_remote_branch_private_root_replacement_cannot_redirect_mutation_or_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.workspaces import _runner_branch

    marker = "# CAYU_TEST_AFTER_OPERATION_RECORD_LOAD"
    original_program = _runner_branch.RUNNER_WORKSPACE_BRANCH_PROGRAM
    assert marker in original_program

    def replacement_program(private: Path, displaced: Path) -> str:
        replacement = (
            marker
            + "\n"
            + f"        os.rename({str(private)!r}, {str(displaced)!r})\n"
            + f"        os.mkdir({str(private)!r}, 0o700)\n"
            + f"        with open(os.path.join({str(private)!r}, RECORD_NAME), 'wb') as victim:\n"
            + "            victim.write(b'UNRELATED-VICTIM')\n"
            + "        for area in ('baseline', 'overlay'):\n"
            + f"            os.mkdir(os.path.join({str(private)!r}, area), 0o700)\n"
            + f"            with open(os.path.join({str(private)!r}, area, 'must-survive.txt'), 'wb') as sentinel:\n"
            + "                sentinel.write(b'UNRELATED-SENTINEL')"
        )
        return original_program.replace(marker, replacement)

    async def scenario() -> None:
        write_root = _populated_root(tmp_path / "write")
        write_source = _workspace(write_root)
        write_created = await write_source.create_branch(await _request(write_source))
        assert write_created.branch is not None
        write_private = _private_branch_directory(write_root)
        write_displaced = Path(str(write_private) + ".displaced")
        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            replacement_program(write_private, write_displaced),
        )

        await write_created.branch.write_bytes("new.txt", b"branch-data")

        assert (write_private / "record.json").read_bytes() == b"UNRELATED-VICTIM"
        for area in ("baseline", "overlay"):
            assert (write_private / area / "must-survive.txt").read_bytes() == (
                b"UNRELATED-SENTINEL"
            )
        assert (write_displaced / "overlay" / "new.txt").read_bytes() == b"branch-data"
        displaced_record = json.loads((write_displaced / "record.json").read_text(encoding="utf-8"))
        assert displaced_record["state"] == "open"
        assert "new.txt" in displaced_record["overlay"]

        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program,
        )
        publish_root = _populated_root(tmp_path / "publish")
        publish_source = _workspace(publish_root)
        publish_created = await publish_source.create_branch(await _request(publish_source))
        assert publish_created.branch is not None
        assert publish_created.evidence.baseline_revision is not None
        await publish_created.branch.write_bytes("original.txt", b"published")
        publish_changes = await publish_created.branch.changes()
        publish_private = _private_branch_directory(publish_root)
        publish_displaced = Path(str(publish_private) + ".displaced")
        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            replacement_program(publish_private, publish_displaced),
        )

        published = await publish_created.branch.publish(
            WorkspaceBranchPublicationRequest(
                branch_id=publish_created.branch.branch_id,
                baseline_revision=publish_created.evidence.baseline_revision,
                change_set_digest=publish_changes.digest,
            )
        )

        assert published.status is WorkspaceBranchOutcomeStatus.COMMITTED
        assert (publish_root / "original.txt").read_bytes() == b"published"
        assert (publish_private / "record.json").read_bytes() == b"UNRELATED-VICTIM"
        for area in ("baseline", "overlay"):
            assert (publish_private / area / "must-survive.txt").read_bytes() == (
                b"UNRELATED-SENTINEL"
            )
        publish_record = json.loads((publish_displaced / "record.json").read_text(encoding="utf-8"))
        assert publish_record["state"] == "committed"
        assert not (publish_displaced / "baseline").exists()
        assert not (publish_displaced / "overlay").exists()

        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program,
        )
        rollback_root = _populated_root(tmp_path / "rollback")
        rollback_source = _workspace(rollback_root)
        rollback_created = await rollback_source.create_branch(await _request(rollback_source))
        assert rollback_created.branch is not None
        await rollback_created.branch.write_bytes("new.txt", b"branch-data")
        rollback_private = _private_branch_directory(rollback_root)
        rollback_displaced = Path(str(rollback_private) + ".displaced")
        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            replacement_program(rollback_private, rollback_displaced),
        )

        rolled_back = await rollback_created.branch.rollback()

        assert rolled_back.status is WorkspaceBranchOutcomeStatus.ROLLED_BACK
        assert (rollback_private / "record.json").read_bytes() == b"UNRELATED-VICTIM"
        for area in ("baseline", "overlay"):
            assert (rollback_private / area / "must-survive.txt").read_bytes() == (
                b"UNRELATED-SENTINEL"
            )
        rollback_record = json.loads(
            (rollback_displaced / "record.json").read_text(encoding="utf-8")
        )
        assert rollback_record["state"] == "rolled_back"
        assert not (rollback_displaced / "baseline").exists()
        assert not (rollback_displaced / "overlay").exists()

    asyncio.run(scenario())


def test_remote_creation_fences_a_replaced_canonical_root_before_acknowledgement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.workspaces import _runner_branch

    marker = "# CAYU_TEST_BEFORE_CREATION_ACKNOWLEDGEMENT"
    original_program = _runner_branch.RUNNER_WORKSPACE_BRANCH_PROGRAM
    assert marker in original_program
    replacement = (
        marker
        + "\n"
        + "            os.rename(private, private + '.displaced')\n"
        + "            os.mkdir(private, 0o700)\n"
        + "            with open(os.path.join(private, RECORD_NAME), 'wb') as victim:\n"
        + "                victim.write(b'UNRELATED-VICTIM')"
    )
    monkeypatch.setattr(
        _runner_branch,
        "RUNNER_WORKSPACE_BRANCH_PROGRAM",
        original_program.replace(marker, replacement),
    )

    async def scenario() -> None:
        root = _populated_root(tmp_path)
        source = _workspace(root)

        with pytest.raises(WorkspaceBranchFencedError):
            await source.create_branch(await _request(source))

        canonical = tuple(
            path for path in root.parent.glob(".cayu-workspace-branch-*.stage") if path.is_dir()
        )
        assert len(canonical) == 1
        private = canonical[0]
        displaced = Path(str(private) + ".displaced")
        assert (private / "record.json").read_bytes() == b"UNRELATED-VICTIM"
        record = json.loads((displaced / "record.json").read_text(encoding="utf-8"))
        assert record["state"] == "open"
        assert (displaced / "baseline" / "original.txt").read_bytes() == b"original"
        assert (root / "original.txt").read_bytes() == b"original"
        assert tuple(root.parent.glob(".cayu-workspace-branch-*.stage.claim"))
        assert tuple(root.parent.glob(".cayu-workspace-branch-*.stage.claim.bound"))

    asyncio.run(scenario())


def test_remote_creation_retry_keeps_a_canonical_replacement_untouched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.workspaces import _runner_branch

    capture_marker = "# CAYU_TEST_AFTER_CREATION_RECORD"
    reset_marker = "# CAYU_TEST_BEFORE_CREATION_RETRY_RESET"
    original_program = _runner_branch.RUNNER_WORKSPACE_BRANCH_PROGRAM
    assert capture_marker in original_program
    assert reset_marker in original_program

    async def scenario() -> None:
        root = _populated_root(tmp_path)
        source, authority = _durable_workspace(root, identity="retry-staging-race")
        request = await _request(
            source,
            authority=authority,
            branch_id="retry-staging-race",
            idempotency_key="retry-staging-race-create",
        )
        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program.replace(capture_marker, "os._exit(86)"),
        )
        with pytest.raises(WorkspaceBranchFencedError):
            await source.create_branch(request)

        replacement = (
            reset_marker
            + "\n"
            + "                os.rename(staging, staging + '.owned-race')\n"
            + "                os.mkdir(staging, 0o700)\n"
            + "                with open(os.path.join(staging, 'must-survive.txt'), 'wb') as victim:\n"
            + "                    victim.write(b'UNRELATED-STAGING')"
        )
        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program.replace(reset_marker, replacement),
        )
        with pytest.raises(WorkspaceBranchFencedError):
            await source.create_branch(request)

        staged = tuple(root.parent.glob(".cayu-workspace-branch-*.stage"))
        displaced = tuple(root.parent.glob(".cayu-workspace-branch-*.stage.owned-race"))
        claims = tuple(root.parent.glob(".cayu-workspace-branch-*.stage.claim"))
        assert len(staged) == len(displaced) == len(claims) == 1
        assert (staged[0] / "must-survive.txt").read_bytes() == b"UNRELATED-STAGING"
        owned_record = json.loads((displaced[0] / "record.json").read_text(encoding="utf-8"))
        assert owned_record["state"] == "open"
        assert (displaced[0] / "baseline" / "original.txt").read_bytes() == b"original"

    asyncio.run(scenario())


def test_remote_creation_does_not_replace_a_preexisting_canonical_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.workspaces import _runner_branch

    marker = "# CAYU_TEST_BEFORE_CREATION_DIRECTORY"
    original_program = _runner_branch.RUNNER_WORKSPACE_BRANCH_PROGRAM
    assert marker in original_program
    replacement = (
        marker
        + "\n"
        + "\n"
        + "            os.mkdir(private, 0o700)\n"
        + "            with open(os.path.join(private, 'must-survive.txt'), 'wb') as victim:\n"
        + "                victim.write(b'UNRELATED-PRIVATE')"
    )
    monkeypatch.setattr(
        _runner_branch,
        "RUNNER_WORKSPACE_BRANCH_PROGRAM",
        original_program.replace(marker, replacement),
    )

    async def scenario() -> None:
        root = _populated_root(tmp_path)
        source, authority = _durable_workspace(root, identity="private-destination-race")

        with pytest.raises(WorkspaceBranchOperationConflict):
            await source.create_branch(
                await _request(
                    source,
                    authority=authority,
                    branch_id="private-destination-race",
                    idempotency_key="private-destination-race-create",
                )
            )

        private = tuple(
            path for path in root.parent.glob(".cayu-workspace-branch-*.stage") if path.is_dir()
        )
        assert len(private) == 1
        assert (private[0] / "must-survive.txt").read_bytes() == b"UNRELATED-PRIVATE"
        assert tuple(root.parent.glob(".cayu-workspace-branch-*.stage.claim"))
        assert not tuple(root.parent.glob(".cayu-workspace-branch-*.stage.claim.bound"))

    asyncio.run(scenario())


def test_remote_partial_creation_claim_is_bounded_and_does_not_poison_other_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.workspaces import _runner_branch

    marker = "# CAYU_TEST_AFTER_CREATION_CLAIM_TEMPORARY_OPEN"
    original_program = _runner_branch.RUNNER_WORKSPACE_BRANCH_PROGRAM
    assert marker in original_program

    async def scenario() -> None:
        root = _populated_root(tmp_path)
        source, authority = _durable_workspace(root, identity="partial-creation-claim")
        limits = WorkspaceBranchLimits(max_active_branches=2)
        request = await _request(
            source,
            limits=limits,
            authority=authority,
            branch_id="partial-creation-claim",
            idempotency_key="partial-creation-claim-create",
        )
        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program.replace(marker, "os._exit(86)"),
        )

        with pytest.raises(WorkspaceBranchFencedError):
            await source.create_branch(request)
        pending = tuple(root.parent.glob(".cayu-workspace-branch-*.stage.claim.pending"))
        assert len(pending) == 1
        assert pending[0].stat().st_size == 0
        assert not tuple(root.parent.glob(".cayu-workspace-branch-*.stage.claim"))

        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program,
        )
        for _ in range(2):
            with pytest.raises(WorkspaceBranchFencedError):
                await source.create_branch(request)
        assert tuple(root.parent.glob(".cayu-workspace-branch-*.stage.claim.pending")) == pending

        unrelated = await source.create_branch(
            await _request(
                source,
                limits=limits,
                authority=authority,
                branch_id="partial-creation-claim-other",
                idempotency_key="partial-creation-claim-other-create",
            )
        )
        assert unrelated.status is WorkspaceBranchOutcomeStatus.CREATED
        assert unrelated.branch is not None

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("marker", "linked_before_loss"),
    (
        ("# CAYU_TEST_AFTER_CREATION_CLAIM_TEMPORARY_SYNC", False),
        ("# CAYU_TEST_AFTER_CREATION_CLAIM_LINK", True),
    ),
)
def test_remote_complete_creation_claim_publication_recovers_after_process_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    marker: str,
    linked_before_loss: bool,
) -> None:
    from cayu.workspaces import _runner_branch

    original_program = _runner_branch.RUNNER_WORKSPACE_BRANCH_PROGRAM
    assert marker in original_program

    async def scenario() -> None:
        root = _populated_root(tmp_path)
        source, authority = _durable_workspace(root, identity="recover-creation-claim")
        request = await _request(
            source,
            authority=authority,
            branch_id="recover-creation-claim",
            idempotency_key="recover-creation-claim-create",
        )
        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program.replace(marker, "os._exit(86)"),
        )

        with pytest.raises(WorkspaceBranchFencedError):
            await source.create_branch(request)
        pending = tuple(root.parent.glob(".cayu-workspace-branch-*.stage.claim.pending"))
        claims = tuple(root.parent.glob(".cayu-workspace-branch-*.stage.claim"))
        assert len(pending) == 1
        assert pending[0].stat().st_size > 0
        if linked_before_loss:
            assert len(claims) == 1
            assert claims[0].stat().st_nlink == 2
        else:
            assert not claims

        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program,
        )
        recovered = await source.create_branch(request)

        assert recovered.status is WorkspaceBranchOutcomeStatus.CREATED
        assert recovered.branch is not None
        pending = tuple(root.parent.glob(".cayu-workspace-branch-*.stage.claim.pending"))
        assert len(pending) == 1
        assert pending[0].stat().st_nlink == 2
        claims = tuple(root.parent.glob(".cayu-workspace-branch-*.stage.claim"))
        assert len(claims) == 1
        assert claims[0].stat().st_nlink == 2

    asyncio.run(scenario())


def test_process_local_complete_pending_claim_releases_capacity_at_expiry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.workspaces import _runner_branch

    marker = "# CAYU_TEST_AFTER_CREATION_CLAIM_TEMPORARY_SYNC"
    original_program = _runner_branch.RUNNER_WORKSPACE_BRANCH_PROGRAM
    assert marker in original_program

    async def scenario() -> None:
        root = _populated_root(tmp_path)
        source = _workspace(root)
        limits = WorkspaceBranchLimits(lifetime_ms=1, max_active_branches=1)
        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program.replace(marker, "os._exit(86)"),
        )

        with pytest.raises(WorkspaceBranchFencedError):
            await source.create_branch(await _request(source, limits=limits))
        pending = tuple(root.parent.glob(".cayu-workspace-branch-*.stage.claim.pending"))
        assert len(pending) == 1
        assert pending[0].stat().st_size > 0
        await asyncio.sleep(0.01)

        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program,
        )
        replacement = await source.create_branch(await _request(source, limits=limits))

        assert replacement.status is WorkspaceBranchOutcomeStatus.CREATED
        assert replacement.branch is not None
        pending = tuple(root.parent.glob(".cayu-workspace-branch-*.stage.claim.pending"))
        assert len(pending) == 2
        assert all(path.stat().st_nlink == 2 for path in pending)
        states = sorted(
            json.loads((path / "record.json").read_text(encoding="utf-8"))["state"]
            for path in root.parent.glob(".cayu-workspace-branch-*.stage")
            if path.is_dir()
        )
        assert states == ["expired", "open"]

    asyncio.run(scenario())


def test_remote_expiry_never_adopts_an_unbound_creation_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.workspaces import _runner_branch

    marker = "# CAYU_TEST_AFTER_UNBOUND_CREATION_DIRECTORY"
    original_program = _runner_branch.RUNNER_WORKSPACE_BRANCH_PROGRAM
    assert marker in original_program

    async def scenario() -> None:
        root = _populated_root(tmp_path)
        source, authority = _durable_workspace(root, identity="unbound-creation-expiry")
        limits = WorkspaceBranchLimits(lifetime_ms=1, max_active_branches=1)
        request = await _request(
            source,
            limits=limits,
            authority=authority,
            branch_id="unbound-creation-expiry",
            idempotency_key="unbound-creation-expiry-create",
        )
        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program.replace(marker, "os._exit(86)"),
        )

        with pytest.raises(WorkspaceBranchFencedError):
            await source.create_branch(request)
        await asyncio.sleep(0.01)
        staged = tuple(root.parent.glob(".cayu-workspace-branch-*.stage"))
        claims = tuple(root.parent.glob(".cayu-workspace-branch-*.stage.claim"))
        bound_claims = tuple(root.parent.glob(".cayu-workspace-branch-*.stage.claim.bound"))
        assert len(staged) == len(claims) == 1
        assert not bound_claims
        assert not tuple(staged[0].iterdir())
        original_identity = staged[0].stat().st_dev, staged[0].stat().st_ino

        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program,
        )
        with pytest.raises(WorkspaceBranchFencedError):
            await source.create_branch(request)
        result = await source.create_branch(
            await _request(
                source,
                limits=limits,
                authority=authority,
                branch_id="unbound-creation-expiry-other",
                idempotency_key="unbound-creation-expiry-other-create",
            )
        )

        assert result.status is WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED
        assert result.evidence.detail_code == "active_branch_limit_exceeded"
        assert (staged[0].stat().st_dev, staged[0].stat().st_ino) == original_identity
        assert not tuple(staged[0].iterdir())
        assert claims[0].is_file()
        assert not tuple(root.parent.glob(".cayu-workspace-branch-*.stage.claim.bound"))

    asyncio.run(scenario())


def test_remote_creation_replay_requires_the_bound_staging_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.workspaces import _runner_branch

    marker = "# CAYU_TEST_AFTER_BRANCH_CAPTURE"
    original_program = _runner_branch.RUNNER_WORKSPACE_BRANCH_PROGRAM
    assert marker in original_program

    async def scenario() -> None:
        root = _populated_root(tmp_path)
        source, authority = _durable_workspace(root, identity="missing-staging-claim")
        request = await _request(
            source,
            authority=authority,
            branch_id="missing-staging-claim",
            idempotency_key="missing-staging-claim-create",
        )
        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program.replace(marker, "os._exit(86)"),
        )

        with pytest.raises(WorkspaceBranchFencedError):
            await source.create_branch(request)
        staged = tuple(root.parent.glob(".cayu-workspace-branch-*.stage"))
        claims = tuple(root.parent.glob(".cayu-workspace-branch-*.stage.claim"))
        assert len(staged) == len(claims) == 1
        claims[0].unlink()
        Path(str(claims[0]) + ".pending").unlink()
        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program,
        )

        with pytest.raises(WorkspaceBranchFencedError):
            await source.create_branch(request)

        assert staged[0].is_dir()
        assert (
            json.loads((staged[0] / "record.json").read_text(encoding="utf-8"))["state"] == "open"
        )
        assert not Path(str(staged[0]).removesuffix(".stage")).exists()

    asyncio.run(scenario())


def test_remote_creation_replay_reuses_the_bound_directory_and_immutable_claims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.workspaces import _runner_branch

    capture_marker = "# CAYU_TEST_AFTER_CREATION_RECORD"
    original_program = _runner_branch.RUNNER_WORKSPACE_BRANCH_PROGRAM
    assert capture_marker in original_program

    async def scenario() -> None:
        root = _populated_root(tmp_path)
        source, authority = _durable_workspace(root, identity="replay-same-inode")
        request = await _request(
            source,
            authority=authority,
            branch_id="replay-same-inode",
            idempotency_key="replay-same-inode-create",
        )
        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program.replace(capture_marker, "os._exit(86)"),
        )
        with pytest.raises(WorkspaceBranchFencedError):
            await source.create_branch(request)
        staged = tuple(root.parent.glob(".cayu-workspace-branch-*.stage"))
        claims = tuple(root.parent.glob(".cayu-workspace-branch-*.stage.claim"))
        bound_claims = tuple(root.parent.glob(".cayu-workspace-branch-*.stage.claim.bound"))
        assert len(staged) == len(claims) == len(bound_claims) == 1
        original_identity = staged[0].stat().st_dev, staged[0].stat().st_ino
        claim_evidence = (
            claims[0].read_bytes(),
            bound_claims[0].read_bytes(),
            claims[0].stat().st_ino,
            bound_claims[0].stat().st_ino,
        )
        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program,
        )
        replayed = await source.create_branch(request)

        assert replayed.status is WorkspaceBranchOutcomeStatus.CREATED
        assert replayed.branch is not None
        assert (staged[0].stat().st_dev, staged[0].stat().st_ino) == original_identity
        assert (
            claims[0].read_bytes(),
            bound_claims[0].read_bytes(),
            claims[0].stat().st_ino,
            bound_claims[0].stat().st_ino,
        ) == claim_evidence
        assert (
            json.loads((staged[0] / "record.json").read_text(encoding="utf-8"))["state"] == "open"
        )

    asyncio.run(scenario())


def test_remote_expiry_does_not_delete_a_canonical_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.workspaces import _runner_branch

    capture_marker = "# CAYU_TEST_AFTER_CREATION_DIRECTORY"
    cleanup_marker = "# CAYU_TEST_BEFORE_EXPIRED_CREATION_TERMINALIZATION"
    original_program = _runner_branch.RUNNER_WORKSPACE_BRANCH_PROGRAM
    assert capture_marker in original_program
    assert cleanup_marker in original_program

    async def scenario() -> None:
        root = _populated_root(tmp_path)
        source = _workspace(root)
        limits = WorkspaceBranchLimits(lifetime_ms=1, max_active_branches=1)
        request = await _request(
            source,
            limits=limits,
        )
        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program.replace(capture_marker, "os._exit(86)"),
        )
        with pytest.raises(WorkspaceBranchFencedError):
            await source.create_branch(request)
        await asyncio.sleep(0.01)

        replacement = (
            cleanup_marker
            + "\n"
            + "\n"
            + "                os.rename(staging, staging + '.owned-expired')\n"
            + "                os.mkdir(staging, 0o700)\n"
            + "                with open(os.path.join(staging, 'must-survive.txt'), 'wb') as victim:\n"
            + "                    victim.write(b'UNRELATED-EXPIRY')"
        )
        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program.replace(cleanup_marker, replacement),
        )

        result = await source.create_branch(await _request(source, limits=limits))
        assert result.status is WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED
        assert result.evidence.detail_code == "active_branch_limit_exceeded"

        staged = tuple(root.parent.glob(".cayu-workspace-branch-*.stage"))
        displaced = tuple(root.parent.glob(".cayu-workspace-branch-*.stage.owned-expired"))
        claims = tuple(root.parent.glob(".cayu-workspace-branch-*.stage.claim"))
        bound_claims = tuple(root.parent.glob(".cayu-workspace-branch-*.stage.claim.bound"))
        cleanup = tuple(root.parent.glob(".cayu-workspace-branch-*.stage.cleanup-*"))
        assert len(staged) == len(displaced) == len(claims) == len(bound_claims) == 1
        assert not cleanup
        assert (staged[0] / "must-survive.txt").read_bytes() == b"UNRELATED-EXPIRY"
        assert (
            json.loads((displaced[0] / "record.json").read_text(encoding="utf-8"))["state"]
            == "expired"
        )
        assert not (displaced[0] / "baseline").exists()
        assert not (displaced[0] / "overlay").exists()

    asyncio.run(scenario())


def test_durable_remote_branch_recovers_a_process_lost_overlay_mutation_within_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.workspaces import _runner_branch

    marker = "# CAYU_TEST_AFTER_PRIVATE_MUTATION"
    original_program = _runner_branch.RUNNER_WORKSPACE_BRANCH_PROGRAM
    assert marker in original_program

    async def scenario() -> None:
        root = _populated_root(tmp_path)
        binding = WorkspaceBranchBindingAuthority(
            environment_name="remote-env",
            binding_generation="mutation-generation",
            binding_identity="mutation-allocation",
        )
        resolver = _DurableTestAuthorityProvider(binding)
        authority = WorkspaceBranchAuthority(
            session_id="mutation-session",
            expected_run_epoch=3,
            environment_name=binding.environment_name,
            binding_generation=binding.binding_generation,
            binding_identity=binding.binding_identity,
            creating_authority="mutation-worker",
            resource_policy="remote-branch-defaults",
        )
        source = _workspace(root, resolver=resolver)
        created = await source.create_branch(
            await _request(
                source,
                authority=authority,
                branch_id="mutation-branch",
                idempotency_key="mutation-create",
                limits=WorkspaceBranchLimits(max_overlay_bytes=1),
            )
        )
        assert created.branch is not None
        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program.replace(marker, "os._exit(86)"),
        )

        with pytest.raises(WorkspaceBranchFencedError):
            await created.branch.write_bytes("first.txt", b"x")

        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program,
        )
        recovered = await _workspace(root, resolver=resolver).recover_branch(
            WorkspaceBranchRecoveryRequest(
                branch_id="mutation-branch",
                session_id=authority.session_id,
                expected_run_epoch=authority.expected_run_epoch,
                binding_generation=authority.binding_generation,
                binding_identity=authority.binding_identity,
                recovery_id="mutation-recover",
            )
        )
        assert recovered.branch is not None
        assert (await recovered.branch.read_bytes("first.txt")).content == b"x"
        assert [change.path for change in (await recovered.branch.changes()).changes] == [
            "first.txt"
        ]

        with pytest.raises(
            WorkspaceBranchResourceExhaustedError,
            match="overlay_byte_limit_exceeded",
        ):
            await recovered.branch.write_bytes("second.txt", b"y")
        assert [change.path for change in (await recovered.branch.changes()).changes] == [
            "first.txt"
        ]
        overlay_files = tuple(
            path.relative_to(_private_branch_directory(root) / "overlay").as_posix()
            for path in (_private_branch_directory(root) / "overlay").rglob("*")
            if path.is_file()
        )
        assert overlay_files == ("first.txt",)

    asyncio.run(scenario())


def test_terminal_remote_branch_counts_capacity_until_cleanup_settles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.workspaces import _runner_branch

    marker = "# CAYU_TEST_BEFORE_COMMITTED_CLEANUP"
    original_program = _runner_branch.RUNNER_WORKSPACE_BRANCH_PROGRAM
    assert marker in original_program

    async def scenario() -> None:
        root = tmp_path / "workspace"
        root.mkdir()
        source = _workspace(root)
        request = await _request(
            source,
            limits=WorkspaceBranchLimits(max_active_branches=1),
        )
        created = await source.create_branch(request)
        assert created.branch is not None
        await created.branch.write_bytes("published.txt", b"published")
        changes = await created.branch.changes()
        publication = WorkspaceBranchPublicationRequest(
            branch_id=created.branch.branch_id,
            baseline_revision=changes.baseline_revision,
            change_set_digest=changes.digest,
        )
        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program.replace(marker, 'raise OSError("cleanup failure")'),
        )

        with pytest.raises(WorkspaceBranchFencedError, match="operation failed"):
            await created.branch.publish(publication)
        assert (root / "published.txt").read_bytes() == b"published"

        blocked = await source.create_branch(await _request(source, limits=request.limits))
        assert blocked.status is WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED

        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program,
        )
        settled = await created.branch.publish(publication)
        assert settled.status is WorkspaceBranchOutcomeStatus.COMMITTED
        replacement = await source.create_branch(await _request(source, limits=request.limits))
        assert replacement.status is WorkspaceBranchOutcomeStatus.CREATED
        assert replacement.branch is not None

    asyncio.run(scenario())


def test_failed_publication_replay_settles_private_cleanup_and_capacity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.workspaces import _runner_branch

    progress_marker = "# CAYU_TEST_AFTER_PUBLICATION_PROGRESS"
    cleanup_marker = "# CAYU_TEST_BEFORE_FAILED_CLEANUP"
    original_program = _runner_branch.RUNNER_WORKSPACE_BRANCH_PROGRAM
    assert progress_marker in original_program
    assert cleanup_marker in original_program

    async def scenario() -> None:
        root = _populated_root(tmp_path)
        source = _workspace(root)
        limits = WorkspaceBranchLimits(max_active_branches=1)
        created = await source.create_branch(await _request(source, limits=limits))
        assert created.branch is not None
        await created.branch.write_bytes("original.txt", b"restored-after-failure")
        changes = await created.branch.changes()
        request = WorkspaceBranchPublicationRequest(
            branch_id=created.branch.branch_id,
            baseline_revision=changes.baseline_revision,
            change_set_digest=changes.digest,
        )
        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program.replace(
                progress_marker,
                'raise OSError("injected publication failure")',
            ).replace(
                cleanup_marker,
                'raise OSError("injected failed cleanup")',
            ),
        )

        with pytest.raises(WorkspaceBranchFencedError, match="operation failed"):
            await created.branch.publish(request)
        assert (root / "original.txt").read_bytes() == b"original"
        private = _private_branch_directory(root)
        assert json.loads((private / "record.json").read_text(encoding="utf-8"))["state"] == (
            "failed"
        )
        assert (private / "baseline").exists()

        blocked = await source.create_branch(await _request(source, limits=limits))
        assert blocked.status is WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED

        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program,
        )
        replayed = await created.branch.publish(request)
        assert replayed.status is WorkspaceBranchOutcomeStatus.FAILED
        assert not (private / "baseline").exists()
        assert not (private / "overlay").exists()
        replacement = await source.create_branch(await _request(source, limits=limits))
        assert replacement.status is WorkspaceBranchOutcomeStatus.CREATED

    asyncio.run(scenario())


def test_fresh_recovery_settles_failed_publication_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.workspaces import _runner_branch

    progress_marker = "# CAYU_TEST_AFTER_PUBLICATION_PROGRESS"
    cleanup_marker = "# CAYU_TEST_BEFORE_FAILED_CLEANUP"
    original_program = _runner_branch.RUNNER_WORKSPACE_BRANCH_PROGRAM

    async def scenario() -> None:
        root = _populated_root(tmp_path)
        binding = WorkspaceBranchBindingAuthority(
            environment_name="remote-env",
            binding_generation="failed-cleanup-generation",
            binding_identity="failed-cleanup-binding",
        )
        resolver = _DurableTestAuthorityProvider(binding)
        authority = WorkspaceBranchAuthority(
            session_id="failed-cleanup-session",
            expected_run_epoch=47,
            environment_name=binding.environment_name,
            binding_generation=binding.binding_generation,
            binding_identity=binding.binding_identity,
            creating_authority="failed-cleanup-worker",
            resource_policy="remote-branch-defaults",
        )
        source = _workspace(root, resolver=resolver)
        limits = WorkspaceBranchLimits(max_active_branches=1)
        created = await source.create_branch(
            await _request(
                source,
                limits=limits,
                authority=authority,
                branch_id="failed-cleanup-branch",
                idempotency_key="failed-cleanup-create",
            )
        )
        assert created.branch is not None
        await created.branch.write_bytes("original.txt", b"must-roll-back")
        changes = await created.branch.changes()
        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program.replace(
                progress_marker,
                'raise OSError("injected publication failure")',
            ).replace(
                cleanup_marker,
                'raise OSError("injected failed cleanup")',
            ),
        )
        with pytest.raises(WorkspaceBranchFencedError, match="operation failed"):
            await created.branch.publish(
                WorkspaceBranchPublicationRequest(
                    branch_id=created.branch.branch_id,
                    baseline_revision=changes.baseline_revision,
                    change_set_digest=changes.digest,
                    idempotency_key="failed-cleanup-publish",
                    expected_run_epoch=authority.expected_run_epoch,
                    binding_generation=authority.binding_generation,
                )
            )
        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program,
        )

        recovered = await _workspace(root, resolver=resolver).recover_branch(
            WorkspaceBranchRecoveryRequest(
                branch_id=created.branch.branch_id,
                session_id=authority.session_id,
                expected_run_epoch=authority.expected_run_epoch,
                binding_generation=authority.binding_generation,
                binding_identity=authority.binding_identity,
                recovery_id="failed-cleanup-recover",
            )
        )
        assert recovered.state.value == "failed"
        private = _private_branch_directory(root)
        assert not (private / "baseline").exists()
        assert not (private / "overlay").exists()
        replacement = await source.create_branch(
            await _request(
                source,
                limits=limits,
                authority=authority,
                branch_id="failed-cleanup-replacement",
                idempotency_key="failed-cleanup-replacement-create",
            )
        )
        assert replacement.status is WorkspaceBranchOutcomeStatus.CREATED

    asyncio.run(scenario())


def test_remote_branch_list_overflow_is_typed_and_pageable(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = tmp_path / "workspace"
        root.mkdir()
        for index in range(40):
            (root / f"long-baseline-file-name-{index:03d}.txt").write_bytes(b"value")
        source = _workspace(root)
        created = await source.create_branch(
            await _request(
                source,
                limits=WorkspaceBranchLimits(max_evidence_bytes=1024),
            )
        )
        assert created.branch is not None

        with pytest.raises(
            WorkspaceBranchResourceExhaustedError,
            match="list_evidence_limit_exceeded",
        ):
            await created.branch.list()
        page = await created.branch.list(limit=1)
        assert len(page.paths) == 1
        assert page.total_count == 40
        assert page.truncated is True

    asyncio.run(scenario())


def test_remote_creation_resource_failure_is_retryable_without_orphaning_capacity(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        root = _populated_root(tmp_path)
        source = _workspace(root)
        request = await _request(
            source,
            limits=WorkspaceBranchLimits(max_file_bytes=3, max_active_branches=1),
        )

        first = await source.create_branch(request)
        second = await source.create_branch(request)

        assert first.status is WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED
        assert second.status is WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED
        assert (
            first.evidence.detail_code == second.evidence.detail_code == "file_byte_limit_exceeded"
        )
        assert not tuple(root.parent.glob(".cayu-workspace-branch-*"))

    asyncio.run(scenario())


def test_malformed_retained_material_does_not_corrupt_creation_protocol(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = _populated_root(tmp_path)
        malformed = root.parent / ".cayu-workspace-branch-malformed"
        malformed.mkdir()
        (malformed / "record.json").write_text("not-json", encoding="utf-8")
        source = _workspace(root)

        result = await source.create_branch(
            await _request(source, limits=WorkspaceBranchLimits(max_active_branches=1))
        )

        assert result.status is WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED
        assert result.evidence.detail_code == "active_branch_limit_exceeded"

    asyncio.run(scenario())


def test_process_local_creation_acknowledgement_loss_reclaims_capacity_at_expiry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.workspaces import _runner_branch

    original_run = _runner_branch._run
    create_calls = 0

    async def lose_first_create_acknowledgement(*args, **kwargs):
        nonlocal create_calls
        result = await original_run(*args, **kwargs)
        if args[1] == "create":
            create_calls += 1
            if create_calls == 1:
                raise RuntimeError("injected acknowledgement loss")
        return result

    monkeypatch.setattr(
        _runner_branch,
        "_run",
        lose_first_create_acknowledgement,
    )

    async def scenario() -> None:
        root = _populated_root(tmp_path)
        source = _workspace(root)
        limits = WorkspaceBranchLimits(lifetime_ms=1, max_active_branches=1)

        with pytest.raises(RuntimeError, match="acknowledgement loss"):
            await source.create_branch(await _request(source, limits=limits))
        await asyncio.sleep(0.01)

        replacement = await source.create_branch(await _request(source, limits=limits))
        assert replacement.status is WorkspaceBranchOutcomeStatus.CREATED
        assert replacement.branch is not None
        states = sorted(
            json.loads((path / "record.json").read_text(encoding="utf-8"))["state"]
            for path in root.parent.glob(".cayu-workspace-branch-*.stage")
            if path.is_dir()
        )
        assert states == ["expired", "open"]

    asyncio.run(scenario())


def test_process_local_creation_process_loss_reclaims_staging_capacity_at_expiry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.workspaces import _runner_branch

    marker = "# CAYU_TEST_AFTER_CREATION_RECORD"
    original_program = _runner_branch.RUNNER_WORKSPACE_BRANCH_PROGRAM
    assert marker in original_program
    monkeypatch.setattr(
        _runner_branch,
        "RUNNER_WORKSPACE_BRANCH_PROGRAM",
        original_program.replace(marker, "os._exit(86)"),
    )

    async def scenario() -> None:
        root = _populated_root(tmp_path)
        source = _workspace(root)
        limits = WorkspaceBranchLimits(lifetime_ms=1, max_active_branches=1)

        with pytest.raises(WorkspaceBranchFencedError):
            await source.create_branch(await _request(source, limits=limits))
        staged = tuple(root.parent.glob(".cayu-workspace-branch-*.stage"))
        assert len(staged) == 1
        assert json.loads((staged[0] / "record.json").read_text(encoding="utf-8"))["state"] == (
            "creating"
        )

        await asyncio.sleep(0.01)
        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program,
        )

        replacement = await source.create_branch(await _request(source, limits=limits))
        assert replacement.status is WorkspaceBranchOutcomeStatus.CREATED
        assert replacement.branch is not None
        states = sorted(
            json.loads((path / "record.json").read_text(encoding="utf-8"))["state"]
            for path in root.parent.glob(".cayu-workspace-branch-*.stage")
            if path.is_dir()
        )
        assert states == ["expired", "open"]

    asyncio.run(scenario())


def test_exact_durable_creation_retry_recovers_a_recordless_staging_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.workspaces import _runner_branch

    marker = "# CAYU_TEST_AFTER_CREATION_DIRECTORY"
    original_program = _runner_branch.RUNNER_WORKSPACE_BRANCH_PROGRAM
    assert marker in original_program

    async def scenario() -> None:
        root = _populated_root(tmp_path)
        binding = WorkspaceBranchBindingAuthority(
            environment_name="remote-env",
            binding_generation="creation-generation",
            binding_identity="creation-allocation",
        )
        resolver = _DurableTestAuthorityProvider(binding)
        authority = WorkspaceBranchAuthority(
            session_id="creation-session",
            expected_run_epoch=5,
            environment_name=binding.environment_name,
            binding_generation=binding.binding_generation,
            binding_identity=binding.binding_identity,
            creating_authority="creation-worker",
            resource_policy="remote-branch-defaults",
        )
        source = _workspace(root, resolver=resolver)
        request = await _request(
            source,
            authority=authority,
            branch_id="recordless-creation",
            idempotency_key="recordless-create",
            limits=WorkspaceBranchLimits(max_active_branches=1),
        )
        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program.replace(marker, "os._exit(86)"),
        )

        with pytest.raises(WorkspaceBranchFencedError):
            await source.create_branch(request)
        staged = tuple(root.parent.glob(".cayu-workspace-branch-*.stage"))
        claims = tuple(root.parent.glob(".cayu-workspace-branch-*.stage.claim"))
        assert len(staged) == len(claims) == 1
        assert not (staged[0] / "record.json").exists()
        creation_claim = json.loads(claims[0].read_text(encoding="utf-8"))
        await asyncio.sleep(0.01)

        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program,
        )
        replayed = await source.create_branch(request)

        assert replayed.status is WorkspaceBranchOutcomeStatus.CREATED
        assert replayed.branch is not None
        assert len(tuple(root.parent.glob(".cayu-workspace-branch-*.stage"))) == 1
        assert len(tuple(root.parent.glob(".cayu-workspace-branch-*.stage.claim"))) == 1
        assert len(tuple(root.parent.glob(".cayu-workspace-branch-*.stage.claim.bound"))) == 1
        record = json.loads((staged[0] / "record.json").read_text(encoding="utf-8"))
        assert record["state"] == "open"
        assert record["created_at_ms"] == creation_claim["created_at_ms"]
        assert record["expires_at_ms"] == creation_claim["expires_at_ms"]

    asyncio.run(scenario())


def test_recordless_creation_claim_releases_capacity_at_expiry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.workspaces import _runner_branch

    marker = "# CAYU_TEST_AFTER_CREATION_DIRECTORY"
    original_program = _runner_branch.RUNNER_WORKSPACE_BRANCH_PROGRAM
    monkeypatch.setattr(
        _runner_branch,
        "RUNNER_WORKSPACE_BRANCH_PROGRAM",
        original_program.replace(marker, "os._exit(86)"),
    )

    async def scenario() -> None:
        root = _populated_root(tmp_path)
        source = _workspace(root)
        limits = WorkspaceBranchLimits(lifetime_ms=1, max_active_branches=1)

        with pytest.raises(WorkspaceBranchFencedError):
            await source.create_branch(await _request(source, limits=limits))
        await asyncio.sleep(0.01)
        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program,
        )

        replacement = await source.create_branch(await _request(source, limits=limits))

        assert replacement.status is WorkspaceBranchOutcomeStatus.CREATED
        assert replacement.branch is not None
        states = sorted(
            json.loads((path / "record.json").read_text(encoding="utf-8"))["state"]
            for path in root.parent.glob(".cayu-workspace-branch-*.stage")
            if path.is_dir()
        )
        assert states == ["expired", "open"]

    asyncio.run(scenario())


def test_exact_creation_retry_preserves_a_replacement_staging_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.workspaces import _runner_branch

    marker = "# CAYU_TEST_AFTER_CREATION_DIRECTORY"
    original_program = _runner_branch.RUNNER_WORKSPACE_BRANCH_PROGRAM

    async def scenario() -> None:
        root = _populated_root(tmp_path)
        binding = WorkspaceBranchBindingAuthority(
            environment_name="remote-env",
            binding_generation="replacement-generation",
            binding_identity="replacement-allocation",
        )
        resolver = _DurableTestAuthorityProvider(binding)
        authority = WorkspaceBranchAuthority(
            session_id="replacement-session",
            expected_run_epoch=5,
            environment_name=binding.environment_name,
            binding_generation=binding.binding_generation,
            binding_identity=binding.binding_identity,
            creating_authority="replacement-worker",
            resource_policy="remote-branch-defaults",
        )
        source = _workspace(root, resolver=resolver)
        request = await _request(
            source,
            authority=authority,
            branch_id="replaced-recordless-stage",
            idempotency_key="replaced-recordless-create",
        )
        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program.replace(marker, "os._exit(86)"),
        )

        with pytest.raises(WorkspaceBranchFencedError):
            await source.create_branch(request)
        staged = tuple(root.parent.glob(".cayu-workspace-branch-*.stage"))
        assert len(staged) == 1
        displaced = staged[0].with_name(staged[0].name + ".displaced")
        staged[0].rename(displaced)
        staged[0].mkdir()
        victim = staged[0] / "unrelated" / "must-survive.txt"
        victim.parent.mkdir()
        victim.write_text("unowned", encoding="utf-8")

        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program,
        )
        with pytest.raises(WorkspaceBranchFencedError):
            await source.create_branch(request)

        assert victim.read_text(encoding="utf-8") == "unowned"
        assert displaced.is_dir()
        assert tuple(root.parent.glob(".cayu-workspace-branch-*.stage.claim"))

    asyncio.run(scenario())


def test_expired_creation_claim_preserves_a_replacement_staging_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.workspaces import _runner_branch

    marker = "# CAYU_TEST_AFTER_CREATION_DIRECTORY"
    original_program = _runner_branch.RUNNER_WORKSPACE_BRANCH_PROGRAM
    monkeypatch.setattr(
        _runner_branch,
        "RUNNER_WORKSPACE_BRANCH_PROGRAM",
        original_program.replace(marker, "os._exit(86)"),
    )

    async def scenario() -> None:
        root = _populated_root(tmp_path)
        source = _workspace(root)
        limits = WorkspaceBranchLimits(lifetime_ms=1, max_active_branches=1)

        with pytest.raises(WorkspaceBranchFencedError):
            await source.create_branch(await _request(source, limits=limits))
        staged = tuple(root.parent.glob(".cayu-workspace-branch-*.stage"))
        assert len(staged) == 1
        displaced = staged[0].with_name(staged[0].name + ".displaced")
        staged[0].rename(displaced)
        staged[0].mkdir()
        victim = staged[0] / "unrelated" / "must-survive.txt"
        victim.parent.mkdir()
        victim.write_text("unowned", encoding="utf-8")
        await asyncio.sleep(0.01)

        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program,
        )
        result = await source.create_branch(await _request(source, limits=limits))
        assert result.status is WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED
        assert result.evidence.detail_code == "active_branch_limit_exceeded"

        assert victim.read_text(encoding="utf-8") == "unowned"
        assert displaced.is_dir()
        assert tuple(root.parent.glob(".cayu-workspace-branch-*.stage.claim"))

    asyncio.run(scenario())


def test_remote_creation_bounds_a_file_that_grows_after_stat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.workspaces import _runner_branch

    marker = "# CAYU_TEST_AFTER_SOURCE_FILE_STAT"
    original_program = _runner_branch.RUNNER_WORKSPACE_BRANCH_PROGRAM
    assert marker in original_program
    monkeypatch.setattr(
        _runner_branch,
        "RUNNER_WORKSPACE_BRANCH_PROGRAM",
        original_program.replace(
            marker,
            "race_fd = os.open(name, os.O_WRONLY | os.O_APPEND, dir_fd=directory_fd); "
            'os.write(race_fd, b"xx"); os.close(race_fd)',
        ),
    )

    async def scenario() -> None:
        root = tmp_path / "workspace"
        root.mkdir()
        (root / "source.txt").write_bytes(b"abc")
        source = _workspace(root)
        result = await source.create_branch(
            await _request(
                source,
                limits=WorkspaceBranchLimits(max_file_bytes=3),
            )
        )
        assert result.status is WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED
        assert result.evidence.detail_code == "file_byte_limit_exceeded"
        assert not tuple(root.parent.glob(".cayu-workspace-branch-*"))

    asyncio.run(scenario())


def test_concurrent_remote_publications_serialize_to_commit_and_conflict(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        source = _workspace(_populated_root(tmp_path))
        first = await source.create_branch(await _request(source))
        second = await source.create_branch(await _request(source))
        assert first.branch is not None
        assert second.branch is not None
        await first.branch.write_bytes("original.txt", b"first-winner")
        await second.branch.write_bytes("original.txt", b"second-winner")
        first_changes = await first.branch.changes()
        second_changes = await second.branch.changes()

        results = await asyncio.gather(
            first.branch.publish(
                WorkspaceBranchPublicationRequest(
                    branch_id=first.branch.branch_id,
                    baseline_revision=first_changes.baseline_revision,
                    change_set_digest=first_changes.digest,
                )
            ),
            second.branch.publish(
                WorkspaceBranchPublicationRequest(
                    branch_id=second.branch.branch_id,
                    baseline_revision=second_changes.baseline_revision,
                    change_set_digest=second_changes.digest,
                )
            ),
        )
        assert sorted(result.status.value for result in results) == ["committed", "conflicted"]
        assert (root_content := (await source.read_bytes("original.txt")).content) in {
            b"first-winner",
            b"second-winner",
        }
        assert root_content != b"original"

    asyncio.run(scenario())


def test_durable_runner_branch_recovers_in_fresh_workspace_and_commits(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = _populated_root(tmp_path)
        binding = WorkspaceBranchBindingAuthority(
            environment_name="remote-env",
            binding_generation="generation-1",
            binding_identity="allocation-binding-1",
        )
        resolver = _DurableTestAuthorityProvider(binding)
        authority = WorkspaceBranchAuthority(
            session_id="session-1",
            expected_run_epoch=3,
            environment_name=binding.environment_name,
            binding_generation=binding.binding_generation,
            binding_identity=binding.binding_identity,
            creating_authority="worker-1",
            resource_policy="remote-branch-defaults",
        )
        source = _workspace(root, resolver=resolver)
        request = await _request(
            source,
            authority=authority,
            branch_id="durable-remote-branch",
            idempotency_key="create-1",
        )
        created = await source.create_branch(request)
        assert created.branch is not None
        await created.branch.write_bytes("original.txt", b"fresh-process")
        change_set = await created.branch.changes()

        fresh = _workspace(root, resolver=resolver)
        recovery_request = WorkspaceBranchRecoveryRequest(
            branch_id="durable-remote-branch",
            session_id="session-1",
            expected_run_epoch=3,
            binding_generation=binding.binding_generation,
            binding_identity=binding.binding_identity,
            recovery_id="recover-1",
        )
        recovered = await fresh.recover_branch(recovery_request)
        assert recovered.branch is not None
        publication = await recovered.branch.publish(
            WorkspaceBranchPublicationRequest(
                branch_id="durable-remote-branch",
                baseline_revision=change_set.baseline_revision,
                change_set_digest=change_set.digest,
                idempotency_key="publish-1",
                expected_run_epoch=3,
                binding_generation=binding.binding_generation,
            )
        )
        assert publication.status is WorkspaceBranchOutcomeStatus.COMMITTED
        assert (await fresh.read_bytes("original.txt")).content == b"fresh-process"

        terminal = await _workspace(root, resolver=resolver).recover_branch(recovery_request)
        assert terminal.publication == publication

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("branch_id", "different-branch"),
        ("source.workspace_id", "different-source"),
        ("authority.session_id", "different-session"),
        ("authority.expected_run_epoch", 99),
        ("authority.environment_name", "different-environment"),
        ("authority.binding_generation", "different-generation"),
        ("authority.binding_identity", "different-binding"),
        ("authority.creating_authority", "different-worker"),
        ("authority.resource_policy", "different-policy"),
    ),
)
def test_runner_branch_recovery_authenticates_complete_returned_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: str | int,
) -> None:
    async def scenario() -> None:
        root = _populated_root(tmp_path)
        binding = WorkspaceBranchBindingAuthority(
            environment_name="remote-env",
            binding_generation="recovery-generation",
            binding_identity="recovery-binding",
        )
        resolver = _DurableTestAuthorityProvider(binding)
        authority = WorkspaceBranchAuthority(
            session_id="recovery-session",
            expected_run_epoch=31,
            environment_name=binding.environment_name,
            binding_generation=binding.binding_generation,
            binding_identity=binding.binding_identity,
            creating_authority="recovery-worker",
            resource_policy="remote-branch-defaults",
        )
        resolver.authorize_operation(authority)
        source = _workspace(root, resolver=resolver)
        response: dict[str, object] = {
            "ok": True,
            "state": "open",
            "branch_id": "recovery-branch",
            "source": {
                "workspace_id": source.id,
                "observer": "runner-branch-tests",
            },
            "baseline_revision": "sha256:" + "1" * 64,
            "limits": WorkspaceBranchLimits().model_dump(mode="json"),
            "authority": authority.model_dump(mode="json"),
            "detail_code": "remote_workspace_branch_created",
        }
        owner, _, nested = field.partition(".")
        if nested:
            child = dict(response[owner])
            child[nested] = replacement
            response[owner] = child
        else:
            response[owner] = replacement

        async def forged_exec(command, **kwargs):
            del command, kwargs
            return ExecResult(stdout=json.dumps(response), exit_code=0)

        monkeypatch.setattr(source._runner, "_exec", forged_exec)
        with pytest.raises(WorkspaceBranchFencedError, match="recovery result"):
            await source.recover_branch(
                WorkspaceBranchRecoveryRequest(
                    branch_id="recovery-branch",
                    session_id=authority.session_id,
                    expected_run_epoch=authority.expected_run_epoch,
                    binding_generation=binding.binding_generation,
                    binding_identity=binding.binding_identity,
                    recovery_id="recovery-request",
                )
            )

    asyncio.run(scenario())


def test_recovery_rejects_changed_complete_authority_before_publication_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.workspaces import _runner_branch

    marker = "# CAYU_TEST_AFTER_PUBLICATION_INTENT"
    original_program = _runner_branch.RUNNER_WORKSPACE_BRANCH_PROGRAM
    assert marker in original_program

    async def scenario() -> None:
        root = _populated_root(tmp_path)
        binding = WorkspaceBranchBindingAuthority(
            environment_name="remote-env",
            binding_generation="recovery-preflight-generation",
            binding_identity="recovery-preflight-binding",
        )
        resolver = _DurableTestAuthorityProvider(binding)
        authority = WorkspaceBranchAuthority(
            session_id="recovery-preflight-session",
            expected_run_epoch=37,
            environment_name=binding.environment_name,
            binding_generation=binding.binding_generation,
            binding_identity=binding.binding_identity,
            creating_authority="recovery-preflight-worker",
            resource_policy="remote-branch-defaults",
        )
        source = _workspace(root, resolver=resolver)
        created = await source.create_branch(
            await _request(
                source,
                authority=authority,
                branch_id="recovery-preflight-branch",
                idempotency_key="recovery-preflight-create",
            )
        )
        assert created.branch is not None
        await created.branch.write_bytes("original.txt", b"must-not-publish")
        changes = await created.branch.changes()
        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program.replace(marker, 'raise OSError("injected publication loss")'),
        )
        with pytest.raises(WorkspaceBranchFencedError, match="operation failed"):
            await created.branch.publish(
                WorkspaceBranchPublicationRequest(
                    branch_id=created.branch.branch_id,
                    baseline_revision=changes.baseline_revision,
                    change_set_digest=changes.digest,
                    idempotency_key="recovery-preflight-publish",
                    expected_run_epoch=authority.expected_run_epoch,
                    binding_generation=authority.binding_generation,
                )
            )

        record_path = _private_branch_directory(root) / "record.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        assert record["state"] == "publication_intent"
        record["authority"]["creating_authority"] = "forged-worker"
        record.pop("integrity", None)
        encoded_record = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        record["integrity"] = f"sha256:{hashlib.sha256(encoded_record).hexdigest()}"
        record_path.write_text(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program,
        )

        with pytest.raises(
            WorkspaceBranchOperationConflict,
            match="workspace_branch_recovery_authority_changed",
        ):
            await _workspace(root, resolver=resolver).recover_branch(
                WorkspaceBranchRecoveryRequest(
                    branch_id=created.branch.branch_id,
                    session_id=authority.session_id,
                    expected_run_epoch=authority.expected_run_epoch,
                    binding_generation=authority.binding_generation,
                    binding_identity=authority.binding_identity,
                    recovery_id="recovery-preflight-recover",
                )
            )
        assert (root / "original.txt").read_bytes() == b"original"

    asyncio.run(scenario())


def test_durable_runner_branch_rejects_a_replaced_run_epoch_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        root = _populated_root(tmp_path)
        binding = WorkspaceBranchBindingAuthority(
            environment_name="remote-env",
            binding_generation="stable-generation",
            binding_identity="stable-binding",
        )
        resolver = _DurableTestAuthorityProvider(binding)
        authority = WorkspaceBranchAuthority(
            session_id="stale-run-session",
            expected_run_epoch=7,
            environment_name=binding.environment_name,
            binding_generation=binding.binding_generation,
            binding_identity=binding.binding_identity,
            creating_authority="stale-run-worker",
            resource_policy="remote-branch-defaults",
        )
        source = _workspace(root, resolver=resolver)
        created = await source.create_branch(
            await _request(
                source,
                authority=authority,
                branch_id="stale-run-branch",
                idempotency_key="stale-run-create",
            )
        )
        assert created.branch is not None
        replacement = authority.model_copy(
            update={
                "expected_run_epoch": 8,
                "creating_authority": "replacement-worker",
            }
        )
        resolver.authorize_operation(replacement)

        async def unexpected_exec(command, **kwargs):
            del command, kwargs
            pytest.fail("stale run authority reached remote dispatch")

        monkeypatch.setattr(source._runner, "_exec", unexpected_exec)
        with pytest.raises(WorkspaceBranchOperationConflict, match="no longer current"):
            await created.branch.write_bytes("stale.txt", b"must-not-land")
        with pytest.raises(WorkspaceBranchOperationConflict, match="no longer current"):
            await source.recover_branch(
                WorkspaceBranchRecoveryRequest(
                    branch_id=created.branch.branch_id,
                    session_id=authority.session_id,
                    expected_run_epoch=authority.expected_run_epoch,
                    binding_generation=authority.binding_generation,
                    binding_identity=authority.binding_identity,
                    recovery_id="stale-run-recovery",
                )
            )

        assert created.branch.lifecycle_status is WorkspaceBranchLifecycleStatus.FENCED
        assert not (root / "stale.txt").exists()

    asyncio.run(scenario())


def test_replacement_run_epoch_recovers_open_branch_and_owns_guest_operations(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        from cayu.workspaces import _runner_branch

        root = _populated_root(tmp_path)
        binding = WorkspaceBranchBindingAuthority(
            environment_name="remote-env",
            binding_generation="replacement-generation",
            binding_identity="replacement-binding",
        )
        resolver = _DurableTestAuthorityProvider(binding)
        creating = WorkspaceBranchAuthority(
            session_id="replacement-session",
            expected_run_epoch=7,
            environment_name=binding.environment_name,
            binding_generation=binding.binding_generation,
            binding_identity=binding.binding_identity,
            creating_authority="creating-worker",
            resource_policy="remote-branch-defaults",
        )
        source = _workspace(root, resolver=resolver)
        created = await source.create_branch(
            await _request(
                source,
                authority=creating,
                branch_id="replacement-open-branch",
                idempotency_key="replacement-open-create",
            )
        )
        assert created.branch is not None
        await created.branch.write_bytes("original.txt", b"replacement-private")
        record_path = _private_branch_directory(root) / "record.json"
        original_record = json.loads(record_path.read_text(encoding="utf-8"))

        replacement = creating.model_copy(
            update={
                "expected_run_epoch": 8,
                "creating_authority": "replacement-worker",
            }
        )
        resolver.authorize_operation(replacement)
        recovery = WorkspaceBranchRecoveryRequest(
            branch_id=created.branch.branch_id,
            session_id=replacement.session_id,
            expected_run_epoch=replacement.expected_run_epoch,
            binding_generation=replacement.binding_generation,
            binding_identity=replacement.binding_identity,
            recovery_id="replacement-open-recovery",
        )
        recovered = await _workspace(root, resolver=resolver).recover_branch(recovery)
        assert recovered.branch is not None
        await recovered.branch.write_bytes("replacement.txt", b"replacement-owned")

        transferred_record = json.loads(record_path.read_text(encoding="utf-8"))
        assert transferred_record["authority"] == creating.model_dump(mode="json")
        assert transferred_record["operation_authority"] == replacement.model_dump(mode="json")
        assert transferred_record["source"] == original_record["source"]
        assert transferred_record["source_root"] == original_record["source_root"]
        assert transferred_record["baseline_revision"] == original_record["baseline_revision"]
        assert (
            transferred_record["allocation_fingerprint"]
            == (original_record["allocation_fingerprint"])
        )

        with pytest.raises(WorkspaceBranchOperationConflict, match="no longer current"):
            await created.branch.read_bytes("original.txt")

        capability = source._branch_capability
        assert capability is not None
        with pytest.raises(
            WorkspaceBranchOperationConflict,
            match="workspace_branch_operation_authority_changed",
        ):
            await _runner_branch._run(
                source,
                "read",
                {
                    "branch_id": created.branch.branch_id,
                    "allocation_fingerprint": capability.allocation_fingerprint,
                    "binding_authority": binding.model_dump(mode="json"),
                    "operation_authority": creating.model_dump(mode="json"),
                    "path": "original.txt",
                    "offset": 0,
                    "limit": 64,
                },
                output_limit_bytes=16 * 1024,
                settlement=_runner_branch._RunnerOperationSettlement(),
            )

        rolled_back = await recovered.branch.rollback(
            WorkspaceBranchRollbackRequest(
                branch_id=created.branch.branch_id,
                idempotency_key="replacement-open-rollback",
                expected_run_epoch=replacement.expected_run_epoch,
                binding_generation=replacement.binding_generation,
            )
        )
        assert rolled_back.status is WorkspaceBranchOutcomeStatus.ROLLED_BACK
        assert (root / "original.txt").read_bytes() == b"original"
        assert not (root / "replacement.txt").exists()

    asyncio.run(scenario())


def test_replacement_run_epoch_recovery_survives_loss_after_authority_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.workspaces import _runner_branch

    publication_marker = "# CAYU_TEST_AFTER_PUBLICATION_INTENT"
    recovery_marker = "# CAYU_TEST_AFTER_RECOVERY_AUTHORITY_HANDOFF"
    original_program = _runner_branch.RUNNER_WORKSPACE_BRANCH_PROGRAM
    assert publication_marker in original_program
    assert recovery_marker in original_program

    async def scenario() -> None:
        root = _populated_root(tmp_path)
        binding = WorkspaceBranchBindingAuthority(
            environment_name="remote-env",
            binding_generation="handoff-generation",
            binding_identity="handoff-binding",
        )
        resolver = _DurableTestAuthorityProvider(binding)
        creating = WorkspaceBranchAuthority(
            session_id="handoff-session",
            expected_run_epoch=12,
            environment_name=binding.environment_name,
            binding_generation=binding.binding_generation,
            binding_identity=binding.binding_identity,
            creating_authority="handoff-creating-worker",
            resource_policy="remote-branch-defaults",
        )
        source = _workspace(root, resolver=resolver)
        created = await source.create_branch(
            await _request(
                source,
                authority=creating,
                branch_id="handoff-publication-branch",
                idempotency_key="handoff-publication-create",
            )
        )
        assert created.branch is not None
        await created.branch.write_bytes("original.txt", b"handoff-published")
        changes = await created.branch.changes()
        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program.replace(
                publication_marker,
                'raise OSError("injected publication owner loss")',
            ),
        )
        with pytest.raises(WorkspaceBranchFencedError, match="operation failed"):
            await created.branch.publish(
                WorkspaceBranchPublicationRequest(
                    branch_id=created.branch.branch_id,
                    baseline_revision=changes.baseline_revision,
                    change_set_digest=changes.digest,
                    idempotency_key="handoff-publication-publish",
                    expected_run_epoch=creating.expected_run_epoch,
                    binding_generation=creating.binding_generation,
                )
            )

        replacement = creating.model_copy(
            update={
                "expected_run_epoch": 13,
                "creating_authority": "handoff-replacement-worker",
            }
        )
        resolver.authorize_operation(replacement)
        recovery = WorkspaceBranchRecoveryRequest(
            branch_id=created.branch.branch_id,
            session_id=replacement.session_id,
            expected_run_epoch=replacement.expected_run_epoch,
            binding_generation=replacement.binding_generation,
            binding_identity=replacement.binding_identity,
            recovery_id="handoff-publication-recovery",
        )
        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program.replace(
                recovery_marker,
                'raise OSError("injected recovery owner loss")',
            ),
        )
        with pytest.raises(WorkspaceBranchFencedError, match="operation failed"):
            await _workspace(root, resolver=resolver).recover_branch(recovery)

        record_path = _private_branch_directory(root) / "record.json"
        interrupted_record = json.loads(record_path.read_text(encoding="utf-8"))
        assert interrupted_record["state"] == "publication_intent"
        assert interrupted_record["authority"] == creating.model_dump(mode="json")
        assert interrupted_record["operation_authority"] == replacement.model_dump(mode="json")
        assert (root / "original.txt").read_bytes() == b"original"

        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program,
        )
        recovered = await _workspace(root, resolver=resolver).recover_branch(recovery)
        assert recovered.publication is not None
        assert recovered.publication.status is WorkspaceBranchOutcomeStatus.COMMITTED
        assert (root / "original.txt").read_bytes() == b"handoff-published"

    asyncio.run(scenario())


def test_failed_binding_claim_release_is_retained_and_retried(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = _populated_root(tmp_path)
        binding = WorkspaceBranchBindingAuthority(
            environment_name="remote-env",
            binding_generation="generation-release-retry",
            binding_identity="allocation-release-retry",
        )
        resolver = _FailOnceReleaseAuthorityProvider(binding)
        authority = WorkspaceBranchAuthority(
            session_id="session-release-retry",
            expected_run_epoch=4,
            environment_name=binding.environment_name,
            binding_generation=binding.binding_generation,
            binding_identity=binding.binding_identity,
            creating_authority="worker-release-retry",
            resource_policy="remote-branch-defaults",
        )
        source = _workspace(root, resolver=resolver)
        request = await _request(
            source,
            authority=authority,
            branch_id="durable-release-retry",
            idempotency_key="create-release-retry",
        )

        with pytest.raises(RuntimeError, match="claim release failure"):
            await source.create_branch(request)
        with pytest.raises(WorkspaceBranchOperationConflict):
            resolver.replace(
                binding.model_copy(update={"binding_generation": "replacement-generation"})
            )

        replayed = await source.create_branch(request)
        assert replayed.status is WorkspaceBranchOutcomeStatus.CREATED
        assert replayed.branch is not None
        resolver.replace(
            binding.model_copy(update={"binding_generation": "replacement-generation"})
        )

    asyncio.run(scenario())


def test_private_mutation_claim_release_failure_fences_until_recovery(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = _populated_root(tmp_path)
        binding = WorkspaceBranchBindingAuthority(
            environment_name="remote-env",
            binding_generation="mutation-release-generation",
            binding_identity="mutation-release-binding",
        )
        resolver = _FailOnceReleaseAuthorityProvider(binding)
        resolver._wrap_next_claim = False
        authority = WorkspaceBranchAuthority(
            session_id="mutation-release-session",
            expected_run_epoch=43,
            environment_name=binding.environment_name,
            binding_generation=binding.binding_generation,
            binding_identity=binding.binding_identity,
            creating_authority="mutation-release-worker",
            resource_policy="remote-branch-defaults",
        )
        source = _workspace(root, resolver=resolver)
        created = await source.create_branch(
            await _request(
                source,
                authority=authority,
                branch_id="mutation-release-branch",
                idempotency_key="mutation-release-create",
            )
        )
        assert created.branch is not None
        resolver._wrap_next_claim = True

        with pytest.raises(RuntimeError, match="claim release failure"):
            await created.branch.create_bytes("created.txt", b"created-before-release-failure")
        assert created.branch.lifecycle_status is WorkspaceBranchLifecycleStatus.FENCED

        recovered = await _workspace(root, resolver=resolver).recover_branch(
            WorkspaceBranchRecoveryRequest(
                branch_id=created.branch.branch_id,
                session_id=authority.session_id,
                expected_run_epoch=authority.expected_run_epoch,
                binding_generation=authority.binding_generation,
                binding_identity=authority.binding_identity,
                recovery_id="mutation-release-recovery",
            )
        )
        assert recovered.branch is not None
        assert (await recovered.branch.read_bytes("created.txt")).content == (
            b"created-before-release-failure"
        )

    asyncio.run(scenario())


def test_unsettled_remote_command_claim_is_not_released_by_later_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.workspaces import _runner_branch

    async def scenario() -> None:
        root = _populated_root(tmp_path)
        binding = WorkspaceBranchBindingAuthority(
            environment_name="remote-env",
            binding_generation="unsettled-generation",
            binding_identity="unsettled-binding",
        )
        resolver = _TrackNextReleaseAuthorityProvider(binding)
        authority = WorkspaceBranchAuthority(
            session_id="unsettled-session",
            expected_run_epoch=44,
            environment_name=binding.environment_name,
            binding_generation=binding.binding_generation,
            binding_identity=binding.binding_identity,
            creating_authority="unsettled-worker",
            resource_policy="remote-branch-defaults",
        )
        source = _workspace(root, resolver=resolver)
        created = await source.create_branch(
            await _request(
                source,
                authority=authority,
                branch_id="unsettled-branch",
                idempotency_key="unsettled-create",
            )
        )
        assert created.branch is not None

        original_exec = source._runner._exec
        original_settlement = _runner_branch.runner_workspace_mutation_settlement
        fail_next = True

        async def fail_once(command, **kwargs):
            nonlocal fail_next
            if fail_next:
                fail_next = False
                raise RuntimeError("injected unknown command outcome")
            return await original_exec(command, **kwargs)

        def classify_unknown(*, result, error):
            if isinstance(error, RuntimeError) and str(error) == "injected unknown command outcome":
                return "uncertain"
            return original_settlement(result=result, error=error)

        resolver.track_next_claim = True
        monkeypatch.setattr(source._runner, "_exec", fail_once)
        monkeypatch.setattr(
            _runner_branch,
            "runner_workspace_mutation_settlement",
            classify_unknown,
        )
        with pytest.raises(RuntimeError, match="unknown command outcome"):
            await created.branch.write_bytes("unsettled.txt", b"possibly-written")

        tracked = resolver.tracked_claim
        assert tracked is not None
        assert tracked.release_calls == 0
        recovered = await _workspace(root, resolver=resolver).recover_branch(
            WorkspaceBranchRecoveryRequest(
                branch_id=created.branch.branch_id,
                session_id=authority.session_id,
                expected_run_epoch=authority.expected_run_epoch,
                binding_generation=authority.binding_generation,
                binding_identity=authority.binding_identity,
                recovery_id="unsettled-recovery",
            )
        )
        assert recovered.branch is not None
        assert tracked.release_calls == 0
        with pytest.raises(WorkspaceBranchOperationConflict):
            resolver.replace(
                binding.model_copy(update={"binding_generation": "replacement-generation"})
            )

        retained = _runner_branch._RETAINED_BINDING_CLAIMS.pop(id(tracked))
        assert retained is tracked
        tracked.release()

    asyncio.run(scenario())


def test_durable_runner_branch_recovers_rollback_and_replays_terminal_evidence(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        root = _populated_root(tmp_path)
        binding = WorkspaceBranchBindingAuthority(
            environment_name="remote-env",
            binding_generation="generation-1",
            binding_identity="allocation-binding-1",
        )
        resolver = _DurableTestAuthorityProvider(binding)
        authority = WorkspaceBranchAuthority(
            session_id="session-rollback",
            expected_run_epoch=5,
            environment_name=binding.environment_name,
            binding_generation=binding.binding_generation,
            binding_identity=binding.binding_identity,
            creating_authority="worker-1",
            resource_policy="remote-branch-defaults",
        )
        source = _workspace(root, resolver=resolver)
        created = await source.create_branch(
            await _request(
                source,
                authority=authority,
                branch_id="durable-rollback",
                idempotency_key="create-rollback",
            )
        )
        assert created.branch is not None
        await created.branch.write_bytes("original.txt", b"private-only")

        recovery_request = WorkspaceBranchRecoveryRequest(
            branch_id="durable-rollback",
            session_id=authority.session_id,
            expected_run_epoch=authority.expected_run_epoch,
            binding_generation=binding.binding_generation,
            binding_identity=binding.binding_identity,
            recovery_id="recover-rollback",
        )
        recovered = await _workspace(root, resolver=resolver).recover_branch(recovery_request)
        assert recovered.branch is not None
        rollback = await recovered.branch.rollback(
            WorkspaceBranchRollbackRequest(
                branch_id="durable-rollback",
                idempotency_key="rollback-1",
                expected_run_epoch=authority.expected_run_epoch,
                binding_generation=binding.binding_generation,
                reason="explicit",
            )
        )
        assert rollback.status is WorkspaceBranchOutcomeStatus.ROLLED_BACK
        assert (root / "original.txt").read_bytes() == b"original"

        terminal = await _workspace(root, resolver=resolver).recover_branch(recovery_request)
        assert terminal.rollback == rollback

    asyncio.run(scenario())


def test_durable_rollback_intent_rejects_conflicting_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.workspaces import _runner_branch

    marker = "# CAYU_TEST_AFTER_ROLLBACK_INTENT"
    original_program = _runner_branch.RUNNER_WORKSPACE_BRANCH_PROGRAM
    assert marker in original_program

    async def scenario() -> None:
        root = _populated_root(tmp_path)
        binding = WorkspaceBranchBindingAuthority(
            environment_name="remote-env",
            binding_generation="rollback-intent-generation",
            binding_identity="rollback-intent-binding",
        )
        resolver = _DurableTestAuthorityProvider(binding)
        authority = WorkspaceBranchAuthority(
            session_id="rollback-intent-session",
            expected_run_epoch=12,
            environment_name=binding.environment_name,
            binding_generation=binding.binding_generation,
            binding_identity=binding.binding_identity,
            creating_authority="rollback-intent-worker",
            resource_policy="remote-branch-defaults",
        )
        source = _workspace(root, resolver=resolver)
        created = await source.create_branch(
            await _request(
                source,
                authority=authority,
                branch_id="rollback-intent-branch",
                idempotency_key="rollback-intent-create",
            )
        )
        assert created.branch is not None
        await created.branch.write_bytes("original.txt", b"private-value")
        accepted = WorkspaceBranchRollbackRequest(
            branch_id=created.branch.branch_id,
            idempotency_key="rollback-intent-accepted",
            expected_run_epoch=authority.expected_run_epoch,
            binding_generation=authority.binding_generation,
            reason="explicit",
        )
        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program.replace(marker, 'raise OSError("lost rollback acknowledgement")'),
        )

        with pytest.raises(WorkspaceBranchFencedError, match="operation failed"):
            await created.branch.rollback(accepted)

        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program,
        )
        with pytest.raises(WorkspaceBranchOperationConflict, match="identity_reused"):
            await created.branch.rollback(
                accepted.model_copy(update={"idempotency_key": "conflicting-rollback"})
            )
        settled = await created.branch.rollback(accepted)
        assert settled.status is WorkspaceBranchOutcomeStatus.ROLLED_BACK
        assert (root / "original.txt").read_bytes() == b"original"

    asyncio.run(scenario())


def test_durable_rollback_rejects_a_different_branch_identity(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = _populated_root(tmp_path)
        binding = WorkspaceBranchBindingAuthority(
            environment_name="remote-env",
            binding_generation="generation-rollback-identity",
            binding_identity="allocation-rollback-identity",
        )
        resolver = _DurableTestAuthorityProvider(binding)
        authority = WorkspaceBranchAuthority(
            session_id="session-rollback-identity",
            expected_run_epoch=6,
            environment_name=binding.environment_name,
            binding_generation=binding.binding_generation,
            binding_identity=binding.binding_identity,
            creating_authority="worker-rollback-identity",
            resource_policy="remote-branch-defaults",
        )
        source = _workspace(root, resolver=resolver)
        created = await source.create_branch(
            await _request(
                source,
                authority=authority,
                branch_id="durable-rollback-identity",
                idempotency_key="create-rollback-identity",
            )
        )
        assert created.branch is not None
        await created.branch.write_bytes("original.txt", b"private-only")

        with pytest.raises(WorkspaceBranchOperationConflict, match="authority_changed"):
            await created.branch.rollback(
                WorkspaceBranchRollbackRequest(
                    branch_id="different-branch",
                    idempotency_key="rollback-wrong-branch",
                    expected_run_epoch=authority.expected_run_epoch,
                    binding_generation=binding.binding_generation,
                    reason="explicit",
                )
            )

        assert (await created.branch.read_bytes("original.txt")).content == b"private-only"
        assert (root / "original.txt").read_bytes() == b"original"

    asyncio.run(scenario())


def test_mutated_remote_branch_requests_fail_before_serialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    canary = "secret-from-mutated-workspace-branch-request"

    async def scenario() -> list[BaseException]:
        root = _populated_root(tmp_path)
        binding = WorkspaceBranchBindingAuthority(
            environment_name="remote-env",
            binding_generation="generation-defensive-copy",
            binding_identity="allocation-defensive-copy",
        )
        resolver = _DurableTestAuthorityProvider(binding)
        authority = WorkspaceBranchAuthority(
            session_id="session-defensive-copy",
            expected_run_epoch=23,
            environment_name=binding.environment_name,
            binding_generation=binding.binding_generation,
            binding_identity=binding.binding_identity,
            creating_authority="worker-defensive-copy",
            resource_policy="remote-branch-defaults",
        )
        source = _workspace(root, resolver=resolver)
        creation_request = await _request(
            source,
            authority=authority,
            branch_id="durable-defensive-copy",
            idempotency_key="create-defensive-copy",
        )
        created = await source.create_branch(creation_request)
        assert created.branch is not None
        await created.branch.write_bytes("original.txt", b"private-value")
        changes = await created.branch.changes()

        async def unexpected_exec(command, **kwargs):
            del command, kwargs
            pytest.fail("invalid caller-owned request reached runner dispatch")

        monkeypatch.setattr(source._runner, "_exec", unexpected_exec)
        wrong_type = _SecretBearingWrongType(canary)
        errors: list[BaseException] = []

        malformed_creation = creation_request.model_copy()
        object.__setattr__(malformed_creation, "branch_id", wrong_type)
        with pytest.raises(TypeError) as raised_creation:
            await source.create_branch(malformed_creation)
        errors.append(raised_creation.value)

        malformed_publication = WorkspaceBranchPublicationRequest(
            branch_id=created.branch.branch_id,
            baseline_revision=changes.baseline_revision,
            change_set_digest=changes.digest,
            idempotency_key="publish-defensive-copy",
            expected_run_epoch=authority.expected_run_epoch,
            binding_generation=binding.binding_generation,
        )
        object.__setattr__(malformed_publication, "change_set_digest", wrong_type)
        with pytest.raises(TypeError) as raised_publication:
            await created.branch.publish(malformed_publication)
        errors.append(raised_publication.value)

        malformed_rollback = WorkspaceBranchRollbackRequest(
            branch_id=created.branch.branch_id,
            idempotency_key="rollback-defensive-copy",
            expected_run_epoch=authority.expected_run_epoch,
            binding_generation=binding.binding_generation,
            reason="explicit",
        )
        object.__setattr__(malformed_rollback, "reason", wrong_type)
        with pytest.raises(TypeError) as raised_rollback:
            await created.branch.rollback(malformed_rollback)
        errors.append(raised_rollback.value)

        malformed_recovery = WorkspaceBranchRecoveryRequest(
            branch_id=created.branch.branch_id,
            session_id=authority.session_id,
            expected_run_epoch=authority.expected_run_epoch,
            binding_generation=binding.binding_generation,
            binding_identity=binding.binding_identity,
            recovery_id="recover-defensive-copy",
        )
        object.__setattr__(malformed_recovery, "recovery_id", wrong_type)
        with pytest.raises(TypeError) as raised_recovery:
            await source.recover_branch(malformed_recovery)
        errors.append(raised_recovery.value)
        return errors

    errors = asyncio.run(scenario())
    captured = capsys.readouterr()
    diagnostic_output = "\n".join(
        (
            repr(errors),
            caplog.text,
            captured.out,
            captured.err,
            *(str(warning.message) for warning in recwarn),
        )
    )
    assert canary not in diagnostic_output


def test_runner_branch_stale_binding_is_fenced_before_mutation(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = _populated_root(tmp_path)
        original = WorkspaceBranchBindingAuthority(
            environment_name="remote-env",
            binding_generation="generation-1",
            binding_identity="allocation-binding-1",
        )
        resolver = _DurableTestAuthorityProvider(original)
        authority = WorkspaceBranchAuthority(
            session_id="session-stale",
            expected_run_epoch=7,
            environment_name=original.environment_name,
            binding_generation=original.binding_generation,
            binding_identity=original.binding_identity,
            creating_authority="worker-1",
            resource_policy="remote-branch-defaults",
        )
        source = _workspace(root, resolver=resolver)
        created = await source.create_branch(
            await _request(
                source,
                authority=authority,
                branch_id="durable-stale",
                idempotency_key="create-stale",
            )
        )
        assert created.branch is not None
        resolver.replace(
            WorkspaceBranchBindingAuthority(
                environment_name="remote-env",
                binding_generation="generation-2",
                binding_identity="allocation-binding-2",
            )
        )

        with pytest.raises(WorkspaceBranchOperationConflict, match="no longer current"):
            await created.branch.write_bytes("original.txt", b"must-not-land")
        assert created.branch.lifecycle_status is WorkspaceBranchLifecycleStatus.FENCED
        assert (root / "original.txt").read_bytes() == b"original"

    asyncio.run(scenario())


def test_runner_branch_corruption_fails_closed_without_leaking_record_content(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        root = _populated_root(tmp_path)
        binding = WorkspaceBranchBindingAuthority(
            environment_name="remote-env",
            binding_generation="generation-1",
            binding_identity="allocation-binding-1",
        )
        resolver = _DurableTestAuthorityProvider(binding)
        authority = WorkspaceBranchAuthority(
            session_id="session-corrupt",
            expected_run_epoch=9,
            environment_name=binding.environment_name,
            binding_generation=binding.binding_generation,
            binding_identity=binding.binding_identity,
            creating_authority="worker-1",
            resource_policy="remote-branch-defaults",
        )
        source = _workspace(root, resolver=resolver)
        created = await source.create_branch(
            await _request(
                source,
                authority=authority,
                branch_id="durable-corrupt",
                idempotency_key="create-corrupt",
            )
        )
        assert created.branch is not None
        (_private_branch_directory(root) / "record.json").write_text(
            '{"private":"sensitive-branch-content"}',
            encoding="utf-8",
        )
        recovery = WorkspaceBranchRecoveryRequest(
            branch_id="durable-corrupt",
            session_id=authority.session_id,
            expected_run_epoch=authority.expected_run_epoch,
            binding_generation=binding.binding_generation,
            binding_identity=binding.binding_identity,
            recovery_id="recover-corrupt",
        )

        with pytest.raises(WorkspaceBranchFencedError) as raised:
            await _workspace(root, resolver=resolver).recover_branch(recovery)
        assert "sensitive-branch-content" not in str(raised.value)

    asyncio.run(scenario())


def test_runner_branch_invalid_command_output_is_not_retained_in_exception_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canary = "provider-command-output-secret-canary"

    async def scenario() -> None:
        source = _workspace(_populated_root(tmp_path))
        created = await source.create_branch(await _request(source))
        assert created.branch is not None

        async def malformed_exec(command, **kwargs):
            del command, kwargs
            return ExecResult(stdout="not-json:" + canary, stderr=canary, exit_code=1)

        monkeypatch.setattr(source._runner, "_exec", malformed_exec)
        with pytest.raises(WorkspaceBranchFencedError) as raised:
            await created.branch.changes()

        pending: list[BaseException | None] = [raised.value]
        representations: list[str] = []
        seen: set[int] = set()
        while pending:
            error = pending.pop()
            if error is None or id(error) in seen:
                continue
            seen.add(id(error))
            representations.extend((str(error), repr(error)))
            pending.extend((error.__cause__, error.__context__))
        assert canary not in "\n".join(representations)

    asyncio.run(scenario())


def test_runner_branch_expiry_cleans_private_contents_and_publication_attempts_are_bounded(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        expiry_root = _populated_root(tmp_path / "expiry")
        expiry_source = _workspace(expiry_root)
        expiring = await expiry_source.create_branch(
            await _request(expiry_source, limits=WorkspaceBranchLimits(lifetime_ms=10))
        )
        assert expiring.branch is not None
        await asyncio.sleep(0.03)
        with pytest.raises(WorkspaceBranchClosedError, match="expired"):
            await expiring.branch.read_bytes("original.txt")
        assert expiring.branch.lifecycle_status is WorkspaceBranchLifecycleStatus.ROLLED_BACK
        private = _private_branch_directory(expiry_root)
        assert not (private / "baseline").exists()
        assert not (private / "overlay").exists()

        bounded_root = _populated_root(tmp_path / "bounded")
        bounded_source = _workspace(bounded_root)
        created = await bounded_source.create_branch(
            await _request(
                bounded_source,
                limits=WorkspaceBranchLimits(max_publication_attempts=1),
            )
        )
        assert created.branch is not None
        await created.branch.write_bytes("original.txt", b"branch-change")
        first_changes = await created.branch.changes()
        await bounded_source.write_bytes("original.txt", b"external-change")
        first = await created.branch.publish(
            WorkspaceBranchPublicationRequest(
                branch_id=created.branch.branch_id,
                baseline_revision=first_changes.baseline_revision,
                change_set_digest=first_changes.digest,
            )
        )
        assert first.status is WorkspaceBranchOutcomeStatus.CONFLICTED
        await bounded_source.write_bytes("original.txt", b"original")
        await created.branch.write_bytes("original.txt", b"second-branch-change")
        second_changes = await created.branch.changes()
        with pytest.raises(
            WorkspaceBranchResourceExhaustedError,
            match="publication_attempt_limit_exceeded",
        ):
            await created.branch.publish(
                WorkspaceBranchPublicationRequest(
                    branch_id=created.branch.branch_id,
                    baseline_revision=second_changes.baseline_revision,
                    change_set_digest=second_changes.digest,
                )
            )

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["publish", "rollback"])
def test_expired_runner_branch_stays_terminal_through_lifecycle_operations(
    tmp_path: Path,
    operation: str,
) -> None:
    async def scenario() -> None:
        root = _populated_root(tmp_path / operation)
        source = _workspace(root)
        created = await source.create_branch(await _request(source))
        assert created.branch is not None
        await created.branch.write_bytes("original.txt", b"branch-change")
        changes = await created.branch.changes()
        record_path = _private_branch_directory(root) / "record.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["expires_at_ms"] = 0
        record.pop("integrity", None)
        encoded_record = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        record["integrity"] = f"sha256:{hashlib.sha256(encoded_record).hexdigest()}"
        record_path.write_text(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )

        if operation == "publish":
            with pytest.raises(WorkspaceBranchClosedError, match="expired"):
                await created.branch.publish(
                    WorkspaceBranchPublicationRequest(
                        branch_id=created.branch.branch_id,
                        baseline_revision=changes.baseline_revision,
                        change_set_digest=changes.digest,
                    )
                )
        else:
            result = await created.branch.rollback()
            assert result.status is WorkspaceBranchOutcomeStatus.ROLLED_BACK

        assert created.branch.lifecycle_status is WorkspaceBranchLifecycleStatus.ROLLED_BACK

    asyncio.run(scenario())


def test_cancelled_publication_recovers_from_durable_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.workspaces import _runner_branch

    marker = "# CAYU_TEST_AFTER_PUBLICATION_INTENT"
    assert marker in _runner_branch.RUNNER_WORKSPACE_BRANCH_PROGRAM
    monkeypatch.setattr(
        _runner_branch,
        "RUNNER_WORKSPACE_BRANCH_PROGRAM",
        _runner_branch.RUNNER_WORKSPACE_BRANCH_PROGRAM.replace(marker, "time.sleep(30)"),
    )

    async def scenario() -> None:
        root = _populated_root(tmp_path)
        binding = WorkspaceBranchBindingAuthority(
            environment_name="remote-env",
            binding_generation="generation-1",
            binding_identity="allocation-binding-1",
        )
        resolver = _DurableTestAuthorityProvider(binding)
        authority = WorkspaceBranchAuthority(
            session_id="session-cancel",
            expected_run_epoch=11,
            environment_name=binding.environment_name,
            binding_generation=binding.binding_generation,
            binding_identity=binding.binding_identity,
            creating_authority="worker-1",
            resource_policy="remote-branch-defaults",
        )
        source = _workspace(root, resolver=resolver)
        created = await source.create_branch(
            await _request(
                source,
                authority=authority,
                branch_id="durable-cancel",
                idempotency_key="create-cancel",
            )
        )
        assert created.branch is not None
        await created.branch.write_bytes("original.txt", b"commit-after-cancel")
        changes = await created.branch.changes()
        publication_request = WorkspaceBranchPublicationRequest(
            branch_id=created.branch.branch_id,
            baseline_revision=changes.baseline_revision,
            change_set_digest=changes.digest,
            idempotency_key="publish-cancel",
            expected_run_epoch=authority.expected_run_epoch,
            binding_generation=binding.binding_generation,
        )
        publication_task = asyncio.create_task(created.branch.publish(publication_request))
        record_path = _private_branch_directory(root) / "record.json"
        for _ in range(200):
            await asyncio.sleep(0.01)
            try:
                if json.loads(record_path.read_text(encoding="utf-8"))["state"] == (
                    "publication_intent"
                ):
                    break
            except (OSError, ValueError, KeyError):
                continue
        else:
            pytest.fail("publication did not persist its intent before cancellation")
        publication_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await publication_task

        recovery_request = WorkspaceBranchRecoveryRequest(
            branch_id="durable-cancel",
            session_id=authority.session_id,
            expected_run_epoch=authority.expected_run_epoch,
            binding_generation=binding.binding_generation,
            binding_identity=binding.binding_identity,
            recovery_id="recover-cancel",
        )
        recovered = await _workspace(root, resolver=resolver).recover_branch(recovery_request)
        assert recovered.publication is not None
        assert recovered.publication.status is WorkspaceBranchOutcomeStatus.COMMITTED
        assert (root / "original.txt").read_bytes() == b"commit-after-cancel"

    asyncio.run(scenario())


def test_cancelled_remote_command_retains_binding_claim_until_quiescent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        root = _populated_root(tmp_path)
        binding = WorkspaceBranchBindingAuthority(
            environment_name="remote-env",
            binding_generation="settlement-generation",
            binding_identity="settlement-allocation",
        )
        resolver = _DurableTestAuthorityProvider(binding)
        authority = WorkspaceBranchAuthority(
            session_id="settlement-session",
            expected_run_epoch=19,
            environment_name=binding.environment_name,
            binding_generation=binding.binding_generation,
            binding_identity=binding.binding_identity,
            creating_authority="settlement-worker",
            resource_policy="remote-branch-defaults",
        )
        runner = _BranchRunner(root, inherit_env=False)
        source = RunnerWorkspace(
            runner,
            workspace_id="remote-source",
            python_executable=sys.executable,
            enable_workspace_branches=True,
            branch_authority_resolver=resolver,
        )
        created = await source.create_branch(
            await _request(
                source,
                authority=authority,
                branch_id="settlement-branch",
                idempotency_key="settlement-create",
            )
        )
        assert created.branch is not None
        await created.branch.write_bytes("original.txt", b"branch-value")
        changes = await created.branch.changes()

        dispatched = asyncio.Event()
        settlement_started = asyncio.Event()
        quiescent = asyncio.Event()

        async def deferred_exec(command, **kwargs):
            del command, kwargs
            dispatched.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as cancellation:
                attach_cancellation_artifacts(
                    cancellation,
                    [
                        {
                            "type": "cayu.runner_cleanup.v1",
                            "action": "kill_command",
                            "status": "deferred",
                        }
                    ],
                )
                raise

        async def await_settlement() -> bool:
            settlement_started.set()
            await quiescent.wait()
            return True

        monkeypatch.setattr(runner, "_exec", deferred_exec)
        monkeypatch.setattr(runner, "await_pending_command_settlement", await_settlement)
        monkeypatch.setattr(
            _BranchRunner,
            "pending_command_settlement_cancellation_safe",
            True,
            raising=False,
        )
        publication = asyncio.create_task(
            created.branch.publish(
                WorkspaceBranchPublicationRequest(
                    branch_id=created.branch.branch_id,
                    baseline_revision=changes.baseline_revision,
                    change_set_digest=changes.digest,
                    idempotency_key="settlement-publish",
                    expected_run_epoch=authority.expected_run_epoch,
                    binding_generation=binding.binding_generation,
                )
            )
        )
        await dispatched.wait()
        publication.cancel("stop branch publication")
        await settlement_started.wait()
        assert publication.cancelling() == 1
        with pytest.raises(WorkspaceBranchOperationConflict):
            resolver.replace(
                binding.model_copy(update={"binding_generation": "replacement-generation"})
            )

        quiescent.set()
        with pytest.raises(asyncio.CancelledError, match="stop branch publication"):
            await publication
        assert publication.cancelled()
        resolver.replace(
            binding.model_copy(update={"binding_generation": "replacement-generation"})
        )

    asyncio.run(scenario())


def test_timed_out_publication_recovers_from_durable_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.workspaces import _runner_branch

    marker = "# CAYU_TEST_AFTER_PUBLICATION_INTENT"
    monkeypatch.setattr(
        _runner_branch,
        "RUNNER_WORKSPACE_BRANCH_PROGRAM",
        _runner_branch.RUNNER_WORKSPACE_BRANCH_PROGRAM.replace(marker, "time.sleep(30)"),
    )

    async def scenario() -> None:
        root = _populated_root(tmp_path)
        binding = WorkspaceBranchBindingAuthority(
            environment_name="remote-env",
            binding_generation="generation-1",
            binding_identity="allocation-binding-1",
        )
        resolver = _DurableTestAuthorityProvider(binding)
        authority = WorkspaceBranchAuthority(
            session_id="session-timeout",
            expected_run_epoch=13,
            environment_name=binding.environment_name,
            binding_generation=binding.binding_generation,
            binding_identity=binding.binding_identity,
            creating_authority="worker-1",
            resource_policy="remote-branch-defaults",
        )
        source = RunnerWorkspace(
            _BranchRunner(root, inherit_env=False),
            workspace_id="remote-source",
            python_executable=sys.executable,
            enable_workspace_branches=True,
            branch_operation_timeout_s=1,
            branch_authority_resolver=resolver,
        )
        created = await source.create_branch(
            await _request(
                source,
                authority=authority,
                branch_id="durable-timeout",
                idempotency_key="create-timeout",
            )
        )
        assert created.branch is not None
        await created.branch.write_bytes("original.txt", b"commit-after-timeout")
        changes = await created.branch.changes()
        with pytest.raises(WorkspaceBranchFencedError, match="timed_out"):
            await created.branch.publish(
                WorkspaceBranchPublicationRequest(
                    branch_id=created.branch.branch_id,
                    baseline_revision=changes.baseline_revision,
                    change_set_digest=changes.digest,
                    idempotency_key="publish-timeout",
                    expected_run_epoch=authority.expected_run_epoch,
                    binding_generation=binding.binding_generation,
                )
            )

        recovery_request = WorkspaceBranchRecoveryRequest(
            branch_id="durable-timeout",
            session_id=authority.session_id,
            expected_run_epoch=authority.expected_run_epoch,
            binding_generation=binding.binding_generation,
            binding_identity=binding.binding_identity,
            recovery_id="recover-timeout",
        )
        recovered = await _workspace(root, resolver=resolver).recover_branch(recovery_request)
        assert recovered.publication is not None
        assert recovered.publication.status is WorkspaceBranchOutcomeStatus.COMMITTED
        assert (root / "original.txt").read_bytes() == b"commit-after-timeout"

    asyncio.run(scenario())


def test_timed_out_creation_is_reconstructed_by_exact_fresh_owner_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.workspaces import _runner_branch

    marker = "# CAYU_TEST_AFTER_BRANCH_CAPTURE"
    original_program = _runner_branch.RUNNER_WORKSPACE_BRANCH_PROGRAM
    monkeypatch.setattr(
        _runner_branch,
        "RUNNER_WORKSPACE_BRANCH_PROGRAM",
        original_program.replace(marker, "time.sleep(30)"),
    )

    async def scenario() -> None:
        root = _populated_root(tmp_path)
        binding = WorkspaceBranchBindingAuthority(
            environment_name="remote-env",
            binding_generation="generation-1",
            binding_identity="allocation-binding-1",
        )
        resolver = _DurableTestAuthorityProvider(binding)
        authority = WorkspaceBranchAuthority(
            session_id="session-create-timeout",
            expected_run_epoch=15,
            environment_name=binding.environment_name,
            binding_generation=binding.binding_generation,
            binding_identity=binding.binding_identity,
            creating_authority="worker-1",
            resource_policy="remote-branch-defaults",
        )
        source = RunnerWorkspace(
            _BranchRunner(root, inherit_env=False),
            workspace_id="remote-source",
            python_executable=sys.executable,
            enable_workspace_branches=True,
            branch_operation_timeout_s=1,
            branch_authority_resolver=resolver,
        )
        request = await _request(
            source,
            authority=authority,
            branch_id="durable-create-timeout",
            idempotency_key="create-timeout",
        )
        with pytest.raises(WorkspaceBranchFencedError, match="timed_out"):
            await source.create_branch(request)

        monkeypatch.setattr(
            _runner_branch,
            "RUNNER_WORKSPACE_BRANCH_PROGRAM",
            original_program,
        )
        retried = await _workspace(root, resolver=resolver).create_branch(request)
        assert retried.status is WorkspaceBranchOutcomeStatus.CREATED
        assert retried.branch is not None
        assert (await retried.branch.read_bytes("original.txt")).content == b"original"
        assert len(tuple(root.parent.glob(".cayu-workspace-branch-*.stage"))) == 1
        assert len(tuple(root.parent.glob(".cayu-workspace-branch-*.stage.claim"))) == 1
        assert len(tuple(root.parent.glob(".cayu-workspace-branch-*.stage.claim.bound"))) == 1

    asyncio.run(scenario())
