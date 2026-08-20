from __future__ import annotations

from cayu.runtime.workspace_mutation_attribution import (
    DirectWorkspaceMutationCollector,
    begin_workspace_mutation_window,
    classify_workspace_mutation_attribution,
    direct_workspace_mutation_payload,
    observed_pre_window_change,
    reconcile_direct_workspace_mutations,
)
from cayu.workspaces import (
    LocalWorkspace,
    WorkspaceDirectMutationReconciliation,
    WorkspaceIdentity,
    WorkspaceMutationAttributionConfidence,
    WorkspaceMutationResult,
    WorkspacePathRevision,
    WorkspaceRevisionDeltaStatus,
    WorkspaceRevisionObservation,
    WorkspaceRevisionObservationStatus,
    WorkspaceWriterIsolationEvidence,
    WorkspaceWriterIsolationStatus,
)


def _observation(
    identity: WorkspaceIdentity,
    *,
    revision: str,
    paths: tuple[WorkspacePathRevision, ...] = (),
) -> WorkspaceRevisionObservation:
    return WorkspaceRevisionObservation(
        identity=identity,
        status=WorkspaceRevisionObservationStatus.SUPPORTED,
        revision=revision,
        paths=paths,
        total_paths=len(paths),
    )


def _exclusive_isolation() -> WorkspaceWriterIsolationEvidence:
    return WorkspaceWriterIsolationEvidence(
        status=WorkspaceWriterIsolationStatus.EXCLUSIVE,
        mechanism="test-lease",
        generation="generation-1",
        detail_code=None,
    )


def test_overlapping_windows_downgrade_both_exclusive_adapter_claims(tmp_path) -> None:
    workspace = LocalWorkspace(tmp_path, workspace_id="shared-workspace")
    first = begin_workspace_mutation_window(workspace, window_id="same-durable-window")
    second = begin_workspace_mutation_window(workspace, window_id="same-durable-window")
    try:
        for window in (first, second):
            attribution = classify_workspace_mutation_attribution(
                window=window,
                isolation_before=_exclusive_isolation(),
                isolation_after=_exclusive_isolation(),
                direct_reconciliation=WorkspaceDirectMutationReconciliation.CONSISTENT,
            )
            assert (
                attribution.confidence
                is WorkspaceMutationAttributionConfidence.CONCURRENT_AMBIGUITY
            )
            assert attribution.overlap_detected is True
            assert attribution.detail_code == "overlapping_workspace_mutation_windows"
    finally:
        first.close()
        second.close()


def test_edit_between_windows_is_retained_as_external_gap_evidence(tmp_path) -> None:
    workspace = LocalWorkspace(tmp_path, workspace_id="gap-workspace")
    identity = WorkspaceIdentity(workspace_id=workspace.id, observer="test-observer")
    first_after = _observation(
        identity,
        revision="revision-1",
        paths=(WorkspacePathRevision(path="file.txt", content_sha256="a" * 64),),
    )
    first = begin_workspace_mutation_window(workspace, window_id="window-gap-1")
    first.close(first_after)

    second_before = _observation(
        identity,
        revision="revision-2",
        paths=(WorkspacePathRevision(path="file.txt", content_sha256="b" * 64),),
    )
    second = begin_workspace_mutation_window(workspace, window_id="window-gap-2")
    try:
        gap = observed_pre_window_change(second, second_before)
        assert gap is not None
        assert gap.status is WorkspaceRevisionDeltaStatus.CHANGED
        assert [(path.path, path.change) for path in gap.paths] == [("file.txt", "modified")]
    finally:
        second.close(second_before)


def test_quarantined_window_discards_process_local_gap_history(tmp_path) -> None:
    workspace = LocalWorkspace(tmp_path, workspace_id="quarantined-history-workspace")
    identity = WorkspaceIdentity(workspace_id=workspace.id, observer="test-observer")
    prior_observation = _observation(identity, revision="revision-before-private-window")
    prior = begin_workspace_mutation_window(workspace, window_id="window-before-private")
    prior.close(prior_observation)

    private = begin_workspace_mutation_window(workspace, window_id="private-window")
    assert private.prior_observation == prior_observation
    private.close(discard_history=True)

    following = begin_workspace_mutation_window(workspace, window_id="window-after-private")
    try:
        assert following.prior_observation is None
    finally:
        following.close()


def test_direct_operation_contradiction_is_retained_and_downgrades_attribution(
    tmp_path,
) -> None:
    identity = WorkspaceIdentity(workspace_id="workspace-1", observer="test-observer")
    before = _observation(identity, revision="revision-before")
    after = _observation(
        identity,
        revision="revision-after",
        paths=(WorkspacePathRevision(path="created.txt", content_sha256="b" * 64),),
    )
    collector = DirectWorkspaceMutationCollector()
    collector.record(
        "create_bytes",
        "created.txt",
        WorkspaceMutationResult(
            operation="create",
            before_revision=None,
            after_revision="file-revision",
            before_sha256=None,
            after_sha256="a" * 64,
            before_bytes=None,
            after_bytes=1,
        ),
    )
    reconciliation = reconcile_direct_workspace_mutations(
        before=before,
        after=after,
        collector=collector,
    )
    assert reconciliation is WorkspaceDirectMutationReconciliation.CONTRADICTORY

    workspace = LocalWorkspace(tmp_path, workspace_id="workspace-1")
    window = begin_workspace_mutation_window(workspace, window_id="window-contradiction")
    try:
        attribution = classify_workspace_mutation_attribution(
            window=window,
            isolation_before=_exclusive_isolation(),
            isolation_after=_exclusive_isolation(),
            direct_reconciliation=reconciliation,
        )
        assert attribution.confidence is WorkspaceMutationAttributionConfidence.CONCURRENT_AMBIGUITY
        assert attribution.detail_code == "direct_and_observed_workspace_evidence_conflict"
    finally:
        window.close(after)


def test_direct_operation_chain_contradiction_is_not_hidden_by_matching_endpoints() -> None:
    identity = WorkspaceIdentity(workspace_id="workspace-chain", observer="test-observer")
    before = _observation(
        identity,
        revision="revision-before",
        paths=(WorkspacePathRevision(path="file.txt", content_sha256="a" * 64),),
    )
    after = _observation(
        identity,
        revision="revision-after",
        paths=(WorkspacePathRevision(path="file.txt", content_sha256="c" * 64),),
    )
    collector = DirectWorkspaceMutationCollector()
    collector.record(
        "replace_bytes",
        "file.txt",
        WorkspaceMutationResult(
            operation="replace",
            before_revision="revision-a",
            after_revision="revision-b",
            before_sha256="a" * 64,
            after_sha256="b" * 64,
            before_bytes=1,
            after_bytes=2,
        ),
    )
    collector.record(
        "replace_bytes",
        "file.txt",
        WorkspaceMutationResult(
            operation="replace",
            before_revision="revision-unrelated",
            after_revision="revision-c",
            before_sha256="d" * 64,
            after_sha256="c" * 64,
            before_bytes=4,
            after_bytes=3,
        ),
    )

    assert (
        reconcile_direct_workspace_mutations(
            before=before,
            after=after,
            collector=collector,
        )
        is WorkspaceDirectMutationReconciliation.CONTRADICTORY
    )


def test_direct_operation_projection_is_bounded_and_content_free() -> None:
    collector = DirectWorkspaceMutationCollector()
    for index in range(65):
        collector.record(
            "create_bytes",
            f"private/{index}.txt",
            WorkspaceMutationResult(
                operation="create",
                before_revision=None,
                after_revision=f"revision-{index}",
                before_sha256=None,
                after_sha256=f"{index:064x}",
                before_bytes=None,
                after_bytes=index,
            ),
        )

    payload = direct_workspace_mutation_payload(
        collector,
        window_id="window-bounded",
        evidence_available=True,
    )

    assert payload["total_operations"] == 65
    assert payload["retained_operations"] == 16
    assert payload["truncated"] is True
    assert all("private/" not in str(operation) for operation in payload["operations"])
    assert all("before_revision" not in operation for operation in payload["operations"])


def test_direct_operation_projection_is_fixed_when_evidence_is_quarantined() -> None:
    collector = DirectWorkspaceMutationCollector()
    private_path = "PRIVATE_ARGUMENT_PATH_CANARY.txt"
    collector.record(
        "create_bytes",
        private_path,
        WorkspaceMutationResult(
            operation="create",
            before_revision=None,
            after_revision="private-revision",
            before_sha256=None,
            after_sha256="a" * 64,
            before_bytes=None,
            after_bytes=7,
        ),
    )

    quarantined = direct_workspace_mutation_payload(
        collector,
        window_id="private-window",
        evidence_available=False,
    )
    empty = direct_workspace_mutation_payload(
        DirectWorkspaceMutationCollector(),
        window_id="different-window",
        evidence_available=False,
    )

    expected = {
        "operations": [],
        "retained_operations": 0,
        "total_operations": 0,
        "truncated": True,
    }
    assert quarantined == empty == expected
    assert private_path not in str(quarantined)


def test_direct_operation_collector_accepts_workspace_valid_surrounding_whitespace() -> None:
    collector = DirectWorkspaceMutationCollector()
    collector.record(
        "create_bytes",
        " leading.txt",
        WorkspaceMutationResult(
            operation="create",
            before_revision=None,
            after_revision="revision",
            before_sha256=None,
            after_sha256="b" * 64,
            before_bytes=None,
            after_bytes=7,
        ),
    )

    assert collector.total_operations == 1
    assert collector.truncated is False
    assert [operation.path for operation in collector.operations] == [" leading.txt"]
