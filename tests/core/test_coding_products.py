from __future__ import annotations

import asyncio
import hashlib

import pytest

import cayu.coding_products as coding_products
from cayu._validation import canonical_durable_json_bytes
from cayu.artifacts import ArtifactMetadata, ArtifactReadResult, ArtifactScope, LocalArtifactStore
from cayu.coding_products import (
    CODING_PRODUCT_EVIDENCE_KIND,
    CodingGitBaselineAuthority,
    CodingLifecycleReceipt,
    CodingProductAdmissionError,
    CodingProductArtifactRepository,
    CodingProductEvidenceError,
    CodingProductReconstructionRequiredError,
    CodingProductRequest,
    CodingProductRunner,
    CodingProductState,
    CodingReviewSettlement,
    CodingRuntimeAuthority,
    CodingSettlementPolicy,
    CodingSourceAuthority,
    CodingTaskAuthority,
    coding_product_completion_decision,
    collect_coding_product_events,
    compile_coding_product_candidate,
)
from cayu.core.events import Event, EventType
from cayu.core.messages import Message
from cayu.runtime.app import CayuApp
from cayu.runtime.sessions import InMemorySessionStore, RunRequest, session_input_messages_sha256
from cayu.runtime.work_contracts import (
    CompletionResultReference,
    CompletionVerdict,
    WorkEvidenceReference,
)
from cayu.vaults import REDACTED_SECRET
from cayu.workspaces import LocalWorkspace
from cayu.workspaces.revisions import (
    WorkspaceRevisionObservationLimits,
    observe_deterministic_workspace,
)


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _publication_receipt(
    *,
    destination_workspace_id: str = "source-workspace",
) -> dict[str, object]:
    snapshot_material: dict[str, object] = {
        "schema": "cayu.source_publication_snapshot.v1",
        "destination_workspace_id": destination_workspace_id,
        "workload_workspace_id": "docker-workspace",
        "source": "sync",
        "outcome": "completed",
        "source_conflict_policy": "require_revision",
        "sync_back": "always",
        "delete_missing": True,
        "copied_files": 1,
        "copied_bytes": 16,
        "deleted_files": 0,
    }
    material: dict[str, object] = {
        "schema": "cayu.source_publication_receipt.v1",
        "snapshot_sha256": "sha256:"
        + hashlib.sha256(
            canonical_durable_json_bytes(
                snapshot_material,
                "source_publication_snapshot",
            )
        ).hexdigest(),
        "destination_workspace_id": destination_workspace_id,
        "workload_workspace_id": "docker-workspace",
        "outcome": "completed",
        "source_conflict_policy": "require_revision",
        "sync_back": "always",
        "delete_missing": True,
        "copied_files": 1,
        "copied_bytes": 16,
        "deleted_files": 0,
    }
    return {
        **material,
        "receipt_sha256": "sha256:"
        + hashlib.sha256(
            canonical_durable_json_bytes(material, "source_publication_receipt")
        ).hexdigest(),
    }


def _request(*, baseline: str, reviewer_required: bool = False) -> CodingProductRequest:
    return CodingProductRequest(
        product_run_id="product-run-1",
        session_id="session-1",
        agent_name="coder",
        source=CodingSourceAuthority(
            origin_id="application-source-1",
            workspace_id="source-workspace",
            baseline_revision=baseline,
            destination_id="application-destination-1",
            git_baseline=CodingGitBaselineAuthority(
                head_revision="a" * 40,
                staged_entries_sha256=_digest("clean-index"),
                tracked_flags_sha256=_digest("clean-index-flags"),
                status_sha256=_digest("clean-status"),
                diff_sha256=_digest("clean-diff"),
            ),
            observation_limits=WorkspaceRevisionObservationLimits(),
        ),
        task=CodingTaskAuthority(
            task_id="task-1",
            instruction_sha256=_digest("repair the project"),
        ),
        runtime=CodingRuntimeAuthority(
            toolchain_profile_id="python",
            toolchain_profile_revision="1",
            toolchain_profile_fingerprint=_digest("toolchain"),
            image_fingerprint=_digest("image"),
            dependency_identity=_digest("dependencies"),
            execution_profile_fingerprint=_digest("execution"),
            tool_manifest_fingerprint=_digest("tools"),
            tool_policy_fingerprint=_digest("policy"),
            approval_policy_fingerprint=_digest("approval"),
            redaction_profile_fingerprint=_digest("redaction"),
        ),
        settlement=CodingSettlementPolicy(
            required_checks=("format", "lint", "test"),
            reviewer_required=reviewer_required,
        ),
    )


def _tool_event(
    tool_name: str,
    structured: dict[str, object],
    *,
    event_id: str,
    content: str = "bounded",
) -> Event:
    return Event(
        id=event_id,
        type=EventType.TOOL_CALL_COMPLETED,
        session_id="session-1",
        interaction_id="interaction-1",
        agent_name="coder",
        environment_name="coding",
        tool_name=tool_name,
        payload={
            "execution_profile_fingerprint": _digest("execution").removeprefix("sha256:"),
            "result": {"content": content, "structured": structured},
        },
    )


def _check_event(name: str, *, workspace_revision: str = _digest("[]")) -> Event:
    return _tool_event(
        "run_check",
        {
            "check": name,
            "check_profile_fingerprint": _digest(f"check:{name}"),
            "status": "passed",
            "exit_code": 0,
            "duration_ms": 1,
            "output_sha256": _digest(f"output:{name}"),
            "stdout_truncated": False,
            "stderr_truncated": False,
            "timed_out": False,
            "cancelled": False,
            "workspace_mutation_settlement": "runner_quiescent",
            "workspace_revision": workspace_revision,
        },
        event_id=f"check-{name}",
    )


def _command_event(
    *,
    status: str = "succeeded",
    exit_code: int = 0,
    exit_code_admitted: bool = True,
    output_collection_complete: bool = True,
    output_publication_complete: bool = True,
) -> Event:
    return _tool_event(
        "run_command",
        {
            "selector": "focused-test",
            "selector_fingerprint": _digest("selector:focused-test"),
            "argv_sha256": _digest("argv:focused-test"),
            "status": status,
            "exit_code": exit_code,
            "exit_code_admitted": exit_code_admitted,
            "duration_ms": 1,
            "output_sha256": _digest("output:focused-test"),
            "stdout_truncated": False,
            "stderr_truncated": False,
            "stdout_runner_truncated": False,
            "stderr_runner_truncated": False,
            "stdout_projection_truncated": False,
            "stderr_projection_truncated": False,
            "output_artifact_status": "not_requested",
            "output_collection_complete": output_collection_complete,
            "output_publication_complete": output_publication_complete,
            "timed_out": False,
            "cancelled": False,
            "workspace_mutation_settlement": "runner_quiescent",
        },
        event_id="command-focused-test",
    )


def _git_events(*, changed: bool) -> tuple[Event, Event, Event]:
    status_change = {
        "path": "example.py",
        "index": " ",
        "worktree": "M",
        "original_path": None,
    }
    changes = [status_change] if changed else []
    summary_changes = (
        [{**status_change, "additions": 1, "deletions": 1, "count_kind": "text"}] if changed else []
    )
    return (
        _tool_event(
            "git_changes",
            {
                "mode": "status",
                "scope": "all",
                "changes": changes,
                "returned": len(changes),
                "offset": 0,
                "limit": 200,
                "truncated": False,
                "next_offset": None,
                "truncation_reasons": [],
            },
            event_id="git-status-1",
        ),
        _tool_event(
            "git_changes",
            {
                "mode": "summary",
                "scope": "all",
                "changes": summary_changes,
                "returned": len(summary_changes),
                "offset": 0,
                "limit": 200,
                "truncated": False,
                "next_offset": None,
                "truncation_reasons": [],
            },
            event_id="git-summary-1",
        ),
        _tool_event(
            "git_changes",
            {
                "mode": "diff",
                "scope": "all",
                "changes": changes,
                "returned": len(changes),
                "offset": 0,
                "limit": 200,
                "diff_offset": 0,
                "truncated": False,
                "next_offset": None,
                "next_diff_offset": None,
                "truncation_reasons": [],
                "binary_omitted": False,
            },
            event_id="git-diff-1",
            content=("--- a/example.py\n+++ b/example.py\n" if changed else ""),
        ),
    )


def _final_git_receipt(
    *,
    request: CodingProductRequest,
    workspace_revision: str,
    changed: bool,
) -> dict[str, object]:
    status_event, summary_event, diff_event = _git_events(changed=changed)
    status_result = status_event.payload["result"]
    summary_result = summary_event.payload["result"]
    diff_result = diff_event.payload["result"]
    assert type(status_result) is dict
    assert type(summary_result) is dict
    assert type(diff_result) is dict
    material: dict[str, object] = {
        "schema": "cayu.final_git_receipt.v1",
        "request_fingerprint": request.fingerprint,
        "destination_workspace_id": "source-workspace",
        "workload_workspace_id": "docker-workspace",
        "baseline_revision": request.source.baseline_revision,
        "workspace_revision": workspace_revision,
        "status": {"structured": status_result["structured"]},
        "summary": {"structured": summary_result["structured"]},
        "diff": {
            "content": diff_result["content"],
            "structured": diff_result["structured"],
        },
    }
    return _seal_final_git_receipt(material)


def _seal_final_git_receipt(material: dict[str, object]) -> dict[str, object]:
    return {
        **material,
        "receipt_sha256": "sha256:"
        + hashlib.sha256(canonical_durable_json_bytes(material, "final_git_receipt")).hexdigest(),
    }


def _terminal_events(
    *,
    request: CodingProductRequest,
    workspace_revision: str,
    changed: bool = False,
) -> tuple[Event, Event]:
    final_git_receipt = _final_git_receipt(
        request=request,
        workspace_revision=workspace_revision,
        changed=changed,
    )
    return (
        Event(
            id="publication-1",
            type=EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED,
            session_id="session-1",
            interaction_id="interaction-1",
            agent_name="coder",
            environment_name="coding",
            payload={
                "execution_profile_fingerprint": _digest("execution").removeprefix("sha256:"),
                "source_publication_receipt": _publication_receipt(),
                "final_git_receipt": final_git_receipt,
                "final_snapshot": {
                    "snapshot_id": "snapshot-1",
                    "metadata": {
                        "copied_files": 1,
                        "copied_bytes": 16,
                        "deleted_files": 0,
                    },
                },
            },
        ),
        Event(
            id="terminal-1",
            type=EventType.SESSION_COMPLETED,
            session_id="session-1",
            interaction_id="interaction-1",
            agent_name="coder",
            environment_name="coding",
            payload={"execution_profile_fingerprint": _digest("execution").removeprefix("sha256:")},
        ),
    )


def test_compile_patch_ready_candidate_and_verify_exact_evidence(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "example.py").write_text("before\n", encoding="utf-8")
    workspace = LocalWorkspace(source, workspace_id="source-workspace")
    limits = WorkspaceRevisionObservationLimits()
    initial = asyncio.run(
        observe_deterministic_workspace(
            workspace,
            observer="cayu-coding-product-source",
            limits=limits,
        )
    )
    assert initial.revision is not None
    request = _request(baseline=initial.revision)
    (source / "example.py").write_text("after\n", encoding="utf-8")
    final = asyncio.run(
        observe_deterministic_workspace(
            workspace,
            observer="cayu-coding-product-source",
            limits=limits,
        )
    )
    assert final.revision is not None
    final_revision = final.revision
    repository = CodingProductArtifactRepository(
        LocalArtifactStore(tmp_path / "artifacts", store_id="coding-product-test")
    )
    events = (
        *(
            _check_event(name, workspace_revision=final_revision)
            for name in request.settlement.required_checks
        ),
        *_terminal_events(
            request=request,
            workspace_revision=final_revision,
            changed=True,
        ),
    )

    candidate = asyncio.run(
        compile_coding_product_candidate(
            request,
            events,
            initial_observation=initial,
            final_observation=final,
            repository=repository,
        )
    )

    assert candidate.state is CodingProductState.PATCH_READY_FOR_DELIVERY
    assert candidate.interaction_id == "interaction-1"
    assert candidate.initial_source.revision == initial.revision
    assert candidate.final_source.revision == final.revision
    assert candidate.publication.outcome == "copied"
    assert all(check.duration_ms == 1 for check in candidate.checks)
    assert candidate.git_status is not None
    assert candidate.git_summary is not None
    assert candidate.git is not None
    assert candidate.git.artifact.sha256.startswith("sha256:")
    with pytest.raises(ValueError, match="trusted-Docker warning"):
        type(candidate).model_validate(
            {
                **candidate.model_dump(mode="python"),
                "warnings": (),
            }
        )
    publication = asyncio.run(repository.publish_candidate(candidate))
    decision = coding_product_completion_decision(
        request,
        candidate,
        evidence=WorkEvidenceReference(
            kind=CODING_PRODUCT_EVIDENCE_KIND,
            reference_id=publication.artifact.artifact_id,
            version="v1",
            digest=candidate.digest,
            available=True,
        ),
    )
    assert decision.verdict is CompletionVerdict.ACCEPTED
    assert asyncio.run(repository.read_candidate(publication.result_reference)) == candidate

    async def publish_noncanonical_alias() -> CompletionResultReference:
        stored = await repository.store.read_bytes(publication.artifact.artifact_id)
        alias_id = "art_11111111111111111111111111111111"
        await repository.store.put_bytes(
            stored.content,
            artifact_id=alias_id,
            filename="coding-product-result.json",
            content_type="application/json",
            scope=ArtifactScope.SESSION,
            session_id=candidate.session_id,
            metadata={
                "content_sha256": "sha256:" + hashlib.sha256(stored.content).hexdigest(),
            },
        )
        return CompletionResultReference(
            kind=publication.result_reference.kind,
            reference_id=alias_id,
            digest=publication.result_reference.digest,
        )

    alias = asyncio.run(publish_noncanonical_alias())
    with pytest.raises(ValueError, match="content-addressed"):
        asyncio.run(repository.read_candidate(alias))


def test_missing_check_duration_cannot_be_patch_ready(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    workspace = LocalWorkspace(source, workspace_id="source-workspace")
    observation = asyncio.run(
        observe_deterministic_workspace(
            workspace,
            observer="cayu-coding-product-source",
            limits=WorkspaceRevisionObservationLimits(),
        )
    )
    assert observation.revision is not None
    request = _request(baseline=observation.revision)
    checks = [_check_event(name) for name in request.settlement.required_checks]
    result = checks[0].payload["result"]
    assert type(result) is dict
    structured = result["structured"]
    assert type(structured) is dict
    structured.pop("duration_ms")

    candidate = asyncio.run(
        compile_coding_product_candidate(
            request,
            (
                *checks,
                *_terminal_events(
                    request=request,
                    workspace_revision=observation.revision,
                ),
            ),
            initial_observation=observation,
            final_observation=observation,
            repository=CodingProductArtifactRepository(LocalArtifactStore(tmp_path / "artifacts")),
        )
    )

    assert candidate.state is CodingProductState.CHECKS_FAILED
    assert candidate.checks[0].duration_ms is None


@pytest.mark.parametrize("later_ordinal", [3, 4])
def test_lifecycle_reconstruction_rejects_any_gap(tmp_path, later_ordinal: int) -> None:
    request = _request(baseline=_digest("baseline"))
    repository = CodingProductArtifactRepository(LocalArtifactStore(tmp_path / "artifacts"))
    first = CodingLifecycleReceipt(
        product_run_id=request.product_run_id,
        session_id=request.session_id,
        request_fingerprint=request.fingerprint,
        ordinal=1,
        state=CodingProductState.ADMITTED,
    )
    later = CodingLifecycleReceipt(
        product_run_id=request.product_run_id,
        session_id=request.session_id,
        request_fingerprint=request.fingerprint,
        ordinal=later_ordinal,
        prior_state=CodingProductState.PREPARING_WORKSPACE,
        state=CodingProductState.ACTIVE,
    )
    asyncio.run(repository.append_lifecycle(first))
    asyncio.run(repository.append_lifecycle(later))

    with pytest.raises(ValueError, match="reconstruction gap"):
        asyncio.run(
            repository.load_lifecycle(
                request.product_run_id,
                session_id=request.session_id,
                request_fingerprint=request.fingerprint,
            )
        )


def test_runner_does_not_replay_with_a_hidden_later_lifecycle_receipt(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    workspace = LocalWorkspace(source, workspace_id="source-workspace")
    observation = asyncio.run(
        observe_deterministic_workspace(
            workspace,
            observer="cayu-coding-product-source",
            limits=WorkspaceRevisionObservationLimits(),
        )
    )
    assert observation.revision is not None
    messages = [Message.text("user", "repair the project")]
    request = _request(baseline=observation.revision)
    request = request.model_copy(
        update={
            "task": CodingTaskAuthority(
                task_id=request.task.task_id,
                instruction_sha256="sha256:" + session_input_messages_sha256(messages),
            )
        }
    )
    repository = CodingProductArtifactRepository(LocalArtifactStore(tmp_path / "artifacts"))
    first = CodingLifecycleReceipt(
        product_run_id=request.product_run_id,
        session_id=request.session_id,
        request_fingerprint=request.fingerprint,
        ordinal=1,
        state=CodingProductState.ADMITTED,
    )
    fourth = CodingLifecycleReceipt(
        product_run_id=request.product_run_id,
        session_id=request.session_id,
        request_fingerprint=request.fingerprint,
        ordinal=4,
        prior_state=CodingProductState.ACTIVE,
        state=CodingProductState.FAILED,
        reason_code="prior_execution_failed",
    )
    asyncio.run(repository.append_lifecycle(first))
    asyncio.run(repository.append_lifecycle(fourth))
    app = CayuApp()
    calls = 0

    async def fail_if_replayed(run_request):
        nonlocal calls
        del run_request
        calls += 1
        if False:
            yield

    monkeypatch.setattr(app, "run", fail_if_replayed)

    async def accept_git_authority(_expected: CodingGitBaselineAuthority) -> None:
        return None

    runner = CodingProductRunner(
        app,
        source_workspace=workspace,
        repository=repository,
        source_git_authority_validator=accept_git_authority,
    )

    with pytest.raises(ValueError, match="reconstruction gap"):
        asyncio.run(
            runner.run(
                request,
                RunRequest(
                    agent_name=request.agent_name,
                    session_id=request.session_id,
                    environment_name=request.runtime.environment_name,
                    messages=messages,
                ),
            )
        )
    assert calls == 0


def test_collect_coding_product_events_enforces_the_byte_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(coding_products, "CODING_PRODUCT_MAX_EVENT_BYTES", 1)

    async def events():
        yield _check_event("format")

    with pytest.raises(ValueError, match="event bytes"):
        asyncio.run(collect_coding_product_events(events()))


def test_missing_required_check_is_not_patch_ready(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "example.py").write_text("content\n", encoding="utf-8")
    workspace = LocalWorkspace(source, workspace_id="source-workspace")
    observation = asyncio.run(
        observe_deterministic_workspace(
            workspace,
            observer="cayu-coding-product-source",
            limits=WorkspaceRevisionObservationLimits(),
        )
    )
    assert observation.revision is not None
    request = _request(baseline=observation.revision)
    repository = CodingProductArtifactRepository(LocalArtifactStore(tmp_path / "artifacts"))

    candidate = asyncio.run(
        compile_coding_product_candidate(
            request,
            (
                *(_check_event(name) for name in ("format", "lint")),
                *_terminal_events(
                    request=request,
                    workspace_revision=observation.revision,
                ),
            ),
            initial_observation=observation,
            final_observation=observation,
            repository=repository,
        )
    )

    assert candidate.state is CodingProductState.CHECKS_NOT_RUN


def test_complete_admitted_nonzero_command_can_remain_patch_ready(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    workspace = LocalWorkspace(source, workspace_id="source-workspace")
    observation = asyncio.run(
        observe_deterministic_workspace(
            workspace,
            observer="cayu-coding-product-source",
            limits=WorkspaceRevisionObservationLimits(),
        )
    )
    assert observation.revision is not None
    request = _request(baseline=observation.revision)
    candidate = asyncio.run(
        compile_coding_product_candidate(
            request,
            (
                _command_event(status="nonzero", exit_code=1),
                *(_check_event(name) for name in request.settlement.required_checks),
                *_git_events(changed=False),
                *_terminal_events(
                    request=request,
                    workspace_revision=observation.revision,
                ),
            ),
            initial_observation=observation,
            final_observation=observation,
            repository=CodingProductArtifactRepository(LocalArtifactStore(tmp_path / "artifacts")),
        )
    )

    assert candidate.state is CodingProductState.PATCH_READY_FOR_DELIVERY
    assert candidate.commands[0].settled is True
    with pytest.raises(ValueError, match="unsettled command evidence"):
        type(candidate).model_validate(
            {
                **candidate.model_dump(mode="python"),
                "commands": (
                    candidate.commands[0].model_copy(update={"output_publication_complete": False}),
                ),
            }
        )


def test_git_evidence_before_the_last_workspace_effect_requires_reconstruction(
    tmp_path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    workspace = LocalWorkspace(source, workspace_id="source-workspace")
    observation = asyncio.run(
        observe_deterministic_workspace(
            workspace,
            observer="cayu-coding-product-source",
            limits=WorkspaceRevisionObservationLimits(),
        )
    )
    assert observation.revision is not None
    request = _request(baseline=observation.revision)
    late_mutation = _tool_event(
        "apply_patch",
        {
            "outcome": "applied",
            "patch_id": "late-patch",
            "requires_fresh_read": False,
        },
        event_id="late-mutation",
    )

    finalization, terminal = _terminal_events(
        request=request,
        workspace_revision=observation.revision,
    )
    candidate = asyncio.run(
        compile_coding_product_candidate(
            request,
            (
                *(_check_event(name) for name in request.settlement.required_checks),
                *_git_events(changed=False),
                finalization,
                late_mutation,
                terminal,
            ),
            initial_observation=observation,
            final_observation=observation,
            repository=CodingProductArtifactRepository(LocalArtifactStore(tmp_path / "artifacts")),
        )
    )

    assert candidate.state is CodingProductState.RECONSTRUCTION_REQUIRED


@pytest.mark.parametrize(
    ("mode", "origin_field"),
    [
        ("status", "offset"),
        ("summary", "offset"),
        ("diff", "diff_offset"),
    ],
)
def test_non_initial_git_page_cannot_supply_complete_evidence(
    tmp_path,
    mode: str,
    origin_field: str,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    workspace = LocalWorkspace(source, workspace_id="source-workspace")
    observation = asyncio.run(
        observe_deterministic_workspace(
            workspace,
            observer="cayu-coding-product-source",
            limits=WorkspaceRevisionObservationLimits(),
        )
    )
    assert observation.revision is not None
    request = _request(baseline=observation.revision)
    events = [
        *(
            _check_event(name, workspace_revision=observation.revision)
            for name in request.settlement.required_checks
        ),
        *_git_events(changed=False),
        *_terminal_events(
            request=request,
            workspace_revision=observation.revision,
        ),
    ]
    finalization_index = next(
        index
        for index, event in enumerate(events)
        if event.type is EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED
    )
    event = events[finalization_index]
    receipt = event.payload["final_git_receipt"]
    assert type(receipt) is dict
    evidence = receipt[mode]
    assert type(evidence) is dict
    structured = evidence["structured"]
    assert type(structured) is dict
    material = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    material[mode] = {
        **evidence,
        "structured": {**structured, origin_field: 999},
    }
    events[finalization_index] = event.model_copy(
        update={
            "payload": {
                **event.payload,
                "final_git_receipt": _seal_final_git_receipt(material),
            }
        },
        deep=True,
    )

    candidate = asyncio.run(
        compile_coding_product_candidate(
            request,
            events,
            initial_observation=observation,
            final_observation=observation,
            repository=CodingProductArtifactRepository(LocalArtifactStore(tmp_path / "artifacts")),
        )
    )

    assert candidate.git_status is None
    assert candidate.git_summary is None
    assert candidate.git is None
    assert candidate.state is CodingProductState.RECONSTRUCTION_REQUIRED


def test_redacted_final_git_receipt_requires_reconstruction(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    workspace = LocalWorkspace(source, workspace_id="source-workspace")
    observation = asyncio.run(
        observe_deterministic_workspace(
            workspace,
            observer="cayu-coding-product-source",
            limits=WorkspaceRevisionObservationLimits(),
        )
    )
    assert observation.revision is not None
    request = _request(baseline=observation.revision)
    finalization, terminal = _terminal_events(
        request=request,
        workspace_revision=observation.revision,
    )
    receipt = finalization.payload["final_git_receipt"]
    assert type(receipt) is dict
    diff = receipt["diff"]
    assert type(diff) is dict
    finalization = finalization.model_copy(
        update={
            "payload": {
                **finalization.payload,
                "final_git_receipt": {
                    **receipt,
                    "diff": {**diff, "content": REDACTED_SECRET},
                },
            }
        },
        deep=True,
    )

    candidate = asyncio.run(
        compile_coding_product_candidate(
            request,
            (
                *(
                    _check_event(name, workspace_revision=observation.revision)
                    for name in request.settlement.required_checks
                ),
                finalization,
                terminal,
            ),
            initial_observation=observation,
            final_observation=observation,
            repository=CodingProductArtifactRepository(LocalArtifactStore(tmp_path / "artifacts")),
        )
    )

    assert candidate.state is CodingProductState.RECONSTRUCTION_REQUIRED
    assert candidate.git is None


def test_required_checks_before_the_final_workspace_revision_require_reconstruction(
    tmp_path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "example.py").write_text("before\n", encoding="utf-8")
    workspace = LocalWorkspace(source, workspace_id="source-workspace")
    limits = WorkspaceRevisionObservationLimits()
    initial = asyncio.run(
        observe_deterministic_workspace(
            workspace,
            observer="cayu-coding-product-source",
            limits=limits,
        )
    )
    assert initial.revision is not None
    request = _request(baseline=initial.revision)
    (source / "example.py").write_text("after\n", encoding="utf-8")
    final = asyncio.run(
        observe_deterministic_workspace(
            workspace,
            observer="cayu-coding-product-source",
            limits=limits,
        )
    )
    assert final.revision is not None

    candidate = asyncio.run(
        compile_coding_product_candidate(
            request,
            (
                *(
                    _check_event(name, workspace_revision=initial.revision)
                    for name in request.settlement.required_checks
                ),
                _tool_event(
                    "apply_patch",
                    {
                        "outcome": "applied",
                        "patch_id": "post-check-patch",
                        "requires_fresh_read": False,
                    },
                    event_id="post-check-mutation",
                ),
                *_git_events(changed=True),
                *_terminal_events(
                    request=request,
                    workspace_revision=final.revision,
                    changed=True,
                ),
            ),
            initial_observation=initial,
            final_observation=final,
            repository=CodingProductArtifactRepository(LocalArtifactStore(tmp_path / "artifacts")),
        )
    )

    assert candidate.state is CodingProductState.RECONSTRUCTION_REQUIRED


def test_patch_ready_candidate_requires_checks_bound_to_final_revision(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    workspace = LocalWorkspace(source, workspace_id="source-workspace")
    observation = asyncio.run(
        observe_deterministic_workspace(
            workspace,
            observer="cayu-coding-product-source",
            limits=WorkspaceRevisionObservationLimits(),
        )
    )
    assert observation.revision is not None
    request = _request(baseline=observation.revision)
    repository = CodingProductArtifactRepository(LocalArtifactStore(tmp_path / "artifacts"))
    candidate = asyncio.run(
        compile_coding_product_candidate(
            request,
            (
                *(
                    _check_event(name, workspace_revision=observation.revision)
                    for name in request.settlement.required_checks
                ),
                *_git_events(changed=False),
                *_terminal_events(
                    request=request,
                    workspace_revision=observation.revision,
                ),
            ),
            initial_observation=observation,
            final_observation=observation,
            repository=repository,
        )
    )
    stale_checks = tuple(
        check.model_copy(update={"workspace_revision": _digest("stale")})
        for check in candidate.checks
    )
    stale_candidate = candidate.model_copy(update={"checks": stale_checks})
    stale_payload = {
        **candidate.model_dump(mode="python"),
        "checks": stale_checks,
    }

    with pytest.raises(ValueError, match="final source revision"):
        type(candidate).model_validate(stale_payload)
    with pytest.raises(ValueError, match="final source revision"):
        asyncio.run(repository.publish_candidate(stale_candidate))
    decision = coding_product_completion_decision(
        request,
        stale_candidate,
        evidence=WorkEvidenceReference(
            kind=CODING_PRODUCT_EVIDENCE_KIND,
            reference_id="stale-check-evidence",
            version="v1",
            digest=stale_candidate.digest,
            available=True,
        ),
    )
    assert decision.verdict is CompletionVerdict.REJECTED


def test_failed_mutation_without_structured_receipt_requires_reconstruction(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    workspace = LocalWorkspace(source, workspace_id="source-workspace")
    observation = asyncio.run(
        observe_deterministic_workspace(
            workspace,
            observer="cayu-coding-product-source",
            limits=WorkspaceRevisionObservationLimits(),
        )
    )
    assert observation.revision is not None
    request = _request(baseline=observation.revision)
    failed_mutation = Event(
        id="failed-mutation-without-receipt",
        type=EventType.TOOL_CALL_FAILED,
        session_id=request.session_id,
        interaction_id="interaction-1",
        agent_name=request.agent_name,
        environment_name=request.runtime.environment_name,
        tool_name="apply_patch",
        payload={
            "execution_profile_fingerprint": request.runtime.execution_profile_fingerprint,
        },
    )

    candidate = asyncio.run(
        compile_coding_product_candidate(
            request,
            (
                *(
                    _check_event(name, workspace_revision=observation.revision)
                    for name in request.settlement.required_checks
                ),
                failed_mutation,
                *_git_events(changed=False),
                *_terminal_events(
                    request=request,
                    workspace_revision=observation.revision,
                ),
            ),
            initial_observation=observation,
            final_observation=observation,
            repository=CodingProductArtifactRepository(LocalArtifactStore(tmp_path / "artifacts")),
        )
    )

    assert candidate.state is CodingProductState.RECONSTRUCTION_REQUIRED


def test_finalize_completion_without_publication_receipt_requires_reconstruction(
    tmp_path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    workspace = LocalWorkspace(source, workspace_id="source-workspace")
    observation = asyncio.run(
        observe_deterministic_workspace(
            workspace,
            observer="cayu-coding-product-source",
            limits=WorkspaceRevisionObservationLimits(),
        )
    )
    assert observation.revision is not None
    request = _request(baseline=observation.revision)
    finalization, terminal = _terminal_events(
        request=request,
        workspace_revision=observation.revision,
    )
    finalization = finalization.model_copy(
        update={
            "payload": {
                "execution_profile_fingerprint": _digest("execution").removeprefix("sha256:")
            }
        },
        deep=True,
    )

    candidate = asyncio.run(
        compile_coding_product_candidate(
            request,
            (
                *(
                    _check_event(name, workspace_revision=observation.revision)
                    for name in request.settlement.required_checks
                ),
                *_git_events(changed=False),
                finalization,
                terminal,
            ),
            initial_observation=observation,
            final_observation=observation,
            repository=CodingProductArtifactRepository(LocalArtifactStore(tmp_path / "artifacts")),
        )
    )

    assert candidate.state is CodingProductState.RECONSTRUCTION_REQUIRED
    assert candidate.publication.detail_code == "source_publication_receipt_invalid"


def test_publication_receipt_for_another_workspace_requires_reconstruction(
    tmp_path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    workspace = LocalWorkspace(source, workspace_id="source-workspace")
    observation = asyncio.run(
        observe_deterministic_workspace(
            workspace,
            observer="cayu-coding-product-source",
            limits=WorkspaceRevisionObservationLimits(),
        )
    )
    assert observation.revision is not None
    request = _request(baseline=observation.revision)
    finalization, terminal = _terminal_events(
        request=request,
        workspace_revision=observation.revision,
    )
    finalization = finalization.model_copy(
        update={
            "payload": {
                **finalization.payload,
                "source_publication_receipt": _publication_receipt(
                    destination_workspace_id="different-source-workspace"
                ),
            }
        },
        deep=True,
    )

    candidate = asyncio.run(
        compile_coding_product_candidate(
            request,
            (
                *(
                    _check_event(name, workspace_revision=observation.revision)
                    for name in request.settlement.required_checks
                ),
                *_git_events(changed=False),
                finalization,
                terminal,
            ),
            initial_observation=observation,
            final_observation=observation,
            repository=CodingProductArtifactRepository(LocalArtifactStore(tmp_path / "artifacts")),
        )
    )

    assert candidate.state is CodingProductState.RECONSTRUCTION_REQUIRED
    assert candidate.publication.detail_code == "source_publication_receipt_invalid"


def test_publication_receipt_accepts_only_the_store_verified_workspace_alias(
    tmp_path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    workspace = LocalWorkspace(source, workspace_id="source-workspace")
    observation = asyncio.run(
        observe_deterministic_workspace(
            workspace,
            observer="cayu-coding-product-source",
            limits=WorkspaceRevisionObservationLimits(),
        )
    )
    assert observation.revision is not None
    request = _request(baseline=observation.revision)
    alias_codec = InMemorySessionStore().public_authority_alias_codec
    destination_alias = alias_codec.encode(
        request.source.workspace_id,
        field_name="workspace_observation_workspace_id",
        session_id=request.session_id,
    )
    finalization, terminal = _terminal_events(
        request=request,
        workspace_revision=observation.revision,
    )
    final_git_receipt = finalization.payload["final_git_receipt"]
    assert type(final_git_receipt) is dict
    final_git_material = {
        key: value for key, value in final_git_receipt.items() if key != "receipt_sha256"
    }
    final_git_material["destination_workspace_id"] = destination_alias
    finalization = finalization.model_copy(
        update={
            "payload": {
                **finalization.payload,
                "source_publication_receipt": _publication_receipt(
                    destination_workspace_id=destination_alias
                ),
                "final_git_receipt": _seal_final_git_receipt(final_git_material),
            }
        },
        deep=True,
    )

    candidate = asyncio.run(
        compile_coding_product_candidate(
            request,
            (
                *(
                    _check_event(name, workspace_revision=observation.revision)
                    for name in request.settlement.required_checks
                ),
                *_git_events(changed=False),
                finalization,
                terminal,
            ),
            initial_observation=observation,
            final_observation=observation,
            repository=CodingProductArtifactRepository(LocalArtifactStore(tmp_path / "artifacts")),
            public_authority_alias_codec=alias_codec,
        )
    )
    evidence = WorkEvidenceReference(
        kind=CODING_PRODUCT_EVIDENCE_KIND,
        reference_id="coding-product-evidence",
        version="v1",
        digest=candidate.digest,
        available=True,
    )

    assert candidate.state is CodingProductState.PATCH_READY_FOR_DELIVERY
    assert (
        coding_product_completion_decision(
            request,
            candidate,
            evidence=evidence,
            public_authority_alias_codec=alias_codec,
        ).verdict
        is CompletionVerdict.ACCEPTED
    )
    assert (
        coding_product_completion_decision(
            request,
            candidate,
            evidence=evidence,
        ).verdict
        is CompletionVerdict.REJECTED
    )


def test_failed_check_event_cannot_supply_positive_evidence(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    workspace = LocalWorkspace(source, workspace_id="source-workspace")
    observation = asyncio.run(
        observe_deterministic_workspace(
            workspace,
            observer="cayu-coding-product-source",
            limits=WorkspaceRevisionObservationLimits(),
        )
    )
    assert observation.revision is not None
    request = _request(baseline=observation.revision)
    checks = [_check_event(name) for name in request.settlement.required_checks]
    checks[0] = checks[0].model_copy(update={"type": EventType.TOOL_CALL_FAILED})

    candidate = asyncio.run(
        compile_coding_product_candidate(
            request,
            (
                *checks,
                *_git_events(changed=False),
                *_terminal_events(
                    request=request,
                    workspace_revision=observation.revision,
                ),
            ),
            initial_observation=observation,
            final_observation=observation,
            repository=CodingProductArtifactRepository(LocalArtifactStore(tmp_path / "artifacts")),
        )
    )

    assert candidate.state is CodingProductState.CHECKS_FAILED


def test_evidence_from_another_environment_is_rejected(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    workspace = LocalWorkspace(source, workspace_id="source-workspace")
    observation = asyncio.run(
        observe_deterministic_workspace(
            workspace,
            observer="cayu-coding-product-source",
            limits=WorkspaceRevisionObservationLimits(),
        )
    )
    assert observation.revision is not None
    request = _request(baseline=observation.revision)
    events = (
        *(
            _check_event(name, workspace_revision=observation.revision)
            for name in request.settlement.required_checks
        ),
        *_git_events(changed=False),
        *_terminal_events(
            request=request,
            workspace_revision=observation.revision,
        ),
    )
    events = tuple(event.model_copy(update={"environment_name": "host"}) for event in events)

    with pytest.raises(CodingProductEvidenceError, match="environment authority"):
        asyncio.run(
            compile_coding_product_candidate(
                request,
                events,
                initial_observation=observation,
                final_observation=observation,
                repository=CodingProductArtifactRepository(
                    LocalArtifactStore(tmp_path / "artifacts")
                ),
            )
        )


def test_tool_evidence_without_an_agent_binding_is_rejected(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    workspace = LocalWorkspace(source, workspace_id="source-workspace")
    observation = asyncio.run(
        observe_deterministic_workspace(
            workspace,
            observer="cayu-coding-product-source",
            limits=WorkspaceRevisionObservationLimits(),
        )
    )
    assert observation.revision is not None
    request = _request(baseline=observation.revision)
    checks = [_check_event(name) for name in request.settlement.required_checks]
    checks[0] = checks[0].model_copy(update={"agent_name": None})

    with pytest.raises(CodingProductEvidenceError, match="agent authority"):
        asyncio.run(
            compile_coding_product_candidate(
                request,
                (
                    *checks,
                    *_git_events(changed=False),
                    *_terminal_events(
                        request=request,
                        workspace_revision=observation.revision,
                    ),
                ),
                initial_observation=observation,
                final_observation=observation,
                repository=CodingProductArtifactRepository(
                    LocalArtifactStore(tmp_path / "artifacts")
                ),
            )
        )


@pytest.mark.parametrize("authority_field", ["execution_profile_fingerprint", "interaction_id"])
def test_patch_ready_requires_authority_on_every_evidence_event(
    tmp_path,
    authority_field: str,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    workspace = LocalWorkspace(source, workspace_id="source-workspace")
    observation = asyncio.run(
        observe_deterministic_workspace(
            workspace,
            observer="cayu-coding-product-source",
            limits=WorkspaceRevisionObservationLimits(),
        )
    )
    assert observation.revision is not None
    request = _request(baseline=observation.revision)
    events = [
        *(
            _check_event(name, workspace_revision=observation.revision)
            for name in request.settlement.required_checks
        ),
        *_git_events(changed=False),
        *_terminal_events(
            request=request,
            workspace_revision=observation.revision,
        ),
    ]
    if authority_field == "interaction_id":
        events[0] = events[0].model_copy(update={"interaction_id": None})
    else:
        payload = dict(events[0].payload)
        payload.pop(authority_field)
        events[0] = events[0].model_copy(update={"payload": payload}, deep=True)

    candidate = asyncio.run(
        compile_coding_product_candidate(
            request,
            events,
            initial_observation=observation,
            final_observation=observation,
            repository=CodingProductArtifactRepository(LocalArtifactStore(tmp_path / "artifacts")),
        )
    )

    assert candidate.state is CodingProductState.RECONSTRUCTION_REQUIRED


def test_session_terminal_does_not_supply_tool_or_profile_authority(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    workspace = LocalWorkspace(source, workspace_id="source-workspace")
    observation = asyncio.run(
        observe_deterministic_workspace(
            workspace,
            observer="cayu-coding-product-source",
            limits=WorkspaceRevisionObservationLimits(),
        )
    )
    assert observation.revision is not None
    request = _request(baseline=observation.revision)
    finalization, terminal = _terminal_events(
        request=request,
        workspace_revision=observation.revision,
    )
    terminal = terminal.model_copy(update={"interaction_id": None, "payload": {}}, deep=True)

    candidate = asyncio.run(
        compile_coding_product_candidate(
            request,
            (
                *(
                    _check_event(name, workspace_revision=observation.revision)
                    for name in request.settlement.required_checks
                ),
                *_git_events(changed=False),
                finalization,
                terminal,
            ),
            initial_observation=observation,
            final_observation=observation,
            repository=CodingProductArtifactRepository(LocalArtifactStore(tmp_path / "artifacts")),
        )
    )

    assert candidate.interaction_id == "interaction-1"
    assert candidate.state is CodingProductState.PATCH_READY_FOR_DELIVERY


def test_session_artifact_validation_rejects_redaction_truncation() -> None:
    content = b"{}"
    result = ArtifactReadResult(
        metadata=ArtifactMetadata(
            id="art_request",
            filename="coding-product-request.json",
            content_type="application/json",
            size_bytes=len(content),
            scope=ArtifactScope.SESSION,
            session_id="session-1",
            metadata={"content_sha256": "sha256:" + hashlib.sha256(content).hexdigest()},
        ),
        content=content,
        total_bytes=len(content),
        redaction_truncated=True,
    )

    with pytest.raises(ValueError, match="artifact authority"):
        CodingProductArtifactRepository._validate_session_json_artifact(
            result,
            artifact_id="art_request",
            session_id="session-1",
            filename="coding-product-request.json",
        )


def test_projection_truncation_without_stored_artifact_fails_check(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    workspace = LocalWorkspace(source, workspace_id="source-workspace")
    observation = asyncio.run(
        observe_deterministic_workspace(
            workspace,
            observer="cayu-coding-product-source",
            limits=WorkspaceRevisionObservationLimits(),
        )
    )
    assert observation.revision is not None
    request = _request(baseline=observation.revision)
    repository = CodingProductArtifactRepository(LocalArtifactStore(tmp_path / "artifacts"))
    events = list(_check_event(name) for name in request.settlement.required_checks)
    events[0] = _tool_event(
        "run_check",
        {
            "check": "format",
            "check_profile_fingerprint": _digest("check:format"),
            "status": "passed",
            "exit_code": 0,
            "duration_ms": 1,
            "stdout_truncated": True,
            "stdout_projection_truncated": True,
            "output_artifact_status": "not_requested",
            "workspace_mutation_settlement": "runner_quiescent",
        },
        event_id="check-format",
    )

    candidate = asyncio.run(
        compile_coding_product_candidate(
            request,
            (
                *events,
                *_terminal_events(
                    request=request,
                    workspace_revision=observation.revision,
                ),
            ),
            initial_observation=observation,
            final_observation=observation,
            repository=repository,
        )
    )

    assert candidate.state is CodingProductState.CHECKS_FAILED


def test_runner_recovers_published_result_without_replaying_session(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "example.py").write_text("content\n", encoding="utf-8")
    workspace = LocalWorkspace(source, workspace_id="source-workspace")
    observation = asyncio.run(
        observe_deterministic_workspace(
            workspace,
            observer="cayu-coding-product-source",
            limits=WorkspaceRevisionObservationLimits(),
        )
    )
    assert observation.revision is not None
    observation_revision = observation.revision
    request = _request(baseline=observation.revision)
    messages = [Message.text("user", "repair the project")]
    request = request.model_copy(
        update={
            "task": CodingTaskAuthority(
                task_id=request.task.task_id,
                instruction_sha256="sha256:" + session_input_messages_sha256(messages),
            )
        }
    )
    repository = CodingProductArtifactRepository(
        LocalArtifactStore(tmp_path / "artifacts", store_id="runner-recovery")
    )
    app = CayuApp()
    calls = 0

    async def run_once(run_request):
        nonlocal calls
        del run_request
        calls += 1
        events = (
            *(
                _check_event(name, workspace_revision=observation_revision)
                for name in request.settlement.required_checks
            ),
            *_git_events(changed=False),
            *_terminal_events(
                request=request,
                workspace_revision=observation_revision,
            ),
        )
        for event in events:
            yield event

    monkeypatch.setattr(app, "run", run_once)
    run_request = RunRequest(
        agent_name=request.agent_name,
        session_id=request.session_id,
        environment_name=request.runtime.environment_name,
        messages=messages,
    )
    validated_git_authorities: list[CodingGitBaselineAuthority] = []

    async def validate_git_authority(expected: CodingGitBaselineAuthority) -> None:
        validated_git_authorities.append(expected)

    runner = CodingProductRunner(
        app,
        source_workspace=workspace,
        repository=repository,
        source_git_authority_validator=validate_git_authority,
    )

    first = asyncio.run(runner.run(request, run_request))
    recovered = asyncio.run(runner.run(request, run_request))

    assert first.candidate.state is CodingProductState.PATCH_READY_FOR_DELIVERY
    assert recovered == first
    assert calls == 1
    assert validated_git_authorities == [request.source.git_baseline] * 3


def test_runner_recovers_published_non_success_result_without_replaying_session(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "example.py").write_text("content\n", encoding="utf-8")
    workspace = LocalWorkspace(source, workspace_id="source-workspace")
    observation = asyncio.run(
        observe_deterministic_workspace(
            workspace,
            observer="cayu-coding-product-source",
            limits=WorkspaceRevisionObservationLimits(),
        )
    )
    assert observation.revision is not None
    messages = [Message.text("user", "repair the project")]
    request = _request(baseline=observation.revision).model_copy(
        update={
            "task": CodingTaskAuthority(
                task_id="task-1",
                instruction_sha256="sha256:" + session_input_messages_sha256(messages),
            )
        }
    )
    repository = CodingProductArtifactRepository(
        LocalArtifactStore(tmp_path / "artifacts", store_id="non-success-recovery")
    )
    app = CayuApp()
    calls = 0

    async def run_once(run_request):
        nonlocal calls
        del run_request
        calls += 1
        for event in (
            *_git_events(changed=False),
            *_terminal_events(
                request=request,
                workspace_revision=observation.revision,
            ),
        ):
            yield event

    monkeypatch.setattr(app, "run", run_once)

    async def accept_git_authority(_expected: CodingGitBaselineAuthority) -> None:
        return None

    runner = CodingProductRunner(
        app,
        source_workspace=workspace,
        repository=repository,
        source_git_authority_validator=accept_git_authority,
    )
    run_request = RunRequest(
        agent_name=request.agent_name,
        session_id=request.session_id,
        environment_name=request.runtime.environment_name,
        messages=messages,
    )
    original_append_lifecycle = repository.append_lifecycle
    stopped = False

    async def stop_before_final_lifecycle(receipt):
        nonlocal stopped
        if receipt.state is CodingProductState.CHECKS_NOT_RUN and not stopped:
            stopped = True
            raise RuntimeError("process stopped before final lifecycle append")
        return await original_append_lifecycle(receipt)

    monkeypatch.setattr(repository, "append_lifecycle", stop_before_final_lifecycle)
    with pytest.raises(RuntimeError, match="process stopped before final lifecycle append"):
        asyncio.run(runner.run(request, run_request))

    receipts, _ = asyncio.run(
        repository.load_lifecycle(
            request.product_run_id,
            session_id=request.session_id,
            request_fingerprint=request.fingerprint,
        )
    )
    assert receipts[-1].state is CodingProductState.PUBLISHING
    assert receipts[-1].evidence_sha256 is not None
    durable = asyncio.run(
        repository.load_publication(
            request_fingerprint=request.fingerprint,
            digest=receipts[-1].evidence_sha256,
        )
    )
    assert durable.candidate.state is CodingProductState.CHECKS_NOT_RUN

    monkeypatch.setattr(repository, "append_lifecycle", original_append_lifecycle)
    recovered = asyncio.run(runner.run(request, run_request))

    assert recovered == durable
    assert calls == 1
    recovered_receipts, _ = asyncio.run(
        repository.load_lifecycle(
            request.product_run_id,
            session_id=request.session_id,
            request_fingerprint=request.fingerprint,
        )
    )
    assert recovered_receipts[-1].state is CodingProductState.CHECKS_NOT_RUN


def test_runner_rejects_git_authority_drift_before_final_source_settlement(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "example.py").write_text("content\n", encoding="utf-8")
    workspace = LocalWorkspace(source, workspace_id="source-workspace")
    observation = asyncio.run(
        observe_deterministic_workspace(
            workspace,
            observer="cayu-coding-product-source",
            limits=WorkspaceRevisionObservationLimits(),
        )
    )
    assert observation.revision is not None
    messages = [Message.text("user", "repair the project")]
    request = _request(baseline=observation.revision).model_copy(
        update={
            "task": CodingTaskAuthority(
                task_id="task-1",
                instruction_sha256="sha256:" + session_input_messages_sha256(messages),
            )
        }
    )
    repository = CodingProductArtifactRepository(
        LocalArtifactStore(tmp_path / "artifacts", store_id="git-drift-product-run")
    )
    app = CayuApp()

    async def run_once(run_request):
        del run_request
        for event in (
            *(
                _check_event(name, workspace_revision=observation.revision)
                for name in request.settlement.required_checks
            ),
            *_git_events(changed=False),
            *_terminal_events(
                request=request,
                workspace_revision=observation.revision,
            ),
        ):
            yield event

    monkeypatch.setattr(app, "run", run_once)
    validation_count = 0

    async def validate_git_authority(expected: CodingGitBaselineAuthority) -> None:
        nonlocal validation_count
        assert expected == request.source.git_baseline
        validation_count += 1
        if validation_count == 2:
            raise RuntimeError("host HEAD changed")

    runner = CodingProductRunner(
        app,
        source_workspace=workspace,
        repository=repository,
        source_git_authority_validator=validate_git_authority,
    )

    with pytest.raises(CodingProductAdmissionError, match="Git authority changed"):
        asyncio.run(
            runner.run(
                request,
                RunRequest(
                    agent_name=request.agent_name,
                    session_id=request.session_id,
                    environment_name=request.runtime.environment_name,
                    messages=messages,
                ),
            )
        )

    receipts, _artifact_ids = asyncio.run(
        repository.load_lifecycle(
            request.product_run_id,
            session_id=request.session_id,
            request_fingerprint=request.fingerprint,
        )
    )
    assert validation_count == 2
    assert receipts[-1].state is CodingProductState.SOURCE_CONFLICT
    assert receipts[-1].reason_code == "source_git_authority_mismatch"


def test_runner_rejects_source_revision_drift_before_patch_ready_publication(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    source_file = source / "example.py"
    source_file.write_text("content\n", encoding="utf-8")
    workspace = LocalWorkspace(source, workspace_id="source-workspace")
    observation = asyncio.run(
        observe_deterministic_workspace(
            workspace,
            observer="cayu-coding-product-source",
            limits=WorkspaceRevisionObservationLimits(),
        )
    )
    assert observation.revision is not None
    messages = [Message.text("user", "repair the project")]
    request = _request(baseline=observation.revision).model_copy(
        update={
            "task": CodingTaskAuthority(
                task_id="task-1",
                instruction_sha256="sha256:" + session_input_messages_sha256(messages),
            )
        }
    )
    repository = CodingProductArtifactRepository(
        LocalArtifactStore(tmp_path / "artifacts", store_id="source-drift-product-run")
    )
    app = CayuApp()

    async def run_once(run_request):
        del run_request
        for event in (
            *(
                _check_event(name, workspace_revision=observation.revision)
                for name in request.settlement.required_checks
            ),
            *_git_events(changed=False),
            *_terminal_events(
                request=request,
                workspace_revision=observation.revision,
            ),
        ):
            yield event

    monkeypatch.setattr(app, "run", run_once)
    validation_count = 0

    async def validate_git_authority(expected: CodingGitBaselineAuthority) -> None:
        nonlocal validation_count
        assert expected == request.source.git_baseline
        validation_count += 1
        if validation_count == 3:
            source_file.write_text("concurrent external edit\n", encoding="utf-8")

    runner = CodingProductRunner(
        app,
        source_workspace=workspace,
        repository=repository,
        source_git_authority_validator=validate_git_authority,
    )

    with pytest.raises(
        CodingProductAdmissionError,
        match="source changed after final evidence compilation",
    ):
        asyncio.run(
            runner.run(
                request,
                RunRequest(
                    agent_name=request.agent_name,
                    session_id=request.session_id,
                    environment_name=request.runtime.environment_name,
                    messages=messages,
                ),
            )
        )

    receipts, _artifact_ids = asyncio.run(
        repository.load_lifecycle(
            request.product_run_id,
            session_id=request.session_id,
            request_fingerprint=request.fingerprint,
        )
    )
    assert validation_count == 3
    assert receipts[-1].state is CodingProductState.SOURCE_CONFLICT
    assert receipts[-1].reason_code == "source_changed_before_publication"


def test_runner_preserves_cancellation_during_async_git_authority_validation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = LocalWorkspace(tmp_path, workspace_id="cancelled-git-validator")
    repository = CodingProductArtifactRepository(LocalArtifactStore(tmp_path / "artifacts"))

    async def cancel_git_authority(_expected: CodingGitBaselineAuthority) -> None:
        raise asyncio.CancelledError

    runner = CodingProductRunner(
        CayuApp(),
        source_workspace=workspace,
        repository=repository,
        source_git_authority_validator=cancel_git_authority,
    )
    recorded: list[tuple[CodingProductState, str | None]] = []

    async def record_state(
        _request,
        _receipts,
        _artifact_ids,
        state: CodingProductState,
        *,
        evidence_sha256: str | None = None,
        reason_code: str | None = None,
    ) -> None:
        del evidence_sha256
        recorded.append((state, reason_code))

    monkeypatch.setattr(runner, "_append_state", record_state)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            runner._require_source_git_authority(
                _request(baseline=_digest("cancelled-validator-baseline")),
                [],
                [],
            )
        )

    assert recorded == [(CodingProductState.CANCELLED, "caller_cancelled")]


def test_concurrent_identical_product_runs_dispatch_exactly_once(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "example.py").write_text("content\n", encoding="utf-8")
    workspace = LocalWorkspace(source, workspace_id="source-workspace")
    observation = asyncio.run(
        observe_deterministic_workspace(
            workspace,
            observer="cayu-coding-product-source",
            limits=WorkspaceRevisionObservationLimits(),
        )
    )
    assert observation.revision is not None
    observation_revision = observation.revision
    messages = [Message.text("user", "repair the project")]
    request = _request(baseline=observation.revision).model_copy(
        update={
            "task": CodingTaskAuthority(
                task_id="task-1",
                instruction_sha256="sha256:" + session_input_messages_sha256(messages),
            )
        }
    )
    repository = CodingProductArtifactRepository(
        LocalArtifactStore(tmp_path / "artifacts", store_id="concurrent-product-run")
    )
    original_load_lifecycle = repository.load_lifecycle
    both_loaded = asyncio.Event()
    load_count = 0

    async def synchronized_load_lifecycle(*args, **kwargs):
        nonlocal load_count
        loaded = await original_load_lifecycle(*args, **kwargs)
        load_count += 1
        if load_count == 2:
            both_loaded.set()
        await both_loaded.wait()
        return loaded

    monkeypatch.setattr(repository, "load_lifecycle", synchronized_load_lifecycle)
    app = CayuApp()
    dispatch_count = 0

    async def run_once(run_request):
        nonlocal dispatch_count
        assert run_request.metadata != {}
        dispatch_count += 1
        for event in (
            *(
                _check_event(name, workspace_revision=observation_revision)
                for name in request.settlement.required_checks
            ),
            *_terminal_events(
                request=request,
                workspace_revision=observation_revision,
            ),
        ):
            yield event

    monkeypatch.setattr(app, "run", run_once)

    async def accept_git_authority(_expected: CodingGitBaselineAuthority) -> None:
        return None

    runner = CodingProductRunner(
        app,
        source_workspace=workspace,
        repository=repository,
        source_git_authority_validator=accept_git_authority,
    )
    run_request = RunRequest(
        agent_name=request.agent_name,
        session_id=request.session_id,
        environment_name=request.runtime.environment_name,
        messages=messages,
    )

    async def run_concurrently():
        return await asyncio.gather(
            runner.run(request, run_request),
            runner.run(request, run_request),
            return_exceptions=True,
        )

    results = asyncio.run(run_concurrently())

    publications = [result for result in results if not isinstance(result, BaseException)]
    failures = [result for result in results if isinstance(result, BaseException)]
    assert len(publications) == 1
    assert publications[0].candidate.state is CodingProductState.PATCH_READY_FOR_DELIVERY
    assert len(failures) == 1
    assert isinstance(failures[0], CodingProductReconstructionRequiredError)
    assert "durably claimed" in str(failures[0])
    assert dispatch_count == 1
    receipts, _ = asyncio.run(
        original_load_lifecycle(
            request.product_run_id,
            session_id=request.session_id,
            request_fingerprint=request.fingerprint,
        )
    )
    assert receipts[-1].state is CodingProductState.PATCH_READY_FOR_DELIVERY
    assert all(
        receipt.state is not CodingProductState.RECONSTRUCTION_REQUIRED for receipt in receipts
    )


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("stale_revision", CodingProductState.SOURCE_CONFLICT),
        ("partial_patch", CodingProductState.PARTIAL),
        ("policy_denied", CodingProductState.DENIED),
        ("check_timeout", CodingProductState.CHECKS_FAILED),
        ("cancelled", CodingProductState.CANCELLED),
        ("stale_toolchain", CodingProductState.TOOLCHAIN_REBUILD_REQUIRED),
        ("failed_check", CodingProductState.CHECKS_FAILED),
        ("destination_conflict", CodingProductState.SOURCE_CONFLICT),
        ("reconstruction", CodingProductState.RECONSTRUCTION_REQUIRED),
        ("profile_drift", CodingProductState.RECONSTRUCTION_REQUIRED),
        ("cleanup_uncertain", CodingProductState.CHECKS_FAILED),
        ("partial_command", CodingProductState.PARTIAL),
        ("partial_git_scope", CodingProductState.RECONSTRUCTION_REQUIRED),
    ],
)
def test_product_failure_evidence_never_becomes_patch_ready(
    tmp_path,
    case: str,
    expected: CodingProductState,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    workspace = LocalWorkspace(source, workspace_id="source-workspace")
    observation = asyncio.run(
        observe_deterministic_workspace(
            workspace,
            observer="cayu-coding-product-source",
            limits=WorkspaceRevisionObservationLimits(),
        )
    )
    assert observation.revision is not None
    request = _request(baseline=observation.revision)
    checks = [_check_event(name) for name in request.settlement.required_checks]
    events: list[Event] = [
        *checks,
        *_git_events(changed=False),
        *_terminal_events(
            request=request,
            workspace_revision=observation.revision,
        ),
    ]
    if case in {"stale_revision", "partial_patch"}:
        events.insert(
            0,
            _tool_event(
                "apply_patch",
                {
                    "outcome": ("stale_revision" if case == "stale_revision" else "partial"),
                    "requires_fresh_read": True,
                },
                event_id=f"mutation-{case}",
            ),
        )
    elif case == "policy_denied":
        events.insert(
            0,
            Event(
                id="policy-denied",
                type=EventType.TOOL_CALL_APPROVAL_DENIED,
                session_id="session-1",
                interaction_id="interaction-1",
                agent_name="coder",
                environment_name="coding",
                tool_name="run_command",
                payload={},
            ),
        )
    elif case in {"check_timeout", "failed_check", "cleanup_uncertain"}:
        status = (
            "timed_out"
            if case == "check_timeout"
            else "failed"
            if case == "failed_check"
            else "partial"
        )
        events[2] = _tool_event(
            "run_check",
            {
                "check": "test",
                "check_profile_fingerprint": _digest("check:test"),
                "status": status,
                "exit_code": 1,
                "timed_out": case == "check_timeout",
                "cancelled": False,
                "workspace_mutation_settlement": (
                    "uncertain" if case == "cleanup_uncertain" else "complete"
                ),
            },
            event_id="check-test",
        )
    elif case == "cancelled":
        events = [
            *checks,
            *_git_events(changed=False),
            Event(
                id="cancelled-terminal",
                type=EventType.SESSION_INTERRUPTED,
                session_id="session-1",
                interaction_id="interaction-1",
                agent_name="coder",
                environment_name="coding",
                payload={},
            ),
        ]
    elif case == "stale_toolchain":
        events.insert(
            0,
            _tool_event(
                "run_command",
                {
                    "selector": "focused-test",
                    "selector_fingerprint": _digest("selector"),
                    "status": "stale_toolchain",
                    "error": "dependency_inputs_changed",
                },
                event_id="stale-toolchain",
            ),
        )
    elif case == "partial_command":
        events.insert(
            0,
            _command_event(
                status="partial",
                output_collection_complete=False,
                output_publication_complete=False,
            ),
        )
    elif case == "destination_conflict":
        events = [
            *checks,
            *_git_events(changed=False),
            Event(
                id="destination-conflict",
                type=EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED,
                session_id="session-1",
                interaction_id="interaction-1",
                agent_name="coder",
                environment_name="coding",
                payload={"error_type": "SyncBindingSourceConflictError"},
            ),
            Event(
                id="failed-terminal",
                type=EventType.SESSION_FAILED,
                session_id="session-1",
                interaction_id="interaction-1",
                agent_name="coder",
                environment_name="coding",
                payload={},
            ),
        ]
    elif case == "reconstruction":
        events = [*checks, *_git_events(changed=False)]
    elif case == "profile_drift":
        event = events[0]
        events[0] = event.model_copy(
            update={
                "payload": {
                    **event.payload,
                    "execution_profile_fingerprint": _digest("drift"),
                }
            },
            deep=True,
        )
    elif case == "partial_git_scope":
        finalization_index = next(
            index
            for index, event in enumerate(events)
            if event.type is EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED
        )
        event = events[finalization_index]
        receipt = event.payload["final_git_receipt"]
        assert type(receipt) is dict
        material = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        for mode in ("status", "summary", "diff"):
            evidence = material[mode]
            assert type(evidence) is dict
            structured = evidence["structured"]
            assert type(structured) is dict
            material[mode] = {
                **evidence,
                "structured": {**structured, "scope": "staged"},
            }
        events[finalization_index] = event.model_copy(
            update={
                "payload": {
                    **event.payload,
                    "final_git_receipt": _seal_final_git_receipt(material),
                }
            },
            deep=True,
        )

    candidate = asyncio.run(
        compile_coding_product_candidate(
            request,
            events,
            initial_observation=observation,
            final_observation=observation,
            repository=CodingProductArtifactRepository(LocalArtifactStore(tmp_path / "artifacts")),
        )
    )

    assert candidate.state is expected


def test_projection_truncation_with_stored_output_artifact_can_settle(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    workspace = LocalWorkspace(source, workspace_id="source-workspace")
    observation = asyncio.run(
        observe_deterministic_workspace(
            workspace,
            observer="cayu-coding-product-source",
            limits=WorkspaceRevisionObservationLimits(),
        )
    )
    assert observation.revision is not None
    request = _request(baseline=observation.revision)
    artifact_store = LocalArtifactStore(tmp_path / "artifacts")
    output_content = b'{"check":"format","stdout":"complete"}'
    output_sha256 = "sha256:" + hashlib.sha256(output_content).hexdigest()
    output_artifact = asyncio.run(
        artifact_store.put_bytes(
            output_content,
            artifact_id="art_11111111111111111111111111111111",
            filename="check-format-output.json",
            content_type="application/json",
            scope=ArtifactScope.SESSION,
            session_id=request.session_id,
            agent_name=request.agent_name,
            environment_name=request.runtime.environment_name,
            metadata={
                "operation": "run_check",
                "check": "format",
                "check_profile_fingerprint": _digest("check:format"),
                "content_sha256": output_sha256,
            },
        )
    )
    checks = [_check_event(name) for name in request.settlement.required_checks]
    checks[0] = _tool_event(
        "run_check",
        {
            "check": "format",
            "check_profile_fingerprint": _digest("check:format"),
            "status": "passed",
            "exit_code": 0,
            "duration_ms": 1,
            "stdout_truncated": True,
            "stdout_projection_truncated": True,
            "output_artifact_status": "stored",
            "output_sha256": output_sha256,
            "artifacts": [output_artifact.model_dump(mode="json")],
            "workspace_mutation_settlement": "complete",
            "workspace_revision": observation.revision,
        },
        event_id="check-format",
    )

    candidate = asyncio.run(
        compile_coding_product_candidate(
            request,
            (
                *checks,
                *_git_events(changed=False),
                *_terminal_events(
                    request=request,
                    workspace_revision=observation.revision,
                ),
            ),
            initial_observation=observation,
            final_observation=observation,
            repository=CodingProductArtifactRepository(artifact_store),
        )
    )

    assert candidate.state is CodingProductState.PATCH_READY_FOR_DELIVERY


def test_projection_truncation_with_missing_output_artifact_fails_closed(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    workspace = LocalWorkspace(source, workspace_id="source-workspace")
    observation = asyncio.run(
        observe_deterministic_workspace(
            workspace,
            observer="cayu-coding-product-source",
            limits=WorkspaceRevisionObservationLimits(),
        )
    )
    assert observation.revision is not None
    request = _request(baseline=observation.revision)
    checks = [_check_event(name) for name in request.settlement.required_checks]
    checks[0] = _tool_event(
        "run_check",
        {
            "check": "format",
            "check_profile_fingerprint": _digest("check:format"),
            "status": "passed",
            "exit_code": 0,
            "stdout_truncated": True,
            "stdout_projection_truncated": True,
            "output_artifact_status": "stored",
            "output_artifact_id": "art_missing",
            "workspace_mutation_settlement": "complete",
            "workspace_revision": observation.revision,
        },
        event_id="check-format",
    )

    candidate = asyncio.run(
        compile_coding_product_candidate(
            request,
            (
                *checks,
                *_git_events(changed=False),
                *_terminal_events(
                    request=request,
                    workspace_revision=observation.revision,
                ),
            ),
            initial_observation=observation,
            final_observation=observation,
            repository=CodingProductArtifactRepository(LocalArtifactStore(tmp_path / "artifacts")),
        )
    )

    assert candidate.state is CodingProductState.CHECKS_FAILED


def test_projection_truncation_with_fabricated_output_reference_is_rejected(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    workspace = LocalWorkspace(source, workspace_id="source-workspace")
    observation = asyncio.run(
        observe_deterministic_workspace(
            workspace,
            observer="cayu-coding-product-source",
            limits=WorkspaceRevisionObservationLimits(),
        )
    )
    assert observation.revision is not None
    request = _request(baseline=observation.revision)
    output_content = b'{"check":"format","stdout":"fabricated"}'
    output_sha256 = "sha256:" + hashlib.sha256(output_content).hexdigest()
    fabricated = ArtifactMetadata(
        id="art_22222222222222222222222222222222",
        filename="check-format-output.json",
        content_type="application/json",
        size_bytes=len(output_content),
        scope=ArtifactScope.SESSION,
        session_id=request.session_id,
        agent_name=request.agent_name,
        environment_name=request.runtime.environment_name,
        metadata={
            "operation": "run_check",
            "check": "format",
            "check_profile_fingerprint": _digest("check:format"),
            "content_sha256": output_sha256,
        },
    )
    checks = [_check_event(name) for name in request.settlement.required_checks]
    checks[0] = _tool_event(
        "run_check",
        {
            "check": "format",
            "check_profile_fingerprint": _digest("check:format"),
            "status": "passed",
            "exit_code": 0,
            "stdout_truncated": True,
            "stdout_projection_truncated": True,
            "output_artifact_status": "stored",
            "output_sha256": output_sha256,
            "artifacts": [fabricated.model_dump(mode="json")],
            "workspace_mutation_settlement": "complete",
            "workspace_revision": observation.revision,
        },
        event_id="check-format",
    )

    with pytest.raises(FileNotFoundError):
        asyncio.run(
            compile_coding_product_candidate(
                request,
                (
                    *checks,
                    *_git_events(changed=False),
                    *_terminal_events(
                        request=request,
                        workspace_revision=observation.revision,
                    ),
                ),
                initial_observation=observation,
                final_observation=observation,
                repository=CodingProductArtifactRepository(
                    LocalArtifactStore(tmp_path / "artifacts")
                ),
            )
        )


def test_positive_review_settlement_requires_linked_result_event(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    workspace = LocalWorkspace(source, workspace_id="source-workspace")
    observation = asyncio.run(
        observe_deterministic_workspace(
            workspace,
            observer="cayu-coding-product-source",
            limits=WorkspaceRevisionObservationLimits(),
        )
    )
    assert observation.revision is not None
    request = _request(baseline=observation.revision, reviewer_required=True)
    repository = CodingProductArtifactRepository(LocalArtifactStore(tmp_path / "artifacts"))

    with pytest.raises(CodingProductEvidenceError, match="reviewer settlement"):
        asyncio.run(
            compile_coding_product_candidate(
                request,
                _terminal_events(
                    request=request,
                    workspace_revision=observation.revision,
                ),
                initial_observation=observation,
                final_observation=observation,
                repository=repository,
                review_settlement=CodingReviewSettlement(
                    reviewer="passed",
                    human="not_required",
                    reviewer_event_id="missing-review-event",
                ),
            )
        )
