from __future__ import annotations

import asyncio
import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from cayu._validation import canonical_durable_json_bytes
from cayu.artifacts import ArtifactReadResult, LocalArtifactStore
from cayu.coding_products import (
    CodingGitBaselineAuthority,
    CodingProductArtifactRepository,
    CodingProductRequest,
    CodingProductState,
    CodingRuntimeAuthority,
    CodingSettlementPolicy,
    CodingSourceAuthority,
    CodingTaskAuthority,
    compile_coding_product_candidate,
)
from cayu.core.events import Event, EventType
from cayu.remote_git_delivery import (
    RemoteGitBrokerProfile,
    RemoteGitCommitAuthority,
    RemoteGitDeliveryAdmissionError,
    RemoteGitDeliveryApproval,
    RemoteGitDeliveryBroker,
    RemoteGitDeliveryError,
    RemoteGitDeliveryReconstructionRequiredError,
    RemoteGitDeliveryRepository,
    RemoteGitDeliveryRequest,
    RemoteGitDeliveryState,
    RemoteGitHttpCredentials,
    RemoteGitLifecycleReceipt,
    RemoteGitPreparedIntent,
    RemoteGitRemoteConfig,
    RemoteGitRepositoryAuthority,
    RemoteGitSecurityAuthority,
    RemoteGitSourceAuthority,
    approve_remote_git_delivery,
    remote_git_broker_behavior_fingerprint,
    remote_git_delivery_request,
)
from cayu.vaults import REDACTED_SECRET, SecretRef, StaticVault
from cayu.workspaces import LocalWorkspace
from cayu.workspaces.revisions import (
    WorkspaceRevisionObservationLimits,
    observe_deterministic_workspace,
)


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _git(cwd: Path, *arguments: str) -> str:
    completed = subprocess.run(
        [shutil.which("git") or "git", *arguments],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
            "HOME": str(cwd),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "LC_ALL": "C",
        },
    )
    return completed.stdout.strip()


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
        session_id="coding-session",
        interaction_id="coding-interaction",
        agent_name="coder",
        environment_name="coding",
        tool_name=tool_name,
        payload={
            "execution_profile_fingerprint": _digest("execution").removeprefix("sha256:"),
            "result": {"content": content, "structured": structured},
        },
    )


def _check_event(name: str, *, workspace_revision: str) -> Event:
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


def _git_events(diff_content: str | None = None) -> tuple[Event, Event, Event]:
    change = {
        "path": "app.py",
        "index": " ",
        "worktree": "M",
        "original_path": None,
    }
    return (
        _tool_event(
            "git_changes",
            {
                "mode": "status",
                "scope": "all",
                "changes": [change],
                "next_offset": None,
                "truncation_reasons": [],
            },
            event_id="git-status",
        ),
        _tool_event(
            "git_changes",
            {
                "mode": "summary",
                "scope": "all",
                "changes": [{**change, "additions": 1, "deletions": 1, "count_kind": "text"}],
                "next_offset": None,
                "truncation_reasons": [],
            },
            event_id="git-summary",
        ),
        _tool_event(
            "git_changes",
            {
                "mode": "diff",
                "scope": "all",
                "changes": [change],
                "next_offset": None,
                "next_diff_offset": None,
                "truncation_reasons": [],
                "binary_omitted": False,
            },
            event_id="git-diff",
            content=diff_content
            or (
                "diff --git a/app.py b/app.py\n"
                "--- a/app.py\n"
                "+++ b/app.py\n"
                "@@ -1 +1 @@\n"
                "-VALUE = 'before'\n"
                "+VALUE = 'after'\n"
            ),
        ),
    )


def _publication_receipt() -> dict[str, object]:
    snapshot_material: dict[str, object] = {
        "schema": "cayu.source_publication_snapshot.v1",
        "destination_workspace_id": "source-workspace",
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
    receipt_material: dict[str, object] = {
        "schema": "cayu.source_publication_receipt.v1",
        "snapshot_sha256": "sha256:"
        + hashlib.sha256(
            canonical_durable_json_bytes(
                snapshot_material,
                "source_publication_snapshot",
            )
        ).hexdigest(),
        "destination_workspace_id": "source-workspace",
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
        **receipt_material,
        "receipt_sha256": "sha256:"
        + hashlib.sha256(
            canonical_durable_json_bytes(
                receipt_material,
                "source_publication_receipt",
            )
        ).hexdigest(),
    }


def _terminal_events(
    request: CodingProductRequest, workspace_revision: str, diff_content: str | None = None
) -> tuple[Event, Event]:
    fingerprint = _digest("execution").removeprefix("sha256:")
    status, summary, diff = (event.payload["result"] for event in _git_events(diff_content))
    for result in (status, summary, diff):
        result["structured"].update(
            returned=len(result["structured"]["changes"]),
            offset=0,
            limit=200,
            truncated=False,
        )
    diff["structured"]["diff_offset"] = 0
    material = {
        "schema": "cayu.final_git_receipt.v1",
        "request_fingerprint": request.fingerprint,
        "destination_workspace_id": "source-workspace",
        "workload_workspace_id": "docker-workspace",
        "baseline_revision": request.source.baseline_revision,
        "workspace_revision": workspace_revision,
        "status": {"structured": status["structured"]},
        "summary": {"structured": summary["structured"]},
        "diff": {"content": diff["content"], "structured": diff["structured"]},
    }
    final_git_receipt = {
        **material,
        "receipt_sha256": "sha256:"
        + hashlib.sha256(canonical_durable_json_bytes(material, "final_git_receipt")).hexdigest(),
    }
    return (
        Event(
            id="publication",
            type=EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED,
            session_id="coding-session",
            interaction_id="coding-interaction",
            agent_name="coder",
            environment_name="coding",
            payload={
                "execution_profile_fingerprint": fingerprint,
                "source_publication_receipt": _publication_receipt(),
                "final_git_receipt": final_git_receipt,
                "final_snapshot": {
                    "snapshot_id": "final-source",
                    "metadata": {"copied_files": 1, "copied_bytes": 16, "deleted_files": 0},
                },
            },
        ),
        Event(
            id="completed",
            type=EventType.SESSION_COMPLETED,
            session_id="coding-session",
            interaction_id="coding-interaction",
            agent_name="coder",
            environment_name="coding",
            payload={"execution_profile_fingerprint": fingerprint},
        ),
    )


def _remote_fixture(
    tmp_path: Path,
    *,
    attributes: str | None = None,
) -> tuple[Path, str]:
    seed = tmp_path / "seed"
    seed.mkdir()
    _git(seed, "init", "--quiet", "--initial-branch=main")
    (seed / "app.py").write_text("VALUE = 'before'\n", encoding="utf-8")
    if attributes is not None:
        (seed / ".gitattributes").write_text(attributes, encoding="utf-8")
    _git(seed, "add", "-A")
    _git(
        seed,
        "-c",
        "user.name=Cayu Test",
        "-c",
        "user.email=cayu@example.invalid",
        "commit",
        "--quiet",
        "--no-verify",
        "-m",
        "base",
    )
    base = _git(seed, "rev-parse", "HEAD")
    remote = tmp_path / "remote.git"
    _git(tmp_path, "clone", "--quiet", "--bare", str(seed), str(remote))
    return remote, base


async def _coding_publication(
    tmp_path: Path,
    source: Path,
    *,
    final_content: bytes = b"VALUE = 'after'\n",
    baseline_commit: str | None = None,
):
    workspace = LocalWorkspace(source, workspace_id="source-workspace")
    limits = WorkspaceRevisionObservationLimits()
    initial = await observe_deterministic_workspace(
        workspace,
        observer="coding-source",
        limits=limits,
    )
    assert initial.revision is not None
    (source / "app.py").write_bytes(final_content)
    final = await observe_deterministic_workspace(
        workspace,
        observer="coding-source",
        limits=limits,
    )
    assert final.revision is not None
    seed = tmp_path / "seed"
    baseline = baseline_commit or _git(seed, "rev-parse", "HEAD")
    diff_content = (
        _git(seed, "--work-tree=" + str(source), "diff", "--no-ext-diff", "HEAD", "--") + "\n"
    )
    request = CodingProductRequest(
        product_run_id="coding-product",
        session_id="coding-session",
        agent_name="coder",
        source=CodingSourceAuthority(
            origin_id="application-source",
            workspace_id=workspace.id,
            baseline_revision=initial.revision,
            destination_id="application-destination",
            git_baseline=CodingGitBaselineAuthority(
                head_revision=baseline,
                staged_entries_sha256=_digest("clean-index"),
                tracked_flags_sha256=_digest("clean-index-flags"),
                status_sha256=_digest("clean-status"),
                diff_sha256=_digest("clean-diff"),
            ),
            observation_limits=limits,
        ),
        task=CodingTaskAuthority(
            task_id="coding-task",
            instruction_sha256=_digest("repair source"),
        ),
        runtime=CodingRuntimeAuthority(
            toolchain_profile_id="python",
            toolchain_profile_revision="1",
            toolchain_profile_fingerprint=_digest("toolchain"),
            image_fingerprint=_digest("image"),
            dependency_identity=_digest("dependencies"),
            execution_profile_fingerprint=_digest("execution"),
            tool_manifest_fingerprint=_digest("tools"),
            tool_policy_fingerprint=_digest("tool-policy"),
            approval_policy_fingerprint=_digest("coding-approval"),
            redaction_profile_fingerprint=_digest("coding-redaction"),
        ),
        settlement=CodingSettlementPolicy(
            required_checks=("format", "lint", "test"),
            reviewer_required=False,
        ),
    )
    store = LocalArtifactStore(tmp_path / "artifacts", store_id="delivery-artifacts")
    coding_repository = CodingProductArtifactRepository(store)
    await coding_repository.ensure_request(request)
    candidate = await compile_coding_product_candidate(
        request,
        (
            *(
                _check_event(name, workspace_revision=final.revision)
                for name in request.settlement.required_checks
            ),
            *_git_events(diff_content),
            *_terminal_events(request, final.revision, diff_content),
        ),
        initial_observation=initial,
        final_observation=final,
        repository=coding_repository,
    )
    assert candidate.state is CodingProductState.PATCH_READY_FOR_DELIVERY
    publication = await coding_repository.publish_candidate(candidate)
    return workspace, publication, coding_repository, store


def _broker(
    tmp_path: Path,
    remote: Path,
    coding_repository: CodingProductArtifactRepository,
    store: LocalArtifactStore,
) -> RemoteGitDeliveryBroker:
    remote_config = RemoteGitRemoteConfig(
        alias="origin",
        remote_identity="fixture-remote",
        url=str(remote),
        default_branch_ref="refs/heads/main",
    )
    profile = RemoteGitBrokerProfile(
        broker_id="fixture-broker",
        repository_id="fixture-repository",
        broker_repository_id="fixture-broker-repository",
        root=tmp_path / "broker",
        git_executable=shutil.which("git") or "/usr/bin/git",
        behavior_fingerprint=remote_git_broker_behavior_fingerprint(),
        remotes={"origin": remote_config},
    )
    return RemoteGitDeliveryBroker(
        profile,
        repository=RemoteGitDeliveryRepository(store),
        coding_repository=coding_repository,
    )


def _request(publication, base: str):
    return remote_git_delivery_request(
        publication,
        delivery_id="delivery-1",
        session_id="delivery-session",
        idempotency_key="application-delivery-1",
        repository=RemoteGitRepositoryAuthority(
            repository_id="fixture-repository",
            broker_repository_id="fixture-broker-repository",
            remote_alias="origin",
            remote_identity="fixture-remote",
            base_ref="refs/heads/main",
            expected_base_commit=base,
            destination_ref="refs/heads/cayu/delivery-1",
        ),
        commit=RemoteGitCommitAuthority(
            author_name="Cayu Delivery",
            author_email="cayu@example.invalid",
            committer_name="Cayu Delivery",
            committer_email="cayu@example.invalid",
            authored_at="2026-08-30T12:00:00-07:00",
            title="Repair source",
            body="Prepared and checked by Cayu.",
        ),
        security=RemoteGitSecurityAuthority(
            broker_behavior_fingerprint=remote_git_broker_behavior_fingerprint(),
            credential_profile_id="none",
            egress_profile_id="application-local",
            policy_fingerprint=_digest("delivery-policy"),
            approval_policy_fingerprint=_digest("approval-policy"),
            redaction_profile_fingerprint=_digest("delivery-redaction"),
        ),
    )


def test_broker_prepares_approves_pushes_and_recovers_exact_commit(tmp_path: Path) -> None:
    remote, base = _remote_fixture(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("VALUE = 'before'\n", encoding="utf-8")
    workspace, product, coding_repository, store = asyncio.run(
        _coding_publication(tmp_path, source)
    )
    broker = _broker(tmp_path, remote, coding_repository, store)
    request = _request(product, base)

    awaiting = asyncio.run(broker.run(request, product, source_workspace=workspace))
    assert awaiting.result.state is RemoteGitDeliveryState.APPROVAL_REQUIRED
    assert (
        _git(
            remote,
            "for-each-ref",
            "--format=%(refname)",
            request.repository.destination_ref,
        )
        == ""
    )
    prepared = asyncio.run(broker.repository.load_prepared(request))
    assert prepared is not None
    approval = approve_remote_git_delivery(
        request,
        prepared,
        approval_id="approval-1",
    )

    pushed = asyncio.run(
        broker.run(
            request,
            product,
            source_workspace=workspace,
            approval=approval,
        )
    )
    recovered = asyncio.run(
        broker.run(
            request,
            product,
            source_workspace=workspace,
            approval=approval,
        )
    )

    assert pushed.result.state is RemoteGitDeliveryState.PUSHED
    assert pushed.result.destination_after == pushed.result.local_commit
    assert pushed.result.next_commit == pushed.result.local_commit
    assert pushed.result.next_ref == request.repository.destination_ref
    assert pushed.result.cleanup_settled is True
    assert pushed.result.session_id == request.session_id
    assert pushed.result.product_result_artifact_id == request.source.product_result_artifact_id
    assert pushed.result.product_run_id == request.source.product_run_id
    assert pushed.result.diff_artifact_id == request.source.diff_artifact_id
    assert pushed.result.diff_sha256 == request.source.diff_sha256
    assert pushed.result.limits == request.limits
    assert recovered == pushed
    assert _git(remote, "rev-parse", request.repository.base_ref) == base
    assert (
        _git(remote, "rev-parse", request.repository.destination_ref) == pushed.result.local_commit
    )
    assert _git(remote, "rev-parse", f"{pushed.result.local_commit}^") == base
    assert "VALUE = 'after'" in _git(remote, "show", f"{pushed.result.local_commit}:app.py")


def test_delivery_rejects_default_branch_and_changed_source(tmp_path: Path) -> None:
    remote, base = _remote_fixture(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("VALUE = 'before'\n", encoding="utf-8")
    workspace, product, coding_repository, store = asyncio.run(
        _coding_publication(tmp_path, source)
    )
    broker = _broker(tmp_path, remote, coding_repository, store)
    request = _request(product, base)
    prepared = asyncio.run(broker.prepare(request, product, source_workspace=workspace))
    approval = approve_remote_git_delivery(request, prepared, approval_id="approval-1")
    (source / "app.py").write_text("VALUE = 'drifted'\n", encoding="utf-8")

    conflict = asyncio.run(
        broker.run(
            request,
            product,
            source_workspace=workspace,
            approval=approval,
        )
    )
    assert conflict.result.state is RemoteGitDeliveryState.CONFLICT
    assert conflict.result.reason_code == "source_evidence_changed_before_delivery"
    (source / "app.py").write_text("VALUE = 'after'\n", encoding="utf-8")

    default_request = request.model_copy(
        update={
            "delivery_id": "default-delivery",
            "idempotency_key": "default-delivery",
            "repository": request.repository.model_copy(
                update={"destination_ref": "refs/heads/main"}
            ),
        }
    )
    with pytest.raises(RemoteGitDeliveryAdmissionError, match="Default-branch"):
        asyncio.run(broker.prepare(default_request, product, source_workspace=workspace))

    wrong_repository = request.model_copy(
        update={
            "delivery_id": "wrong-repository",
            "idempotency_key": "wrong-repository",
            "repository": request.repository.model_copy(
                update={"repository_id": "unconfigured-repository"}
            ),
        }
    )
    with pytest.raises(RemoteGitDeliveryAdmissionError, match="Configured remote authority"):
        asyncio.run(broker.prepare(wrong_repository, product, source_workspace=workspace))

    storage_bounded = request.model_copy(
        update={
            "delivery_id": "storage-bounded",
            "idempotency_key": "storage-bounded",
            "limits": request.limits.model_copy(update={"max_git_storage_bytes": 1024}),
        }
    )
    # The process file cap may now reject before the aggregate storage census.
    with pytest.raises(RemoteGitDeliveryError):
        asyncio.run(broker.prepare(storage_bounded, product, source_workspace=workspace))
    assert asyncio.run(broker.repository.load_prepared(storage_bounded)) is None


def test_delivery_denial_and_existing_destination_never_push(tmp_path: Path) -> None:
    remote, base = _remote_fixture(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("VALUE = 'before'\n", encoding="utf-8")
    workspace, product, coding_repository, store = asyncio.run(
        _coding_publication(tmp_path, source)
    )
    broker = _broker(tmp_path, remote, coding_repository, store)
    request = _request(product, base)
    prepared = asyncio.run(broker.prepare(request, product, source_workspace=workspace))
    denied_approval = RemoteGitDeliveryApproval(
        approval_id="denied-approval",
        request_fingerprint=request.fingerprint,
        prepared_tree=prepared.tree,
        policy_fingerprint=request.security.policy_fingerprint,
        commit_approved=False,
        push_approved=False,
    )

    denied = asyncio.run(
        broker.run(
            request,
            product,
            source_workspace=workspace,
            approval=denied_approval,
        )
    )
    assert denied.result.state is RemoteGitDeliveryState.DENIED
    assert (
        _git(
            remote,
            "for-each-ref",
            "--format=%(refname)",
            request.repository.destination_ref,
        )
        == ""
    )

    existing_request = request.model_copy(
        update={
            "delivery_id": "existing-destination",
            "idempotency_key": "existing-destination",
            "repository": request.repository.model_copy(
                update={"destination_ref": "refs/heads/cayu/existing"}
            ),
        }
    )
    _git(remote, "update-ref", existing_request.repository.destination_ref, base)
    existing = asyncio.run(
        broker.run(
            existing_request,
            product,
            source_workspace=workspace,
        )
    )
    assert existing.result.state is RemoteGitDeliveryState.CONFLICT
    assert existing.result.reason_code == "destination_ref_already_exists"
    assert existing.result.destination_after == base

    tree = _git(remote, "rev-parse", f"{base}^{{tree}}")
    advanced = _git(
        remote,
        "-c",
        "user.name=Cayu Test",
        "-c",
        "user.email=cayu@example.invalid",
        "commit-tree",
        tree,
        "-p",
        base,
        "-m",
        "advanced base",
    )
    _git(remote, "update-ref", request.repository.base_ref, advanced)
    changed_base_request = request.model_copy(
        update={
            "delivery_id": "changed-base",
            "idempotency_key": "changed-base",
            "repository": request.repository.model_copy(
                update={"destination_ref": "refs/heads/cayu/changed-base"}
            ),
        }
    )
    changed_base = asyncio.run(
        broker.run(
            changed_base_request,
            product,
            source_workspace=workspace,
        )
    )
    assert changed_base.result.state is RemoteGitDeliveryState.CONFLICT
    assert changed_base.result.reason_code == "remote_base_changed_before_preparation"
    assert changed_base.result.observed_base_commit == advanced


def test_lost_push_acknowledgement_reconciles_without_duplicate_push(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote, base = _remote_fixture(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("VALUE = 'before'\n", encoding="utf-8")
    workspace, product, coding_repository, store = asyncio.run(
        _coding_publication(tmp_path, source)
    )
    broker = _broker(tmp_path, remote, coding_repository, store)
    request = _request(product, base)
    prepared = asyncio.run(broker.prepare(request, product, source_workspace=workspace))
    approval = approve_remote_git_delivery(request, prepared, approval_id="approval-1")
    original_push = broker._push

    async def push_then_lose_ack(*args, **kwargs):
        await original_push(*args, **kwargs)
        raise RemoteGitDeliveryError("simulated acknowledgement loss")

    monkeypatch.setattr(broker, "_push", push_then_lose_ack)
    reconciled = asyncio.run(
        broker.run(
            request,
            product,
            source_workspace=workspace,
            approval=approval,
        )
    )

    assert reconciled.result.state is RemoteGitDeliveryState.PUSHED
    assert (
        _git(remote, "rev-parse", request.repository.destination_ref)
        == reconciled.result.local_commit
    )


def test_broker_repository_hooks_cannot_run_during_delivery(tmp_path: Path) -> None:
    remote, base = _remote_fixture(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("VALUE = 'before'\n", encoding="utf-8")
    workspace, product, coding_repository, store = asyncio.run(
        _coding_publication(tmp_path, source)
    )
    broker = _broker(tmp_path, remote, coding_repository, store)
    request = _request(product, base)
    prepared = asyncio.run(broker.prepare(request, product, source_workspace=workspace))
    approval = approve_remote_git_delivery(request, prepared, approval_id="approval-1")
    marker = tmp_path / "hook-ran"
    hook = broker._delivery_root(request) / ".git" / "hooks" / "pre-push"
    hook.write_text(f"#!/bin/sh\ntouch {marker}\nexit 1\n", encoding="utf-8")
    hook.chmod(0o700)

    pushed = asyncio.run(
        broker.run(
            request,
            product,
            source_workspace=workspace,
            approval=approval,
        )
    )

    assert pushed.result.state is RemoteGitDeliveryState.PUSHED
    assert marker.exists() is False


def test_unobserved_push_failure_is_ambiguous_and_recoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote, base = _remote_fixture(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("VALUE = 'before'\n", encoding="utf-8")
    workspace, product, coding_repository, store = asyncio.run(
        _coding_publication(tmp_path, source)
    )
    broker = _broker(tmp_path, remote, coding_repository, store)
    request = _request(product, base)
    prepared = asyncio.run(broker.prepare(request, product, source_workspace=workspace))
    approval = approve_remote_git_delivery(request, prepared, approval_id="approval-1")
    original_push = broker._push

    async def lose_before_push(*args, **kwargs):
        del args, kwargs
        raise RemoteGitDeliveryError("simulated transport failure")

    monkeypatch.setattr(broker, "_push", lose_before_push)
    ambiguous = asyncio.run(
        broker.run(
            request,
            product,
            source_workspace=workspace,
            approval=approval,
        )
    )
    assert ambiguous.result.state is RemoteGitDeliveryState.AMBIGUOUS
    assert ambiguous.result.local_commit is not None
    monkeypatch.setattr(broker, "_push", original_push)

    recovered = asyncio.run(
        broker.run(
            request,
            product,
            source_workspace=workspace,
            approval=approval,
        )
    )
    assert recovered.result.state is RemoteGitDeliveryState.PUSHED
    assert recovered.result.local_commit == ambiguous.result.local_commit


def test_cleanup_failure_is_partial_after_exact_remote_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote, base = _remote_fixture(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("VALUE = 'before'\n", encoding="utf-8")
    workspace, product, coding_repository, store = asyncio.run(
        _coding_publication(tmp_path, source)
    )
    broker = _broker(tmp_path, remote, coding_repository, store)
    request = _request(product, base)
    prepared = asyncio.run(broker.prepare(request, product, source_workspace=workspace))
    approval = approve_remote_git_delivery(request, prepared, approval_id="approval-1")

    async def fail_cleanup(request, root):
        return False

    monkeypatch.setattr(broker, "_cleanup_delivery_root", fail_cleanup)

    partial = asyncio.run(
        broker.run(
            request,
            product,
            source_workspace=workspace,
            approval=approval,
        )
    )

    assert partial.result.state is RemoteGitDeliveryState.PARTIAL
    assert partial.result.reason_code == "cleanup_unsettled"
    assert partial.result.destination_after == partial.result.local_commit
    assert partial.result.next_commit is None
    assert partial.result.cleanup_settled is False


def test_cancellation_after_commit_recovers_the_same_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote, base = _remote_fixture(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("VALUE = 'before'\n", encoding="utf-8")
    workspace, product, coding_repository, store = asyncio.run(
        _coding_publication(tmp_path, source)
    )
    broker = _broker(tmp_path, remote, coding_repository, store)
    request = _request(product, base)
    prepared = asyncio.run(broker.prepare(request, product, source_workspace=workspace))
    approval = approve_remote_git_delivery(request, prepared, approval_id="approval-1")
    original_push = broker._push

    async def cancel_push(*args, **kwargs):
        del args, kwargs
        raise asyncio.CancelledError

    monkeypatch.setattr(broker, "_push", cancel_push)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            broker.run(
                request,
                product,
                source_workspace=workspace,
                approval=approval,
            )
        )
    receipts = asyncio.run(broker.repository.load_lifecycle(request))
    committed = next(
        receipt.commit
        for receipt in reversed(receipts)
        if receipt.state is RemoteGitDeliveryState.COMMITTED_LOCALLY
    )
    monkeypatch.setattr(broker, "_push", original_push)

    recovered = asyncio.run(
        broker.run(
            request,
            product,
            source_workspace=workspace,
            approval=approval,
        )
    )
    assert recovered.result.state is RemoteGitDeliveryState.PUSHED
    assert recovered.result.local_commit == committed


def test_cancellation_before_commit_records_no_remote_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote, base = _remote_fixture(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("VALUE = 'before'\n", encoding="utf-8")
    workspace, product, coding_repository, store = asyncio.run(
        _coding_publication(tmp_path, source)
    )
    broker = _broker(tmp_path, remote, coding_repository, store)
    request = _request(product, base)

    async def cancel_prepare(*args, **kwargs):
        del args, kwargs
        raise asyncio.CancelledError

    monkeypatch.setattr(broker, "_prepare_owned", cancel_prepare)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(broker.run(request, product, source_workspace=workspace))
    receipts = asyncio.run(broker.repository.load_lifecycle(request))
    assert receipts[-1].state is RemoteGitDeliveryState.CANCELLED
    assert (
        _git(
            remote,
            "for-each-ref",
            "--format=%(refname)",
            request.repository.destination_ref,
        )
        == ""
    )


def test_local_commit_failure_is_definite_and_never_pushes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote, base = _remote_fixture(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("VALUE = 'before'\n", encoding="utf-8")
    workspace, product, coding_repository, store = asyncio.run(
        _coding_publication(tmp_path, source)
    )
    broker = _broker(tmp_path, remote, coding_repository, store)
    request = _request(product, base)
    prepared = asyncio.run(broker.prepare(request, product, source_workspace=workspace))
    approval = approve_remote_git_delivery(request, prepared, approval_id="approval-1")

    async def fail_commit(*args, **kwargs):
        del args, kwargs
        raise RemoteGitDeliveryError("simulated local commit failure")

    monkeypatch.setattr(broker, "_commit_tree", fail_commit)
    failed = asyncio.run(
        broker.run(
            request,
            product,
            source_workspace=workspace,
            approval=approval,
        )
    )

    assert failed.result.state is RemoteGitDeliveryState.FAILED
    assert failed.result.reason_code == "local_commit_failed"
    assert failed.result.local_commit is None
    assert (
        _git(
            remote,
            "for-each-ref",
            "--format=%(refname)",
            request.repository.destination_ref,
        )
        == ""
    )


def test_source_drift_after_local_commit_prevents_push(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote, base = _remote_fixture(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("VALUE = 'before'\n", encoding="utf-8")
    workspace, product, coding_repository, store = asyncio.run(
        _coding_publication(tmp_path, source)
    )
    broker = _broker(tmp_path, remote, coding_repository, store)
    request = _request(product, base)
    prepared = asyncio.run(broker.prepare(request, product, source_workspace=workspace))
    approval = approve_remote_git_delivery(request, prepared, approval_id="approval-1")
    original_commit = broker._commit_tree

    async def commit_then_change_source(*args, **kwargs):
        result = await original_commit(*args, **kwargs)
        (source / "app.py").write_text("VALUE = 'changed-again'\n", encoding="utf-8")
        return result

    monkeypatch.setattr(broker, "_commit_tree", commit_then_change_source)
    conflict = asyncio.run(
        broker.run(
            request,
            product,
            source_workspace=workspace,
            approval=approval,
        )
    )

    assert conflict.result.state is RemoteGitDeliveryState.CONFLICT
    assert conflict.result.reason_code == "source_evidence_changed_before_push"
    assert conflict.result.local_commit is not None
    assert (
        _git(
            remote,
            "for-each-ref",
            "--format=%(refname)",
            request.repository.destination_ref,
        )
        == ""
    )


def test_push_rejection_is_not_reported_as_success(tmp_path: Path) -> None:
    remote, base = _remote_fixture(tmp_path)
    hook = remote / "hooks" / "pre-receive"
    hook.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    hook.chmod(0o700)
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("VALUE = 'before'\n", encoding="utf-8")
    workspace, product, coding_repository, store = asyncio.run(
        _coding_publication(tmp_path, source)
    )
    broker = _broker(tmp_path, remote, coding_repository, store)
    request = _request(product, base)
    prepared = asyncio.run(broker.prepare(request, product, source_workspace=workspace))
    approval = approve_remote_git_delivery(request, prepared, approval_id="approval-1")

    rejected = asyncio.run(
        broker.run(
            request,
            product,
            source_workspace=workspace,
            approval=approval,
        )
    )
    assert rejected.result.state is RemoteGitDeliveryState.FAILED
    assert rejected.result.reason_code == "push_rejected"


def test_vault_credentials_are_redacted_before_step_evidence(tmp_path: Path) -> None:
    username = "delivery-user-secret"
    password = "delivery-password-secret"
    vault = StaticVault({"username": username, "password": password})
    executable = tmp_path / "fake-git"
    executable.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$CAYU_GIT_USERNAME\"\n"
        "printf '%s\\n' \"$CAYU_GIT_PASSWORD\" >&2\n",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    remote = RemoteGitRemoteConfig(
        alias="origin",
        remote_identity="authenticated-remote",
        url="https://git.example.invalid/repository.git",
        default_branch_ref="refs/heads/main",
        credential_profile_id="vault-test",
        egress_profile_id="https-only-test",
        credentials=RemoteGitHttpCredentials(
            credential_profile_id="vault-test",
            username=SecretRef(name="username"),
            password=SecretRef(name="password"),
            resolver=vault,
        ),
    )
    profile = RemoteGitBrokerProfile(
        broker_id="redaction-broker",
        repository_id="repository",
        broker_repository_id="broker-repository",
        root=tmp_path / "broker",
        git_executable=str(executable),
        behavior_fingerprint=remote_git_broker_behavior_fingerprint(),
        remotes={"origin": remote},
    )
    store = LocalArtifactStore(tmp_path / "artifacts", store_id="redaction-store")
    broker = RemoteGitDeliveryBroker(
        profile,
        repository=RemoteGitDeliveryRepository(store),
        coding_repository=CodingProductArtifactRepository(store),
    )
    request = RemoteGitDeliveryRequest(
        delivery_id="redaction-delivery",
        session_id="redaction-session",
        idempotency_key="redaction-delivery",
        source=RemoteGitSourceAuthority(
            product_result_artifact_id="art_11111111111111111111111111111111",
            product_result_sha256=_digest("product"),
            product_request_fingerprint=_digest("request"),
            product_run_id="product",
            source_workspace_id="workspace",
            final_source_revision=_digest("source"),
            diff_artifact_id="art_22222222222222222222222222222222",
            diff_sha256=_digest("diff"),
            check_evidence_sha256=_digest("checks"),
        ),
        repository=RemoteGitRepositoryAuthority(
            repository_id="repository",
            broker_repository_id="broker-repository",
            remote_alias="origin",
            remote_identity="authenticated-remote",
            base_ref="refs/heads/main",
            expected_base_commit="1" * 40,
            destination_ref="refs/heads/cayu/redaction",
        ),
        commit=RemoteGitCommitAuthority(
            author_name="Cayu",
            author_email="cayu@example.invalid",
            committer_name="Cayu",
            committer_email="cayu@example.invalid",
            authored_at="2026-08-30T12:00:00-07:00",
            title="Redaction test",
        ),
        security=RemoteGitSecurityAuthority(
            broker_behavior_fingerprint=remote_git_broker_behavior_fingerprint(),
            credential_profile_id="vault-test",
            egress_profile_id="https-only-test",
            policy_fingerprint=_digest("policy"),
            approval_policy_fingerprint=_digest("approval"),
            redaction_profile_fingerprint=_digest("redaction"),
        ),
    )

    result, evidence = asyncio.run(
        broker._git(
            request,
            "credential_redaction_probe",
            ("version",),
            cwd=profile.root,
            remote=remote,
        )
    )

    published = f"{result.stdout}\n{result.stderr}\n{evidence.model_dump_json()}"
    assert username not in published
    assert password not in published
    assert REDACTED_SECRET in published


def test_git_attributes_cannot_transform_the_approved_source_tree(tmp_path: Path) -> None:
    attributes = "*.py text\n"
    remote, base = _remote_fixture(tmp_path, attributes=attributes)
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("VALUE = 'before'\n", encoding="utf-8")
    (source / ".gitattributes").write_text(attributes, encoding="utf-8")
    workspace, product, coding_repository, store = asyncio.run(
        _coding_publication(
            tmp_path,
            source,
            final_content=b"VALUE = 'after'\r\n",
        )
    )
    broker = _broker(tmp_path, remote, coding_repository, store)
    request = _request(product, base)

    with pytest.raises(RemoteGitDeliveryAdmissionError, match="attributes or filters"):
        asyncio.run(broker.prepare(request, product, source_workspace=workspace))


@pytest.mark.parametrize("field", ["force_update", "delete_ref"])
def test_force_and_delete_authority_are_unrepresentable(field: str) -> None:
    payload = {
        "delivery_id": "prohibited-effect",
        "session_id": "delivery-session",
        "idempotency_key": "prohibited-effect",
        "source": {
            "product_result_artifact_id": "art_11111111111111111111111111111111",
            "product_result_sha256": _digest("product"),
            "product_request_fingerprint": _digest("request"),
            "product_run_id": "product",
            "source_workspace_id": "workspace",
            "final_source_revision": _digest("source"),
            "diff_artifact_id": "art_22222222222222222222222222222222",
            "diff_sha256": _digest("diff"),
            "check_evidence_sha256": _digest("checks"),
        },
        "repository": {
            "repository_id": "repository",
            "broker_repository_id": "broker-repository",
            "remote_alias": "origin",
            "remote_identity": "remote",
            "base_ref": "refs/heads/main",
            "expected_base_commit": "1" * 40,
            "destination_ref": "refs/heads/cayu/change",
        },
        "commit": {
            "author_name": "Cayu",
            "author_email": "cayu@example.invalid",
            "committer_name": "Cayu",
            "committer_email": "cayu@example.invalid",
            "authored_at": "2026-08-30T12:00:00-07:00",
            "title": "Prohibited effect",
        },
        "security": {
            "broker_behavior_fingerprint": remote_git_broker_behavior_fingerprint(),
            "credential_profile_id": "none",
            "egress_profile_id": "local",
            "policy_fingerprint": _digest("policy"),
            "approval_policy_fingerprint": _digest("approval"),
            "redaction_profile_fingerprint": _digest("redaction"),
        },
        field: True,
    }
    with pytest.raises(ValidationError):
        RemoteGitDeliveryRequest.model_validate(payload)


def test_git_object_ids_require_an_exact_supported_hash_width() -> None:
    with pytest.raises(ValidationError, match="full Git object ID"):
        RemoteGitRepositoryAuthority(
            repository_id="repository",
            broker_repository_id="broker-repository",
            remote_alias="origin",
            remote_identity="remote",
            base_ref="refs/heads/main",
            expected_base_commit="1" * 41,
            destination_ref="refs/heads/cayu/change",
        )


def test_credentialed_remote_requires_https() -> None:
    credentials = RemoteGitHttpCredentials(
        credential_profile_id="vault-test",
        username=SecretRef(name="username"),
        password=SecretRef(name="password"),
        resolver=StaticVault({"username": "user", "password": "secret"}),
    )

    with pytest.raises(ValueError, match="require HTTPS"):
        RemoteGitRemoteConfig(
            alias="origin",
            remote_identity="authenticated-remote",
            url="http://git.example.invalid/repository.git",
            default_branch_ref="refs/heads/main",
            credential_profile_id="vault-test",
            credentials=credentials,
        )


@pytest.mark.parametrize("url", ["--upload-pack=attacker", "ext::attacker-command"])
def test_remote_url_cannot_invoke_git_options_or_helpers(url: str) -> None:
    with pytest.raises(ValueError, match="options or remote helpers"):
        RemoteGitRemoteConfig(
            alias="origin",
            remote_identity="unsafe-remote",
            url=url,
            default_branch_ref="refs/heads/main",
        )


def test_broker_profile_rejects_relative_host_paths(tmp_path: Path) -> None:
    remote = RemoteGitRemoteConfig(
        alias="origin",
        remote_identity="fixture-remote",
        url=str(tmp_path / "remote.git"),
        default_branch_ref="refs/heads/main",
    )
    with pytest.raises(ValueError, match="bounded absolute directory"):
        RemoteGitBrokerProfile(
            broker_id="fixture-broker",
            repository_id="fixture-repository",
            broker_repository_id="fixture-broker-repository",
            root=Path("relative-broker"),
            git_executable=shutil.which("git") or "/usr/bin/git",
            behavior_fingerprint=remote_git_broker_behavior_fingerprint(),
            remotes={"origin": remote},
        )


def test_broker_rejects_symlinked_delivery_root(tmp_path: Path) -> None:
    remote, base = _remote_fixture(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("VALUE = 'before'\n", encoding="utf-8")
    workspace, product, coding_repository, store = asyncio.run(
        _coding_publication(tmp_path, source)
    )
    broker = _broker(tmp_path, remote, coding_repository, store)
    request = _request(product, base)
    protected = broker.profile.root / "protected"
    protected.mkdir()
    marker = protected / "marker"
    marker.write_text("retain", encoding="utf-8")
    broker._delivery_root(request).symlink_to(protected, target_is_directory=True)

    with pytest.raises(RemoteGitDeliveryAdmissionError, match="delivery root is unsafe"):
        asyncio.run(broker.prepare(request, product, source_workspace=workspace))
    assert marker.read_text(encoding="utf-8") == "retain"


@pytest.mark.parametrize("ordinal", [2, 3, 48])
def test_repository_detects_lifecycle_reconstruction_gap(tmp_path: Path, ordinal: int) -> None:
    _, base = _remote_fixture(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("VALUE = 'before'\n", encoding="utf-8")
    _, product, _, store = asyncio.run(_coding_publication(tmp_path, source))
    repository = RemoteGitDeliveryRepository(store)
    request = _request(product, base)
    receipt = RemoteGitLifecycleReceipt(
        delivery_id=request.delivery_id,
        request_fingerprint=request.fingerprint,
        ordinal=ordinal,
        prior_state=RemoteGitDeliveryState.PREPARING,
        state=RemoteGitDeliveryState.PREPARED,
    )
    asyncio.run(repository.append_lifecycle(request, receipt))

    with pytest.raises(
        RemoteGitDeliveryReconstructionRequiredError,
        match="reconstruction gap",
    ):
        asyncio.run(repository.load_lifecycle(request))


def test_repository_rejects_wrong_session_artifact_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, base = _remote_fixture(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("VALUE = 'before'\n", encoding="utf-8")
    _, product, _, store = asyncio.run(_coding_publication(tmp_path, source))
    repository = RemoteGitDeliveryRepository(store)
    request = _request(product, base)
    asyncio.run(repository.ensure_request(request))
    request_artifact_id = repository.request_artifact_id(request.delivery_id)
    original_read = store.read_bytes

    async def wrong_session_read(artifact_id: str, *, max_bytes: int | None = None):
        result = await original_read(artifact_id, max_bytes=max_bytes)
        if artifact_id != request_artifact_id:
            return result
        return ArtifactReadResult(
            metadata=result.metadata.model_copy(update={"session_id": "different-session"}),
            content=result.content,
            total_bytes=result.total_bytes,
            truncated=result.truncated,
            source_bytes_read=result.source_bytes_read,
            redaction_truncated=result.redaction_truncated,
        )

    monkeypatch.setattr(store, "read_bytes", wrong_session_read)
    with pytest.raises(
        RemoteGitDeliveryReconstructionRequiredError,
        match="artifact authority",
    ):
        asyncio.run(repository.ensure_request(request))


def test_broker_rejects_wrong_diff_artifact_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote, base = _remote_fixture(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("VALUE = 'before'\n", encoding="utf-8")
    workspace, product, coding_repository, store = asyncio.run(
        _coding_publication(tmp_path, source)
    )
    broker = _broker(tmp_path, remote, coding_repository, store)
    request = _request(product, base)
    diff_artifact_id = request.source.diff_artifact_id
    original_read = store.read_bytes

    async def wrong_filename_read(artifact_id: str, *, max_bytes: int | None = None):
        result = await original_read(artifact_id, max_bytes=max_bytes)
        if artifact_id != diff_artifact_id:
            return result
        return ArtifactReadResult(
            metadata=result.metadata.model_copy(update={"filename": "untrusted.diff"}),
            content=result.content,
            total_bytes=result.total_bytes,
            truncated=result.truncated,
            source_bytes_read=result.source_bytes_read,
            redaction_truncated=result.redaction_truncated,
        )

    monkeypatch.setattr(store, "read_bytes", wrong_filename_read)
    with pytest.raises(RemoteGitDeliveryAdmissionError, match="artifact authority"):
        asyncio.run(broker.prepare(request, product, source_workspace=workspace))


def test_askpass_helper_rejects_hard_link_alias(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts", store_id="askpass-artifacts")
    remote = RemoteGitRemoteConfig(
        alias="origin",
        remote_identity="fixture-remote",
        url=str(tmp_path / "remote.git"),
        default_branch_ref="refs/heads/main",
    )
    profile = RemoteGitBrokerProfile(
        broker_id="fixture-broker",
        repository_id="fixture-repository",
        broker_repository_id="fixture-broker-repository",
        root=tmp_path / "broker",
        git_executable=shutil.which("git") or "/usr/bin/git",
        behavior_fingerprint=remote_git_broker_behavior_fingerprint(),
        remotes={"origin": remote},
    )
    broker = RemoteGitDeliveryBroker(
        profile,
        repository=RemoteGitDeliveryRepository(store),
        coding_repository=CodingProductArtifactRepository(store),
    )
    helper = broker._askpass_path()
    target = helper.parent / "hard-link-target"
    target.write_bytes(helper.read_bytes())
    helper.unlink()
    helper.hardlink_to(target)

    with pytest.raises(RemoteGitDeliveryAdmissionError, match="helper identity"):
        broker._askpass_path()


@pytest.mark.parametrize(
    "path",
    [
        ".git/config",
        ".GIT/config",
        ".cayu/authority.json",
        ".runtime/state",
        "safe\\..\\.git/config",
    ],
)
def test_prepared_intent_rejects_protected_source_paths(path: str) -> None:
    with pytest.raises(RemoteGitDeliveryAdmissionError, match="protected path"):
        RemoteGitPreparedIntent.model_validate(
            {
                "delivery_id": "protected-path",
                "request_fingerprint": _digest("request"),
                "repository_id": "repository",
                "remote_identity": "remote",
                "observed_base_commit": "1" * 40,
                "tree": "2" * 40,
                "changed_paths": (path,),
                "source_revision": _digest("source"),
                "source_manifest_sha256": _digest("manifest"),
                "commit_message_sha256": _digest("message"),
                "steps": (),
            }
        )
