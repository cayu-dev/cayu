"""Local Git integration assertions; no GitHub or paid provider access."""

import importlib
import json
import os
import shutil
import subprocess
from hashlib import sha256
from unittest.mock import patch

import httpx
import pytest

from cayu import (
    RemoteGitCommitAuthority,
    RemoteGitDeliveryApproval,
    RemoteGitDeliveryError,
    RemoteGitDeliveryLimits,
    RemoteGitDeliveryRequest,
    RemoteGitDeliveryState,
    RemoteGitRepositoryAuthority,
    RemoteGitSecurityAuthority,
    TaskQuery,
    TaskStatus,
    approve_remote_git_delivery,
    remote_git_broker_behavior_fingerprint,
    remote_git_delivery_request,
    run_task_worker,
)
from cayu.server import ProductPrincipal
from tests.qualification.repository_maintenance_case import SEED_BASE_REVISION
from tests.qualification.repository_maintenance_github_case import exercise_github_delivery
from tests.qualification.repository_maintenance_restart_case import exercise_approval_restart


def local_git(root, *arguments):
    return subprocess.run(
        [
            shutil.which("git") or "/usr/bin/git",
            "-c",
            f"core.hooksPath={os.devnull}",
            "-C",
            str(root),
            *arguments,
        ],
        env={
            "PATH": os.defpath,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "LC_ALL": "C",
        },
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout.strip()


def build_journey_http(application, reservations, *, tenant, subject):
    """Use the actual host with its budget guard, not a fixture-only HTTP owner."""
    auth = importlib.import_module("integrations.maintenance_auth")
    access = auth.MaintenanceAccess(
        product_tokens={"fixture-product": ProductPrincipal(tenant_id=tenant, subject_id=subject)},
        operator_token="fixture-operator",
    )
    return importlib.import_module("operations.maintenance_http").build_maintenance_server(
        application, reservations, access
    )


async def exercise_local_delivery(
    application,
    task,
    publication,
    *,
    root,
    remote,
    incorrect_probe,
    lose_push_ack=False,
    lose_github_ack=False,
    reservations=None,
    identity=None,
    restart_approval=False,
):
    integration = importlib.import_module("integrations.remote_git")
    workflow = importlib.import_module("workflows.maintenance_delivery")
    assert integration.REMOTE_GIT_DELIVERY_ENABLED
    assert importlib.import_module("integrations.github").GITHUB_DELIVERY_ENABLED

    def build_broker():
        return integration.build_remote_git_delivery_broker(
            artifact_store=application.artifact_store,
            broker_root=root / "broker",
            git_executable=shutil.which("git") or "/usr/bin/git",
            repository_id="fixture",
            broker_repository_id="fixture-broker",
            remote_url=str(remote),
            remote_identity="fixture-remote",
            default_branch_ref="refs/heads/main",
        )

    broker = build_broker()
    policy = "sha256:" + sha256(b"local-qualification-explicit-human-approval").hexdigest()
    configured = {
        "repository": RemoteGitRepositoryAuthority(
            repository_id="fixture",
            broker_repository_id="fixture-broker",
            remote_alias="origin",
            remote_identity="fixture-remote",
            base_ref="refs/heads/main",
            expected_base_commit=SEED_BASE_REVISION,
            destination_ref="refs/heads/cayu/verified-delivery",
        ).model_dump(mode="json"),
        "commit": RemoteGitCommitAuthority(
            author_name="Qualification",
            author_email="fixture@example.invalid",
            committer_name="Qualification",
            committer_email="fixture@example.invalid",
            authored_at="2000-01-02T00:00:00+00:00",
            title="Repair inclusive endpoint",
        ).model_dump(mode="json"),
        "security": RemoteGitSecurityAuthority(
            broker_behavior_fingerprint=remote_git_broker_behavior_fingerprint(),
            credential_profile_id="none",
            egress_profile_id="application-local",
            policy_fingerprint=policy,
            approval_policy_fingerprint=policy,
            redaction_profile_fingerprint=policy,
        ).model_dump(mode="json"),
        "limits": RemoteGitDeliveryLimits(
            max_paths=3, max_file_bytes=16384, max_total_file_bytes=49152
        ).model_dump(mode="json"),
    }
    configuration = importlib.import_module("configuration.maintenance")
    queued_request = None
    queued_approval = None
    with patch.dict(os.environ, {"CAYU_MAINTENANCE_GIT_JSON": json.dumps(configured)}):
        repository, commit, security, limits = configuration.configured_maintenance_git_authority()
        if reservations is not None and not incorrect_probe:
            assert identity is not None
            # Complete real Runtime queue/claim -> native pending approval result.
            api = build_journey_http(
                application,
                reservations,
                tenant=identity.intent.tenant,
                subject=identity.intent.subject,
            )

            async def operator_request(method, suffix, body=None, *, token="fixture-operator"):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=api), base_url="http://fixture"
                ) as client:
                    return await client.request(
                        method,
                        f"/operator/runs/{identity.public_id}/git/{suffix}",
                        params={"tenant": identity.intent.tenant},
                        headers={"authorization": f"Bearer {token}"},
                        json=body,
                    )

            assert (
                await operator_request("POST", "preparation", {}, token="fixture-product")
            ).status_code == 401
            response = await operator_request("POST", "preparation", {})
            assert response.status_code == 202, response.text
            queued = await application.app.task_store.load_task(identity.git_preparation_task_id)
            assert queued is not None
            try:
                native = RemoteGitDeliveryRequest.model_validate_json(queued.input["request_json"])
                assert native.delivery_id == identity.git_delivery_task_id
                assert native.source.product_run_id == task.product_run_id
                assert native.source.product_result_sha256 == publication.artifact.sha256
                assert queued.status is TaskStatus.PENDING
                assert queued.invocation.origin.subject == "maintenance-operator"
                assert not list(broker.profile.root.iterdir())
                assert (await operator_request("POST", "preparation", {})).status_code == 202
                assert await application.app.task_store.load_task(queued.id) == queued
                worker = importlib.import_module("operations.maintenance_git_worker")

                async def handle(_app, claimed, worker_id):
                    await worker.handle_git_preparation_task(
                        application, reservations, claimed, worker_id, broker
                    )

                if restart_approval:
                    exercise_approval_restart(
                        application, reservations, identity, root=root, remote=remote
                    )
                else:
                    assert (
                        await run_task_worker(
                            application.app,
                            application.app.task_store,
                            handle,
                            worker_id="qualification-git-preparation",
                            query=TaskQuery(type="maintenance.git_preparation"),
                            max_tasks=1,
                            recover_interrupted_handoffs=False,
                        )
                        == 1
                    )
                completed = await application.app.task_store.load_task(queued.id)
                assert completed.status is TaskStatus.COMPLETED
                assert completed.worker_id is None and completed.lease_expires_at is None
                assert completed.result["request_fingerprint"] == native.fingerprint
                pending = await broker.repository.load_result(
                    native, completed.result["result_digest"]
                )
                assert pending.result.state is RemoteGitDeliveryState.APPROVAL_REQUIRED
                assert not pending.result.cleanup_settled
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=api), base_url="http://fixture"
                ) as client:
                    evidence = await client.get(
                        f"/operator/runs/{identity.public_id}/delivery",
                        params={"tenant": identity.intent.tenant},
                        headers={"authorization": "Bearer fixture-operator"},
                    )
                assert evidence.status_code == 200, evidence.text
                observed = evidence.json()["git"]
                assert observed["state"] == "approval_required"
                assert observed["cleanup_evidence"] == "recorded_unsettled"
                assert observed["task_receipts"]["preparation"] == "matches_latest"
                response = await operator_request("GET", "approval")
                assert response.status_code == 200, response.text
                review = response.json()
                assert review["request"] == native.model_dump(mode="json")
                assert review["pending_result_digest"] == pending.artifact.sha256
                approval_body = {
                    "request_fingerprint": review["request_fingerprint"],
                    "prepared_tree": review["prepared_tree"],
                    "approval_id": "fixture-human-decision",
                }
                before_approval = await application.app.task_store.load_task(
                    identity.git_delivery_task_id
                )
                assert (
                    await operator_request(
                        "POST", "approval", dict(approval_body, prepared_tree="0" * 40)
                    )
                ).status_code == 409
                assert (
                    await application.app.task_store.load_task(identity.git_delivery_task_id)
                    == before_approval
                )
                response = await operator_request("POST", "approval", approval_body)
                assert response.status_code == 202, response.text
                approved_task = await application.app.task_store.load_task(
                    identity.git_delivery_task_id
                )
                assert approved_task is not None
                try:
                    queued_approval = RemoteGitDeliveryApproval.model_validate_json(
                        approved_task.input["approval_json"]
                    )
                    assert queued_approval.request_fingerprint == native.fingerprint
                    assert queued_approval.prepared_tree == review["prepared_tree"]
                    assert approved_task.status is TaskStatus.PENDING
                    assert (
                        await operator_request("POST", "approval", approval_body)
                    ).status_code == 202
                    assert (
                        await application.app.task_store.load_task(approved_task.id)
                        == approved_task
                    )
                except BaseException:
                    stopped = await application.app.task_store.cancel_task(approved_task.id)
                    assert stopped.status is TaskStatus.CANCELLED
                    raise
                queued_request = native
            finally:
                current = await application.app.task_store.load_task(queued.id)
                if current.status is TaskStatus.PENDING:
                    stopped = await application.app.task_store.cancel_task(queued.id)
                    assert stopped.status is TaskStatus.CANCELLED
    request = queued_request or remote_git_delivery_request(
        publication,
        delivery_id="verified-delivery",
        session_id=task.session_id,
        idempotency_key="verified-delivery",
        repository=repository,
        commit=commit,
        security=security,
        limits=limits,
    )
    refs_before = local_git(remote, "show-ref")
    if incorrect_probe:
        rejection = importlib.import_module(
            "domain.maintenance_acceptance"
        ).MaintenanceAcceptanceRejected
        with pytest.raises(rejection):
            await workflow.prepare_verified_git_delivery(
                application, task, publication, broker, request
            )
        with pytest.raises(rejection):
            await workflow.run_verified_git_delivery(
                application, task, publication, broker, request
            )
        assert not list(broker.profile.root.iterdir())
        assert local_git(remote, "show-ref") == refs_before
        return
    prepared = await workflow.prepare_verified_git_delivery(
        application, task, publication, broker, request
    )
    awaiting = await workflow.run_verified_git_delivery(
        application, task, publication, broker, request
    )
    assert awaiting.result.state is RemoteGitDeliveryState.APPROVAL_REQUIRED
    # Pending approval deliberately retains its prepared private repository;
    # cleanup_settled describes removal, not whether run's exclusive owner exited.
    assert not awaiting.result.cleanup_settled
    assert local_git(remote, "show-ref") == refs_before
    previous_broker = broker
    broker = build_broker()
    assert broker.repository is not previous_broker.repository
    reconstructed = await broker.repository.load_prepared(request)
    assert reconstructed == prepared
    assert await broker.repository.load_result(request, awaiting.result.digest) == awaiting
    prepared = reconstructed
    assert prepared is not None
    # This is an explicit test-owned decision, not proof of a human/UI workflow.
    approval = queued_approval or approve_remote_git_delivery(
        request, prepared, approval_id="fixture-human-decision"
    )
    source_file = root / "source" / "range_ops.py"
    approved_bytes = source_file.read_bytes()
    receipts_before = await broker.repository.load_lifecycle(request)
    source_file.write_bytes(approved_bytes + b"\n# changed after approval\n")
    try:
        rejection = importlib.import_module(
            "domain.maintenance_acceptance"
        ).MaintenanceAcceptanceRejected
        with pytest.raises(rejection):
            await workflow.run_verified_git_delivery(
                application, task, publication, broker, request, approval=approval
            )
        assert await broker.repository.load_lifecycle(request) == receipts_before
        assert local_git(remote, "show-ref") == refs_before
    finally:
        source_file.write_bytes(approved_bytes)
    changed_destination = request.model_copy(
        update={
            "delivery_id": "changed-destination",
            "idempotency_key": "changed-destination",
            "repository": request.repository.model_copy(
                update={
                    "destination_ref": "refs/heads/cayu/changed-destination",
                }
            ),
        }
    )
    denied = await workflow.run_verified_git_delivery(
        application,
        task,
        publication,
        broker,
        changed_destination,
        approval=approval,
    )
    assert denied.result.state is RemoteGitDeliveryState.DENIED
    assert denied.result.next_commit is None
    assert denied.result.cleanup_settled
    assert local_git(remote, "show-ref") == refs_before
    original_push = broker._push
    push_count = 0

    async def observed_push(*args, **kwargs):
        nonlocal push_count
        push_count += 1
        result = await original_push(*args, **kwargs)
        # Observe actual acceptance before injecting acknowledgement loss.
        assert local_git(remote, "rev-parse", request.repository.destination_ref)
        if lose_push_ack:
            raise RemoteGitDeliveryError("fixture lost acknowledgement after accepted push")
        return result

    with patch.object(broker, "_push", observed_push):
        if queued_request is not None:
            # The stored operator decision, not a fixture argument, authorizes push.
            async def deliver(_app, claimed, worker_id):
                await worker.handle_git_delivery_task(
                    application, reservations, claimed, worker_id, broker
                )

            with patch.dict(os.environ, {"CAYU_MAINTENANCE_GIT_JSON": json.dumps(configured)}):
                assert (
                    await run_task_worker(
                        application.app,
                        application.app.task_store,
                        deliver,
                        worker_id="qualification-git-delivery",
                        query=TaskQuery(type="maintenance.git_delivery"),
                        max_tasks=1,
                        recover_interrupted_handoffs=False,
                    )
                    == 1
                )
                completed_delivery = await application.app.task_store.load_task(approved_task.id)
                assert completed_delivery.status is TaskStatus.COMPLETED
                assert completed_delivery.worker_id is None
                assert completed_delivery.lease_expires_at is None
                assert completed_delivery.result["request_fingerprint"] == request.fingerprint
                delivered = await broker.repository.load_result(
                    request, completed_delivery.result["result_digest"]
                )
                assert (
                    await operator_request("POST", "approval", approval_body)
                ).status_code == 202
                assert (
                    await application.app.task_store.load_task(approved_task.id)
                    == completed_delivery
                )
        else:
            delivered = await workflow.run_verified_git_delivery(
                application, task, publication, broker, request, approval=approval
            )
        replay = await workflow.run_verified_git_delivery(
            application, task, publication, broker, request, approval=approval
        )
    assert push_count == 1
    assert replay == delivered
    assert delivered.result.state is RemoteGitDeliveryState.PUSHED
    assert delivered.result.cleanup_settled
    assert (
        local_git(remote, "rev-parse", request.repository.destination_ref)
        == delivered.result.next_commit
    )
    assert local_git(remote, "rev-parse", "refs/heads/main") == SEED_BASE_REVISION
    assert "lower <= value <= upper" in local_git(
        remote, "show", f"{delivered.result.next_commit}:range_ops.py"
    )
    assert local_git(remote, "rev-list", "--count", "--all") == "2"
    if not lose_push_ack or queued_request is not None:
        if queued_request is not None:
            intake = importlib.import_module("operations.maintenance_git_intake")
            with patch.dict(os.environ, {"CAYU_MAINTENANCE_GIT_JSON": json.dumps(configured)}):
                (
                    restored_task,
                    restored_product,
                    restored_remote,
                ) = await intake.load_verified_git_result(application, reservations, identity)
                assert restored_task == task
                assert restored_product == publication and restored_remote == delivered
                # Feed reconstructed durable evidence, not fixture-held authority.
                task, publication, delivered = restored_task, restored_product, restored_remote
        await exercise_github_delivery(
            application,
            task,
            publication,
            delivered,
            root=root,
            remote=remote,
            git=local_git,
            lose_ack=lose_github_ack,
            reservations=reservations if queued_request is not None else None,
            identity=identity if queued_request is not None else None,
            git_configuration=configured,
        )
