"""Native artifact and task receipt joins; upstream Git/coding is controlled here."""

import asyncio
import json
import os

import pytest

from cayu import (
    GitHubCheckState,
    GitHubDeliveryState,
    GitHubPullRequestDeliveryRequest,
    GitHubReviewState,
    TaskQuery,
    approve_github_delivery,
)
from tests.core.test_github_delivery import FakeTransport, _connector, _pr
from tests.qualification.test_repository_maintenance_github_intake import enqueue
from tests.qualification.test_repository_maintenance_github_intake import (
    github_intake as github_intake,
)
from tests.qualification.test_repository_maintenance_intake import intake as intake


@pytest.mark.parametrize(
    "case",
    [
        "passed",
        "approved",
        "approved-pending",
        "ambiguous",
        "changes-requested",
        "truncated",
        "closed",
        "merged",
        "no-pr",
        "wrong-approval",
        "stale-receipt",
        "claimed",
        "config",
        "owner",
        "absent-result",
        "required-review",
        "required-approved",
        "feedback-truncated",
        "pending-poll",
        "approval-fingerprint",
    ],
)
def test_exact_native_delivery_readback(github_intake, tmp_path, monkeypatch, case):
    module, application, reservations, identity = github_intake

    async def scenario():
        if case in {"required-review", "required-approved"}:
            configured = json.loads(os.environ["CAYU_MAINTENANCE_GITHUB_JSON"])
            configured["reviews"] = {"approval_required": True, "required_approvers": ["reviewer"]}
            monkeypatch.setenv("CAYU_MAINTENANCE_GITHUB_JSON", json.dumps(configured))
        queued = await enqueue(github_intake)
        native = GitHubPullRequestDeliveryRequest.model_validate_json(queued.input["request_json"])
        connector, _config = _connector(tmp_path, native, FakeTransport(native))
        application.artifact_store = connector.repository.store
        try:
            approval = approve_github_delivery(
                native, approval_id="different" if case == "wrong-approval" else "github-consent"
            )
            state = {
                "approved": GitHubDeliveryState.APPROVED,
                "required-approved": GitHubDeliveryState.APPROVED,
                "approved-pending": GitHubDeliveryState.APPROVED,
                "ambiguous": GitHubDeliveryState.AMBIGUOUS,
                "changes-requested": GitHubDeliveryState.CHANGES_REQUESTED,
            }.get(case, GitHubDeliveryState.CHECKS_PASSED)
            pr = _pr(native)
            if case in {"closed", "merged"}:
                pr = pr.model_copy(update={"state": "closed", "merged": case == "merged"})
            candidate = connector._result(
                native,
                state,
                approval=approval,
                pr=None if case == "no-pr" else pr,
                checks_state=GitHubCheckState.PENDING
                if case == "approved-pending"
                else GitHubCheckState.PASSED,
                review_state=(
                    GitHubReviewState.APPROVED
                    if case in {"approved", "approved-pending", "required-approved"}
                    else GitHubReviewState.CHANGES_REQUESTED
                    if case == "changes-requested"
                    else GitHubReviewState.NONE
                ),
                checks_truncated=case == "truncated",
                feedback_truncated=case == "feedback-truncated",
            )
            if case == "pending-poll":
                candidate = candidate.model_copy(
                    update={
                        "next_poll_at": native.requested_at,
                        "next_poll_after_seconds": 1,
                    }
                )
            if case == "approval-fingerprint":
                candidate = candidate.model_copy(
                    update={"approval_fingerprint": "sha256:" + "0" * 64}
                )
            publication = await connector.repository.publish(native, candidate)
            claimed = await application.app.task_store.claim_task(
                "worker", TaskQuery(type="maintenance.github_delivery")
            )
            if case != "claimed":
                await application.app.task_store.complete_task(
                    claimed.id,
                    {
                        "request_fingerprint": native.fingerprint,
                        "result_digest": "sha256:" + "0" * 64
                        if case == "stale-receipt"
                        else publication.artifact.sha256,
                    },
                    worker_id="worker",
                    lease_expires_at=claimed.lease_expires_at,
                )
            if case == "config":
                monkeypatch.setattr(module, "_configured_github_request", lambda *args: None)
            if case == "absent-result":

                async def absent(_self, _request):
                    return None

                monkeypatch.setattr(module.GitHubDeliveryRepository, "latest", absent)
            if case == "owner":
                original = application.app.task_store.load_task

                async def still_owned(task_id):
                    task = await original(task_id)
                    return task.model_copy(update={"worker_id": "unfinished-owner"})

                monkeypatch.setattr(application.app.task_store, "load_task", still_owned)
            before = await application.app.task_store.list_tasks(TaskQuery())
            if case in {"passed", "approved", "required-approved"}:
                for _ in range(2):
                    *_, restored = await module.load_verified_github_result(
                        application, reservations, identity
                    )
                    assert restored == publication
            else:
                with pytest.raises(module.MaintenanceTaskConflict):
                    await module.load_verified_github_result(application, reservations, identity)
            assert await application.app.task_store.list_tasks(TaskQuery()) == before
        finally:
            assert await connector.aclose(timeout_s=1) is True

    asyncio.run(scenario())


def test_readback_cancellation_preserves_task_receipt(github_intake, tmp_path, monkeypatch):
    module, application, reservations, identity = github_intake

    async def scenario():
        queued = await enqueue(github_intake)
        claimed = await application.app.task_store.claim_task("worker", TaskQuery(type=queued.type))
        native = GitHubPullRequestDeliveryRequest.model_validate_json(queued.input["request_json"])
        await application.app.task_store.complete_task(
            claimed.id,
            {"request_fingerprint": native.fingerprint, "result_digest": "sha256:" + "a" * 64},
            worker_id="worker",
            lease_expires_at=claimed.lease_expires_at,
        )
        connector, _config = _connector(tmp_path, native, FakeTransport(native))
        application.artifact_store = connector.repository.store
        entered = asyncio.Event()

        async def blocked(_self, _request):
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(module.GitHubDeliveryRepository, "latest", blocked)
        before = await application.app.task_store.list_tasks(TaskQuery())
        owner = asyncio.create_task(
            module.load_verified_github_result(application, reservations, identity)
        )
        try:
            await asyncio.wait_for(entered.wait(), 5)
            owner.cancel("stop-github-readback")
            with pytest.raises(asyncio.CancelledError, match="stop-github-readback"):
                await owner
            assert owner.cancelled() and owner.cancelling() == 1
            assert await application.app.task_store.list_tasks(TaskQuery()) == before
        finally:
            if not owner.done():
                owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)
            assert await connector.aclose(timeout_s=1) is True

    asyncio.run(scenario())
