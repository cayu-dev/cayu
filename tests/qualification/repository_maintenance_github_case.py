"""Generated-consumer GitHub contract checks using local Git and controlled HTTP."""

import asyncio
import importlib
import json
import os
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from unittest.mock import patch

import httpx
import pytest

from cayu import (
    GitHubCheckPolicy,
    GitHubCheckState,
    GitHubDeliveryLimits,
    GitHubDeliveryState,
    GitHubOperation,
    GitHubPullRequestMetadata,
    GitHubRestTransport,
    GitHubReviewPolicy,
    GitHubSecurityAuthority,
    SecretRef,
    StaticVault,
    TaskQuery,
    TaskStatus,
    approve_github_delivery,
    github_connector_behavior_fingerprint,
    github_pull_request_delivery_request,
    run_task_worker,
)


async def exercise_github_delivery(
    application,
    task,
    publication,
    delivered,
    *,
    root,
    remote,
    git,
    lose_ack=False,
    reservations=None,
    identity=None,
    git_configuration=None,
):
    integration = importlib.import_module("integrations.github")
    workflow = importlib.import_module("workflows.maintenance_delivery")
    head = delivered.result.next_commit
    destination = delivered.result.next_ref
    assert head is not None and destination is not None
    token = "qualification-host-only-token"
    policy = "sha256:" + sha256(b"qualification-github-policy").hexdigest()
    request = github_pull_request_delivery_request(
        publication,
        delivered,
        connector_run_id="maintenance-github"
        if identity is None
        else identity.github_delivery_task_id,
        session_id=task.session_id,
        idempotency_key="maintenance-github"
        if identity is None
        else identity.github_delivery_task_id,
        requested_at=datetime.now(UTC).isoformat(),
        repository_alias="github",
        installation_id="fixture-installation",
        account_id="fixture-account",
        mode="create",
        existing_pull_request_number=None,
        metadata=GitHubPullRequestMetadata(title="Repair inclusive endpoint"),
        checks=GitHubCheckPolicy(required_checks=("test",)),
        reviews=GitHubReviewPolicy(),
        limits=GitHubDeliveryLimits(poll_interval_seconds=1),
        security=GitHubSecurityAuthority(
            connector_id="maintenance-app-github",
            connector_behavior_fingerprint=github_connector_behavior_fingerprint(),
            credential_profile_id="github-installation-token",
            egress_profile_id="github-api-only",
            policy_fingerprint=policy,
            approval_policy_fingerprint=policy,
            redaction_profile_fingerprint=policy,
            allowed_operations=(GitHubOperation.CREATE_PULL_REQUEST,),
        ),
    )
    pull_request = None
    calls = []
    writes = []
    check_mode = "missing"
    now = datetime.now(UTC)
    reconciliation_unavailable = False
    automatic_checks = 0
    accepted_reconciliation_reads = 0

    async def provider(http_request):
        nonlocal \
            pull_request, \
            reconciliation_unavailable, \
            automatic_checks, \
            accepted_reconciliation_reads
        assert http_request.url.host == "api.github.example"
        assert http_request.headers["authorization"] == f"Bearer {token}"
        path = http_request.url.path
        calls.append((http_request.method, path))
        if http_request.method == "GET" and "/git/ref/" in path:
            ref = "refs/" + path.split("/git/ref/", 1)[1]
            return httpx.Response(200, json={"object": {"sha": git(remote, "rev-parse", ref)}})
        if http_request.method == "POST" and path.endswith("/pulls"):
            assert pull_request is None
            writes.append("create")
            payload = json.loads(http_request.content)
            pull_request = {
                "number": 7,
                "node_id": "PR_maintenance_7",
                "html_url": "https://github.example/fixture/maintenance/pull/7",
                "state": "open",
                "draft": payload["draft"],
                "merged": False,
                "base": {
                    "ref": "main",
                    "sha": git(remote, "rev-parse", "refs/heads/main"),
                    "repo": {"full_name": "fixture/maintenance"},
                },
                "head": {
                    "ref": destination.removeprefix("refs/heads/"),
                    "sha": git(remote, "rev-parse", destination),
                    "repo": {"full_name": "fixture/maintenance"},
                },
                "title": payload["title"],
                "body": payload["body"],
                "labels": [],
            }
            if lose_ack:
                reconciliation_unavailable = True
                raise httpx.ReadError(
                    "fixture accepted create but lost response", request=http_request
                )
            return httpx.Response(201, json=pull_request)
        assert http_request.method == "GET", "Unexpected provider mutation"
        if path.endswith("/pulls"):
            if pull_request is not None:
                accepted_reconciliation_reads += 1
            if reconciliation_unavailable:
                reconciliation_unavailable = False
                return httpx.Response(503, json={"message": "fixture unavailable"})
            return httpx.Response(200, json=[] if pull_request is None else [pull_request])
        if path.endswith("/pulls/7"):
            return httpx.Response(200, json=pull_request)
        if path.endswith("/check-runs"):
            assert f"/commits/{head}/" in path
            mode = check_mode
            if mode == "advancing":
                automatic_checks += 1
                mode = "missing" if automatic_checks == 1 else "current"
            if mode == "missing":
                return httpx.Response(200, json={"total_count": 0, "check_runs": []})
            return httpx.Response(
                200,
                json={
                    "total_count": 1,
                    "check_runs": [
                        {
                            "id": 10,
                            "name": "test",
                            "head_sha": head if mode == "current" else "0" * 40,
                            "status": "completed",
                            "conclusion": "success",
                        }
                    ],
                },
            )
        if path.endswith(("/statuses", "/comments", "/reviews")):
            return httpx.Response(200, json=[])
        raise AssertionError(f"Unexpected fixed endpoint: {path}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:

        def build_connector():
            with patch.object(
                integration, "GitHubRestTransport", lambda: GitHubRestTransport(client)
            ):
                return integration.build_github_connector(
                    artifact_store=application.artifact_store,
                    repository_id="fixture",
                    installation_id="fixture-installation",
                    account_id="fixture-account",
                    owner="fixture",
                    repository_name="maintenance",
                    token_ref=SecretRef(name="fixture-github-token"),
                    secret_resolver=StaticVault({"fixture-github-token": token}),
                    api_base_url="https://api.github.example",
                )

        connector = build_connector()
        connector.clock = lambda: now
        try:
            if reservations is not None:
                assert identity is not None and git_configuration is not None
                from tests.qualification.repository_maintenance_delivery_case import (
                    build_journey_http,
                )

                configured = {
                    name: getattr(request, name)
                    for name in ("requested_at", "mode", "existing_pull_request_number")
                }
                configured.update(
                    {
                        name: getattr(request.repository, name)
                        for name in ("repository_alias", "installation_id", "account_id")
                    }
                )
                configured.update(
                    {
                        name: getattr(request, name).model_dump(mode="json")
                        for name in ("metadata", "checks", "reviews", "security", "limits")
                    }
                )
                api = build_journey_http(
                    application,
                    reservations,
                    tenant=identity.intent.tenant,
                    subject=identity.intent.subject,
                )
                with patch.dict(
                    os.environ,
                    {
                        "CAYU_MAINTENANCE_GIT_JSON": json.dumps(git_configuration),
                        "CAYU_MAINTENANCE_GITHUB_JSON": json.dumps(configured),
                        "CAYU_MAINTENANCE_GITHUB_HOST_JSON": json.dumps(
                            {
                                "owner": "fixture",
                                "repository_name": "maintenance",
                                "api_base_url": "https://api.github.example",
                            }
                        ),
                        "CAYU_MAINTENANCE_GITHUB_WEB_ORIGIN": "https://github.example",
                    },
                ):
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=api), base_url="http://fixture"
                    ) as http:
                        path = f"/operator/runs/{identity.public_id}/github/approval?tenant={identity.intent.tenant}"
                        headers = {"authorization": "Bearer fixture-operator"}
                        final_path = f"/operator/runs/{identity.public_id}/result?tenant={identity.intent.tenant}"
                        delivery_path = f"/operator/runs/{identity.public_id}/delivery?tenant={identity.intent.tenant}"
                        review = await http.get(path, headers=headers)
                        assert review.status_code == 200, review.text
                        assert review.json()["request"] == request.model_dump(mode="json")
                        body = {
                            "request_fingerprint": request.fingerprint,
                            "approval_id": "fixture-github-human",
                        }
                        queued = await http.post(path, headers=headers, json=body)
                        assert queued.status_code == 202, queued.text
                        assert not writes
                        assert (await http.get(final_path, headers=headers)).status_code == 409
                        readback = importlib.import_module("operations.maintenance_github_intake")
                        with pytest.raises(readback.MaintenanceTaskConflict):
                            await readback.load_verified_github_result(
                                application, reservations, identity
                            )
                        check_mode = "advancing"
                        loop = asyncio.get_running_loop()
                        began = loop.time()
                        connector.clock = lambda: now + timedelta(seconds=loop.time() - began)
                        worker = importlib.import_module("operations.maintenance_github")

                        async def handle(_app, claimed, worker_id):
                            await worker.handle_github_delivery_task(
                                application, reservations, claimed, worker_id, lambda: connector
                            )

                        original_publish = connector.repository.publish
                        unavailable_results = []

                        async def observe_publication(native, candidate):
                            recorded = await original_publish(native, candidate)
                            if (
                                candidate.reason_code
                                == "provider_mutation_reconciliation_unavailable"
                            ):
                                assert await connector.repository.latest(native) == recorded
                                assert candidate.state is GitHubDeliveryState.AMBIGUOUS
                                assert candidate.checks_state is not GitHubCheckState.PASSED
                                assert candidate.next_poll_at is not None
                                assert candidate.next_poll_after_seconds is not None
                                held = await application.app.task_store.load_task(
                                    identity.github_delivery_task_id
                                )
                                assert held.status is TaskStatus.CLAIMED
                                assert held.worker_id == "qualification-github"
                                assert held.lease_expires_at is not None and held.result is None
                                unavailable_results.append(recorded)
                                observation = await http.get(delivery_path, headers=headers)
                                assert observation.status_code == 200, observation.text
                                observed = observation.json()["github"]
                                assert observed["state"] == "ambiguous"
                                assert observed["task_receipt"] == "not_completed"
                                assert observed["result_digest"] == recorded.artifact.sha256
                            return recorded

                        with patch.object(connector.repository, "publish", observe_publication):
                            assert (
                                await run_task_worker(
                                    application.app,
                                    application.app.task_store,
                                    handle,
                                    worker_id="qualification-github",
                                    query=TaskQuery(type="maintenance.github_delivery"),
                                    max_tasks=1,
                                    recover_interrupted_handoffs=False,
                                )
                                == 1
                            )
                        completed = await application.app.task_store.load_task(
                            identity.github_delivery_task_id
                        )
                        assert completed.status is TaskStatus.COMPLETED
                        assert completed.worker_id is None and completed.lease_expires_at is None
                        result = await connector.repository.latest(request)
                        assert completed.result == {
                            "request_fingerprint": request.fingerprint,
                            "result_digest": result.artifact.sha256,
                        }
                        assert result.result.state is GitHubDeliveryState.CHECKS_PASSED
                        assert result.result.pull_request.head_commit == head
                        assert not result.result.pull_request.merged
                        assert automatic_checks == 2 and writes == ["create"]
                        if lose_ack:
                            assert accepted_reconciliation_reads >= 2
                            assert len(unavailable_results) == 1
                        else:
                            assert unavailable_results == []
                        assert token not in result.result.model_dump_json()
                        assert await connector.aclose(timeout_s=0) is True
                        observation = await http.get(delivery_path, headers=headers)
                        assert observation.status_code == 200, observation.text
                        observed = observation.json()
                        assert observed["git"]["state"] == "pushed"
                        assert observed["git"]["cleanup_evidence"] == "recorded_settled"
                        assert observed["git"]["task_receipts"]["delivery"] == "matches_latest"
                        assert observed["github"]["state"] == "checks_passed"
                        assert observed["github"]["task_receipt"] == "matches_latest"
                        assert observed["github"]["head_commit"] == head
                        assert observed["github"]["pull_request_number"] == 7
                        assert observed["github"]["cleanup_evidence"] == "not_in_native_result"
                        call_count = len(calls)
                        final_response = await http.get(final_path, headers=headers)
                        assert final_response.status_code == 200, final_response.text
                        final_data = final_response.json()
                        assert final_data["outcome"] == "recorded_verified_delivery"
                        assert final_data["observation"] == "durable_history"
                        assert final_data["commit"] == head
                        assert final_data["tree"] == delivered.result.tree
                        assert final_data["pull_request"] == {
                            "url": "https://github.example/fixture/maintenance/pull/7",
                            "number": 7,
                        }
                        for evidence in ("acceptance", "cost"):
                            linked = await http.get(final_data[evidence]["href"], headers=headers)
                            assert linked.status_code == 200, linked.text
                            assert {
                                key: value for key, value in linked.json().items() if key != "id"
                            } == {
                                key: value
                                for key, value in final_data[evidence].items()
                                if key not in {"href", "id"}
                            }
                        assert final_data["cost"]["billing_completeness"] == "not_established"
                        assert len(calls) == call_count and writes == ["create"]
                        for _ in range(2):
                            (
                                restored_task,
                                restored_product,
                                restored_git,
                                restored_github,
                            ) = await readback.load_verified_github_result(
                                application, reservations, identity
                            )
                            assert restored_task == task
                            assert restored_product == publication
                            assert restored_git == delivered
                            assert restored_github == result
                        assert (
                            await http.post(path, headers=headers, json=body)
                        ).status_code == 202
                        assert await application.app.task_store.load_task(completed.id) == completed
                return
            awaiting = await workflow.run_verified_github_delivery(
                application, task, publication, delivered, connector, request
            )
            assert awaiting.result.state is GitHubDeliveryState.APPROVAL_REQUIRED
            assert not writes
            approval = approve_github_delivery(request, approval_id="fixture-github-human")
            source = root / "source" / "range_ops.py"
            original = source.read_bytes()
            calls_before = list(calls)
            source.write_bytes(original + b"\n# changed after GitHub approval\n")
            try:
                rejection = importlib.import_module(
                    "domain.maintenance_acceptance"
                ).MaintenanceAcceptanceRejected
                with pytest.raises(rejection):
                    await workflow.run_verified_github_delivery(
                        application,
                        task,
                        publication,
                        delivered,
                        connector,
                        request,
                        approval=approval,
                    )
                assert calls == calls_before
            finally:
                source.write_bytes(original)
            missing = await workflow.run_verified_github_delivery(
                application, task, publication, delivered, connector, request, approval=approval
            )
            if lose_ack:
                assert missing.result.state is GitHubDeliveryState.AMBIGUOUS
                assert missing.result.checks_state is not GitHubCheckState.PASSED
                assert await connector.repository.latest(request) == missing
                assert writes == ["create"]
                assert await connector.aclose(timeout_s=5)
                old_connector = connector
                connector = build_connector()
                assert connector.repository is not old_connector.repository
                now += timedelta(seconds=60)
                connector.clock = lambda: now
                # No caller approval: reconstruct the exact durable consent.
                approval = None
                missing = await workflow.run_verified_github_delivery(
                    application, task, publication, delivered, connector, request
                )
            assert missing.result.state is GitHubDeliveryState.CHECKS_PENDING
            assert missing.result.checks_state is GitHubCheckState.MISSING
            check_mode = "stale"
            now += timedelta(seconds=60)
            stale = await workflow.run_verified_github_delivery(
                application, task, publication, delivered, connector, request, approval=approval
            )
            assert stale.result.state is not GitHubDeliveryState.CHECKS_PASSED
            assert stale.result.checks_state is not GitHubCheckState.PASSED
            check_mode = "advancing"
            now += timedelta(seconds=60)
            loop = asyncio.get_running_loop()
            started = loop.time()

            def clock():
                return now + timedelta(seconds=loop.time() - started)

            connector.clock = clock
            observer = importlib.import_module("operations.maintenance_github")
            result = await observer.observe_github_delivery(
                application, task, publication, delivered, connector, request, approval=approval
            )
            assert automatic_checks == 2
            assert await connector.aclose(timeout_s=0) is True
            assert result.result.state is GitHubDeliveryState.CHECKS_PASSED
            assert result.result.checks_state is GitHubCheckState.PASSED
            assert result.result.pull_request.head_commit == head
            assert not result.result.pull_request.merged
            # The phase owns sealing; exact terminal replay uses a fresh instance
            # only after positive local settlement, never after a timed-out close.
            connector = build_connector()
            connector.clock = clock
            replay = await workflow.run_verified_github_delivery(
                application, task, publication, delivered, connector, request, approval=approval
            )
            assert replay == result
            assert writes == ["create"]
            assert token not in result.result.model_dump_json()
        finally:
            assert await connector.aclose(timeout_s=5)
