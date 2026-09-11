"""Real HTTP/store/native-artifact reads; effect production is controlled here."""

import asyncio
import importlib
import json
import warnings

import pytest

from cayu import (
    GitHubDeliveryState,
    GitHubPullRequestDeliveryRequest,
    RemoteGitDeliveryRepository,
    RemoteGitDeliveryState,
    RemoteGitLifecycleReceipt,
    TaskQuery,
    approve_github_delivery,
)
from tests.core.test_github_delivery import FakeTransport, _connector, _pr
from tests.qualification.test_repository_maintenance_git_approval import (
    approval_context as approval_context,
)
from tests.qualification.test_repository_maintenance_git_http import OP, PRODUCT
from tests.qualification.test_repository_maintenance_git_intake import git_intake as git_intake
from tests.qualification.test_repository_maintenance_github_intake import (
    github_intake as github_intake,
)
from tests.qualification.test_repository_maintenance_http import client
from tests.qualification.test_repository_maintenance_intake import intake as intake
from tests.qualification.test_repository_maintenance_operator_tasks import phases as phases


@pytest.fixture
def view(phases, approval_context, tmp_path):
    api, app, identity = phases
    _module, application, _reservations, _, native, *_ = approval_context
    task = asyncio.run(app.task_store.load_task(identity.github_delivery_task_id))
    github = GitHubPullRequestDeliveryRequest.model_validate_json(task.input["request_json"])
    connector, _configuration = _connector(tmp_path, github, FakeTransport(github))
    application.artifact_store = connector.repository.store
    try:
        yield api, application, identity, native, github, connector
    finally:
        assert asyncio.run(connector.aclose(timeout_s=1)) is True


def url(identity):
    return f"/operator/runs/{identity.public_id}/delivery?tenant=tenant-a"


def test_authorization_and_absent_native_evidence(view, monkeypatch):
    api, application, identity, _native, _github, _connector = view

    async def scenario():
        async with client(api) as http:
            assert (await http.get(url(identity), headers=PRODUCT)).status_code == 401
            assert (
                await http.get(url(identity).replace("tenant-a", "other"), headers=OP)
            ).status_code == 404
            response = await http.get(url(identity), headers=OP)
            assert response.status_code == 200
            assert (
                response.json()["git"]["evidence"]
                == response.json()["github"]["evidence"]
                == "absent"
            )
            load = application.app.task_store.load_task

            async def absent(_task_id):
                return None

            monkeypatch.setattr(application.app.task_store, "load_task", absent)
            response = await http.get(url(identity), headers=OP)
            assert (
                response.json()["git"] == response.json()["github"] == {"evidence": "not_requested"}
            )
            monkeypatch.setattr(application.app.task_store, "load_task", load)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "state",
    [
        RemoteGitDeliveryState.PREPARED,
        RemoteGitDeliveryState.CANCELLED,
        RemoteGitDeliveryState.FAILED,
    ],
)
def test_lifecycle_hash_is_not_blindly_treated_as_result(view, state):
    api, application, identity, native, _, _ = view

    async def scenario():
        repository = RemoteGitDeliveryRepository(application.artifact_store)
        await repository.append_lifecycle(
            native,
            RemoteGitLifecycleReceipt(
                delivery_id=native.delivery_id,
                request_fingerprint=native.fingerprint,
                ordinal=1,
                state=RemoteGitDeliveryState.PREPARING,
            ),
        )
        await repository.append_lifecycle(
            native,
            RemoteGitLifecycleReceipt(
                delivery_id=native.delivery_id,
                request_fingerprint=native.fingerprint,
                ordinal=2,
                prior_state=RemoteGitDeliveryState.PREPARING,
                state=state,
                evidence_sha256=None
                if state is RemoteGitDeliveryState.CANCELLED
                else "sha256:" + "0" * 64,
            ),
        )
        async with client(api) as http:
            response = await http.get(url(identity), headers=OP)
            assert response.status_code == 200
            row = response.json()["git"]
            if state is RemoteGitDeliveryState.FAILED:
                assert row == {"evidence": "unavailable"}
            else:
                assert row["evidence"] == "recorded" and row["state"] == state.value
                assert row["result_evidence"] == row["cleanup_evidence"] == "not_recorded"
            assert response.json()["github"]["evidence"] == "absent"

    asyncio.run(scenario())


def test_conflicting_preparation_never_falls_back_to_delivery(view, monkeypatch):
    api, application, identity, native, _, _ = view
    load = application.app.task_store.load_task

    async def different(task_id):
        task = await load(task_id)
        if task_id == identity.git_preparation_task_id:
            changed = native.model_copy(
                update={
                    "commit": native.commit.model_copy(update={"title": "Different exact request"})
                }
            )
            module = importlib.import_module("operations.maintenance_git_intake")
            task.input["request_json"] = module._encode(changed)
        return task

    async def scenario():
        monkeypatch.setattr(application.app.task_store, "load_task", different)
        async with client(api) as http:
            response = await http.get(url(identity), headers=OP)
            assert response.json()["git"] == {"evidence": "conflicting"}
            assert response.json()["github"]["evidence"] == "absent"

    asyncio.run(scenario())


@pytest.mark.parametrize("receipt", ["missing", "matching", "stale"])
def test_native_unknown_is_not_task_success_and_unsafe_fields_are_omitted(
    view, receipt, caplog, capsys
):
    api, application, identity, _, github, connector = view
    canary = "private-projection-canary"

    async def scenario():
        # Pure native result construction, not a provider-success simulation.
        approval = approve_github_delivery(github, approval_id="github-consent")
        candidate = connector._result(
            github,
            GitHubDeliveryState.AMBIGUOUS,
            approval=approval,
            pr=_pr(github).model_copy(update={"url": "javascript:" + canary, "body": canary}),
            reason=canary,
        )
        publication = await connector.repository.publish(github, candidate)
        if receipt != "missing":
            claimed = await application.app.task_store.claim_task(
                "fixture", TaskQuery(type="maintenance.github_delivery")
            )
            await application.app.task_store.complete_task(
                claimed.id,
                {
                    "request_fingerprint": github.fingerprint,
                    "result_digest": publication.artifact.sha256
                    if receipt == "matching"
                    else "sha256:" + "0" * 64,
                },
                worker_id="fixture",
                lease_expires_at=claimed.lease_expires_at,
            )
        before = await application.app.task_store.list_tasks(TaskQuery())
        async with client(api) as http:
            response = await http.get(url(identity), headers=OP)
            assert response.status_code == 200 and canary not in response.text
            row = response.json()["github"]
            assert row["evidence"] == "recorded" and row["state"] == "ambiguous"
            assert (
                row["task_receipt"]
                == {
                    "missing": "not_completed",
                    "matching": "matches_latest",
                    "stale": "different_result",
                }[receipt]
            )
            assert row["cleanup_evidence"] == "not_in_native_result"
            assert row["pull_request_number"] == 7
        assert await application.app.task_store.list_tasks(TaskQuery()) == before

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        asyncio.run(scenario())
    output = capsys.readouterr()
    assert all(canary not in str(item.message) for item in captured)
    assert canary not in caplog.text + output.out + output.err


def test_read_failure_preserves_sibling_and_cancellation(view, monkeypatch):
    api, application, identity, _, _, connector = view
    module = importlib.import_module("operations.maintenance_delivery_view")

    async def unavailable(*args):
        raise ConnectionError("private-store-canary")

    async def scenario():
        before = await application.app.task_store.list_tasks(TaskQuery())
        monkeypatch.setattr(module.RemoteGitDeliveryRepository, "load_lifecycle", unavailable)
        async with client(api) as http:
            response = await http.get(url(identity), headers=OP)
            assert response.json()["git"] == {"evidence": "unavailable"}
            assert response.json()["github"]["evidence"] == "absent"
            assert "private-store-canary" not in response.text
            entered = asyncio.Event()

            async def blocked(*args):
                entered.set()
                await asyncio.Event().wait()

            monkeypatch.setattr(type(connector.repository), "latest", blocked)
            owner = asyncio.create_task(http.get(url(identity), headers=OP))
            try:
                await asyncio.wait_for(entered.wait(), 5)
                owner.cancel("operator-disconnected")
                with pytest.raises(asyncio.CancelledError, match="operator-disconnected"):
                    await owner
                assert owner.cancelled() and owner.cancelling() == 1
            finally:
                if not owner.done():
                    owner.cancel()
                await asyncio.gather(owner, return_exceptions=True)
        assert await application.app.task_store.list_tasks(TaskQuery()) == before

    asyncio.run(scenario())


@pytest.mark.parametrize("malformed", ["json", "state"])
def test_corrupt_native_artifact_is_conflicting_without_diagnostic_leak(
    view, malformed, caplog, capsys
):
    api, application, identity, native, _, _ = view
    canary = "private-corrupt-artifact-canary"

    async def scenario():
        repository = RemoteGitDeliveryRepository(application.artifact_store)
        receipt = RemoteGitLifecycleReceipt(
            delivery_id=native.delivery_id,
            request_fingerprint=native.fingerprint,
            ordinal=1,
            state=RemoteGitDeliveryState.PREPARING,
            reason_code=canary,
        ).model_dump(mode="json")
        receipt["state"] = True
        raw = canary.encode() if malformed == "json" else json.dumps(receipt).encode()
        await application.artifact_store.put_bytes(
            raw,
            artifact_id=repository.lifecycle_artifact_id(native.delivery_id, 1),
            filename="remote-git-lifecycle-1.json",
            session_id=native.session_id,
            content_type="application/json",
        )
        async with client(api) as http:
            response = await http.get(url(identity), headers=OP)
            assert response.status_code == 200 and canary not in response.text
            assert response.json()["git"] == {"evidence": "conflicting"}
            assert response.json()["github"]["evidence"] == "absent"

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        asyncio.run(scenario())
    output = capsys.readouterr()
    assert all(canary not in str(item.message) for item in captured)
    assert canary not in caplog.text + output.out + output.err
