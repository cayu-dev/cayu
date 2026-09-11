"""GitHub consent queue with real Runtime stores; upstream publication is controlled."""

import asyncio
import importlib
import json
import traceback
import warnings
from datetime import datetime
from types import SimpleNamespace

import pytest

from cayu import (
    GitHubPullRequestDeliveryRequest,
    GitHubRepositoryAuthority,
    GitHubSourceAuthority,
    TaskQuery,
)
from tests.qualification.test_repository_maintenance_intake import intake as intake


def configuration():
    fingerprint = "sha256:" + "a" * 64
    return {
        "requested_at": "2026-09-10T00:00:00+00:00",
        "repository_alias": "github",
        "installation_id": "installation",
        "account_id": "account",
        "mode": "create",
        "existing_pull_request_number": None,
        "metadata": {"title": "Repair endpoint", "draft": True},
        "checks": {"required_checks": ["test"]},
        "reviews": {},
        "limits": {},
        "security": {
            "connector_id": "maintenance-app-github",
            "connector_behavior_fingerprint": fingerprint,
            "credential_profile_id": "github-installation-token",
            "egress_profile_id": "github-api-only",
            "policy_fingerprint": fingerprint,
            "approval_policy_fingerprint": fingerprint,
            "redaction_profile_fingerprint": fingerprint,
            "allowed_operations": ["create_pull_request"],
        },
    }


@pytest.fixture
def github_intake(intake, monkeypatch):
    _coding, _domain, reservations, app, identity = intake
    module = importlib.import_module("operations.maintenance_github_intake")
    application = SimpleNamespace(app=app)
    monkeypatch.setenv("CAYU_MAINTENANCE_GITHUB_JSON", json.dumps(configuration()))

    async def verified(given, registry, expected):
        assert given is application and registry is reservations and expected == identity
        return None, object(), object()

    def request(_product, _remote, **kwargs):
        repository = GitHubRepositoryAuthority(
            repository_id="target",
            repository_alias=kwargs.pop("repository_alias"),
            installation_id=kwargs.pop("installation_id"),
            account_id=kwargs.pop("account_id"),
            base_ref="refs/heads/main",
            expected_base_commit="1" * 40,
            head_ref="refs/heads/cayu/fix",
            head_commit="2" * 40,
        )
        source = GitHubSourceAuthority(
            product_result_artifact_id="product",
            product_result_sha256="sha256:" + "a" * 64,
            product_request_fingerprint="sha256:" + "b" * 64,
            product_run_id=identity.product_run_id,
            source_workspace_id="workspace",
            source_revision="sha256:" + "c" * 64,
            diff_artifact_id="diff",
            diff_sha256="sha256:" + "d" * 64,
            remote_result_artifact_id="remote",
            remote_result_sha256="sha256:" + "e" * 64,
            remote_request_fingerprint="sha256:" + "f" * 64,
            remote_delivery_id=identity.git_delivery_task_id,
        )
        return GitHubPullRequestDeliveryRequest(source=source, repository=repository, **kwargs)

    monkeypatch.setattr(module, "load_verified_git_result", verified)
    monkeypatch.setattr(module, "github_pull_request_delivery_request", request)
    return module, application, reservations, identity


async def enqueue(context, **changes):
    module, application, reservations, identity = context
    native = await module.load_github_approval_request(application, reservations, identity)
    options = dict(
        actor_subject="operator",
        expected_request_fingerprint=native.fingerprint,
        approval_id="github-consent",
    )
    options.update(changes)
    return await module.ensure_github_delivery_task(application, reservations, identity, **options)


def test_approval_queue_and_json_claim_reconstruction(github_intake):
    module, application, reservations, identity = github_intake

    async def scenario():
        queued = await enqueue(github_intake)
        assert await enqueue(github_intake) == queued
        assert queued.id == identity.github_delivery_task_id
        assert queued.invocation.origin.subject == "operator"
        claimed = await application.app.task_store.claim_task(
            "worker", TaskQuery(type="maintenance.github_delivery")
        )
        restored, native, approval = await module.load_claimed_github_delivery(
            application.app, reservations, claimed, "worker"
        )
        assert restored == identity
        assert datetime.fromisoformat(native.requested_at) == datetime.fromisoformat(
            configuration()["requested_at"]
        )
        assert native.checks.required_checks == ("test",)
        assert approval.approved_operations == native.security.allowed_operations
        assert approval.approval_id == "github-consent"
        assert approval.request_fingerprint == native.fingerprint
        conflict = importlib.import_module("operations.maintenance_intake").MaintenanceTaskConflict
        with pytest.raises(conflict):
            await module.load_claimed_github_delivery(
                application.app, reservations, claimed, "wrong-worker"
            )
        altered = claimed.model_copy(deep=True)
        altered.input["approval_json"] = altered.input["approval_json"].replace(
            "github-consent", "different"
        )
        with pytest.raises(conflict):
            await module.load_claimed_github_delivery(
                application.app, reservations, altered, "worker"
            )
        assert await application.app.task_store.load_task(claimed.id) == claimed

    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["timestamp", "title", "checks", "actor", "approval"])
def test_changed_authority_cannot_replace_reserved_task(github_intake, monkeypatch, change):
    _module, application, _reservations, _identity = github_intake
    conflict = importlib.import_module("operations.maintenance_intake").MaintenanceTaskConflict

    async def scenario():
        queued = await enqueue(github_intake)
        config = configuration()
        if change == "timestamp":
            config["requested_at"] = "2026-09-11T00:00:00+00:00"
        elif change == "title":
            config["metadata"]["title"] = "Different title"
        elif change == "checks":
            config["checks"]["required_checks"] = ["other"]
        monkeypatch.setenv("CAYU_MAINTENANCE_GITHUB_JSON", json.dumps(config))
        changes = (
            {"actor_subject": "different"}
            if change == "actor"
            else {"approval_id": "different"}
            if change == "approval"
            else {}
        )
        with pytest.raises(conflict):
            await enqueue(github_intake, **changes)
        assert await application.app.task_store.load_task(queued.id) == queued

    asyncio.run(scenario())


def test_committed_enqueue_ack_loss_reconciles_same_task(github_intake, monkeypatch):
    _module, application, _reservations, identity = github_intake
    create = application.app.create_task
    calls = 0

    async def lose(request):
        nonlocal calls
        calls += 1
        await create(request)
        raise ConnectionError("ack lost")

    monkeypatch.setattr(application.app, "create_task", lose)

    async def scenario():
        with pytest.raises(ConnectionError):
            await enqueue(github_intake)
        stored = await application.app.task_store.load_task(identity.github_delivery_task_id)
        assert await enqueue(github_intake) == stored and calls == 1

    asyncio.run(scenario())


def test_cancel_verification_does_not_enqueue(github_intake, monkeypatch):
    module, application, _reservations, identity = github_intake

    async def scenario():
        entered = asyncio.Event()

        async def wait(*args):
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(module, "load_verified_git_result", wait)
        owner = asyncio.create_task(enqueue(github_intake))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            owner.cancel("stop-pr-intake")
            with pytest.raises(asyncio.CancelledError, match="stop-pr-intake"):
                await owner
            assert owner.cancelled() and owner.cancelling() == 1
            assert (
                await application.app.task_store.load_task(identity.github_delivery_task_id) is None
            )
        finally:
            if not owner.done():
                owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("bad", ["operation", "boolean", "duplicate", "extra", "oversize"])
def test_invalid_configuration_is_sanitized_and_never_enqueued(
    github_intake, monkeypatch, caplog, capsys, bad
):
    _module, application, _reservations, identity = github_intake
    conflict = importlib.import_module("operations.maintenance_intake").MaintenanceTaskConflict
    canary = "private-github-config-canary"
    config = configuration()
    config["metadata"]["body"] = canary
    if bad == "operation":
        config["security"]["allowed_operations"] = ["merge"]
    elif bad == "boolean":
        config["limits"]["max_polls"] = True
    elif bad == "extra":
        config["token"] = canary
    raw = json.dumps(config)
    if bad == "duplicate":
        raw = '{"metadata":"' + canary + '","metadata":{}}'
    elif bad == "oversize":
        raw = canary * 4000
    monkeypatch.setenv("CAYU_MAINTENANCE_GITHUB_JSON", raw)

    async def scenario():
        with pytest.raises(conflict) as caught:
            await enqueue(github_intake)
        assert canary not in "".join(traceback.format_exception(caught.value))
        assert await application.app.task_store.load_task(identity.github_delivery_task_id) is None

    with warnings.catch_warnings(record=True) as captured:
        asyncio.run(scenario())
    output = capsys.readouterr()
    assert not captured and canary not in caplog.text + output.out + output.err


@pytest.mark.parametrize("bad", ["duplicate", "boolean", "operation"])
def test_corrupt_saved_approval_cannot_authorize_claim(github_intake, monkeypatch, bad):
    module, application, reservations, _identity = github_intake
    conflict = importlib.import_module("operations.maintenance_intake").MaintenanceTaskConflict

    async def scenario():
        await enqueue(github_intake)
        claimed = await application.app.task_store.claim_task(
            "worker", TaskQuery(type="maintenance.github_delivery")
        )
        corrupt = claimed.model_copy(deep=True)
        if bad == "duplicate":
            corrupt.input["approval_json"] = '{"approval_id":"a","approval_id":"b"}'
        elif bad == "boolean":
            corrupt.input["approval_json"] = True
        else:
            raw = json.loads(corrupt.input["approval_json"])
            raw["approved_operations"] = ["merge"]
            corrupt.input["approval_json"] = json.dumps(raw)

        async def stored(_task_id):
            return corrupt

        monkeypatch.setattr(application.app.task_store, "load_task", stored)
        with pytest.raises(conflict):
            await module.load_claimed_github_delivery(
                application.app, reservations, corrupt, "worker"
            )

    asyncio.run(scenario())
