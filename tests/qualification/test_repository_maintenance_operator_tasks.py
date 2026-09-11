"""Operator task readback through generated HTTP and native Memory/SQLite stores."""

import asyncio
import warnings
from datetime import datetime
from uuid import uuid4

import pytest

from cayu import TaskQuery, TaskStatus
from tests.qualification.test_repository_maintenance_git_approval import (
    approval_context as approval_context,
)
from tests.qualification.test_repository_maintenance_git_http import OP, PRODUCT, server
from tests.qualification.test_repository_maintenance_git_intake import git_intake as git_intake
from tests.qualification.test_repository_maintenance_github_intake import (
    enqueue,
)
from tests.qualification.test_repository_maintenance_github_intake import (
    github_intake as github_intake,
)
from tests.qualification.test_repository_maintenance_http import client
from tests.qualification.test_repository_maintenance_intake import intake as intake


@pytest.fixture
def phases(intake, github_intake, approval_context):
    coding, _domain, reservations, app, identity = intake
    git, application, _, _, _native, _pending, _receipts, options = approval_context

    async def setup():
        await coding.ensure_coding_task(app, reservations, identity)
        await git.ensure_git_delivery_task(application, reservations, identity, **options)
        await enqueue(github_intake)

    asyncio.run(setup())
    return server(application, reservations), app, identity


def path(identity):
    return f"/operator/runs/{identity.public_id}/tasks?tenant=tenant-a"


def test_four_phases_are_independent_durable_observations(phases):
    api, app, identity = phases

    async def scenario():
        worker = f"maintenance.git_delivery-{uuid4().hex}"
        claimed = await app.task_store.claim_task(
            worker, TaskQuery(type="maintenance.git_delivery")
        )
        before = await app.task_store.list_tasks(TaskQuery())
        async with client(api) as http:
            response = await http.get(path(identity), headers=OP)
            assert response.status_code == 200
            data = response.json()
            assert data["effect_evidence"] == data["cleanup_evidence"] == "not_inspected"
            rows = data["phases"]
            assert list(rows) == ["coding", "git_preparation", "git_delivery", "github_delivery"]
            assert all(row["evidence"] == "recorded" for row in rows.values())
            assert (
                rows["coding"]["task_status"] == rows["github_delivery"]["task_status"] == "pending"
            )
            assert rows["git_preparation"]["task_status"] == "completed"
            delivery = rows["git_delivery"]
            assert delivery["task_status"] == "claimed"
            assert delivery["recorded_owner"] == {"kind": "registered_role", "id": worker}
            assert delivery["recorded_lease_expires_at"] == claimed.lease_expires_at.isoformat()
            assert delivery["lease_observation"] == "not_expired"
            assert delivery["cancellation_marker"] == "not_recorded"
            assert datetime.fromisoformat(delivery["observed_at"]) >= claimed.updated_at
            assert rows["coding"]["recorded_owner"] == {"kind": "not_recorded"}
            assert "request_json" not in response.text and "approval_json" not in response.text
        assert await app.task_store.list_tasks(TaskQuery()) == before

    asyncio.run(scenario())


def test_auth_tenant_and_uuid_validation_precede_phase_reads(phases, monkeypatch):
    api, app, identity = phases

    async def forbidden(*args, **kwargs):
        pytest.fail("Unauthorized observation reached task storage")

    monkeypatch.setattr(app.task_store, "load_task", forbidden)

    async def scenario():
        async with client(api) as http:
            for headers, suffix, expected in (
                ({}, "?tenant=tenant-a", 401),
                (PRODUCT, "?tenant=tenant-a", 401),
                (OP, "", 422),
                (OP, "?tenant=tenant-a&tenant=other", 422),
                (OP, "?tenant=other", 404),
            ):
                response = await http.get(
                    f"/operator/runs/{identity.public_id}/tasks{suffix}", headers=headers
                )
                assert response.status_code == expected
            assert (
                await http.get("/operator/runs/invalid/tasks?tenant=tenant-a", headers=OP)
            ).status_code == 404

    asyncio.run(scenario())


@pytest.mark.parametrize("missing", ["all", "git_delivery"])
def test_absent_is_not_pending_and_does_not_recreate(phases, monkeypatch, missing):
    api, app, identity = phases
    load = app.task_store.load_task

    async def absent(task_id):
        if missing == "all" or task_id == identity.git_delivery_task_id:
            return None
        return await load(task_id)

    async def scenario():
        before = await app.task_store.list_tasks(TaskQuery())
        monkeypatch.setattr(app.task_store, "load_task", absent)
        async with client(api) as http:
            rows = (await http.get(path(identity), headers=OP)).json()["phases"]
            for phase, row in rows.items():
                expected = "absent" if missing in {"all", phase} else "recorded"
                assert row["evidence"] == expected
                if expected == "absent":
                    assert set(row) == {"task_id", "evidence"}
        assert await app.task_store.list_tasks(TaskQuery()) == before

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "failure", ["unavailable", "identity", "status", "boolean", "timestamp", "reason", "approval"]
)
def test_bad_phase_does_not_erase_siblings_or_echo_diagnostics(
    phases, monkeypatch, caplog, capsys, failure
):
    api, app, identity = phases
    load = app.task_store.load_task
    canary = "private-task-observation-canary"

    class Hostile:
        def __repr__(self):
            return canary

        def __str__(self):
            return canary

    async def broken(task_id):
        task = await load(task_id)
        if task_id != identity.git_delivery_task_id:
            return task
        if failure == "unavailable":
            raise ConnectionError(canary)
        changed = task.model_copy(deep=True)
        changed.error = {"secret": canary}
        changed.result = {"secret": canary}
        changed.status_payload = {"secret": canary}
        if failure == "identity":
            changed.input["maintenance_run_id"] = Hostile()
        elif failure == "status":
            changed.status = Hostile()
        elif failure == "boolean":
            changed.status = True
        elif failure == "timestamp":
            changed.lease_expires_at = Hostile()
        elif failure == "reason":
            changed.status_reason = Hostile()
        else:
            changed.input["approval_json"] = '{"secret":"' + canary + '"}'
        return changed

    async def scenario():
        before = await app.task_store.list_tasks(TaskQuery())
        monkeypatch.setattr(app.task_store, "load_task", broken)
        async with client(api) as http:
            response = await http.get(path(identity), headers=OP)
            assert response.status_code == 200 and canary not in response.text
            rows = response.json()["phases"]
            assert rows["coding"]["evidence"] == rows["github_delivery"]["evidence"] == "recorded"
            row = rows["git_delivery"]
            assert row == {
                "task_id": identity.git_delivery_task_id,
                "evidence": "unavailable" if failure == "unavailable" else "conflicting",
            }
            monkeypatch.setattr(app.task_store, "load_task", load)
            retry = (await http.get(path(identity), headers=OP)).json()
            assert all(row["evidence"] == "recorded" for row in retry["phases"].values())
        assert await app.task_store.list_tasks(TaskQuery()) == before

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        asyncio.run(scenario())
    assert all(canary not in str(item.message) for item in captured)
    output = capsys.readouterr()
    assert canary not in caplog.text + output.out + output.err


def test_expired_cancelled_owner_is_still_fenced(phases):
    api, app, identity = phases

    async def scenario():
        query = TaskQuery(type="maintenance.git_delivery")
        await app.task_store.claim_task("private-owner", query, lease_seconds=1)
        fenced = await app.task_store.cancel_task(identity.git_delivery_task_id)
        assert fenced.status_reason == "cancellation_requested"
        await asyncio.sleep(1.05)
        async with client(api) as http:
            response = await http.get(path(identity), headers=OP)
            row = response.json()["phases"]["git_delivery"]
            assert row["evidence"] == "recorded"
            assert row["recorded_owner"] == {"kind": "present_unprojected"}
            assert row["lease_observation"] == "expired"
            assert row["cancellation_marker"] == "requested"
            assert "private-owner" not in response.text
        assert await app.task_store.load_task(fenced.id) == fenced
        assert await app.task_store.claim_task("replacement", query) is None

    asyncio.run(scenario())


def test_request_cancellation_during_middle_read_preserves_signal(phases, monkeypatch):
    api, app, identity = phases
    load = app.task_store.load_task

    async def scenario():
        before = await app.task_store.list_tasks(TaskQuery())
        entered = asyncio.Event()
        seen = []

        async def blocked(task_id):
            seen.append(task_id)
            if task_id == identity.git_delivery_task_id:
                entered.set()
                await asyncio.Event().wait()
            return await load(task_id)

        monkeypatch.setattr(app.task_store, "load_task", blocked)
        async with client(api) as http:
            owner = asyncio.create_task(http.get(path(identity), headers=OP))
            try:
                await asyncio.wait_for(entered.wait(), 5)
                owner.cancel("operator-left")
                with pytest.raises(asyncio.CancelledError, match="operator-left"):
                    await owner
                assert owner.cancelled() and owner.cancelling() == 1
                assert seen == [
                    identity.task_id,
                    identity.git_preparation_task_id,
                    identity.git_delivery_task_id,
                ]
            finally:
                if not owner.done():
                    owner.cancel()
                await asyncio.gather(owner, return_exceptions=True)
        assert await app.task_store.list_tasks(TaskQuery()) == before

    asyncio.run(scenario())


def test_all_task_states_remain_distinct_from_business_outcome(phases, monkeypatch):
    api, app, identity = phases
    load = app.task_store.load_task

    async def scenario():
        task = await load(identity.task_id)
        async with client(api) as http:
            for status in TaskStatus:

                async def stored(task_id, observed_status=status):
                    if task_id == identity.task_id:
                        return task.model_copy(
                            update={"status": observed_status, "status_reason": "private-reason"}
                        )
                    return await load(task_id)

                monkeypatch.setattr(app.task_store, "load_task", stored)
                response = await http.get(path(identity), headers=OP)
                data = response.json()
                assert data["phases"]["coding"]["task_status"] == status.value
                assert data["phases"]["coding"]["cancellation_marker"] == "unrecognized"
                assert data["effect_evidence"] == data["cleanup_evidence"] == "not_inspected"
                assert "private-reason" not in response.text

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["coding", "git_preparation", "git_delivery", "github_delivery"])
def test_conflicting_identity_never_hides_valid_phases(phases, monkeypatch, phase):
    api, app, identity = phases
    load = app.task_store.load_task
    selected = identity.task_id if phase == "coding" else getattr(identity, f"{phase}_task_id")

    async def conflicting(task_id):
        task = await load(task_id)
        if task_id == selected:
            task.input["maintenance_run_id"] = "different-run"
        return task

    async def scenario():
        before = await app.task_store.list_tasks(TaskQuery())
        monkeypatch.setattr(app.task_store, "load_task", conflicting)
        async with client(api) as http:
            response = await http.get(path(identity), headers=OP)
            assert response.status_code == 200
            for name, row in response.json()["phases"].items():
                assert row["evidence"] == ("conflicting" if name == phase else "recorded")
        assert await app.task_store.list_tasks(TaskQuery()) == before

    asyncio.run(scenario())


def test_reserved_but_never_enqueued_tasks_are_absent(github_intake):
    _module, application, reservations, identity = github_intake
    api = server(application, reservations)

    async def scenario():
        async with client(api) as http:
            response = await http.get(path(identity), headers=OP)
            assert response.status_code == 200
            assert all(row["evidence"] == "absent" for row in response.json()["phases"].values())
        assert await application.app.task_store.list_tasks(TaskQuery()) == []

    asyncio.run(scenario())
