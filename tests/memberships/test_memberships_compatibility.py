"""A project that moves from its generated policy to `cayu.memberships` keeps its queued work.

`tests/fixtures/memberships/` holds the `policies/access.py` and `database/memberships.py`
every Cayu release from 0.16.1 to 0.20.0 generated, byte for byte. Cayu stores the authority
and grant with each piece of work it admits, so the library policy must keep both.
"""

from __future__ import annotations

import asyncio
import hashlib
import sys
import types
from pathlib import Path

import pytest

from cayu import CayuApp
from cayu.memberships import (
    InMemoryMembershipDirectory,
    Membership,
    MembershipAccessPolicy,
    conversation_labels,
)
from cayu.resource_access import ResourceAccessPolicy, encode_grant
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks.creation import TaskCreate
from cayu.tasks.worker import run_task_worker

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "memberships"
# The released generated files' SHA-256, as `cayu check` knows them.
RELEASED = {
    "policies_access.py.fixture": "40c51983cd0ba925de7769d3c3a60f4362cbffd4839449ab013cf6489ffcbcb0",
    "database_memberships.py.fixture": (
        "a4e39fd2440c9c1f6cdfc053797c1a12a8d6252db0cd669bdeb9b98cd7f863e9"
    ),
}
ALICE = Membership("acme", "acme/finance")
CASES = {
    "nobody": [],
    "organization only": [Membership("acme")],
    "one department": [ALICE],
    "several": [
        Membership("globex", "globex/support"),
        ALICE,
        Membership("initech"),
        Membership("acme", "acme/legal"),
    ],
    "the most": [Membership(f"org{index:0>59}", f"org{index:0>59}/d") for index in range(16)],
}


def _generated_policy_class() -> type[ResourceAccessPolicy]:
    """The generated `MembershipAccessPolicy`, loaded from the released file as it is."""

    source = (FIXTURES / "policies_access.py.fixture").read_text(encoding="utf-8")
    # The project modules it imports; the policy reads memberships only from the directory
    # it is given.
    stand_ins = {
        "database": types.ModuleType("database"),
        "database.memberships": types.ModuleType("database.memberships"),
        "domain": types.ModuleType("domain"),
        "domain.organizations": types.ModuleType("domain.organizations"),
    }
    stand_ins["database.memberships"].DatabaseMembershipDirectory = object  # type: ignore[attr-defined]
    stand_ins["domain.organizations"].MembershipDirectory = object  # type: ignore[attr-defined]
    saved = {name: sys.modules.get(name) for name in stand_ins}
    sys.modules.update(stand_ins)
    try:
        module = types.ModuleType("generated_policies_access")
        exec(compile(source, "policies/access.py", "exec"), module.__dict__)
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return module.MembershipAccessPolicy


@pytest.mark.parametrize("name", sorted(RELEASED))
def test_the_fixtures_are_the_released_files(name: str) -> None:
    content = (FIXTURES / name).read_bytes()
    assert hashlib.sha256(content).hexdigest() == RELEASED[name]


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_library_decides_as_the_generated_policy_did(case: str) -> None:
    generated = _generated_policy_class()

    async def run() -> None:
        directory = InMemoryMembershipDirectory({"alice": CASES[case]})
        directory.set_memberships("bob", [])
        old = await generated(directory).resolve("alice")
        new = await MembershipAccessPolicy(directory).resolve("alice")
        assert generated(directory).authority == MembershipAccessPolicy(directory).authority
        assert encode_grant(new.grant) == encode_grant(old.grant)
        assert new.revision == old.revision == 1
        if old.valid_until is None:
            assert new.valid_until is None
        else:
            assert new.valid_until is not None
            assert abs((new.valid_until - old.valid_until).total_seconds()) < 5

    asyncio.run(run())


def test_a_task_admitted_under_the_generated_policy_runs_under_the_library_policy(
    tmp_path: Path,
) -> None:
    directory = InMemoryMembershipDirectory({"alice": [ALICE]})
    path = tmp_path / "tasks.db"

    async def queue(policy: ResourceAccessPolicy) -> str:
        store = SQLiteTaskStore(path)
        try:
            app = CayuApp(task_store=store, resource_access_policy=policy)
            access = await app.access("alice")
            labels = conversation_labels(subject="alice", membership=ALICE, sharing="department")
            task = await access.tasks.create(
                TaskCreate(type="report", title="Quarterly report"), labels=labels
            )
            return task.id
        finally:
            await store.close()

    async def work(policy: ResourceAccessPolicy) -> list[str]:
        store = SQLiteTaskStore(path)
        try:
            app = CayuApp(task_store=store, resource_access_policy=policy)
            handled: list[str] = []

            async def handler(app, task, worker_id) -> None:
                handled.append(task.id)

            await run_task_worker(app, store, handler, worker_id="worker", max_tasks=1)
            return handled
        finally:
            await store.close()

    # Queued before the project moved, with the policy it generated.
    task_id = asyncio.run(queue(_generated_policy_class()(directory)))
    # Run after it moved to the library.
    handled = asyncio.run(work(MembershipAccessPolicy(directory)))
    # The worker admitted it and handed it to the handler; refused work never gets there
    # (the next test).
    assert handled == [task_id]


def test_a_task_admitted_under_another_authority_is_refused(tmp_path: Path) -> None:
    """The control for the test above: the authority is what keeps queued work admitted."""

    directory = InMemoryMembershipDirectory({"alice": [ALICE]})
    path = tmp_path / "tasks.db"

    async def run() -> None:
        store = SQLiteTaskStore(path)
        try:
            app = CayuApp(
                task_store=store, resource_access_policy=MembershipAccessPolicy(directory)
            )
            access = await app.access("alice")
            labels = conversation_labels(subject="alice", membership=ALICE, sharing="department")
            task = await access.tasks.create(
                TaskCreate(type="report", title="Quarterly report"), labels=labels
            )
            renamed = CayuApp(
                task_store=store,
                resource_access_policy=MembershipAccessPolicy(directory, authority="renamed-v2"),
            )
            handled: list[str] = []

            async def handler(app, task, worker_id) -> None:
                handled.append(task.id)

            await run_task_worker(renamed, store, handler, worker_id="worker", max_tasks=1)
            assert handled == []
            assert (await store.load_task(task.id)).status == "failed"
        finally:
            await store.close()

    asyncio.run(run())
