"""A project that moves from its generated policy to `cayu.memberships` keeps its queued work.

`tests/fixtures/memberships/` holds the `policies/access.py` and `database/memberships.py`
every Cayu release from 0.16.1 to 0.20.0 generated, byte for byte. Cayu stores the authority
with each piece of work it admits and checks the work against both the grant it was admitted
with and the current one, so the library keeps the authority and admits the owner's work.
Those releases also shared conversations with departments, which the library has no more:
such a conversation narrows to its owner, and nobody sees more than they did before.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import sys
import types
from pathlib import Path

import pytest

from cayu import CayuApp
from cayu.memberships import (
    InMemoryMembershipDirectory,
    Membership,
    MembershipAccessPolicy,
)
from cayu.resource_access import ResourceAccessPolicy
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
# What the released policy's directory reported: an organization and at most one of its
# departments. The library's memberships are the organizations alone.
OLD_MEMBERSHIPS = {
    "alice": [("acme", "acme/finance")],
    "carol": [("acme", "acme/finance")],
    "bob": [("acme", "acme/legal")],
    "erin": [("acme", None), ("globex", "globex/support")],
    "dave": [("globex", "globex/support")],
}
# Conversations as the released projects labelled them, by each sharing they offered.
SHARED = (
    [
        {"organization": organization, "owner": owner, "sharing": "personal"}
        for organization, owner in itertools.product(("acme", "globex"), OLD_MEMBERSHIPS)
    ]
    + [
        {"organization": organization, "owner": owner, "sharing": "organization"}
        for organization, owner in itertools.product(("acme", "globex"), OLD_MEMBERSHIPS)
    ]
    + [
        {
            "organization": organization,
            "department": department,
            "owner": owner,
            "sharing": "department",
        }
        for (organization, department), owner in itertools.product(
            (("acme", "acme/finance"), ("acme", "acme/legal"), ("globex", "globex/support")),
            OLD_MEMBERSHIPS,
        )
    ]
)
DEPARTMENT_LABELS = {
    "organization": "acme",
    "department": "acme/finance",
    "owner": "alice",
    "sharing": "department",
}


class _OldMembership:
    """A membership as the released policy read it: `organization` and `department`."""

    def __init__(self, organization: str, department: str | None) -> None:
        self.organization = organization
        self.department = department


class _OldDirectory:
    def __init__(self, memberships: dict[str, list[tuple[str, str | None]]]) -> None:
        self._memberships = memberships

    async def current(self, subject: str):
        return types.SimpleNamespace(
            memberships=tuple(_OldMembership(*item) for item in self._memberships.get(subject, [])),
            # The revision `_new_directory` reaches: one change per person.
            revision=len(OLD_MEMBERSHIPS),
        )


def _new_directory() -> InMemoryMembershipDirectory:
    """The same people in the library's terms, at the same revision: the project's tables
    didn't change when it moved."""

    directory = InMemoryMembershipDirectory()
    for subject, items in OLD_MEMBERSHIPS.items():
        directory.set_memberships(subject, [Membership(organization) for organization, _ in items])
    return directory


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


@pytest.mark.parametrize("subject", sorted(OLD_MEMBERSHIPS))
def test_nobody_sees_or_continues_more_than_under_the_generated_policy(subject: str) -> None:
    generated = _generated_policy_class()

    async def run() -> None:
        old = (await generated(_OldDirectory(OLD_MEMBERSHIPS)).resolve(subject)).grant.sessions
        new = (await MembershipAccessPolicy(_new_directory()).resolve(subject)).grant.sessions
        assert (
            generated(_OldDirectory({})).authority
            == MembershipAccessPolicy(_new_directory()).authority
        )
        for labels in SHARED:
            for action in ("read", "execute", "modify", "delete", "update_labels"):
                before, after = old.matches(action, labels), new.matches(action, labels)
                if labels["sharing"] != "department":
                    # Personal and organization conversations are decided as before.
                    assert after == before, (action, labels)
                elif labels["owner"] != subject:
                    # A department's conversation is nobody's but its owner's now.
                    assert not after, (action, labels)
                else:
                    # Its owner keeps it, in the organization it was started in.
                    assert after == (
                        labels["organization"] in {o for o, _ in OLD_MEMBERSHIPS[subject]}
                    )

    asyncio.run(run())


def test_a_task_admitted_under_the_generated_policy_runs_under_the_library_policy(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tasks.db"

    async def queue(policy: ResourceAccessPolicy) -> str:
        store = SQLiteTaskStore(path)
        try:
            app = CayuApp(task_store=store, resource_access_policy=policy)
            access = await app.access("alice")
            # Shared with her department, which the released projects offered.
            task = await access.tasks.create(
                TaskCreate(type="report", title="Quarterly report"), labels=DEPARTMENT_LABELS
            )
            return task.id
        finally:
            await store.close()

    async def work(policy: ResourceAccessPolicy) -> tuple[list[str], list[str]]:
        store = SQLiteTaskStore(path)
        try:
            app = CayuApp(task_store=store, resource_access_policy=policy)
            handled: list[str] = []

            async def handler(app, task, worker_id) -> None:
                handled.append(task.id)
                # Work it makes takes its labels and is still alice's to make.
                await store.create_task(TaskCreate(type="report-part", title="Appendix"))

            await run_task_worker(app, store, handler, worker_id="worker", max_tasks=1)
            listed = await store.list_tasks()
            return handled, [item.status for item in listed if item.type == "report-part"]
        finally:
            await store.close()

    # Queued before the project moved, with the policy it generated.
    task_id = asyncio.run(queue(_generated_policy_class()(_OldDirectory(OLD_MEMBERSHIPS))))
    # Run after it moved to the library: still alice's, so still admitted.
    handled, children = asyncio.run(work(MembershipAccessPolicy(_new_directory())))
    # The worker admitted it and handed it to the handler; refused work never gets there
    # (the next test).
    assert handled == [task_id]
    # Its handler made its child task rather than being refused.
    assert children == ["pending"]


def test_the_owner_still_makes_outputs_in_a_department_conversation_but_no_new_one() -> None:
    async def run() -> None:
        grant = (await MembershipAccessPolicy(_new_directory()).resolve("alice")).grant
        for family in (grant.tasks, grant.artifacts):
            assert family is not None and family.matches("create", DEPARTMENT_LABELS)
        # Not a new conversation shared that way, nor outputs for someone else.
        assert grant.sessions is not None
        assert not grant.sessions.matches("create", DEPARTMENT_LABELS)
        carol = (await MembershipAccessPolicy(_new_directory()).resolve("carol")).grant
        assert carol.tasks is not None and not carol.tasks.matches("create", DEPARTMENT_LABELS)

    asyncio.run(run())


def test_a_task_admitted_under_another_authority_is_refused(tmp_path: Path) -> None:
    """The control for the test above: the authority is what keeps queued work admitted."""

    directory = _new_directory()
    path = tmp_path / "tasks.db"

    async def run() -> None:
        store = SQLiteTaskStore(path)
        try:
            app = CayuApp(
                task_store=store, resource_access_policy=MembershipAccessPolicy(directory)
            )
            access = await app.access("alice")
            task = await access.tasks.create(
                TaskCreate(type="report", title="Quarterly report"),
                labels={"organization": "acme", "owner": "alice", "sharing": "personal"},
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
