"""Closure admission checks the complete bounded target set and retired identities."""

import asyncio
from uuid import uuid4

import pytest
from tests.core.session_closure_conformance import create_closure_session
from tests.core.test_tool_effect_store_conformance import _stores

from cayu import CayuApp, RunRequest
from cayu.events import Event
from cayu.runtime.session_closure import (
    SessionClosureChildPolicy,
    SessionClosureDisposition,
    SessionClosurePolicy,
    SessionClosureRecord,
)
from cayu.sessions.base import SessionIdentity, fork_session_invocation


class RecordingDependent:
    store_id = "admission-dependent"

    def __init__(self):
        self.deleted = []

    async def inspect_session_closure(self, session_id, *, policy):
        return SessionClosureRecord(
            store_id=self.store_id,
            record_class="records",
            disposition=SessionClosureDisposition.OWNED_ELIGIBLE,
        )

    async def export_session_closure(self, session_id, *, policy):
        return {"records": []}

    async def erase_session_closure(self, session_id, *, policy, plan_id):
        self.deleted.append(session_id)
        return SessionClosureRecord(
            store_id=self.store_id,
            record_class="records",
            disposition=SessionClosureDisposition.ERASED,
        )


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("bound", ["records", "bytes"])
def test_recursive_child_native_inventory_is_bounded(backend, bound, tmp_path, request):
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    async def scenario():
        async with _stores(backend, tmp_path / "bounded.db", dsn) as open_store:
            store = open_store()
            root, child = f"root-{uuid4().hex}", f"child-{uuid4().hex}"
            await create_closure_session(store, root)
            await create_closure_session(store, child, root)
            for index in range(25):
                await store.append_event(
                    child,
                    Event(
                        id=f"{child}-extra-{index}",
                        session_id=child,
                        type="custom.inventory",
                        payload={"text": "x" * 1024},
                    ),
                )
            dependent = RecordingDependent()
            app = CayuApp(
                enable_logging=False, session_store=store, session_closure_stores=(dependent,)
            )
            policy = SessionClosurePolicy(
                child_policy=SessionClosureChildPolicy.RECURSIVE,
                **({"max_records": 20} if bound == "records" else {"max_bytes": 20000}),
            )
            assert (await app.inspect_session_closure(root, policy=policy)).complete
            assert not (await app.inspect_session_closure(child, policy=policy)).complete
            with pytest.raises(ValueError):
                await app.erase_session_closure(root, policy=policy)
            assert dependent.deleted == []
            assert await store.load(root) is not None
            assert (await store.load(child)).parent_session_id == root
            # Rejection precedes ownership publication too.
            await create_closure_session(store, f"another-{uuid4().hex}", root)

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("target_kind", ["root", "child"])
@pytest.mark.parametrize("bound", ["records", "bytes"])
def test_native_inventory_rejection_does_not_publish_claim(
    backend, target_kind, bound, tmp_path, request, monkeypatch
):
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    async def scenario():
        async with _stores(backend, tmp_path / "growth.db", dsn) as open_store:
            store, other = open_store(), open_store()
            root, child = uuid4().hex, uuid4().hex
            await create_closure_session(store, root)
            await create_closure_session(store, child, root)
            dependent = RecordingDependent()
            app = CayuApp(
                enable_logging=False, session_store=store, session_closure_stores=(dependent,)
            )
            original = store.claim_session_closure_progress
            attempted = []

            async def claim(progress):
                attempted.append(progress)
                target = root if target_kind == "root" else child
                for _ in range(25 if bound == "records" else 1):
                    await other.append_event(
                        target,
                        Event(
                            id=uuid4().hex,
                            session_id=target,
                            type="custom.inventory",
                            payload={"text": "x" * (25000 if bound == "bytes" else 1)},
                        ),
                    )
                await original(progress)

            monkeypatch.setattr(store, "claim_session_closure_progress", claim)
            with pytest.raises(ValueError, match="exceed"):
                await app.erase_session_closure(
                    root,
                    policy=SessionClosurePolicy(
                        child_policy=SessionClosureChildPolicy.RECURSIVE,
                        **({"max_records": 20} if bound == "records" else {"max_bytes": 20000}),
                    ),
                )
            assert len(attempted) == 1
            assert await other.load_session_closure_progress(root, attempted[0]["plan_id"]) is None
            assert dependent.deleted == []
            assert await other.load(root) is not None
            assert (await other.load(child)).parent_session_id == root
            # A fresh coordinator can admit a sufficient policy; rejection must
            # not strand either target under the insufficient immutable plan.
            monkeypatch.setattr(store, "claim_session_closure_progress", original)
            reopened = open_store()
            fresh = CayuApp(
                enable_logging=False, session_store=reopened, session_closure_stores=(dependent,)
            )
            report = await fresh.erase_session_closure(
                root,
                policy=SessionClosurePolicy(
                    child_policy=SessionClosureChildPolicy.RECURSIVE,
                    max_records=1000,
                    max_bytes=100000,
                ),
            )
            assert report.complete
            assert await reopened.load(root) is None
            assert await reopened.load(child) is None

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_receipt_alone_retires_creation_identity(backend, tmp_path, request):
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    async def scenario():
        async with _stores(backend, tmp_path / "receipt-only.db", dsn) as open_store:
            store = open_store()
            target = uuid4().hex
            await create_closure_session(store, target)
            await store.delete_session(
                target,
                closure_receipt={
                    "session_id": target,
                    "plan_id": uuid4().hex * 2,
                    "complete": True,
                },
            )
            other = open_store()
            with pytest.raises(ValueError, match="identity was retired"):
                await create_closure_session(other, target)
            assert await other.load(target) is None

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("parent", ["root", "child", "empty", "reject"])
def test_recursive_claim_rejects_new_edge_before_any_mutation(
    backend, parent, tmp_path, request, monkeypatch
):
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    async def scenario():
        async with _stores(backend, tmp_path / "topology.db", dsn) as open_store:
            store, other = open_store(), open_store()
            root, child, late = (uuid4().hex for _ in range(3))
            await create_closure_session(store, root)
            if parent not in {"empty", "reject"}:
                await create_closure_session(store, child, root)
            dependent = RecordingDependent()
            app = CayuApp(
                enable_logging=False, session_store=store, session_closure_stores=(dependent,)
            )
            original = store.claim_session_closure_progress
            entered, release = asyncio.Event(), asyncio.Event()
            proposed = []

            async def claim(progress):
                proposed.append(progress)
                entered.set()
                await release.wait()
                return await original(progress)

            monkeypatch.setattr(store, "claim_session_closure_progress", claim)
            policy = SessionClosurePolicy(
                child_policy=SessionClosureChildPolicy.REJECT
                if parent == "reject"
                else SessionClosureChildPolicy.RECURSIVE
            )
            closing = asyncio.create_task(app.erase_session_closure(root, policy=policy))
            try:
                await asyncio.wait_for(entered.wait(), 10)
                await create_closure_session(other, late, child if parent == "child" else root)
                release.set()
                with pytest.raises(ValueError, match="lineage changed"):
                    await closing
                assert dependent.deleted == []
                assert (
                    await store.load_session_closure_progress(root, proposed[0]["plan_id"]) is None
                )
                for target in (
                    (root, late) if parent in {"empty", "reject"} else (root, child, late)
                ):
                    assert await other.load(target) is not None
                monkeypatch.setattr(store, "claim_session_closure_progress", original)
                policy = SessionClosurePolicy(child_policy=SessionClosureChildPolicy.RECURSIVE)
                assert (await app.erase_session_closure(root, policy=policy)).complete
                assert await other.load(late) is None
            finally:
                release.set()
                if not closing.done():
                    closing.cancel()
                await asyncio.gather(closing, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_retired_identity_rejected_after_reopening(backend, tmp_path, request):
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    async def scenario():
        async with _stores(backend, tmp_path / "retired.db", dsn) as open_store:
            store = open_store()
            root, child, source_id = (uuid4().hex for _ in range(3))
            await create_closure_session(store, root)
            await create_closure_session(store, child, root)
            await create_closure_session(store, source_id)
            app = CayuApp(enable_logging=False, session_store=store)
            policy = SessionClosurePolicy(child_policy=SessionClosureChildPolicy.RECURSIVE)
            report = await app.erase_session_closure(root, policy=policy)
            assert report.complete
            other = open_store()
            source = await other.load(source_id)
            for target in (root, child):
                with pytest.raises(ValueError):
                    await other.create(
                        RunRequest(agent_name="closure", messages=[], session_id=target),
                        identity=SessionIdentity(provider_name="test", model="test"),
                    )
                assert await other.load(target) is None
                for method in ("create_fork", "create_fork_with_transcript_validation"):
                    fork = source.model_copy(
                        update={
                            "id": target,
                            "parent_session_id": source_id,
                            "invocation": fork_session_invocation(source),
                        }
                    )
                    with pytest.raises(ValueError):
                        await getattr(other, method)(
                            source_session_id=source_id,
                            fork=fork,
                            source_statuses={source.status},
                            expected_source_run_epoch=source.run_epoch,
                            transcript_cursor=None,
                            checkpoint_transform=lambda *_: None,
                            **(
                                {"transcript_validator": lambda *_: None}
                                if method.endswith("validation")
                                else {}
                            ),
                        )
                    assert await other.load(target) is None
            fresh_app = CayuApp(enable_logging=False, session_store=other)
            replay = await fresh_app.erase_session_closure(root, policy=policy)
            assert replay.complete and replay.already_absent

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_receipt_replay_rejects_live_replacement(backend, tmp_path, request, monkeypatch):
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    async def scenario():
        async with _stores(backend, tmp_path / "replay.db", dsn) as open_store:
            store = open_store()
            target = uuid4().hex
            await create_closure_session(store, target)
            original = await store.load(target)
            app = CayuApp(enable_logging=False, session_store=store)
            assert (await app.erase_session_closure(target)).complete
            # A custom/broken backend reports a live replacement under the exact
            # retained receipt. The public replay boundary must not certify absence.
            load = store.load

            async def load_replacement(session_id):
                if session_id == target:
                    return original.model_copy(update={"instance_id": uuid4().hex})
                return await load(session_id)

            monkeypatch.setattr(store, "load", load_replacement)
            with pytest.raises(ValueError, match="receipt conflicts with a live session"):
                await app.erase_session_closure(target)

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("field", ["max_records", "max_bytes"])
def test_claim_replay_preserves_admitted_limits(backend, field, tmp_path, request, monkeypatch):
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    async def scenario():
        async with _stores(backend, tmp_path / "claim-replay.db", dsn) as open_store:
            store = open_store()
            root = uuid4().hex
            await create_closure_session(store, root)
            original = store.claim_session_closure_progress
            attempted = []

            async def lost_ack(progress):
                attempted.append(progress)
                await original(progress)
                raise OSError("claim acknowledgement lost")

            monkeypatch.setattr(store, "claim_session_closure_progress", lost_ack)
            app = CayuApp(enable_logging=False, session_store=store)
            policy = SessionClosurePolicy(child_policy=SessionClosureChildPolicy.RECURSIVE)
            with pytest.raises(OSError, match="acknowledgement lost"):
                await app.erase_session_closure(root, policy=policy)
            other = open_store()
            # Memory reopens the same instance; restore its real admission seam.
            monkeypatch.setattr(store, "claim_session_closure_progress", original)
            saved = await other.load_session_closure_progress(root, attempted[0]["plan_id"])
            assert saved[field] == getattr(policy, field)
            await other.claim_session_closure_progress(saved)
            with pytest.raises(ValueError, match="progress identity conflict"):
                await other.claim_session_closure_progress({**saved, field: saved[field] + 1})
            assert await other.load_session_closure_progress(root, saved["plan_id"]) == saved
            fresh = CayuApp(enable_logging=False, session_store=other)
            assert (await fresh.erase_session_closure(root, policy=policy)).complete

    asyncio.run(scenario())


def test_postgres_cancellation_during_native_admission_rolls_back(
    tmp_path, postgres_dsn, monkeypatch
):
    async def scenario():
        async with _stores("postgres", tmp_path / "cancel.db", postgres_dsn) as open_store:
            store, other = open_store(), open_store()
            root = uuid4().hex
            await create_closure_session(store, root)
            app = CayuApp(enable_logging=False, session_store=store)
            policy = SessionClosurePolicy(child_policy=SessionClosureChildPolicy.RECURSIVE)
            plan_id = (await app.inspect_session_closure(root, policy=policy)).plan_id
            original = store._load_session_closure_records
            entered, release = asyncio.Event(), asyncio.Event()

            async def inventory(cur, session_id, **limits):
                result = await original(cur, session_id, **limits)
                # Inspection uses a read-only connection; pause only the claim's
                # write transaction, after its actual inventory query settles.
                await cur.execute("SHOW transaction_read_only")
                if (await cur.fetchone())[0] == "off":
                    entered.set()
                    await release.wait()
                return result

            monkeypatch.setattr(store, "_load_session_closure_records", inventory)
            closing = asyncio.create_task(app.erase_session_closure(root, policy=policy))
            try:
                await asyncio.wait_for(entered.wait(), 10)
                closing.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await closing
                assert closing.cancelled() and closing.cancelling() == 1
                assert await other.load_session_closure_progress(root, plan_id) is None
                assert await other.load(root) is not None
                # Both the transaction lock and logical ownership must be gone.
                await create_closure_session(other, uuid4().hex, root)
                fresh = CayuApp(enable_logging=False, session_store=other)
                assert (await fresh.erase_session_closure(root, policy=policy)).complete
            finally:
                release.set()
                if not closing.done():
                    closing.cancel()
                await asyncio.gather(closing, return_exceptions=True)

    asyncio.run(scenario())
