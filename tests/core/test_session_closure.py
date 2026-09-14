from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace

import pytest
from pydantic import SecretStr
from tests.core.session_closure_conformance import (
    assert_detach_closure_conformance,
    create_closure_session,
)

from cayu import CayuApp, SessionClosureExportIncomplete
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.runtime.session_closure import (
    ArtifactSessionClosureStore,
    BudgetSessionClosureStore,
    RetainedSessionClosureStore,
    SessionClosureCoordinator,
    SessionClosureDisposition,
    SessionClosurePolicy,
    SessionClosureRecord,
    SessionEvidenceClosureStore,
    TaskSessionClosureStore,
    session_closure_target_plan_id,
)
from cayu.sessions.base import (
    InMemorySessionStore,
    RunRequest,
    SessionIdentity,
    SessionLineageNode,
    SessionLineageResult,
)
from cayu.storage.sqlite import SQLiteSessionStore, SQLiteTaskStore
from cayu.tasks.base import InMemoryTaskStore, TaskCreate
from cayu.vaults.redaction import SecretRedactor


class _SessionStore:
    supports_session_closure_receipts = True

    def __init__(self, *, status: str = "completed") -> None:
        self.session = SimpleNamespace(status=SimpleNamespace(value=status))
        self.deleted = False
        self.receipts = {}
        self.progress = {}

    async def claim_session_closure_progress(self, progress):
        await self._validate_claim_native_inventory(progress)
        self.progress[(progress["root_session_id"], progress["plan_id"])] = dict(progress)

    async def _validate_claim_native_inventory(self, progress):
        for target in (
            progress["root_session_id"],
            *(item["session_id"] for item in progress["descendants"]),
        ):
            await self.load_session_closure_records(
                target, max_records=progress["max_records"], max_bytes=progress["max_bytes"]
            )

    async def load(self, _session_id: str):
        return None if self.deleted else self.session

    async def validate_session_closure_admission(self, session_id):
        if self.deleted or self.session.status.value in {"running", "interrupting"}:
            raise ValueError("Closure target is not quiescent.")

    async def load_session_closure_records(self, session_id, *, max_records, max_bytes):
        from cayu.runtime._session_closure_records import (
            SESSION_CLOSURE_NATIVE_CLASSES,
            ClosureRecordsBuilder,
        )

        builder = ClosureRecordsBuilder(max_records=max_records, max_bytes=max_bytes)
        for name in SESSION_CLOSURE_NATIVE_CLASSES:
            builder.add_class(
                name, [{"id": session_id}] if name == "session" and not self.deleted else []
            )
        return builder.finish()

    async def delete_session(self, _session_id: str, *, closure_receipt=None) -> None:
        self.deleted = True
        if closure_receipt is not None:
            self.receipts[(_session_id, closure_receipt["plan_id"])] = closure_receipt

    async def load_session_closure_receipt(self, session_id: str, plan_id: str):
        return self.receipts.get((session_id, plan_id))

    async def list_sessions(self, query):
        return []


class _DependentStore:
    store_id = "dependent"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.erased = False

    async def inspect_session_closure(self, _session_id: str, *, policy):
        return {
            "store_id": self.store_id,
            "record_class": "records",
            "disposition": SessionClosureDisposition.OWNED_ELIGIBLE,
            "count": 1,
        }

    async def erase_session_closure(self, _session_id: str, *, policy, plan_id: str):
        if self.fail:
            raise RuntimeError("dependent store unavailable")
        self.erased = True
        return {
            "store_id": self.store_id,
            "record_class": "records",
            "disposition": SessionClosureDisposition.ERASED,
            "count": 1,
        }


class _PagedFallbackSessionStore(_SessionStore):
    def __init__(self, count: int) -> None:
        super().__init__()
        self.children = tuple(
            SimpleNamespace(
                id=f"child-{index}",
                status=SimpleNamespace(value="completed"),
            )
            for index in range(count)
        )

    async def list_sessions(self, query):
        start = int(query.cursor or "0")
        page = self.children[start : start + query.limit]
        next_cursor = str(start + len(page)) if start + len(page) < len(self.children) else None
        return SimpleNamespace(sessions=list(page), next_cursor=next_cursor)


class _LargeExportStore(_DependentStore):
    async def export_session_closure(self, _session_id: str, *, policy):
        return {"payload": "x" * (17 * 1024 * 1024)}


class _ExportSessionStore(_SessionStore):
    async def load_session_export_snapshot(self, _session_id: str, *, limits):
        return SimpleNamespace(document=lambda: {})


class _DescendantSessionStore(_SessionStore):
    supports_session_lineage = True
    supports_session_closure_progress = True
    supports_session_closure_detachment = True
    supports_session_closure_recursive_deletion = True

    def __init__(self) -> None:
        super().__init__()
        now = __import__("datetime").datetime.now(__import__("datetime").UTC)
        self.sessions = {
            "session": SimpleNamespace(status=SimpleNamespace(value="completed")),
            "child": SimpleNamespace(status=SimpleNamespace(value="completed")),
            "grandchild": SimpleNamespace(status=SimpleNamespace(value="completed")),
        }
        self.parents = {"child": "session", "grandchild": "child"}
        self.deleted_order: list[str] = []
        self.detached: tuple[str, ...] = ()
        self.progress = {}
        self.fail_delete_once: str | None = None
        self.fail_progress_after_child = False
        self.now = now

    async def load_session_closure_progress(self, session_id, plan_id):
        return self.progress.get((session_id, plan_id))

    async def claim_session_closure_progress(self, progress):
        await self._validate_claim_native_inventory(progress)
        await self.save_session_closure_progress(progress)

    async def save_session_closure_progress(self, progress):
        if progress.get("completed") and self.fail_progress_after_child:
            self.fail_progress_after_child = False
            raise RuntimeError("simulated progress publication failure")
        self.progress[(progress["root_session_id"], progress["plan_id"])] = dict(progress)

    async def load(self, session_id: str):
        return self.sessions.get(session_id)

    async def query_session_lineage(self, query):
        children = [
            SessionLineageNode(
                id=child_id,
                parent_session_id=query.parent_session_id,
                created_at=self.now,
            )
            for child_id, parent_id in self.parents.items()
            if parent_id == query.parent_session_id
        ]
        return SessionLineageResult(
            parent_session_id=query.parent_session_id,
            children=tuple(children),
        )

    async def detach_session_children(
        self, parent_session_id, child_session_ids, *, closure_receipt
    ):
        self.detached = tuple(child_session_ids)
        for child_id in child_session_ids:
            self.parents.pop(child_id, None)
        return tuple(
            {
                "root_session_id": closure_receipt["root_session_id"],
                "plan_id": closure_receipt["plan_id"],
                "child_session_id": child_id,
                "original_parent_session_id": parent_session_id,
                "detached_at": self.now.isoformat(),
            }
            for child_id in child_session_ids
        )

    async def delete_session(self, session_id: str, *, closure_receipt=None):
        if session_id == self.fail_delete_once:
            self.fail_delete_once = None
            raise RuntimeError("simulated child deletion failure")
        self.deleted_order.append(session_id)
        self.sessions.pop(session_id, None)
        self.parents.pop(session_id, None)
        if closure_receipt is not None:
            self.receipts[(session_id, closure_receipt["plan_id"])] = dict(closure_receipt)


def test_closure_detach_preserves_children_and_records_exact_edge() -> None:
    async def run() -> None:
        store = _DescendantSessionStore()
        policy = SessionClosurePolicy(child_policy="detach")
        report = await SessionClosureCoordinator(store).erase("session", policy=policy)
        assert report.complete is True
        assert store.detached == ("child",)
        assert store.deleted_order == ["session"]
        assert store.parents == {"grandchild": "child"}

    asyncio.run(run())


def test_fallback_lineage_paginates_beyond_single_page() -> None:
    async def run() -> None:
        store = _PagedFallbackSessionStore(1001)
        manifest = await SessionClosureCoordinator(store).inspect(
            "session",
            policy=SessionClosurePolicy(max_descendants=1001),
        )
        child_record = next(
            record for record in manifest.records if record.record_class == "child_sessions"
        )
        assert child_record.count == 1001
        assert manifest.complete is True

    asyncio.run(run())


def test_closure_recursive_deletes_descendants_postorder() -> None:
    async def run() -> None:
        store = _DescendantSessionStore()
        policy = SessionClosurePolicy(child_policy="recursive")
        report = await SessionClosureCoordinator(store).erase("session", policy=policy)
        assert report.complete is True
        assert store.deleted_order == ["grandchild", "child", "session"]
        assert store.sessions == {}

    asyncio.run(run())


def test_recursive_closure_retries_from_durable_progress() -> None:
    async def run() -> None:
        store = _DescendantSessionStore()
        store.fail_delete_once = "grandchild"
        policy = SessionClosurePolicy(child_policy="recursive")
        first = await SessionClosureCoordinator(store).erase("session", policy=policy)
        assert first.complete is False
        assert store.deleted_order == []
        second = await SessionClosureCoordinator(store).erase("session", policy=policy)
        assert second.complete is True
        assert store.deleted_order == ["grandchild", "child", "session"]

    asyncio.run(run())


def test_recursive_retry_reconciles_child_deleted_before_progress_failure() -> None:
    async def run() -> None:
        store = _DescendantSessionStore()
        store.fail_progress_after_child = True
        policy = SessionClosurePolicy(child_policy="recursive")
        first = await SessionClosureCoordinator(store).erase("session", policy=policy)
        assert first.complete is False
        assert store.deleted_order == ["grandchild"]
        second = await SessionClosureCoordinator(store).erase("session", policy=policy)
        assert second.complete is True
        assert store.deleted_order == ["grandchild", "child", "session"]

    asyncio.run(run())


def test_recursive_progress_rejects_future_schema() -> None:
    async def run() -> None:
        store = _DescendantSessionStore()
        policy = SessionClosurePolicy(child_policy="recursive")
        coordinator = SessionClosureCoordinator(store)
        await coordinator.inspect("session", policy=policy)
        plan_id = coordinator._plan_id("session", policy)
        store.progress[("session", plan_id)] = {
            "schema_version": 2,
            "max_records": policy.max_records,
            "max_bytes": policy.max_bytes,
            "root_session_id": "session",
            "plan_id": plan_id,
            "policy_digest": "0" * 64,
            "phase": "recursive",
            "descendants": [],
            "completed": [],
        }
        report = await coordinator.erase("session", policy=policy)
        assert report.complete is False
        assert report.error == "Recursive closure progress could not be validated."

    asyncio.run(run())


def test_recursive_retry_rejects_conflicting_child_receipt() -> None:
    async def run() -> None:
        store = _DescendantSessionStore()
        policy = SessionClosurePolicy(child_policy="recursive")
        coordinator = SessionClosureCoordinator(store)
        await coordinator.inspect("session", policy=policy)
        plan_id = coordinator._plan_id("session", policy)
        child_plan_id = session_closure_target_plan_id(plan_id, "grandchild")
        store.receipts[("grandchild", child_plan_id)] = {
            "root_session_id": "other-session",
            "target_session_id": "grandchild",
            "plan_id": child_plan_id,
            "operation": "recursive",
            "complete": True,
        }
        report = await coordinator.erase("session", policy=policy)
        assert report.complete is False
        assert report.error == "Recursive closure stopped before durable completion."

    asyncio.run(run())


def test_recursive_export_marks_partial_progress_incomplete() -> None:
    async def run() -> None:
        store = _DescendantSessionStore()
        policy = SessionClosurePolicy(child_policy="recursive")
        coordinator = SessionClosureCoordinator(store)
        manifest = await coordinator.inspect("session", policy=policy)
        store.progress[("session", manifest.plan_id)] = {
            "schema_version": 1,
            "max_records": policy.max_records,
            "max_bytes": policy.max_bytes,
            "root_session_id": "session",
            "plan_id": manifest.plan_id,
            "policy_digest": "0" * 64,
            "phase": "recursive",
            "descendants": [
                {"session_id": "child", "parent_session_id": "session"},
                {"session_id": "grandchild", "parent_session_id": "child"},
            ],
            "completed": ["grandchild"],
        }
        export = await coordinator.export("session", policy=policy, allow_partial=True)
        assert export.manifest.complete is False
        assert "session-store/closure_progress" in export.session_records

    asyncio.run(run())


def test_recursive_closure_requires_explicit_progress_capability() -> None:
    async def run() -> None:
        store = _DescendantSessionStore()
        store.supports_session_closure_progress = False
        manifest = await SessionClosureCoordinator(store).inspect(
            "session", policy=SessionClosurePolicy(child_policy="recursive")
        )
        child_record = next(
            record for record in manifest.records if record.record_class == "child_sessions"
        )
        assert child_record.disposition is SessionClosureDisposition.UNSUPPORTED

    asyncio.run(run())


def test_recursive_bound_rejects_before_mutation() -> None:
    async def run() -> None:
        store = _DescendantSessionStore()
        manifest = await SessionClosureCoordinator(store).inspect(
            "session",
            policy=SessionClosurePolicy(child_policy="recursive", max_descendants=1),
        )
        child_record = next(
            record for record in manifest.records if record.record_class == "child_sessions"
        )
        assert manifest.complete is False
        assert child_record.disposition is SessionClosureDisposition.TRUNCATED
        assert store.deleted_order == []

    asyncio.run(run())


def test_recursive_byte_bound_rejects_before_mutation() -> None:
    async def run() -> None:
        store = _DescendantSessionStore()
        manifest = await SessionClosureCoordinator(store).inspect(
            "session",
            policy=SessionClosurePolicy(child_policy="recursive", max_bytes=1),
        )
        child_record = next(
            record for record in manifest.records if record.record_class == "child_sessions"
        )
        assert manifest.complete is False
        assert child_record.disposition is SessionClosureDisposition.TRUNCATED
        assert store.deleted_order == []

    asyncio.run(run())


def test_closure_inspection_reports_unsupported_capability() -> None:
    class Unsupported:
        store_id = "unsupported"

    async def run() -> None:
        manifest = await SessionClosureCoordinator(
            _SessionStore(), dependent_stores=(Unsupported(),)
        ).inspect("session")
        record = manifest.records[-1]
        assert record.disposition is SessionClosureDisposition.UNSUPPORTED
        assert manifest.complete is False

    asyncio.run(run())


def test_closure_erases_dependents_before_session_store() -> None:
    async def run() -> None:
        session_store = _SessionStore()
        dependent = _DependentStore()
        report = await SessionClosureCoordinator(
            session_store, dependent_stores=(dependent,)
        ).erase("session")
        assert report.complete is True
        assert dependent.erased is True
        assert session_store.deleted is True
        assert all(
            record.disposition is SessionClosureDisposition.ERASED
            for record in report.manifest.records
        )

    asyncio.run(run())


def test_closure_failure_is_partial_and_does_not_delete_session() -> None:
    async def run() -> None:
        session_store = _SessionStore()
        report = await SessionClosureCoordinator(
            session_store, dependent_stores=(_DependentStore(fail=True),)
        ).erase("session")
        assert report.complete is False
        assert session_store.deleted is False
        assert report.error is not None

    asyncio.run(run())


def test_closure_rejects_nonterminal_sessions() -> None:
    async def run() -> None:
        with pytest.raises(ValueError, match="terminal"):
            await SessionClosureCoordinator(_SessionStore(status="running")).erase("session")

    asyncio.run(run())


def test_closure_export_is_bounded_and_marks_export_operation() -> None:
    async def run() -> None:
        export = await SessionClosureCoordinator(_SessionStore()).export(
            "session", allow_partial=True
        )
        assert export.manifest.operation.value == "export"
        payload = export.to_bytes()
        assert b"session" in payload
        assert b"content_redacted" in payload

    asyncio.run(run())


def test_closure_export_honors_larger_requested_byte_limit() -> None:
    async def run() -> None:
        policy = SessionClosurePolicy(max_bytes=32 * 1024 * 1024)
        export = await SessionClosureCoordinator(
            _ExportSessionStore(), dependent_stores=(_LargeExportStore(),)
        ).export("session", policy=policy)
        assert len(export.to_bytes()) > 16 * 1024 * 1024

    asyncio.run(run())


@pytest.mark.parametrize("disposition", ["unavailable", "unsupported", "truncated"])
def test_public_closure_export_requires_partial_opt_in(disposition) -> None:
    class Adapter(_DependentStore):
        export_calls = 0

        async def inspect_session_closure(self, session_id, *, policy):
            return {
                "store_id": self.store_id,
                "record_class": "records",
                "disposition": disposition,
            }

        async def export_session_closure(self, session_id, *, policy):
            self.export_calls += 1
            return {"records": ["diagnostic"]}

    async def run():
        adapter = Adapter()
        app = CayuApp(session_closure_stores=(adapter,))
        with pytest.raises(SessionClosureExportIncomplete) as rejected:
            await app.export_session_closure("session")
        assert not rejected.value.manifest.complete
        assert rejected.value.manifest.operation.value == "export"
        assert adapter.export_calls == 0
        budget = next(
            item for item in rejected.value.manifest.records if item.store_id == "budget-store"
        )
        assert budget.disposition is SessionClosureDisposition.RETAINED

        result = await app.export_session_closure("session", allow_partial=True)
        assert not result.manifest.complete
        assert result.session_records[adapter.store_id] == {"records": ["diagnostic"]}
        assert adapter.export_calls == 1

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["ordinary", "unsupported", "missing"])
def test_public_closure_export_rejects_late_export_failure(monkeypatch, failure) -> None:
    class Adapter(_DependentStore):
        async def export_session_closure(self, session_id, *, policy):
            if self.fail:
                if failure == "unsupported":
                    raise NotImplementedError("private provider failure")
                raise RuntimeError("private provider failure")
            return {"records": []}

    async def run():
        adapter = Adapter(fail=True)
        if failure == "missing":
            monkeypatch.setattr(adapter, "export_session_closure", None)
        app = CayuApp(session_closure_stores=(adapter,))
        await app.session_store.create(
            RunRequest(agent_name="test", messages=[], session_id="session"),
            identity=SessionIdentity(provider_name="test", model="test"),
        )
        with pytest.raises(SessionClosureExportIncomplete) as rejected:
            await app.export_session_closure("session")
        assert not rejected.value.manifest.complete
        assert "private provider failure" not in str(rejected.value)
        failed = next(r for r in rejected.value.manifest.records if r.store_id == "dependent")
        expected = "unavailable" if failure == "ordinary" else "unsupported"
        assert failed.disposition.value == expected
        assert failed.capability == "export_session_closure"
        result = await app.export_session_closure("session", allow_partial=True)
        assert not result.manifest.complete
        assert result.session_records["dependent"]["disposition"] == expected
        adapter.fail = False
        if failure == "missing":
            monkeypatch.delattr(adapter, "export_session_closure")
        complete = await app.export_session_closure("session")
        assert complete.manifest.complete
        assert complete.session_records["dependent"] == {"records": []}

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("unsupported", [False, True])
@pytest.mark.parametrize(
    "method,child_policy,record_class",
    [
        ("load_session_closure_records", "reject", "session"),
        ("load_session_closure_tombstones", "detach", "child_sessions"),
        ("load_session_closure_progress", "recursive", "child_sessions"),
    ],
)
def test_public_export_source_failure_evidence(
    tmp_path, monkeypatch, capsys, caplog, backend, unsupported, method, child_policy, record_class
):
    async def run():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "export.sqlite")
        )
        try:
            app = CayuApp(session_store=store)
            await store.create(
                RunRequest(agent_name="test", messages=[], session_id="session"),
                identity=SessionIdentity(provider_name="test", model="test"),
            )

            async def fail(*args, **kwargs):
                if unsupported:
                    raise NotImplementedError("private-export-canary")
                raise RuntimeError("private-export-canary")

            monkeypatch.setattr(store, method, fail)
            policy = SessionClosurePolicy(child_policy=child_policy)
            expected = "unsupported" if unsupported else "unavailable"
            with pytest.raises(SessionClosureExportIncomplete) as error:
                await app.export_session_closure("session", policy=policy)
            failed = next(
                r
                for r in error.value.manifest.records
                if r.store_id == "session-store" and r.record_class == record_class
            )
            assert failed.disposition.value == expected
            assert failed.capability == f"session.{method}"
            assert "private-export-canary" not in str(error.value)
            partial = await app.export_session_closure("session", policy=policy, allow_partial=True)
            assert not partial.manifest.complete
            assert partial.session_records["budget-store"]["disposition"] == "retained"
            assert "private-export-canary" not in partial.to_bytes().decode()
            if method != "load_session_closure_records":
                assert "session-store/session" in partial.session_records
        finally:
            if backend == "sqlite":
                await store.close()

    asyncio.run(run())
    captured = capsys.readouterr()
    assert "private-export-canary" not in captured.out + captured.err + caplog.text


@pytest.mark.parametrize(
    "method,child_policy",
    [
        ("load_session_closure_records", "reject"),
        ("load_session_closure_tombstones", "detach"),
        ("load_session_closure_progress", "recursive"),
    ],
)
def test_public_export_preserves_owner_cancellation(monkeypatch, method, child_policy):
    async def run():
        app = CayuApp()
        await app.session_store.create(
            RunRequest(agent_name="test", messages=[], session_id="session"),
            identity=SessionIdentity(provider_name="test", model="test"),
        )
        entered = asyncio.Event()

        async def blocked(*args, **kwargs):
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(app.session_store, method, blocked)
        task = asyncio.create_task(
            app.export_session_closure(
                "session",
                policy=SessionClosurePolicy(child_policy=child_policy),
                allow_partial=True,
            )
        )
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()
        assert task.cancelling() == 1

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("outcome", ["failure", "cancellation", "changed"])
def test_public_export_rechecks_native_snapshot_after_inspection(
    tmp_path, monkeypatch, capsys, caplog, backend, outcome
):
    async def run():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "export-read.sqlite")
        )
        try:
            await store.create(
                RunRequest(agent_name="test", messages=[], session_id="session"),
                identity=SessionIdentity(provider_name="test", model="test"),
            )
            original = store.load_session_closure_records
            calls = 0
            entered = asyncio.Event()

            async def read(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    if outcome == "failure":
                        raise RuntimeError("private-export-read-canary")
                    if outcome == "cancellation":
                        entered.set()
                        await asyncio.Event().wait()
                    await store.update_labels("session", {"added": "between reads"})
                return await original(*args, **kwargs)

            monkeypatch.setattr(store, "load_session_closure_records", read)
            app = CayuApp(session_store=store)
            if outcome == "failure":
                with pytest.raises(SessionClosureExportIncomplete) as error:
                    await app.export_session_closure("session")
                assert not error.value.manifest.complete
                assert (
                    next(
                        item
                        for item in error.value.manifest.records
                        if item.record_class == "session"
                    ).disposition
                    is SessionClosureDisposition.UNAVAILABLE
                )
            elif outcome == "cancellation":
                task = asyncio.create_task(app.export_session_closure("session"))
                await entered.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert task.cancelled() and task.cancelling() == 1
            else:
                result = await app.export_session_closure("session")
                snapshot = result.session_records["session-store/session"]
                assert snapshot["records"]["labels"] == [{"key": "added", "value": "between reads"}]
                record = next(
                    item for item in result.manifest.records if item.record_class == "labels"
                )
                assert record.count == 1
                assert record.bytes == snapshot["record_bytes"]["labels"]
                assert record.disposition is SessionClosureDisposition.OWNED_ELIGIBLE
            assert calls == 2
            assert await store.load("session") is not None
        finally:
            if backend == "sqlite":
                await store.close()

    asyncio.run(run())
    captured = capsys.readouterr()
    assert "private-export-read-canary" not in captured.out + captured.err + caplog.text


def test_public_export_preserves_lineage_alongside_snapshot(monkeypatch):
    async def run():
        app = CayuApp()
        await app.session_store.create(
            RunRequest(agent_name="test", messages=[], session_id="session"),
            identity=SessionIdentity(provider_name="test", model="test"),
        )

        async def tombstones(*args, **kwargs):
            return ({"parent_session_id": "session", "child_session_id": "child"},)

        monkeypatch.setattr(app.session_store, "load_session_closure_tombstones", tombstones)
        result = await app.export_session_closure(
            "session", policy=SessionClosurePolicy(child_policy="detach")
        )
        assert result.manifest.complete
        assert "session-store/session" in result.session_records
        assert result.session_records["session-store/lineage_tombstones"]["items"] == [
            {"parent_session_id": "session", "child_session_id": "child"}
        ]

    asyncio.run(run())


@pytest.mark.parametrize("document", [None, [], "invalid"])
def test_public_export_rejects_malformed_snapshot_document(monkeypatch, document):
    async def run():
        app = CayuApp()

        async def snapshot(*args, **kwargs):
            return document

        monkeypatch.setattr(app.session_store, "load_session_closure_records", snapshot)
        with pytest.raises(SessionClosureExportIncomplete) as error:
            await app.export_session_closure("session")
        failed = next(r for r in error.value.manifest.records if r.record_class == "session")
        assert failed.disposition is SessionClosureDisposition.UNAVAILABLE

    asyncio.run(run())


def test_closure_rejects_plan_identity_conflict_before_deletion() -> None:
    async def run() -> None:
        session_store = _SessionStore()
        with pytest.raises(ValueError, match="plan identity"):
            await SessionClosureCoordinator(session_store).erase(
                "session", expected_plan_id="wrong"
            )
        assert session_store.deleted is False

    asyncio.run(run())


def test_closure_does_not_relabel_retained_records_as_erased() -> None:
    async def run() -> None:
        report = await SessionClosureCoordinator(
            _SessionStore(),
            dependent_stores=(RetainedSessionClosureStore("budget-store"),),
        ).erase("session")
        assert report.complete is True
        assert report.manifest.records[-1].disposition is SessionClosureDisposition.RETAINED

    asyncio.run(run())


def test_budget_closure_is_explicitly_retain_only_in_manifest_and_export() -> None:
    async def run() -> None:
        coordinator = SessionClosureCoordinator(
            _SessionStore(), dependent_stores=(BudgetSessionClosureStore(),)
        )
        policy = SessionClosurePolicy()
        manifest = await coordinator.inspect("session", policy=policy)
        budget = next(record for record in manifest.records if record.store_id == "budget-store")
        assert budget.disposition is SessionClosureDisposition.RETAINED
        assert budget.capability == "budget.retention-only"

        report = await coordinator.erase("session", policy=policy)
        erased_budget = next(
            record for record in report.manifest.records if record.store_id == "budget-store"
        )
        assert erased_budget.disposition is SessionClosureDisposition.RETAINED
        assert erased_budget.capability == "budget.retention-only"

        export = await coordinator.export("session", policy=policy, allow_partial=True)
        assert export.session_records["budget-store"]["disposition"] == "retained"

    asyncio.run(run())


def test_closure_retries_same_plan_after_session_deletion() -> None:
    async def run() -> None:
        coordinator = SessionClosureCoordinator(_SessionStore())
        first = await coordinator.erase("session")
        second = await coordinator.erase("session", expected_plan_id=first.plan_id)
        assert first.complete is True
        assert second.complete is True
        assert second.already_absent is True

    asyncio.run(run())


def test_in_memory_task_closure_removes_terminal_session_tasks() -> None:
    async def run() -> None:
        store = InMemoryTaskStore()
        await store.create_task(TaskCreate(task_id="task", type="job", session_id="session"))
        await store.complete_task("task", {})
        adapter = TaskSessionClosureStore(store)
        policy = SessionClosurePolicy()
        record = await adapter.inspect_session_closure("session", policy=policy)
        assert record.disposition is SessionClosureDisposition.OWNED_ELIGIBLE
        await adapter.erase_session_closure("session", policy=policy, plan_id="a" * 64)
        assert await store.load_task("task") is None

    asyncio.run(run())


def test_sqlite_task_closure_removes_terminal_session_tasks(tmp_path) -> None:
    async def run() -> None:
        store = SQLiteTaskStore(tmp_path / "closure.sqlite")
        try:
            await store.create_task(
                TaskCreate(task_id="sqlite-task", type="job", session_id="session")
            )
            await store.complete_task("sqlite-task", {})
            adapter = TaskSessionClosureStore(store)
            policy = SessionClosurePolicy()
            record = await adapter.inspect_session_closure("session", policy=policy)
            assert record.disposition is SessionClosureDisposition.OWNED_ELIGIBLE
            await adapter.erase_session_closure("session", policy=policy, plan_id="a" * 64)
            assert await store.load_task("sqlite-task") is None
        finally:
            await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_detach_closure_conformance(tmp_path, backend):
    async def run():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "detach.sqlite")
        )
        competitor = (
            store if backend == "memory" else SQLiteSessionStore(tmp_path / "detach.sqlite")
        )
        try:
            await assert_detach_closure_conformance(store, competitor)
        finally:
            if backend == "sqlite":
                await competitor.close()
                await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("policy", ["detach", "recursive"])
def test_public_closure_rejects_active_root_before_descendants(monkeypatch, policy):
    async def run():
        dependent = _DependentStore()
        app = CayuApp(session_closure_stores=(dependent,))
        await create_closure_session(app.session_store, "root")
        await create_closure_session(app.session_store, "child", "root")

        async def active(session_id):
            if session_id == "root":
                raise ValueError("Session closure requires no active model operation.")

        monkeypatch.setattr(app.session_store, "validate_session_closure_admission", active)
        with pytest.raises(ValueError, match="active model"):
            await app.erase_session_closure(
                "root", policy=SessionClosurePolicy(child_policy=policy)
            )
        assert (await app.session_store.load("child")).parent_session_id == "root"
        assert await app.session_store.load("root") is not None
        assert not dependent.erased

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("policy", ["detach", "recursive"])
def test_public_closure_retry_fences_child_creation(tmp_path, monkeypatch, backend, policy):
    async def run():
        path = tmp_path / "retry.sqlite"
        store = InMemorySessionStore() if backend == "memory" else SQLiteSessionStore(path)
        try:
            await create_closure_session(store, "root")
            await create_closure_session(store, "child", "root")
            original = store.delete_session

            async def fail(session_id, **kwargs):
                raise RuntimeError("delete unavailable")

            monkeypatch.setattr(store, "delete_session", fail)
            selected = SessionClosurePolicy(child_policy=policy)
            first = await CayuApp(session_store=store).erase_session_closure(
                "root", policy=selected
            )
            assert not first.complete
            monkeypatch.setattr(store, "delete_session", original)
            if backend == "sqlite":
                await store.close()
                store = SQLiteSessionStore(path)
            with pytest.raises(ValueError, match="owned by"):
                await create_closure_session(store, "late", "root")
            retry_app = CayuApp(session_store=store)
            retry = await retry_app.erase_session_closure("root", policy=selected)
            assert retry.complete
            assert await store.load("late") is None
            assert await store.load("root") is None
        finally:
            if backend == "sqlite":
                await store.close()

    asyncio.run(run())


def test_public_detach_retry_preserves_receipt_and_surviving_lineage(monkeypatch):
    async def run():
        store = InMemorySessionStore()
        for session_id, parent in (("root", None), ("child", "root"), ("grandchild", "child")):
            await create_closure_session(store, session_id, parent)
        original = store.delete_session

        async def fail(session_id, **kwargs):
            raise RuntimeError("root cleanup failed")

        monkeypatch.setattr(store, "delete_session", fail)
        policy = SessionClosurePolicy(child_policy="detach")
        first = await CayuApp(session_store=store).erase_session_closure("root", policy=policy)
        assert not first.complete
        monkeypatch.setattr(store, "delete_session", original)
        report = await CayuApp(session_store=store).erase_session_closure("root", policy=policy)
        assert report.complete
        receipt = await store.load_session_closure_receipt("root", report.plan_id)
        children = next(
            r for r in receipt["manifest"]["records"] if r["record_class"] == "child_sessions"
        )
        assert children["disposition"] == "retained"
        assert receipt["manifest"]["metadata"]["detached_edges"][0]["child_session_id"] == "child"
        assert (await store.load("child")).parent_session_id is None
        assert (await store.load("grandchild")).parent_session_id == "child"

    asyncio.run(run())


@pytest.mark.parametrize("policy", ["detach", "recursive"])
def test_public_finalization_fences_child_creation(monkeypatch, policy):
    async def run():
        app = CayuApp()
        store = app.session_store
        await create_closure_session(store, "root")
        await create_closure_session(store, "child", "root")
        original = store.delete_session

        async def create_late_child(session_id, **kwargs):
            if session_id == "root":
                with pytest.raises(ValueError, match="owned by"):
                    await create_closure_session(store, "late", "root")
            return await original(session_id, **kwargs)

        monkeypatch.setattr(store, "delete_session", create_late_child)
        report = await app.erase_session_closure(
            "root", policy=SessionClosurePolicy(child_policy=policy)
        )
        assert report.complete
        assert await store.load("late") is None
        assert await store.load_session_closure_receipt("root", report.plan_id) is not None

    asyncio.run(run())


@pytest.mark.parametrize("active_target", ["root", "child"])
def test_public_recursive_closure_preflights_active_tasks(active_target):
    async def run():
        sessions = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        dependent = _DependentStore()
        await create_closure_session(sessions, "root")
        await create_closure_session(sessions, "child", "root")
        await tasks.create_task(TaskCreate(type="test", session_id=active_target))
        app = CayuApp(session_store=sessions, task_store=tasks, session_closure_stores=(dependent,))
        try:
            result = await app.erase_session_closure(
                "root", policy=SessionClosurePolicy(child_policy="recursive")
            )
        except ValueError as exc:
            assert "admission" in str(exc)
        else:
            assert not result.complete
        assert not dependent.erased
        assert await sessions.load("root") is not None
        assert (await sessions.load("child")).parent_session_id == "root"

    asyncio.run(run())


@pytest.mark.parametrize(
    "disposition", ["unavailable", "unsupported", "truncated", "owned_eligible"]
)
def test_public_recursive_closure_rejects_incomplete_settlement(disposition):
    class Incomplete(_DependentStore):
        async def erase_session_closure(self, session_id, *, policy, plan_id):
            return {
                "store_id": self.store_id,
                "record_class": "records",
                "disposition": disposition,
            }

    async def run():
        sessions = InMemorySessionStore()
        await create_closure_session(sessions, "root")
        await create_closure_session(sessions, "child", "root")
        app = CayuApp(session_store=sessions, session_closure_stores=(Incomplete(),))
        result = await app.erase_session_closure(
            "root", policy=SessionClosurePolicy(child_policy="recursive")
        )
        assert not result.complete
        assert await sessions.load("child") is not None
        assert (
            await sessions.load_session_closure_receipt(
                "child", session_closure_target_plan_id(result.plan_id, "child")
            )
            is None
        )

    asyncio.run(run())


def test_public_failed_session_delete_does_not_report_cascade_erased(monkeypatch):
    async def run():
        store = InMemorySessionStore()
        await create_closure_session(store, "session")
        original = store.delete_session

        async def fail(*args, **kwargs):
            raise RuntimeError("delete unavailable")

        monkeypatch.setattr(store, "delete_session", fail)
        app = CayuApp(session_store=store)
        report = await app.erase_session_closure("session")
        assert not report.complete
        evidence = next(
            item for item in report.manifest.records if item.record_class == "recall_receipts"
        )
        assert evidence.disposition is not SessionClosureDisposition.ERASED
        monkeypatch.setattr(store, "delete_session", original)
        report = await app.erase_session_closure("session")
        assert report.complete
        receipt = await store.load_session_closure_receipt("session", report.plan_id)
        evidence = next(
            item
            for item in receipt["manifest"]["records"]
            if item["record_class"] == "recall_receipts"
        )
        assert evidence["disposition"] == "erased"

    asyncio.run(run())


def test_evidence_export_uses_actual_identities_and_rejects_truncation():
    class Evidence:
        truncated = False

        async def list_recall_receipts(self, query):
            return SimpleNamespace(
                items=[
                    SimpleNamespace(receipt_id="receipt-a", session_id="session"),
                    SimpleNamespace(receipt_id="receipt-b", session_id="session"),
                ],
                truncated=self.truncated,
                next_cursor=None,
            )

        async def list_context_exposures(self, query):
            return SimpleNamespace(
                items=[SimpleNamespace(exposure_id="exposure", session_id="session")],
                truncated=False,
                next_cursor=None,
            )

    async def run():
        source = Evidence()
        adapter = SessionEvidenceClosureStore(source)
        result = await adapter.export_session_closure("session", policy=SessionClosurePolicy())
        identities = [item["id_digest"] for items in result.values() for item in items]
        assert len(set(identities)) == 3
        below = SessionClosurePolicy(max_records=2)
        inspection = await adapter.inspect_session_closure("session", policy=below)
        assert inspection.disposition is SessionClosureDisposition.TRUNCATED
        with pytest.raises(ValueError, match="truncated"):
            await adapter.export_session_closure("session", policy=below)
        at_limit = await adapter.export_session_closure(
            "session", policy=SessionClosurePolicy(max_records=3)
        )
        assert at_limit == result
        source.truncated = True
        with pytest.raises(ValueError, match="truncated"):
            await adapter.export_session_closure("session", policy=SessionClosurePolicy())

    asyncio.run(run())


def test_artifact_export_rejects_truncation_and_honors_omission():
    class Artifacts:
        id = "test"

        async def list(self, **kwargs):
            from cayu.artifacts import ArtifactListResult

            return ArtifactListResult(artifacts=(), truncated=True, total_count=1)

    async def run():
        adapter = ArtifactSessionClosureStore(Artifacts())
        with pytest.raises(ValueError, match="truncated"):
            await adapter.export_session_closure("session", policy=SessionClosurePolicy())
        result = await adapter.export_session_closure(
            "session", policy=SessionClosurePolicy(include_artifact_metadata=False)
        )
        assert result["disposition"] == "omitted"

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_public_closure_redacts_adapter_exports_and_receipts(
    tmp_path, capsys, caplog, recwarn, backend
):
    canary = "closure-private-payload-canary"
    keyring = PublicAuthorityAliasKeyring(
        active_key_id="test",
        keys={"test": SecretStr(base64.urlsafe_b64encode(bytes(range(32))).decode().rstrip("="))},
    )

    class Sensitive(_DependentStore):
        async def inspect_session_closure(self, session_id, *, policy):
            result = await super().inspect_session_closure(session_id, policy=policy)
            result["detail"] = canary
            return result

        async def erase_session_closure(self, session_id, *, policy, plan_id):
            result = await super().erase_session_closure(session_id, policy=policy, plan_id=plan_id)
            result["detail"] = canary
            return result

        async def export_session_closure(self, session_id, *, policy):
            return {canary: {"value": canary}}

    async def run():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(
                tmp_path / "redaction.sqlite",
                public_authority_alias_codec=PublicAuthorityAliasCodec(keyring),
            )
        )
        try:
            await create_closure_session(store, "root")
            app = CayuApp(
                session_store=store,
                session_closure_stores=(Sensitive(),),
                secret_redactor=SecretRedactor(canary),
            )
            inspected = await app.inspect_session_closure("root")
            assert canary not in inspected.model_dump_json()
            exported = await app.export_session_closure("root")
            assert canary.encode() not in exported.to_bytes()
            assert exported.content_redacted
            report = await app.erase_session_closure("root")
            assert report.complete
            assert canary not in report.model_dump_json()
            receipt = await store.load_session_closure_receipt("root", report.plan_id)
            assert canary not in repr(receipt)
            replay = await app.erase_session_closure("root", expected_plan_id=report.plan_id)
            assert replay.complete
            assert replay.plan_id == report.plan_id
        finally:
            if backend == "sqlite":
                await store.close()

    asyncio.run(run())
    captured = capsys.readouterr()
    assert canary not in captured.out + captured.err + caplog.text
    assert all(canary not in str(warning.message) for warning in recwarn)


def test_public_incomplete_export_redacts_manifest(capsys, caplog):
    canary = "closure-unavailable-private-canary"

    class Unavailable(_DependentStore):
        async def inspect_session_closure(self, session_id, *, policy):
            return {
                "store_id": self.store_id,
                "record_class": "records",
                "disposition": "unavailable",
                "detail": canary,
            }

    async def run():
        app = CayuApp(
            session_closure_stores=(Unavailable(),), secret_redactor=SecretRedactor(canary)
        )
        await create_closure_session(app.session_store, "root")
        with pytest.raises(SessionClosureExportIncomplete) as caught:
            await app.export_session_closure("root")
        assert canary not in caught.value.manifest.model_dump_json()

    asyncio.run(run())
    captured = capsys.readouterr()
    assert canary not in captured.out + captured.err + caplog.text


def test_public_closure_revalidates_mutated_adapter_record(capsys, caplog, recwarn):
    canary = "closure-mutated-private-canary"

    class PrivateValue:
        def __repr__(self):
            return canary

    record = SessionClosureRecord(
        store_id="dependent", record_class="records", disposition="owned_eligible"
    )
    object.__setattr__(record, "detail", PrivateValue())

    class Malformed(_DependentStore):
        async def inspect_session_closure(self, session_id, *, policy):
            return record

    async def run():
        app = CayuApp(session_closure_stores=(Malformed(),))
        await create_closure_session(app.session_store, "root")
        manifest = await app.inspect_session_closure("root")
        assert not manifest.complete
        assert canary not in manifest.model_dump_json()
        report = await app.erase_session_closure("root")
        assert not report.complete
        assert await app.session_store.load("root") is not None

    asyncio.run(run())
    captured = capsys.readouterr()
    assert canary not in captured.out + captured.err + caplog.text
    assert all(canary not in str(warning.message) for warning in recwarn)


def test_public_closure_preserves_fixed_dispositions_under_secret_collision():
    async def run():
        app = CayuApp(secret_redactor=SecretRedactor("retained"))
        manifest = await app.inspect_session_closure("absent")
        budget = next(record for record in manifest.records if record.store_id == "budget-store")
        assert budget.disposition is SessionClosureDisposition.RETAINED

    asyncio.run(run())


@pytest.mark.parametrize(
    "failure", ["repeated_cursor", "duplicate_identity", "foreign_session", "invalid_completeness"]
)
def test_public_evidence_pagination_rejects_untrusted_pages(monkeypatch, failure):
    async def run():
        sessions = InMemorySessionStore()
        app = CayuApp(
            session_store=sessions,
            session_closure_stores=(SessionEvidenceClosureStore(sessions),),
        )
        await create_closure_session(app.session_store, "root")
        calls = 0

        async def malformed(query):
            nonlocal calls
            calls += 1
            return SimpleNamespace(
                items=[
                    SimpleNamespace(
                        receipt_id="first"
                        if calls == 1 or failure == "duplicate_identity"
                        else "second",
                        session_id="other" if failure == "foreign_session" else "root",
                    )
                ],
                truncated=1 if failure == "invalid_completeness" else True,
                next_cursor="cursor-one",
            )

        monkeypatch.setattr(app.session_store, "list_recall_receipts", malformed)
        manifest = await app.inspect_session_closure("root")
        assert not manifest.complete
        assert calls <= 2
        record = next(
            item for item in manifest.records if item.store_id == "session-store-evidence"
        )
        assert record.disposition is SessionClosureDisposition.UNAVAILABLE
        report = await app.erase_session_closure("root")
        assert not report.complete
        assert await app.session_store.load("root") is not None

    asyncio.run(run())
