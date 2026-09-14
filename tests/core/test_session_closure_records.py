"""Native snapshot bounds and serializer-free record conversion."""

import asyncio
from datetime import UTC, datetime

import pytest
from pydantic import BaseModel
from tests.core.session_closure_conformance import create_closure_session

from cayu import CayuApp
from cayu.runtime._session_closure_records import ClosureRecordsBuilder, ClosureRecordsTooLarge
from cayu.sessions.base import InMemorySessionStore
from cayu.storage.sqlite import SQLiteSessionStore


@pytest.mark.parametrize(
    "store_ids", [("session-store",), ("session-store/session",), ("same", "same")]
)
def test_dependent_stores_cannot_impersonate_native_export_authority(store_ids):
    class Store:
        def __init__(self, store_id):
            self.store_id = store_id

        async def inspect_session_closure(self, *args, **kwargs):
            raise AssertionError("Rejected store must not be invoked.")

    with pytest.raises(ValueError, match="identity is invalid or reserved"):
        CayuApp(session_closure_stores=tuple(Store(store_id) for store_id in store_ids))


@pytest.mark.parametrize("secret", ["x", "a-long-private-label-value"])
def test_public_export_byte_evidence_matches_redacted_native_records(secret):
    from cayu._validation import canonical_durable_json_bytes
    from cayu.vaults.redaction import SecretRedactor

    async def run():
        app = CayuApp(secret_redactor=SecretRedactor(secret))
        await create_closure_session(app.session_store, "root")
        await app.session_store.update_labels("root", {"label": secret})
        original = await app.session_store.load_session_closure_records(
            "root", max_records=100, max_bytes=100_000
        )
        exported = await app.export_session_closure("root")
        native = exported.session_records["session-store/session"]
        assert native["record_bytes"]["labels"] != original["record_bytes"]["labels"]
        for name, rows in native["records"].items():
            expected = sum(
                len(canonical_durable_json_bytes(row, "projected record")) for row in rows
            )
            assert native["counts"][name] == len(rows)
            assert native["record_bytes"][name] == expected
            record = next(item for item in exported.manifest.records if item.record_class == name)
            assert record.count == len(rows)
            assert record.bytes == expected
        assert (await app.session_store.load("root")).labels == {"label": secret}

    asyncio.run(run())


def test_public_export_enforces_combined_adapter_byte_budget():
    from cayu.runtime.session_closure import (
        SessionClosureExportIncomplete,
        SessionClosurePolicy,
        SessionClosureRecord,
    )

    class ExportStore:
        def __init__(self, store_id):
            self.store_id = store_id

        async def inspect_session_closure(self, session_id, *, policy):
            return SessionClosureRecord(
                store_id=self.store_id, record_class="records", disposition="retained", count=1
            )

        async def export_session_closure(self, session_id, *, policy):
            return {"payload": self.store_id * 7500}

    async def run():
        app = CayuApp(session_closure_stores=(ExportStore("aa"), ExportStore("bb")))
        await create_closure_session(app.session_store, "aggregate-export")
        policy = SessionClosurePolicy(max_bytes=25_000)
        with pytest.raises(SessionClosureExportIncomplete) as rejected:
            await app.export_session_closure("aggregate-export", policy=policy)
        failed = next(
            record for record in rejected.value.manifest.records if record.store_id == "bb"
        )
        assert failed.disposition.value == "truncated"
        partial = await app.export_session_closure(
            "aggregate-export", policy=policy, allow_partial=True
        )
        assert not partial.manifest.complete
        assert partial.session_records["aa"]["payload"] == "aa" * 7500
        assert partial.session_records["bb"]["disposition"] == "truncated"
        assert len(partial.to_bytes()) <= policy.max_bytes
        assert await app.session_store.load("aggregate-export") is not None

    asyncio.run(run())


def test_public_export_serialization_is_bounded_without_model_dump(
    monkeypatch, capsys, caplog, recwarn
):
    from cayu.runtime.session_closure import SessionClosureExport

    canary = "private-export-serialization-canary"

    class PrivateValue:
        def __repr__(self):
            return canary

    async def run():
        app = CayuApp()
        await create_closure_session(app.session_store, "bounded-export")
        export = await app.export_session_closure("bounded-export")

        def forbidden_dump(*args, **kwargs):
            raise AssertionError("Export attempted an unrestricted model serializer.")

        monkeypatch.setattr(SessionClosureExport, "model_dump", forbidden_dump)
        expected = export.to_bytes()
        export._max_bytes = len(expected)
        assert export.to_bytes() == expected
        export._max_bytes = len(expected) + 1
        assert export.to_bytes() == expected
        export._max_bytes = len(expected) - 1
        with pytest.raises(ValueError):
            export.to_bytes()
        export._max_bytes = 100_000
        export.session_records["mutated"] = PrivateValue()
        with pytest.raises(TypeError, match="unsupported value") as caught:
            export.to_bytes()
        assert canary not in str(caught.value)

    asyncio.run(run())
    captured = capsys.readouterr()
    assert canary not in captured.out + captured.err + caplog.text
    assert all(canary not in str(warning.message) for warning in recwarn)


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("revoke_before_closure", [False, True])
def test_public_closure_fences_native_writes_but_allows_exact_replay(
    tmp_path, request, backend, revoke_before_closure
):
    from tests.core.test_session_store_shared_conformance import (
        _planned_context_exposure,
        _recall_receipt,
    )
    from tests.core.test_targeted_tool_grants import _codec, _open_targeted_grant

    from cayu.events import Event, EventType
    from cayu.memory.evidence import ContextExposureTransitionRequest
    from cayu.runtime.session_closure import SessionClosureRecord
    from cayu.sessions.base import SessionStatus

    if backend == "postgres":
        dsn = request.getfixturevalue("postgres_dsn")

    async def run():
        if backend == "memory":
            store = competitor = InMemorySessionStore(public_authority_alias_codec=_codec())
        elif backend == "sqlite":
            path = tmp_path / "closure-evidence-owner.sqlite"
            store = SQLiteSessionStore(path, public_authority_alias_codec=_codec())
            competitor = SQLiteSessionStore(path, public_authority_alias_codec=_codec())
        else:
            from cayu.storage.migrations import SchemaMode
            from cayu.storage.postgres import PostgresSessionStore

            store = PostgresSessionStore(
                dsn, schema_mode=SchemaMode.CREATE, public_authority_alias_codec=_codec()
            )
            competitor = PostgresSessionStore(
                dsn, schema_mode=SchemaMode.CREATE, public_authority_alias_codec=_codec()
            )
        entered, release = asyncio.Event(), asyncio.Event()

        class BarrierStore:
            store_id = "closure-barrier"

            async def inspect_session_closure(self, session_id, *, policy):
                return SessionClosureRecord(
                    store_id=self.store_id, record_class="records", disposition="owned_eligible"
                )

            async def erase_session_closure(self, session_id, *, policy, plan_id):
                entered.set()
                await release.wait()
                return SessionClosureRecord(
                    store_id=self.store_id, record_class="records", disposition="erased"
                )

        closing = None
        try:
            session_id = f"closure-memory-write-owner-{revoke_before_closure}"
            _, _, stream, _, grant, _ = await _open_targeted_grant(store, session_id=session_id)
            async for _ in stream:
                pass
            completed_session = await store.load(session_id)
            assert completed_session.status is SessionStatus.COMPLETED
            revocation = dict(
                session_id=session_id,
                expected_run_epoch=completed_session.run_epoch,
                reason="operator_revoked",
                revoked_at=datetime.now(UTC),
            )
            if revoke_before_closure:
                revoked = await store.revoke_targeted_tool_grant(grant.tool_ref, **revocation)
            receipt = _recall_receipt("closure-owner", session_id=session_id)
            exposure, items = _planned_context_exposure("closure-owner", receipt)
            await store.create_recall_receipt(receipt)
            await store.create_context_exposure(exposure, items)
            preparation = ContextExposureTransitionRequest(
                transition_id="closure-preparation",
                expected_state="planned",
                expected_revision=0,
                state="prepared",
                occurred_at=exposure.created_at,
                evidence_kind="request_prepared",
                evidence_ref="closure-prepared",
            )
            prepared = await store.transition_context_exposure(
                session_id, exposure.exposure_id, preparation
            )
            app = CayuApp(session_store=store, session_closure_stores=(BarrierStore(),))
            closing = asyncio.create_task(app.erase_session_closure(session_id))
            await asyncio.wait_for(entered.wait(), timeout=10)
            before = await competitor.load_session_closure_records(
                session_id, max_records=100, max_bytes=1_000_000
            )
            assert await competitor.create_recall_receipt(receipt) == receipt
            assert await competitor.create_context_exposure(exposure, items) == prepared
            assert (
                await competitor.transition_context_exposure(
                    session_id, exposure.exposure_id, preparation
                )
                == prepared
            )
            with pytest.raises(ValueError, match="owned"):
                await competitor.create_recall_receipt(
                    _recall_receipt("closure-new", session_id=session_id)
                )
            new_exposure, new_items = _planned_context_exposure("closure-new", receipt)
            with pytest.raises(ValueError, match="owned"):
                await competitor.create_context_exposure(new_exposure, new_items)
            with pytest.raises(ValueError, match="owned"):
                await competitor.transition_context_exposure(
                    session_id,
                    exposure.exposure_id,
                    ContextExposureTransitionRequest(
                        transition_id="closure-late-transition",
                        expected_state="prepared",
                        expected_revision=1,
                        state="dispatch_started",
                        occurred_at=exposure.created_at,
                        evidence_kind="dispatch_intent_committed",
                        evidence_ref="closure-dispatched",
                    ),
                )
            with pytest.raises(ValueError, match="owned"):
                await competitor.transition_status_if_no_queued_messages(
                    session_id,
                    from_statuses={SessionStatus.COMPLETED},
                    to_status=SessionStatus.RUNNING,
                )

            def forbidden_transform(*args):
                raise AssertionError("Closure must reject before calling the transform.")

            with pytest.raises(ValueError, match="owned"):
                await competitor.reserve_stalled_run_recovery(
                    session_id,
                    statuses={SessionStatus.COMPLETED},
                    inactive_for_seconds=None,
                    checkpoint_transform=forbidden_transform,
                )
            with pytest.raises(ValueError, match="owned"):
                await competitor.fence_run_and_transform_checkpoint(
                    session_id,
                    statuses={SessionStatus.COMPLETED},
                    checkpoint_transform=forbidden_transform,
                )
            with pytest.raises(ValueError, match="owned"):
                await competitor.publish_interaction_transition(
                    session_id,
                    event=Event(
                        id="closure-late-interaction",
                        type=EventType.INTERACTION_INTERRUPTED,
                        session_id=session_id,
                        interaction_id="late-interaction",
                    ),
                    from_statuses={SessionStatus.COMPLETED},
                    to_status=SessionStatus.INTERRUPTED,
                )
            with pytest.raises(ValueError, match="owned"):
                await competitor.claim_budget_reservation_identity(
                    reservation_id="late-reservation",
                    publication_session_id=session_id,
                    publication_id="late-publication",
                )
            with pytest.raises(ValueError, match="owned"):
                await competitor.update_labels(session_id, {"late": "label"})
            with pytest.raises(ValueError, match="owned"):
                await competitor.update_metadata(session_id, {"late": "metadata"})
            if revoke_before_closure:
                assert (
                    await competitor.revoke_targeted_tool_grant(grant.tool_ref, **revocation)
                    == revoked
                )
            else:
                with pytest.raises(ValueError, match="owned"):
                    await competitor.revoke_targeted_tool_grant(grant.tool_ref, **revocation)
            after = await competitor.load_session_closure_records(
                session_id, max_records=100, max_bytes=1_000_000
            )
            assert after == before
            release.set()
            assert (await closing).complete
        finally:
            release.set()
            if closing is not None:
                await asyncio.gather(closing, return_exceptions=True)
            if backend != "memory":
                await competitor.close()
                await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_public_closure_includes_session_memory_evidence(tmp_path, request, backend):
    from tests.core.test_session_store_shared_conformance import (
        _planned_context_exposure,
        _recall_receipt,
    )
    from tests.core.test_targeted_tool_grants import _codec

    if backend == "postgres":
        dsn = request.getfixturevalue("postgres_dsn")

    async def run():
        if backend == "memory":
            store = InMemorySessionStore()
        elif backend == "sqlite":
            store = SQLiteSessionStore(tmp_path / "closure-memory.sqlite")
        else:
            from cayu.storage.migrations import SchemaMode
            from cayu.storage.postgres import PostgresSessionStore

            store = PostgresSessionStore(
                dsn, schema_mode=SchemaMode.CREATE, public_authority_alias_codec=_codec()
            )
        try:
            session_id = "closure-memory-evidence"
            await create_closure_session(store, session_id)
            receipt = _recall_receipt("closure-evidence", session_id=session_id)
            exposure, items = _planned_context_exposure("closure-evidence", receipt)
            await store.create_recall_receipt(receipt)
            await store.create_context_exposure(exposure, items)
            app = CayuApp(session_store=store)
            manifest = await app.inspect_session_closure(session_id)
            assert manifest.complete
            classes = ("recall_receipts", "context_exposures", "recall_item_exposures")
            for name in classes:
                rows = [record for record in manifest.records if record.record_class == name]
                assert len(rows) == 1
                assert rows[0].count == 1
            assert not any(
                record.store_id == "session-store-evidence" for record in manifest.records
            )
            snapshot = await store.load_session_closure_records(
                session_id, max_records=100, max_bytes=100_000
            )
            for name in classes:
                assert snapshot["counts"][name] == 1
                assert snapshot["record_bytes"][name] > 0
            exported = await app.export_session_closure(session_id)
            assert exported.manifest.complete
            assert receipt.receipt_id in exported.to_bytes().decode()
            assert exposure.exposure_id in exported.to_bytes().decode()
            total = sum(snapshot["counts"].values())
            with pytest.raises(ClosureRecordsTooLarge):
                await store.load_session_closure_records(
                    session_id, max_records=total - 1, max_bytes=100_000
                )
            assert await store.load_recall_receipt(session_id, receipt.receipt_id) is not None
            report = await app.erase_session_closure(session_id)
            assert report.complete
            assert await store.load_recall_receipt(session_id, receipt.receipt_id) is None
            assert await store.load_context_exposure(session_id, exposure.exposure_id) is None
        finally:
            if backend != "memory":
                await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_native_closure_snapshot_counts_actual_records(tmp_path, backend):
    async def run():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "closure-records.sqlite")
        )
        try:
            await create_closure_session(store, "root")
            await store.update_labels("root", {"one": "first", "two": "second"})
            result = await store.load_session_closure_records(
                "root", max_records=100, max_bytes=100_000
            )
            assert result["counts"]["session"] == 1
            assert result["counts"]["labels"] == 2
            assert result["counts"]["events"] == 1
            assert result["counts"]["transcript"] == 0
            assert result["counts"]["queued_messages"] == 0
            assert result["counts"]["checkpoint"] == 0
            assert "metadata" not in result["records"]["session"][0]
            assert "labels" not in result["records"]["session"][0]
            result["records"]["labels"][0]["value"] = "changed"
            assert (await store.load("root")).labels == {"one": "first", "two": "second"}
            with pytest.raises(ClosureRecordsTooLarge):
                await store.load_session_closure_records("root", max_records=1, max_bytes=100_000)
            assert await store.load("root") is not None
        finally:
            if backend == "sqlite":
                await store.close()

    asyncio.run(run())


def test_closure_records_do_not_call_model_serializers():
    class Record(BaseModel):
        text: str
        timestamp: datetime

        def model_dump(self, *args, **kwargs):
            raise AssertionError("unbounded serializer called")

    builder = ClosureRecordsBuilder(max_records=2, max_bytes=1000)
    builder.add_class("records", [Record(text="safe", timestamp=datetime(2026, 9, 1, tzinfo=UTC))])
    result = builder.finish()
    assert result["records"]["records"][0]["text"] == "safe"
    assert result["records"]["records"][0]["timestamp"].endswith("+00:00")


def test_closure_records_accept_message_content_but_not_arbitrary_tuple_subclasses():
    from cayu.messages import Message

    class UntrustedTuple(tuple):
        pass

    builder = ClosureRecordsBuilder(max_records=2, max_bytes=1000)
    builder.add_class("messages", [Message.text("user", "safe")])
    assert builder.finish()["records"]["messages"][0]["content"] == [
        {"type": "text", "text": "safe"}
    ]
    with pytest.raises(TypeError, match="unsupported value"):
        builder.add_class("untrusted", [UntrustedTuple(("safe",))])


@pytest.mark.parametrize("bad_value", ["x" * 2000, [1] * 2000])
def test_closure_records_reject_overflow_without_partial_snapshot(bad_value):
    builder = ClosureRecordsBuilder(max_records=2, max_bytes=100)
    with pytest.raises(ValueError):
        builder.add_class("records", [{"value": bad_value}])
    with pytest.raises(ValueError, match="already failed"):
        builder.finish()


def test_closure_record_failure_does_not_render_private_values(capsys, caplog, recwarn):
    canary = "private-closure-value"

    class Private:
        def __repr__(self):
            return canary

    builder = ClosureRecordsBuilder(max_records=2, max_bytes=1000)
    with pytest.raises(TypeError) as caught:
        builder.add_class("records", [{"value": Private()}])
    captured = capsys.readouterr()
    assert canary not in str(caught.value) + captured.out + captured.err + caplog.text
    assert all(canary not in str(warning.message) for warning in recwarn)


@pytest.mark.parametrize("corruption", ["count", "bytes", "missing_class", "foreign_session"])
def test_public_closure_rejects_conflicting_native_inventory(monkeypatch, corruption):
    async def run():
        app = CayuApp()
        store = app.session_store
        await create_closure_session(store, "root")
        original = store.load_session_closure_records

        async def wrong(session_id, **kwargs):
            result = await original(session_id, **kwargs)
            if corruption == "count":
                result["counts"]["session"] = True
            elif corruption == "bytes":
                result["record_bytes"]["session"] += 1
            elif corruption == "missing_class":
                del result["records"]["queued_messages"]
            else:
                result["records"]["session"][0]["id"] = "evil"
            return result

        monkeypatch.setattr(store, "load_session_closure_records", wrong)
        manifest = await app.inspect_session_closure("root")
        assert not manifest.complete
        report = await app.erase_session_closure("root")
        assert not report.complete
        assert await store.load("root") is not None

    asyncio.run(run())
