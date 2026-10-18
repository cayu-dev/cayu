"""One contract suite for the in-memory, SQLite and PostgreSQL agent snapshot stores."""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest
from tests.core.test_agent_snapshot_merkle import _access, _digest, _snapshot
from tests.core.test_agent_snapshots import _access as _memory_access
from tests.core.test_agent_snapshots import _capture_request, _providers
from tests.core.test_agent_snapshots import _snapshot as _memory_snapshot

from cayu.snapshots import (
    AgentSnapshotAuthorityRef,
    AgentSnapshotAuthorizationError,
    AgentSnapshotComponentKind,
    AgentSnapshotCoordinator,
    AgentSnapshotGCRequest,
    AgentSnapshotIdentityBinding,
    AgentSnapshotLogicalRef,
    AgentSnapshotMaterialization,
    AgentSnapshotMaterializationProgress,
    AgentSnapshotMaterializationRequest,
    AgentSnapshotPinRequest,
    AgentSnapshotProtection,
    AgentSnapshotProtectionKind,
    AgentSnapshotReleaseRequest,
    AgentSnapshotResultBinding,
    AgentSnapshotRetentionClass,
    AgentSnapshotStoreConflict,
    AgentSnapshotTerminalDisposition,
    AgentSnapshotTrialStateMode,
    AgentSnapshotVerificationError,
    InMemoryAgentSnapshotStore,
    SQLiteAgentSnapshotStore,
)
from cayu.storage._agent_snapshot_schema import AGENT_SNAPSHOT_TABLES
from cayu.storage.migrations import SchemaMode

pytestmark = pytest.mark.anyio


async def _clear_postgres_snapshots(dsn: str) -> None:
    import psycopg

    async with await psycopg.AsyncConnection.connect(dsn) as connection:
        await connection.execute("TRUNCATE " + ", ".join(AGENT_SNAPSHOT_TABLES))
        await connection.commit()


class _Backend:
    """Opens store instances that share one durable database."""

    def __init__(self, name: str, address: str | None) -> None:
        self.name = name
        self.address = address
        self._memory = InMemoryAgentSnapshotStore()
        self._opened: list[object] = []

    def open(self):
        if self.name == "memory":
            return self._memory
        if self.name == "sqlite":
            return SQLiteAgentSnapshotStore(self.address)
        from cayu.storage import PostgresAgentSnapshotStore

        store = PostgresAgentSnapshotStore(self.address, schema_mode=SchemaMode.CREATE)
        self._opened.append(store)
        return store

    @property
    def durable(self) -> bool:
        return self.name != "memory"

    async def sql(self, statement: str, parameters: tuple[object, ...] = ()) -> None:
        """Change stored rows behind the store, to prove reads fail closed."""

        if self.name == "sqlite":
            with sqlite3.connect(self.address) as connection:
                connection.execute(statement, parameters)
            return
        import psycopg

        async with await psycopg.AsyncConnection.connect(self.address) as connection:
            await connection.execute(statement.replace("?", "%s"), parameters)
            await connection.commit()

    async def close(self) -> None:
        for store in self._opened:
            await store.close()


@pytest.fixture(params=["memory", "sqlite", "postgres"])
async def backend(request, tmp_path):
    name = request.param
    if name == "postgres":
        address = request.getfixturevalue("postgres_dsn")
        opened = _Backend(name, address).open()
        await opened.ensure_schema()
        await opened.close()
        await _clear_postgres_snapshots(address)
    else:
        address = str(tmp_path / "snapshots.db") if name == "sqlite" else None
    selected = _Backend(name, address)
    yield selected
    await selected.close()


async def test_put_get_enumerate_and_inspect_survive_a_new_store_instance(backend) -> None:
    baseline = _snapshot()
    changed = _snapshot(changed_kind=AgentSnapshotComponentKind.STANDING_POLICY)
    store = backend.open()

    first = await store.put_snapshot(baseline, baseline.identity_binding)
    assert await store.put_snapshot(baseline, baseline.identity_binding) == first
    await store.put_snapshot(changed, changed.identity_binding)
    assert (await store.get_snapshot(_access(baseline))) == baseline
    assert await store.load_snapshot(baseline.snapshot_root) == baseline
    assert await store.load_identity_binding(baseline.identity_binding.binding_id) == (
        baseline.identity_binding
    )
    assert await store.load_identity_binding(_digest("unknown-binding")) is None
    inspection = await store.inspect_snapshot(_access(baseline))
    assert inspection.shared_bytes > 0
    assert inspection.unique_stored_bytes > 0
    assert (
        inspection.unique_stored_bytes + inspection.shared_bytes == inspection.logical_closure_bytes
    )
    closure = await store.enumerate_snapshot_closure(_access(baseline))
    assert closure[0].digest == baseline.snapshot_root

    reopened = backend.open()
    assert await reopened.inspect_snapshot(_access(baseline)) == inspection
    assert await reopened.enumerate_snapshot_closure(_access(baseline)) == closure
    assert await reopened.save_snapshot(baseline) == baseline


async def test_reads_require_the_exact_binding_and_scope(backend) -> None:
    snapshot = _snapshot()
    store = backend.open()
    await store.put_snapshot(snapshot, snapshot.identity_binding)

    wrong_scope = _access(snapshot).model_copy(
        update={"authority_scope_fingerprint": _digest("another-scope")}
    )
    with pytest.raises(AgentSnapshotAuthorizationError, match="scope"):
        await store.get_snapshot(wrong_scope)
    missing_binding = _access(snapshot).model_copy(update={"binding_id": _digest("missing")})
    with pytest.raises(AgentSnapshotAuthorizationError, match="binding"):
        await store.get_snapshot(missing_binding)
    broadened = AgentSnapshotIdentityBinding.create(
        subject=snapshot.subject,
        snapshot=snapshot.ref,
        authority_scope_fingerprint=_digest("broadened"),
    )
    with pytest.raises(AgentSnapshotStoreConflict, match="scope"):
        await store.put_snapshot(snapshot, broadened)


async def test_corrupt_stored_rows_fail_closed(backend) -> None:
    if not backend.durable:
        pytest.skip("The in-memory store has no rows to corrupt.")
    snapshot = _snapshot()
    store = backend.open()
    await store.put_snapshot(snapshot, snapshot.identity_binding)

    await backend.sql(
        "UPDATE cayu_agent_snapshot_nodes SET node_kind = 'component' WHERE digest = ?",
        (snapshot.snapshot_root,),
    )
    with pytest.raises(AgentSnapshotVerificationError, match="columns|kind"):
        await store.enumerate_snapshot_closure(_access(snapshot))
    await backend.sql(
        "UPDATE cayu_agent_snapshot_nodes SET node_kind = 'manifest' WHERE digest = ?",
        (snapshot.snapshot_root,),
    )
    await backend.sql(
        "UPDATE cayu_agent_snapshot_roots SET manifest_document = ? WHERE snapshot_root = ?",
        (snapshot.component_nodes()[0].model_dump_json(), snapshot.snapshot_root),
    )
    with pytest.raises(AgentSnapshotVerificationError, match="root manifest"):
        await store.get_snapshot(_access(snapshot))


async def test_exact_lifecycle_operations_converge_and_gc_collects(backend) -> None:
    snapshot = _snapshot(capture_request_id="lifecycle")
    stores = (backend.open(), backend.open())
    receipts = await asyncio.gather(
        *(stores[index % 2].put_snapshot(snapshot, snapshot.identity_binding) for index in range(8))
    )
    assert len({receipt.receipt_id for receipt in receipts}) == 1
    access = _access(snapshot)
    pin_request = AgentSnapshotPinRequest(
        operation_id="pin",
        access=access,
        owner="operator",
        reason="candidate-under-evaluation",
        retention_class=AgentSnapshotRetentionClass.CANDIDATE,
    )
    pins = await asyncio.gather(
        *(stores[index % 2].pin_snapshot(pin_request) for index in range(8))
    )
    assert len({pin.receipt_id for pin in pins}) == 1
    with pytest.raises(AgentSnapshotStoreConflict, match="another request"):
        await stores[0].pin_snapshot(pin_request.model_copy(update={"reason": "other"}))

    blocked = await stores[0].plan_snapshot_gc(
        AgentSnapshotGCRequest(operation_id="gc-while-pinned", candidates=(access,))
    )
    assert blocked.blocked_roots == (snapshot.snapshot_root,)
    assert blocked.collectable_roots == ()

    release_request = AgentSnapshotReleaseRequest(
        operation_id="release",
        access=access,
        pin_id=pins[0].pin_id,
        owner="operator",
        reason="candidate-evaluation-complete",
    )
    with pytest.raises(AgentSnapshotAuthorizationError, match="owner"):
        await stores[0].release_snapshot_pin(
            release_request.model_copy(update={"operation_id": "intruder", "owner": "intruder"})
        )
    releases = await asyncio.gather(
        *(stores[index % 2].release_snapshot_pin(release_request) for index in range(8))
    )
    assert len({release.receipt_id for release in releases}) == 1

    request = AgentSnapshotGCRequest(operation_id="gc", candidates=(access,))
    plans = await asyncio.gather(
        *(stores[index % 2].plan_snapshot_gc(request) for index in range(4))
    )
    assert len({plan.plan_id for plan in plans}) == 1
    assert plans[0].collectable_roots == (snapshot.snapshot_root,)
    gc_receipts = await asyncio.gather(
        *(stores[index % 2].execute_snapshot_gc(plans[0]) for index in range(4))
    )
    assert len({receipt.receipt_id for receipt in gc_receipts}) == 1
    assert gc_receipts[0].deleted_bytes == plans[0].bytes_to_delete
    with pytest.raises(AgentSnapshotAuthorizationError):
        await stores[0].get_snapshot(access)
    assert await stores[1].load_snapshot(snapshot.snapshot_root) is None
    assert await stores[1].pin_snapshot(pin_request) == pins[0]
    assert await stores[1].release_snapshot_pin(release_request) == releases[0]


async def test_gc_keeps_held_roots_and_shared_nodes(backend) -> None:
    store = backend.open()
    kept = _snapshot(changed_kind=AgentSnapshotComponentKind.ROLE_PROFILE, agent_id="kept")
    protected = _snapshot(changed_kind=AgentSnapshotComponentKind.KNOWLEDGE, agent_id="protected")
    losing = _snapshot(changed_kind=AgentSnapshotComponentKind.ARTIFACTS, agent_id="losing")
    for snapshot in (kept, protected, losing):
        await store.put_snapshot(snapshot, snapshot.identity_binding)
    await store.pin_snapshot(
        AgentSnapshotPinRequest(
            operation_id="champion",
            access=_access(kept),
            owner="operator",
            reason="champion",
            retention_class=AgentSnapshotRetentionClass.CHAMPION,
        )
    )
    protection = await store.protect_snapshot(
        AgentSnapshotProtection.create(
            operation_id="materializing",
            access=_access(protected),
            kind=AgentSnapshotProtectionKind.MATERIALIZING,
            owner="runtime",
            reason="materializing-in-progress",
        )
    )
    assert await store.protect_snapshot(protection) == protection

    candidates = tuple(
        sorted(
            (_access(snapshot) for snapshot in (kept, protected, losing)),
            key=lambda item: item.snapshot.snapshot_root,
        )
    )
    plan = await store.plan_snapshot_gc(
        AgentSnapshotGCRequest(operation_id="protected-gc", candidates=candidates)
    )
    assert plan.collectable_roots == (losing.snapshot_root,)
    assert set(plan.blocked_roots) == {kept.snapshot_root, protected.snapshot_root}
    assert plan.retained_shared_node_digests
    receipt = await store.execute_snapshot_gc(plan)
    assert receipt.deleted_roots == (losing.snapshot_root,)
    for snapshot in (kept, protected):
        assert (await store.get_snapshot(_access(snapshot))) == snapshot

    released = await store.release_snapshot_protection(
        operation_id="materialized",
        access=_access(protected),
        protection_id=protection.protection_id,
    )
    assert released == protection
    plan = await store.plan_snapshot_gc(
        AgentSnapshotGCRequest(operation_id="after-release", candidates=(_access(protected),))
    )
    assert plan.collectable_roots == (protected.snapshot_root,)


async def test_gc_execution_refuses_a_plan_that_changed(backend) -> None:
    store = backend.open()
    snapshot = _snapshot(agent_id="raced")
    await store.put_snapshot(snapshot, snapshot.identity_binding)
    plan = await store.plan_snapshot_gc(
        AgentSnapshotGCRequest(operation_id="raced-gc", candidates=(_access(snapshot),))
    )
    await store.pin_snapshot(
        AgentSnapshotPinRequest(
            operation_id="late-pin",
            access=_access(snapshot),
            owner="operator",
            reason="pinned-after-planning",
            retention_class=AgentSnapshotRetentionClass.CHAMPION,
        )
    )
    with pytest.raises(AgentSnapshotStoreConflict, match="protection"):
        await store.execute_snapshot_gc(plan)
    assert (await store.get_snapshot(_access(snapshot))) == snapshot


async def test_atomic_import_pin_must_name_the_put_binding(backend) -> None:
    store = backend.open()
    snapshot = _snapshot(agent_id="imported")
    other = _snapshot(agent_id="other")
    pin = AgentSnapshotPinRequest(
        operation_id="import-pin",
        access=_access(other),
        owner="importer",
        reason="imported",
        retention_class=AgentSnapshotRetentionClass.RUN_EVIDENCE,
    )
    with pytest.raises(AgentSnapshotAuthorizationError, match="another binding"):
        await store.put_snapshot_and_pin(snapshot, snapshot.identity_binding, pin)
    with pytest.raises(AgentSnapshotAuthorizationError):
        await store.get_snapshot(_access(snapshot))

    pin = pin.model_copy(update={"access": _access(snapshot)})
    put, pinned = await store.put_snapshot_and_pin(snapshot, snapshot.identity_binding, pin)
    assert put.snapshot == snapshot.ref
    assert pinned.snapshot == snapshot.ref
    plan = await store.plan_snapshot_gc(
        AgentSnapshotGCRequest(operation_id="import-gc", candidates=(_access(snapshot),))
    )
    assert plan.blocked_roots == (snapshot.snapshot_root,)


async def test_retention_sees_identifiers_only_of_held_snapshots(backend) -> None:
    if backend.name == "memory":
        pytest.skip("The in-memory store does not report held identifiers.")
    store = backend.open()
    snapshot = _snapshot(agent_id="held-agent")
    await store.put_snapshot(snapshot, snapshot.identity_binding)
    assert snapshot.snapshot_root not in await store.retention_held_identifiers()

    pin = await store.pin_snapshot(
        AgentSnapshotPinRequest(
            operation_id="retention-pin",
            access=_access(snapshot),
            owner="operator",
            reason="kept",
            retention_class=AgentSnapshotRetentionClass.CHAMPION,
        )
    )
    held = await store.retention_held_identifiers()
    assert {snapshot.snapshot_root, "held-agent"} <= held
    await store.release_snapshot_pin(
        AgentSnapshotReleaseRequest(
            operation_id="retention-release",
            access=_access(snapshot),
            pin_id=pin.pin_id,
            owner="operator",
            reason="done",
        )
    )
    assert snapshot.snapshot_root not in await store.retention_held_identifiers()


async def test_coordinator_capture_materialization_and_lineage_recover_in_a_new_instance(
    backend,
) -> None:
    now = [datetime(2026, 8, 23, tzinfo=UTC)]
    evaluator = _digest("hidden-evaluator-v1")
    snapshot = _memory_snapshot(
        evaluator=AgentSnapshotAuthorityRef(
            identity=AgentSnapshotLogicalRef(fingerprint=evaluator, revision="hidden-evaluator-v1")
        )
    )
    first = AgentSnapshotCoordinator(
        _providers(snapshot), store=backend.open(), clock=lambda: now[0]
    )
    captured = await first.capture(_capture_request(snapshot))
    request = AgentSnapshotMaterializationRequest(
        access=_memory_access(captured),
        candidate_id="candidate-a",
        trial_id="trial-1",
        state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
    )
    materialization = await first.materialize(request)
    trial = await first.begin_trial(
        materialization, case_id="case-1", trial_id="trial-1", evaluator_fingerprint=evaluator
    )
    result = AgentSnapshotResultBinding.create(
        trial=trial,
        session_id="candidate-session-1",
        terminal_disposition=AgentSnapshotTerminalDisposition.COMPLETED,
        runtime_evidence_fingerprint=_digest("runtime-evidence"),
        eval_result_revision=_digest("eval-result"),
        memory_evidence_fingerprint=_digest("memory-evidence"),
        usage_fingerprint=_digest("usage"),
        cost_fingerprint=_digest("cost"),
        recorded_at=now[0],
    )
    assert await first.record_result(result) == result

    now[0] += timedelta(minutes=5)
    providers = _providers(snapshot)
    restarted = AgentSnapshotCoordinator(providers, store=backend.open(), clock=lambda: now[0])
    recovered = await restarted.materialize(request)
    assert recovered.fingerprint == materialization.fingerprint
    assert sum(provider.materialize_calls for provider in providers) == 0
    assert await restarted.store.load_materialization(materialization.fingerprint) is not None
    assert await restarted.store.load_trial(trial.fingerprint) == trial
    assert await restarted.store.load_result(result.fingerprint) == result


async def test_materialization_progress_is_compare_and_set_and_finalization_replays(
    backend,
) -> None:
    store = backend.open()
    snapshot = _memory_snapshot()
    request = AgentSnapshotMaterializationRequest(
        access=_memory_access(snapshot),
        candidate_id="candidate-a",
        trial_id="trial-1",
        state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
    )
    initial = AgentSnapshotMaterializationProgress.create(
        request=request,
        created_at=datetime(2026, 8, 23, tzinfo=UTC),
        components=snapshot.components,
    )
    progress = await store.begin_materialization(initial)
    assert progress == initial
    assert await store.begin_materialization(initial) == initial
    assert await store.load_materialization_progress_for_scope(request) == initial
    assert await store.load_materialization_for_scope(request) is None

    providers = {provider.kind: provider for provider in _providers(snapshot)}
    for operation in progress.operations:
        claimed = await store.claim_materialization_operation(progress, operation.operation_id)
        with pytest.raises(AgentSnapshotStoreConflict, match="compare-and-set"):
            await store.claim_materialization_operation(progress, operation.operation_id)
        result = providers[operation.component_kind]._new_materialization(
            snapshot.component(operation.component_kind), request
        )
        progress = await store.complete_materialization_operation(
            claimed, operation.operation_id, result
        )
    prefinal = progress
    materialization = AgentSnapshotMaterialization.create(
        progress_id=progress.progress_id,
        request=request,
        created_at=progress.created_at,
        components=progress.components,
    )
    first = await store.finalize_materialization(progress, materialization)
    assert await store.finalize_materialization(prefinal, materialization) == first
    final_progress = await store.load_materialization_progress_for_scope(request)
    assert final_progress is not None
    assert final_progress.materialization_fingerprint == materialization.fingerprint
    assert await store.finalize_materialization(final_progress, materialization) == first
    assert await store.save_materialization(materialization) == first
    assert await backend.open().load_materialization_for_scope(request) == first

    changed = list(materialization.components)
    changed[0] = changed[0].model_copy(update={"baseline_fingerprint": _digest("other")})
    different = AgentSnapshotMaterialization.create(
        progress_id=progress.progress_id,
        request=request,
        created_at=progress.created_at,
        components=changed,
    )
    with pytest.raises(AgentSnapshotStoreConflict, match="another identity"):
        await store.save_materialization(different)


async def test_postgres_writes_wait_for_a_retention_share_fence_without_deadlock(
    backend,
) -> None:
    if backend.name != "postgres":
        pytest.skip("Only PostgreSQL retention fences snapshot tables with LOCK TABLE.")
    import psycopg

    store = backend.open()
    await store.ensure_schema()
    snapshot = _snapshot(agent_id="fenced")
    async with await psycopg.AsyncConnection.connect(backend.address) as retention:
        # Retention locks its tables one at a time in sorted order. Take the
        # first, start a put, then take one the put would otherwise write first.
        await retention.execute("LOCK TABLE cayu_agent_snapshot_bindings IN SHARE MODE")
        put = asyncio.create_task(store.put_snapshot(snapshot, snapshot.identity_binding))
        await asyncio.sleep(0.5)
        assert not put.done()
        await asyncio.wait_for(
            retention.execute("LOCK TABLE cayu_agent_snapshot_nodes IN SHARE MODE"), timeout=5
        )
        await retention.commit()
    receipt = await asyncio.wait_for(put, timeout=10)
    assert receipt.snapshot == snapshot.ref


async def test_postgres_artifact_retention_and_a_snapshot_put_run_concurrently(
    backend,
) -> None:
    if backend.name != "postgres":
        pytest.skip("Only PostgreSQL retention fences snapshot tables with LOCK TABLE.")
    import psycopg

    from cayu.artifacts.base import ArtifactMetadata
    from cayu.storage._artifact_retention import artifact_reference_guard, reference_databases
    from cayu.storage.retention import RetentionProtection

    store = backend.open()
    await store.ensure_schema()
    snapshot = _snapshot(agent_id="retained-while-captured")
    databases = await reference_databases((store,))
    assert databases is not None
    metadata = ArtifactMetadata(
        id="retained-artifact", filename="a.txt", size_bytes=1, session_id="retained-session"
    )

    async def retention() -> set[RetentionProtection]:
        async with artifact_reference_guard(databases, metadata, apply=True) as reasons:
            return set(reasons)

    async with await psycopg.AsyncConnection.connect(backend.address) as gate:
        # Hold the put after its first writes, as a slow transaction would, so
        # retention starts its sorted SHARE fence while the put is in flight.
        await gate.execute("LOCK TABLE cayu_agent_snapshot_roots IN SHARE MODE")
        put = asyncio.create_task(store.put_snapshot(snapshot, snapshot.identity_binding))
        await asyncio.sleep(0.5)
        guard = asyncio.create_task(retention())
        await asyncio.sleep(0.5)
        assert not put.done()
        assert not guard.done()
        await gate.commit()
    receipt, reasons = await asyncio.wait_for(asyncio.gather(put, guard), timeout=10)
    assert receipt.snapshot == snapshot.ref
    assert RetentionProtection.SNAPSHOT_PIN not in reasons
