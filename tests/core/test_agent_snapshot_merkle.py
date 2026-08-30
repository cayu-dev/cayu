from __future__ import annotations

import asyncio
import hashlib
import sqlite3
from datetime import UTC, datetime, timedelta
from functools import wraps

import pytest
from pydantic import ValidationError

from cayu import (
    AgentSnapshot,
    AgentSnapshotAccess,
    AgentSnapshotAuthorityRef,
    AgentSnapshotAuthorizationError,
    AgentSnapshotCompleteness,
    AgentSnapshotComponentKind,
    AgentSnapshotComponentRef,
    AgentSnapshotConsistency,
    AgentSnapshotExecutionProfileComponent,
    AgentSnapshotExecutionProfileRef,
    AgentSnapshotGCRequest,
    AgentSnapshotIdentityBinding,
    AgentSnapshotLogicalRef,
    AgentSnapshotMaterializationCapability,
    AgentSnapshotNode,
    AgentSnapshotNodeChild,
    AgentSnapshotNodeKind,
    AgentSnapshotPinRequest,
    AgentSnapshotProtection,
    AgentSnapshotProtectionKind,
    AgentSnapshotRedaction,
    AgentSnapshotReleaseRequest,
    AgentSnapshotRetentionClass,
    AgentSnapshotStoreConflict,
    AgentSnapshotSubject,
    AgentSnapshotVerificationError,
    InMemoryAgentSnapshotStore,
    SQLiteAgentSnapshotStore,
)
from cayu.agent_snapshots import _verify_snapshot_nodes


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _async_test(function):
    @wraps(function)
    def run(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return run


def _logical(
    value: str,
    *,
    scope: str | None,
    source_ref: str | None = None,
) -> AgentSnapshotLogicalRef:
    return AgentSnapshotLogicalRef(
        fingerprint=_digest(value),
        revision=f"revision:{value}",
        scope_fingerprint=scope,
        source_ref=source_ref,
    )


def _profile(value: str = "v1") -> AgentSnapshotExecutionProfileRef:
    return AgentSnapshotExecutionProfileRef(
        schema_version=7,
        fingerprint=_digest(f"execution-profile:{value}"),
        components=(
            AgentSnapshotExecutionProfileComponent(
                name="provider_target",
                fingerprint=_digest(f"provider-target:{value}"),
                availability="available",
            ),
            AgentSnapshotExecutionProfileComponent(
                name="runtime",
                fingerprint=_digest(f"runtime:{value}"),
                availability="available",
            ),
        ),
    )


_PRECISE_COMPONENT_KINDS = (
    AgentSnapshotComponentKind.BODY,
    AgentSnapshotComponentKind.EXECUTION_PROFILE,
    AgentSnapshotComponentKind.ROLE_PROFILE,
    AgentSnapshotComponentKind.TOOL_CATALOGUE,
    AgentSnapshotComponentKind.TOOL_EXPOSURE_POLICY,
    AgentSnapshotComponentKind.KNOWLEDGE,
    AgentSnapshotComponentKind.SESSION,
    AgentSnapshotComponentKind.WORK_CONTEXT,
    AgentSnapshotComponentKind.RECALL_POLICY,
    AgentSnapshotComponentKind.CONTEXT_PROJECTION_POLICY,
    AgentSnapshotComponentKind.LEARNING_POLICY,
    AgentSnapshotComponentKind.WORKSPACE,
    AgentSnapshotComponentKind.ARTIFACTS,
    AgentSnapshotComponentKind.ENVIRONMENT,
    AgentSnapshotComponentKind.EXTERNAL_BINDINGS,
    AgentSnapshotComponentKind.STANDING_POLICY,
)


def _snapshot(
    *,
    agent_id: str = "agent-a",
    application_id: str = "application-a",
    project_id: str = "project-a",
    captured_at: datetime | None = None,
    capture_request_id: str = "capture-a",
    source_suffix: str = "a",
    changed_kind: AgentSnapshotComponentKind | None = None,
    authority_scope: str | None = None,
    parent_snapshot_fingerprint: str | None = None,
) -> AgentSnapshot:
    scope = authority_scope or _digest("tenant-scope")
    profile = _profile(
        "v2" if changed_kind is AgentSnapshotComponentKind.EXECUTION_PROFILE else "v1"
    )
    body = _logical(
        "body-v2" if changed_kind is AgentSnapshotComponentKind.BODY else "body-v1",
        scope=scope,
        source_ref=f"cayu-ref:body:{source_suffix}",
    )
    subject = AgentSnapshotSubject(
        agent_id=agent_id,
        application_id=application_id,
        project_id=project_id,
        body_release=body,
    )
    components = []
    for kind in _PRECISE_COMPONENT_KINDS:
        if kind is AgentSnapshotComponentKind.BODY:
            logical = body
        elif kind is AgentSnapshotComponentKind.EXECUTION_PROFILE:
            logical = AgentSnapshotLogicalRef(
                fingerprint=profile.fingerprint,
                revision="execution-profile:7",
                scope_fingerprint=scope,
            )
        else:
            version = "v2" if kind is changed_kind else "v1"
            logical = _logical(f"{kind.value}:{version}", scope=scope)
        components.append(
            AgentSnapshotComponentRef(
                kind=kind,
                provider_id=f"test.{kind.value}.v1",
                logical=logical,
                consistency=AgentSnapshotConsistency.FRONTIER_CONSISTENT,
                completeness=AgentSnapshotCompleteness.COMPLETE,
                redaction=AgentSnapshotRedaction.BOUNDED_PROJECTION,
                materialization=(
                    AgentSnapshotMaterializationCapability.REFERENCE_ONLY
                    if kind is AgentSnapshotComponentKind.EXTERNAL_BINDINGS
                    else AgentSnapshotMaterializationCapability.REPLAYABLE
                ),
                required=kind
                in {
                    AgentSnapshotComponentKind.BODY,
                    AgentSnapshotComponentKind.EXECUTION_PROFILE,
                },
            )
        )
    lineage = () if parent_snapshot_fingerprint is None else (parent_snapshot_fingerprint,)
    return AgentSnapshot.create(
        capture_request_id=capture_request_id,
        captured_at=captured_at or datetime(2026, 8, 29, tzinfo=UTC),
        subject=subject,
        authority_scope_fingerprint=scope,
        execution_profile=profile,
        components=components,
        parent_snapshot_fingerprint=parent_snapshot_fingerprint,
        lineage=lineage,
    )


def _access(snapshot: AgentSnapshot) -> AgentSnapshotAccess:
    binding = snapshot.identity_binding
    return AgentSnapshotAccess(
        snapshot=snapshot.ref,
        binding_id=binding.binding_id,
        authority_scope_fingerprint=binding.authority_scope_fingerprint,
    )


def _stores(tmp_path):
    return (
        InMemoryAgentSnapshotStore(),
        SQLiteAgentSnapshotStore(tmp_path / "snapshots.db"),
    )


@_async_test
async def test_registration_and_provenance_are_outside_one_shared_snapshot_root(tmp_path) -> None:
    first = _snapshot()
    second = _snapshot(
        agent_id="agent-b",
        application_id="application-b",
        project_id="project-b",
        captured_at=first.captured_at + timedelta(days=1),
        capture_request_id="capture-b",
        source_suffix="relocated",
        parent_snapshot_fingerprint=_digest("provenance-parent"),
    )

    assert first.snapshot_root == second.snapshot_root
    assert _snapshot(authority_scope=_digest("second-authority-scope")).snapshot_root == (
        first.snapshot_root
    )
    assert first.identity_binding.binding_id != second.identity_binding.binding_id
    changed_authorities = AgentSnapshot.model_validate(
        first.model_copy(
            update={
                "evaluator": AgentSnapshotAuthorityRef(
                    identity=_logical("evaluator-b", scope=None)
                ),
                "promotion_authority": AgentSnapshotAuthorityRef(
                    identity=_logical("promotion-b", scope=None)
                ),
            }
        ).model_dump(mode="json")
    )
    assert changed_authorities.snapshot_root == first.snapshot_root

    for store in _stores(tmp_path):
        first_receipt, second_receipt = await asyncio.gather(
            store.put_snapshot(first, first.identity_binding),
            store.put_snapshot(second, second.identity_binding),
        )
        assert first_receipt.snapshot == second_receipt.snapshot == first.ref
        assert first_receipt.binding_id != second_receipt.binding_id
        assert (await store.get_snapshot(_access(first))).subject.agent_id == "agent-a"
        assert (await store.get_snapshot(_access(second))).subject.agent_id == "agent-b"
        assert tuple(
            node.digest for node in await store.enumerate_snapshot_closure(_access(first))
        ) == tuple(node.digest for node in await store.enumerate_snapshot_closure(_access(second)))
        one_binding_plan = await store.plan_snapshot_gc(
            AgentSnapshotGCRequest(
                operation_id=f"one-binding-cannot-collect-{type(store).__name__}",
                candidates=(_access(first),),
            )
        )
        assert one_binding_plan.blocked_roots == (first.snapshot_root,)
        all_accesses = tuple(
            sorted(
                (_access(first), _access(second)),
                key=lambda access: (
                    access.snapshot.snapshot_root,
                    access.binding_id,
                    access.authority_scope_fingerprint,
                ),
            )
        )
        all_bindings_plan = await store.plan_snapshot_gc(
            AgentSnapshotGCRequest(
                operation_id=f"all-bindings-can-collect-{type(store).__name__}",
                candidates=all_accesses,
            )
        )
        assert all_bindings_plan.collectable_roots == (first.snapshot_root,)
        third = _snapshot(
            agent_id="agent-c",
            application_id="application-c",
            project_id="project-c",
            capture_request_id="capture-c",
        )
        await store.put_snapshot(third, third.identity_binding)
        with pytest.raises(AgentSnapshotStoreConflict, match="binding reachability"):
            await store.execute_snapshot_gc(all_bindings_plan)


@_async_test
async def test_save_snapshot_returns_the_canonical_persisted_manifest(tmp_path) -> None:
    first = _snapshot(capture_request_id="capture-first")
    repeated = _snapshot(
        capture_request_id="capture-repeated",
        captured_at=first.captured_at + timedelta(seconds=1),
    )

    assert first.snapshot_root == repeated.snapshot_root
    assert first.identity_binding.binding_id == repeated.identity_binding.binding_id
    assert first != repeated

    for store in _stores(tmp_path):
        assert await store.save_snapshot(first) == first
        saved = await store.save_snapshot(repeated)
        loaded = await store.get_snapshot(_access(repeated))

        assert saved == loaded == first


def test_every_precise_component_changes_the_root_independently() -> None:
    baseline = _snapshot()

    for kind in _PRECISE_COMPONENT_KINDS:
        assert _snapshot(changed_kind=kind).snapshot_root != baseline.snapshot_root

    assert AgentSnapshotComponentKind.MEMORY not in {
        component.kind for component in baseline.components
    }
    assert AgentSnapshotComponentKind.POLICIES not in {
        component.kind for component in baseline.components
    }


@_async_test
async def test_shared_nodes_have_exact_deterministic_size_accounting(tmp_path) -> None:
    baseline = _snapshot()
    changed = _snapshot(changed_kind=AgentSnapshotComponentKind.STANDING_POLICY)

    inspections = []
    closure_orders = []
    for store in _stores(tmp_path):
        await store.put_snapshot(baseline, baseline.identity_binding)
        await store.put_snapshot(changed, changed.identity_binding)
        baseline_inspection = await store.inspect_snapshot(_access(baseline))
        changed_inspection = await store.inspect_snapshot(_access(changed))
        assert baseline_inspection.shared_bytes > 0
        assert baseline_inspection.unique_stored_bytes > 0
        assert changed_inspection.shared_bytes == baseline_inspection.shared_bytes
        assert (
            baseline_inspection.unique_stored_bytes + baseline_inspection.shared_bytes
            == baseline_inspection.logical_closure_bytes
        )
        inspections.append(baseline_inspection)
        closure_orders.append(
            tuple(node.digest for node in await store.enumerate_snapshot_closure(_access(baseline)))
        )

    assert inspections[0] == inspections[1]
    assert closure_orders[0] == closure_orders[1]
    restarted = SQLiteAgentSnapshotStore(tmp_path / "snapshots.db")
    assert (
        tuple(node.digest for node in await restarted.enumerate_snapshot_closure(_access(baseline)))
        == closure_orders[0]
    )


@_async_test
async def test_digest_alone_wrong_scope_and_broadened_binding_never_authorize_reads(
    tmp_path,
) -> None:
    snapshot = _snapshot()

    broadened_component = snapshot.components[0].model_copy(
        update={
            "logical": snapshot.components[0].logical.model_copy(
                update={"scope_fingerprint": _digest("broadened-component-scope")}
            )
        }
    )
    forged_components = snapshot.model_copy(
        update={
            "components": (
                broadened_component,
                *snapshot.components[1:],
            )
        }
    ).model_dump(mode="json")
    with pytest.raises(ValidationError, match="components cannot broaden"):
        AgentSnapshot.model_validate(forged_components)

    mismatched_body = snapshot.model_copy(
        update={
            "subject": snapshot.subject.model_copy(
                update={
                    "body_release": snapshot.subject.body_release.model_copy(
                        update={"revision": "revision:another-body-release"}
                    )
                }
            )
        }
    ).model_dump(mode="json")
    with pytest.raises(ValidationError, match="Body component does not match"):
        AgentSnapshot.model_validate(mismatched_body)

    for store in _stores(tmp_path):
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


@_async_test
async def test_missing_corrupt_wrong_kind_and_indexed_objects_fail_closed(tmp_path) -> None:
    snapshot = _snapshot()
    in_memory = InMemoryAgentSnapshotStore()
    await in_memory.put_snapshot(snapshot, snapshot.identity_binding)
    closure = await in_memory.enumerate_snapshot_closure(_access(snapshot))
    missing_digest = closure[-1].digest
    missing_node = in_memory._snapshot_nodes.pop(missing_digest)
    with pytest.raises(AgentSnapshotVerificationError, match="missing"):
        await in_memory.enumerate_snapshot_closure(_access(snapshot))
    in_memory._snapshot_nodes[missing_digest] = missing_node

    root = in_memory._snapshot_nodes[snapshot.snapshot_root]
    in_memory._snapshot_nodes[snapshot.snapshot_root] = root.model_copy(
        update={"schema_id": "cayu.agent-snapshot.manifest.v999"}
    )
    with pytest.raises(AgentSnapshotVerificationError, match="corrupt|schema"):
        await in_memory.enumerate_snapshot_closure(_access(snapshot))

    sqlite = SQLiteAgentSnapshotStore(tmp_path / "corrupt.db")
    await sqlite.put_snapshot(snapshot, snapshot.identity_binding)
    with sqlite3.connect(sqlite.path) as connection:
        connection.execute(
            "UPDATE cayu_agent_snapshot_nodes SET node_kind = 'component' WHERE digest = ?",
            (snapshot.snapshot_root,),
        )
    with pytest.raises(AgentSnapshotVerificationError, match="columns|kind"):
        await sqlite.enumerate_snapshot_closure(_access(snapshot))

    manifest_store = SQLiteAgentSnapshotStore(tmp_path / "corrupt-root-manifest.db")
    await manifest_store.put_snapshot(snapshot, snapshot.identity_binding)
    component_node = snapshot.component_nodes()[0]
    with sqlite3.connect(manifest_store.path) as connection:
        connection.execute(
            "UPDATE cayu_agent_snapshot_roots SET manifest_document = ? WHERE snapshot_root = ?",
            (component_node.model_dump_json(), snapshot.snapshot_root),
        )
    with pytest.raises(AgentSnapshotVerificationError, match="root manifest"):
        await manifest_store.enumerate_snapshot_closure(_access(snapshot))


def test_node_payloads_are_strict_canonical_json() -> None:
    with pytest.raises((TypeError, ValueError)):
        AgentSnapshotNode.create(
            node_kind=AgentSnapshotNodeKind.COMPONENT,
            schema_id="test.component.v1",
            payload={"ambiguous": {"not", "json"}},
        )
    node = AgentSnapshotNode.create(
        node_kind=AgentSnapshotNodeKind.COMPONENT,
        schema_id="test.component.v1",
        payload={"stable": [1, "two", True, None]},
    )
    forged = node.model_dump(mode="json")
    forged["digest"] = _digest("forged")
    with pytest.raises(ValidationError, match="digest"):
        AgentSnapshotNode.model_validate(forged)


def test_fault_injected_cycle_is_rejected_before_manifest_use(monkeypatch) -> None:
    snapshot = _snapshot()
    root = snapshot.root_node()
    cyclic = root.model_copy(
        update={
            "children": (
                AgentSnapshotNodeChild(
                    relation=AgentSnapshotComponentKind.BODY,
                    node_kind=AgentSnapshotNodeKind.MANIFEST,
                    schema_id=root.schema_id,
                    digest=root.digest,
                ),
            )
        }
    )
    monkeypatch.setattr(
        AgentSnapshotNode,
        "model_validate",
        classmethod(lambda cls, value: cyclic),
    )

    with pytest.raises(AgentSnapshotVerificationError, match="cycle"):
        _verify_snapshot_nodes(snapshot, {root.digest: cyclic})


@_async_test
async def test_exact_put_pin_release_and_gc_operations_converge(tmp_path) -> None:
    for index, store in enumerate(_stores(tmp_path)):
        snapshot = _snapshot(capture_request_id=f"concurrent-{index}")
        receipts = await asyncio.gather(
            *(store.put_snapshot(snapshot, snapshot.identity_binding) for _ in range(16))
        )
        assert len({receipt.receipt_id for receipt in receipts}) == 1
        access = _access(snapshot)
        pin_request = AgentSnapshotPinRequest(
            operation_id=f"pin-{index}",
            access=access,
            owner="n9-operator",
            reason="candidate-under-evaluation",
            retention_class=AgentSnapshotRetentionClass.CANDIDATE,
        )
        pins = await asyncio.gather(*(store.pin_snapshot(pin_request) for _ in range(16)))
        assert len({pin.receipt_id for pin in pins}) == 1
        release_request = AgentSnapshotReleaseRequest(
            operation_id=f"release-{index}",
            access=access,
            pin_id=pins[0].pin_id,
            owner="n9-operator",
            reason="candidate-evaluation-complete",
        )
        releases = await asyncio.gather(
            *(store.release_snapshot_pin(release_request) for _ in range(16))
        )
        assert len({release.receipt_id for release in releases}) == 1
        request = AgentSnapshotGCRequest(
            operation_id=f"gc-{index}",
            candidates=(access,),
        )
        plans = await asyncio.gather(*(store.plan_snapshot_gc(request) for _ in range(8)))
        assert len({plan.plan_id for plan in plans}) == 1
        gc_receipts = await asyncio.gather(*(store.execute_snapshot_gc(plans[0]) for _ in range(8)))
        assert len({receipt.receipt_id for receipt in gc_receipts}) == 1
        with pytest.raises(AgentSnapshotAuthorizationError):
            await store.get_snapshot(access)
        assert await store.pin_snapshot(pin_request) == pins[0]
        assert await store.release_snapshot_pin(release_request) == releases[0]


@_async_test
async def test_two_sqlite_store_instances_converge_on_exact_lifecycle_writes(tmp_path) -> None:
    path = tmp_path / "multi-instance.db"
    stores = (SQLiteAgentSnapshotStore(path), SQLiteAgentSnapshotStore(path))
    snapshot = _snapshot(capture_request_id="multi-instance")
    puts = await asyncio.gather(
        *(
            stores[index % 2].put_snapshot(snapshot, snapshot.identity_binding)
            for index in range(20)
        )
    )
    assert len({receipt.receipt_id for receipt in puts}) == 1

    request = AgentSnapshotPinRequest(
        operation_id="multi-instance-pin",
        access=_access(snapshot),
        owner="operator",
        reason="cross-process-convergence",
        retention_class=AgentSnapshotRetentionClass.RUN_EVIDENCE,
    )
    pins = await asyncio.gather(*(stores[index % 2].pin_snapshot(request) for index in range(20)))
    assert len({receipt.receipt_id for receipt in pins}) == 1
    release = AgentSnapshotReleaseRequest(
        operation_id="multi-instance-release",
        access=request.access,
        pin_id=pins[0].pin_id,
        owner="operator",
        reason="cross-process-convergence-complete",
    )
    releases = await asyncio.gather(
        *(stores[index % 2].release_snapshot_pin(release) for index in range(20))
    )
    assert len({receipt.receipt_id for receipt in releases}) == 1


@_async_test
async def test_gc_preserves_all_pin_and_lifecycle_protection_classes(tmp_path) -> None:
    for store_index, store in enumerate(_stores(tmp_path)):
        snapshots = [
            _snapshot(
                changed_kind=kind,
                agent_id=f"agent-{kind.value}",
                capture_request_id=f"capture-{store_index}-{kind.value}",
            )
            for kind in (
                AgentSnapshotComponentKind.ROLE_PROFILE,
                AgentSnapshotComponentKind.TOOL_CATALOGUE,
                AgentSnapshotComponentKind.KNOWLEDGE,
                AgentSnapshotComponentKind.WORKSPACE,
                AgentSnapshotComponentKind.ENVIRONMENT,
                AgentSnapshotComponentKind.STANDING_POLICY,
            )
        ]
        losing = _snapshot(
            changed_kind=AgentSnapshotComponentKind.ARTIFACTS,
            agent_id="losing-agent",
            capture_request_id=f"losing-{store_index}",
        )
        for snapshot in (*snapshots, losing):
            await store.put_snapshot(snapshot, snapshot.identity_binding)

        pin = await store.pin_snapshot(
            AgentSnapshotPinRequest(
                operation_id=f"pinned-{store_index}",
                access=_access(snapshots[0]),
                owner="operator",
                reason="champion",
                retention_class=AgentSnapshotRetentionClass.CHAMPION,
            )
        )
        assert pin.snapshot == snapshots[0].ref
        for protection_index, (snapshot, kind) in enumerate(
            zip(
                snapshots[1:],
                (
                    AgentSnapshotProtectionKind.ACTIVE,
                    AgentSnapshotProtectionKind.OUTCOME_UNKNOWN,
                    AgentSnapshotProtectionKind.IMPORTING,
                    AgentSnapshotProtectionKind.EXPORTING,
                    AgentSnapshotProtectionKind.MATERIALIZING,
                ),
                strict=True,
            )
        ):
            await store.protect_snapshot(
                AgentSnapshotProtection.create(
                    operation_id=(f"protect-{store_index}-{protection_index}-{kind.value}"),
                    access=_access(snapshot),
                    kind=kind,
                    owner="runtime",
                    reason=f"{kind.value}-in-progress",
                )
            )

        accesses = tuple(
            sorted(
                (_access(snapshot) for snapshot in (*snapshots, losing)),
                key=lambda item: item.snapshot.snapshot_root,
            )
        )
        plan = await store.plan_snapshot_gc(
            AgentSnapshotGCRequest(
                operation_id=f"protected-gc-{store_index}",
                candidates=accesses,
            )
        )
        assert plan.collectable_roots == (losing.snapshot_root,)
        assert set(plan.blocked_roots) == {snapshot.snapshot_root for snapshot in snapshots}
        assert plan.retained_shared_node_digests
        receipt = await store.execute_snapshot_gc(plan)
        assert receipt.deleted_roots == (losing.snapshot_root,)
        for snapshot in snapshots:
            assert (await store.get_snapshot(_access(snapshot))).snapshot_root == (
                snapshot.snapshot_root
            )


def test_content_nodes_exclude_registration_scope_paths_and_private_values() -> None:
    snapshot = _snapshot(
        agent_id="secret-agent-registration",
        application_id="secret-application-registration",
        project_id="secret-project-registration",
        source_suffix="private-path-alias",
        capture_request_id="super-secret-credential-value",
    )
    encoded_nodes = "\n".join(node.model_dump_json() for node in snapshot.merkle_nodes())
    for forbidden in (
        "secret-agent-registration",
        "secret-application-registration",
        "secret-project-registration",
        "private-path-alias",
        snapshot.authority_scope_fingerprint,
        "super-secret-credential-value",
        "sealed-answer-is-42",
        "provider-state-token-123",
    ):
        assert forbidden not in encoded_nodes
    assert "hidden_evaluator_truth" in snapshot.exclusions
    assert "provider_continuation_state" in snapshot.exclusions
