from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

from cayu import (
    MAX_KNOWLEDGE_ACTIVATION_RECEIPT_BYTES,
    MAX_KNOWLEDGE_ACTIVATION_REQUEST_BYTES,
    KnowledgeAccessDenied,
    KnowledgeAccessScope,
    KnowledgeActivationConflict,
    KnowledgeActivationDecision,
    KnowledgeActivationDisposition,
    KnowledgeActivationRequest,
    KnowledgeActivationSource,
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeGovernanceConfig,
    KnowledgeGovernanceMode,
    KnowledgePublicationConflict,
    KnowledgeReviewWorkflow,
    KnowledgeRevisionConflict,
    KnowledgeStatus,
    KnowledgeStore,
    decide_knowledge_activation,
    prepare_knowledge_activation_request,
)
from cayu._validation import canonical_durable_json_bytes

_NOW = datetime(2026, 8, 30, 12, tzinfo=UTC)


class _ActivatePolicy:
    identity = "conformance.activation-policy"
    version = "1"

    async def decide_activation(
        self,
        request: KnowledgeActivationRequest,
    ) -> KnowledgeActivationDecision:
        return KnowledgeActivationDecision(
            request_sha256=request.fingerprint,
            disposition=KnowledgeActivationDisposition.ACTIVATE,
            policy_identity=self.identity,
            policy_version=self.version,
            code="conformance_activate",
        )


def _material(entry_id: str, text: str) -> tuple[KnowledgeEntry, list[KnowledgeChunk]]:
    entry = KnowledgeEntry(
        id=entry_id,
        text=text,
        status=KnowledgeStatus.PENDING,
        created_at=_NOW,
        updated_at=_NOW,
    )
    return entry, [
        KnowledgeChunk(
            id=f"{entry_id}:r1:0",
            entry_id=entry_id,
            text=text,
            chunk_index=0,
        )
    ]


async def assert_knowledge_governance_conformance(
    store: KnowledgeStore,
    *,
    access_scope: KnowledgeAccessScope,
) -> None:
    reviewed_entry, reviewed_chunks = _material(
        "governance-conformance-reviewed",
        "Reviewed knowledge remains pending until attributed approval.",
    )
    reviewed_entry = reviewed_entry.model_copy(update={"expires_at": _NOW + timedelta(hours=1)})
    reviewed_request = prepare_knowledge_activation_request(
        reviewed_entry,
        reviewed_chunks,
        access_scope=access_scope,
        operation_id="governance-conformance-reviewed-publication",
        governance_mode=KnowledgeGovernanceMode.REVIEWED,
        source=KnowledgeActivationSource.CURATOR,
    )
    reviewed_authority = await decide_knowledge_activation(
        reviewed_request,
        config=KnowledgeGovernanceConfig(),
    )
    reviewed_publication = await store.publish_entry_revision(
        reviewed_entry,
        reviewed_chunks,
        access_scope=access_scope,
        operation_id=reviewed_request.operation_id,
        activation_authority=reviewed_authority,
    )
    reviewed_activation = await store.load_activation_receipt(
        reviewed_request.operation_id,
        access_scope=access_scope,
    )
    reviewed_replay = await store.publish_entry_revision(
        reviewed_entry,
        reviewed_chunks,
        access_scope=access_scope,
        operation_id=reviewed_request.operation_id,
        activation_authority=reviewed_authority,
    )
    assert reviewed_activation is not None
    assert reviewed_activation.publication_request_sha256 == reviewed_publication.request_sha256
    assert reviewed_activation.committed_at == reviewed_publication.committed_at
    assert reviewed_activation.authority == reviewed_authority
    assert reviewed_replay.replayed is True

    workflow = KnowledgeReviewWorkflow(store, access_scope=access_scope)
    try:
        await workflow.approve(
            reviewed_entry.id,
            operation_id=reviewed_request.operation_id,
            reviewer_identity="conformance.reviewer",
            reviewer_version="1",
        )
    except KnowledgeActivationConflict as exc:
        assert exc.reason == "operation_mismatch"
    else:
        raise AssertionError("A non-review activation operation was accepted as an approval.")
    approval = await workflow.approve(
        reviewed_entry.id,
        operation_id="governance-conformance-reviewed-approval",
        reviewer_identity="conformance.reviewer",
        reviewer_version="1",
    )
    approval_replay = await workflow.approve(
        reviewed_entry.id,
        operation_id="governance-conformance-reviewed-approval",
        reviewer_identity="conformance.reviewer",
        reviewer_version="1",
    )
    approval_publication = await store.load_entry_publication_receipt(
        "governance-conformance-reviewed-approval",
        access_scope=access_scope,
    )
    assert approval.entry.status is KnowledgeStatus.ACTIVE
    assert approval.entry.revision == 2
    assert approval_publication is not None
    assert approval_publication.request_sha256 == approval.receipt.publication_request_sha256
    assert approval_publication.committed_at == approval.receipt.committed_at
    assert approval.receipt.authority.decision.policy_identity == "conformance.reviewer"
    assert approval_replay.entry == approval.entry
    assert approval_replay.receipt.replayed is True
    for replay_scope in (
        {"expected_namespace": "outside-review-scope"},
        {"expected_labels": {"review-scope": "outside"}},
    ):
        try:
            await store.approve_pending_entry(
                approval.receipt.authority,
                access_scope=access_scope,
                **replay_scope,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("Approval replay bypassed its direct-store review scope.")
    for restricted_workflow in (
        KnowledgeReviewWorkflow(
            store,
            access_scope=access_scope,
            namespace="outside-review-scope",
        ),
        KnowledgeReviewWorkflow(
            store,
            access_scope=access_scope,
            labels={"review-scope": "outside"},
        ),
    ):
        try:
            await restricted_workflow.approve(
                reviewed_entry.id,
                operation_id="governance-conformance-reviewed-approval",
                reviewer_identity="conformance.reviewer",
                reviewer_version="1",
            )
        except PermissionError:
            pass
        else:
            raise AssertionError("Approval replay bypassed its workflow review scope.")
    pruned = await store.prune_expired(
        access_scope=access_scope,
        now=_NOW + timedelta(days=1),
    )
    assert pruned == 1
    pruned_replay = await workflow.approve(
        reviewed_entry.id,
        operation_id="governance-conformance-reviewed-approval",
        reviewer_identity="conformance.reviewer",
        reviewer_version="1",
    )
    assert pruned_replay.entry == approval.entry
    assert pruned_replay.receipt.replayed is True

    automatic_entry, automatic_chunks = _material(
        "governance-conformance-automatic",
        "Automatic activation requires explicit application policy authority.",
    )
    automatic_config = KnowledgeGovernanceConfig(
        mode=KnowledgeGovernanceMode.POLICY_AUTOMATIC,
        policy_identity=_ActivatePolicy.identity,
        policy_version=_ActivatePolicy.version,
    )
    automatic_request = prepare_knowledge_activation_request(
        automatic_entry,
        automatic_chunks,
        access_scope=access_scope,
        operation_id="governance-conformance-automatic-publication",
        governance_mode=automatic_config.mode,
        source=KnowledgeActivationSource.CURATOR,
    )
    automatic_authority = await decide_knowledge_activation(
        automatic_request,
        config=automatic_config,
        policy=_ActivatePolicy(),
    )
    active_entry = automatic_entry.model_copy(update={"status": KnowledgeStatus.ACTIVE})
    automatic_publication = await store.publish_entry_revision(
        active_entry,
        automatic_chunks,
        access_scope=access_scope,
        operation_id=automatic_request.operation_id,
        activation_authority=automatic_authority,
    )
    automatic_activation = await store.load_activation_receipt(
        automatic_request.operation_id,
        access_scope=access_scope,
    )
    stored_active = await store.get_entry(automatic_entry.id, access_scope=access_scope)
    assert automatic_activation is not None
    assert automatic_activation.publication_request_sha256 == automatic_publication.request_sha256
    assert automatic_activation.committed_at == automatic_publication.committed_at
    assert automatic_activation.authority == automatic_authority
    assert stored_active is not None
    assert stored_active.status is KnowledgeStatus.ACTIVE

    oversized_successor = active_entry.model_copy(
        update={
            "revision": 2,
            "updated_at": _NOW + timedelta(minutes=1),
            "labels": {f"retirement-scope-{index}": "x" * 512 for index in range(2_100)},
        }
    )
    oversized_chunks = [
        KnowledgeChunk(
            id=f"{active_entry.id}:r2:0",
            entry_id=active_entry.id,
            entry_revision=2,
            text=oversized_successor.text,
            chunk_index=0,
        )
    ]
    try:
        await store.append_entry_revision(
            oversized_successor,
            oversized_chunks,
            expected_revision=1,
            access_scope=access_scope,
        )
    except ValueError as exc:
        assert "activation retirement authority exceeds its canonical byte limit" in str(exc)
    else:
        raise AssertionError("A governed successor exceeded its retirement authority capacity.")
    unchanged_active = await store.get_entry(active_entry.id, access_scope=access_scope)
    assert unchanged_active == active_entry

    expiring_successor = active_entry.model_copy(
        update={
            "revision": 2,
            "updated_at": _NOW + timedelta(minutes=1),
            "expires_at": _NOW + timedelta(minutes=2),
        }
    )
    expiring_chunks = [
        KnowledgeChunk(
            id=f"{active_entry.id}:r2:0",
            entry_id=active_entry.id,
            entry_revision=2,
            text=expiring_successor.text,
            chunk_index=0,
        )
    ]
    await store.append_entry_revision(
        expiring_successor,
        expiring_chunks,
        expected_revision=1,
        access_scope=access_scope,
    )
    assert (
        await store.prune_expired(
            access_scope=access_scope,
            now=_NOW + timedelta(minutes=2),
        )
        == 1
    )
    retained_activation = await store.load_activation_receipt(
        automatic_request.operation_id,
        access_scope=access_scope,
    )
    assert retained_activation == automatic_activation

    near_limit_entry = KnowledgeEntry(
        id="governance-conformance-near-limit",
        text="x",
        status=KnowledgeStatus.PENDING,
        created_at=_NOW,
        updated_at=_NOW,
    )
    near_limit_chunks = [
        KnowledgeChunk(
            id=f"n{index}",
            entry_id=near_limit_entry.id,
            text="x",
            chunk_index=index,
        )
        for index in range(6_400)
    ]
    near_limit_request = prepare_knowledge_activation_request(
        near_limit_entry,
        near_limit_chunks,
        access_scope=access_scope,
        operation_id="governance-conformance-near-limit-publication",
        governance_mode=automatic_config.mode,
        source=KnowledgeActivationSource.CURATOR,
    )
    request_bytes = len(
        canonical_durable_json_bytes(
            near_limit_request.model_dump(mode="json"),
            "near-limit activation request",
        )
    )
    assert request_bytes > MAX_KNOWLEDGE_ACTIVATION_REQUEST_BYTES - 10_000
    assert request_bytes <= MAX_KNOWLEDGE_ACTIVATION_REQUEST_BYTES
    near_limit_authority = await decide_knowledge_activation(
        near_limit_request,
        config=automatic_config,
        policy=_ActivatePolicy(),
    )
    await store.publish_entry_revision(
        near_limit_entry.model_copy(update={"status": KnowledgeStatus.ACTIVE}),
        near_limit_chunks,
        access_scope=access_scope,
        operation_id=near_limit_request.operation_id,
        activation_authority=near_limit_authority,
    )
    near_limit_receipt = await store.load_activation_receipt(
        near_limit_request.operation_id,
        access_scope=access_scope,
    )
    assert near_limit_receipt is not None
    assert (
        len(
            canonical_durable_json_bytes(
                near_limit_receipt.model_dump(mode="json"),
                "near-limit activation receipt",
            )
        )
        <= MAX_KNOWLEDGE_ACTIVATION_RECEIPT_BYTES
    )


async def assert_inaccessible_malformed_activation_receipt_is_hidden(
    store: KnowledgeStore,
    *,
    corrupt_receipt: Callable[[str], Awaitable[None]],
) -> None:
    """Require scope authorization before a durable activation body is decoded."""

    owner_scope = KnowledgeAccessScope.for_namespace(
        "governance-owner",
        allowed_statuses=list(KnowledgeStatus),
    )
    foreign_scope = KnowledgeAccessScope.for_namespace(
        "governance-foreign",
        allowed_statuses=list(KnowledgeStatus),
    )
    entry, chunks = _material(
        "governance-inaccessible-malformed-activation",
        "Out-of-scope activation receipt bodies remain invisible.",
    )
    entry = entry.model_copy(update={"namespace": "governance-owner"})
    request = prepare_knowledge_activation_request(
        entry,
        chunks,
        access_scope=owner_scope,
        operation_id="governance-inaccessible-malformed-activation",
        governance_mode=KnowledgeGovernanceMode.REVIEWED,
        source=KnowledgeActivationSource.CURATOR,
    )
    authority = await decide_knowledge_activation(
        request,
        config=KnowledgeGovernanceConfig(),
    )
    await store.publish_entry_revision(
        entry,
        chunks,
        access_scope=owner_scope,
        operation_id=request.operation_id,
        activation_authority=authority,
    )
    await corrupt_receipt(request.operation_id)

    assert (
        await store.load_activation_receipt(
            request.operation_id,
            access_scope=foreign_scope,
        )
        is None
    )
    try:
        await store.load_activation_receipt(
            request.operation_id,
            access_scope=owner_scope,
        )
    except KnowledgeActivationConflict as exc:
        assert exc.reason == "malformed_receipt"
    else:
        raise AssertionError("An in-scope malformed activation receipt was accepted.")


async def assert_activation_receipt_lifecycle_conformance(
    store: KnowledgeStore,
) -> None:
    """Require activation history to follow current-entry access and erasure."""

    namespace = "governance-lifecycle"
    pending_scope = KnowledgeAccessScope.for_namespace(
        namespace,
        allowed_statuses=[KnowledgeStatus.PENDING],
    )
    lifecycle_audit_scope = KnowledgeAccessScope.for_namespace(
        namespace,
        allowed_statuses=[KnowledgeStatus.PENDING, KnowledgeStatus.DELETED],
    )
    entry, chunks = _material(
        "governance-activation-lifecycle",
        "Activation material must not bypass current lifecycle authorization.",
    )
    entry = entry.model_copy(update={"namespace": namespace})
    request = prepare_knowledge_activation_request(
        entry,
        chunks,
        access_scope=pending_scope,
        operation_id="governance-activation-lifecycle-publication",
        governance_mode=KnowledgeGovernanceMode.REVIEWED,
        source=KnowledgeActivationSource.CURATOR,
    )
    authority = await decide_knowledge_activation(
        request,
        config=KnowledgeGovernanceConfig(),
    )
    await store.publish_entry_revision(
        entry,
        chunks,
        access_scope=pending_scope,
        operation_id=request.operation_id,
        activation_authority=authority,
    )
    assert (
        await store.load_activation_receipt(
            request.operation_id,
            access_scope=pending_scope,
        )
        is not None
    )

    tombstone = await store.delete_entry(
        entry.id,
        expected_revision=1,
        access_scope=pending_scope,
    )
    assert tombstone is not None
    assert tombstone.status is KnowledgeStatus.DELETED
    assert (
        await store.load_activation_receipt(
            request.operation_id,
            access_scope=pending_scope,
        )
        is None
    )
    assert (
        await store.load_activation_receipt(
            request.operation_id,
            access_scope=lifecycle_audit_scope,
        )
        is not None
    )
    try:
        await store.publish_entry_revision(
            entry,
            chunks,
            access_scope=pending_scope,
            operation_id=request.operation_id,
            activation_authority=authority,
        )
    except KnowledgeAccessDenied:
        pass
    else:
        raise AssertionError("Publication replay bypassed current lifecycle authorization.")

    removed = await store.delete_entry(
        entry.id,
        expected_revision=tombstone.revision,
        access_scope=lifecycle_audit_scope,
        hard=True,
    )
    assert removed == tombstone
    assert (
        await store.load_activation_receipt(
            request.operation_id,
            access_scope=lifecycle_audit_scope,
        )
        is None
    )

    expired_at = datetime.now(UTC) - timedelta(seconds=1)
    expiration_audit_scope = KnowledgeAccessScope.for_namespace(
        namespace,
        allowed_statuses=[KnowledgeStatus.PENDING],
        include_expired=True,
    )
    ordinary_scope = KnowledgeAccessScope.for_namespace(
        namespace,
        allowed_statuses=[KnowledgeStatus.PENDING],
    )
    expiring_entry, expiring_chunks = _material(
        "governance-expired-activation-audit",
        "Expiration audit remains explicit and unavailable to ordinary reads.",
    )
    expiring_entry = expiring_entry.model_copy(
        update={
            "namespace": namespace,
            "created_at": expired_at - timedelta(hours=1),
            "updated_at": expired_at - timedelta(hours=1),
            "expires_at": expired_at,
        }
    )
    expiring_request = prepare_knowledge_activation_request(
        expiring_entry,
        expiring_chunks,
        access_scope=expiration_audit_scope,
        operation_id="governance-expired-activation-audit-publication",
        governance_mode=KnowledgeGovernanceMode.REVIEWED,
        source=KnowledgeActivationSource.CURATOR,
    )
    expiring_authority = await decide_knowledge_activation(
        expiring_request,
        config=KnowledgeGovernanceConfig(),
    )
    await store.publish_entry_revision(
        expiring_entry,
        expiring_chunks,
        access_scope=expiration_audit_scope,
        operation_id=expiring_request.operation_id,
        activation_authority=expiring_authority,
    )
    assert (
        await store.prune_expired(
            access_scope=expiration_audit_scope,
            now=datetime.now(UTC),
        )
        == 1
    )
    assert (
        await store.load_activation_receipt(
            expiring_request.operation_id,
            access_scope=expiration_audit_scope,
        )
        is not None
    )
    assert (
        await store.load_activation_receipt(
            expiring_request.operation_id,
            access_scope=ordinary_scope,
        )
        is None
    )

    old_label_scope = KnowledgeAccessScope.for_namespace(
        namespace,
        required_labels={"access": "old"},
        allowed_statuses=[KnowledgeStatus.PENDING],
        include_expired=True,
    )
    relabeled_entry, relabeled_chunks = _material(
        "governance-pruned-final-authority",
        "The final access boundary must survive expiration pruning.",
    )
    relabeled_entry = relabeled_entry.model_copy(
        update={
            "namespace": namespace,
            "labels": {"access": "old"},
            "expires_at": expired_at,
        }
    )
    relabeled_request = prepare_knowledge_activation_request(
        relabeled_entry,
        relabeled_chunks,
        access_scope=old_label_scope,
        operation_id="governance-pruned-final-authority-publication",
        governance_mode=KnowledgeGovernanceMode.REVIEWED,
        source=KnowledgeActivationSource.CURATOR,
    )
    relabeled_authority = await decide_knowledge_activation(
        relabeled_request,
        config=KnowledgeGovernanceConfig(),
    )
    await store.publish_entry_revision(
        relabeled_entry,
        relabeled_chunks,
        access_scope=old_label_scope,
        operation_id=relabeled_request.operation_id,
        activation_authority=relabeled_authority,
    )
    final_entry = relabeled_entry.model_copy(
        update={
            "revision": 2,
            "labels": {"access": "new"},
            "updated_at": relabeled_entry.updated_at + timedelta(seconds=1),
        }
    )
    final_chunks = [
        relabeled_chunks[0].model_copy(
            update={
                "id": f"{relabeled_entry.id}:r2:0",
                "entry_revision": 2,
            }
        )
    ]
    await store.append_entry_revision(
        final_entry,
        final_chunks,
        expected_revision=1,
        access_scope=KnowledgeAccessScope.privileged(),
    )
    assert (
        await store.load_activation_receipt(
            relabeled_request.operation_id,
            access_scope=old_label_scope,
        )
        is None
    )
    assert (
        await store.prune_expired(
            access_scope=KnowledgeAccessScope.privileged(),
            now=datetime.now(UTC),
        )
        == 1
    )
    assert (
        await store.load_activation_receipt(
            relabeled_request.operation_id,
            access_scope=old_label_scope,
        )
        is None
    )
    assert (
        await store.load_activation_receipt(
            relabeled_request.operation_id,
            access_scope=KnowledgeAccessScope.privileged(),
        )
        is not None
    )
    try:
        await store.delete_entry(
            relabeled_entry.id,
            expected_revision=2,
            access_scope=old_label_scope,
            hard=True,
        )
    except KnowledgeAccessDenied:
        pass
    else:
        raise AssertionError("A revoked scope erased retained activation history.")

    changed_expiry_entry, changed_expiry_chunks = _material(
        "governance-pruned-changed-expiry",
        "Pruning is explicit lifecycle evidence, not an inferred receipt timestamp.",
    )
    changed_expiry_entry = changed_expiry_entry.model_copy(update={"namespace": namespace})
    changed_expiry_request = prepare_knowledge_activation_request(
        changed_expiry_entry,
        changed_expiry_chunks,
        access_scope=KnowledgeAccessScope.privileged(),
        operation_id="governance-pruned-changed-expiry-publication",
        governance_mode=KnowledgeGovernanceMode.REVIEWED,
        source=KnowledgeActivationSource.CURATOR,
    )
    changed_expiry_authority = await decide_knowledge_activation(
        changed_expiry_request,
        config=KnowledgeGovernanceConfig(),
    )
    await store.publish_entry_revision(
        changed_expiry_entry,
        changed_expiry_chunks,
        access_scope=KnowledgeAccessScope.privileged(),
        operation_id=changed_expiry_request.operation_id,
        activation_authority=changed_expiry_authority,
    )
    expiring_successor = changed_expiry_entry.model_copy(
        update={
            "revision": 2,
            "updated_at": changed_expiry_entry.updated_at + timedelta(seconds=1),
            "expires_at": expired_at,
        }
    )
    expiring_successor_chunks = [
        changed_expiry_chunks[0].model_copy(
            update={
                "id": f"{changed_expiry_entry.id}:r2:0",
                "entry_revision": 2,
            }
        )
    ]
    await store.append_entry_revision(
        expiring_successor,
        expiring_successor_chunks,
        expected_revision=1,
        access_scope=KnowledgeAccessScope.privileged(),
    )
    assert (
        await store.prune_expired(
            access_scope=KnowledgeAccessScope.privileged(),
            now=datetime.now(UTC),
        )
        == 1
    )
    assert (
        await store.load_activation_receipt(
            changed_expiry_request.operation_id,
            access_scope=KnowledgeAccessScope.privileged(),
        )
        is not None
    )
    try:
        await store.create_entry(
            changed_expiry_entry,
            changed_expiry_chunks,
            access_scope=KnowledgeAccessScope.privileged(),
        )
    except KnowledgePublicationConflict as exc:
        assert exc.reason == "entry_retired"
    else:
        raise AssertionError("A retained governed identity was recreated before audit erasure.")
    try:
        await store.delete_entry(
            changed_expiry_entry.id,
            expected_revision=1,
            access_scope=KnowledgeAccessScope.privileged(),
            hard=True,
        )
    except KnowledgeRevisionConflict as exc:
        assert exc.actual_revision == 2
    else:
        raise AssertionError("A stale CAS erased retained activation history.")
    assert (
        await store.load_activation_receipt(
            changed_expiry_request.operation_id,
            access_scope=KnowledgeAccessScope.privileged(),
        )
        is not None
    )
    erased = await store.delete_entry(
        changed_expiry_entry.id,
        expected_revision=2,
        access_scope=KnowledgeAccessScope.privileged(),
        hard=True,
    )
    assert erased is None
    assert (
        await store.load_activation_receipt(
            changed_expiry_request.operation_id,
            access_scope=KnowledgeAccessScope.privileged(),
        )
        is None
    )
    assert (
        await store.create_entry(
            changed_expiry_entry,
            changed_expiry_chunks,
            access_scope=KnowledgeAccessScope.privileged(),
        )
        == changed_expiry_entry
    )


__all__ = [
    "assert_activation_receipt_lifecycle_conformance",
    "assert_inaccessible_malformed_activation_receipt_is_hidden",
    "assert_knowledge_governance_conformance",
]
