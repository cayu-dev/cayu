from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from examples.knowledge_maintenance_governance import main as governance_example
from tests.core.knowledge_maintenance_conformance import (
    _create_proposal_entries,
    maintenance_decision,
    maintenance_proposal,
)
from tests.core.knowledge_maintenance_governance_conformance import (
    assert_knowledge_maintenance_governance_conformance,
)
from tests.core.test_knowledge_maintenance_persistence import (
    _REVIEW_SCOPE,
    _accepted,
    _decision,
    _publisher,
)

import cayu
from cayu.knowledge_maintenance_governance import (
    KnowledgeMaintenanceGovernanceDecision,
    KnowledgeMaintenanceGovernanceDisposition,
    KnowledgeMaintenanceGovernancePolicyError,
    KnowledgeMaintenanceGovernor,
    load_knowledge_maintenance_governance_receipt,
)
from cayu.storage import (
    InMemoryKnowledgeStore,
    KnowledgeGovernanceConfig,
    KnowledgeGovernanceMode,
    KnowledgeMaintenanceDecisionKind,
    KnowledgeMaintenanceStale,
    SQLiteKnowledgeStore,
)
from cayu.storage import migrations as schema_migrations


class _Policy:
    def __init__(
        self,
        disposition: KnowledgeMaintenanceGovernanceDisposition,
        *,
        identity: str = "application-maintenance-policy",
        version: str = "3",
    ) -> None:
        self.disposition = disposition
        self.identity = identity
        self.version = version
        self.calls = 0

    async def decide_maintenance(self, request):
        self.calls += 1
        return KnowledgeMaintenanceGovernanceDecision(
            request_sha256=request.fingerprint,
            disposition=self.disposition,
            policy_identity=self.identity,
            policy_version=self.version,
            code=f"policy_{self.disposition.value}",
            annotations={"risk_tier": "bounded"},
        )


async def _publication(store: Any, prefix: str):
    request, routing, planning = await _accepted(store, prefix)
    return await _publisher(store).publish(request, routing, planning)


def _automatic_config(mode: KnowledgeGovernanceMode) -> KnowledgeGovernanceConfig:
    return KnowledgeGovernanceConfig(
        mode=mode,
        policy_identity="application-maintenance-policy",
        policy_version="3",
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_maintenance_governance_backend_conformance(backend: str, tmp_path) -> None:
    async def run() -> None:
        store = (
            InMemoryKnowledgeStore()
            if backend == "memory"
            else SQLiteKnowledgeStore(tmp_path / "maintenance-governance-conformance.db")
        )
        try:
            await assert_knowledge_maintenance_governance_conformance(
                store,
                access_scope=_REVIEW_SCOPE,
                prefix=f"{backend}-maintenance-governance-conformance",
            )
        finally:
            if isinstance(store, SQLiteKnowledgeStore):
                await store.close()

    asyncio.run(run())


def test_lost_acknowledgement_reconciles_without_repeating_policy() -> None:
    class AcknowledgementLossStore(InMemoryKnowledgeStore):
        lose_route_acknowledgement = True
        lose_decision_acknowledgement = True

        async def record_maintenance_governance_route(self, *args, **kwargs):
            receipt = await super().record_maintenance_governance_route(*args, **kwargs)
            if self.lose_route_acknowledgement:
                self.lose_route_acknowledgement = False
                raise RuntimeError("lost route acknowledgement")
            return receipt

        async def apply_maintenance_decision(self, *args, **kwargs):
            receipt = await super().apply_maintenance_decision(*args, **kwargs)
            if self.lose_decision_acknowledgement:
                self.lose_decision_acknowledgement = False
                raise RuntimeError("lost decision acknowledgement")
            return receipt

    async def run() -> None:
        store = AcknowledgementLossStore()
        reviewed = await _publication(store, "lost-route-acknowledgement")
        reviewed_governor = KnowledgeMaintenanceGovernor(
            store,
            config=KnowledgeGovernanceConfig(mode=KnowledgeGovernanceMode.REVIEWED),
        )
        with pytest.raises(RuntimeError, match="lost route acknowledgement"):
            await reviewed_governor.govern(
                operation_id="lost-route-acknowledgement-operation",
                proposal_id=reviewed.proposal.id,
                access_scope=_REVIEW_SCOPE,
            )
        route_replay = await reviewed_governor.govern(
            operation_id="lost-route-acknowledgement-operation",
            proposal_id=reviewed.proposal.id,
            access_scope=_REVIEW_SCOPE,
        )
        assert route_replay.replayed is True

        automatic = await _publication(store, "lost-decision-acknowledgement")
        policy = _Policy(KnowledgeMaintenanceGovernanceDisposition.APPROVE)
        automatic_governor = KnowledgeMaintenanceGovernor(
            store,
            config=_automatic_config(KnowledgeGovernanceMode.AUTONOMOUS),
            policy=policy,
        )
        with pytest.raises(RuntimeError, match="lost decision acknowledgement"):
            await automatic_governor.govern(
                operation_id="lost-decision-acknowledgement-operation",
                proposal_id=automatic.proposal.id,
                access_scope=_REVIEW_SCOPE,
            )
        decision_replay = await automatic_governor.govern(
            operation_id="lost-decision-acknowledgement-operation",
            proposal_id=automatic.proposal.id,
            access_scope=_REVIEW_SCOPE,
        )
        assert decision_replay.replayed is True
        assert decision_replay.maintenance_receipt is not None
        assert decision_replay.maintenance_receipt.replayed is True
        assert policy.calls == 1

    asyncio.run(run())


def test_loader_ignores_an_ordinary_unpublished_maintenance_decision() -> None:
    async def run() -> None:
        proposal = maintenance_proposal("ordinary-maintenance-decision")
        store = InMemoryKnowledgeStore(access_scope=proposal.access_scope)
        await _create_proposal_entries(store, proposal)
        decision = maintenance_decision(
            proposal,
            operation_id="ordinary-maintenance-operation",
            kind=KnowledgeMaintenanceDecisionKind.REJECT,
        )
        await store.apply_maintenance_decision(
            proposal,
            decision,
            access_scope=proposal.access_scope,
        )

        assert (
            await load_knowledge_maintenance_governance_receipt(
                store,
                operation_id=decision.operation_id,
                access_scope=proposal.access_scope,
            )
            is None
        )

    asyncio.run(run())


def test_review_route_commit_time_cannot_precede_proposal_publication() -> None:
    async def run() -> None:
        now = [datetime(2026, 9, 1, 12, tzinfo=UTC)]
        store = InMemoryKnowledgeStore(clock=lambda: now[0])
        publication = await _publication(store, "route-causal-time")
        now[0] = publication.proposal.created_at

        receipt = await KnowledgeMaintenanceGovernor(
            store,
            config=KnowledgeGovernanceConfig(mode=KnowledgeGovernanceMode.REVIEWED),
        ).govern(
            operation_id="route-causal-time-operation",
            proposal_id=publication.proposal.id,
            access_scope=_REVIEW_SCOPE,
        )

        assert receipt.committed_at == publication.receipt.committed_at

    asyncio.run(run())


def test_sqlite_restart_reconstructs_routes_and_terminal_authority_without_policy(
    tmp_path,
) -> None:
    async def run() -> None:
        database = tmp_path / "maintenance-governance-restart.db"
        original = SQLiteKnowledgeStore(database)
        reviewed = await _publication(original, "restart-reviewed-route")
        reviewed_operation = "restart-reviewed-route-operation"
        await KnowledgeMaintenanceGovernor(
            original,
            config=KnowledgeGovernanceConfig(mode=KnowledgeGovernanceMode.REVIEWED),
        ).govern(
            operation_id=reviewed_operation,
            proposal_id=reviewed.proposal.id,
            access_scope=_REVIEW_SCOPE,
        )
        automatic = await _publication(original, "restart-automatic-decision")
        automatic_operation = "restart-automatic-decision-operation"
        initial_policy = _Policy(KnowledgeMaintenanceGovernanceDisposition.APPROVE)
        await KnowledgeMaintenanceGovernor(
            original,
            config=_automatic_config(KnowledgeGovernanceMode.AUTONOMOUS),
            policy=initial_policy,
        ).govern(
            operation_id=automatic_operation,
            proposal_id=automatic.proposal.id,
            access_scope=_REVIEW_SCOPE,
        )
        assert initial_policy.calls == 1
        await original.close()

        reopened = SQLiteKnowledgeStore(database)
        try:
            route_replay = await KnowledgeMaintenanceGovernor(
                reopened,
                config=KnowledgeGovernanceConfig(mode=KnowledgeGovernanceMode.REVIEWED),
            ).govern(
                operation_id=reviewed_operation,
                proposal_id=reviewed.proposal.id,
                access_scope=_REVIEW_SCOPE,
            )
            terminal_replay = await KnowledgeMaintenanceGovernor(
                reopened,
                config=_automatic_config(KnowledgeGovernanceMode.AUTONOMOUS),
            ).govern(
                operation_id=automatic_operation,
                proposal_id=automatic.proposal.id,
                access_scope=_REVIEW_SCOPE,
            )
            assert route_replay.replayed is True
            assert terminal_replay.replayed is True
            assert terminal_replay.maintenance_receipt is not None
            assert terminal_replay.maintenance_receipt.replayed is True
        finally:
            await reopened.close()

    asyncio.run(run())


def test_stale_source_fails_at_atomic_store_boundary_without_governance_receipt() -> None:
    async def run() -> None:
        store = InMemoryKnowledgeStore()
        publication = await _publication(store, "stale-governance-source")
        source_ref = publication.proposal.sources[0]
        source = await store.get_entry(
            source_ref.entry_id,
            access_scope=_REVIEW_SCOPE,
        )
        assert source is not None
        await store.append_entry_revision(
            source.model_copy(
                update={
                    "revision": source.revision + 1,
                    "text": f"{source.text} Updated after proposal publication.",
                    "updated_at": source.updated_at + timedelta(seconds=1),
                }
            ),
            expected_revision=source.revision,
            access_scope=_REVIEW_SCOPE,
        )
        policy = _Policy(KnowledgeMaintenanceGovernanceDisposition.APPROVE)
        governor = KnowledgeMaintenanceGovernor(
            store,
            config=_automatic_config(KnowledgeGovernanceMode.AUTONOMOUS),
            policy=policy,
        )
        operation_id = "stale-governance-source-operation"
        with pytest.raises(KnowledgeMaintenanceStale) as stale_error:
            await governor.govern(
                operation_id=operation_id,
                proposal_id=publication.proposal.id,
                access_scope=_REVIEW_SCOPE,
            )
        assert stale_error.value.reason == "source_revision"
        assert policy.calls == 1
        assert (
            await load_knowledge_maintenance_governance_receipt(
                store,
                operation_id=operation_id,
                access_scope=_REVIEW_SCOPE,
            )
            is None
        )

    asyncio.run(run())


async def _assert_policy_failures_and_self_authority_fail_closed() -> None:
    store = InMemoryKnowledgeStore()
    publication = await _publication(store, "failed-governance")
    missing = KnowledgeMaintenanceGovernor(
        store,
        config=_automatic_config(KnowledgeGovernanceMode.AUTONOMOUS),
    )
    with pytest.raises(KnowledgeMaintenanceGovernancePolicyError) as missing_error:
        await missing.govern(
            operation_id="missing-policy-operation",
            proposal_id=publication.proposal.id,
            access_scope=_REVIEW_SCOPE,
        )
    assert missing_error.value.code == "policy_missing"

    forbidden_identities = (
        publication.proposal.proposed_by,
        publication.proposal.policy_id,
        publication.accepted_plan.planner_id,
        publication.accepted_plan.evaluator_id,
    )
    for index, identity in enumerate(forbidden_identities):
        self_authorizing = _Policy(
            KnowledgeMaintenanceGovernanceDisposition.APPROVE,
            identity=identity,
        )
        governor = KnowledgeMaintenanceGovernor(
            store,
            config=KnowledgeGovernanceConfig(
                mode=KnowledgeGovernanceMode.AUTONOMOUS,
                policy_identity=identity,
                policy_version="3",
            ),
            policy=self_authorizing,
        )
        operation_id = f"self-authority-operation-{index}"
        with pytest.raises(KnowledgeMaintenanceGovernancePolicyError) as authority_error:
            await governor.govern(
                operation_id=operation_id,
                proposal_id=publication.proposal.id,
                access_scope=_REVIEW_SCOPE,
            )
        assert authority_error.value.code == "policy_output_invalid"
        assert (
            await store.load_maintenance_decision(
                operation_id,
                access_scope=_REVIEW_SCOPE,
            )
            is None
        )


def test_policy_failures_and_self_authority_fail_closed() -> None:
    asyncio.run(_assert_policy_failures_and_self_authority_fail_closed())


async def _assert_malformed_timeout_and_cancellation_fail_without_attribution() -> None:
    store = InMemoryKnowledgeStore()
    publication = await _publication(store, "policy-boundary-governance")

    class WrongFingerprintPolicy:
        async def decide_maintenance(self, request):
            return KnowledgeMaintenanceGovernanceDecision(
                request_sha256="0" * 64,
                disposition=KnowledgeMaintenanceGovernanceDisposition.APPROVE,
                policy_identity="application-maintenance-policy",
                policy_version="3",
                code="wrong_request",
            )

    malformed = KnowledgeMaintenanceGovernor(
        store,
        config=_automatic_config(KnowledgeGovernanceMode.AUTONOMOUS),
        policy=WrongFingerprintPolicy(),
    )
    with pytest.raises(KnowledgeMaintenanceGovernancePolicyError) as malformed_error:
        await malformed.govern(
            operation_id="malformed-policy-operation",
            proposal_id=publication.proposal.id,
            access_scope=_REVIEW_SCOPE,
        )
    assert malformed_error.value.code == "policy_output_invalid"

    class FailingPolicy:
        async def decide_maintenance(self, request):
            raise RuntimeError("secret policy failure")

    failing = KnowledgeMaintenanceGovernor(
        store,
        config=_automatic_config(KnowledgeGovernanceMode.AUTONOMOUS),
        policy=FailingPolicy(),
    )
    with pytest.raises(KnowledgeMaintenanceGovernancePolicyError) as failing_error:
        await failing.govern(
            operation_id="failing-policy-operation",
            proposal_id=publication.proposal.id,
            access_scope=_REVIEW_SCOPE,
        )
    assert failing_error.value.code == "policy_failed"
    assert "secret policy failure" not in str(failing_error.value)

    class WrongIdentityPolicy(_Policy):
        async def decide_maintenance(self, request):
            decision = await super().decide_maintenance(request)
            return decision.model_copy(update={"policy_version": "wrong-version"})

    wrong_identity = KnowledgeMaintenanceGovernor(
        store,
        config=_automatic_config(KnowledgeGovernanceMode.AUTONOMOUS),
        policy=WrongIdentityPolicy(KnowledgeMaintenanceGovernanceDisposition.APPROVE),
    )
    with pytest.raises(KnowledgeMaintenanceGovernancePolicyError) as identity_error:
        await wrong_identity.govern(
            operation_id="wrong-identity-policy-operation",
            proposal_id=publication.proposal.id,
            access_scope=_REVIEW_SCOPE,
        )
    assert identity_error.value.code == "policy_output_invalid"

    class OversizedPolicy:
        async def decide_maintenance(self, request):
            return KnowledgeMaintenanceGovernanceDecision.model_construct(
                schema_version=1,
                request_sha256=request.fingerprint,
                disposition=KnowledgeMaintenanceGovernanceDisposition.APPROVE,
                policy_identity="application-maintenance-policy",
                policy_version="3",
                code="oversized",
                annotations={"detail": "x" * 4_097},
            )

    oversized = KnowledgeMaintenanceGovernor(
        store,
        config=_automatic_config(KnowledgeGovernanceMode.AUTONOMOUS),
        policy=OversizedPolicy(),
    )
    with pytest.raises(KnowledgeMaintenanceGovernancePolicyError) as oversized_error:
        await oversized.govern(
            operation_id="oversized-policy-operation",
            proposal_id=publication.proposal.id,
            access_scope=_REVIEW_SCOPE,
        )
    assert oversized_error.value.code == "policy_output_invalid"

    class SlowPolicy:
        async def decide_maintenance(self, request):
            await asyncio.sleep(60)

    timed_out = KnowledgeMaintenanceGovernor(
        store,
        config=KnowledgeGovernanceConfig(
            mode=KnowledgeGovernanceMode.AUTONOMOUS,
            policy_identity="application-maintenance-policy",
            policy_version="3",
            policy_timeout_seconds=0.001,
        ),
        policy=SlowPolicy(),
    )
    with pytest.raises(KnowledgeMaintenanceGovernancePolicyError) as timeout_error:
        await timed_out.govern(
            operation_id="timed-out-policy-operation",
            proposal_id=publication.proposal.id,
            access_scope=_REVIEW_SCOPE,
        )
    assert timeout_error.value.code == "policy_timed_out"

    entered = asyncio.Event()

    class BlockingPolicy:
        async def decide_maintenance(self, request):
            entered.set()
            await asyncio.Event().wait()

    cancelled = KnowledgeMaintenanceGovernor(
        store,
        config=_automatic_config(KnowledgeGovernanceMode.AUTONOMOUS),
        policy=BlockingPolicy(),
    )
    task = asyncio.create_task(
        cancelled.govern(
            operation_id="cancelled-policy-operation",
            proposal_id=publication.proposal.id,
            access_scope=_REVIEW_SCOPE,
        )
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    for operation_id in (
        "malformed-policy-operation",
        "failing-policy-operation",
        "wrong-identity-policy-operation",
        "oversized-policy-operation",
        "timed-out-policy-operation",
        "cancelled-policy-operation",
    ):
        assert (
            await store.load_maintenance_decision(
                operation_id,
                access_scope=_REVIEW_SCOPE,
            )
            is None
        )
        assert (
            await store.load_maintenance_governance_route(
                operation_id,
                access_scope=_REVIEW_SCOPE,
            )
            is None
        )


def test_malformed_timeout_and_cancellation_fail_without_attribution() -> None:
    asyncio.run(_assert_malformed_timeout_and_cancellation_fail_without_attribution())


def test_public_maintenance_governance_exports_are_stable() -> None:
    for name in (
        "KnowledgeMaintenanceGovernanceAuthority",
        "KnowledgeMaintenanceGovernanceDecision",
        "KnowledgeMaintenanceGovernanceDisposition",
        "KnowledgeMaintenanceGovernancePolicy",
        "KnowledgeMaintenanceGovernancePolicyError",
        "KnowledgeMaintenanceGovernanceReceipt",
        "KnowledgeMaintenanceGovernanceRequest",
        "KnowledgeMaintenanceGovernor",
        "decide_knowledge_maintenance_governance",
        "load_knowledge_maintenance_governance_receipt",
        "prepare_knowledge_maintenance_governance_request",
    ):
        assert name in cayu.__all__
        assert getattr(cayu, name) is not None


def test_deterministic_example_covers_all_governance_modes(capsys) -> None:
    asyncio.run(governance_example())
    assert json.loads(capsys.readouterr().out) == {
        "autonomous": "reject",
        "policy_automatic": "approve",
        "provider_calls": 0,
        "reviewed": "route_to_review",
    }


def test_sqlite_revision_77_does_not_infer_governance_for_reviewed_history(
    tmp_path,
) -> None:
    database = tmp_path / "revision-77-no-inferred-governance.db"

    async def seed() -> tuple[str, str]:
        store = SQLiteKnowledgeStore(database)
        try:
            publication = await _publication(store, "pre-governance-reviewed")
            decision = _decision(
                publication.proposal,
                kind=KnowledgeMaintenanceDecisionKind.REJECT,
                suffix="pre-governance-reviewed",
            )
            await store.apply_maintenance_decision(
                publication.proposal,
                decision,
                access_scope=_REVIEW_SCOPE,
            )
            return publication.proposal.id, decision.operation_id
        finally:
            await store.close()

    proposal_id, decision_operation_id = asyncio.run(seed())
    with sqlite3.connect(database) as connection:
        connection.execute("DROP TABLE cayu_knowledge_semantic_watch_receipts")
        connection.execute("DROP TABLE cayu_knowledge_maintenance_governance_routes")
        connection.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 77")
        connection.execute("PRAGMA user_version = 76")
        connection.commit()

    migrated = SQLiteKnowledgeStore(
        database,
        schema_mode=schema_migrations.SchemaMode.MIGRATE,
    )

    async def verify() -> None:
        try:
            assert (
                await migrated.load_maintenance_decision(
                    decision_operation_id,
                    access_scope=_REVIEW_SCOPE,
                )
                is not None
            )
            assert (
                await migrated.load_maintenance_governance_route(
                    decision_operation_id,
                    access_scope=_REVIEW_SCOPE,
                )
                is None
            )
            assert (
                migrated._connection.execute(
                    "SELECT COUNT(*) FROM cayu_knowledge_maintenance_governance_routes"
                ).fetchone()[0]
                == 0
            )
            assert (
                await migrated.load_maintenance_proposal(
                    proposal_id,
                    access_scope=_REVIEW_SCOPE,
                )
                is not None
            )
        finally:
            await migrated.close()

    asyncio.run(verify())


def test_sqlite_revision_77_rejects_malformed_governance_storage(tmp_path) -> None:
    database = tmp_path / "revision-77-malformed-governance.db"
    store = SQLiteKnowledgeStore(database, access_scope=_REVIEW_SCOPE)
    store._connection.close()
    with sqlite3.connect(database) as connection:
        connection.execute("DROP TABLE cayu_knowledge_maintenance_governance_routes")
        connection.execute(
            "CREATE TABLE cayu_knowledge_maintenance_governance_routes "
            "(operation_id TEXT PRIMARY KEY)"
        )
        connection.commit()

    with pytest.raises(RuntimeError, match="maintenance-governance contract"):
        SQLiteKnowledgeStore(database, access_scope=_REVIEW_SCOPE)
