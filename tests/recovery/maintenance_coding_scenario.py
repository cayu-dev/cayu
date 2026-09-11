"""Generated SQLite application loss; local runner, never live Docker proof."""

import asyncio
import importlib
import importlib.util
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from tests.qualification.repository_maintenance_delivery_case import build_journey_http
from worker_harness import _wait_for_task_lease_expiry, _write_json_atomic

from cayu import (
    BudgetPolicy,
    DockerCodingToolchainProfile,
    ExecutionProfileBehaviorIdentity,
    ModelStreamEvent,
    ResolutionActor,
    ResolutionActorSource,
    ScriptedModelProvider,
    TaskCancellationReconciliationEvent,
    TaskCancellationReconciliationEvidence,
    TaskCancellationReconciliationOutcome,
    TaskCancellationReconciliationRequest,
    TaskClaimLost,
    TaskQuery,
    TaskStatus,
    TaskTerminalizationConflict,
    complete_managed_task,
    run_task_worker,
)
from cayu.cli.project import project_context
from cayu.runtime import (
    RecoveryDecision,
    RecoveryExecutionRequest,
    RecoveryItemExecutionStatus,
    RecoveryPlanAction,
    RecoveryPlanRequest,
    RecoveryPlanSelection,
)
from cayu.storage import SQLiteBudgetLedger


class _Provider(ScriptedModelProvider):
    @property
    def execution_profile_identity(self):
        return ExecutionProfileBehaviorIdentity(
            name="tests:maintenance-local-coding-loss-provider",
            behavior_version="1",
            implementation_version="1",
        )

    def __init__(self, config):
        super().__init__([])
        self.config = config
        self.identity = None

    async def stream(self, request):
        assert self.config["action"] == "start", "Recovery dispatched a model"
        self.requests.append(request)
        assert self.identity is not None
        _write_json_atomic(
            Path(self.config["phase_path"]),
            {"phase": "maintenance_model_dispatched", "identity": self.identity},
        )
        await asyncio.Event().wait()
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


async def run_maintenance_coding(config):
    project = Path(config["project"])
    with project_context(project), pytest.MonkeyPatch.context() as patch:
        operations = importlib.import_module("operations.coding")
        profile = DockerCodingToolchainProfile.model_validate_json(config["toolchain_json"])
        patch.setattr(
            operations, "_configured_docker_authority", lambda root: (profile, "/usr/bin/docker")
        )
        spec = importlib.util.spec_from_file_location(
            "maintenance_coding_fixture", project / "tests/test_coding_composition.py"
        )
        assert spec is not None and spec.loader is not None
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        runners = []
        fixture._install_fake_docker_factory(
            patch, target=Path(config["target"]), created_runners=runners
        )
        provider = _Provider(config)
        application = importlib.import_module("app").build_coding_product_application(
            provider=provider,
            workspace_root=Path(config["source"]),
            budget_policy=BudgetPolicy.model_validate_json(config["budget_json"]),
            budget_ledger=SQLiteBudgetLedger(config["budget_path"]),
        )
        store = application.app.task_store
        try:
            reservations = importlib.import_module(
                "operations.maintenance_runs"
            ).SQLiteMaintenanceRunStore(config["reservation_path"])
            api = build_journey_http(
                application, reservations, tenant="coding-loss", subject="operator"
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=api), base_url="http://fixture"
            ) as client:
                if config["action"] == "start":
                    await reservations.initialize()
                    body = {
                        "instruction": "Repair the inclusive endpoint.",
                        "idempotency_key": "coding-loss",
                    }
                    headers = {"authorization": "Bearer fixture-product"}
                    response = await client.post("/runs", headers=headers, json=body)
                    assert response.status_code == 202, response.text
                    assert (
                        await client.post("/runs", headers=headers, json=body)
                    ).json() == response.json()
                    identity = await reservations.load_owned(
                        tenant="coding-loss", public_id=response.json()["id"]
                    )
                    assert identity is not None
                    provider.identity = identity.model_dump(mode="json")
                    worker = importlib.import_module("operations.maintenance_worker")

                    async def handle(_app, task, worker_id):
                        await worker.handle_coding_task(application, reservations, task, worker_id)

                    await run_task_worker(
                        application.app,
                        store,
                        handle,
                        worker_id="maintenance.coding-" + uuid4().hex,
                        query=TaskQuery(type="maintenance.coding"),
                        lease_seconds=3,
                        poll_interval_s=0.01,
                        max_tasks=1,
                        recover_interrupted_handoffs=False,
                    )
                    raise AssertionError("Worker returned before its SIGKILL barrier")
                identity = await reservations.load_owned(
                    tenant="coding-loss", public_id=config["identity"]["public_id"]
                )
                assert (
                    identity is not None and identity.model_dump(mode="json") == config["identity"]
                )
                await _wait_for_task_lease_expiry(store, identity.task_id)
                query = TaskQuery(type="maintenance.coding")
                assert await store.reclaim_expired(query=query) == []
                assert await store.claim_task("replacement", query, lease_seconds=3) is None
                held = await store.load_task(identity.task_id)
                assert held is not None and held.status_reason == "cancellation_requested"
                assert held.status is TaskStatus.CLAIMED and held.session_id is None
                with pytest.raises((TaskClaimLost, TaskTerminalizationConflict)):
                    await complete_managed_task(store, held, held.worker_id, {"stale": True})
                assert await store.load_task(held.id) == held
                headers = {"authorization": "Bearer fixture-operator"}
                path = f"/operator/runs/{identity.public_id}"
                response = await client.get(
                    path + "/tasks", headers=headers, params={"tenant": "coding-loss"}
                )
                assert response.status_code == 200, response.text
                observed = response.json()["phases"]["coding"]
                assert observed["cancellation_marker"] == "requested"
                assert observed["recorded_owner"]["id"] == held.worker_id
                assert observed["lease_observation"] == "expired"
                plan = await application.app.plan_recovery(
                    RecoveryPlanRequest(
                        selection=RecoveryPlanSelection(session_ids=(identity.session_id,))
                    )
                )
                assert len(plan.items) == 1
                item = plan.items[0]
                assert RecoveryPlanAction.MODEL_MARK_INTERRUPTED in item.allowed_actions, item
                execution = RecoveryExecutionRequest(
                    plan=plan,
                    execution_id="maintenance-coding-loss",
                    decisions=(
                        RecoveryDecision(
                            item_id=item.item_id, action=RecoveryPlanAction.MODEL_MARK_INTERRUPTED
                        ),
                    ),
                )
                receipt = await application.app.execute_recovery(execution)
                assert receipt.items[0].status is RecoveryItemExecutionStatus.EXECUTED
                replay = await application.app.execute_recovery(execution)
                assert replay.items[0].replayed
                assert replay.items[0].receipt_event_id == receipt.items[0].receipt_event_id
                results = importlib.import_module("operations.maintenance_results")
                task = results.coding_task_from_identity(identity)
                publication = await application.recover_settled(task)
                assert await application.recover_settled(task) == publication
                assert publication.candidate.state.value in {"cancelled", "reconstruction_required"}
                # Product recovery requires native release evidence. This fixture's
                # old process was killed/reaped and allocated no external resources.
                # None of those facts is proof about real Docker/provider cleanup.
                event = TaskCancellationReconciliationEvent.model_validate(
                    held.status_payload["event"]
                )
                now = datetime.now(UTC)
                request = TaskCancellationReconciliationRequest(
                    task_id=held.id,
                    original_worker_id=held.worker_id,
                    original_lease_expires_at=held.lease_expires_at,
                    cancellation_requested_at=event.occurred_at,
                    cancellation_idempotency_key=held.status_payload[
                        "terminalization_idempotency_key"
                    ],
                    reconciliation_idempotency_key="maintenance-local-coding-loss",
                    reconciliation_requested_at=now,
                    reconciled_by=ResolutionActor(
                        subject="local-process-harness", source=ResolutionActorSource.SYSTEM
                    ),
                    evidence=TaskCancellationReconciliationEvidence(
                        outcome=TaskCancellationReconciliationOutcome.QUIESCENT,
                        validator_id="local-process-harness",
                        validator_version="1",
                        evidence_id=identity.product_run_id,
                        evidence_sha256=publication.candidate.digest,
                        validated_at=now,
                        execution_profile_fingerprint=held.metadata.get(
                            "execution_profile_fingerprint"
                        ),
                        effect_fingerprint=held.metadata.get("effect_fingerprint"),
                    ),
                    expected_execution_profile_fingerprint=held.metadata.get(
                        "execution_profile_fingerprint"
                    ),
                    expected_effect_fingerprint=held.metadata.get("effect_fingerprint"),
                )
                settlement = await store.reconcile_task_cancellation(request)
                assert await store.reconcile_task_cancellation(request) == settlement
                terminal = await store.load_task(held.id)
                assert terminal.status is TaskStatus.CANCELLED and terminal.result is None
                assert terminal.worker_id is None and terminal.lease_expires_at is None
                with pytest.raises((TaskClaimLost, TaskTerminalizationConflict)):
                    await complete_managed_task(store, held, held.worker_id, {"stale": True})
                assert await store.load_task(held.id) == terminal
                with pytest.raises(results.MaintenanceResultUnavailable):
                    await results.load_verified_coding_result(application, reservations, identity)
                assert not provider.requests and not runners
                return {
                    "identity": identity.model_dump(mode="json"),
                    "state": publication.candidate.state.value,
                    "task_status": terminal.status.value,
                    "provider_calls": 0,
                    "result_digest": publication.candidate.digest,
                }
        finally:
            for name in (
                "drain_background_interruptions",
                "drain_recovery_cleanups",
                "drain_provider_operation_cancellations",
                "drain_environment_cleanups",
                "drain_knowledge_publications",
            ):
                assert await getattr(application.app, name)() is True
            for owned in (
                application.app.session_store,
                store,
                application.app.knowledge_store,
                application.app.budget_ledger,
            ):
                await owned.close()
