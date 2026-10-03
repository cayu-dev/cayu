"""Application-owned settlement example, not a Runtime recovery implementation.

See ``cayu guide authoring#coding-product-host``. A deployment authenticates the
operator before calling this module. Native registered recovery runs first;
this recipe never resumes a session, executes a tool, or performs delivery.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, StrictStr, field_validator

from cayu import (
    CodingProductRequest,
    CodingProductRunner,
    CodingProductState,
    ResolutionActor,
    TaskCancellationReconciliationEvidence,
    TaskCancellationReconciliationOutcome,
    TaskCancellationReconciliationRequest,
    TaskStatus,
)
from cayu.budgets.pricing import PriceBook, copy_price_book
from cayu.guides.coding_host_evidence import _require_serial_check_quiescence
from cayu.guides.coding_host_owner import inspect_stopped_worker as inspect_docker_worker
from cayu.workspaces.revisions import (
    WorkspaceRevisionObservationStatus,
    observe_deterministic_workspace,
)


class SettlementConflict(ValueError):
    """Retained business identity/evidence cannot be replaced by a retry."""


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class Reservation(BaseModel):
    """Trusted application intake record; never accept this from an HTTP body."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)
    tenant: StrictStr
    public_id: StrictStr
    request: CodingProductRequest
    pricing: PriceBook
    cost_basis: Literal["observed", "synthetic"]

    @field_validator("tenant", "public_id")
    @classmethod
    def clean_identity(cls, value):
        if (
            not value
            or len(value) > 256
            or value.strip() != value
            or any(ord(c) < 32 or ord(c) == 127 for c in value)
        ):
            raise ValueError("Invalid application identity.")
        return value


def _reservation(value: Reservation) -> Reservation:
    if type(value) is not Reservation or type(value.request) is not CodingProductRequest:
        raise SettlementConflict("Invalid application reservation type.")
    # Revalidate after reconstruction/model_copy without serializer warnings that
    # could print a rejected value. This is still not HTTP authentication.
    return Reservation(
        tenant=value.tenant,
        public_id=value.public_id,
        pricing=copy_price_book(value.pricing),
        cost_basis=value.cost_basis,
        request=CodingProductRequest.model_validate(
            value.request.model_dump(mode="python", warnings=False)
        ),
    )


def _require_cost_binding(expected, pricing, cost_basis):
    if cost_basis != expected.cost_basis or copy_price_book(pricing) != expected.pricing:
        raise SettlementConflict("Original cost evidence configuration changed.")


class BusinessStore:
    """Bounded SQLite example; Runtime stores remain independent owners.

    Synchronous short transactions deliberately have no await between CAS and
    commit. A production async adapter must own cancellation through commit.
    The original reservation remains readable after its source fence is released.
    No schema upgrade or multi-host PostgreSQL adapter is implied by this recipe.
    """

    def __init__(self, path: Path):
        self.path = path
        with self._transaction() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS coding_host ("
                "tenant TEXT NOT NULL, public_id TEXT NOT NULL, reservation TEXT NOT NULL, "
                "source TEXT NOT NULL, pending TEXT, result TEXT, "
                "PRIMARY KEY (tenant, public_id))"
            )
            db.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS coding_host_source "
                "ON coding_host(source) WHERE result IS NULL"
            )

    @contextmanager
    def _transaction(self):
        db = sqlite3.connect(self.path, timeout=1)
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def reserve(self, expected: Reservation) -> None:
        expected = _reservation(expected)
        encoded = _json(expected.model_dump(mode="json"))
        with self._transaction() as db:
            row = db.execute(
                "SELECT reservation FROM coding_host WHERE tenant=? AND public_id=?",
                (expected.tenant, expected.public_id),
            ).fetchone()
            if row is not None:
                if row[0] != encoded:
                    raise SettlementConflict("Business reservation changed.")
                return
            try:
                db.execute(
                    "INSERT INTO coding_host VALUES (?, ?, ?, ?, NULL, NULL)",
                    (
                        expected.tenant,
                        expected.public_id,
                        encoded,
                        expected.request.source.workspace_id,
                    ),
                )
            except sqlite3.IntegrityError:
                raise SettlementConflict("Source already has an unsettled owner.") from None

    def _row(self, db, expected):
        expected = _reservation(expected)
        row = db.execute(
            "SELECT reservation, pending, result FROM coding_host WHERE tenant=? AND public_id=?",
            (expected.tenant, expected.public_id),
        ).fetchone()
        if row is None or row[0] != _json(expected.model_dump(mode="json")):
            raise SettlementConflict("Original business reservation is absent or conflicting.")
        return row

    def read(self, expected: Reservation) -> dict | None:
        """Historical application result, not an observation of today's source."""
        with self._transaction() as db:
            result = self._row(db, expected)[2]
            return None if result is None else json.loads(result)

    def pending(self, expected: Reservation) -> TaskCancellationReconciliationRequest | None:
        with self._transaction() as db:
            pending = self._row(db, expected)[1]
            return (
                None
                if pending is None
                else TaskCancellationReconciliationRequest.model_validate_json(pending)
            )

    def prepare(self, expected: Reservation, request: TaskCancellationReconciliationRequest):
        """Persist the full native operation BEFORE calling its transaction owner."""
        expected = _reservation(expected)
        if type(request) is not TaskCancellationReconciliationRequest:
            raise SettlementConflict("Invalid cancellation request type.")
        request = TaskCancellationReconciliationRequest.model_validate(
            request.model_dump(mode="python", warnings=False)
        )
        if request.task_id != expected.request.task.task_id:
            raise SettlementConflict("Cancellation belongs to another task.")
        encoded = _json(request.model_dump(mode="json"))
        with self._transaction() as db:
            _, pending, result = self._row(db, expected)
            if result is not None or (pending is not None and pending != encoded):
                raise SettlementConflict("Cancellation selection changed.")
            db.execute(
                "UPDATE coding_host SET pending=? WHERE tenant=? AND public_id=?",
                (encoded, expected.tenant, expected.public_id),
            )

    def settle(self, expected: Reservation, result: dict):
        """One atomic result publication/source release; exact replay cannot free a newer owner."""
        encoded = _json(result)
        with self._transaction() as db:
            old = self._row(db, expected)[2]
            if old is not None:
                if old != encoded:
                    raise SettlementConflict("Application settlement changed.")
                return json.loads(old)
            db.execute(
                "UPDATE coding_host SET result=? WHERE tenant=? AND public_id=?",
                (encoded, expected.tenant, expected.public_id),
            )
        return json.loads(encoded)


async def project_result(
    runner: CodingProductRunner,
    reservation: Reservation,
    *,
    pricing: PriceBook,
    cost_basis: Literal["observed", "synthetic"],
    result_digest: str | None,
    cancellation_receipt: str | None = None,
) -> dict:
    """One projection used by normal and reconstructed paths; no provider call.

    cost_basis is trusted deployment configuration, never inferred from missing
    usage. Synthetic fixture usage is not an observed provider invoice.
    """
    reservation = _reservation(reservation)
    _require_cost_binding(reservation, pricing, cost_basis)
    if type(cost_basis) is not str or cost_basis not in {"observed", "synthetic"}:
        raise ValueError("An explicit cost basis is required.")
    request = reservation.request
    if (result_digest is None) == (cancellation_receipt is None):
        raise SettlementConflict("Exactly one native terminal outcome is required.")
    revision = None
    if result_digest is not None:
        publication = await runner.repository.load_publication(
            request_fingerprint=request.fingerprint, digest=result_digest
        )
        candidate = publication.candidate
        if (
            candidate.task_id != request.task.task_id
            or candidate.session_id != request.session_id
            or candidate.product_run_id != request.product_run_id
            or candidate.state is not CodingProductState.PATCH_READY_FOR_DELIVERY
        ):
            raise SettlementConflict("Retained product is not the exact accepted result.")
        revision = candidate.final_revision
    price_book = copy_price_book(reservation.pricing)
    cost: dict[str, Any] = {
        "basis": cost_basis,
        "availability": "unavailable",
        "estimated_total": None,
        "currency": "USD",
        "billing_completeness": "not_established",
        "pricing_sha256": sha256(_json(price_book.model_dump(mode="json")).encode()).hexdigest(),
    }
    causal = runner.app.project_causal_budget_id_for_exposure(
        request.causal_budget_id or request.session_id, session_ids=(request.session_id,)
    )
    try:
        summary = await runner.app.get_causal_budget_cost(causal, price_book, currency="USD")
    except KeyError:
        pass
    else:
        if summary.causal_budget_id != causal or summary.currency != "USD":
            raise SettlementConflict("Cost evidence belongs to another scope.")
        cost.update(
            availability="recorded" if summary.line_items else "no_observations",
            estimated_total=str(summary.total_cost) if summary.line_items else None,
            model_steps=summary.model_steps,
            unpriced_model_steps=summary.unpriced_model_steps,
            missing_usage_model_steps=summary.missing_usage_model_steps,
            auxiliary_attempts=summary.auxiliary_attempts,
            unpriced_auxiliary_attempts=summary.unpriced_auxiliary_attempts,
            observed_line_items=len(summary.line_items),
            unpriced_line_items=sum(not item.priced for item in summary.line_items),
        )
    return {
        "id": reservation.public_id,
        "task_id": request.task.task_id,
        "session_id": request.session_id,
        "request_fingerprint": request.fingerprint,
        "classification": "cancelled" if cancellation_receipt else "historical_patch_ready",
        "result_digest": result_digest,
        "final_revision": revision,
        "cancellation_receipt": cancellation_receipt,
        "cost": cost,
        "external_delivery_performed": False,
    }


async def settle_completed(
    runner: CodingProductRunner,
    business: BusinessStore,
    expected: Reservation,
    *,
    result_digest: str,
    pricing: PriceBook,
    cost_basis: Literal["observed", "synthetic"],
):
    """Shared normal/completion-ack-loss entrance; never repeats the handler.

    The example's managed handler completes with precisely product_run_id and
    result_digest. Other application task result schemas need their own validator.
    This is patch readiness, not independent domain acceptance or Git delivery.
    """
    expected = _reservation(expected)
    _require_cost_binding(expected, pricing, cost_basis)
    previous = business.read(expected)
    if business.pending(expected) is not None:
        raise SettlementConflict("Cancellation owns this application settlement.")
    task_store = runner.app.task_store
    if task_store is None:
        raise SettlementConflict("Application has no task store.")
    task = await task_store.load_task(expected.request.task.task_id)
    if (
        task is None
        or task.status is not TaskStatus.COMPLETED
        or task.result
        != {
            "product_run_id": expected.request.product_run_id,
            "result_digest": result_digest,
        }
    ):
        raise SettlementConflict("Exact managed task completion is unavailable.")
    if previous is not None:
        if previous["result_digest"] != result_digest:
            raise SettlementConflict("Historical completion identity changed.")
        return previous
    result = await project_result(
        runner,
        expected,
        pricing=pricing,
        cost_basis=cost_basis,
        result_digest=result_digest,
    )
    return business.settle(expected, result)


async def require_current_source(
    runner: CodingProductRunner, reservation: Reservation, result: dict
):
    """Separate new-delivery gate, under the application's exclusive source owner.

    Historical readback does not call this. Delivery must retain its own exact
    source/approval fence through publication; this observation grants no push.
    """
    reservation = _reservation(reservation)
    request = reservation.request
    publication = await runner.repository.load_publication(
        request_fingerprint=request.fingerprint, digest=result["result_digest"]
    )
    if (
        runner.source_workspace.id != request.source.workspace_id
        or result.get("request_fingerprint") != request.fingerprint
        or result.get("task_id") != request.task.task_id
        or result.get("session_id") != request.session_id
        or result.get("id") != reservation.public_id
        or result.get("classification") != "historical_patch_ready"
        or result.get("final_revision") != publication.candidate.final_revision
        or publication.candidate.state is not CodingProductState.PATCH_READY_FOR_DELIVERY
    ):
        raise SettlementConflict("Current delivery result authority conflicts.")
    observed = await observe_deterministic_workspace(
        runner.source_workspace,
        observer="cayu-coding-product-source",
        limits=request.source.observation_limits,
    )
    if (
        observed.status is not WorkspaceRevisionObservationStatus.SUPPORTED
        or observed.revision != publication.candidate.final_revision
    ):
        raise SettlementConflict("Current source bytes differ from the historical accepted result.")


async def settle_cancelled(
    runner: CodingProductRunner,
    business: BusinessStore,
    expected: Reservation,
    *,
    actor: ResolutionActor,
    reconciliation_id: str,
    inspect_stopped_worker=inspect_docker_worker,
    pricing: PriceBook,
    cost_basis: Literal["observed", "synthetic"],
):
    """Operator entrance after registered native recovery, never a retry engine.

    inspect_stopped_worker is the deployment's authenticated generation observer,
    not a JSON field or user callback. It must reject unknown/live generations.
    Native invocation/effect evidence is independently required below.
    """
    expected = _reservation(expected)
    _require_cost_binding(expected, pricing, cost_basis)
    if type(actor) is not ResolutionActor:
        raise SettlementConflict("Authenticated operator identity is required.")
    actor = ResolutionActor.model_validate(actor.model_dump(mode="python", warnings=False))
    if actor.tenant != expected.tenant:
        raise SettlementConflict("Operator tenant does not own this reservation.")
    task_store = runner.app.task_store
    if task_store is None:
        raise SettlementConflict("Application has no task store.")
    previous = business.read(expected)
    request = business.pending(expected)
    if request is None:
        if previous is not None:
            raise SettlementConflict("Business request already has another outcome.")
        task = await task_store.load_task(expected.request.task.task_id)
        if (
            task is None
            or task.status not in {TaskStatus.CLAIMED, TaskStatus.RUNNING}
            or task.status_reason != "cancellation_requested"
            or task.started_at is None
            or task.lease_expires_at is None
            or task.lease_expires_at > datetime.now(UTC)
            or task.interrupted_handoff_id is not None
            or task.worker_id is None
            or task.status_payload is None
        ):
            raise SettlementConflict("Original cancellation is not ready for reconciliation.")
        owner = await inspect_stopped_worker(task.worker_id)
        if (
            type(owner) is not dict
            or owner.get("worker_id") != task.worker_id
            or type(owner.get("container_id")) is not str
            or len(owner["container_id"]) != 64
            or any(c not in "0123456789abcdef" for c in owner["container_id"])
            or type(owner.get("generation")) is not str
            or len(owner["generation"]) != 64
            or any(c not in "0123456789abcdef" for c in owner["generation"])
        ):
            raise SettlementConflict("Stopped worker generation evidence is unavailable.")
        saved = await runner.repository.load_request(
            expected.request.product_run_id, session_id=expected.request.session_id
        )
        if saved != expected.request:
            raise SettlementConflict("Original admitted request changed.")
        inspected = await runner.inspect_settled_execution(saved)
        _require_serial_check_quiescence(
            inspected.events, tool_call_ordinals=inspected.tool_call_ordinals
        )
        now = datetime.now(UTC)
        evidence = {
            "reservation": expected.model_dump(mode="json"),
            "owner": owner,
            "release": inspected.release_fingerprint,
            "events": [event.model_dump(mode="json") for event in inspected.events],
            "ordinals": inspected.tool_call_ordinals,
        }
        request = TaskCancellationReconciliationRequest(
            task_id=task.id,
            original_worker_id=task.worker_id,
            original_lease_expires_at=task.lease_expires_at,
            cancellation_requested_at=task.status_payload["event"]["occurred_at"],
            cancellation_idempotency_key=task.status_payload["terminalization_idempotency_key"],
            reconciliation_idempotency_key=reconciliation_id,
            reconciliation_requested_at=now,
            reconciled_by=actor,
            evidence=TaskCancellationReconciliationEvidence(
                outcome=TaskCancellationReconciliationOutcome.QUIESCENT,
                validator_id="coding-host-example",
                validator_version="1",
                evidence_id=expected.public_id,
                evidence_sha256=sha256(_json(evidence).encode()).hexdigest(),
                validated_at=now,
            ),
        )
        business.prepare(expected, request)
    if (
        request.reconciled_by != actor
        or request.reconciliation_idempotency_key != reconciliation_id
    ):
        raise SettlementConflict("Exact cancellation replay authority changed.")
    receipt = await task_store.reconcile_task_cancellation(request)
    if previous is not None:
        if previous["cancellation_receipt"] != receipt.reconciliation.events[-1].id:
            raise SettlementConflict("Native receipt changed after application settlement.")
        return previous
    result = await project_result(
        runner,
        expected,
        pricing=pricing,
        cost_basis=cost_basis,
        result_digest=None,
        cancellation_receipt=receipt.reconciliation.events[-1].id,
    )
    return business.settle(expected, result)


def extend_generated_application(
    application,
    *,
    business: BusinessStore,
    tenant: str,
    pricing: PriceBook,
    cost_basis: Literal["observed", "synthetic"],
):
    """Extend the generated project-owned admission method, not Runtime internals.

    Call from the generated project, after its ordinary factory constructs the
    application. This single-tenant example receives tenant from authenticated
    deployment configuration. It is not a multi-tenant HTTP authenticator.
    """
    from workflows.coding_product import CodingProductApplication  # ty: ignore[unresolved-import]

    if type(application) is not CodingProductApplication:
        raise TypeError("Expected the generated CodingProductApplication.")
    configured_pricing = copy_price_book(pricing)

    def reserved(request):
        return Reservation(
            tenant=tenant,
            public_id=request.product_run_id,
            request=request,
            pricing=configured_pricing,
            cost_basis=cost_basis,
        )

    class ReservedApplication:
        # Keep the original generated object as the sole preparation/registration
        # owner. Reconstructing it around the same CayuApp would duplicate verifier
        # registration and split its process-local registration bookkeeping.
        app = application.app

        async def run(self, task):
            runner, request, run_request = await self._prepare(task, require_existing=False)
            expected = reserved(request)
            if business.read(expected) is not None:
                return await runner.recover_settled_execution(
                    request,
                    review_settlement=task.review_settlement,
                )
            return await runner.run(request, run_request, review_settlement=task.review_settlement)

        async def _prepare(self, task, *, require_existing):
            runner, request, run_request = await application._prepare(
                task, require_existing=require_existing
            )
            expected = reserved(request)
            if require_existing:
                # Missing business provenance must never be synthesized by recovery.
                business.read(expected)
            else:
                # Native preparation is read-only with respect to source execution.
                # Commit the source fence before runner.run can dispatch anything.
                business.reserve(expected)
            return runner, request, run_request

        async def recover_settled(self, task):
            runner, request, _ = await self._prepare(task, require_existing=True)
            return await runner.recover_settled_execution(
                request,
                review_settlement=task.review_settlement,
            )

        async def settle_completed(self, task, *, result_digest, pricing, cost_basis):
            runner, request, _ = await self._prepare(task, require_existing=True)
            return await settle_completed(
                runner,
                business,
                reserved(request),
                result_digest=result_digest,
                pricing=pricing,
                cost_basis=cost_basis,
            )

        async def settle_cancelled(
            self,
            task,
            *,
            actor,
            reconciliation_id,
            inspect_stopped_worker=inspect_docker_worker,
            pricing,
            cost_basis,
        ):
            runner, request, _ = await self._prepare(task, require_existing=True)
            return await settle_cancelled(
                runner,
                business,
                reserved(request),
                actor=actor,
                reconciliation_id=reconciliation_id,
                inspect_stopped_worker=inspect_stopped_worker,
                pricing=pricing,
                cost_basis=cost_basis,
            )

    return ReservedApplication()
