from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from contextlib import asynccontextmanager

import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient
from pydantic import ValidationError

from cayu import (
    REDACTED_SECRET,
    AgentSpec,
    CayuApp,
    InMemorySessionStore,
    InMemoryTaskStore,
    Message,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
    SecretRedactor,
    SessionStatus,
    SQLiteSessionStore,
    SQLiteTaskStore,
    TaskCreate,
    TaskStatus,
    run_to_completion,
)
from cayu.server import (
    AuthenticatedAccess,
    AuthenticatedProductAccess,
    BasicAuth,
    CayuService,
    PlaceholderOperatorAccess,
    PlaceholderProductAccess,
    ProductExecutionClaimLost,
    ProductIdempotencyConflict,
    ProductOperation,
    ProductOperationExecutionClaim,
    ProductOperationReservation,
    ProductOperationSettlementConflict,
    ProductPrincipal,
    ServiceIdentityStoreKind,
    ServiceMode,
    create_agent_service,
)
from cayu.server.service import (
    MAX_PRODUCT_REQUEST_BYTES,
    MAX_PUBLIC_RESULT_CHARS,
    _is_maintained_service,
    _product_auth_dependency,
)


class MemoryProductStore:
    category = ServiceIdentityStoreKind.DURABLE

    def __init__(self) -> None:
        self.by_public_id: dict[str, ProductOperation] = {}
        self.by_work_id: dict[str, ProductOperation] = {}
        self.by_idempotency_key: dict[str, ProductOperation] = {}
        self.claimed_work_ids: list[str] = []
        self.execution_claims: dict[str, tuple[str, float]] = {}

    async def reserve(
        self,
        *,
        tenant_id: str,
        idempotency_key: str,
        request_fingerprint: str,
        public_id: str,
        work_id: str,
        session_id: str,
        task_id: str,
        request_text: str,
    ) -> ProductOperationReservation:
        existing = self.by_idempotency_key.get(idempotency_key)
        if existing is not None:
            if (
                existing.tenant_id != tenant_id
                or existing.request_fingerprint != request_fingerprint
            ):
                raise ProductIdempotencyConflict
            return ProductOperationReservation(operation=existing, created=False)
        operation = ProductOperation(
            tenant_id=tenant_id,
            public_id=public_id,
            work_id=work_id,
            idempotency_key=idempotency_key,
            request_fingerprint=request_fingerprint,
            session_id=session_id,
            task_id=task_id,
            request_text=request_text,
            status="pending",
            result=None,
        )
        self.by_public_id[public_id] = operation
        self.by_work_id[work_id] = operation
        self.by_idempotency_key[idempotency_key] = operation
        return ProductOperationReservation(operation=operation, created=True)

    async def find(self, *, tenant_id: str, public_id: str) -> ProductOperation | None:
        operation = self.by_public_id.get(public_id)
        return operation if operation is not None and operation.tenant_id == tenant_id else None

    async def claim_execution(
        self,
        *,
        work_id: str,
        claim_id: str,
        lease_seconds: int,
    ) -> ProductOperationExecutionClaim | None:
        self.claimed_work_ids.append(work_id)
        operation = self.by_work_id.get(work_id)
        if operation is None:
            return None
        if operation.status != "pending":
            return ProductOperationExecutionClaim(operation=operation, acquired=False)
        now = asyncio.get_running_loop().time()
        current = self.execution_claims.get(work_id)
        if current is not None and current[0] != claim_id and current[1] > now:
            return ProductOperationExecutionClaim(operation=operation, acquired=False)
        self.execution_claims[work_id] = (claim_id, now + lease_seconds)
        return ProductOperationExecutionClaim(operation=operation, acquired=True)

    async def heartbeat_execution(
        self,
        *,
        work_id: str,
        claim_id: str,
        lease_seconds: int,
    ) -> bool:
        current = self.execution_claims.get(work_id)
        operation = self.by_work_id.get(work_id)
        if current is None or current[0] != claim_id or operation is None:
            return False
        if operation.status != "pending":
            return True
        self.execution_claims[work_id] = (
            claim_id,
            asyncio.get_running_loop().time() + lease_seconds,
        )
        return True

    async def release_execution(
        self,
        *,
        work_id: str,
        claim_id: str,
    ) -> bool:
        operation = self.by_work_id.get(work_id)
        if operation is None:
            raise RuntimeError("Product work disappeared during execution-claim release.")
        current = self.execution_claims.get(work_id)
        if operation.status != "pending" or (current is not None and current[0] != claim_id):
            return False
        if current is not None:
            self.execution_claims.pop(work_id)
        return True

    async def finish(
        self,
        *,
        work_id: str,
        claim_id: str,
        status: str,
        result: str | None,
    ) -> ProductOperation:
        operation = self.by_work_id[work_id]
        if operation.status != "pending":
            current = self.execution_claims.get(work_id)
            if current is None or current[0] != claim_id:
                raise ProductExecutionClaimLost
            if operation.status == status and operation.result == result:
                return operation
            raise ProductOperationSettlementConflict
        current = self.execution_claims.get(work_id)
        if current is None or current[0] != claim_id:
            raise ProductExecutionClaimLost
        operation = operation.model_copy(update={"status": status, "result": result})
        self.by_work_id[work_id] = operation
        self.by_public_id[operation.public_id] = operation
        self.by_idempotency_key[operation.idempotency_key] = operation
        return operation


class BlockingFinalizationStore(MemoryProductStore):
    def __init__(self) -> None:
        super().__init__()
        self.finalization_started = asyncio.Event()
        self.allow_finalization = asyncio.Event()

    async def finish(
        self,
        *,
        work_id: str,
        claim_id: str,
        status: str,
        result: str | None,
    ) -> ProductOperation:
        self.finalization_started.set()
        await self.allow_finalization.wait()
        return await super().finish(
            work_id=work_id,
            claim_id=claim_id,
            status=status,
            result=result,
        )


def _build_service(
    provider=None,
    *,
    agent_name="assistant",
    registered_agent_name="assistant",
    operator_access=None,
    product_access=None,
    session_store=None,
    task_store=None,
    product_store=None,
    secret_redactor=None,
    product_api_path="/api",
    control_plane_path="/cayu",
):
    provider = provider or ScriptedModelProvider(
        [
            ModelStreamEvent.text_delta("safe product answer"),
            ModelStreamEvent.completed(
                {"finish_reason": "stop", "provider_body": "provider-body-sentinel"}
            ),
        ]
    )
    app = CayuApp(
        session_store=session_store if session_store is not None else InMemorySessionStore(),
        task_store=task_store if task_store is not None else InMemoryTaskStore(),
        secret_redactor=secret_redactor,
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name=registered_agent_name, model="scripted-model"))
    product_store = product_store if product_store is not None else MemoryProductStore()

    async def product_auth(request: Request) -> ProductPrincipal:
        token = request.headers.get("authorization")
        principals = {
            "Bearer tenant-a": ProductPrincipal(tenant_id="tenant-a", subject_id="alice"),
            "Bearer tenant-b": ProductPrincipal(tenant_id="tenant-b", subject_id="bob"),
        }
        principal = principals.get(token or "")
        if principal is None:
            raise HTTPException(status_code=401, detail="Authentication required.")
        return principal

    service = create_agent_service(
        app,
        agent_name=agent_name,
        mode=ServiceMode.PRODUCTION,
        product_access=(
            product_access
            if product_access is not None
            else AuthenticatedProductAccess(dependency=product_auth)
        ),
        operator_access=operator_access
        or AuthenticatedAccess(
            dependency=BasicAuth(username="operator", password="operator-secret")
        ),
        product_store=product_store,
        product_api_path=product_api_path,
        control_plane_path=control_plane_path,
    )
    return service, product_store, provider


def _product_request_fingerprint(request_text: str, *, agent_name: str = "assistant") -> str:
    encoded = json.dumps(
        {"agent_name": agent_name, "request": request_text, "schema_version": 1},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_public_service_enforces_tenant_lookup_idempotency_and_safe_projection() -> None:
    service, store, provider = _build_service()
    client = TestClient(service.asgi_app)
    headers_a = {"Authorization": "Bearer tenant-a", "Idempotency-Key": "request-1"}

    assert client.post("/api/operations", json={"request": "summarize"}).status_code == 401
    created = client.post("/api/operations", headers=headers_a, json={"request": "summarize"})
    assert created.status_code == 201
    assert created.headers["cache-control"] == "private, no-store"
    assert created.json() == {
        "id": created.json()["id"],
        "status": "completed",
        "result": "safe product answer",
    }
    public_id = created.json()["id"]
    operation = store.by_public_id[public_id]
    assert operation.tenant_id == "tenant-a"
    assert operation.session_id
    assert operation.task_id
    assert operation.work_id in store.claimed_work_ids

    repeated = client.post("/api/operations", headers=headers_a, json={"request": "summarize"})
    assert repeated.status_code == 200
    assert repeated.json() == created.json()
    assert len(provider.requests) == 1

    assert client.get(f"/api/operations/{public_id}").status_code == 401
    assert (
        client.get(
            f"/api/operations/{public_id}", headers={"Authorization": "Bearer tenant-b"}
        ).status_code
        == 404
    )
    assert (
        client.get(
            f"/api/operations/{operation.session_id}",
            headers={"Authorization": "Bearer tenant-a"},
        ).status_code
        == 404
    )
    authorized_read = client.get(f"/api/operations/{public_id}", headers=headers_a)
    assert authorized_read.headers["cache-control"] == "private, no-store"
    assert authorized_read.json() == created.json()

    same_tenant_conflict = client.post(
        "/api/operations",
        headers=headers_a,
        json={"request": "different work"},
    )
    assert same_tenant_conflict.status_code == 409
    other_tenant_conflict = client.post(
        "/api/operations",
        headers={"Authorization": "Bearer tenant-b", "Idempotency-Key": "request-1"},
        json={"request": "summarize"},
    )
    assert other_tenant_conflict.status_code == 409
    assert same_tenant_conflict.json() == other_tenant_conflict.json()

    encoded = repr(created.json())
    for private_value in (
        operation.tenant_id,
        operation.work_id,
        operation.session_id,
        operation.task_id,
        operation.request_fingerprint,
        operation.idempotency_key,
    ):
        assert private_value not in encoded


def test_pending_reservation_is_reconstructed_on_idempotent_redelivery() -> None:
    store = MemoryProductStore()
    asyncio.run(
        store.reserve(
            tenant_id="tenant-a",
            idempotency_key="restart-redelivery",
            request_fingerprint=_product_request_fingerprint("recover work"),
            public_id="op_restart",
            work_id="work_restart",
            session_id="session_restart",
            task_id="task_restart",
            request_text="recover work",
        )
    )
    service, _store, provider = _build_service(product_store=store)

    response = TestClient(service.asgi_app).post(
        "/api/operations",
        headers={
            "Authorization": "Bearer tenant-a",
            "Idempotency-Key": "restart-redelivery",
        },
        json={"request": "recover work"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "id": "op_restart",
        "status": "completed",
        "result": "safe product answer",
    }
    assert len(provider.requests) == 1


def test_background_execution_rejects_agent_fingerprint_drift_and_releases_claim() -> None:
    async def scenario() -> None:
        store = MemoryProductStore()
        session_store = InMemorySessionStore()
        task_store = InMemoryTaskStore()
        reservation = await store.reserve(
            tenant_id="tenant-a",
            idempotency_key="agent-drift",
            request_fingerprint=_product_request_fingerprint(
                "bound work",
                agent_name="original-agent",
            ),
            public_id="op_agent_drift",
            work_id="work_agent_drift",
            session_id="session_agent_drift",
            task_id="task_agent_drift",
            request_text="bound work",
        )
        wrong_service, _store, wrong_provider = _build_service(
            agent_name="replacement-agent",
            registered_agent_name="replacement-agent",
            product_store=store,
            session_store=session_store,
            task_store=task_store,
        )

        with pytest.raises(RuntimeError, match="does not match this service"):
            await wrong_service.execute_work(reservation.operation.work_id)

        assert reservation.operation.work_id not in store.execution_claims
        assert await task_store.load_task(reservation.operation.task_id) is None
        assert wrong_provider.requests == []

        correct_service, _store, correct_provider = _build_service(
            agent_name="original-agent",
            registered_agent_name="original-agent",
            product_store=store,
            session_store=session_store,
            task_store=task_store,
        )
        completed = await correct_service.execute_work(reservation.operation.work_id)

        assert completed is not None and completed.status == "completed"
        assert len(correct_provider.requests) == 1

    asyncio.run(scenario())


def test_precreated_product_task_is_verified_before_redelivery() -> None:
    async def scenario() -> None:
        store = MemoryProductStore()
        service, _store, provider = _build_service(product_store=store)
        reservation = await store.reserve(
            tenant_id="tenant-a",
            idempotency_key="task-created-redelivery",
            request_fingerprint=_product_request_fingerprint("recover work"),
            public_id="op_task_created",
            work_id="work_task_created",
            session_id="session_task_created",
            task_id="task_task_created",
            request_text="recover work",
        )
        await service.cayu_app.create_task(
            TaskCreate(
                task_id=reservation.operation.task_id,
                type="public_agent_operation",
                session_id=reservation.operation.session_id,
                assigned_agent_name=service.agent_name,
            )
        )

        completed = await service.execute_work(reservation.operation.work_id)

        assert completed is not None
        assert completed.status == "completed"
        assert len(provider.requests) == 1

    asyncio.run(scenario())


def test_product_task_creation_acknowledgement_loss_is_reconstructed() -> None:
    class CommitThenRaiseTaskStore(InMemoryTaskStore):
        async def create_task(self, request):
            await super().create_task(request)
            raise RuntimeError("task creation acknowledgement lost")

    async def scenario() -> None:
        store = MemoryProductStore()
        service, _store, provider = _build_service(
            product_store=store,
            task_store=CommitThenRaiseTaskStore(),
        )
        reservation = await store.reserve(
            tenant_id="tenant-a",
            idempotency_key="task-acknowledgement-loss",
            request_fingerprint=_product_request_fingerprint("recover work"),
            public_id="op_task_acknowledgement",
            work_id="work_task_acknowledgement",
            session_id="session_task_acknowledgement",
            task_id="task_task_acknowledgement",
            request_text="recover work",
        )

        completed = await service.execute_work(reservation.operation.work_id)

        assert completed is not None
        assert completed.status == "completed"
        assert len(provider.requests) == 1

    asyncio.run(scenario())


def test_existing_session_is_not_redispatched_while_product_task_is_pending() -> None:
    async def scenario() -> None:
        store = MemoryProductStore()
        service, _store, provider = _build_service(product_store=store)
        reservation = await store.reserve(
            tenant_id="tenant-a",
            idempotency_key="session-created-redelivery",
            request_fingerprint=_product_request_fingerprint("do not repeat"),
            public_id="op_session_created",
            work_id="work_session_created",
            session_id="session_session_created",
            task_id="task_session_created",
            request_text="do not repeat",
        )
        await service.cayu_app.create_task(
            TaskCreate(
                task_id=reservation.operation.task_id,
                type="public_agent_operation",
                session_id=reservation.operation.session_id,
                assigned_agent_name=service.agent_name,
            )
        )
        outcome = await run_to_completion(
            service.cayu_app,
            RunRequest(
                agent_name=service.agent_name,
                messages=[Message.text("user", reservation.operation.request_text)],
                session_id=reservation.operation.session_id,
            ),
        )
        assert outcome.ok
        assert len(provider.requests) == 1

        redelivered = await service.execute_work(reservation.operation.work_id)

        assert redelivered is not None
        assert redelivered.status == "pending"
        assert store.by_work_id[reservation.operation.work_id].status == "pending"
        task_store = service.cayu_app.task_store
        assert task_store is not None
        pending_task = await task_store.load_task(reservation.operation.task_id)
        assert pending_task is not None
        assert pending_task.status == TaskStatus.PENDING
        assert len(provider.requests) == 1
        assert reservation.operation.work_id not in store.execution_claims
        takeover = await store.claim_execution(
            work_id=reservation.operation.work_id,
            claim_id="recovery-worker",
            lease_seconds=120,
        )
        assert takeover is not None and takeover.acquired

    asyncio.run(scenario())


def test_completed_cayu_work_is_not_redispatched_or_falsely_failed() -> None:
    class FailingFinalizationStore(MemoryProductStore):
        fail_finalization = True

        async def finish(self, **kwargs) -> ProductOperation:
            if self.fail_finalization:
                raise RuntimeError("product settlement unavailable")
            return await super().finish(**kwargs)

    async def scenario() -> None:
        store = FailingFinalizationStore()
        service, _store, provider = _build_service(product_store=store)
        reservation = await store.reserve(
            tenant_id="tenant-a",
            idempotency_key="completed-redelivery",
            request_fingerprint=_product_request_fingerprint("complete once"),
            public_id="op_completed_redelivery",
            work_id="work_completed_redelivery",
            session_id="session_completed_redelivery",
            task_id="task_completed_redelivery",
            request_text="complete once",
        )

        with pytest.raises(RuntimeError, match="product settlement unavailable"):
            await service.execute_work(reservation.operation.work_id)
        assert len(provider.requests) == 1
        assert store.by_work_id[reservation.operation.work_id].status == "pending"
        task_store = service.cayu_app.task_store
        assert task_store is not None
        completed_task = await task_store.load_task(reservation.operation.task_id)
        assert completed_task is not None
        assert completed_task.status == TaskStatus.COMPLETED

        store.fail_finalization = False
        # Simulate the original execution lease expiring after settlement failed.
        store.execution_claims.pop(reservation.operation.work_id)
        redelivered = await service.execute_work(reservation.operation.work_id)

        assert redelivered is not None
        assert redelivered.status == "pending"
        assert store.by_work_id[reservation.operation.work_id].status == "pending"
        assert reservation.operation.work_id not in store.execution_claims
        assert len(provider.requests) == 1

    asyncio.run(scenario())


def test_progressed_product_work_release_reconciles_lost_acknowledgement() -> None:
    class ReleaseAcknowledgementLossStore(MemoryProductStore):
        release_calls = 0

        async def release_execution(self, **kwargs) -> bool:
            released = await super().release_execution(**kwargs)
            self.release_calls += 1
            if self.release_calls == 1:
                raise RuntimeError("release acknowledgement lost")
            return released

    async def scenario() -> None:
        store = ReleaseAcknowledgementLossStore()
        service, _store, provider = _build_service(product_store=store)
        reservation = await store.reserve(
            tenant_id="tenant-a",
            idempotency_key="release-acknowledgement-loss",
            request_fingerprint=_product_request_fingerprint("do not repeat"),
            public_id="op_release_acknowledgement",
            work_id="work_release_acknowledgement",
            session_id="session_release_acknowledgement",
            task_id="task_release_acknowledgement",
            request_text="do not repeat",
        )
        task_store = service.cayu_app.task_store
        assert task_store is not None
        await task_store.create_task(
            TaskCreate(
                task_id=reservation.operation.task_id,
                type="public_agent_operation",
                session_id=reservation.operation.session_id,
                assigned_agent_name=service.agent_name,
            )
        )
        await task_store.start_task(reservation.operation.task_id)

        redelivered = await service.execute_work(reservation.operation.work_id)

        assert redelivered is not None and redelivered.status == "pending"
        assert store.release_calls == 2
        assert reservation.operation.work_id not in store.execution_claims
        takeover = await store.claim_execution(
            work_id=reservation.operation.work_id,
            claim_id="replacement-worker",
            lease_seconds=120,
        )
        assert takeover is not None and takeover.acquired
        assert provider.requests == []

    asyncio.run(scenario())


def test_progressed_product_work_release_resists_caller_cancellation() -> None:
    class BlockingReleaseStore(MemoryProductStore):
        def __init__(self) -> None:
            super().__init__()
            self.release_started = asyncio.Event()
            self.allow_release = asyncio.Event()

        async def release_execution(self, **kwargs) -> bool:
            self.release_started.set()
            await self.allow_release.wait()
            return await super().release_execution(**kwargs)

    async def scenario() -> None:
        store = BlockingReleaseStore()
        service, _store, provider = _build_service(product_store=store)
        reservation = await store.reserve(
            tenant_id="tenant-a",
            idempotency_key="cancelled-release",
            request_fingerprint=_product_request_fingerprint("do not repeat"),
            public_id="op_cancelled_release",
            work_id="work_cancelled_release",
            session_id="session_cancelled_release",
            task_id="task_cancelled_release",
            request_text="do not repeat",
        )
        task_store = service.cayu_app.task_store
        assert task_store is not None
        await task_store.create_task(
            TaskCreate(
                task_id=reservation.operation.task_id,
                type="public_agent_operation",
                session_id=reservation.operation.session_id,
                assigned_agent_name=service.agent_name,
            )
        )
        await task_store.start_task(reservation.operation.task_id)

        execution = asyncio.create_task(service.execute_work(reservation.operation.work_id))
        await store.release_started.wait()
        execution.cancel("caller stopped during recovery handoff")
        store.allow_release.set()

        with pytest.raises(asyncio.CancelledError, match="caller stopped"):
            await execution
        assert store.by_work_id[reservation.operation.work_id].status == "pending"
        assert reservation.operation.work_id not in store.execution_claims
        assert provider.requests == []

    asyncio.run(scenario())


def test_product_reconciliation_failure_releases_execution_claim() -> None:
    class UnavailableTaskStore(InMemoryTaskStore):
        async def create_task(self, request):
            del request
            raise RuntimeError("task create unavailable")

        async def load_task(self, task_id):
            del task_id
            raise RuntimeError("task load unavailable")

    async def scenario() -> None:
        store = MemoryProductStore()
        service, _store, provider = _build_service(
            product_store=store,
            task_store=UnavailableTaskStore(),
        )
        reservation = await store.reserve(
            tenant_id="tenant-a",
            idempotency_key="reconciliation-release",
            request_fingerprint=_product_request_fingerprint("recover later"),
            public_id="op_reconciliation_release",
            work_id="work_reconciliation_release",
            session_id="session_reconciliation_release",
            task_id="task_reconciliation_release",
            request_text="recover later",
        )

        with pytest.raises(
            RuntimeError,
            match="acknowledgement could not be reconciled safely",
        ):
            await service.execute_work(reservation.operation.work_id)

        assert reservation.operation.work_id not in store.execution_claims
        takeover = await store.claim_execution(
            work_id=reservation.operation.work_id,
            claim_id="retry-worker",
            lease_seconds=120,
        )
        assert takeover is not None and takeover.acquired
        assert provider.requests == []

    asyncio.run(scenario())


@pytest.mark.parametrize("secret_field", ["tenant_id", "idempotency_key", "request"])
def test_secret_bearing_product_values_are_rejected_before_reservation(
    secret_field: str,
) -> None:
    secret = "workload-secret-value"
    store = MemoryProductStore()

    async def product_auth(_request: Request) -> ProductPrincipal:
        return ProductPrincipal(
            tenant_id=(secret if secret_field == "tenant_id" else "tenant-a"),
            subject_id="alice",
        )

    service, _store, provider = _build_service(
        product_store=store,
        product_access=AuthenticatedProductAccess(dependency=product_auth),
        secret_redactor=SecretRedactor(secret),
    )
    response = TestClient(service.asgi_app).post(
        "/api/operations",
        headers={
            "Authorization": "Bearer tenant-a",
            "Idempotency-Key": (secret if secret_field == "idempotency_key" else "secret-boundary"),
        },
        json={"request": secret if secret_field == "request" else "safe work"},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid product request."}
    assert response.headers["cache-control"] == "private, no-store"
    assert store.by_work_id == {}
    assert store.claimed_work_ids == []
    assert provider.requests == []


def test_concurrent_product_redelivery_does_not_execute_claimed_work_twice(
    monkeypatch,
) -> None:
    async def scenario() -> None:
        store = MemoryProductStore()
        service, _store, provider = _build_service(product_store=store)
        await store.reserve(
            tenant_id="tenant-a",
            idempotency_key="concurrent-redelivery",
            request_fingerprint=_product_request_fingerprint("work"),
            public_id="op_concurrent",
            work_id="work_concurrent",
            session_id="session_concurrent",
            task_id="task_concurrent",
            request_text="work",
        )
        execution_started = asyncio.Event()
        allow_execution = asyncio.Event()
        assert service.cayu_app.task_store is not None
        original_create_task = service.cayu_app.task_store.create_task

        async def blocking_create_task(request):
            execution_started.set()
            await allow_execution.wait()
            return await original_create_task(request)

        monkeypatch.setattr(service.cayu_app.task_store, "create_task", blocking_create_task)
        first = asyncio.create_task(service.execute_work("work_concurrent"))
        await execution_started.wait()

        duplicate = await service.execute_work("work_concurrent")
        assert duplicate is not None
        assert duplicate.status == "pending"
        assert provider.requests == []

        allow_execution.set()
        completed = await first
        assert completed is not None
        assert completed.status == "completed"
        assert len(provider.requests) == 1

    asyncio.run(scenario())


def test_execution_claim_is_rechecked_before_provider_execution() -> None:
    class ClaimLostBeforeExecutionStore(MemoryProductStore):
        async def heartbeat_execution(
            self,
            *,
            work_id: str,
            claim_id: str,
            lease_seconds: int,
        ) -> bool:
            del claim_id
            self.execution_claims[work_id] = (
                "successor-claim",
                asyncio.get_running_loop().time() + lease_seconds,
            )
            return False

    async def scenario() -> None:
        store = ClaimLostBeforeExecutionStore()
        service, _store, provider = _build_service(product_store=store)
        await store.reserve(
            tenant_id="tenant-a",
            idempotency_key="pre-provider-fence",
            request_fingerprint=_product_request_fingerprint("work"),
            public_id="op_pre_provider_fence",
            work_id="work_pre_provider_fence",
            session_id="session_pre_provider_fence",
            task_id="task_pre_provider_fence",
            request_text="work",
        )

        with pytest.raises(BaseExceptionGroup):
            await service.execute_work("work_pre_provider_fence")

        assert store.by_work_id["work_pre_provider_fence"].status == "pending"
        assert provider.requests == []

    asyncio.run(scenario())


def test_product_claim_and_finish_reconcile_commit_then_raise_acknowledgements() -> None:
    class AcknowledgementLossStore(MemoryProductStore):
        def __init__(self) -> None:
            super().__init__()
            self.raise_after_claim = True
            self.raise_after_finish = True

        async def claim_execution(self, **kwargs) -> ProductOperationExecutionClaim | None:
            claim = await super().claim_execution(**kwargs)
            if self.raise_after_claim:
                self.raise_after_claim = False
                raise RuntimeError("claim-acknowledgement-lost")
            return claim

        async def finish(self, **kwargs) -> ProductOperation:
            operation = await super().finish(**kwargs)
            if self.raise_after_finish:
                self.raise_after_finish = False
                raise RuntimeError("finish-acknowledgement-lost")
            return operation

    store = AcknowledgementLossStore()
    service, _store, provider = _build_service(product_store=store)
    response = TestClient(service.asgi_app).post(
        "/api/operations",
        headers={"Authorization": "Bearer tenant-a", "Idempotency-Key": "lost-acks"},
        json={"request": "work"},
    )

    assert response.status_code == 201
    assert response.json()["status"] == "completed"
    assert len(provider.requests) == 1
    assert len(store.by_work_id) == 1


def test_product_reservation_must_preserve_authenticated_tenant_authority() -> None:
    class CrossTenantReservationStore(MemoryProductStore):
        async def reserve(self, **kwargs) -> ProductOperationReservation:
            reservation = await super().reserve(**kwargs)
            return reservation.model_copy(
                update={
                    "operation": reservation.operation.model_copy(update={"tenant_id": "tenant-b"})
                }
            )

    service, _store, provider = _build_service(product_store=CrossTenantReservationStore())
    response = TestClient(service.asgi_app, raise_server_exceptions=False).post(
        "/api/operations",
        headers={"Authorization": "Bearer tenant-a", "Idempotency-Key": "cross-tenant"},
        json={"request": "work"},
    )

    assert response.status_code == 500
    assert response.headers["cache-control"] == "private, no-store"
    assert response.json() == {"detail": "Internal server error."}
    assert "tenant-b" not in response.text
    assert provider.requests == []


def test_product_store_accepts_enriched_operation_subclasses() -> None:
    class EnrichedProductOperation(ProductOperation):
        revision: int

    class EnrichedOperationStore(MemoryProductStore):
        @staticmethod
        def enrich(operation: ProductOperation) -> EnrichedProductOperation:
            return EnrichedProductOperation(
                **operation.model_dump(),
                revision=1,
            )

        async def reserve(self, **kwargs) -> ProductOperationReservation:
            reservation = await super().reserve(**kwargs)
            return reservation.model_copy(update={"operation": self.enrich(reservation.operation)})

        async def find(self, *, tenant_id: str, public_id: str) -> ProductOperation | None:
            operation = await super().find(tenant_id=tenant_id, public_id=public_id)
            return None if operation is None else self.enrich(operation)

        async def claim_execution(self, **kwargs) -> ProductOperationExecutionClaim | None:
            claim = await super().claim_execution(**kwargs)
            if claim is None:
                return None
            return claim.model_copy(update={"operation": self.enrich(claim.operation)})

    service, _store, provider = _build_service(product_store=EnrichedOperationStore())
    client = TestClient(service.asgi_app)
    created = client.post(
        "/api/operations",
        headers={"Authorization": "Bearer tenant-a", "Idempotency-Key": "enriched"},
        json={"request": "work"},
    )
    read = client.get(
        f"/api/operations/{created.json()['id']}",
        headers={"Authorization": "Bearer tenant-a"},
    )

    assert created.status_code == 201
    assert read.status_code == 200
    assert read.json() == created.json()
    assert len(provider.requests) == 1


def test_completed_idempotency_replay_allows_request_data_minimization() -> None:
    class RedactingReplayStore(MemoryProductStore):
        async def reserve(self, **kwargs) -> ProductOperationReservation:
            reservation = await super().reserve(**kwargs)
            if reservation.created:
                return reservation
            return reservation.model_copy(
                update={
                    "operation": reservation.operation.model_copy(
                        update={"request_text": "[redacted]"}
                    )
                }
            )

    service, _store, provider = _build_service(product_store=RedactingReplayStore())
    client = TestClient(service.asgi_app)
    headers = {"Authorization": "Bearer tenant-a", "Idempotency-Key": "minimized"}
    created = client.post("/api/operations", headers=headers, json={"request": "private work"})
    replayed = client.post(
        "/api/operations",
        headers=headers,
        json={"request": "private work"},
    )

    assert created.status_code == 201
    assert replayed.status_code == 200
    assert replayed.json() == created.json()
    assert len(provider.requests) == 1


def test_product_lookup_must_preserve_authenticated_tenant_authority() -> None:
    class CrossTenantLookupStore(MemoryProductStore):
        async def find(self, *, tenant_id: str, public_id: str) -> ProductOperation | None:
            del tenant_id
            return ProductOperation(
                tenant_id="tenant-b",
                public_id=public_id,
                work_id="foreign-work",
                idempotency_key="foreign-key",
                request_fingerprint="foreign-fingerprint",
                session_id="foreign-session",
                task_id="foreign-task",
                request_text="foreign-request",
                status="completed",
                result="foreign-result-sentinel",
            )

    service, _store, _provider = _build_service(product_store=CrossTenantLookupStore())
    response = TestClient(service.asgi_app, raise_server_exceptions=False).get(
        "/api/operations/op_guessed",
        headers={"Authorization": "Bearer tenant-a"},
    )

    assert response.status_code == 500
    assert response.headers["cache-control"] == "private, no-store"
    assert response.json() == {"detail": "Internal server error."}
    assert "foreign-result-sentinel" not in response.text


def test_product_authentication_revalidates_constructed_principals() -> None:
    async def invalid_auth(_request: Request) -> ProductPrincipal:
        return ProductPrincipal.model_construct(tenant_id="", subject_id="subject")

    service, store, provider = _build_service(
        ScriptedModelProvider([]),
        product_access=AuthenticatedProductAccess(dependency=invalid_auth),
    )

    response = TestClient(service.asgi_app, raise_server_exceptions=False).post(
        "/api/operations",
        headers={"Idempotency-Key": "invalid-principal"},
        json={"request": "work"},
    )

    assert response.status_code == 500
    assert response.headers["cache-control"] == "private, no-store"
    assert response.json() == {"detail": "Internal server error."}
    assert store.by_work_id == {}
    assert provider.requests == []


def test_product_authentication_accepts_enriched_principal_subclasses() -> None:
    class EnrichedProductPrincipal(ProductPrincipal):
        roles: tuple[str, ...]

    async def enriched_auth(_request: Request) -> ProductPrincipal:
        return EnrichedProductPrincipal(
            tenant_id="tenant-a",
            subject_id="subject",
            roles=("customer",),
        )

    service, _store, provider = _build_service(
        product_access=AuthenticatedProductAccess(dependency=enriched_auth)
    )
    response = TestClient(service.asgi_app).post(
        "/api/operations",
        headers={"Idempotency-Key": "enriched-principal"},
        json={"request": "work"},
    )

    assert response.status_code == 201
    assert response.json()["status"] == "completed"
    assert len(provider.requests) == 1


def test_product_execution_claim_must_preserve_requested_work_authority() -> None:
    class InconsistentWorkStore(MemoryProductStore):
        async def claim_execution(self, **kwargs) -> ProductOperationExecutionClaim | None:
            claim = await super().claim_execution(**kwargs)
            assert claim is not None
            return claim.model_copy(
                update={
                    "operation": claim.operation.model_copy(update={"work_id": "different-work"})
                }
            )

    async def scenario() -> None:
        store = InconsistentWorkStore()
        service, _store, provider = _build_service(product_store=store)
        await store.reserve(
            tenant_id="tenant-a",
            idempotency_key="work-authority",
            request_fingerprint=_product_request_fingerprint("work"),
            public_id="op_authority",
            work_id="work_authority",
            session_id="session_authority",
            task_id="task_authority",
            request_text="work",
        )

        with pytest.raises(RuntimeError, match="inconsistent authority from claim_execution"):
            await service.execute_work("work_authority")
        assert provider.requests == []

    asyncio.run(scenario())


def test_product_lookup_identifiers_are_bounded_before_store_access() -> None:
    class TrackingLookupStore(MemoryProductStore):
        def __init__(self) -> None:
            super().__init__()
            self.public_ids: list[str] = []

        async def find(self, *, tenant_id: str, public_id: str) -> ProductOperation | None:
            self.public_ids.append(public_id)
            return await super().find(tenant_id=tenant_id, public_id=public_id)

    store = TrackingLookupStore()
    service, _store, _provider = _build_service(product_store=store)
    response = TestClient(service.asgi_app).get(
        f"/api/operations/{'x' * 513}",
        headers={"Authorization": "Bearer tenant-a"},
    )

    assert response.status_code == 404
    assert response.headers["cache-control"] == "private, no-store"
    assert store.public_ids == []

    with pytest.raises(ValueError, match="work_id must not exceed 512 characters"):
        asyncio.run(service.execute_work("x" * 513))
    assert store.claimed_work_ids == []


def test_product_finalization_must_confirm_terminal_state() -> None:
    class InconsistentFinalizationStore(MemoryProductStore):
        async def finish(
            self,
            *,
            work_id: str,
            claim_id: str,
            status: str,
            result: str | None,
        ) -> ProductOperation:
            operation = await super().finish(
                work_id=work_id,
                claim_id=claim_id,
                status=status,
                result=result,
            )
            return operation.model_copy(update={"status": "pending", "result": None})

    service, _store, provider = _build_service(product_store=InconsistentFinalizationStore())
    response = TestClient(service.asgi_app, raise_server_exceptions=False).post(
        "/api/operations",
        headers={"Authorization": "Bearer tenant-a", "Idempotency-Key": "bad-finish"},
        json={"request": "work"},
    )

    assert response.status_code == 500
    assert response.headers["cache-control"] == "private, no-store"
    assert response.json() == {"detail": "Internal server error."}
    assert len(provider.requests) == 1


def test_public_service_keeps_operator_control_plane_separate_and_authenticated() -> None:
    service, _store, _provider = _build_service()
    client = TestClient(service.asgi_app)

    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404
    assert client.get("/cayu/").status_code == 401
    assert client.get("/cayu/", headers={"Authorization": "Bearer tenant-a"}).status_code == 401
    assert client.get("/cayu/", auth=("operator", "operator-secret")).status_code == 200
    assert client.get("/cayu/assets/missing.js").status_code == 401
    assert client.get("/cayu/api/sessions").status_code == 401
    assert client.delete("/cayu/api/sessions/guessed-private-id").status_code == 401
    assert client.get("/cayu/api/sessions", auth=("operator", "operator-secret")).status_code == 200
    assert service.manifest.product_access == "authenticated"
    assert service.manifest.operator_access == "authenticated"
    assert service.manifest.identity_store == "durable"
    assert service.manifest.runtime_session_store == "development"
    assert service.manifest.runtime_task_store == "development"
    assert service.manifest.host_routing == "maintained"


def test_sqlite_memory_runtime_stores_are_not_reported_as_durable() -> None:
    service, _store, _provider = _build_service(
        session_store=SQLiteSessionStore(":memory:"),
        task_store=SQLiteTaskStore(":memory:"),
    )

    assert service.manifest.runtime_session_store == "development"
    assert service.manifest.runtime_task_store == "development"


def test_placeholder_product_access_always_uses_runtime_owned_denial() -> None:
    service, store, provider = _build_service(product_access=PlaceholderProductAccess())
    response = TestClient(service.asgi_app, raise_server_exceptions=False).post(
        "/api/operations",
        headers={"Idempotency-Key": "must-deny"},
        json={"request": "work"},
    )

    assert response.status_code == 503
    assert response.headers["cache-control"] == "private, no-store"
    assert store.by_work_id == {}
    assert provider.requests == []

    with pytest.raises(ValidationError, match="Invalid placeholder product-access"):
        PlaceholderProductAccess(dependency=lambda _request: None)


def test_request_fingerprint_is_stable_and_does_not_expose_request_text() -> None:
    service, store, _provider = _build_service()
    request_text = "private customer prompt"
    response = TestClient(service.asgi_app).post(
        "/api/operations",
        headers={"Authorization": "Bearer tenant-a", "Idempotency-Key": "stable-key"},
        json={"request": request_text},
    )
    operation = store.by_public_id[response.json()["id"]]
    fingerprint_input = (
        '{"agent_name":"assistant","request":"private customer prompt","schema_version":1}'
    )
    assert operation.request_fingerprint == hashlib.sha256(fingerprint_input.encode()).hexdigest()
    assert request_text not in response.text


def test_public_projection_redacts_durable_result_on_read_and_idempotent_post() -> None:
    secret = "workload-secret-value"
    store = MemoryProductStore()
    pending = asyncio.run(
        store.reserve(
            tenant_id="tenant-a",
            idempotency_key="stored-secret",
            request_fingerprint=_product_request_fingerprint("safe work"),
            public_id="op_stored_secret",
            work_id="work_stored_secret",
            session_id="session_stored_secret",
            task_id="task_stored_secret",
            request_text="safe work",
        )
    ).operation
    completed = pending.model_copy(update={"status": "completed", "result": secret})
    store.by_work_id[completed.work_id] = completed
    store.by_public_id[completed.public_id] = completed
    store.by_idempotency_key[completed.idempotency_key] = completed
    service, _store, provider = _build_service(
        product_store=store,
        secret_redactor=SecretRedactor(secret),
    )
    client = TestClient(service.asgi_app)
    headers = {
        "Authorization": "Bearer tenant-a",
        "Idempotency-Key": completed.idempotency_key,
    }

    read = client.get(f"/api/operations/{completed.public_id}", headers=headers)
    repeated = client.post(
        "/api/operations",
        headers=headers,
        json={"request": completed.request_text},
    )

    for response in (read, repeated):
        assert response.status_code == 200
        assert response.json()["result"] == REDACTED_SECRET
        assert secret not in response.text
    assert provider.requests == []


def test_split_model_secret_is_redacted_before_product_store_persistence() -> None:
    secret = "workload-secret-value"
    provider = ScriptedModelProvider(
        [
            ModelStreamEvent.text_delta("workload-"),
            ModelStreamEvent.text_delta("secret-value"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    service, store, _provider = _build_service(
        provider,
        secret_redactor=SecretRedactor(secret),
    )

    response = TestClient(service.asgi_app).post(
        "/api/operations",
        headers={
            "Authorization": "Bearer tenant-a",
            "Idempotency-Key": "split-result-secret",
        },
        json={"request": "safe work"},
    )

    operation = next(iter(store.by_work_id.values()))
    assert response.status_code == 201
    assert response.json()["result"] == REDACTED_SECRET
    assert operation.result == REDACTED_SECRET
    assert secret not in repr(operation)


def test_result_limit_does_not_publish_or_store_a_partial_secret() -> None:
    secret = "workload-secret-value"
    safe_prefix = "x" * (MAX_PUBLIC_RESULT_CHARS - 5)
    provider = ScriptedModelProvider(
        [
            ModelStreamEvent.text_delta(safe_prefix),
            ModelStreamEvent.text_delta(secret),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    service, store, _provider = _build_service(
        provider,
        secret_redactor=SecretRedactor(secret),
    )

    response = TestClient(service.asgi_app).post(
        "/api/operations",
        headers={
            "Authorization": "Bearer tenant-a",
            "Idempotency-Key": "boundary-result-secret",
        },
        json={"request": "safe work"},
    )

    operation = next(iter(store.by_work_id.values()))
    assert response.status_code == 201
    assert response.json()["result"] == safe_prefix
    assert operation.result == safe_prefix
    assert secret not in repr(operation)


def test_result_redaction_capacity_fails_product_after_draining_runtime(
    monkeypatch,
) -> None:
    secret = "s" * 100
    monkeypatch.setattr(
        "cayu.server.service._PUBLIC_RESULT_CAPTURE_BYTES",
        64,
    )
    provider = ScriptedModelProvider(
        [
            ModelStreamEvent.text_delta("s" * 80),
            ModelStreamEvent.text_delta("safe suffix"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    service, store, _provider = _build_service(
        provider,
        secret_redactor=SecretRedactor(secret),
    )

    response = TestClient(service.asgi_app).post(
        "/api/operations",
        headers={
            "Authorization": "Bearer tenant-a",
            "Idempotency-Key": "result-redaction-capacity",
        },
        json={"request": "safe work"},
    )

    operation = next(iter(store.by_work_id.values()))
    state = asyncio.run(service.cayu_app.session_store.load_state(operation.session_id))
    assert response.status_code == 201
    assert response.json()["status"] == "failed"
    assert operation.status == "failed"
    assert operation.result is None
    assert state is not None and state.status is SessionStatus.COMPLETED
    assert len(provider.requests) == 1


def test_public_projection_fails_closed_when_public_id_contains_a_secret() -> None:
    secret = "workload-secret-value"
    store = MemoryProductStore()
    operation = ProductOperation(
        tenant_id="tenant-a",
        public_id=secret,
        work_id="work_unsafe_authority",
        idempotency_key="unsafe-authority",
        request_fingerprint=_product_request_fingerprint("safe work"),
        session_id="session_unsafe_authority",
        task_id="task_unsafe_authority",
        request_text="safe work",
        status="completed",
        result="safe result",
    )
    store.by_work_id[operation.work_id] = operation
    store.by_public_id[operation.public_id] = operation
    store.by_idempotency_key[operation.idempotency_key] = operation
    service, _store, _provider = _build_service(
        product_store=store,
        secret_redactor=SecretRedactor(secret),
    )

    response = TestClient(service.asgi_app, raise_server_exceptions=False).get(
        f"/api/operations/{secret}",
        headers={"Authorization": "Bearer tenant-a"},
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error."}
    assert secret not in response.text


def test_product_result_capture_is_bounded_without_losing_terminal_status() -> None:
    oversized_result = "x" * (MAX_PUBLIC_RESULT_CHARS + 25_000)
    provider = ScriptedModelProvider(
        [
            ModelStreamEvent.text_delta(oversized_result),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    service, _store, _provider = _build_service(provider)

    response = TestClient(service.asgi_app).post(
        "/api/operations",
        headers={
            "Authorization": "Bearer tenant-a",
            "Idempotency-Key": "bounded-result",
        },
        json={"request": "produce a long result"},
    )

    assert response.status_code == 201
    assert response.json()["status"] == "completed"
    assert response.json()["result"] == oversized_result[:MAX_PUBLIC_RESULT_CHARS]
    assert len(provider.requests) == 1


@pytest.mark.parametrize("request_text", ["   ", "contains-\x00-nul"])
def test_invalid_request_text_is_rejected_before_product_reservation(
    request_text: str,
) -> None:
    service, store, provider = _build_service()
    response = TestClient(service.asgi_app, raise_server_exceptions=False).post(
        "/api/operations",
        headers={"Authorization": "Bearer tenant-a", "Idempotency-Key": "invalid"},
        json={"request": request_text},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid product request."}
    assert store.by_work_id == {}
    assert provider.requests == []


def test_nonportable_json_text_is_rejected_before_product_reservation() -> None:
    service, store, provider = _build_service()
    response = TestClient(service.asgi_app, raise_server_exceptions=False).post(
        "/api/operations",
        headers={
            "Authorization": "Bearer tenant-a",
            "Content-Type": "application/json",
            "Idempotency-Key": "invalid-surrogate",
        },
        content=b'{"request":"\\ud800"}',
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid product request."}
    assert store.by_work_id == {}
    assert provider.requests == []


def test_product_request_boundary_rejects_duplicate_json_keys_before_reservation() -> None:
    service, store, provider = _build_service()
    response = TestClient(service.asgi_app, raise_server_exceptions=False).post(
        "/api/operations",
        headers={
            "Authorization": "Bearer tenant-a",
            "Content-Type": "application/json",
            "Idempotency-Key": "duplicate-json",
        },
        content=b'{"request":"first","request":"second"}',
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid product request."}
    assert response.headers["cache-control"] == "private, no-store"
    assert store.by_work_id == {}
    assert provider.requests == []


def test_product_request_boundary_rejects_declared_and_streamed_oversize() -> None:
    async def invoke(
        service,
        *,
        headers: list[tuple[bytes, bytes]],
        messages: list[dict[str, object]],
    ) -> tuple[int, dict[str, str], dict[str, object]]:
        sent: list[dict[str, object]] = []

        async def receive() -> dict[str, object]:
            if messages:
                return messages.pop(0)
            return {"type": "http.disconnect"}

        async def send(message: dict[str, object]) -> None:
            sent.append(message)

        await service.asgi_app(
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/api/operations",
                "raw_path": b"/api/operations",
                "query_string": b"",
                "root_path": "",
                "headers": headers,
                "client": ("127.0.0.1", 12345),
                "server": ("127.0.0.1", 8000),
            },
            receive,
            send,
        )
        start = next(message for message in sent if message["type"] == "http.response.start")
        body = b"".join(
            message.get("body", b"") for message in sent if message["type"] == "http.response.body"
        )
        response_headers = {
            key.decode("latin-1"): value.decode("latin-1") for key, value in start["headers"]
        }
        return start["status"], response_headers, json.loads(body)

    async def scenario() -> None:
        service, store, provider = _build_service()
        common_headers = [
            (b"authorization", b"Bearer tenant-a"),
            (b"content-type", b"application/json"),
            (b"idempotency-key", b"oversized"),
        ]
        declared = await invoke(
            service,
            headers=[
                *common_headers,
                (b"content-length", str(MAX_PRODUCT_REQUEST_BYTES + 1).encode("ascii")),
            ],
            messages=[],
        )
        streamed = await invoke(
            service,
            headers=common_headers,
            messages=[
                {
                    "type": "http.request",
                    "body": b"x" * MAX_PRODUCT_REQUEST_BYTES,
                    "more_body": True,
                },
                {"type": "http.request", "body": b"x", "more_body": False},
            ],
        )

        for status, headers, body in (declared, streamed):
            assert status == 413
            assert headers["cache-control"] == "private, no-store"
            assert body == {"detail": "Product request exceeds the server byte limit."}
        assert store.by_work_id == {}
        assert provider.requests == []

    asyncio.run(scenario())


def test_public_service_requires_its_selected_agent_to_be_registered() -> None:
    with pytest.raises(ValueError, match="agent_name must identify a registered agent"):
        _build_service(agent_name="missing", registered_agent_name="registered")


@pytest.mark.parametrize(
    "field_name, value",
    [
        ("product_api_path", "/api?tenant=one"),
        ("product_api_path", "/api#fragment"),
        ("product_api_path", "/api%2Fhidden"),
        ("product_api_path", "/api/{tenant}"),
        ("product_api_path", "api"),
        ("product_api_path", "/api/"),
        ("control_plane_path", "/cayu\\admin"),
    ],
)
def test_public_service_rejects_nonliteral_or_nonportable_mount_paths(
    field_name: str,
    value: str,
) -> None:
    kwargs = {field_name: value}

    with pytest.raises(ValueError):
        _build_service(**kwargs)


def test_public_service_serves_custom_fixed_paths() -> None:
    service, _store, _provider = _build_service(
        product_api_path="/product/v1",
        control_plane_path="/operators",
    )

    assert service.manifest.product_api_path == "/product/v1"
    assert service.manifest.control_plane_path == "/operators"
    response = TestClient(service.asgi_app).post(
        "/product/v1/operations",
        headers={"Authorization": "Bearer tenant-a", "Idempotency-Key": "custom-path"},
        json={"request": "work"},
    )
    assert response.status_code == 201
    assert (
        TestClient(service.asgi_app)
        .get("/operators/", auth=("operator", "operator-secret"))
        .status_code
        == 200
    )


def test_public_service_does_not_project_provider_failure_bodies() -> None:
    service, _store, _provider = _build_service(
        ScriptedModelProvider(
            [
                ModelStreamEvent.error("provider-error-sentinel"),
                ModelStreamEvent.completed({"finish_reason": "error"}),
            ]
        )
    )
    response = TestClient(service.asgi_app).post(
        "/api/operations",
        headers={"Authorization": "Bearer tenant-a", "Idempotency-Key": "failure"},
        json={"request": "private-prompt-sentinel"},
    )

    assert response.status_code == 201
    assert response.json()["status"] == "failed"
    assert response.json()["result"] is None
    assert "provider-error-sentinel" not in response.text
    assert "private-prompt-sentinel" not in response.text


def test_product_access_validation_redacts_rejected_dependency_inputs() -> None:
    sentinel = "PRODUCT_ACCESS_SECRET_SENTINEL"

    with pytest.raises(ValidationError) as captured:
        AuthenticatedProductAccess(dependency=sentinel)

    rendered = str(captured.value) + repr(captured.value.errors()) + captured.value.json()
    assert sentinel not in rendered
    assert captured.value.errors()[0]["input"] is None


def test_cayu_service_cannot_be_fabricated_through_its_public_constructor() -> None:
    service, store, _provider = _build_service()

    with pytest.raises(TypeError, match="created only by create_agent_service"):
        CayuService(
            cayu_app=service.cayu_app,
            asgi_app=service.asgi_app,
            manifest=service.manifest,
            product_store=store,
            agent_name="assistant",
        )


def test_placeholder_operator_access_denies_the_assembled_control_plane() -> None:
    service, _store, _provider = _build_service(operator_access=PlaceholderOperatorAccess())

    assert service.manifest.operator_access == "placeholder"
    assert TestClient(service.asgi_app).get("/cayu/").status_code == 503


def test_maintained_service_provenance_rejects_appended_routes() -> None:
    service, _store, _provider = _build_service()
    assert _is_maintained_service(service)

    @service.asgi_app.get("/anonymous")
    async def anonymous():
        return {"unsafe": True}

    assert not _is_maintained_service(service)


def test_maintained_service_provenance_survives_normal_asgi_startup() -> None:
    service, _store, _provider = _build_service()
    assert _is_maintained_service(service)

    with TestClient(service.asgi_app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/cayu/", auth=("operator", "operator-secret")).status_code == 200

    assert _is_maintained_service(service)


def test_maintained_service_provenance_rejects_lifespan_replacement() -> None:
    service, _store, _provider = _build_service()
    assert _is_maintained_service(service)

    @asynccontextmanager
    async def unsafe_lifespan(app):
        @app.get("/late-anonymous")
        async def late_anonymous():
            return {"unsafe": True}

        yield

    service.asgi_app.router.lifespan_context = unsafe_lifespan

    assert not _is_maintained_service(service)


def test_maintained_service_provenance_rejects_router_dispatch_replacement() -> None:
    service, _store, _provider = _build_service()
    assert _is_maintained_service(service)

    async def anonymous_default(_scope, _receive, send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [],
            }
        )
        await send({"type": "http.response.body", "body": b"unsafe"})

    service.asgi_app.router.default = anonymous_default

    assert not _is_maintained_service(service)


def test_maintained_service_provenance_rejects_middleware_stack_replacement() -> None:
    service, _store, _provider = _build_service()
    assert _is_maintained_service(service)

    service.asgi_app.middleware_stack = object()

    assert not _is_maintained_service(service)


def test_sync_product_authentication_runs_outside_the_event_loop_thread() -> None:
    dependency_threads: list[int] = []

    def authenticate(_request: Request) -> ProductPrincipal:
        dependency_threads.append(threading.get_ident())
        return ProductPrincipal(tenant_id="tenant", subject_id="subject")

    resolve = _product_auth_dependency(authenticate)

    async def invoke() -> tuple[int, ProductPrincipal]:
        event_loop_thread = threading.get_ident()
        principal = await resolve(Request({"type": "http"}))
        return event_loop_thread, principal

    event_loop_thread, principal = asyncio.run(invoke())

    assert principal == ProductPrincipal(tenant_id="tenant", subject_id="subject")
    assert dependency_threads
    assert dependency_threads[0] != event_loop_thread


def test_execution_failure_terminalizes_the_reserved_product_operation(
    monkeypatch,
) -> None:
    service, store, provider = _build_service()

    async def fail_task_creation(_request) -> None:
        raise RuntimeError("transient-task-store-sentinel")

    assert service.cayu_app.task_store is not None
    monkeypatch.setattr(service.cayu_app.task_store, "create_task", fail_task_creation)
    client = TestClient(service.asgi_app, raise_server_exceptions=False)
    headers = {"Authorization": "Bearer tenant-a", "Idempotency-Key": "retry-me"}

    first = client.post("/api/operations", headers=headers, json={"request": "work"})
    repeated = client.post("/api/operations", headers=headers, json={"request": "work"})

    assert first.status_code == 500
    assert first.headers["cache-control"] == "private, no-store"
    assert "transient-task-store-sentinel" not in first.text
    assert repeated.status_code == 200
    assert repeated.json()["status"] == "failed"
    assert repeated.json()["result"] is None
    assert len(store.by_public_id) == 1
    assert len(provider.requests) == 0


def test_execution_and_finalization_failures_preserve_both_exception_objects(
    monkeypatch,
) -> None:
    class FinalizationFailureStore(MemoryProductStore):
        async def finish(
            self,
            *,
            work_id: str,
            claim_id: str,
            status: str,
            result: str | None,
        ) -> ProductOperation:
            del work_id, claim_id, status, result
            raise RuntimeError("product-finalization-sentinel")

    service, _store, _provider = _build_service(product_store=FinalizationFailureStore())

    async def fail_task_creation(_request) -> None:
        raise RuntimeError("task-creation-sentinel")

    assert service.cayu_app.task_store is not None
    monkeypatch.setattr(service.cayu_app.task_store, "create_task", fail_task_creation)

    with pytest.raises(BaseExceptionGroup) as captured:
        TestClient(service.asgi_app).post(
            "/api/operations",
            headers={"Authorization": "Bearer tenant-a", "Idempotency-Key": "failure"},
            json={"request": "work"},
        )

    assert [str(error) for error in captured.value.exceptions] == [
        "task-creation-sentinel",
        "product-finalization-sentinel",
    ]


def test_grouped_execution_failure_also_terminalizes_the_operation(monkeypatch) -> None:
    service, store, _provider = _build_service()

    async def fail_task_creation(_request) -> None:
        raise BaseExceptionGroup(
            "grouped-execution-sentinel",
            [RuntimeError("ordinary-member"), asyncio.CancelledError("cancelled-member")],
        )

    assert service.cayu_app.task_store is not None
    monkeypatch.setattr(service.cayu_app.task_store, "create_task", fail_task_creation)

    with pytest.raises(BaseExceptionGroup, match="grouped-execution-sentinel"):
        TestClient(service.asgi_app).post(
            "/api/operations",
            headers={"Authorization": "Bearer tenant-a", "Idempotency-Key": "grouped"},
            json={"request": "work"},
        )

    operation = next(iter(store.by_work_id.values()))
    assert operation.status == "failed"


def test_execution_cancellation_remains_authoritative_and_terminalizes_work(
    monkeypatch,
) -> None:
    store = BlockingFinalizationStore()
    service, _store, _provider = _build_service(product_store=store)
    execution_started = asyncio.Event()

    async def block_task_creation(_request) -> None:
        execution_started.set()
        await asyncio.Event().wait()

    assert service.cayu_app.task_store is not None
    monkeypatch.setattr(service.cayu_app.task_store, "create_task", block_task_creation)

    async def cancel_during_execution() -> asyncio.CancelledError:
        await store.reserve(
            tenant_id="tenant-a",
            idempotency_key="cancelled",
            request_fingerprint=_product_request_fingerprint("work"),
            public_id="op_cancelled",
            work_id="work_cancelled",
            session_id="session_cancelled",
            task_id="task_cancelled",
            request_text="work",
        )
        execution = asyncio.create_task(service.execute_work("work_cancelled"))
        await execution_started.wait()
        execution.cancel("first-cancellation")
        await store.finalization_started.wait()
        execution.cancel("repeated-cancellation")
        store.allow_finalization.set()
        with pytest.raises(asyncio.CancelledError) as captured:
            await execution
        assert execution.cancelled()
        return captured.value

    cancellation = asyncio.run(cancel_during_execution())

    assert store.by_work_id["work_cancelled"].status == "failed"
    assert cancellation.args == ("first-cancellation",)


def test_completion_finalization_settles_before_cancellation_is_redelivered() -> None:
    store = BlockingFinalizationStore()
    service, _store, provider = _build_service(product_store=store)

    async def cancel_during_completion() -> asyncio.CancelledError:
        await store.reserve(
            tenant_id="tenant-a",
            idempotency_key="completed",
            request_fingerprint=_product_request_fingerprint("work"),
            public_id="op_completed",
            work_id="work_completed",
            session_id="session_completed",
            task_id="task_completed",
            request_text="work",
        )
        execution = asyncio.create_task(service.execute_work("work_completed"))
        await store.finalization_started.wait()
        execution.cancel("completion-cancellation")
        store.allow_finalization.set()
        with pytest.raises(asyncio.CancelledError) as captured:
            await execution
        assert execution.cancelled()
        return captured.value

    cancellation = asyncio.run(cancel_during_completion())

    assert store.by_work_id["work_completed"].status == "completed"
    assert store.by_work_id["work_completed"].result == "safe product answer"
    assert cancellation.args == ("completion-cancellation",)
    assert len(provider.requests) == 1


def test_execution_heartbeat_remains_active_through_terminal_settlement(
    monkeypatch,
) -> None:
    class HeartbeatDuringFinalizationStore(BlockingFinalizationStore):
        def __init__(self) -> None:
            super().__init__()
            self.heartbeat_during_finalization = asyncio.Event()

        async def heartbeat_execution(self, **kwargs) -> bool:
            retained = await super().heartbeat_execution(**kwargs)
            if self.finalization_started.is_set():
                self.heartbeat_during_finalization.set()
            return retained

    monkeypatch.setattr(
        "cayu.server.service.PRODUCT_EXECUTION_HEARTBEAT_SECONDS",
        0.01,
    )
    store = HeartbeatDuringFinalizationStore()
    service, _store, _provider = _build_service(product_store=store)

    async def scenario() -> None:
        await store.reserve(
            tenant_id="tenant-a",
            idempotency_key="heartbeat-through-finish",
            request_fingerprint=_product_request_fingerprint("work"),
            public_id="op_heartbeat",
            work_id="work_heartbeat",
            session_id="session_heartbeat",
            task_id="task_heartbeat",
            request_text="work",
        )
        execution = asyncio.create_task(service.execute_work("work_heartbeat"))
        await store.finalization_started.wait()
        await asyncio.wait_for(store.heartbeat_during_finalization.wait(), timeout=1)
        store.allow_finalization.set()
        completed = await execution
        assert completed is not None
        assert completed.status == "completed"

    asyncio.run(scenario())


def test_terminal_settlement_wins_an_in_flight_heartbeat(monkeypatch) -> None:
    class SettlementBeforeHeartbeatStore(MemoryProductStore):
        def __init__(self) -> None:
            super().__init__()
            self.heartbeat_started = asyncio.Event()
            self.allow_heartbeat = asyncio.Event()
            self.heartbeat_calls = 0

        async def heartbeat_execution(self, **kwargs) -> bool:
            self.heartbeat_calls += 1
            if self.heartbeat_calls == 1:
                return await super().heartbeat_execution(**kwargs)
            self.heartbeat_started.set()
            await self.allow_heartbeat.wait()
            return await super().heartbeat_execution(**kwargs)

        async def finish(self, **kwargs) -> ProductOperation:
            await self.heartbeat_started.wait()
            operation = await super().finish(**kwargs)
            self.allow_heartbeat.set()
            return operation

    monkeypatch.setattr(
        "cayu.server.service.PRODUCT_EXECUTION_HEARTBEAT_SECONDS",
        0.001,
    )
    store = SettlementBeforeHeartbeatStore()
    service, _store, _provider = _build_service(product_store=store)

    async def scenario() -> None:
        await store.reserve(
            tenant_id="tenant-a",
            idempotency_key="settlement-heartbeat-race",
            request_fingerprint=_product_request_fingerprint("work"),
            public_id="op_settlement_heartbeat",
            work_id="work_settlement_heartbeat",
            session_id="session_settlement_heartbeat",
            task_id="task_settlement_heartbeat",
            request_text="work",
        )

        completed = await asyncio.wait_for(
            service.execute_work("work_settlement_heartbeat"),
            timeout=2,
        )

        assert completed is not None
        assert completed.status == "completed"
        assert await store.heartbeat_execution(
            work_id="work_settlement_heartbeat",
            claim_id=store.execution_claims["work_settlement_heartbeat"][0],
            lease_seconds=120,
        )

    asyncio.run(scenario())


def test_execution_stops_before_a_blocked_heartbeat_can_outlive_its_lease(
    monkeypatch,
) -> None:
    class BlockedHeartbeatStore(MemoryProductStore):
        async def heartbeat_execution(self, **_kwargs) -> bool:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    monkeypatch.setattr(
        "cayu.server.service.PRODUCT_EXECUTION_HEARTBEAT_SECONDS",
        0.001,
    )
    monkeypatch.setattr(
        "cayu.server.service.PRODUCT_EXECUTION_HEARTBEAT_TIMEOUT_SECONDS",
        0.01,
    )
    store = BlockedHeartbeatStore()
    service, _store, _provider = _build_service(product_store=store)

    async def block_task_creation(_request) -> None:
        await asyncio.Event().wait()

    assert service.cayu_app.task_store is not None
    monkeypatch.setattr(service.cayu_app.task_store, "create_task", block_task_creation)

    async def scenario() -> None:
        await store.reserve(
            tenant_id="tenant-a",
            idempotency_key="heartbeat-timeout",
            request_fingerprint=_product_request_fingerprint("work"),
            public_id="op_heartbeat_timeout",
            work_id="work_heartbeat_timeout",
            session_id="session_heartbeat_timeout",
            task_id="task_heartbeat_timeout",
            request_text="work",
        )

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                service.execute_work("work_heartbeat_timeout"),
                timeout=2,
            )

        assert store.by_work_id["work_heartbeat_timeout"].status == "failed"

    asyncio.run(scenario())
