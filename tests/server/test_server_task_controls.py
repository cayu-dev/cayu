"""HTTP server task controls contracts."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")
pytest.importorskip("httpx")

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from fastapi.testclient import TestClient
from tests.core.task_invocation_fixtures import task_backed_session_invocation
from tests.server.server_test_support import _LOCAL_SERVER_CONFIG

from cayu.applications import CayuApp
from cayu.server import create_server
from cayu.tasks.contracts import (
    CompletionResultResolverRef,
    CompletionVerifierRef,
    WorkContractDraft,
    WorkCriterion,
    work_contract_from_draft,
)
from cayu.tasks.creation import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.records import TaskRetryPolicy, TaskStatus
from cayu.tasks.retry import TaskRetryAttemptDisposition, TaskRetrySettlementRequest
from cayu.vaults.redaction import SecretRedactor


def test_server_task_list_exposes_worker_lease_state() -> None:
    task_store = InMemoryTaskStore()

    async def setup_task() -> None:
        await task_store.create_task(
            TaskCreate(
                task_id="leased_task",
                type="review",
                assigned_agent_name="assistant",
            )
        )
        claimed = await task_store.claim_task("worker_a", lease_seconds=300)
        assert claimed is not None

    asyncio.run(setup_task())

    client = TestClient(create_server(CayuApp(task_store=task_store), config=_LOCAL_SERVER_CONFIG))
    tasks = client.get("/api/tasks").json()

    assert len(tasks) == 1
    assert tasks[0]["id"] == "leased_task"
    assert tasks[0]["type"] == "review"
    assert tasks[0]["title"] is None
    assert tasks[0]["status"] == "claimed"
    assert tasks[0]["status_reason"] is None
    assert tasks[0]["status_payload"] is None
    assert tasks[0]["session_id"] is None
    assert tasks[0]["parent_task_id"] is None
    assert tasks[0]["assigned_agent_name"] == "assistant"
    assert tasks[0]["worker_id"] == "worker_a"
    assert tasks[0]["completed_at"] is None
    assert isinstance(tasks[0]["lease_expires_at"], str)
    assert isinstance(tasks[0]["created_at"], str)
    assert tasks[0]["description"] is None
    assert "input" not in tasks[0]
    assert "result" not in tasks[0]
    assert "error" not in tasks[0]
    assert "metadata" not in tasks[0]
    assert isinstance(tasks[0]["updated_at"], str)


def test_server_task_status_omits_private_work_contract_fingerprint() -> None:
    secret = "private-verification-threshold"
    task_store = InMemoryTaskStore()
    contract = work_contract_from_draft(
        WorkContractDraft(
            contract_id="private-status-contract",
            version=1,
            objective=f"Approve only when the score exceeds {secret}.",
            criteria=(
                WorkCriterion(
                    criterion_id="threshold",
                    ordinal=1,
                    description=f"Confirm the private threshold {secret}.",
                ),
            ),
            verifier=CompletionVerifierRef(
                verifier_id="private-status-verifier",
                version="v1",
                configuration_fingerprint="0" * 64,
            ),
            result_resolver=CompletionResultResolverRef(
                resolver_id="private-status-result",
                version="v1",
                configuration_fingerprint="1" * 64,
            ),
        )
    )

    async def setup_task() -> None:
        await task_store.publish_work_contract(contract)
        task = await task_store.create_task(
            TaskCreate(
                task_id="private-status-task",
                type="verified-work",
                work_contract=contract.reference(),
            )
        )
        claimed = await task_store.claim_task("ordinary-worker")
        assert claimed is not None
        assert claimed.id == task.id
        assert claimed.lease_expires_at is not None
        await task_store.hold_claimed_work_contract_task(
            task.id,
            worker_id="ordinary-worker",
            lease_expires_at=claimed.lease_expires_at,
            contract=contract.reference(),
        )

    asyncio.run(setup_task())
    client = TestClient(
        create_server(
            CayuApp(
                task_store=task_store,
                secret_redactor=SecretRedactor(secret),
            ),
            config=_LOCAL_SERVER_CONFIG,
        )
    )

    list_response = client.get("/api/tasks")
    detail_response = client.get("/api/tasks/private-status-task")
    assert list_response.status_code == 200
    assert detail_response.status_code == 200
    listed = list_response.json()[0]
    detailed = detail_response.json()
    expected_status = {
        "contract_id": contract.contract_id,
        "contract_version": contract.version,
    }
    assert listed["status_payload"] == expected_status
    assert detailed["status_payload"] == expected_status
    for rendered in (list_response.text, detail_response.text):
        assert secret not in rendered
        assert contract.fingerprint not in rendered


def test_server_task_endpoints_serialize_availability_in_canonical_utc() -> None:
    task_store = InMemoryTaskStore()
    local_time = datetime(
        2026,
        8,
        8,
        9,
        30,
        tzinfo=timezone(timedelta(hours=5, minutes=30)),
    )
    asyncio.run(
        task_store.create_task(
            TaskCreate(
                task_id="scheduled_task",
                type="review",
                available_at=local_time,
            )
        )
    )

    client = TestClient(create_server(CayuApp(task_store=task_store), config=_LOCAL_SERVER_CONFIG))
    task_list = client.get("/api/tasks")
    task_detail = client.get("/api/tasks/scheduled_task")

    assert task_list.status_code == 200
    assert task_detail.status_code == 200
    assert task_list.json()[0]["available_at"] == "2026-08-08T04:00:00+00:00"
    assert task_detail.json()["available_at"] == "2026-08-08T04:00:00+00:00"


def test_server_task_endpoints_project_retry_series_authority() -> None:
    task_store = InMemoryTaskStore()
    asyncio.run(
        task_store.create_task(
            TaskCreate(
                task_id="retry_task",
                type="review",
                retry_policy=TaskRetryPolicy(
                    max_attempts=3,
                    max_elapsed_seconds=60,
                    max_total_tokens=100,
                    max_estimated_cost=Decimal("1.25"),
                    cost_currency="usd",
                    initial_backoff_seconds=2,
                ),
            )
        )
    )

    client = TestClient(create_server(CayuApp(task_store=task_store), config=_LOCAL_SERVER_CONFIG))
    projected = client.get("/api/tasks").json()[0]["retry_series"]
    detail = client.get("/api/tasks/retry_task").json()["retry_series"]

    assert detail == projected
    assert projected["attempt"] == 1
    assert projected["causal_budget_id"] == projected["series_id"]
    assert projected["attempts_remaining"] == 2
    assert projected["tokens_remaining"] == "100"
    assert projected["estimated_cost_remaining"] == "1.25"
    assert projected["elapsed_deadline"] is not None
    assert projected["next_eligible_at"] is None
    assert projected["disposition"] == "active"
    assert projected["policy"]["cost_currency"] == "USD"


@pytest.mark.parametrize("secret", ["1.25", "0"])
def test_server_task_retry_numeric_authority_survives_secret_collisions(secret: str) -> None:
    task_store = InMemoryTaskStore()
    asyncio.run(
        task_store.create_task(
            TaskCreate(
                task_id=f"retry_numeric_secret_{secret.replace('.', '_')}",
                type="review",
                retry_policy=TaskRetryPolicy(
                    max_attempts=2,
                    max_total_tokens=100,
                    max_estimated_cost=Decimal("1.25"),
                ),
            )
        )
    )
    app = CayuApp(
        task_store=task_store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    list_response = client.get("/api/tasks")
    detail_response = client.get(f"/api/tasks/retry_numeric_secret_{secret.replace('.', '_')}")

    assert list_response.status_code == 200
    assert detail_response.status_code == 200
    for projected in (
        list_response.json()[0]["retry_series"],
        detail_response.json()["retry_series"],
    ):
        assert projected["cumulative_tokens"] == "0"
        assert projected["tokens_remaining"] == "100"
        assert projected["cumulative_estimated_cost"] == "0"
        assert projected["estimated_cost_remaining"] == "1.25"
        assert projected["policy"]["max_total_tokens"] == "100"
        assert projected["policy"]["max_estimated_cost"] == "1.25"


def test_server_task_retry_series_redacts_opaque_causal_budget_authority() -> None:
    secret = "TASK_RETRY_CAUSAL_BUDGET_SECRET_CANARY"
    task_store = InMemoryTaskStore()
    task = asyncio.run(
        task_store.create_task(
            TaskCreate(
                task_id="retry_secret",
                type="review",
                retry_policy=TaskRetryPolicy(max_attempts=2),
            )
        )
    )
    assert task.retry_series is not None
    task_store._tasks[task.id] = task.model_copy(
        update={"retry_series": task.retry_series.model_copy(update={"causal_budget_id": secret})}
    )
    app = CayuApp(
        task_store=task_store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    response = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG)).get(
        "/api/tasks/retry_secret"
    )

    assert response.status_code == 200
    assert response.json()["retry_series"]["causal_budget_id"] == "[REDACTED_SECRET]"
    assert secret not in response.text


def test_server_task_retry_series_redacts_caller_controlled_currency(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    secret = "secretcurrency"
    task_store = InMemoryTaskStore()

    async def create_and_settle() -> None:
        await task_store.create_task(
            TaskCreate(
                task_id="retry_currency_secret",
                type="review",
                retry_policy=TaskRetryPolicy(
                    max_attempts=2,
                    cost_currency=secret,
                ),
            )
        )
        claimed = await task_store.claim_task("currency-worker")
        assert claimed is not None
        assert claimed.retry_series is not None
        await task_store.settle_task_retry_attempt(
            TaskRetrySettlementRequest(
                task_id=claimed.id,
                worker_id="currency-worker",
                lease_expires_at=claimed.lease_expires_at,
                idempotency_key="currency-settlement",
                causal_budget_id=claimed.retry_series.causal_budget_id,
                disposition=TaskRetryAttemptDisposition.SUCCEEDED,
                result={"ok": True},
            )
        )

    asyncio.run(create_and_settle())
    app = CayuApp(
        task_store=task_store,
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    responses = [
        client.get("/api/tasks"),
        client.get("/api/tasks/retry_currency_secret"),
    ]

    assert all(response.status_code == 200 for response in responses)
    assert responses[0].json()[0]["retry_series"]["policy"]["cost_currency"] == (
        "[REDACTED_SECRET]"
    )
    assert responses[1].json()["retry_series"]["policy"]["cost_currency"] == ("[REDACTED_SECRET]")
    assert responses[0].json()[0]["status_payload"]["cost_currency"] == "[REDACTED_SECRET]"
    assert responses[1].json()["status_payload"]["cost_currency"] == "[REDACTED_SECRET]"
    captured = capsys.readouterr()
    diagnostic_text = "\n".join(
        [
            *(record.getMessage() for record in caplog.records),
            *(str(warning.message) for warning in recwarn),
            captured.out,
            captured.err,
            *(response.text for response in responses),
        ]
    )
    assert secret not in diagnostic_text.casefold()


def test_server_task_list_filters_lifecycle_states() -> None:
    task_store = InMemoryTaskStore()

    async def setup_task() -> None:
        await task_store.create_task(
            TaskCreate(
                task_id="blocked_task",
                type="review",
                assigned_agent_name="assistant",
            )
        )
        await task_store.create_task(
            TaskCreate(
                task_id="ready_task",
                type="review",
                assigned_agent_name="assistant",
            )
        )
        await task_store.block_task(
            "blocked_task",
            reason="Waiting on upstream import",
            payload={"dependency": "import_123"},
        )

    asyncio.run(setup_task())

    client = TestClient(create_server(CayuApp(task_store=task_store), config=_LOCAL_SERVER_CONFIG))
    response = client.get(
        "/api/tasks",
        params={
            "status": TaskStatus.BLOCKED.value,
            "type": "review",
            "assigned_agent_name": "assistant",
        },
    )

    assert response.status_code == 200
    tasks = response.json()
    assert [task["id"] for task in tasks] == ["blocked_task"]
    assert tasks[0]["status"] == "blocked"
    assert tasks[0]["status_reason"] == "Waiting on upstream import"
    assert tasks[0]["status_payload"] == {"dependency": "import_123"}

    oldest_first_response = client.get(
        "/api/tasks",
        params={"order_by": "created_at_asc"},
    )
    assert oldest_first_response.status_code == 200
    assert [task["id"] for task in oldest_first_response.json()] == [
        "blocked_task",
        "ready_task",
    ]

    search_response = client.get("/api/tasks", params={"q": "upstream"})
    assert search_response.status_code == 200
    assert [task["id"] for task in search_response.json()] == ["blocked_task"]


def test_server_task_detail_returns_full_payload() -> None:
    task_store = InMemoryTaskStore()

    async def setup_task() -> None:
        await task_store.create_task(
            TaskCreate(
                task_id="detail_task",
                type="review",
                title="Inspect detail",
                description="Full task payload should stay off the list endpoint.",
                input={"document": "invoice.pdf", "amount": 42},
                metadata={"tenant": "acme", "priority": "high"},
            )
        )
        await task_store.start_task(
            "detail_task",
            session_id="sess_detail",
            session_invocation=await task_backed_session_invocation(
                task_store,
                "detail_task",
                "sess_detail",
            ),
        )
        await task_store.complete_task("detail_task", {"accepted": True})

    asyncio.run(setup_task())

    client = TestClient(create_server(CayuApp(task_store=task_store), config=_LOCAL_SERVER_CONFIG))

    list_response = client.get("/api/tasks")
    assert list_response.status_code == 200
    list_task = list_response.json()[0]
    assert list_task["id"] == "detail_task"
    assert "input" not in list_task
    assert "result" not in list_task
    assert "error" not in list_task
    assert "metadata" not in list_task
    assert "started_at" not in list_task

    detail_response = client.get("/api/tasks/detail_task")
    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert detail["id"] == "detail_task"
    assert detail["description"] == "Full task payload should stay off the list endpoint."
    assert detail["session_id"] == "sess_detail"
    assert detail["input"] == {"document": "invoice.pdf", "amount": 42}
    assert detail["result"] == {"accepted": True}
    assert detail["error"] is None
    assert detail["metadata"] == {"tenant": "acme", "priority": "high"}
    assert isinstance(detail["started_at"], str)


def test_server_task_detail_reports_missing_store_and_task() -> None:
    missing_store_client = TestClient(create_server(CayuApp(), config=_LOCAL_SERVER_CONFIG))
    missing_store_response = missing_store_client.get("/api/tasks/task_1")
    assert missing_store_response.status_code == 404
    assert missing_store_response.json()["detail"] == "Task store is not configured."

    task_store = InMemoryTaskStore()
    client = TestClient(create_server(CayuApp(task_store=task_store), config=_LOCAL_SERVER_CONFIG))
    missing_task_response = client.get("/api/tasks/missing_task")
    assert missing_task_response.status_code == 404
    assert missing_task_response.json()["detail"] == "Task not found: missing_task"


def test_server_task_lifecycle_endpoints_hold_and_resume_tasks() -> None:
    task_store = InMemoryTaskStore()

    async def setup_task() -> None:
        await task_store.create_task(
            TaskCreate(
                task_id="review_task",
                type="review",
                input={"document": "invoice.pdf"},
                metadata={"tenant": "acme"},
            )
        )

    asyncio.run(setup_task())

    client = TestClient(create_server(CayuApp(task_store=task_store), config=_LOCAL_SERVER_CONFIG))
    block_response = client.post(
        "/api/tasks/review_task/block",
        json={
            "reason": "Waiting on operator",
            "payload": {"queue": "ops"},
        },
    )

    assert block_response.status_code == 200
    blocked = block_response.json()
    assert blocked["id"] == "review_task"
    assert blocked["status"] == "blocked"
    assert blocked["status_reason"] == "Waiting on operator"
    assert blocked["status_payload"] == {"queue": "ops"}
    assert blocked["input"] == {"document": "invoice.pdf"}
    assert blocked["metadata"] == {"tenant": "acme"}

    list_response = client.get("/api/tasks", params={"status": "blocked"})
    assert list_response.status_code == 200
    listed_tasks = list_response.json()
    assert [task["id"] for task in listed_tasks] == ["review_task"]
    assert "input" not in listed_tasks[0]
    assert "metadata" not in listed_tasks[0]

    resume_response = client.post("/api/tasks/review_task/resume")

    assert resume_response.status_code == 200
    resumed = resume_response.json()
    assert resumed["status"] == "pending"
    assert resumed["status_reason"] is None
    assert resumed["status_payload"] is None


def test_server_task_lifecycle_endpoints_support_pause_and_needs_attention() -> None:
    task_store = InMemoryTaskStore()

    async def setup_tasks() -> None:
        await task_store.create_task(TaskCreate(task_id="pause_task", type="review"))
        await task_store.create_task(TaskCreate(task_id="attention_task", type="review"))

    asyncio.run(setup_tasks())

    client = TestClient(create_server(CayuApp(task_store=task_store), config=_LOCAL_SERVER_CONFIG))

    pause_response = client.post(
        "/api/tasks/pause_task/pause",
        json={"reason": "Worker maintenance"},
    )
    assert pause_response.status_code == 200
    assert pause_response.json()["status"] == "paused"
    assert pause_response.json()["status_reason"] == "Worker maintenance"

    attention_response = client.post(
        "/api/tasks/attention_task/needs-attention",
        json={"payload": {"field": "amount"}},
    )
    assert attention_response.status_code == 200
    assert attention_response.json()["status"] == "needs_attention"
    assert attention_response.json()["status_payload"] == {"field": "amount"}


def test_server_task_lifecycle_endpoints_report_invalid_transitions() -> None:
    task_store = InMemoryTaskStore()

    async def setup_task() -> None:
        await task_store.create_task(TaskCreate(task_id="attached_task", type="review"))
        await task_store.start_task(
            "attached_task",
            session_id="sess_attached",
            session_invocation=await task_backed_session_invocation(
                task_store,
                "attached_task",
                "sess_attached",
            ),
        )

    asyncio.run(setup_task())

    client = TestClient(create_server(CayuApp(task_store=task_store), config=_LOCAL_SERVER_CONFIG))

    hold_response = client.post(
        "/api/tasks/attached_task/block",
        json={"reason": "not allowed"},
    )
    assert hold_response.status_code == 409
    assert "already attached to session sess_attached" in hold_response.json()["detail"]

    resume_response = client.post("/api/tasks/attached_task/resume")
    assert resume_response.status_code == 409
    assert "not paused, blocked, or waiting for attention" in resume_response.json()["detail"]


def test_server_task_lifecycle_endpoints_report_missing_task_store_and_task() -> None:
    missing_store_client = TestClient(create_server(CayuApp(), config=_LOCAL_SERVER_CONFIG))

    missing_store_response = missing_store_client.post("/api/tasks/task_1/block")
    assert missing_store_response.status_code == 404
    assert missing_store_response.json()["detail"] == "Task store is not configured."

    task_store = InMemoryTaskStore()
    client = TestClient(create_server(CayuApp(task_store=task_store), config=_LOCAL_SERVER_CONFIG))

    missing_task_response = client.post("/api/tasks/missing_task/block")
    assert missing_task_response.status_code == 404
    assert "missing_task" in missing_task_response.json()["detail"]


def test_server_task_lifecycle_endpoints_validate_request_body() -> None:
    task_store = InMemoryTaskStore()

    async def setup_task() -> None:
        await task_store.create_task(TaskCreate(task_id="task_1", type="review"))

    asyncio.run(setup_task())

    client = TestClient(create_server(CayuApp(task_store=task_store), config=_LOCAL_SERVER_CONFIG))
    response = client.post("/api/tasks/task_1/block", json={"reason": "   "})

    assert response.status_code == 422
