"""Fresh generated-consumer processes around a durable Git approval request.

SQLite/native Git are real. Docker admission is fixture-controlled; neither
coding execution nor production Docker/PostgreSQL qualification is claimed here.
"""

import asyncio
import importlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import httpx
from tests.qualification.repository_maintenance_delivery_case import build_journey_http
from worker_harness import _write_json_atomic

from cayu import (
    BudgetPolicy,
    DockerCodingToolchainProfile,
    ScriptedModelProvider,
    TaskQuery,
    TaskStatus,
    run_task_worker,
)
from cayu.cli.project import project_context
from cayu.storage import SQLiteBudgetLedger


async def run_maintenance_approval(config):
    with project_context(Path(config["project"])):
        operations = importlib.import_module("operations.coding")
        profile = DockerCodingToolchainProfile.model_validate_json(config["toolchain_json"])
        provider = ScriptedModelProvider([])
        with patch.object(
            operations, "_configured_docker_authority", lambda root: (profile, "/usr/bin/docker")
        ):
            application = importlib.import_module("app").build_coding_product_application(
                provider=provider,
                workspace_root=Path(config["source"]),
                budget_policy=BudgetPolicy.model_validate_json(config["budget_json"]),
                budget_ledger=SQLiteBudgetLedger(config["budget_path"]),
            )
        try:
            reservations = importlib.import_module(
                "operations.maintenance_runs"
            ).SQLiteMaintenanceRunStore(config["reservation_path"])
            identity = await reservations.load_owned(
                tenant=config["tenant"], public_id=config["public_id"]
            )
            assert identity is not None and identity.model_dump(mode="json") == config["identity"]
            assert str(application.app.session_store.path) == config["backend"]["session_path"]
            assert str(application.app.task_store.path) == config["backend"]["task_path"]
            api = build_journey_http(
                application,
                reservations,
                tenant=identity.intent.tenant,
                subject=identity.intent.subject,
            )
            path = f"/operator/runs/{identity.public_id}/git/approval"
            headers = {"authorization": "Bearer fixture-operator"}
            params = {"tenant": identity.intent.tenant}
            if config["action"] == "start":
                with patch.dict(
                    os.environ, {"CAYU_MAINTENANCE_GIT_HOST_JSON": json.dumps(config["git_host"])}
                ):
                    broker = importlib.import_module(
                        "integrations.maintenance_git_host"
                    ).configured_git_broker(application.artifact_store)
                worker = importlib.import_module("operations.maintenance_git_worker")

                async def prepare(_app, task, worker_id):
                    await worker.handle_git_preparation_task(
                        application, reservations, task, worker_id, broker
                    )

                assert (
                    await run_task_worker(
                        application.app,
                        application.app.task_store,
                        prepare,
                        worker_id="approval-process-producer",
                        query=TaskQuery(type="maintenance.git_preparation"),
                        max_tasks=1,
                        recover_interrupted_handoffs=False,
                    )
                    == 1
                )
            completed = await application.app.task_store.load_task(identity.git_preparation_task_id)
            assert completed.status is TaskStatus.COMPLETED
            assert completed.worker_id is None and completed.lease_expires_at is None
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=api), base_url="http://fixture"
            ) as client:
                response = await client.get(path, headers=headers, params=params)
                assert response.status_code == 200, response.text
                review = response.json()
                assert review["pending_result_digest"] == completed.result["result_digest"]
                assert not provider.requests
                if config["action"] == "start":
                    assert (
                        await application.app.task_store.load_task(identity.git_delivery_task_id)
                        is None
                    )
                    _write_json_atomic(
                        Path(config["phase_path"]),
                        {
                            "phase": "maintenance_approval_durable",
                            "review": review,
                            "preparation": completed.model_dump(mode="json"),
                        },
                    )
                    await asyncio.Event().wait()
                    raise AssertionError(
                        "Producer must be terminated at the durable approval barrier"
                    )
                assert review == config["expected_review"]
                assert completed.model_dump(mode="json") == config["expected_preparation"]
                body = {
                    "request_fingerprint": review["request_fingerprint"],
                    "prepared_tree": review["prepared_tree"],
                    "approval_id": "fixture-human-decision",
                }
                stale = await client.post(
                    path, headers=headers, params=params, json=dict(body, prepared_tree="0" * 40)
                )
                assert stale.status_code == 409
                assert (
                    await application.app.task_store.load_task(identity.git_delivery_task_id)
                    is None
                )
                created = await client.post(path, headers=headers, params=params, json=body)
                assert created.status_code == 202, created.text
                queued = await application.app.task_store.load_task(identity.git_delivery_task_id)
                assert queued.status is TaskStatus.PENDING and queued.worker_id is None
                assert (
                    await client.post(path, headers=headers, params=params, json=body)
                ).status_code == 202
                assert await application.app.task_store.load_task(queued.id) == queued
                assert not provider.requests
                return {
                    "review": review,
                    "approved_task": queued.model_dump(mode="json"),
                    "provider_calls": 0,
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
            await application.app.session_store.close()
            await application.app.task_store.close()
            await application.app.knowledge_store.close()
            await application.app.budget_ledger.close()
