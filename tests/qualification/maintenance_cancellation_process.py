"""Fresh-process fixture: real generated worker and HTTP, controlled local runner.

Not Docker qualification. The only replaced owners are Docker admission/factory
and the stopped-process observation. Runtime execution/release, task receipts,
application intake, profiles, artifacts and HTTP reconciliation are real.
"""

import asyncio
import importlib
import importlib.util
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from cayu import ExecCommand, LocalRunner
from cayu.cli.project import project_context
from cayu.coding_products import CodingProductRunner
from cayu.evals.testing import ScriptedModelProvider
from cayu.providers.base import ModelStreamEvent
from cayu.runners.docker_workload import DockerImageIdentity
from cayu.storage.budget_ledger import SQLiteBudgetLedger
from cayu.tasks.queries import TaskQuery
from cayu.tasks.records import TaskStatus
from cayu.tasks.worker import run_task_worker
from tests.qualification.repository_maintenance_delivery_case import build_journey_http
from tests.qualification.repository_maintenance_toolchain import maintenance_toolchain
from tests.qualification.test_repository_maintenance_application import scripted_journey_budget


async def run(root, action, old_pid):
    project, source = root / "maintenance-app", root / "source"
    dsn = os.environ.get("CAYU_DATABASE_URL")
    if dsn and action == "start":
        from cayu.storage.migrations import SchemaMode
        from cayu.storage.postgres import PostgresSessionStore

        bootstrap = PostgresSessionStore(dsn, schema_mode=SchemaMode.CREATE)
        try:
            await bootstrap.ensure_schema()
        finally:
            await bootstrap.close()
    with project_context(project), pytest.MonkeyPatch.context() as patches:
        operations = importlib.import_module("operations.coding")
        profile = maintenance_toolchain(
            image_identity=DockerImageIdentity(reference="fixture@sha256:" + "a" * 64),
            architecture="amd64",
            build_context_sha256="sha256:" + "b" * 64,
        )
        patches.setattr(
            operations, "_configured_docker_authority", lambda _: (profile, "/usr/bin/docker")
        )
        spec = importlib.util.spec_from_file_location(
            "local_composition", project / "tests/test_coding_composition.py"
        )
        assert spec is not None and spec.loader is not None
        harness = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(harness)
        target = root / "target"
        target.mkdir(exist_ok=True)
        runners = []
        harness._install_fake_docker_factory(patches, target=target, created_runners=runners)
        original_exec = harness._LocalDockerRunner.exec

        async def execute(runner, command, **kwargs):
            if tuple(command.argv or ())[:1] == ("/opt/cayu-project/.venv/bin/pytest",):
                # Execute a real bounded local subprocess under the test adapter;
                # do not claim the pinned Docker toolchain was exercised.
                local = LocalRunner(runner.root)
                try:
                    return await local.exec(
                        ExecCommand.process(sys.executable, "-c", "print('checked once')")
                    )
                finally:
                    await local.close()
            return await original_exec(runner, command, **kwargs)

        patches.setattr(harness._LocalDockerRunner, "exec", execute)
        usage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="check", name="run_check", arguments={"check": "test"}
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls", "usage": usage}),
                ],
                [
                    ModelStreamEvent.text_delta("Stopped without a verified repair."),
                    ModelStreamEvent.completed({"finish_reason": "stop", "usage": usage}),
                ],
            ]
        )
        if dsn:
            from cayu.storage.budget_postgres import PostgresBudgetLedger

            ledger = PostgresBudgetLedger(dsn)
        else:
            ledger = SQLiteBudgetLedger(root / "budget.sqlite")
        application = importlib.import_module("app").build_coding_product_application(
            provider=provider,
            workspace_root=source,
            budget_policy=scripted_journey_budget(),
            budget_ledger=ledger,
        )
        stores = importlib.import_module("operations.maintenance_runs")
        registry = (
            stores.PostgresMaintenanceRunStore(dsn)
            if dsn
            else stores.SQLiteMaintenanceRunStore(root / "reservations.sqlite")
        )
        await registry.initialize()
        api = build_journey_http(application, registry, tenant="fixture", subject="operator")
        tasks = application.app.task_store
        headers = {"authorization": "Bearer fixture-operator"}
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api), base_url="http://fixture"
        ) as client:
            if action == "start":
                response = await client.post(
                    "/runs",
                    headers={"authorization": "Bearer fixture-product"},
                    json={
                        "instruction": "Inspect the admitted files.",
                        "idempotency_key": "one-task",
                    },
                )
                assert response.status_code == 202, response.text
                identity = await registry.load_owned(
                    tenant="fixture", public_id=response.json()["id"]
                )
                (root / "identity.json").write_text(identity.model_dump_json())

                async def pause(runner, *args, **kwargs):
                    # Real model/tool execution has settled, but publication and
                    # outer-task completion have not occurred. Never fake release.
                    inspected = await runner.inspect_settled_execution(args[0])
                    assert inspected.request_fingerprint
                    assert len(provider.requests) == 2
                    assert runners and all(item.closed for item in runners)
                    (root / "ready").write_text(str(os.getpid()))
                    await asyncio.Event().wait()

                patches.setattr(CodingProductRunner, "_compile_and_publish", pause)
                worker = importlib.import_module("operations.maintenance_worker")

                async def handler(_app, claimed, owner):
                    await worker.handle_coding_task(application, registry, claimed, owner)

                await run_task_worker(
                    application.app,
                    tasks,
                    handler,
                    worker_id="fixture-process",
                    query=TaskQuery(type="maintenance.coding"),
                    lease_seconds=1,
                    max_tasks=1,
                    recover_interrupted_handoffs=False,
                )
                raise AssertionError("Parent must kill the worker at the release barrier")
            identity_data = json.loads((root / "identity.json").read_text())
            identity = await registry.load_owned(
                tenant="fixture", public_id=identity_data["public_id"]
            )
            assert identity.model_dump(mode="json") == identity_data
            task = await tasks.load_task(identity.task_id)
            for _ in range(100):
                if task.lease_expires_at is None or task.lease_expires_at <= datetime.now(UTC):
                    break
                await asyncio.sleep(0.05)
            await tasks.reclaim_expired(query=TaskQuery(type="maintenance.coding"))
            module = importlib.import_module("operations.maintenance_reconciliation")

            async def stopped(owner):
                assert owner == "fixture-process"
                try:
                    os.kill(old_pid, 0)
                except ProcessLookupError:
                    return {"worker_id": owner, "container_id": "a" * 64, "generation": "b" * 64}
                raise AssertionError("Original process still alive")

            patches.setattr(module, "inspect_stopped_worker", stopped)
            path = f"/operator/runs/{identity.public_id}/coding/cancellation?tenant=fixture"
            events_before = await application.app.session_store.load_events(identity.session_id)
            if action == "ack-loss":
                assert task.status is TaskStatus.CLAIMED
                assert (
                    await tasks.claim_task(
                        "competitor", TaskQuery(type="maintenance.coding"), lease_seconds=1
                    )
                    is None
                )
                plan = await client.get(path, headers=headers)
                if plan.status_code != 200:
                    await module.inspect_cancellation(
                        application, registry, identity, actor_subject="operator"
                    )
                assert plan.status_code == 200, plan.text
                body = {
                    "plan_fingerprint": plan.json()["plan_fingerprint"],
                    "reconciliation_id": "one-receipt",
                }
                (root / "replay.json").write_text(json.dumps(body))
                reconcile = tasks.reconcile_task_cancellation

                async def lost(request):
                    await reconcile(request)
                    raise ConnectionError("accepted cancellation acknowledgement lost")

                patches.setattr(tasks, "reconcile_task_cancellation", lost)
                assert (await client.post(path, headers=headers, json=body)).status_code == 503
            else:
                body = json.loads((root / "replay.json").read_text())
                response = await client.post(path, headers=headers, json=body)
                assert response.status_code == 200, response.text
                assert (
                    await client.post(path, headers=headers, json=body)
                ).json() == response.json()
                assert (
                    await client.post(
                        path, headers=headers, json={**body, "reconciliation_id": "different"}
                    )
                ).status_code == 409
                public = await client.get(
                    f"/runs/{identity.public_id}",
                    headers={"authorization": "Bearer fixture-product"},
                )
                assert public.json()["coding_task_status"] == "cancelled"
                assert (
                    await client.get(
                        f"/operator/runs/{identity.public_id}/result?tenant=fixture",
                        headers=headers,
                    )
                ).status_code == 409
                cost = await client.get(
                    f"/operator/runs/{identity.public_id}/cost?tenant=fixture", headers=headers
                )
                assert cost.status_code == 200 and cost.json()["model_steps"] == 2, cost.text
                (root / "settled.json").write_text(json.dumps(response.json()))
            terminal = await tasks.load_task(identity.task_id)
            assert terminal.status is TaskStatus.CANCELLED and terminal.result is None
            assert terminal.worker_id is None and terminal.lease_expires_at is None
            if action == "ack-loss":
                (root / "committed-task.json").write_text(terminal.model_dump_json())
            else:
                assert terminal.model_dump(mode="json") == json.loads(
                    (root / "committed-task.json").read_text()
                )
            assert (
                await application.app.session_store.load_events(identity.session_id)
                == events_before
            )
            assert not provider.requests
        await application.app.aclose()


if __name__ == "__main__":
    asyncio.run(run(Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])))
