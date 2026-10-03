"""Execute the shipped managed-handler fragment, not a coding acceptance stand-in."""

import asyncio
import json
import re
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from cayu import CayuApp, InMemoryTaskStore, SQLiteTaskStore, TaskCreate, TaskQuery, TaskStatus
from cayu.cli import main


def test_host_guidance_is_discoverable_and_preserves_application_ownership(capsys):
    assert main(["guide", "authoring#coding-composition"]) == 0
    composition = capsys.readouterr().out
    assert "cayu guide authoring#coding-product-host" in composition
    assert "not a prohibition" in composition
    assert main(["guide", "authoring#coding-product-host", "--json"]) == 0
    guide = json.loads(capsys.readouterr().out)
    assert guide["package_source"] == "cayu.guides/authoring.md"
    content = guide["content"].replace("\n", " ")
    for boundary in (
        "authenticated tenant and public ID",
        "not an atomic reservation",
        "run_task_worker",
        "complete_managed_task",
        "invocation-release evidence",
        "Do not label cancellation settlement as a verified repair",
        "settles the original business reservation",
        "make replay of the same receipt idempotent",
        "Native session recovery alone does not perform this application transaction",
    ):
        assert boundary in content
    assert "## 2. Use the project factory" not in content


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("lose_ack", [False, True])
def test_coding_host_fragment_uses_managed_completion_and_durable_readback(
    backend, lose_ack, tmp_path, capsys
):
    assert main(["guide", "authoring#coding-product-host", "--json"]) == 0
    content = json.loads(capsys.readouterr().out)["content"]
    snippets = re.findall(r"```python\n(.*?)```", content, re.DOTALL)
    assert len(snippets) == 2
    flat = content.replace("\n", " ")
    assert "does not create a caller-keyed `TaskTerminalizationReceipt`" in flat
    assert "Extending the lease duration does not remove the race" in flat
    assert "diagnostics#external-tool-coverage-unknown" in flat
    assert "module import expressions" in flat
    assert "command journal before destructive disposal" in flat
    assert "Changing `max_steps` from eight to one" in flat
    assert "not an observation-only switch" in flat
    assert "may continue the model" in flat

    async def scenario():
        store = (
            InMemoryTaskStore() if backend == "memory" else SQLiteTaskStore(tmp_path / "tasks.db")
        )
        app = CayuApp(task_store=store, enable_logging=False)
        renewed = asyncio.Event()
        leases = []
        verification_calls = []
        completion_calls = []
        failures = []
        expected = {"product_run_id": "retained-product", "result_digest": "a" * 64}
        original_heartbeat = store.heartbeat
        original_complete = store.complete_task
        lost = ConnectionError("Completion committed before reply")

        async def heartbeat(*args, **kwargs):
            updated = await original_heartbeat(*args, **kwargs)
            leases.append(updated.lease_expires_at)
            if len(leases) >= 2:
                renewed.set()
            return updated

        async def complete(*args, **kwargs):
            completion_calls.append(kwargs["lease_expires_at"])
            result = await original_complete(*args, **kwargs)
            if lose_ack:
                raise lost
            return result

        async def verify_product_for_claim(_app, claimed):
            verification_calls.append(claimed.id)
            await asyncio.wait_for(renewed.wait(), timeout=5)
            assert leases[-1] != claimed.lease_expires_at
            # Domain acceptance is deliberately characterized here; this test
            # proves only the documented task-owner composition.
            return SimpleNamespace(**expected)

        namespace: dict[str, Any] = {"verify_product_for_claim": verify_product_for_claim}
        exec(compile("\n\n".join(snippets), "authoring#coding-product-host", "exec"), namespace)

        async def handler(app, claimed, worker):
            try:
                await namespace["handle_coding_task"](app, claimed, worker)
            except ConnectionError as error:
                assert error is lost and lose_ack
                failures.append(error)
                # Do not repeat the handler after a lost acknowledgement.
                completed = await store.load_task(claimed.id)
                assert completed is not None and completed.status is TaskStatus.COMPLETED

        from cayu import run_task_worker

        try:
            created = await app.create_task(TaskCreate(task_id="guide-task", type="guide-coding"))
            with (
                patch.object(store, "heartbeat", heartbeat),
                patch.object(store, "complete_task", complete),
            ):
                assert (
                    await run_task_worker(
                        app,
                        store,
                        handler,
                        worker_id="guide-worker",
                        query=TaskQuery(type="guide-coding"),
                        lease_seconds=3,
                        max_tasks=1,
                        reclaim=False,
                    )
                    == 1
                )
            assert completion_calls == [leases[-1]]
            assert verification_calls == [created.id]
            assert failures == ([lost] if lose_ack else [])
            if isinstance(store, SQLiteTaskStore):
                await store.close()
                store = SQLiteTaskStore(tmp_path / "tasks.db")
            retained = await store.load_task(created.id)
            assert retained is not None
            assert retained.status is TaskStatus.COMPLETED
            assert retained.result == expected
            assert retained.worker_id is None and retained.lease_expires_at is None
            assert await store.load_task(created.id) == retained
            assert verification_calls == [created.id]
        finally:
            if isinstance(store, SQLiteTaskStore):
                await store.close()

    asyncio.run(scenario())
