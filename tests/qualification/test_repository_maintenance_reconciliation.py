"""Emitted operator/store boundaries with explicitly controlled host evidence."""

import asyncio
import importlib
from types import SimpleNamespace

import pytest

from cayu import Event, EventType, Message, TaskStatus
from cayu.sessions.transcript_input import session_input_messages_sha256
from tests.qualification.test_repository_maintenance_http import _A, _BODY, _OP, client
from tests.qualification.test_repository_maintenance_http import host as host
from tests.qualification.test_repository_maintenance_request import consumer as consumer
from tests.qualification.test_repository_maintenance_request import project as project

_WORKSPACE_TOOLS = (
    "read_file",
    "write_file",
    "edit_file",
    "apply_patch",
    "delete_file",
    "list_files",
    "search_text",
    "git_changes",
)


def _events():
    return tuple(
        Event(type=kind, session_id="fixture", payload=payload, tool_name=tool)
        for kind, payload, tool in (
            (EventType.MODEL_STARTED, {"step": 1, "attempt": 1, "model": "fixture"}, None),
            (
                EventType.MODEL_COMPLETED,
                {
                    "step": 1,
                    "attempt": 1,
                    "model": "fixture",
                    "status": "completed",
                    "completion": {"finish_reason": "tool_calls", "status": "completed"},
                },
                None,
            ),
            (EventType.TOOL_CALL_STARTED, {}, "run_check"),
            (
                EventType.TOOL_CALL_FAILED,
                {
                    "result": {
                        "structured": {
                            "status": "timed_out",
                            "workspace_mutation_settlement": "complete",
                            "cleanup_uncertain": False,
                        }
                    }
                },
                "run_check",
            ),
        )
    )


@pytest.mark.parametrize("consumer", ["memory", "sqlite"], indirect=True)
@pytest.mark.parametrize("instruction_prefix", ["", "sha256:"])
@pytest.mark.parametrize("completed_file", [None, *_WORKSPACE_TOOLS])
def test_operator_cancellation_exact_receipt_and_ack_loss(
    host, monkeypatch, instruction_prefix, completed_file
):
    server, application, registry, provider, _ = host
    module = importlib.import_module("operations.maintenance_reconciliation")

    async def scenario():
        async with client(server) as http:
            intake = await http.post("/runs", headers=_A, json=_BODY)
            identity = await registry.load_owned(tenant="tenant-a", public_id=intake.json()["id"])
            store = application.app.task_store
            claimed = await store.claim_task("controlled-stopped-owner", lease_seconds=1)
            await store.mark_claimed_task_execution_started(
                claimed.id, claimed.worker_id, claimed.lease_expires_at
            )
            await store.cancel_task(claimed.id, {"code": "operator"})
            await asyncio.sleep(1.05)
            held = await store.load_task(claimed.id)
            accepted = module.decode_request(identity.intent.request_json)
            saved = SimpleNamespace(
                product_run_id=identity.product_run_id,
                session_id=identity.session_id,
                agent_name=application.agent_name,
                settlement=module.CodingSettlementPolicy.model_validate_json(
                    accepted.settlement_json
                ),
                fingerprint="sha256:" + "a" * 64,
                task=SimpleNamespace(
                    task_id=identity.task_id,
                    instruction_sha256=instruction_prefix
                    + session_input_messages_sha256([Message.text("user", accepted.instruction)]),
                ),
                source=SimpleNamespace(
                    workspace_id=accepted.source_workspace_id,
                    origin_id=accepted.source_origin_id,
                    destination_id=accepted.source_destination_id,
                    git_baseline=SimpleNamespace(head_revision=accepted.base_revision),
                ),
                parent_session_id=identity.workflow_session_id,
                causal_budget_id=identity.workflow_session_id,
                runtime=SimpleNamespace(
                    execution_profile_fingerprint=accepted.execution_profile_fingerprint,
                    toolchain_profile_fingerprint=accepted.toolchain_profile_fingerprint,
                ),
            )

            async def request(*args, **kwargs):
                return saved

            async def stopped(worker):
                assert worker == claimed.worker_id
                return {"worker_id": worker}

            events = _events()
            if completed_file is not None:
                events = (*events[:2], *_file_events(completed_file), *events[2:])
            inspection = SimpleNamespace(
                events=events,
                request_fingerprint=saved.fingerprint,
                release_fingerprint="sha256:" + "c" * 64,
                tool_call_ordinals=_fixture_ordinals(events),
            )

            async def released(self, request):
                assert request is saved
                return inspection

            monkeypatch.setattr(module, "inspect_stopped_worker", stopped)
            monkeypatch.setattr(module.CodingProductArtifactRepository, "load_request", request)
            monkeypatch.setattr(module.CodingProductRunner, "inspect_settled_execution", released)
            url = f"/operator/runs/{identity.public_id}/coding/cancellation?tenant=tenant-a"
            for headers in ({}, _A):
                assert (await http.get(url, headers=headers)).status_code == 401
            assert (
                await http.get(url.replace("tenant-a", "tenant-b"), headers=_OP)
            ).status_code == 404
            assert (await http.post(url, headers=_OP, json={"quiescent": True})).status_code == 422
            plan = await http.get(url, headers=_OP)
            assert plan.status_code == 200, plan.text
            body = {
                "plan_fingerprint": plan.json()["plan_fingerprint"],
                "reconciliation_id": "settle-once",
            }
            assert (
                await http.post(url, headers=_OP, json={**body, "tool_call_ordinals": [0, 0]})
            ).status_code == 422
            original_ordinals = inspection.tool_call_ordinals
            inspection.tool_call_ordinals = tuple(
                None if ordinal is None else ordinal + 1 for ordinal in original_ordinals
            )
            # Both transcripts are internally correlated, but the old plan must
            # not authorize a different inspected dispatch-correlation tuple.
            assert (await http.get(url, headers=_OP)).status_code == 200
            assert (await http.post(url, headers=_OP, json=body)).status_code == 409
            assert await store.load_task(claimed.id) == held
            inspection.tool_call_ordinals = original_ordinals
            original_instruction = saved.task.instruction_sha256
            saved.task.instruction_sha256 = instruction_prefix + "0" * 64
            assert (await http.get(url, headers=_OP)).status_code == 409
            assert (await http.post(url, headers=_OP, json=body)).status_code == 409
            assert await store.load_task(claimed.id) == held
            saved.task.instruction_sha256 = original_instruction
            for target, field, changed in (
                (saved, "product_run_id", "different-product"),
                (saved, "session_id", "different-session"),
                (saved, "agent_name", "different-agent"),
                (saved.source.git_baseline, "head_revision", "0" * 40),
                (
                    saved,
                    "settlement",
                    saved.settlement.model_copy(update={"reviewer_required": True}),
                ),
            ):
                original_value = getattr(target, field)
                setattr(target, field, changed)
                assert (await http.get(url, headers=_OP)).status_code == 409
                assert (await http.post(url, headers=_OP, json=body)).status_code == 409
                assert await store.load_task(claimed.id) == held
                setattr(target, field, original_value)
            assert (
                await http.post(
                    url, headers=_OP, json={**body, "plan_fingerprint": "sha256:" + "b" * 64}
                )
            ).status_code == 409
            assert await store.load_task(claimed.id) == held
            inspection.release_fingerprint = "sha256:" + "d" * 64
            assert (await http.post(url, headers=_OP, json=body)).status_code == 409
            assert await store.load_task(claimed.id) == held
            inspection.release_fingerprint = "sha256:" + "c" * 64
            # A fresh plan must not hide a newly unknown check outcome.
            events[-1].payload["result"]["structured"]["cleanup_uncertain"] = True
            assert (await http.get(url, headers=_OP)).status_code == 409
            assert (await http.post(url, headers=_OP, json=body)).status_code == 409
            assert await store.load_task(claimed.id) == held
            events[-1].payload["result"]["structured"]["cleanup_uncertain"] = False
            reconcile = store.reconcile_task_cancellation

            async def lost_ack(request):
                await reconcile(request)
                raise OSError("commit acknowledgement lost")

            monkeypatch.setattr(store, "reconcile_task_cancellation", lost_ack)
            assert (await http.post(url, headers=_OP, json=body)).status_code == 503
            monkeypatch.setattr(store, "reconcile_task_cancellation", reconcile)
            replay = await http.post(url, headers=_OP, json=body)
            assert replay.status_code == 200, replay.text
            assert replay.json()["coding_task_status"] == "cancelled"
            committed = await store.load_task(claimed.id)
            acknowledged = asyncio.Event()

            async def response_wait(request):
                result = await reconcile(request)
                acknowledged.set()
                await asyncio.Event().wait()
                return result

            monkeypatch.setattr(store, "reconcile_task_cancellation", response_wait)
            pending = asyncio.create_task(http.post(url, headers=_OP, json=body))
            try:
                await asyncio.wait_for(acknowledged.wait(), 5)
                pending.cancel("operator-disconnected")
                assert pending.cancelling() == 1
                with pytest.raises(asyncio.CancelledError, match="operator-disconnected"):
                    await pending
                assert pending.cancelled()
            finally:
                if not pending.done():
                    pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
                monkeypatch.setattr(store, "reconcile_task_cancellation", reconcile)
            assert await store.load_task(claimed.id) == committed
            # Reopen the durable backend before exact replay. No fresh host or
            # invocation proof is manufactured to replace the committed receipt.
            if hasattr(store, "path"):
                from cayu import SQLiteTaskStore

                reopened = SQLiteTaskStore(store.path)
                monkeypatch.setattr(
                    store, "reconcile_task_cancellation", reopened.reconcile_task_cancellation
                )
                try:
                    assert (await http.post(url, headers=_OP, json=body)).json() == replay.json()
                finally:
                    await reopened.close()
                monkeypatch.setattr(store, "reconcile_task_cancellation", reconcile)
            assert (await http.post(url, headers=_OP, json=body)).json() == replay.json()
            assert (
                await http.post(url, headers=_OP, json={**body, "reconciliation_id": "other"})
            ).status_code == 409
            terminal = await store.load_task(claimed.id)
            assert terminal.status is TaskStatus.CANCELLED and terminal.result is None
            assert terminal.worker_id is None and terminal.lease_expires_at is None
            assert not provider.requests

    asyncio.run(scenario())


def _fixture_ordinals(events):
    # Controlled serial fixture linkage; production derives this from private
    # Runtime dispatch tuples, never from names or these test events.
    current = -1
    values = []
    for event in events:
        if event.type is EventType.TOOL_CALL_STARTED:
            current += 1
        values.append(current if event.tool_name else None)
    return tuple(values)


@pytest.mark.parametrize("suffix", ["a" * 64 + ":" + "b" * 64, "a" * 64 + ":unknown"])
def test_operator_projects_recorded_generation_without_claiming_quiescence(host, suffix):
    server, application, _registry, _provider, _auth = host

    async def scenario():
        async with client(server) as http:
            intake = await http.post("/runs", headers=_A, json=_BODY)
            worker_id = "maintenance.coding:docker:" + suffix
            claimed = await application.app.task_store.claim_task(worker_id, lease_seconds=10)
            response = await http.get(
                f"/operator/runs/{intake.json()['id']}/tasks?tenant=tenant-a", headers=_OP
            )
            assert response.status_code == 200
            observed = response.json()
            owner = observed["phases"]["coding"]["recorded_owner"]
            assert owner == (
                {"kind": "present_unprojected"}
                if suffix.endswith("unknown")
                else {"kind": "registered_role", "id": worker_id}
            )
            assert observed["cleanup_evidence"] == observed["effect_evidence"] == "not_inspected"
            assert await application.app.task_store.load_task(claimed.id) == claimed

    asyncio.run(scenario())


def _file_events(name):
    common = {"path": "range_ops.py"}
    values = {
        "read_file": {
            **common,
            "source": "workspace",
            "encoding": "utf-8",
            "bytes": 10,
            "total_bytes": 10,
            "offset": 0,
            "truncated": False,
        },
        "write_file": {
            **common,
            "revision": "revision",
            "sha256": "a" * 64,
            "bytes": 10,
            "encoding": "utf-8",
            "mode": "overwrite",
        },
        "edit_file": {
            **common,
            "before_revision": "before",
            "after_revision": "after",
            "before_sha256": "a" * 64,
            "after_sha256": "b" * 64,
            "before_bytes": 10,
            "after_bytes": 11,
            "edit_count": 1,
            "replacement_count": 1,
        },
        "apply_patch": {
            "version": 2,
            "patch_id": "patch",
            "behavior_profile_id": "profile",
            "outcome": "applied",
            "failure_category": None,
            "requires_fresh_read": False,
            "operation_count": 1,
        },
        "delete_file": {
            **common,
            "deleted_bytes": 10,
            "deleted_revision": "revision",
            "deleted_sha256": "a" * 64,
        },
        "list_files": {
            "pattern": "**/*",
            "files": [],
            "offset": 0,
            "total_files": 0,
            "truncated": False,
        },
        "search_text": {
            **common,
            "pattern": "range",
            "mode": "files",
            "matches": [],
            "returned": 0,
            "offset": 0,
            "limit": 10,
            "stdout_bytes": 0,
            "truncated": False,
        },
        "git_changes": {
            "mode": "diff",
            "scope": "all",
            "changes": [],
            "returned": 0,
            "offset": 0,
            "limit": 10,
            "truncated": False,
        },
    }
    return (
        Event(type=EventType.TOOL_CALL_STARTED, session_id="fixture", tool_name=name),
        Event(
            type=EventType.TOOL_CALL_COMPLETED,
            session_id="fixture",
            tool_name=name,
            payload={"result": {"is_error": False, "structured": values[name]}},
        ),
    )
