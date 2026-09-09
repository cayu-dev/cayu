"""Public-API, fresh-process allocation recovery without Docker or model access."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentAllocationScope,
    EnvironmentAllocationState,
    EnvironmentFactory,
    EnvironmentFactoryOperation,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryResult,
    EnvironmentSpec,
    ExecutionProfileBehaviorIdentity,
    IncompleteSessionRecoveryRequest,
    Message,
    ModelStreamEvent,
    ResumeRequest,
    RunRequest,
    ScriptedModelProvider,
    SQLiteSessionStore,
    WorkspaceBinding,
)


async def _child(root: Path, mode: str, fault: str) -> None:
    def call(operation: str) -> None:
        with (root / "calls.jsonl").open("a") as stream:
            stream.write(json.dumps(operation) + "\n")

    class Store(SQLiteSessionStore):
        invocation_lifecycle_command_version = 1
        lost_ack = False

        async def transform_checkpoint(self, session_id, checkpoint_transform):
            await super().transform_checkpoint(session_id, checkpoint_transform)
            checkpoint = await self.load_checkpoint(session_id) or {}
            record = checkpoint.get("environment_factory_allocation_intents", {}).get("remote")
            if (
                mode == "run"
                and fault == "after_publication"
                and checkpoint.get("environment_factory_allocation_receipts")
            ):
                os._exit(73)
            if (
                mode == "run"
                and fault == "binding_rejection"
                and record is not None
                and record["state"] == "reaping"
            ):
                os._exit(75)
            if mode == "after_reaped" and record is not None and record["state"] == "reaped":
                os._exit(74)
            if (
                mode == "lost_cleanup_ack"
                and not self.lost_ack
                and record is not None
                and record["state"] == "reaped"
            ):
                self.lost_ack = True
                raise TimeoutError("lost cleanup store acknowledgement")

    class RejectedBinding(WorkspaceBinding):
        async def bind(self, workspace, runner, **kwargs):
            raise ValueError("fixture binding rejected")

        async def finalize(self, bound, **kwargs):
            raise AssertionError("Rejected binding must not finalize")

    class Factory(EnvironmentFactory):
        @property
        def execution_profile_identity(self):
            return ExecutionProfileBehaviorIdentity(
                name="tests:allocation-factory",
                behavior_version="1",
                implementation_version="1",
            )

        def allocation_scope(self, request):
            return EnvironmentAllocationScope(provider="fixture", adapter_generation="fixture.v1")

        async def create_recoverable(self, request, allocation):
            if mode == "run" and fault == "before_intent":
                os._exit(73)
            if allocation.state is EnvironmentAllocationState.UNPREPARED:
                await allocation.prepare({"resource": allocation.intent.allocation_id})
            if mode == "run" and fault == "after_intent":
                os._exit(73)
            if allocation.state is EnvironmentAllocationState.PREPARED:
                await allocation.mark_dispatched()
            if mode == "run" and fault == "after_dispatch":
                os._exit(73)
            path = root / (allocation.intent.allocation_id + ".resource")
            identity = allocation.intent.to_payload()
            if path.exists():
                assert json.loads(path.read_text()) == identity
                call("lookup")
            else:
                call("create")
                path.write_text(json.dumps(identity))
            if mode == "run" and fault == "after_remote_create":
                os._exit(73)
            reconnect = {"resource": allocation.intent.allocation_id}
            await allocation.acknowledge(reconnect)
            if mode == "run" and fault == "after_acknowledgement":
                os._exit(73)
            if mode == "run" and fault == "factory_failure":
                raise RuntimeError("fixture failed after acknowledgement")
            return self.result(request, path, reconnect)

        async def create(self, request):
            assert request.operation is EnvironmentFactoryOperation.RECONNECT
            call("reconnect")
            path = root / (request.reconnect_metadata["resource"] + ".resource")
            assert path.exists()
            return self.result(request, path, request.reconnect_metadata)

        def result(self, request, path, reconnect):
            async def release(action):
                if action is EnvironmentFactoryReleaseAction.DISCARD:
                    call("discard")
                    path.unlink(missing_ok=True)

            return EnvironmentFactoryResult(
                environment=Environment(
                    EnvironmentSpec(name=request.environment_name),
                    binding=RejectedBinding() if fault == "binding_rejection" else None,
                ),
                reconnect_metadata=reconnect,
                release=release,
            )

        async def reap_allocation(self, request, allocation):
            assert allocation.intent.session_id == request.session_id
            assert allocation.intent.environment_name == request.environment_name
            assert allocation.intent.scope == self.allocation_scope(request)
            if mode == "cleanup_failure":
                raise RuntimeError("fixture cleanup unavailable")
            if mode == "cleanup_cancel":
                raise asyncio.CancelledError("fixture cleanup cancelled")
            if mode == "cleanup_timeout":
                await asyncio.sleep(60)
            if mode.startswith("concurrent_"):
                await asyncio.sleep(1)
            path = root / (allocation.intent.allocation_id + ".resource")
            if path.exists():
                if json.loads(path.read_text()) != allocation.intent.to_payload():
                    raise RuntimeError("fixture resource belongs to another generation")
            elif allocation.state is EnvironmentAllocationState.DISPATCHED:
                raise RuntimeError("fixture allocation is still ambiguous")
            if not await allocation.mark_reaping():
                return
            if mode == "after_reaping":
                os._exit(74)
            call("reap")
            path.unlink(missing_ok=True)
            if mode == "after_delete":
                os._exit(74)
            await allocation.mark_reaped()

    def clock():
        return datetime.now(UTC) + timedelta(
            seconds=600 if (root / "advance_store_clock").exists() else 0,
        )

    store = Store(root / "sessions.db", ownership_clock=clock)
    app = CayuApp(session_store=store, enable_logging=False, clock=clock)
    app.register_provider(
        ScriptedModelProvider(
            [[ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()]]
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="probe", model="scripted-model"))
    app.register_environment_factory(
        EnvironmentSpec(
            name="remote",
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name="tests:abandoned-allocation",
                behavior_version="1",
                implementation_version="changed" if mode == "profile_drift" else "1",
            ),
        ),
        Factory(),
        default=True,
    )
    try:
        if mode == "run":
            async for _ in app.run(
                RunRequest(
                    agent_name="probe",
                    session_id="abandoned",
                    messages=[Message.text("user", "finish")],
                    max_steps=1,
                )
            ):
                pass
        elif mode == "resume":
            async for _ in app.resume(
                ResumeRequest(
                    session_id="abandoned", messages=[Message.text("user", "continue")], max_steps=1
                )
            ):
                pass
        else:
            result = await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id="abandoned",
                    reason="fixture_process_loss",
                )
            )
            (root / (mode + ".json")).write_text(result.model_dump_json())
        checkpoint = await store.load_checkpoint("abandoned")
        (root / "checkpoint.json").write_text(json.dumps(checkpoint))
        events = await store.load_events("abandoned")
        (root / "events.json").write_text(
            json.dumps([event.model_dump(mode="json") for event in events])
        )
    except asyncio.CancelledError as exc:
        (root / "cancelled.json").write_text(json.dumps(str(exc)))
    finally:
        await store.close()


def _process(
    root: Path, mode: str, fault: str = "after_acknowledgement"
) -> subprocess.CompletedProcess:
    script = (
        "import asyncio,sys; from pathlib import Path; "
        "from tests.core.test_abandoned_environment_allocations import _child; "
        "asyncio.run(_child(Path(sys.argv[1]),sys.argv[2],sys.argv[3]))"
    )
    return subprocess.run(
        [sys.executable, "-c", script, str(root), mode, fault],
        text=True,
        capture_output=True,
        timeout=45,
    )


def _recover(root: Path, mode: str = "recover") -> dict:
    process = _process(root, mode)
    assert process.returncode == 0, process.stderr
    return json.loads((root / (mode + ".json")).read_text())


@pytest.mark.parametrize(
    "fault",
    [
        "before_intent",
        "after_intent",
        "after_remote_create",
        "after_acknowledgement",
        "after_publication",
    ],
)
def test_abandoned_setup_recovery_crosses_every_publication_boundary(tmp_path: Path, fault: str):
    process = _process(tmp_path, "run", fault)
    assert process.returncode == 73, process.stderr
    result = _recover(tmp_path)
    assert result["status"] == "interrupted"
    assert not list(tmp_path.glob("*.resource"))
    repeated = _recover(tmp_path, "repeat")
    assert repeated["actions"] == ["skipped_terminal"]
    events = json.loads((tmp_path / "events.json").read_text())
    assert sum(event["type"] == "session.interrupted" for event in events) == 1
    assert not any(event["type"] == "model.requested" for event in events)
    if fault != "before_intent":
        assert "reaped_allocation" in result["actions"]
        checkpoint = json.loads((tmp_path / "checkpoint.json").read_text())
        assert checkpoint["environment_factory_allocation_intents"]["remote"]["state"] == "reaped"


@pytest.mark.parametrize(
    "mode",
    [
        "cleanup_failure",
        "cleanup_cancel",
        "cleanup_timeout",
        "after_reaping",
        "after_delete",
        "after_reaped",
        "lost_cleanup_ack",
    ],
)
def test_cleanup_failures_and_process_loss_remain_retryable(tmp_path: Path, mode: str):
    assert _process(tmp_path, "run").returncode == 73
    first = _process(tmp_path, mode)
    assert first.returncode == (74 if mode.startswith("after_") else 0), first.stderr
    if mode in {"cleanup_failure", "cleanup_timeout"}:
        result = json.loads((tmp_path / (mode + ".json")).read_text())
        assert result["actions"] == ["pending_allocation_cleanup"]
        assert list(tmp_path.glob("*.resource"))
    if mode == "cleanup_cancel":
        assert json.loads((tmp_path / "cancelled.json").read_text()) == "fixture cleanup cancelled"
    if mode.startswith("after_"):
        assert _recover(tmp_path, "still_leased")["actions"] == ["skipped_active"]
        # Advance the injected app and store clocks together after proving
        # immediate takeover is refused. No private SQL or lease edits.
        (tmp_path / "advance_store_clock").touch()
    result = _recover(tmp_path)
    assert result["status"] == "interrupted"
    assert not list(tmp_path.glob("*.resource"))
    assert _recover(tmp_path, "repeat")["actions"] == ["skipped_terminal"]


def test_ambiguous_dispatch_is_pending_until_exact_resource_appears(tmp_path: Path):
    assert _process(tmp_path, "run", "after_dispatch").returncode == 73
    result = _recover(tmp_path)
    assert result["actions"] == ["pending_allocation_cleanup"]
    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text())
    record = checkpoint["environment_factory_allocation_intents"]["remote"]
    assert record["state"] == "dispatched"
    # Complete the original provider operation, not a recovery allocation.
    path = tmp_path / (record["intent"]["allocation_id"] + ".resource")
    path.write_text(json.dumps(record["intent"]))
    assert "reaped_allocation" in _recover(tmp_path, "repeat")["actions"]
    assert not path.exists()
    assert "create" not in (tmp_path / "calls.jsonl").read_text()


def test_cleanup_refuses_a_different_resource_generation(tmp_path: Path):
    assert _process(tmp_path, "run").returncode == 73
    (path,) = tmp_path.glob("*.resource")
    other = json.loads(path.read_text())
    other["adapter_generation"] = "newer"
    path.write_text(json.dumps(other))
    result = _recover(tmp_path)
    assert result["actions"] == ["pending_allocation_cleanup"]
    assert json.loads(path.read_text()) == other


def test_cleanup_keeps_profile_and_missing_transcript_guards(tmp_path: Path):
    assert _process(tmp_path, "run").returncode == 73
    rejected = _process(tmp_path, "profile_drift")
    assert rejected.returncode != 0
    assert "execution profile changed" in rejected.stderr
    assert list(tmp_path.glob("*.resource"))
    assert "reaped_allocation" in _recover(tmp_path)["actions"]
    rejected = _process(tmp_path, "resume")
    assert rejected.returncode != 0
    assert "initial transcript" in rejected.stderr
    assert not list(tmp_path.glob("*.resource"))
    assert (tmp_path / "calls.jsonl").read_text().count('"create"') == 1


def test_terminal_setup_failure_still_recovers_its_unpublished_allocation(tmp_path: Path):
    process = _process(tmp_path, "run", "factory_failure")
    assert process.returncode == 0, process.stderr
    assert list(tmp_path.glob("*.resource"))
    result = _recover(tmp_path)
    assert result["status"] == "failed"
    assert "reaped_allocation" in result["actions"]
    assert not list(tmp_path.glob("*.resource"))
    assert _recover(tmp_path, "repeat")["actions"] == ["skipped_terminal"]


def test_concurrent_recovery_uses_one_claim_and_preserves_other_allocations(tmp_path: Path):
    assert _process(tmp_path, "run").returncode == 73
    foreign = tmp_path / "foreign.resource"
    foreign.write_text("owned by another session")
    with ThreadPoolExecutor(max_workers=2) as executor:
        processes = list(
            executor.map(
                lambda mode: _process(tmp_path, mode),
                ("concurrent_one", "concurrent_two"),
            )
        )
    assert all(result.returncode == 0 for result in processes), [
        result.stderr for result in processes
    ]
    actions = [
        action
        for mode in ("concurrent_one", "concurrent_two")
        for action in json.loads((tmp_path / (mode + ".json")).read_text())["actions"]
    ]
    assert "reaped_allocation" in actions
    assert "skipped_active" in actions or "skipped_terminal" in actions
    assert list(tmp_path.glob("*.resource")) == [foreign]
    assert foreign.read_text() == "owned by another session"
    assert (tmp_path / "calls.jsonl").read_text().count('"create"') == 1


def test_valid_continuation_reconnects_exact_published_allocation(tmp_path: Path):
    first = _process(tmp_path, "run", "none")
    assert first.returncode == 0, first.stderr
    resources = list(tmp_path.glob("*.resource"))
    assert len(resources) == 1
    continued = _process(tmp_path, "resume")
    assert continued.returncode == 0, continued.stderr
    assert list(tmp_path.glob("*.resource")) == resources
    calls = (tmp_path / "calls.jsonl").read_text()
    assert calls.count('"create"') == 1
    assert calls.count('"reconnect"') == 1


def test_failed_binding_reaping_survives_process_loss_after_transcript_publication(tmp_path: Path):
    process = _process(tmp_path, "run", "binding_rejection")
    assert process.returncode == 75, process.stderr
    assert len(list(tmp_path.glob("*.resource"))) == 1
    (tmp_path / "advance_store_clock").touch()
    result = _recover(tmp_path)
    assert "reaped_allocation" in result["actions"]
    assert not list(tmp_path.glob("*.resource"))
    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text())
    assert "initial_transcript_pending" not in checkpoint
    assert checkpoint["environment_factory_allocation_intents"]["remote"]["state"] == "reaped"
    assert _recover(tmp_path, "repeat")["actions"] == ["skipped_terminal"]
    events = json.loads((tmp_path / "events.json").read_text())
    assert any(event["type"] == "environment.factory.completed" for event in events)
    assert not any(event["type"] == "model.requested" for event in events)
    calls = (tmp_path / "calls.jsonl").read_text()
    assert calls.count('"create"') == 1
    assert calls.count('"reap"') == 1
    assert '"reconnect"' not in calls
