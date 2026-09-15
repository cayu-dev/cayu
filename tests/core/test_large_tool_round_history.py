"""Admitted round history remains inspectable across query-sized call batches."""

import asyncio

import pytest

from cayu.events import Event, EventType
from cayu.sessions.base import InMemorySessionStore, RunRequest, SessionIdentity, ToolRoundIdentity
from cayu.storage.sqlite import SQLiteSessionStore


def _history_store(backend, path, fixture_request):
    if backend == "memory":
        return InMemorySessionStore()
    if backend == "sqlite":
        return SQLiteSessionStore(path)
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresSessionStore

    return PostgresSessionStore(
        fixture_request.getfixturevalue("postgres_dsn"),
        min_size=1,
        max_size=2,
        schema_mode=SchemaMode.CREATE,
    )


@pytest.mark.parametrize(
    "backend", ["memory", "sqlite", pytest.param("postgres", marks=pytest.mark.postgres)]
)
@pytest.mark.parametrize("count", [255, 256, 257])
def test_large_round_lifecycle_lookup_keeps_every_terminal(tmp_path, backend, count, request):
    async def run():
        store = _history_store(backend, tmp_path / "history.sqlite3", request)
        session_id = f"large-round-history-{count}-{backend}"
        identity = ToolRoundIdentity(
            model_step_id="mstep_" + "1" * 32,
            model_attempt_id="matt_" + "2" * 32,
            tool_round_id="tround_" + "3" * 32,
        )
        try:
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=SessionIdentity(provider_name="fake", model="fake"),
            )
            ids = [f"call-{i}" for i in range(count)]
            await store.checkpoint(
                session_id,
                {
                    "pending_tool_round": {
                        **identity.model_dump(),
                        "agent_name": "assistant",
                        "tool_calls": [
                            {"tool_call_id": call_id, "tool_name": "lookup", "arguments": {}}
                            for call_id in ids
                        ],
                    }
                },
            )
            events = [
                Event(
                    id=f"terminal-{i}",
                    type=EventType.TOOL_CALL_COMPLETED,
                    session_id=session_id,
                    tool_name="lookup",
                    payload={
                        **identity.model_dump(),
                        "tool_call_id": call_id,
                        "idempotency_key": f"operation-{i}",
                        "result": {"content": str(i), "is_error": False},
                    },
                )
                for i, call_id in enumerate(ids)
            ]
            await store.append_events(session_id, events)
            observed = await store.load_tool_round_lifecycle_events_for_round(
                session_id, ids, tool_round_identity=identity
            )
            assert [event.id for event in observed] == [event.id for event in events]
        finally:
            if backend != "memory":
                await store.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "backend", ["memory", "sqlite", pytest.param("postgres", marks=pytest.mark.postgres)]
)
@pytest.mark.parametrize("defect", [None, "missing_tail", "duplicate_tail", "foreign_round_tail"])
def test_large_round_publishes_all_results_and_exactly_replays(tmp_path, backend, defect, request):
    fixture_request = request
    from cayu.approvals.tools import PendingToolCallApproval
    from cayu.messages import ToolResultPart
    from cayu.runtime._tool_execution import tool_idempotency_key
    from cayu.runtime._tool_round_publication import build_tool_round_publication_request
    from cayu.runtime._tool_round_recovery import PendingToolRound
    from cayu.sessions.base import SessionStatus
    from cayu.tools.base import ToolResult

    async def run():
        store = _history_store(backend, tmp_path / "publication.sqlite3", fixture_request)
        session_id = (
            f"large-round-publication-{defect}"
            if backend == "postgres"
            else "large-round-publication"
        )
        pending = PendingToolRound(
            model_step_id="mstep_" + "1" * 32,
            model_attempt_id="matt_" + "2" * 32,
            tool_round_id="tround_" + "3" * 32,
            agent_name="assistant",
            environment_name=None,
            tool_calls=[
                PendingToolCallApproval(
                    tool_call_id=f"call-{i}", tool_name="lookup", arguments={"index": i}
                )
                for i in range(257)
            ],
        )
        identity = ToolRoundIdentity(
            model_step_id=pending.model_step_id,
            model_attempt_id=pending.model_attempt_id,
            tool_round_id=pending.tool_round_id,
        )
        try:
            session = await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=SessionIdentity(provider_name="fake", model="fake"),
            )
            checkpoint = {
                "pending_tool_round": pending.model_dump(mode="json"),
                "unrelated": "keep",
            }
            await store.checkpoint(session_id, checkpoint)
            events = []
            for i, call in enumerate(pending.tool_calls):
                payload = {
                    **identity.model_dump(),
                    "tool_call_id": call.tool_call_id,
                    "idempotency_key": tool_idempotency_key(
                        session_id=session_id,
                        tool_round_id=pending.tool_round_id,
                        tool_call_id=call.tool_call_id,
                    ),
                }
                for kind in (EventType.TOOL_CALL_STARTED, EventType.TOOL_CALL_COMPLETED):
                    event_payload = (
                        {**payload, "arguments": call.arguments}
                        if kind == EventType.TOOL_CALL_STARTED
                        else {
                            **payload,
                            "result": ToolResult(content=str(i)).model_dump(mode="json"),
                        }
                    )
                    events.append(
                        Event(
                            id=f"{kind}-{i}",
                            type=kind,
                            session_id=session_id,
                            interaction_id="interaction",
                            agent_name="assistant",
                            tool_name="lookup",
                            payload=event_payload,
                        )
                    )
            await store.append_events(session_id, events)
            retained = await store.load_tool_round_lifecycle_events_for_round(
                session_id,
                [c.tool_call_id for c in pending.tool_calls],
                tool_round_identity=identity,
            )
            assert len(retained) == 514
            if defect:
                broken = list(retained)
                if defect == "missing_tail":
                    broken.pop()
                elif defect == "duplicate_tail":
                    broken.append(broken[-1])
                else:
                    broken[-1] = broken[-1].model_copy(
                        update={
                            "payload": {**broken[-1].payload, "tool_round_id": "tround_" + "4" * 32}
                        }
                    )
                with pytest.raises(ValueError):
                    build_tool_round_publication_request(
                        session_id=session_id,
                        pending_round=pending,
                        source_checkpoint=checkpoint,
                        durable_events=broken,
                    )
                assert await store.load_transcript(session_id) == []
                assert await store.load_checkpoint(session_id) == checkpoint
                return
            request = build_tool_round_publication_request(
                session_id=session_id,
                pending_round=pending,
                source_checkpoint=checkpoint,
                durable_events=retained,
            )
            result = await store.publish_runtime_publication(
                session_id,
                request=request,
                expected_statuses={SessionStatus.PENDING},
                expected_run_epoch=session.run_epoch,
                expected_transcript_cursor=0,
            )
            assert not result.replayed
            if backend == "sqlite":
                import json
                import subprocess
                import sys

                request_file = tmp_path / "publication-request.json"
                request_file.write_text(json.dumps(checkpoint))
                await store.close()
                process = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        """
import asyncio, json, sys
from pathlib import Path
from cayu.runtime._tool_round_publication import build_tool_round_publication_request
from cayu.runtime._tool_round_recovery import PendingToolRound, pending_tool_round_identity
from cayu.storage.sqlite import SQLiteSessionStore
async def replay():
    store = SQLiteSessionStore(sys.argv[1])
    try:
        checkpoint = json.loads(Path(sys.argv[2]).read_text())
        pending = PendingToolRound.model_validate(checkpoint["pending_tool_round"])
        events = await store.load_tool_round_lifecycle_events_for_round(
            "large-round-publication", [c.tool_call_id for c in pending.tool_calls],
            tool_round_identity=pending_tool_round_identity(pending))
        request = build_tool_round_publication_request(session_id="large-round-publication",
            pending_round=pending, source_checkpoint=checkpoint, durable_events=events)
        result = await store.publish_runtime_publication("large-round-publication", request=request)
        print(json.dumps({"replayed": result.replayed, "receipt": result.receipt.model_dump(mode="json")}))
    finally:
        await store.close()
asyncio.run(replay())
""",
                        str(tmp_path / "publication.sqlite3"),
                        str(request_file),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                child_result = json.loads(process.stdout)
                assert child_result["replayed"] is True
                assert child_result["receipt"] == result.receipt.model_dump(mode="json")
                store = SQLiteSessionStore(tmp_path / "publication.sqlite3")
            replay = await store.publish_runtime_publication(
                session_id,
                request=request,
                expected_statuses={SessionStatus.PENDING},
                expected_run_epoch=session.run_epoch,
                expected_transcript_cursor=0,
            )
            assert replay.replayed and replay.receipt == result.receipt
            assert await store.load_checkpoint(session_id) == {"unrelated": "keep"}
            transcript = await store.load_transcript(session_id)
            assert len(transcript) == 1
            parts = [p for p in transcript[0].content if isinstance(p, ToolResultPart)]
            assert [p.tool_call_id for p in parts] == [c.tool_call_id for c in pending.tool_calls]
        finally:
            if backend != "memory":
                await store.close()

    asyncio.run(run())


def test_native_large_round_cancellation_recovers_every_admitted_call(tmp_path):
    from tests.core._workload_secret_support import FakeProvider, collect_events

    from cayu.agents import AgentSpec
    from cayu.applications import CayuApp
    from cayu.messages import Message, ToolResultPart
    from cayu.providers.base import ModelStreamEvent
    from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
    from cayu.sessions.base import IncompleteSessionRecoveryRequest, ResumeRequest
    from cayu.tools.base import Tool, ToolEffect, ToolResult, ToolSpec

    class Echo(Tool):
        spec = ToolSpec(
            name="echo",
            description="Echo the supplied value",
            parallel_safe=False,
            effect=ToolEffect.IDEMPOTENT,
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name="tests:large-round:echo",
                behavior_version="1",
                implementation_version="1",
            ),
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        )

        def __init__(self):
            self.values = []
            self.second_started = asyncio.Event()

        async def run(self, ctx, args):
            self.values.append(args["value"])
            if args["value"] == "1":
                self.second_started.set()
                await asyncio.Event().wait()
            return ToolResult(content=args["value"])

    async def run():
        store = SQLiteSessionStore(tmp_path / "native.sqlite3")
        tool = Echo()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            FakeProvider(
                [
                    *[
                        ModelStreamEvent.tool_call(
                            id=f"native-{i}", name="echo", arguments={"value": str(i)}
                        )
                        for i in range(257)
                    ],
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])
        consumer = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="native-large-cancel",
                    messages=[Message.text("user", "Echo the synthetic values.")],
                ),
            )
        )
        try:
            await asyncio.wait_for(tool.second_started.wait(), timeout=7200)
            before_cancel = await store.load_events("native-large-cancel")
            assert any(
                e.type == EventType.TOOL_CALL_COMPLETED
                and e.payload.get("tool_call_id") == "native-0"
                and e.payload["result"]["content"] == "0"
                for e in before_cancel
            )
            consumer.cancel("interrupt after a durable first result")
            with pytest.raises(asyncio.CancelledError):
                await consumer
            assert await app.drain_recovery_cleanups(timeout_s=120)
            checkpoint = await store.load_checkpoint("native-large-cancel")
            if "pending_tool_round" in checkpoint:
                # Finite shutdown cleanup may leave a durable pending round.
                # Explicit resume must finish it without repeating staged work.
                await store.close()
                store = SQLiteSessionStore(tmp_path / "native.sqlite3")
                resumed_app = CayuApp(session_store=store, enable_logging=False)
                resumed_app.register_provider(
                    FakeProvider(
                        [
                            ModelStreamEvent.text_delta("done"),
                            ModelStreamEvent.completed(),
                        ]
                    ),
                    default=True,
                )
                resumed_app.register_agent(
                    AgentSpec(name="assistant", model="fake-model"), tools=[tool]
                )
                await resumed_app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(
                        session_id="native-large-cancel",
                        reason="synthetic worker restart after cancelled cleanup",
                    )
                )
                resumed = [
                    event
                    async for event in resumed_app.resume(
                        ResumeRequest(
                            session_id="native-large-cancel",
                            messages=[Message.text("user", "Continue from retained work.")],
                        )
                    )
                ]
                assert resumed[-1].type == EventType.SESSION_COMPLETED
            events = await store.load_events("native-large-cancel")
            terminal = [
                e
                for e in events
                if e.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
            ]
            assert len(terminal) == 257
            assert len({e.payload["tool_call_id"] for e in terminal}) == 257
            completed = [e for e in terminal if e.type == EventType.TOOL_CALL_COMPLETED]
            assert any(e.payload["result"]["content"] == "0" for e in completed)
            assert tool.values.count("0") == 1
            assert "pending_tool_round" not in await store.load_checkpoint("native-large-cancel")
            transcript = await store.load_transcript("native-large-cancel")
            results = [p for m in transcript for p in m.content if isinstance(p, ToolResultPart)]
            assert len(results) == 257
            assert sum(not p.is_error for p in results) == len(completed)
        finally:
            if not consumer.done():
                consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)
            await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("restart", [False, True])
def test_native_large_round_across_worker_processes(tmp_path, restart):
    import json
    import subprocess
    import sys
    import time

    worker = "tests.core._large_round_process_worker"
    with (tmp_path / "worker.log").open("w") as log:
        if restart:
            producer = subprocess.Popen(
                [sys.executable, "-m", worker, "prepare", str(tmp_path)], stdout=log, stderr=log
            )
            try:
                deadline = time.monotonic() + 7200
                while not (tmp_path / "ready.json").exists():
                    assert producer.poll() is None, (tmp_path / "worker.log").read_text()
                    assert time.monotonic() < deadline, (
                        "Worker did not reach its durable interruption point"
                    )
                    time.sleep(0.1)
                assert json.loads((tmp_path / "ready.json").read_text())["first_result_durable"]
                producer.kill()  # Only this test-owned worker: no graceful Runtime cleanup.
                producer.wait(timeout=30)
            finally:
                if producer.poll() is None:
                    producer.kill()
                    producer.wait(timeout=30)
        mode = "resume" if restart else "normal"
        completed = subprocess.run(
            [sys.executable, "-m", worker, mode, str(tmp_path)],
            stdout=log,
            stderr=log,
            timeout=7200,
        )
        assert completed.returncode == 0, (tmp_path / "worker.log").read_text()
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["terminals"] == 257 and result["first_execution_count"] == 1
    if not restart:
        assert result["completed"] == 257
