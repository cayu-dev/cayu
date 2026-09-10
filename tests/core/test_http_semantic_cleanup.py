"""Credential-free semantic-idle shutdown and durable local HTTP receipts."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    RecoveryPlanRequest,
    RecoveryPlanSelection,
    SQLiteSessionStore,
    StepError,
    WorkflowBase,
    WorkflowSpec,
    step,
)
from cayu.providers import deadlines as ds
from cayu.providers.deadlines import ProviderStreamDeadlines
from cayu.providers.openai import HttpxOpenAITransport, OpenAIProvider


class Workflow(WorkflowBase):
    spec = WorkflowSpec(name="semantic-probe")

    async def run(self, sid):
        yield await self.context(sid).start()


def frame(x):
    return b"data: " + json.dumps(x).encode() + b"\n\n"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "mode,traffic",
    [
        (mode, traffic)
        for mode in ("socket", "mock", "delayed", "failure")
        for traffic in ("silence", "heartbeat", "whitespace")
    ]
    + [
        ("noncooperative", "silence"),
        ("receipt_failure", "silence"),
        ("delayed_receipt", "silence"),
        ("socket_delayed_receipt", "whitespace"),
    ],
)
async def test_semantic_idle_http_cleanup(tmp_path, monkeypatch, mode, traffic):
    baseline_tasks = set(asyncio.all_tasks())

    payload = frame(
        {
            "type": "response.created",
            "response": {"id": "resp_synthetic", "status": "in_progress", "output": []},
        }
    ) + frame({"type": "response.reasoning_summary_text.delta", "delta": "synthetic reasoning"})
    extra = (
        b": heartbeat\n\n"
        if traffic == "heartbeat"
        else frame({"type": "response.output_text.delta", "delta": " "})
    )
    handlers = set()
    closed = asyncio.Event()
    release = asyncio.Event()
    stream_started = asyncio.Event()
    requests = 0

    async def serve(reader, writer):
        handlers.add(asyncio.current_task())
        try:
            h = await reader.readuntil(b"\r\n\r\n")
            n = next(
                int(line.split(b":", 1)[1])
                for line in h.split(b"\r\n")
                if line.lower().startswith(b"content-length:")
            )
            await reader.readexactly(n)
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n"
                + payload
            )
            await writer.drain()
            stream_started.set()
            while True:
                try:
                    x = await asyncio.wait_for(reader.read(), 0.025)
                    assert x == b""
                    break
                except TimeoutError:
                    if traffic != "silence":
                        writer.write(extra)
                        await writer.drain()
        except (ConnectionError, BrokenPipeError):
            pass
        finally:
            writer.close()
            await writer.wait_closed()
            handlers.discard(asyncio.current_task())
            closed.set()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]

    class Local(httpx.AsyncBaseTransport):
        def __init__(self):
            self.inner = httpx.AsyncHTTPTransport()

        async def handle_async_request(self, request):
            nonlocal requests
            requests += 1
            request.url = httpx.URL(f"http://127.0.0.1:{port}/responses")
            return await self.inner.handle_async_request(request)

        async def aclose(self):
            await self.inner.aclose()

    class Mock(httpx.AsyncByteStream):
        async def __aiter__(self):
            stream_started.set()
            yield payload
            while True:
                try:
                    await asyncio.sleep(0.025)
                except asyncio.CancelledError:
                    if mode == "noncooperative":
                        await release.wait()
                    raise
                if traffic != "silence":
                    yield extra

        async def aclose(self):
            if mode == "delayed":
                await release.wait()
            if mode == "failure":
                raise RuntimeError("synthetic close failure")
            closed.set()

    def handle(request):
        nonlocal requests
        requests += 1
        return httpx.Response(200, stream=Mock())

    transport = HttpxOpenAITransport()
    path = tmp_path / "sessions.db"
    store = SQLiteSessionStore(path)
    if mode in {"receipt_failure", "delayed_receipt", "socket_delayed_receipt"}:
        append_event = store.append_event

        async def reject_receipt(session_id, event):
            if event.type == "model.http_cleanup":
                if mode in {"delayed_receipt", "socket_delayed_receipt"}:
                    await release.wait()
                else:
                    raise RuntimeError("synthetic receipt persistence failure")
            await append_event(session_id, event)

        monkeypatch.setattr(store, "append_event", reject_receipt)
    provider = None
    running = opening = None
    try:
        async with httpx.AsyncClient(
            transport=(
                Local()
                if mode in {"socket", "socket_delayed_receipt"}
                else httpx.MockTransport(handle)
            )
        ) as client:
            transport._client._client = client
            provider = OpenAIProvider(
                api_key="synthetic",
                base_url="https://synthetic.invalid",
                transport=transport,
                stream_deadlines=ProviderStreamDeadlines(
                    semantic_progress_timeout_s=0.2,
                    transport_idle_timeout_s=5,
                    protocol_idle_timeout_s=5,
                    absolute_stream_timeout_s=10,
                ),
            )
            app = CayuApp(enable_logging=False, session_store=store)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="probe", model="synthetic"))
            ctx = Workflow(app).context("probe")
            await ctx.start()
            running = asyncio.create_task(
                step(ctx, agent="probe", step_id="one", prompt="synthetic")
            )
            opening = asyncio.create_task(stream_started.wait())
            # The cleanup watchdog measures an active HTTP stream, not session
            # preparation/schema/profile work on a loaded CI host. Setup has a
            # separate deadlock guard; an early step failure must be observed.
            done, _ = await asyncio.wait(
                (running, opening), timeout=30, return_when=asyncio.FIRST_COMPLETED
            )
            assert done, "Workflow did not reach HTTP dispatch within the setup guard."
            if opening not in done:
                await running
                pytest.fail("Workflow finished without opening its HTTP stream.")
            with pytest.raises(StepError) as raised:
                await asyncio.wait_for(running, 3)
            sid = raised.value.session_id
            assert sid is not None

            async def snapshot():
                events = await store.load_events(sid)
                before = await store.load(sid)
                plan = await app.plan_recovery(
                    RecoveryPlanRequest(
                        selection=RecoveryPlanSelection(session_ids=(sid,), inactive_for_seconds=0)
                    )
                )
                assert await store.load(sid) == before
                assert await store.load_events(sid) == events
                return events, plan.items[0]

            events, initial_plan = await snapshot()
            error = next(e for e in events if e.type == "model.error")
            assert error.payload["provider_deadline_kind"] == "semantic_idle"
            assert error.payload["provider_deadline_timeout_s"] == 0.2
            assert error.payload["provider_effect_outcome"] == "unknown"
            assert error.payload["retry_disposition"] == "suppressed"
            assert error.payload["effective_max_attempts"] == 1
            # Socket shutdown and SQLite receipt publication may outlast the
            # bounded 100/50 ms joins on a loaded worker. The original error
            # can therefore report unconfirmed cleanup even when it later
            # succeeds. Require the immutable, exact durable receipt below.
            if mode not in {"socket", "mock"}:
                assert error.payload["stream_cleanup_failed"] is True
            elif not error.payload.get("stream_cleanup_failed", False):
                assert initial_plan.active_model_stage.local_http_cleanup == "succeeded"
            if mode in {"delayed", "noncooperative"}:
                assert not closed.is_set()
                assert ds._PROVIDER_DEADLINE_AWAIT_OWNERS
                assert initial_plan.active_model_stage.local_http_cleanup == "unknown"
            if mode in {"delayed_receipt", "socket_delayed_receipt"}:
                # For a socket, peer EOF observation is independent of the
                # local close/receipt task; join it explicitly before checking.
                await asyncio.wait_for(closed.wait(), 2)
                assert closed.is_set()
                assert ds._PROVIDER_DEADLINE_AWAIT_OWNERS
                assert initial_plan.active_model_stage.local_http_cleanup == "unknown"
            release.set()
            async with asyncio.timeout(2):
                while ds._PROVIDER_DEADLINE_AWAIT_OWNERS:
                    await asyncio.sleep(0.001)
            events, settled_plan = await snapshot()
            receipts = [e for e in events if e.type == "model.http_cleanup"]
            assert len(receipts) == (0 if mode == "receipt_failure" else 1)
            expected = (
                "unknown"
                if mode == "receipt_failure"
                else "failed"
                if mode == "failure"
                else "succeeded"
            )
            assert settled_plan.active_model_stage.local_http_cleanup == expected
            assert settled_plan.allowed_actions == initial_plan.allowed_actions
            assert settled_plan.blockers == initial_plan.blockers
            assert settled_plan.status == "running"
            if receipts:
                receipt = receipts[0]
                assert receipt.payload["local_http_cleanup"] == expected
                assert receipt.payload["source_run_epoch"] == settled_plan.run_epoch == 1
                for key in (
                    "model_attempt_id",
                    "model_step_id",
                    "provider_deadline_kind",
                    "provider_deadline_timeout_s",
                    "provider_stream_elapsed_s",
                ):
                    assert receipt.payload[key] == error.payload[key]
                assert receipt.session_id == error.session_id == sid
                assert receipt.interaction_id == error.interaction_id
                assert receipt.payload["provider_effect_outcome"] == "unknown"
            if (mode, traffic) == ("mock", "silence"):
                query_events = store.query_events
                for field, wrong in (
                    ("model_attempt_id", "matt_unrelated"),
                    ("source_run_epoch", 2),
                    ("provider_effect_outcome", "settled"),
                ):

                    async def wrong_receipt(query=None, field=field, wrong=wrong):
                        records = await query_events(query)
                        if query is not None and query.event_id == receipt.id:
                            assert len(records) == 1
                            changed = receipt.model_copy(
                                update={"payload": {**receipt.payload, field: wrong}}
                            )
                            return [records[0].model_copy(update={"event": changed})]
                        return records

                    with monkeypatch.context() as patch:
                        patch.setattr(store, "query_events", wrong_receipt)
                        _, unrelated_plan = await snapshot()
                    assert unrelated_plan.active_model_stage.local_http_cleanup == "unknown"
                    assert unrelated_plan.allowed_actions == settled_plan.allowed_actions
            assert next(e for e in events if e.id == error.id) == error
            assert requests == 1
            if mode in {"socket", "socket_delayed_receipt"}:
                await asyncio.wait_for(closed.wait(), 2)
            assert closed.is_set() == (mode != "failure")
            assert not handlers
            assert not [t for t in asyncio.all_tasks() - baseline_tasks if not t.done()]
            await store.close()
            store = SQLiteSessionStore(path)
            app = CayuApp(enable_logging=False, session_store=store)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="probe", model="synthetic"))
            readback, readback_plan = await snapshot()
            assert readback == events
            assert readback_plan == settled_plan
            await provider.aclose()
            await store.close()
    finally:
        release.set()
        for owned in (running, opening):
            if owned is not None and not owned.done():
                owned.cancel()
        await asyncio.gather(
            *(owned for owned in (running, opening) if owned is not None), return_exceptions=True
        )
        async with asyncio.timeout(2):
            while ds._PROVIDER_DEADLINE_AWAIT_OWNERS:
                await asyncio.sleep(0.001)
        if provider is not None:
            await provider.aclose()
        await store.close()
        server.close()
        await server.wait_closed()
        for task in list(handlers):
            task.cancel()
        await asyncio.gather(*handlers, return_exceptions=True)


@pytest.mark.anyio
async def test_semantic_http_cleanup_watchdog_excludes_session_setup(tmp_path, monkeypatch):
    from cayu.runtime._session_engine import SessionEngine

    prepare = SessionEngine._prepare_initial_run

    async def slow_prepare(self, request, **kwargs):
        # Longer than the unchanged post-dispatch watchdog. This must not turn
        # the provider's semantic-idle result into an unrelated caller timeout.
        await asyncio.sleep(3.1)
        return await prepare(self, request, **kwargs)

    monkeypatch.setattr(SessionEngine, "_prepare_initial_run", slow_prepare)
    await test_semantic_idle_http_cleanup(tmp_path, monkeypatch, "delayed", "whitespace")


@pytest.mark.anyio
async def test_parent_cancellation_interrupts_semantic_cleanup_grace():
    cancellation_seen = asyncio.Event()
    release = asyncio.Event()
    read_settled = asyncio.Event()
    cancellations = 0
    controller = ds.ProviderStreamDeadlineController(
        ProviderStreamDeadlines(semantic_progress_timeout_s=0.01)
    )

    async def read():
        nonlocal cancellations
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellations += 1
            cancellation_seen.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancellations += 1
                raise
            finally:
                read_settled.set()
            raise

    task = asyncio.create_task(
        controller.wait_for(
            read(), kinds=(ds.ProviderDeadlineKind.SEMANTIC_IDLE,), semantic_cleanup_grace_s=0.1
        )
    )
    try:
        await asyncio.wait_for(cancellation_seen.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        controller.close()
        assert ds._PROVIDER_DEADLINE_AWAIT_OWNERS
        assert not read_settled.is_set()
        release.set()
        await asyncio.wait_for(read_settled.wait(), 1)
        async with asyncio.timeout(1):
            while ds._PROVIDER_DEADLINE_AWAIT_OWNERS:
                await asyncio.sleep(0)
        assert cancellations == 1
    finally:
        release.set()
        controller.close()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.anyio
async def test_accepted_terminal_result_does_not_publish_deadline_cleanup():
    from cayu.providers._stream_cleanup import (
        _local_http_cleanup_observer,
        _LocalHttpCleanupObserver,
    )

    receipts = []

    async def publish(evidence, succeeded):
        receipts.append((evidence, succeeded))

    observer = _LocalHttpCleanupObserver(publish)
    token = _local_http_cleanup_observer.set(observer)
    controller = ds.ProviderStreamDeadlineController(
        ProviderStreamDeadlines(semantic_progress_timeout_s=0.01)
    )
    controller.observe_semantic(ds.ProviderProgressKind.TERMINAL)

    async def read():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await observer.closed(succeeded=True)
            return "accepted terminal"

    try:
        result = await controller.wait_for(
            read(),
            kinds=(ds.ProviderDeadlineKind.SEMANTIC_IDLE,),
            semantic_cleanup_grace_s=0.1,
            accept_cancelled_result=lambda value: value == "accepted terminal",
        )
        assert result == "accepted terminal"
        assert not receipts
    finally:
        controller.close()
        _local_http_cleanup_observer.reset(token)
