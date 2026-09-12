"""Credential-free native child deadlines over an owned loopback HTTP stream."""

from __future__ import annotations

import asyncio
import json

import pytest
from tests.core.test_workflow_failure_evidence import Verifiers

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    ExecutionDeadline,
    SQLiteSessionStore,
    StepError,
    parallel,
    step,
)
from cayu.providers import ChatCompletionsProvider
from cayu.workflows import StepRunOptions


class DeadlineEndpoint:
    def __init__(self, *, output: bool) -> None:
        self.output = output
        self.requests: list[str] = []
        self.handlers: set[asyncio.Task] = set()
        self.errors: list[BaseException] = []
        self.first_frames = asyncio.Event()
        self.closed = asyncio.Event()
        self.expected_slow = 1
        self.slow_frames = 0
        self.slow_closed = 0

    async def __aenter__(self):
        self.server = await asyncio.start_server(self.handle, "127.0.0.1", 0)
        self.base_url = f"http://127.0.0.1:{self.server.sockets[0].getsockname()[1]}/v1"
        return self

    async def __aexit__(self, *args):
        self.server.close()
        await self.server.wait_closed()
        if self.handlers:
            await asyncio.wait_for(asyncio.gather(*self.handlers), 5)

    async def handle(self, reader, writer):
        task = asyncio.current_task()
        self.handlers.add(task)
        slow = False
        try:
            headers = (await reader.readuntil(b"\r\n\r\n")).decode("latin-1").split("\r\n")
            length = next(
                int(line.partition(":")[2])
                for line in headers
                if line.lower().startswith("content-length:")
            )
            request = json.loads(await reader.readexactly(length))
            model = request["model"]
            self.requests.append(model)
            slow = model == "slow"
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                b"Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
            )

            async def send(payload):
                writer.write(f"{len(payload):X}\r\n".encode() + payload + b"\r\n")
                await writer.drain()

            if self.output or not slow:
                payload = {
                    "id": f"completion-{len(self.requests)}",
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "partial" if slow else "sibling result"},
                            "finish_reason": None if slow else "stop",
                        }
                    ],
                }
                await send(f"data: {json.dumps(payload)}\n\n".encode())
            else:
                await send(b": response accepted, no semantic output\n\n")
            if slow:
                self.slow_frames += 1
                if self.slow_frames == self.expected_slow:
                    self.first_frames.set()
                # A client EOF proves this local response socket was released.
                assert await reader.read(1) == b""
            else:
                await send(b"data: [DONE]\n\n")
                writer.write(b"0\r\n\r\n")
                await writer.drain()
        except BaseException as error:
            self.errors.append(error)
        finally:
            writer.close()
            await writer.wait_closed()
            if slow:
                self.slow_closed += 1
                if self.slow_closed == self.expected_slow:
                    self.closed.set()
            self.handlers.remove(task)


@pytest.mark.parametrize("output", [False, True])
@pytest.mark.parametrize("both_expire", [False, True])
def test_native_http_child_deadlines_preserve_terminal_and_sibling(tmp_path, output, both_expire):
    async def scenario():
        loop = asyncio.get_running_loop()
        errors = []
        loop.set_exception_handler(lambda _loop, context: errors.append(context))
        path = tmp_path / "native-http-deadline.sqlite"
        store = SQLiteSessionStore(path)
        async with DeadlineEndpoint(output=output) as endpoint:
            endpoint.expected_slow = 2 if both_expire else 1
            provider = ChatCompletionsProvider(
                api_key="local-contract-key",
                base_url=endpoint.base_url,
                allow_http=True,
                stream_include_usage=False,
            )
            app = CayuApp(enable_logging=False, session_store=store)
            app.register_provider(provider, default=True)
            for model in ("fast", "slow"):
                app.register_agent(AgentSpec(name=model, model=model))
            ctx = Verifiers(app).context("http-deadlines")
            await ctx.start()
            deadlines = [ExecutionDeadline.after(3, scope=f"child-{i}") for i in range(2)]
            result_task = asyncio.create_task(
                parallel(
                    [
                        step(
                            ctx,
                            agent="slow" if both_expire or i == 1 else "fast",
                            step_id=f"child-{i}",
                            prompt="local check",
                            run_options=StepRunOptions(execution_deadline=deadlines[i]),
                        )
                        for i in range(2)
                    ]
                )
            )
            try:
                # Dispatch is attested by parsed requests and drained first frames,
                # not inferred from elapsed time or an unentered provider coroutine.
                await asyncio.wait_for(endpoint.first_frames.wait(), 3)
                result = await asyncio.wait_for(result_task, 10)
                await asyncio.wait_for(endpoint.closed.wait(), 3)
                assert len(result.failures) == endpoint.expected_slow
                assert len(result.successes) == (0 if both_expire else 1)
                if result.successes:
                    assert result.successes[0].text == "sibling result"
                terminals = {}
                for failure in result.failures:
                    evidence = failure.evidence
                    assert evidence.classification == "deadline"
                    assert evidence.deadline_phase == "in_flight"
                    expected = deadlines[int(failure.step_id[-1])]
                    assert evidence.deadline.expires_at == expected.expires_at
                    assert evidence.deadline.scope == expected.scope
                    assert evidence.deadline.source == expected.source
                    assert evidence.session_id == failure.session_id
                    assert evidence.run_epoch is not None
                    assert evidence.terminal_event_id is not None
                    assert evidence.settlement == "unknown"
                    assert not evidence.secondary_failures
                    events = await store.load_events(failure.session_id)
                    assert any(e.type == EventType.MODEL_TEXT_DELTA for e in events) is output
                    terminal = next(e for e in events if e.id == evidence.terminal_event_id)
                    assert not terminal.payload.get("provider_cancellation_failures")
                    terminals[failure.session_id] = terminal
                assert len(endpoint.requests) == 2
                await store.close()
                store = SQLiteSessionStore(path)
                replay_app = CayuApp(enable_logging=False, session_store=store)
                replay_app.register_provider(provider, default=True)
                replay_app.register_agent(AgentSpec(name="slow", model="slow"))
                replay_ctx = Verifiers(replay_app).context("http-deadlines")
                await replay_ctx.start()
                for failure in result.failures:
                    with pytest.raises(StepError) as raised:
                        await step(
                            replay_ctx, agent="slow", step_id=failure.step_id, prompt="check"
                        )
                    for field in (
                        "classification",
                        "deadline_phase",
                        "session_id",
                        "run_epoch",
                        "terminal_event_id",
                        "secondary_failures",
                        "settlement",
                    ):
                        assert getattr(raised.value.evidence, field) == getattr(
                            failure.evidence, field
                        )
                    assert (
                        raised.value.evidence.deadline.model_dump()
                        == failure.evidence.deadline.model_dump()
                    )
                    terminal = terminals[failure.session_id]
                    assert terminal in await store.load_events(failure.session_id)
                assert len(endpoint.requests) == 2
            finally:
                if not result_task.done():
                    result_task.cancel()
                    await asyncio.gather(result_task, return_exceptions=True)
                await provider.aclose()
                await store.close()
            assert endpoint.errors == []
        await loop.shutdown_asyncgens()
        await asyncio.sleep(0)
        assert endpoint.handlers == set()
        assert errors == []

    asyncio.run(scenario())
