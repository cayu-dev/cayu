"""Provider-free native round benchmark; emits one verified JSON record per sample."""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import json
import platform
import tempfile
import time
import tracemalloc
from pathlib import Path

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.evals.testing import ScriptedModelProvider
from cayu.messages import Message, ToolResultPart
from cayu.providers.base import ModelStreamEvent
from cayu.sessions.base import InMemorySessionStore, RunRequest
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.base import Tool, ToolResult, ToolSpec


class Echo(Tool):
    spec = ToolSpec(
        name="echo",
        description="Return a small fixed result",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        parallel_safe=True,
    )

    def __init__(self, payload_bytes: int):
        self.calls = 0
        self.content = "Small retained observation" + "x" * payload_bytes

    async def run(self, ctx, args):
        self.calls += 1
        return ToolResult(content=self.content)


async def measure(n: int, backend: str, payload_bytes: int, repeat: int, allocations: bool):
    with tempfile.TemporaryDirectory() as directory:
        started = time.perf_counter()
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(Path(directory) / "session.db")
        )
        app = CayuApp(session_store=store, enable_logging=False)
        provider = ScriptedModelProvider(
            [
                [
                    *[
                        ModelStreamEvent.tool_call(id=f"call-{i}", name="echo", arguments={})
                        for i in range(n)
                    ],
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        tool = Echo(payload_bytes)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="worker", model="scripted-model"), tools=[tool])
        setup = time.perf_counter() - started
        if allocations:
            tracemalloc.start()
        cpu_started = time.process_time()
        started = time.perf_counter()
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="worker",
                    session_id=f"round-{n}",
                    messages=[Message.text("user", "Run the requested tools")],
                )
            )
        ]
        elapsed = time.perf_counter() - started
        cpu_elapsed = time.process_time() - cpu_started
        peak_bytes = tracemalloc.get_traced_memory()[1] if allocations else None
        if allocations:
            tracemalloc.stop()
        completed = [e for e in events if str(e.type) == "tool.call.completed"]
        assert tool.calls == n and len(completed) == n
        assert sum(str(e.type) == "session.completed" for e in events) == 1
        assert len(provider.requests) == 2
        results = [
            part
            for message in provider.requests[-1].messages
            for part in message.content
            if isinstance(part, ToolResultPart)
        ]
        assert len(results) == n
        assert {part.tool_call_id for part in results} == {f"call-{i}" for i in range(n)}
        assert all(part.content == tool.content and not part.is_error for part in results)
        if isinstance(store, SQLiteSessionStore):
            await store.close()
        print(
            json.dumps(
                {
                    "backend": backend,
                    "calls": n,
                    "repeat": repeat,
                    "payload_bytes": payload_bytes,
                    "setup_seconds": setup,
                    "round_seconds": elapsed,
                    "cpu_seconds": cpu_elapsed,
                    "peak_traced_bytes": peak_bytes,
                    "final_completed": True,
                    "tool_results": len(completed),
                    "model_requests": len(provider.requests),
                    "events": len(events),
                    "python": platform.python_version(),
                }
            ),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, nargs="+", default=[1, 10, 20, 40])
    parser.add_argument(
        "--backends", nargs="+", choices=["memory", "sqlite"], default=["memory", "sqlite"]
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--payload-bytes", type=int, default=0)
    parser.add_argument("--profile", type=Path)
    parser.add_argument(
        "--allocations",
        action="store_true",
        help="Trace peak Python memory separately from timing runs",
    )
    args = parser.parse_args()
    profiler = cProfile.Profile() if args.profile else None
    if profiler:
        profiler.enable()
    for backend in args.backends:
        for n in args.calls:
            for repeat in range(args.repeats):
                asyncio.run(measure(n, backend, args.payload_bytes, repeat, args.allocations))
    if profiler:
        profiler.disable()
        profiler.dump_stats(str(args.profile))


if __name__ == "__main__":
    main()
