"""Measure provider-free SQLite model/tool steps, optionally under cProfile.

Run with PYTHONPATH=src and the same interpreter/dependencies in both checkouts.
Each sample owns a fresh database; setup and transcript verification are excluded.
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import json
import platform
import statistics
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter, process_time

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest, Tool, ToolResult, ToolSpec
from cayu.providers import ModelProvider, ModelStreamEvent
from cayu.storage import SQLiteSessionStore


class Noop(Tool):
    spec = ToolSpec(name="noop", description="No-op", input_schema={"type": "object"})

    async def run(self, ctx, args):
        return ToolResult(content="done")


class Provider(ModelProvider):
    name = "benchmark"

    def __init__(self, steps):
        self.steps = steps
        self.calls = 0

    async def stream(self, request):
        self.calls += 1
        if self.calls < self.steps:
            yield ModelStreamEvent.tool_call(name="noop", arguments={}, id=f"noop-{self.calls}")
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
        else:
            yield ModelStreamEvent.text_delta("done")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})


async def sample(steps):
    with tempfile.TemporaryDirectory() as directory:
        store = SQLiteSessionStore(Path(directory) / "session.sqlite")
        try:
            app = CayuApp(session_store=store, enable_logging=False)
            provider = Provider(steps)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="benchmark", model="scripted"), tools=[Noop()])
            request = RunRequest(
                agent_name="benchmark", max_steps=steps, messages=[Message.text("user", "run")]
            )
            cpu, wall = process_time(), perf_counter()
            events = [event async for event in app.run(request)]
            result = {"cpu_seconds": process_time() - cpu, "wall_seconds": perf_counter() - wall}
            assert events[-1].type is EventType.SESSION_COMPLETED
            assert provider.calls == steps
            return result
        finally:
            await store.close()


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--profile", type=Path)
    args = parser.parse_args()
    if args.steps < 2 or args.samples < 1:
        parser.error("steps must be at least 2 and samples positive")
    await sample(2)
    profiler = cProfile.Profile() if args.profile else None
    if profiler:
        profiler.enable()
    samples = [await sample(args.steps) for _ in range(args.samples)]
    if profiler:
        profiler.disable()
        profiler.dump_stats(args.profile)
    import inspect

    source_root = Path(inspect.getfile(CayuApp)).resolve().parents[2]
    revision = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "-C", str(source_root), "status", "--porcelain", "--", "src/cayu"], text=True
        ).strip()
    )
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "revision": revision,
                "source_modified": dirty,
                "steps": args.steps,
                "samples": samples,
                "median_cpu_seconds_per_step": statistics.median(s["cpu_seconds"] for s in samples)
                / args.steps,
                "median_wall_seconds_per_step": statistics.median(
                    s["wall_seconds"] for s in samples
                )
                / args.steps,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
