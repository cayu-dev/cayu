"""Measure CPU and durable writes for a scripted Chat Completions reasoning stream.

Use the same script and interpreter with PYTHONPATH pointing to each revision's
src directory. No network or paid provider calls occur; each sample owns a fresh
SQLite database. Setup and post-run verification are outside the timed interval.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import tempfile
from pathlib import Path
from time import perf_counter, process_time

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.messages import ThinkingPart
from cayu.providers.chat_completions import ChatCompletionsProvider
from cayu.storage import SQLiteSessionStore


class Transport:
    def __init__(self, chunks):
        self.chunks = chunks

    async def stream_chat_completions(self, **kwargs):
        for _ in range(self.chunks):
            yield {"choices": [{"delta": {"reasoning": "x"}}]}
        yield {"choices": [{"delta": {"content": "done"}}]}
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}


async def sample(chunks):
    with tempfile.TemporaryDirectory() as directory:
        store = SQLiteSessionStore(Path(directory) / "session.sqlite")
        try:
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(
                ChatCompletionsProvider(api_key="test", transport=Transport(chunks)), default=True
            )
            app.register_agent(AgentSpec(name="benchmark", model="scripted-reasoner"))
            request = RunRequest(
                session_id="benchmark",
                agent_name="benchmark",
                messages=[Message.text("user", "run")],
            )
            cpu, wall = process_time(), perf_counter()
            events = [event async for event in app.run(request)]
            result = {"cpu_seconds": process_time() - cpu, "wall_seconds": perf_counter() - wall}
            assert events[-1].type is EventType.SESSION_COMPLETED
            stored = await store.load_events("benchmark")
            deltas = [event for event in stored if event.type is EventType.MODEL_THINKING_DELTA]
            assert "".join(event.payload["delta"] for event in deltas) == "x" * chunks
            transcript = await store.load_transcript("benchmark")
            assert (
                "".join(
                    part.text for part in transcript[-1].content if isinstance(part, ThinkingPart)
                )
                == "x" * chunks
            )
            return {**result, "thinking_events": len(deltas), "total_events": len(stored)}
        finally:
            await store.close()


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=int, default=3000)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()
    if args.chunks < 1 or args.samples < 1:
        parser.error("chunks and samples must be positive")
    await sample(1)
    samples = [await sample(args.chunks) for _ in range(args.samples)]
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "reasoning_chunks": args.chunks,
                "samples": samples,
                "median_cpu_seconds": statistics.median(s["cpu_seconds"] for s in samples),
                "median_wall_seconds": statistics.median(s["wall_seconds"] for s in samples),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
