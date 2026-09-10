"""Provider-free microbenchmark of the private context overlay, not model quality.

Run with PYTHONPATH=src python benchmarks/argument_continuity.py. Fixture creation
is excluded. Timings include one bounded copied read, integrity validation and
materialization; they exclude database/network latency and the rest of a step.
"""

from __future__ import annotations

import asyncio
import json
import tracemalloc
from copy import deepcopy
from statistics import median
from time import perf_counter_ns

from cayu.core import Message, ProviderStatePart, ToolCallPart
from cayu.runtime import InMemorySessionStore, RunRequest, SessionIdentity
from cayu.runtime._argument_continuity import (
    ArgumentContinuity,
    append_record,
    materialize,
)
from cayu.vaults import SecretRedactor


class ReadProbe:
    def __init__(self, raw):
        self.raw = raw
        self.reads = 0

    async def load_session_operation(self, session_id, key):
        self.reads += 1
        return deepcopy(self.raw)


async def main():
    session = await InMemorySessionStore().create(
        RunRequest(agent_name="benchmark", session_id="benchmark", messages=[]),
        identity=SessionIdentity(provider_name="offline", model="offline"),
    )
    call = Message(
        role="assistant",
        content=[
            ToolCallPart(
                tool_call_id="call",
                tool_name="private_tool",
                arguments={},
                tool_round_id="tround_" + "a" * 32,
                model_step_id="mstep_" + "a" * 32,
                model_attempt_id="matt_" + "a" * 32,
            )
        ],
    )
    private = ArgumentContinuity(
        nonce="a" * 32, profile="b" * 64, arguments={"call": {"text": "synthetic " * 200}}
    )
    raw = append_record(
        None, continuity=private, session=session, messages=[call], request_digest="c" * 64
    )
    native_call = call.model_copy(
        update={
            "content": (
                *call.content,
                ProviderStatePart(
                    provider="openai",
                    state={
                        "type": "function_call",
                        "id": "fc_benchmark",
                        "call_id": "call",
                        "name": "private_tool",
                        "arguments": "{}",
                        "status": "completed",
                    },
                ),
            ),
        }
    )
    full = None
    for index in range(16):
        call_id = f"call-{index}"
        last_call = Message(
            role="assistant",
            content=[
                call.content[0].model_copy(
                    update={"tool_call_id": call_id, "tool_round_id": f"tround_{index:032x}"}
                )
            ],
        )
        value = ArgumentContinuity(
            nonce="a" * 32, profile="b" * 64, arguments={call_id: {"text": "x" * 3500}}
        )
        full = append_record(
            full, continuity=value, session=session, messages=[last_call], request_digest="c" * 64
        )
    long = [Message.text("user", f"synthetic message {i}") for i in range(10000)]
    result = []
    for name, messages, names, records in [
        ("disabled_10000_messages", long, frozenset(), None),
        ("enabled_no_calls_10000_messages", long, frozenset({"private_tool"}), None),
        ("one_record", [call], frozenset({"private_tool"}), raw),
        ("one_record_native_replay", [native_call], frozenset({"private_tool"}), raw),
        ("one_record_10000_messages", [*long, call], frozenset({"private_tool"}), raw),
        ("full_cache_10000_messages", [*long, last_call], frozenset({"private_tool"}), full),
    ]:
        probe = ReadProbe(records)
        kwargs = dict(
            store=probe,
            session=session,
            profile="b" * 64,
            messages=messages,
            names=names,
            redactor=SecretRedactor(),
        )
        for _ in range(20):
            await materialize(**kwargs)
        probe.reads = 0
        timings = []
        for _ in range(200):
            start = perf_counter_ns()
            await materialize(**kwargs)
            timings.append((perf_counter_ns() - start) / 1000)
        reads = probe.reads / 200
        tracemalloc.start()
        await materialize(**kwargs)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result.append(
            dict(
                case=name,
                median_us=round(median(timings), 2),
                p95_us=round(sorted(timings)[189], 2),
                peak_allocated_bytes=peak,
                reads_per_step=reads,
                retained_bytes=len(json.dumps(records).encode()) if records else 0,
            )
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
