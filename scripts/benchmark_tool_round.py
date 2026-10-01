#!/usr/bin/env python3
"""Measure a fixed public tool-round workload on the in-memory backend.

Run the same script and dependency versions against both source revisions:
    PYTHONPATH=src python scripts/benchmark_tool_round.py > /tmp/tool-round.json

Each case uses fresh stores and one two-step round per session. Latency samples
have no instrumentation. A separate run counts public async store calls
(including backend-internal calls) and counts full checkpoint admissions at the
existing durable-JSON walker. The public governor supplies peak staged payload
bytes; that is not total heap. Optional --trace-python-allocations measures
Python allocations from batch construction through completion, including the
bytes retained after garbage collection with the application/stores alive and
after releasing them. It excludes native allocations and can be substantially
slower. Allocation tracing never runs during latency samples.
Results describe this scripted workload, not provider or database performance.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import inspect
import itertools
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
from collections import Counter
from contextlib import contextmanager
from functools import wraps
from importlib.metadata import version
from math import isfinite
from pathlib import Path

import cayu._validation as validation
from cayu import (
    AgentSpec,
    CayuApp,
    CayuConfig,
    EventType,
    InMemorySessionStore,
    Message,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
    Tool,
    ToolExecutionConfig,
    ToolResult,
    ToolSpec,
)
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity


class FixedResultTool(Tool):
    spec = ToolSpec(
        name="fixed_result",
        description="Return a fixed-size result without external I/O.",
        input_schema={
            "type": "object",
            "properties": {"index": {"type": "integer"}},
            "required": ["index"],
        },
        max_terminal_payload_bytes=65_536,
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="benchmark:fixed-result", behavior_version="1", implementation_version="1"
        ),
    )

    def __init__(self):
        super().__init__()
        self.calls = Counter()

    async def run(self, ctx, args):
        self.calls[ctx.session_id, args["index"]] += 1
        return ToolResult(content="x" * 128)


@contextmanager
def observe_store(store):
    """Observe class-owned methods without shadowing authenticated instance methods.

    Only the separate measurement pass runs while these wrappers are installed.
    Calls on any other memory store delegate without being counted.
    """
    calls = Counter()
    admissions = Counter()
    original_walk = validation._walk_bounded_durable_json
    store_type = type(store)
    missing = object()
    originals = {}

    def count_checkpoint(value, field_name, **kwargs):
        if field_name == "checkpoint":
            admissions["checkpoint"] += 1
        return original_walk(value, field_name, **kwargs)

    def instrument(name, method):
        @wraps(method)
        async def counted(self, *args, **kwargs):
            if self is store:
                calls[name] += 1
            return await method(self, *args, **kwargs)

        return counted

    try:
        for name in dir(store_type):
            method = getattr(store_type, name)
            if not name.startswith("_") and inspect.iscoroutinefunction(method):
                originals[name] = store_type.__dict__.get(name, missing)
                setattr(store_type, name, instrument(name, method))
        validation._walk_bounded_durable_json = count_checkpoint
        yield calls, admissions
    finally:
        validation._walk_bounded_durable_json = original_walk
        for name, original in originals.items():
            if original is missing:
                delattr(store_type, name)
            else:
                setattr(store_type, name, original)


async def run_batch(
    call_count, history_count, session_count, *, instrumented, trace_python_allocations=False
):
    if trace_python_allocations and not instrumented:
        raise ValueError("Allocation tracing requires a separate instrumented pass.")
    if tracemalloc.is_tracing():
        raise RuntimeError("The benchmark requires its own allocation trace.")
    if not trace_python_allocations:
        return await _run_batch(call_count, history_count, session_count, instrumented=instrumented)

    gc.collect()
    tracemalloc.start()
    try:
        result = await _run_batch(
            call_count,
            history_count,
            session_count,
            instrumented=True,
            trace_python_allocations=True,
        )
        # Let completed task callbacks release their arguments before measuring
        # what survives the fresh application's entire lifetime.
        await asyncio.sleep(0)
        gc.collect()
        result["post_release_traced_python_bytes"] = tracemalloc.get_traced_memory()[0]
        return result
    finally:
        tracemalloc.stop()


async def _run_batch(
    call_count, history_count, session_count, *, instrumented, trace_python_allocations=False
):
    store = InMemorySessionStore()
    tool = FixedResultTool()

    def response(request):
        if request.messages[-1].role.value == "tool":
            return [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        return [
            *(
                ModelStreamEvent.tool_call(
                    id=f"call-{index}", name="fixed_result", arguments={"index": index}
                )
                for index in range(call_count)
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]

    provider = ScriptedModelProvider(response_factory=response)
    app = CayuApp(
        session_store=store,
        enable_logging=False,
        config=CayuConfig(tool_execution=ToolExecutionConfig(max_parallel_tool_calls=4)),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="benchmark", model="scripted-model"), tools=[tool])
    requests = [
        RunRequest(
            session_id=f"benchmark-{index}",
            agent_name="benchmark",
            max_steps=2,
            messages=[
                *(
                    Message.text("user" if index % 2 == 0 else "assistant", "h" * 128)
                    for index in range(history_count)
                ),
                Message.text("user", "run one tool round"),
            ],
        )
        for index in range(session_count)
    ]

    async def run_one(request):
        started = time.perf_counter()
        last = None
        async for event in app.run(request):
            last = event.type
        elapsed = time.perf_counter() - started
        assert last is EventType.SESSION_COMPLETED
        return elapsed

    gc.collect()
    if instrumented:
        with observe_store(store) as (calls, admissions):
            await asyncio.gather(*(run_one(request) for request in requests))
        result = {
            "store_calls": dict(sorted(calls.items())),
            "full_checkpoint_admissions": admissions["checkpoint"],
            "peak_traced_python_bytes": None,
            "retained_traced_python_bytes": None,
            "post_release_traced_python_bytes": None,
            "peak_staged_payload_bytes": app.tool_terminal_publication_status().maximum_staged_bytes,
        }
    else:
        started = time.perf_counter()
        durations = await asyncio.gather(*(run_one(request) for request in requests))
        result = {"batch_seconds": time.perf_counter() - started, "session_seconds": durations}

    assert len(provider.requests) == 2 * session_count
    assert tool.calls == Counter(
        {
            (f"benchmark-{session}", call): 1
            for session in range(session_count)
            for call in range(call_count)
        }
    )
    metrics = app.tool_terminal_publication_status()
    assert metrics.active_round_reservations == metrics.staged_count == 0
    if trace_python_allocations:
        gc.collect()
        retained, peak = tracemalloc.get_traced_memory()
        result.update(peak_traced_python_bytes=peak, retained_traced_python_bytes=retained)
    return result


def positive(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def nonnegative(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def positive_seconds(value):
    parsed = float(value)
    if not isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def empty_case(calls, history, sessions):
    return {
        "calls_per_round": calls,
        "history_messages": history,
        "sessions": sessions,
        "status": "running",
        "latency_samples": [],
        "median_batch_seconds": None,
        "instrumented": None,
    }


async def measure_case(calls, history, sessions, *, samples, trace, output=None):
    result = empty_case(calls, history, sessions)
    started = time.perf_counter()
    for _ in range(samples):
        result["latency_samples"].append(
            await run_batch(calls, history, sessions, instrumented=False)
        )
        save_report(output, result)
    result["instrumented"] = await run_batch(
        calls, history, sessions, instrumented=True, trace_python_allocations=trace
    )
    result["status"] = "completed"
    result["median_batch_seconds"] = statistics.median(
        sample["batch_seconds"] for sample in result["latency_samples"]
    )
    result["elapsed_seconds"] = time.perf_counter() - started
    save_report(output, result)
    return result


def save_report(path, report):
    """Replace an optional report atomically so interrupted runs retain prior cases."""
    if path is None:
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def measure_bounded_case(calls, history, sessions, *, samples, trace, timeout):
    """Give a synthetic memory-only case a process deadline, including cleanup."""
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="cayu-round-benchmark-") as directory:
        output = Path(directory) / "case.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--calls-per-round",
            str(calls),
            "--history-messages",
            str(history),
            "--sessions",
            str(sessions),
            "--samples",
            str(samples),
            "--output",
            str(output),
        ]
        if trace:
            command.append("--trace-python-allocations")
        try:
            subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            # subprocess.run kills and waits for this owned worker. No external
            # effects or durable stores exist in this fixed synthetic workload.
            result = (
                json.loads(output.read_text())
                if output.exists()
                else empty_case(calls, history, sessions)
            )
            result.update(status="timed_out", median_batch_seconds=None, instrumented=None)
            result["elapsed_seconds"] = time.perf_counter() - started
            return result
        except subprocess.CalledProcessError as error:
            print(error.stderr, file=sys.stderr)
            raise
        return json.loads(output.read_text())


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls-per-round", nargs="+", type=positive, default=[2, 16, 64])
    parser.add_argument("--history-messages", nargs="+", type=nonnegative, default=[0, 128, 512])
    parser.add_argument("--sessions", nargs="+", type=positive, default=[1, 8])
    parser.add_argument("--samples", type=positive, default=3)
    parser.add_argument("--trace-python-allocations", action="store_true")
    parser.add_argument("--case-timeout", type=positive_seconds, default=None)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    loaded_source = Path(inspect.getfile(InMemorySessionStore)).resolve()
    if not loaded_source.is_relative_to(root / "src"):
        parser.error("set PYTHONPATH=src to measure this checkout's source")

    if args.worker:
        if (
            any(
                len(values) != 1
                for values in (args.calls_per_round, args.history_messages, args.sessions)
            )
            or args.output is None
        ):
            parser.error("an internal worker requires exactly one case and an output path")
        await run_batch(2, 0, 1, instrumented=False)
        await measure_case(
            args.calls_per_round[0],
            args.history_messages[0],
            args.sessions[0],
            samples=args.samples,
            trace=args.trace_python_allocations,
            output=args.output,
        )
        return

    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()

    if git("status", "--porcelain", "--", "src/cayu"):
        parser.error("commit runtime changes before measuring an exact revision")
    report = {
        "schema_version": 2,
        "revision": git("rev-parse", "HEAD"),
        "source_tree": git("rev-parse", "HEAD:src/cayu"),
        "workload_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: version(name) for name in ("cayu", "pydantic", "pydantic-core")},
        "backend": "memory",
        "allocation_scope": (
            "Fresh batch construction and execution; retained after GC with app/stores alive; "
            "released after returning from the batch and GC. Python-traced allocations only."
            if args.trace_python_allocations
            else None
        ),
        "configuration": {
            key: value for key, value in vars(args).items() if key not in {"output", "worker"}
        },
        "finished": False,
        "complete": False,
        "cases": [],
    }
    save_report(args.output, report)
    # Warm lazy imports once outside measured cases; each case still owns fresh state.
    if args.case_timeout is None:
        await run_batch(2, 0, 1, instrumented=False)
    for calls, history, sessions in itertools.product(
        args.calls_per_round, args.history_messages, args.sessions
    ):
        print(f"calls={calls} history={history} sessions={sessions}", file=sys.stderr, flush=True)
        if args.case_timeout is None:
            result = await measure_case(
                calls,
                history,
                sessions,
                samples=args.samples,
                trace=args.trace_python_allocations,
            )
        else:
            result = measure_bounded_case(
                calls,
                history,
                sessions,
                samples=args.samples,
                trace=args.trace_python_allocations,
                timeout=args.case_timeout,
            )
        report["cases"].append(result)
        save_report(args.output, report)
    report["finished"] = True
    report["complete"] = all(case["status"] == "completed" for case in report["cases"])
    save_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
