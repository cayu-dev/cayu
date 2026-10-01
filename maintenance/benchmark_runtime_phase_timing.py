"""Measure paired, provider-free timing overhead; no sink or durable timing writes.

Without ``--baseline-src`` the benchmark alternates timing enabled and disabled in
one process. With ``--baseline-src`` it also runs the same workload against another
checkout's ``src`` directory (for example the merge base), which has no timing
collection at all, so disabled and enabled are both compared with that baseline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from cayu import (
    AgentSpec,
    CayuApp,
    ExecutionProfileBehaviorIdentity,
    InMemorySessionStore,
    Message,
    ModelProvider,
    ModelStreamEvent,
    RunRequest,
    SQLiteSessionStore,
    Tool,
    ToolResult,
    ToolSpec,
)

_CONFIGURATIONS = ("disabled", "enabled")


class Provider(ModelProvider):
    name = "timing-benchmark"

    def __init__(self):
        self.calls = 0

    @property
    def execution_profile_identity(self):
        return ExecutionProfileBehaviorIdentity(
            name="cayu:benchmark:runtime-timing",
            behavior_version="1",
            implementation_version="1",
        )

    async def stream(self, request):
        self.calls += 1
        if self.calls % 2:
            yield ModelStreamEvent.tool_call(name="noop", arguments={}, id="noop-call")
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
        else:
            yield ModelStreamEvent.completed({"finish_reason": "stop"})


class Noop(Tool):
    spec = ToolSpec(
        name="noop", description="No-op application effect.", input_schema={"type": "object"}
    )

    async def run(self, ctx, args):
        return ToolResult(content="done")


async def run_once(configuration, backend, path):
    store = SQLiteSessionStore(path) if backend == "sqlite" else InMemorySessionStore()
    if configuration == "baseline":
        app = CayuApp(session_store=store, enable_logging=False)
    else:
        from cayu import RuntimeTimingConfig

        app = CayuApp(
            session_store=store,
            enable_logging=False,
            runtime_timing=RuntimeTimingConfig(enabled=configuration == "enabled"),
        )
    app.register_provider(Provider(), default=True)
    app.register_agent(AgentSpec(name="benchmark", model="local"), tools=[Noop()])
    started = perf_counter()
    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="benchmark",
                session_id="benchmark",
                messages=[Message.text("user", "run")],
            )
        )
    ]
    elapsed = perf_counter() - started
    if events[-1].type.value != "session.completed":
        raise RuntimeError("Benchmark workload did not complete.")
    if isinstance(store, SQLiteSessionStore):
        await store.close()
    return elapsed


async def paired(iterations, backend, directory, configurations):
    samples = {name: [] for name in configurations}
    for index in range(iterations + 1):
        # Rotate the order to avoid systematically charging warm-up/drift to one side.
        offset = index % len(configurations)
        for name in configurations[offset:] + configurations[:offset]:
            elapsed = await run_once(name, backend, Path(directory) / f"{index}-{name}.sqlite")
            if index:
                samples[name].append(elapsed)
    return samples


def worker_samples(configuration, iterations, backend, source):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(source), *filter(None, [environment.get("PYTHONPATH")])]
    )
    completed = subprocess.run(
        [
            sys.executable,
            __file__,
            "--worker",
            configuration,
            "--iterations",
            str(iterations),
            "--backend",
            backend,
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return json.loads(completed.stdout)


def summary(backend, samples_per_configuration, samples):
    medians = {name: statistics.median(values) for name, values in samples.items()}
    reference = "baseline" if "baseline" in samples else "disabled"
    result = {
        "backend": backend,
        "samples_per_configuration": samples_per_configuration,
        "sinks": 0,
        "reference": reference,
        "median_seconds": medians,
    }
    for name, value in medians.items():
        if name != reference:
            result[f"{name}_median_difference_percent"] = (value / medians[reference] - 1) * 100
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--backend", choices=("memory", "sqlite"), default="sqlite")
    parser.add_argument(
        "--baseline-src",
        type=Path,
        help="Another checkout's src directory to measure as the no-timing baseline.",
    )
    parser.add_argument("--rounds", type=int, default=5, help="Worker rounds with --baseline-src.")
    parser.add_argument("--worker", choices=("baseline", *_CONFIGURATIONS), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 3 <= args.iterations <= 1000:
        parser.error("--iterations must be between 3 and 1000")
    if not 1 <= args.rounds <= 100:
        parser.error("--rounds must be between 1 and 100")
    if args.worker is not None or args.baseline_src is None:
        configurations = list(_CONFIGURATIONS) if args.worker is None else [args.worker]
        with TemporaryDirectory(prefix="cayu-runtime-timing-") as directory:
            samples = asyncio.run(paired(args.iterations, args.backend, directory, configurations))
        if args.worker is not None:
            print(json.dumps(samples[args.worker]))
        else:
            print(json.dumps(summary(args.backend, args.iterations, samples), indent=2))
        return
    current = Path(__file__).resolve().parents[1] / "src"
    sources = {"baseline": args.baseline_src.resolve(), "disabled": current, "enabled": current}
    names = list(sources)
    samples = {name: [] for name in names}
    for index in range(args.rounds):
        # Each worker is a fresh process; rotate which configuration runs first.
        offset = index % len(names)
        for name in names[offset:] + names[:offset]:
            samples[name].extend(worker_samples(name, args.iterations, args.backend, sources[name]))
    print(json.dumps(summary(args.backend, args.iterations * args.rounds, samples), indent=2))


if __name__ == "__main__":
    main()
