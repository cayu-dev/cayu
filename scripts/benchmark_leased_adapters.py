"""Measure fresh public completion operations using existing characterization fixtures.

Run the same script in separate processes against both checkouts, with the same
Python environment and development dependencies. Setup and one warmup batch
are outside timing. Each sample creates fresh in-memory stores. Instrumentation
runs separately and counts public async store calls, including internal calls.

Example:
    python scripts/benchmark_leased_adapters.py --repo . --output /tmp/leased.json
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import inspect
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path


async def measure(repo: Path, samples: int, *, isolate_gc: bool = False) -> dict[str, object]:
    # Select the target checkout before importing Cayu or its frozen fixtures.
    # A process measures exactly one revision.
    gc_initially_enabled = gc.isenabled()
    sys.path[:0] = [str(repo / "src"), str(repo)]
    from tests.core.test_completion_result_resolvers import _contract as resolver_contract
    from tests.core.test_completion_result_resolvers import (
        _prepared_app,
        _resolution_request,
        _Resolver,
        _result,
    )
    from tests.core.test_completion_verifier_adapters import (
        RecordingVerifier,
        _accepted_decision,
        _execution_request,
        _proposal,
    )
    from tests.core.test_completion_verifier_adapters import _contract as verifier_contract

    from cayu import CayuApp, InMemoryTaskStore
    from cayu.build_provenance import current_runtime_build_provenance

    async def batch(kind: str, size: int, *, counted: bool = False):
        operations = []
        adapters = []
        counts: dict[str, int] = {}
        originals = {}
        try:
            for _ in range(size):
                if kind == "verifier":
                    store = InMemoryTaskStore()
                    contract = verifier_contract()
                    proposal = await _proposal(store, contract)
                    app = CayuApp(task_store=store, enable_logging=False)
                    adapter = RecordingVerifier(_accepted_decision())
                    app.register_completion_verifier(contract.verifier, adapter)
                    request = _execution_request(proposal)
                    operations.append(
                        lambda app=app, request=request: app.verify_completion_proposal(request)
                    )
                    stores = [store]
                else:
                    app, sessions, store, decision = await _prepared_app()
                    adapter = _Resolver(_result("1"))
                    app.register_completion_result_resolver(
                        resolver_contract().result_resolver, adapter
                    )
                    request = _resolution_request(decision)
                    operations.append(
                        lambda app=app, request=request: app.resolve_completion_result(request)
                    )
                    stores = [store, sessions]
                adapters.append(adapter)
                if counted:
                    for store in stores:
                        cls = type(store)
                        for name in dir(cls):
                            if name.startswith("_") or (cls, name) in originals:
                                continue
                            function = getattr(cls, name)
                            if not inspect.iscoroutinefunction(function):
                                continue
                            label = f"{cls.__name__}.{name}"

                            async def wrapper(
                                self, *args, original=function, label=label, **kwargs
                            ):
                                counts[label] = counts.get(label, 0) + 1
                                return await original(self, *args, **kwargs)

                            # Awaiting the exact original to quiescence preserves
                            # the concrete store's mutation guarantee. Instance
                            # shadowing would invalidate that structural proof.
                            originals[cls, name] = (name in vars(cls), function)
                            setattr(cls, name, wrapper)
            counts.clear()  # Exclude later fixture setup under the shared class wrappers.
            gc_was_enabled = gc.isenabled()
            if isolate_gc:
                gc.collect()
                gc.disable()
            try:
                started = time.perf_counter()
                results = await asyncio.gather(*(operation() for operation in operations))
                elapsed = time.perf_counter() - started
            finally:
                if isolate_gc and gc_was_enabled:
                    gc.enable()
            if len(results) != size or any(len(adapter.requests) != 1 for adapter in adapters):
                raise RuntimeError("Benchmark did not complete exactly one dispatch per operation.")
            return elapsed, counts
        finally:
            for (cls, name), (declared, original) in originals.items():
                if declared:
                    setattr(cls, name, original)
                else:
                    delattr(cls, name)

    rows = []
    for kind in ("verifier", "resolver"):
        for concurrency in (1, 8, 32):
            await batch(kind, concurrency)
            elapsed = [(await batch(kind, concurrency))[0] for _ in range(samples)]
            _, counts = await batch(kind, concurrency, counted=True)
            rows.append(
                {
                    "kind": kind,
                    "concurrency": concurrency,
                    "samples_s": elapsed,
                    "median_s": statistics.median(elapsed),
                    "public_store_calls": counts,
                }
            )
    return {
        "python": sys.version,
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "runtime_source_digest": current_runtime_build_provenance().artifact_digest,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "isolate_gc": isolate_gc,
        "gc_initially_enabled": gc_initially_enabled,
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument(
        "--isolate-gc",
        action="store_true",
        help="Collect cyclic garbage before each batch and disable collection during timing.",
    )
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    repo = args.repo.resolve()
    if not (repo / "src/cayu").is_dir() or not (repo / "tests/core").is_dir():
        parser.error("--repo must contain the Cayu source and characterization tests")
    report = (
        json.dumps(asyncio.run(measure(repo, args.samples, isolate_gc=args.isolate_gc)), indent=2)
        + "\n"
    )
    if args.output is None:
        sys.stdout.write(report)
    else:
        args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
