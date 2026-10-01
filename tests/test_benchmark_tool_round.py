"""Benchmark observation must preserve backend authority and restore global state."""

import asyncio
import subprocess
import tracemalloc
from pathlib import Path

import pytest
from scripts import benchmark_tool_round
from scripts.benchmark_tool_round import (
    empty_case,
    measure_bounded_case,
    observe_store,
    run_batch,
    save_report,
)

import cayu._validation as validation
from cayu.sessions.base import InMemorySessionStore, SessionQuery


@pytest.mark.parametrize("fail", [False, True])
def test_observation_preserves_protocols_and_restores_methods(fail):
    store = InMemorySessionStore()
    other_store = InMemorySessionStore()
    methods_before = dict(InMemorySessionStore.__dict__)
    original_walk = validation._walk_bounded_durable_json

    async def scenario():
        with observe_store(store) as (calls, admissions):
            assert store._supports_invocation_lifecycle_command_protocol()
            assert store._supports_terminal_interaction_publication_protocol()
            await store.list_sessions(SessionQuery())
            await other_store.list_sessions(SessionQuery())
            assert calls["list_sessions"] == 1
            before = admissions["checkpoint"]
            validation.copy_durable_json_object({"retained": "safe"}, "checkpoint")
            assert admissions["checkpoint"] == before + 1
            if fail:
                raise RuntimeError("measurement failed")

    if fail:
        with pytest.raises(RuntimeError, match="measurement failed"):
            asyncio.run(scenario())
    else:
        asyncio.run(scenario())
    assert dict(InMemorySessionStore.__dict__) == methods_before
    assert validation._walk_bounded_durable_json is original_walk
    assert store._supports_invocation_lifecycle_command_protocol()
    assert store._supports_terminal_interaction_publication_protocol()


def test_instrumented_batch_completes_each_concurrent_round_once():
    observed = asyncio.run(run_batch(2, 0, 2, instrumented=True))
    assert observed["store_calls"]["publish_runtime_publication"] == 2
    assert observed["full_checkpoint_admissions"] > 0
    assert observed["peak_staged_payload_bytes"] > 0
    assert observed["peak_traced_python_bytes"] is None
    assert observed["retained_traced_python_bytes"] is None
    assert observed["post_release_traced_python_bytes"] is None


def test_allocation_pass_measures_live_and_released_batch_memory():
    observed = asyncio.run(run_batch(1, 0, 1, instrumented=True, trace_python_allocations=True))
    assert observed["peak_traced_python_bytes"] >= observed["retained_traced_python_bytes"] > 0
    assert (
        0 <= observed["post_release_traced_python_bytes"] < observed["retained_traced_python_bytes"]
    )
    assert not tracemalloc.is_tracing()


def test_allocation_trace_is_released_when_the_workload_fails(monkeypatch):
    async def failed_batch(*args, **kwargs):
        assert tracemalloc.is_tracing()
        raise RuntimeError("workload failed")

    monkeypatch.setattr(benchmark_tool_round, "_run_batch", failed_batch)
    with pytest.raises(RuntimeError, match="workload failed"):
        asyncio.run(run_batch(1, 0, 1, instrumented=True, trace_python_allocations=True))
    assert not tracemalloc.is_tracing()


def test_allocation_trace_cannot_contaminate_latency_or_replace_an_existing_trace():
    with pytest.raises(ValueError, match="separate instrumented pass"):
        asyncio.run(run_batch(1, 0, 1, instrumented=False, trace_python_allocations=True))
    tracemalloc.start()
    try:
        with pytest.raises(RuntimeError, match="own allocation trace"):
            asyncio.run(run_batch(1, 0, 1, instrumented=True, trace_python_allocations=True))
        with pytest.raises(RuntimeError, match="own allocation trace"):
            asyncio.run(run_batch(1, 0, 1, instrumented=False))
        assert tracemalloc.is_tracing()
    finally:
        tracemalloc.stop()


def test_case_timeout_retains_finished_samples_without_reporting_completion(monkeypatch):
    completed_sample = {"batch_seconds": 1.0, "session_seconds": [1.0]}

    def timeout_worker(command, **kwargs):
        partial = empty_case(2, 0, 1)
        partial["latency_samples"].append(completed_sample)
        save_report(Path(command[command.index("--output") + 1]), partial)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(benchmark_tool_round.subprocess, "run", timeout_worker)
    result = measure_bounded_case(2, 0, 1, samples=3, trace=False, timeout=0.01)
    assert result["status"] == "timed_out"
    assert result["latency_samples"] == [completed_sample]
    assert result["median_batch_seconds"] is None
    assert result["instrumented"] is None
