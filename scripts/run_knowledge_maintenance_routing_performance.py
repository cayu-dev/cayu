#!/usr/bin/env python3
"""Measure hermetic reviewed-maintenance candidate-routing overhead."""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

from cayu import (
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeEntry,
    KnowledgeMaintenanceCandidateSignal,
    KnowledgeMaintenanceRouter,
    KnowledgeMaintenanceRouterConfig,
    KnowledgeMaintenanceRoutingRequest,
    KnowledgeMaintenanceSignalKind,
    KnowledgeRevisionRef,
    KnowledgeVisibility,
    SQLiteKnowledgeStore,
)

_SCHEMA_VERSION = "cayu.knowledge_maintenance_routing_performance.v1"
_DEFAULT_CANDIDATES = 50
_DEFAULT_ITERATIONS = 30
_WARMUP_ITERATIONS = 3
_NOW = datetime(2026, 8, 25, 12, 0, tzinfo=UTC)

_CEILINGS = {
    "memory_zero_candidate_routing_p95_ms": 5.0,
    "sqlite_zero_candidate_routing_p95_ms": 5.0,
    "memory_bounded_routing_p95_ms": 100.0,
    "sqlite_bounded_routing_p95_ms": 150.0,
    "memory_routing_p95_ms_per_candidate": 5.0,
    "sqlite_routing_p95_ms_per_candidate": 10.0,
    "candidate_payload_bytes_per_candidate": 4_096,
}


def _latency_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = max(0, (95 * len(ordered) + 99) // 100 - 1)
    return {
        "p50_ms": round(median(ordered), 6),
        "p95_ms": round(ordered[p95_index], 6),
    }


def _entries(count: int) -> list[KnowledgeEntry]:
    return [
        KnowledgeEntry(
            id=f"maintenance-routing-performance-{index:03d}",
            text=(
                "A bounded canonical operational fact used to measure deterministic "
                f"candidate routing. Candidate ordinal {index:03d}."
            ),
            namespace="performance:routing",
            labels={"workload": "routing"},
            visibility=KnowledgeVisibility.PROJECT,
            created_at=_NOW,
            updated_at=_NOW,
        )
        for index in range(count)
    ]


def _signals(count: int) -> tuple[KnowledgeMaintenanceCandidateSignal, ...]:
    return tuple(
        KnowledgeMaintenanceCandidateSignal(
            id=f"maintenance-routing-signal-{index:03d}",
            kind=KnowledgeMaintenanceSignalKind.EXACT_REFERENCE,
            references=(
                KnowledgeRevisionRef(
                    entry_id=f"maintenance-routing-performance-{index:03d}",
                    revision=1,
                ),
            ),
            producer_id="hermetic-performance-runner",
            producer_version="1",
            reason_code="explicit_reference",
            observed_at=_NOW,
        )
        for index in range(count)
    )


def _request(
    request_id: str,
    signals: tuple[KnowledgeMaintenanceCandidateSignal, ...],
    scope: KnowledgeAccessScope,
) -> KnowledgeMaintenanceRoutingRequest:
    return KnowledgeMaintenanceRoutingRequest(
        id=request_id,
        policy_id="hermetic-reviewed-routing-v1",
        namespace="performance:routing",
        labels={"workload": "routing"},
        access_scope=scope,
        signals=signals,
        created_at=_NOW,
    )


async def _close(store: Any) -> None:
    close = getattr(store, "close", None)
    if callable(close):
        await close()


async def _backend_result(
    backend: str,
    *,
    candidate_count: int,
    iterations: int,
    directory: Path,
) -> dict[str, Any]:
    scope = KnowledgeAccessScope.for_namespace(
        "performance:routing",
        required_labels={"workload": "routing"},
        allowed_visibilities=[KnowledgeVisibility.PROJECT],
    )
    store = (
        InMemoryKnowledgeStore(access_scope=scope)
        if backend == "memory"
        else SQLiteKnowledgeStore(directory / "routing.sqlite", access_scope=scope)
    )
    try:
        entries = _entries(candidate_count)
        for entry in entries:
            await store.create_entry(entry)
        config = KnowledgeMaintenanceRouterConfig(
            max_signals=candidate_count,
            max_candidate_reads=candidate_count,
            max_candidates=candidate_count,
            max_candidate_bytes=1024 * 1024,
            max_concurrency=min(candidate_count, 8),
        )
        router = KnowledgeMaintenanceRouter(store, config=config)
        zero_request = _request("routing-performance-zero", (), scope)
        populated_request = _request(
            "routing-performance-populated",
            _signals(candidate_count),
            scope,
        )
        for _ in range(_WARMUP_ITERATIONS):
            await router.route(zero_request)
            await router.route(populated_request)

        zero_latencies: list[float] = []
        populated_latencies: list[float] = []
        last_result = None
        for _ in range(iterations):
            started = time.perf_counter_ns()
            zero_result = await router.route(zero_request)
            zero_latencies.append((time.perf_counter_ns() - started) / 1_000_000)
            if zero_result.candidates or zero_result.loaded_reference_count:
                raise RuntimeError("Zero-candidate routing performed candidate work.")
            if zero_result.relation_payload_bytes:
                raise RuntimeError("Zero-candidate routing performed relation work.")

            started = time.perf_counter_ns()
            last_result = await router.route(populated_request)
            populated_latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        if last_result is None or len(last_result.candidates) != candidate_count:
            raise RuntimeError("Bounded routing performance workload lost candidates.")
        if last_result.omissions or last_result.truncated:
            raise RuntimeError("Bounded routing performance workload was incomplete.")
        if last_result.relation_payload_bytes:
            raise RuntimeError("Exact-reference routing performed relation work.")

        revisions_after: list[int] = []
        for entry in entries:
            current = await store.get_entry(entry.id)
            if current is None:
                raise RuntimeError("Read-only routing removed canonical knowledge.")
            revisions_after.append(current.revision)
        if revisions_after != [1] * candidate_count:
            raise RuntimeError("Read-only routing changed canonical knowledge revisions.")
        per_candidate = [latency / candidate_count for latency in populated_latencies]
        return {
            "backend": backend,
            "candidate_count": candidate_count,
            "iteration_count": iterations,
            "zero_candidate_routing_latency": _latency_summary(zero_latencies),
            "routing_latency": _latency_summary(populated_latencies),
            "routing_latency_per_candidate": _latency_summary(per_candidate),
            "candidate_payload_bytes": last_result.candidate_payload_bytes,
            "relation_payload_bytes": last_result.relation_payload_bytes,
            "candidate_payload_bytes_per_candidate": round(
                last_result.candidate_payload_bytes / candidate_count,
                6,
            ),
            "loaded_reference_count": last_result.loaded_reference_count,
            "routed_signal_count": len(last_result.routed_signals),
            "knowledge_revision_mutations": 0,
        }
    finally:
        await _close(store)


def _ceiling_findings(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for result in results:
        backend = result["backend"]
        checks = {
            f"{backend}_zero_candidate_routing_p95_ms": result["zero_candidate_routing_latency"][
                "p95_ms"
            ],
            f"{backend}_bounded_routing_p95_ms": result["routing_latency"]["p95_ms"],
            f"{backend}_routing_p95_ms_per_candidate": result["routing_latency_per_candidate"][
                "p95_ms"
            ],
            "candidate_payload_bytes_per_candidate": result[
                "candidate_payload_bytes_per_candidate"
            ],
        }
        for metric, observed in checks.items():
            ceiling = _CEILINGS[metric]
            if observed > ceiling:
                findings.append(
                    {
                        "backend": backend,
                        "metric": metric,
                        "observed": round(observed, 6),
                        "ceiling": ceiling,
                    }
                )
    return findings


async def _run(candidate_count: int, iterations: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="cayu-maintenance-routing-performance-") as raw:
        directory = Path(raw)
        results = [
            await _backend_result(
                backend,
                candidate_count=candidate_count,
                iterations=iterations,
                directory=directory,
            )
            for backend in ("memory", "sqlite")
        ]
    findings = _ceiling_findings(results)
    return {
        "schema_version": _SCHEMA_VERSION,
        "workload": {
            "candidate_count": candidate_count,
            "iterations": iterations,
            "signal_kind": "exact_reference",
            "provider_calls": 0,
            "model_calls": 0,
            "store_writes_during_routing": 0,
        },
        "control": {
            "kind": "current_runtime_zero_candidates",
            "historical_pre_feature": False,
            "candidate_count": 0,
            "store_reads": 0,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "ceilings": _CEILINGS,
        "results": results,
        "ceiling_findings": findings,
        "within_ceilings": not findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=int, default=_DEFAULT_CANDIDATES)
    parser.add_argument("--iterations", type=int, default=_DEFAULT_ITERATIONS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.candidates <= 50:
        parser.error("--candidates must be between 1 and 50")
    if not 1 <= args.iterations <= 1_000:
        parser.error("--iterations must be between 1 and 1000")

    report = asyncio.run(_run(args.candidates, args.iterations))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 1 if args.check and not report["within_ceilings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
