#!/usr/bin/env python3
"""Measure hermetic knowledge-maintenance planning and evaluation overhead."""

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
    KnowledgeMaintenanceEvaluationVerdict,
    KnowledgeMaintenanceEvaluatorDecision,
    KnowledgeMaintenanceEvaluatorOutput,
    KnowledgeMaintenanceEvidenceMapping,
    KnowledgeMaintenancePlanDraft,
    KnowledgeMaintenancePlanEndpoint,
    KnowledgeMaintenancePlanEndpointKind,
    KnowledgeMaintenancePlannerOutput,
    KnowledgeMaintenancePlanningConfig,
    KnowledgeMaintenancePlanningOutcome,
    KnowledgeMaintenancePlanningWorkflow,
    KnowledgeMaintenanceRelationDraft,
    KnowledgeMaintenanceReplacementDraft,
    KnowledgeMaintenanceRouter,
    KnowledgeMaintenanceRouterConfig,
    KnowledgeMaintenanceRoutingRequest,
    KnowledgeMaintenanceSignalKind,
    KnowledgeRelationKind,
    KnowledgeRevisionRef,
    KnowledgeVisibility,
    SQLiteKnowledgeStore,
)
from cayu._validation import canonical_durable_json_bytes

_SCHEMA_VERSION = "cayu.knowledge_maintenance_planning_performance.v1"
_DEFAULT_CANDIDATES = 50
_DEFAULT_ITERATIONS = 30
_WARMUP_ITERATIONS = 3
_NOW = datetime(2026, 8, 26, 12, 0, tzinfo=UTC)

_CEILINGS = {
    "memory_zero_candidate_planning_p95_ms": 5.0,
    "sqlite_zero_candidate_planning_p95_ms": 5.0,
    "memory_bounded_planning_p95_ms": 400.0,
    "sqlite_bounded_planning_p95_ms": 500.0,
    "memory_planning_p95_ms_per_candidate": 10.0,
    "sqlite_planning_p95_ms_per_candidate": 12.0,
    "planner_input_bytes_per_candidate": 8_192,
    "plan_bytes_per_candidate": 4_096,
    "evaluator_input_bytes_per_candidate": 16_384,
}


def _latency_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = max(0, (95 * len(ordered) + 99) // 100 - 1)
    return {
        "p50_ms": round(median(ordered), 6),
        "p95_ms": round(ordered[p95_index], 6),
    }


def _canonical_bytes(value: Any, field_name: str) -> int:
    return len(canonical_durable_json_bytes(value.model_dump(mode="json"), field_name))


def _entries(count: int) -> list[KnowledgeEntry]:
    return [
        KnowledgeEntry(
            id=f"maintenance-planning-performance-{index:03d}",
            text=(
                "A bounded reviewed operational fact used to measure planning overhead. "
                f"Candidate ordinal {index:03d}."
            ),
            namespace="performance:planning",
            labels={"workload": "planning"},
            visibility=KnowledgeVisibility.PROJECT,
            created_at=_NOW,
            updated_at=_NOW,
            source_type="fixture",
            source_id=f"planning-source-{index:03d}",
            source_hash=f"planning-source-hash-{index:03d}",
        )
        for index in range(count)
    ]


def _signals(count: int) -> tuple[KnowledgeMaintenanceCandidateSignal, ...]:
    return tuple(
        KnowledgeMaintenanceCandidateSignal(
            id=f"maintenance-planning-signal-{index:03d}",
            kind=KnowledgeMaintenanceSignalKind.EXACT_REFERENCE,
            references=(
                KnowledgeRevisionRef(
                    entry_id=f"maintenance-planning-performance-{index:03d}",
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
        policy_id="hermetic-reviewed-planning-v1",
        namespace="performance:planning",
        labels={"workload": "planning"},
        access_scope=scope,
        signals=signals,
        created_at=_NOW,
    )


class _CountingPlanningStore:
    def __init__(self, store: Any) -> None:
        self._store = store
        self.reads = 0

    async def get_entry(self, *args: Any, **kwargs: Any) -> KnowledgeEntry | None:
        self.reads += 1
        return await self._store.get_entry(*args, **kwargs)


class _HermeticPlanner:
    def __init__(self) -> None:
        self.calls = 0
        self.last_input = None
        self.last_output = None

    async def propose_maintenance(self, request):
        self.calls += 1
        self.last_input = request
        sources = tuple(candidate.reference for candidate in request.snapshot.candidates)
        evidence = tuple(
            KnowledgeMaintenanceEvidenceMapping(
                id=f"claim:{index:03d}",
                claim=f"The replacement retains reviewed source {index:03d}.",
                source_references=(reference,),
            )
            for index, reference in enumerate(sources)
        )
        relations = tuple(
            KnowledgeMaintenanceRelationDraft(
                id=f"relation:{index:03d}",
                subject=KnowledgeMaintenancePlanEndpoint(
                    kind=KnowledgeMaintenancePlanEndpointKind.REPLACEMENT
                ),
                object=KnowledgeMaintenancePlanEndpoint(
                    kind=KnowledgeMaintenancePlanEndpointKind.SOURCE,
                    reference=reference,
                ),
                kind=KnowledgeRelationKind.DERIVED_FROM,
                evidence_mapping_ids=(evidence[index].id,),
            )
            for index, reference in enumerate(sources)
        )
        self.last_output = KnowledgeMaintenancePlannerOutput(
            plan=KnowledgeMaintenancePlanDraft(
                id="hermetic-planning-performance-plan",
                routing_request_fingerprint=request.snapshot.routing_request_fingerprint,
                routing_result_fingerprint=request.snapshot.routing_result_fingerprint,
                configuration_fingerprint=request.configuration_fingerprint,
                policy_id=request.snapshot.policy_id,
                source_references=sources,
                replacement=KnowledgeMaintenanceReplacementDraft(
                    text="A bounded consolidation retaining every reviewed fixture fact.",
                    title="Hermetic planning fixture",
                    kind="fact",
                    aspects=("performance",),
                ),
                relations=relations,
                evidence_mappings=evidence,
                rationale="Every routed fixture is represented exactly once.",
                evidence_summary="Every claim maps to its exact routed revision.",
            )
        )
        return self.last_output


class _HermeticEvaluator:
    def __init__(self) -> None:
        self.calls = 0
        self.last_input = None

    async def evaluate_maintenance_plan(self, request):
        self.calls += 1
        self.last_input = request
        return KnowledgeMaintenanceEvaluatorOutput(
            decision=KnowledgeMaintenanceEvaluatorDecision(
                plan_fingerprint=request.plan.fingerprint,
                routing_result_fingerprint=(
                    request.planner_input.snapshot.routing_result_fingerprint
                ),
                configuration_fingerprint=request.planner_input.configuration_fingerprint,
                verdict=KnowledgeMaintenanceEvaluationVerdict.ACCEPTED,
            )
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
        "performance:planning",
        required_labels={"workload": "planning"},
        allowed_visibilities=[KnowledgeVisibility.PROJECT],
    )
    store = (
        InMemoryKnowledgeStore(access_scope=scope)
        if backend == "memory"
        else SQLiteKnowledgeStore(directory / "planning.sqlite", access_scope=scope)
    )
    try:
        entries = _entries(candidate_count)
        for entry in entries:
            await store.create_entry(entry)
        router = KnowledgeMaintenanceRouter(
            store,
            config=KnowledgeMaintenanceRouterConfig(
                max_signals=candidate_count,
                max_candidate_reads=candidate_count,
                max_candidates=candidate_count,
                max_candidate_bytes=1024 * 1024,
                max_concurrency=min(candidate_count, 8),
            ),
        )
        zero_request = _request("planning-performance-zero", (), scope)
        zero_routing = await router.route(zero_request)
        populated_request = _request(
            "planning-performance-populated",
            _signals(candidate_count),
            scope,
        )
        populated_routing = await router.route(populated_request)
        if len(populated_routing.candidates) != candidate_count:
            raise RuntimeError("Planning performance routing lost candidates.")

        counted_store = _CountingPlanningStore(store)
        planner = _HermeticPlanner()
        evaluator = _HermeticEvaluator()
        workflow = KnowledgeMaintenancePlanningWorkflow(
            counted_store,
            planner=planner,
            evaluator=evaluator,
            config=KnowledgeMaintenancePlanningConfig(
                planner_id="hermetic-performance-planner",
                planner_version="1",
                evaluator_id="hermetic-performance-evaluator",
                evaluator_version="1",
                max_planner_model_calls=0,
                max_evaluator_model_calls=0,
                max_planner_cost_micro_usd=0,
                max_evaluator_cost_micro_usd=0,
                max_total_cost_micro_usd=0,
            ),
            clock=lambda: _NOW,
        )
        for _ in range(_WARMUP_ITERATIONS):
            await workflow.plan(zero_request, zero_routing)
            await workflow.plan(populated_request, populated_routing)

        counted_store.reads = 0
        planner.calls = 0
        evaluator.calls = 0
        zero_latencies: list[float] = []
        populated_latencies: list[float] = []
        last_result = None
        for _ in range(iterations):
            before_reads = counted_store.reads
            before_planner_calls = planner.calls
            before_evaluator_calls = evaluator.calls
            started = time.perf_counter_ns()
            zero_result = await workflow.plan(zero_request, zero_routing)
            zero_latencies.append((time.perf_counter_ns() - started) / 1_000_000)
            if zero_result.outcome is not KnowledgeMaintenancePlanningOutcome.NO_CANDIDATES:
                raise RuntimeError("Zero-candidate planning returned the wrong outcome.")
            if (
                counted_store.reads != before_reads
                or planner.calls != before_planner_calls
                or evaluator.calls != before_evaluator_calls
            ):
                raise RuntimeError("Zero-candidate planning performed component work.")

            started = time.perf_counter_ns()
            last_result = await workflow.plan(populated_request, populated_routing)
            populated_latencies.append((time.perf_counter_ns() - started) / 1_000_000)

        if last_result is None or last_result.outcome is not (
            KnowledgeMaintenancePlanningOutcome.ACCEPTED
        ):
            raise RuntimeError("Bounded planning performance workload was not accepted.")
        expected_reads = iterations * candidate_count * 3
        if counted_store.reads != expected_reads:
            raise RuntimeError("Planning currentness checks did not stay exact and bounded.")
        if planner.calls != iterations or evaluator.calls != iterations:
            raise RuntimeError("Planning components did not run exactly once per attempt.")
        if last_result.planner_usage is None or last_result.evaluator_usage is None:
            raise RuntimeError("Planning performance result omitted usage accounting.")
        if last_result.planner_usage.model_calls or last_result.evaluator_usage.model_calls:
            raise RuntimeError("Hermetic planning performance made a model call.")
        if planner.last_input is None or planner.last_output is None:
            raise RuntimeError("Hermetic planner did not retain its measured boundary.")
        if evaluator.last_input is None:
            raise RuntimeError("Hermetic evaluator did not retain its measured boundary.")

        revisions_after: list[int] = []
        for entry in entries:
            current = await store.get_entry(entry.id)
            if current is None:
                raise RuntimeError("Read-only planning removed canonical knowledge.")
            revisions_after.append(current.revision)
        if revisions_after != [1] * candidate_count:
            raise RuntimeError("Read-only planning changed canonical knowledge revisions.")

        planner_input_bytes = _canonical_bytes(planner.last_input, "planner input")
        plan_bytes = _canonical_bytes(planner.last_output.plan, "plan")
        evaluator_input_bytes = _canonical_bytes(evaluator.last_input, "evaluator input")
        per_candidate = [latency / candidate_count for latency in populated_latencies]
        return {
            "backend": backend,
            "candidate_count": candidate_count,
            "iteration_count": iterations,
            "zero_candidate_planning_latency": _latency_summary(zero_latencies),
            "planning_latency": _latency_summary(populated_latencies),
            "planning_latency_per_candidate": _latency_summary(per_candidate),
            "planner_input_bytes": planner_input_bytes,
            "planner_input_bytes_per_candidate": round(
                planner_input_bytes / candidate_count,
                6,
            ),
            "plan_bytes": plan_bytes,
            "plan_bytes_per_candidate": round(plan_bytes / candidate_count, 6),
            "evaluator_input_bytes": evaluator_input_bytes,
            "evaluator_input_bytes_per_candidate": round(
                evaluator_input_bytes / candidate_count,
                6,
            ),
            "source_revalidation_reads_per_attempt": candidate_count * 3,
            "planner_calls_per_attempt": 1,
            "evaluator_calls_per_attempt": 1,
            "model_calls": 0,
            "provider_calls": 0,
            "cost_micro_usd": 0,
            "knowledge_revision_mutations": 0,
        }
    finally:
        await _close(store)


def _ceiling_findings(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for result in results:
        backend = result["backend"]
        checks = {
            f"{backend}_zero_candidate_planning_p95_ms": result["zero_candidate_planning_latency"][
                "p95_ms"
            ],
            f"{backend}_bounded_planning_p95_ms": result["planning_latency"]["p95_ms"],
            f"{backend}_planning_p95_ms_per_candidate": result["planning_latency_per_candidate"][
                "p95_ms"
            ],
            "planner_input_bytes_per_candidate": result["planner_input_bytes_per_candidate"],
            "plan_bytes_per_candidate": result["plan_bytes_per_candidate"],
            "evaluator_input_bytes_per_candidate": result["evaluator_input_bytes_per_candidate"],
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
    with tempfile.TemporaryDirectory(prefix="cayu-maintenance-planning-performance-") as raw:
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
            "planner_calls_per_attempt": 1,
            "evaluator_calls_per_attempt": 1,
            "source_revalidations_per_attempt": 3,
            "provider_calls": 0,
            "model_calls": 0,
            "cost_micro_usd": 0,
            "store_writes_during_planning": 0,
        },
        "control": {
            "kind": "current_runtime_zero_candidates",
            "historical_pre_feature": False,
            "candidate_count": 0,
            "store_reads": 0,
            "planner_calls": 0,
            "evaluator_calls": 0,
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
