#!/usr/bin/env python3
"""Measure hermetic reviewed knowledge-maintenance storage overhead."""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any

from cayu import (
    MAX_KNOWLEDGE_MAINTENANCE_SOURCES,
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeActorType,
    KnowledgeEntry,
    KnowledgeMaintenanceDecision,
    KnowledgeMaintenanceDecisionKind,
    KnowledgeMaintenanceProposal,
    KnowledgeRelation,
    KnowledgeRelationKind,
    KnowledgeRevisionRef,
    KnowledgeStatus,
    SQLiteKnowledgeStore,
    prepare_knowledge_maintenance_decision,
)

_SCHEMA_VERSION = "cayu.knowledge_maintenance_performance.v1"
_DEFAULT_DECISIONS = 20
_DEFAULT_SOURCES = 20
_NOW = datetime(2026, 8, 25, 9, 0, tzinfo=UTC)
_SCOPE = KnowledgeAccessScope.privileged()

_CEILINGS = {
    "memory_zero_decision_entry_publish_p95_ms": 10.0,
    "sqlite_zero_decision_entry_publish_p95_ms": 100.0,
    "preparation_p95_ms_per_source": 2.0,
    "memory_application_p95_ms_per_source": 10.0,
    "sqlite_application_p95_ms_per_source": 30.0,
    "memory_replay_p95_ms": 100.0,
    "sqlite_replay_p95_ms": 200.0,
    "memory_receipt_load_p95_ms": 25.0,
    "sqlite_receipt_load_p95_ms": 50.0,
    "sqlite_storage_bytes_per_applied_source": 65_536,
}


def _latency_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = max(0, (95 * len(ordered) + 99) // 100 - 1)
    return {
        "p50_ms": round(median(ordered), 6),
        "p95_ms": round(ordered[p95_index], 6),
    }


def _storage_bytes(path: Path) -> int:
    return sum(
        candidate.stat().st_size for candidate in (path, Path(f"{path}-wal")) if candidate.exists()
    )


def _materials(
    decision_count: int,
    source_count: int,
) -> list[tuple[KnowledgeMaintenanceProposal, KnowledgeMaintenanceDecision, list[KnowledgeEntry]]]:
    materials = []
    for decision_index in range(decision_count):
        prefix = f"maintenance-performance-{decision_index:04d}"
        timestamp = _NOW + timedelta(seconds=decision_index)
        replacement = KnowledgeRevisionRef(entry_id=f"{prefix}-replacement", revision=1)
        sources = [
            KnowledgeRevisionRef(entry_id=f"{prefix}-source-{index:03d}", revision=1)
            for index in range(source_count)
        ]
        active_replacement = replacement.model_copy(update={"revision": 2})
        relations = [
            KnowledgeRelation(
                id=f"{prefix}-relation-{index:03d}",
                subject=active_replacement,
                object=source,
                kind=KnowledgeRelationKind.SUPERSEDES,
                created_by="performance-runner",
                policy_id="hermetic-maintenance-v1",
                created_at=timestamp,
                metadata={"ordinal": index},
            )
            for index, source in enumerate(sources)
        ]
        proposal = KnowledgeMaintenanceProposal(
            id=f"{prefix}-proposal",
            replacement=replacement,
            sources=sources,
            relations=relations,
            access_scope=_SCOPE,
            policy_id="hermetic-maintenance-v1",
            proposed_by="performance-runner",
            created_at=timestamp,
            rationale="The reviewed replacement preserves the canonical meaning.",
            evidence_summary="Every source is bound to one exact reviewed revision.",
            metadata={"decision_ordinal": decision_index},
        )
        decision = KnowledgeMaintenanceDecision(
            operation_id=f"{prefix}-operation",
            proposal_id=proposal.id,
            proposal_fingerprint=proposal.fingerprint,
            kind=KnowledgeMaintenanceDecisionKind.APPROVE,
            reviewer_type=KnowledgeActorType.USER,
            reviewer="performance-reviewer",
            reason="The exact bounded proposal was reviewed.",
            decided_at=timestamp + timedelta(milliseconds=1),
        )
        entries = [
            KnowledgeEntry(
                id=replacement.entry_id,
                text=f"Pending canonical replacement {decision_index:04d}.",
                status=KnowledgeStatus.PENDING,
                created_at=timestamp,
                updated_at=timestamp,
            ),
            *(
                KnowledgeEntry(
                    id=source.entry_id,
                    text=f"Reviewed predecessor {decision_index:04d}/{index:03d}.",
                    created_at=timestamp,
                    updated_at=timestamp,
                )
                for index, source in enumerate(sources)
            ),
        ]
        materials.append((proposal, decision, entries))
    return materials


async def _close(store: Any) -> None:
    close = getattr(store, "close", None)
    if close is not None:
        await close()


async def _backend_result(
    backend: str,
    *,
    decision_count: int,
    source_count: int,
    directory: Path,
) -> dict[str, Any]:
    zero_path = directory / f"{backend}-zero-decisions.sqlite"
    populated_path = directory / f"{backend}-populated-decisions.sqlite"
    if backend == "memory":
        zero_store = InMemoryKnowledgeStore(access_scope=_SCOPE)
        populated_store = InMemoryKnowledgeStore(access_scope=_SCOPE)
    else:
        zero_store = SQLiteKnowledgeStore(zero_path, access_scope=_SCOPE)
        populated_store = SQLiteKnowledgeStore(populated_path, access_scope=_SCOPE)

    materials = _materials(decision_count, source_count)
    entry_latencies: list[float] = []
    for _, _, entries in materials:
        for entry in entries:
            started = time.perf_counter_ns()
            await zero_store.create_entry(entry)
            entry_latencies.append((time.perf_counter_ns() - started) / 1_000_000)
            await populated_store.create_entry(entry)

    preparation_latencies: list[float] = []
    application_latencies: list[float] = []
    replay_latencies: list[float] = []
    receipt_load_latencies: list[float] = []
    for proposal, decision, _ in materials:
        started = time.perf_counter_ns()
        prepare_knowledge_maintenance_decision(proposal, decision)
        preparation_latencies.append((time.perf_counter_ns() - started) / 1_000_000)

        started = time.perf_counter_ns()
        await populated_store.apply_maintenance_decision(proposal, decision)
        application_latencies.append((time.perf_counter_ns() - started) / 1_000_000)

        started = time.perf_counter_ns()
        replay = await populated_store.apply_maintenance_decision(proposal, decision)
        replay_latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        if not replay.replayed:
            raise RuntimeError("Maintenance performance replay was not idempotent.")

        started = time.perf_counter_ns()
        receipt = await populated_store.load_maintenance_decision_receipt(decision.operation_id)
        receipt_load_latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        if receipt is None or receipt.operation_id != decision.operation_id:
            raise RuntimeError("Maintenance performance receipt was unavailable.")

    await _close(zero_store)
    await _close(populated_store)
    zero_storage_bytes = _storage_bytes(zero_path) if backend == "sqlite" else None
    populated_storage_bytes = _storage_bytes(populated_path) if backend == "sqlite" else None
    storage_overhead = (
        None
        if zero_storage_bytes is None or populated_storage_bytes is None
        else populated_storage_bytes - zero_storage_bytes
    )
    applied_source_count = decision_count * source_count
    return {
        "backend": backend,
        "decision_count": decision_count,
        "sources_per_decision": source_count,
        "applied_source_count": applied_source_count,
        "entry_count": decision_count * (source_count + 1),
        "zero_decision_entry_publish_latency": _latency_summary(entry_latencies),
        "preparation_latency_per_source": _latency_summary(
            [latency / source_count for latency in preparation_latencies]
        ),
        "application_latency_per_source": _latency_summary(
            [latency / source_count for latency in application_latencies]
        ),
        "exact_replay_latency": _latency_summary(replay_latencies),
        "receipt_load_latency": _latency_summary(receipt_load_latencies),
        "zero_decision_storage_bytes": zero_storage_bytes,
        "populated_decision_storage_bytes": populated_storage_bytes,
        "storage_byte_overhead": storage_overhead,
        "storage_bytes_per_applied_source": (
            None if storage_overhead is None else round(storage_overhead / applied_source_count, 6)
        ),
    }


def _ceiling_findings(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for result in results:
        backend = result["backend"]
        checks = {
            f"{backend}_zero_decision_entry_publish_p95_ms": result[
                "zero_decision_entry_publish_latency"
            ]["p95_ms"],
            "preparation_p95_ms_per_source": result["preparation_latency_per_source"]["p95_ms"],
            f"{backend}_application_p95_ms_per_source": result["application_latency_per_source"][
                "p95_ms"
            ],
            f"{backend}_replay_p95_ms": result["exact_replay_latency"]["p95_ms"],
            f"{backend}_receipt_load_p95_ms": result["receipt_load_latency"]["p95_ms"],
        }
        if backend == "sqlite":
            checks["sqlite_storage_bytes_per_applied_source"] = result[
                "storage_bytes_per_applied_source"
            ]
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


async def _run(decision_count: int, source_count: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="cayu-knowledge-maintenance-performance-") as raw:
        directory = Path(raw)
        results = [
            await _backend_result(
                backend,
                decision_count=decision_count,
                source_count=source_count,
                directory=directory,
            )
            for backend in ("memory", "sqlite")
        ]
    findings = _ceiling_findings(results)
    return {
        "schema_version": _SCHEMA_VERSION,
        "workload": {
            "decision_count": decision_count,
            "sources_per_decision": source_count,
            "applied_source_count": decision_count * source_count,
            "entry_count": decision_count * (source_count + 1),
            "provider_calls": 0,
        },
        "control": {
            "kind": "current_runtime_zero_maintenance_decisions",
            "historical_pre_feature": False,
            "decision_count": 0,
            "entry_count": decision_count * (source_count + 1),
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
    parser.add_argument("--decisions", type=int, default=_DEFAULT_DECISIONS)
    parser.add_argument("--sources", type=int, default=_DEFAULT_SOURCES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.decisions <= 100:
        parser.error("--decisions must be between 1 and 100")
    if not 1 <= args.sources <= MAX_KNOWLEDGE_MAINTENANCE_SOURCES:
        parser.error("--sources must be between 1 and MAX_KNOWLEDGE_MAINTENANCE_SOURCES")

    report = asyncio.run(_run(args.decisions, args.sources))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 1 if args.check and not report["within_ceilings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
