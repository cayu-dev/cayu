#!/usr/bin/env python3
"""Measure hermetic evaluated-maintenance governance overhead."""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import runpy
import sys
import tempfile
import time
from pathlib import Path
from statistics import median
from typing import Any

from cayu import (
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeGovernanceConfig,
    KnowledgeGovernanceMode,
    KnowledgeMaintenanceGovernanceDecision,
    KnowledgeMaintenanceGovernanceDisposition,
    KnowledgeMaintenanceGovernor,
    KnowledgeStatus,
    KnowledgeVisibility,
    SQLiteKnowledgeStore,
)

_SCHEMA_VERSION = "cayu.knowledge_maintenance_governance_performance.v1"
_DEFAULT_OPERATIONS_PER_MODE = 5
_SCOPE = KnowledgeAccessScope.for_namespace(
    "example:maintenance",
    allowed_visibilities=[KnowledgeVisibility.PROJECT],
    allowed_statuses=[KnowledgeStatus.ACTIVE, KnowledgeStatus.PENDING],
)
_CEILINGS = {
    "memory_reviewed_route_p95_ms": 50.0,
    "sqlite_reviewed_route_p95_ms": 100.0,
    "memory_automatic_reject_p95_ms": 50.0,
    "sqlite_automatic_reject_p95_ms": 100.0,
    "memory_receipt_replay_p95_ms": 50.0,
    "sqlite_receipt_replay_p95_ms": 100.0,
    "sqlite_storage_bytes_per_governance_outcome": 131_072,
}
_EXAMPLE = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "examples/knowledge_maintenance_governance.py")
)
publish_evaluated_proposal = _EXAMPLE["publish_evaluated_proposal"]


class _RejectPolicy:
    def __init__(self) -> None:
        self.calls = 0

    async def decide_maintenance(self, request):
        self.calls += 1
        return KnowledgeMaintenanceGovernanceDecision(
            request_sha256=request.fingerprint,
            disposition=KnowledgeMaintenanceGovernanceDisposition.REJECT,
            policy_identity="performance.application-maintenance-policy",
            policy_version="1",
            code="performance_reject",
        )


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


async def _close(store: Any) -> None:
    close = getattr(store, "close", None)
    if close is not None:
        await close()


async def _backend_result(
    backend: str,
    *,
    operations_per_mode: int,
    directory: Path,
) -> dict[str, Any]:
    control_path = directory / f"{backend}-control.sqlite"
    governed_path = directory / f"{backend}-governed.sqlite"
    if backend == "memory":
        control_store = InMemoryKnowledgeStore()
        governed_store = InMemoryKnowledgeStore()
    else:
        control_store = SQLiteKnowledgeStore(control_path)
        governed_store = SQLiteKnowledgeStore(governed_path)

    operation_count = operations_per_mode * 2
    control_publications = []
    governed_publications = []
    for index in range(operation_count):
        prefix = f"performance-{index:04d}"
        control_publications.append(await publish_evaluated_proposal(control_store, prefix))
        governed_publications.append(await publish_evaluated_proposal(governed_store, prefix))

    reviewed = KnowledgeMaintenanceGovernor(
        governed_store,
        config=KnowledgeGovernanceConfig(mode=KnowledgeGovernanceMode.REVIEWED),
    )
    policy = _RejectPolicy()
    automatic = KnowledgeMaintenanceGovernor(
        governed_store,
        config=KnowledgeGovernanceConfig(
            mode=KnowledgeGovernanceMode.AUTONOMOUS,
            policy_identity="performance.application-maintenance-policy",
            policy_version="1",
        ),
        policy=policy,
    )
    reviewed_latencies: list[float] = []
    automatic_latencies: list[float] = []
    replay_latencies: list[float] = []
    governed_results = []
    for index, publication in enumerate(governed_publications):
        governor = reviewed if index < operations_per_mode else automatic
        operation_id = f"performance-governance-{index:04d}"
        started = time.perf_counter_ns()
        receipt = await governor.govern(
            operation_id=operation_id,
            proposal_id=publication.proposal.id,
            access_scope=_SCOPE,
        )
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        if index < operations_per_mode:
            reviewed_latencies.append(elapsed_ms)
        else:
            automatic_latencies.append(elapsed_ms)
        governed_results.append((governor, publication, receipt))

    policy_calls_before_replay = policy.calls
    for governor, publication, receipt in governed_results:
        started = time.perf_counter_ns()
        replay = await governor.govern(
            operation_id=receipt.operation_id,
            proposal_id=publication.proposal.id,
            access_scope=_SCOPE,
        )
        replay_latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        if not replay.replayed:
            raise RuntimeError("Governance performance replay was not identified.")
    if policy.calls != policy_calls_before_replay:
        raise RuntimeError("Governance performance replay called the policy again.")

    await _close(control_store)
    await _close(governed_store)
    control_bytes = _storage_bytes(control_path) if backend == "sqlite" else None
    governed_bytes = _storage_bytes(governed_path) if backend == "sqlite" else None
    storage_overhead = (
        None if control_bytes is None or governed_bytes is None else governed_bytes - control_bytes
    )
    return {
        "backend": backend,
        "operations_per_mode": operations_per_mode,
        "operation_count": operation_count,
        "reviewed_route_latency": _latency_summary(reviewed_latencies),
        "automatic_reject_latency": _latency_summary(automatic_latencies),
        "receipt_replay_latency": _latency_summary(replay_latencies),
        "control_storage_bytes": control_bytes,
        "governed_storage_bytes": governed_bytes,
        "storage_byte_overhead": storage_overhead,
        "storage_bytes_per_governance_outcome": (
            None if storage_overhead is None else round(storage_overhead / operation_count, 6)
        ),
    }


def _ceiling_findings(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for result in results:
        backend = result["backend"]
        checks = {
            f"{backend}_reviewed_route_p95_ms": result["reviewed_route_latency"]["p95_ms"],
            f"{backend}_automatic_reject_p95_ms": result["automatic_reject_latency"]["p95_ms"],
            f"{backend}_receipt_replay_p95_ms": result["receipt_replay_latency"]["p95_ms"],
        }
        if backend == "sqlite":
            checks["sqlite_storage_bytes_per_governance_outcome"] = result[
                "storage_bytes_per_governance_outcome"
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


async def _run(operations_per_mode: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(
        prefix="cayu-knowledge-maintenance-governance-performance-"
    ) as raw:
        directory = Path(raw)
        results = [
            await _backend_result(
                backend,
                operations_per_mode=operations_per_mode,
                directory=directory,
            )
            for backend in ("memory", "sqlite")
        ]
    findings = _ceiling_findings(results)
    operation_count = operations_per_mode * 2
    return {
        "schema_version": _SCHEMA_VERSION,
        "workload": {
            "operations_per_mode": operations_per_mode,
            "operation_count": operation_count,
            "reviewed_route_operations": operations_per_mode,
            "automatic_reject_operations": operations_per_mode,
            "provider_calls": 0,
        },
        "control": {
            "kind": "identical_evaluated_proposals_without_governance_outcomes",
            "historical_pre_feature": False,
            "operation_count": operation_count,
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
    parser.add_argument(
        "--operations-per-mode",
        type=int,
        default=_DEFAULT_OPERATIONS_PER_MODE,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.operations_per_mode <= 50:
        parser.error("--operations-per-mode must be between 1 and 50")

    report = asyncio.run(_run(args.operations_per_mode))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 1 if args.check and not report["within_ceilings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
