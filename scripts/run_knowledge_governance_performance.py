#!/usr/bin/env python3
"""Measure hermetic knowledge-governance latency and storage overhead."""

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
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeActivationDecision,
    KnowledgeActivationDisposition,
    KnowledgeActivationRequest,
    KnowledgeActivationSource,
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeGovernanceConfig,
    KnowledgeGovernanceMode,
    KnowledgeStatus,
    SQLiteKnowledgeStore,
    decide_knowledge_activation,
    prepare_knowledge_activation_request,
)

_SCHEMA_VERSION = "cayu.knowledge_governance_performance.v1"
_DEFAULT_OPERATIONS_PER_MODE = 20
_NOW = datetime(2026, 8, 30, 12, tzinfo=UTC)
_SCOPE = KnowledgeAccessScope.privileged()
_AUTOMATIC_CONFIG = KnowledgeGovernanceConfig(
    mode=KnowledgeGovernanceMode.POLICY_AUTOMATIC,
    policy_identity="performance.activation-policy",
    policy_version="1",
)
_REVIEWED_CONFIG = KnowledgeGovernanceConfig()

_CEILINGS = {
    "memory_reviewed_routing_p95_ms": 25.0,
    "sqlite_reviewed_routing_p95_ms": 100.0,
    "memory_automatic_activation_p95_ms": 25.0,
    "sqlite_automatic_activation_p95_ms": 100.0,
    "memory_receipt_lookup_p95_ms": 25.0,
    "sqlite_receipt_lookup_p95_ms": 50.0,
    "sqlite_storage_bytes_per_activation_receipt": 65_536,
}


class _ActivatePolicy:
    async def decide_activation(
        self,
        request: KnowledgeActivationRequest,
    ) -> KnowledgeActivationDecision:
        return KnowledgeActivationDecision(
            request_sha256=request.fingerprint,
            disposition=KnowledgeActivationDisposition.ACTIVATE,
            policy_identity="performance.activation-policy",
            policy_version="1",
            code="performance_activate",
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


def _material(index: int) -> tuple[KnowledgeEntry, list[KnowledgeChunk]]:
    entry_id = f"governance-performance-{index:04d}"
    timestamp = _NOW + timedelta(seconds=index)
    entry = KnowledgeEntry(
        id=entry_id,
        text=f"Governed knowledge performance candidate {index:04d}.",
        status=KnowledgeStatus.PENDING,
        created_at=timestamp,
        updated_at=timestamp,
        source_type="performance",
        source_id=entry_id,
        source_hash=f"performance-source-{index:04d}",
    )
    return entry, [
        KnowledgeChunk(
            id=f"{entry_id}:r1:0",
            entry_id=entry_id,
            text=entry.text,
            chunk_index=0,
        )
    ]


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
        control_store = InMemoryKnowledgeStore(access_scope=_SCOPE)
        governed_store = InMemoryKnowledgeStore(access_scope=_SCOPE)
    else:
        control_store = SQLiteKnowledgeStore(control_path, access_scope=_SCOPE)
        governed_store = SQLiteKnowledgeStore(governed_path, access_scope=_SCOPE)

    reviewed_latencies: list[float] = []
    automatic_latencies: list[float] = []
    receipt_latencies: list[float] = []
    policy = _ActivatePolicy()
    operation_count = operations_per_mode * 2
    for index in range(operation_count):
        entry, chunks = _material(index)
        automatic = index >= operations_per_mode
        config = _AUTOMATIC_CONFIG if automatic else _REVIEWED_CONFIG
        source = (
            KnowledgeActivationSource.MODEL_TOOL if automatic else KnowledgeActivationSource.CURATOR
        )
        operation_id = f"governance-performance-operation-{index:04d}"
        started = time.perf_counter_ns()
        request = prepare_knowledge_activation_request(
            entry,
            chunks,
            access_scope=_SCOPE,
            operation_id=operation_id,
            governance_mode=config.mode,
            source=source,
            forbidden_authority_identities=("performance.model",),
        )
        authority = await decide_knowledge_activation(
            request,
            config=config,
            policy=policy if automatic else None,
        )
        published_entry = entry.model_copy(
            update={"status": (KnowledgeStatus.ACTIVE if automatic else KnowledgeStatus.PENDING)}
        )
        await governed_store.publish_entry_revision(
            published_entry,
            chunks,
            operation_id=operation_id,
            activation_authority=authority,
        )
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        (automatic_latencies if automatic else reviewed_latencies).append(elapsed_ms)

        await control_store.publish_entry_revision(
            published_entry,
            chunks,
            operation_id=f"control-{operation_id}",
        )
        started = time.perf_counter_ns()
        receipt = await governed_store.load_activation_receipt(operation_id)
        receipt_latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        if receipt is None or receipt.operation_id != operation_id:
            raise RuntimeError("Governance performance receipt was unavailable.")

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
        "reviewed_routing_latency": _latency_summary(reviewed_latencies),
        "automatic_activation_latency": _latency_summary(automatic_latencies),
        "receipt_lookup_latency": _latency_summary(receipt_latencies),
        "control_storage_bytes": control_bytes,
        "governed_storage_bytes": governed_bytes,
        "storage_byte_overhead": storage_overhead,
        "storage_bytes_per_activation_receipt": (
            None if storage_overhead is None else round(storage_overhead / operation_count, 6)
        ),
    }


def _ceiling_findings(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for result in results:
        backend = result["backend"]
        checks = {
            f"{backend}_reviewed_routing_p95_ms": result["reviewed_routing_latency"]["p95_ms"],
            f"{backend}_automatic_activation_p95_ms": result["automatic_activation_latency"][
                "p95_ms"
            ],
            f"{backend}_receipt_lookup_p95_ms": result["receipt_lookup_latency"]["p95_ms"],
        }
        if backend == "sqlite":
            checks["sqlite_storage_bytes_per_activation_receipt"] = result[
                "storage_bytes_per_activation_receipt"
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
    with tempfile.TemporaryDirectory(prefix="cayu-knowledge-governance-performance-") as raw:
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
            "reviewed_operations": operations_per_mode,
            "automatic_operations": operations_per_mode,
            "provider_calls": 0,
        },
        "control": {
            "kind": "current_runtime_zero_activation_receipts",
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
    if not 1 <= args.operations_per_mode <= 100:
        parser.error("--operations-per-mode must be between 1 and 100")

    report = asyncio.run(_run(args.operations_per_mode))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 1 if args.check and not report["within_ceilings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
