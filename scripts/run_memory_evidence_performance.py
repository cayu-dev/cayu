#!/usr/bin/env python3
"""Measure memory-evidence preparation, persistence, storage, and projection overhead."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import platform
import sys
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any

from pydantic import SecretStr

from cayu import (
    CayuApp,
    ContextExposure,
    ContextExposureEvidenceKind,
    ContextExposureState,
    ContextExposureTransition,
    InMemorySessionStore,
    KeyedEvidenceFingerprint,
    KeyedEvidenceFingerprintDomain,
    KnowledgeEntryEvidenceLocator,
    Message,
    RecallItemAdmission,
    RecallItemExposure,
    RecallItemSelectionReason,
    RecallReceipt,
    RecallReceiptItem,
    RecallSourceCoverage,
    RecallSourceCoverageState,
    RequestFootprintConfig,
    RetrievalCandidateIdentity,
    RunRequest,
    RuntimeEvidenceRequest,
    SessionIdentity,
    SQLiteSessionStore,
    runtime_evidence,
)
from cayu.runtime.sessions import SessionStore

_SCHEMA_VERSION = "cayu.memory_evidence_performance.v1"
_DEFAULT_SAMPLES = 50
_DEFAULT_PROJECTION_ITERATIONS = 30
_SESSION_ID = "memory-evidence-performance"
_KEY_MATERIAL = "public-hermetic-performance-key-material-v1"
_KEY_CONFIG = RequestFootprintConfig(
    fingerprint_key_id="performance-v1",
    fingerprint_key=SecretStr(_KEY_MATERIAL),
)

_CEILINGS = {
    "preparation_p95_ms": 10.0,
    "memory_persistence_p95_ms": 15.0,
    "sqlite_persistence_p95_ms": 25.0,
    "memory_zero_record_runtime_evidence_p95_ms": 5.0,
    "sqlite_zero_record_runtime_evidence_p95_ms": 10.0,
    "projection_overhead_p95_ms_per_pair": 10.0,
    "projection_bytes_per_pair": 6_000,
    "sqlite_storage_bytes_per_pair": 32_768,
}


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _fingerprint(
    label: str,
    domain: KeyedEvidenceFingerprintDomain,
) -> KeyedEvidenceFingerprint:
    return KeyedEvidenceFingerprint(
        domain=domain,
        key_id="performance-v1",
        digest=hmac.digest(
            _KEY_MATERIAL.encode(),
            f"{domain.value}\x00{label}".encode(),
            "sha256",
        ).hex(),
    )


def _documents(index: int) -> tuple[RecallReceipt, ContextExposure, tuple[RecallItemExposure, ...]]:
    label = f"sample-{index:05d}"
    occurred_at = datetime(2026, 8, 23, tzinfo=UTC) + timedelta(microseconds=index)
    item = RecallReceiptItem(
        ordinal=0,
        identity=RetrievalCandidateIdentity(
            record_type="knowledge_entry",
            record_id=f"entry-{label}",
            revision="1",
        ),
        representation_id="entry_text",
        content_sha256=_digest(f"content-{label}"),
        locator=KnowledgeEntryEvidenceLocator(
            entry_id=f"entry-{label}",
            entry_revision=1,
        ),
        admission=RecallItemAdmission.ADMITTED,
        selection_reason=RecallItemSelectionReason.CALIBRATED_STRONG_MATCH,
        fused_rank=1,
        match_channels=("knowledge.lexical",),
    )
    receipt = RecallReceipt(
        receipt_id=f"receipt-{label}",
        session_id=_SESSION_ID,
        interaction_id=f"interaction-{label}",
        model_step_id=f"mstep_{_digest(f'step-{label}')[:32]}",
        created_at=occurred_at,
        situation_fingerprint=_fingerprint(
            f"situation-{label}",
            KeyedEvidenceFingerprintDomain.SITUATION,
        ),
        engine_version="cayu.recall.v1",
        source_configuration_fingerprint=_fingerprint(
            f"sources-{label}",
            KeyedEvidenceFingerprintDomain.SOURCE_CONFIGURATION,
        ),
        admission_policy_fingerprint=_fingerprint(
            f"policy-{label}",
            KeyedEvidenceFingerprintDomain.ADMISSION_POLICY,
        ),
        access_scope_fingerprint=_fingerprint(
            f"scope-{label}",
            KeyedEvidenceFingerprintDomain.ACCESS_SCOPE,
        ),
        frontier_fingerprint=_fingerprint(
            f"frontier-{label}",
            KeyedEvidenceFingerprintDomain.FRONTIER,
        ),
        sources=(
            RecallSourceCoverage(
                source="knowledge",
                required=True,
                channels=("knowledge.lexical",),
                state=RecallSourceCoverageState.COMPLETE,
                inspected_count=1,
                candidate_limit=10,
            ),
        ),
        inspected_count=1,
        eligible_count=1,
        admitted_count=1,
        offered_count=0,
        silent_count=0,
        omitted_count=0,
        truncated=False,
        items=(item,),
    )
    transition = ContextExposureTransition(
        transition_id=f"transition-{label}",
        revision=0,
        state=ContextExposureState.PLANNED,
        occurred_at=occurred_at,
        evidence_kind=ContextExposureEvidenceKind.COMPOSITION_PLANNED,
        evidence_ref=f"composition-{label}",
    )
    exposure = ContextExposure(
        exposure_id=f"exposure-{label}",
        session_id=_SESSION_ID,
        interaction_id=receipt.interaction_id,
        model_step_id=receipt.model_step_id,
        model_attempt_id=f"matt_{_digest(f'model-{label}')[:32]}",
        provider_attempt_id=f"patt_{_digest(f'provider-{label}')[:32]}",
        provider_name="hermetic",
        model_name="hermetic-model",
        composition_fingerprint=_fingerprint(
            f"composition-{label}",
            KeyedEvidenceFingerprintDomain.COMPOSITION,
        ),
        execution_profile_fingerprint=_fingerprint(
            f"profile-{label}",
            KeyedEvidenceFingerprintDomain.EXECUTION_PROFILE,
        ),
        context_policy_fingerprint=_fingerprint(
            f"context-{label}",
            KeyedEvidenceFingerprintDomain.CONTEXT_POLICY,
        ),
        tool_exposure_fingerprint=_fingerprint(
            f"tools-{label}",
            KeyedEvidenceFingerprintDomain.TOOL_EXPOSURE,
        ),
        request_contract_fingerprint=_fingerprint(
            f"request-{label}",
            KeyedEvidenceFingerprintDomain.REQUEST_CONTRACT,
        ),
        receipt_ids=(receipt.receipt_id,),
        contributor_ids=("automatic_recall",),
        created_at=occurred_at,
        updated_at=occurred_at,
        state=ContextExposureState.PLANNED,
        state_revision=0,
        transitions=(transition,),
    )
    exposure_item = RecallItemExposure(
        exposure_id=exposure.exposure_id,
        receipt_id=receipt.receipt_id,
        ordinal=0,
        receipt_item_ordinal=0,
        identity=item.identity,
        representation_id=item.representation_id,
        content_sha256=item.content_sha256,
        locator=item.locator,
        admission=item.admission,
        selection_reason=item.selection_reason,
    )
    return receipt, exposure, (exposure_item,)


async def _create_session(store: SessionStore) -> None:
    await store.create(
        RunRequest(
            agent_name="performance",
            session_id=_SESSION_ID,
            messages=[Message.text("user", "hermetic benchmark")],
        ),
        identity=SessionIdentity(provider_name="hermetic", model="hermetic-model"),
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


async def _projection_samples(
    zero_record_store: SessionStore,
    populated_store: SessionStore,
    *,
    iterations: int,
) -> tuple[list[float], list[float], int, int]:
    request = RuntimeEvidenceRequest(
        root_session_id=_SESSION_ID,
        max_sessions=1,
        max_events=1,
    )
    zero_record_app = CayuApp(
        session_store=zero_record_store,
        request_footprint=_KEY_CONFIG,
        enable_logging=False,
    )
    populated_app = CayuApp(
        session_store=populated_store,
        request_footprint=_KEY_CONFIG,
        enable_logging=False,
    )
    zero_record_latencies: list[float] = []
    populated_latencies: list[float] = []
    zero_record_report = await runtime_evidence(zero_record_app, request)
    populated_report = await runtime_evidence(populated_app, request)
    for _ in range(iterations):
        started = time.perf_counter_ns()
        await runtime_evidence(zero_record_app, request)
        zero_record_latencies.append((time.perf_counter_ns() - started) / 1_000_000)

        started = time.perf_counter_ns()
        await runtime_evidence(populated_app, request)
        populated_latencies.append((time.perf_counter_ns() - started) / 1_000_000)
    return (
        zero_record_latencies,
        populated_latencies,
        len(zero_record_report.model_dump_json().encode()),
        len(populated_report.model_dump_json().encode()),
    )


async def _backend_result(
    backend: str,
    *,
    samples: int,
    projection_iterations: int,
    directory: Path,
) -> dict[str, Any]:
    zero_record_path = directory / f"{backend}-zero-record.sqlite"
    populated_path = directory / f"{backend}-populated.sqlite"
    if backend == "memory":
        zero_record_store: SessionStore = InMemorySessionStore()
        populated_store: SessionStore = InMemorySessionStore()
    else:
        zero_record_store = SQLiteSessionStore(zero_record_path)
        populated_store = SQLiteSessionStore(populated_path)
    await _create_session(zero_record_store)
    await _create_session(populated_store)
    preparation_latencies: list[float] = []
    documents: list[tuple[RecallReceipt, ContextExposure, tuple[RecallItemExposure, ...]]] = []
    for index in range(samples):
        started = time.perf_counter_ns()
        prepared = _documents(index)
        preparation_latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        documents.append(prepared)

    persistence_latencies: list[float] = []
    for receipt, exposure, items in documents:
        started = time.perf_counter_ns()
        await populated_store.create_recall_receipt(receipt)
        await populated_store.create_context_exposure(exposure, items)
        persistence_latencies.append((time.perf_counter_ns() - started) / 1_000_000)

    (
        zero_record_projection,
        populated_projection,
        zero_record_bytes,
        populated_bytes,
    ) = await _projection_samples(
        zero_record_store,
        populated_store,
        iterations=projection_iterations,
    )
    close_zero_record = getattr(zero_record_store, "close", None)
    close_populated = getattr(populated_store, "close", None)
    if close_zero_record is not None:
        await close_zero_record()
    if close_populated is not None:
        await close_populated()
    zero_record_storage_bytes = _storage_bytes(zero_record_path) if backend == "sqlite" else None
    storage_bytes = _storage_bytes(populated_path) if backend == "sqlite" else None

    zero_record_summary = _latency_summary(zero_record_projection)
    populated_summary = _latency_summary(populated_projection)
    overhead = [
        max(populated - zero_record, 0.0)
        for zero_record, populated in zip(
            zero_record_projection,
            populated_projection,
            strict=True,
        )
    ]
    overhead_summary = _latency_summary(overhead)
    return {
        "backend": backend,
        "sample_count": samples,
        "projection_iteration_count": projection_iterations,
        "preparation_latency": _latency_summary(preparation_latencies),
        "persistence_latency": _latency_summary(persistence_latencies),
        "zero_record_runtime_evidence_latency": zero_record_summary,
        "populated_runtime_evidence_latency": populated_summary,
        "incremental_projection_latency": overhead_summary,
        "populated_p95_ratio_to_zero_record": round(
            populated_summary["p95_ms"] / max(zero_record_summary["p95_ms"], 0.000001),
            6,
        ),
        "zero_record_report_bytes": zero_record_bytes,
        "populated_report_bytes": populated_bytes,
        "incremental_projection_bytes": populated_bytes - zero_record_bytes,
        "projection_bytes_per_pair": round(
            (populated_bytes - zero_record_bytes) / samples,
            6,
        ),
        "zero_record_storage_bytes": zero_record_storage_bytes,
        "populated_storage_bytes": storage_bytes,
        "storage_byte_overhead": (
            None
            if storage_bytes is None or zero_record_storage_bytes is None
            else storage_bytes - zero_record_storage_bytes
        ),
        "storage_bytes_per_pair": (
            None
            if storage_bytes is None or zero_record_storage_bytes is None
            else round((storage_bytes - zero_record_storage_bytes) / samples, 6)
        ),
    }


def _ceiling_findings(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for result in results:
        backend = result["backend"]
        checks = {
            "preparation_p95_ms": result["preparation_latency"]["p95_ms"],
            f"{backend}_persistence_p95_ms": result["persistence_latency"]["p95_ms"],
            f"{backend}_zero_record_runtime_evidence_p95_ms": result[
                "zero_record_runtime_evidence_latency"
            ]["p95_ms"],
            "projection_overhead_p95_ms_per_pair": (
                result["incremental_projection_latency"]["p95_ms"] / result["sample_count"]
            ),
            "projection_bytes_per_pair": result["projection_bytes_per_pair"],
        }
        if backend == "sqlite":
            checks["sqlite_storage_bytes_per_pair"] = result["storage_bytes_per_pair"]
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


async def _run(samples: int, projection_iterations: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="cayu-memory-evidence-performance-") as raw:
        directory = Path(raw)
        results = [
            await _backend_result(
                backend,
                samples=samples,
                projection_iterations=projection_iterations,
                directory=directory,
            )
            for backend in ("memory", "sqlite")
        ]
    findings = _ceiling_findings(results)
    return {
        "schema_version": _SCHEMA_VERSION,
        "workload": {
            "receipt_exposure_pairs": samples,
            "items_per_pair": 1,
            "provider_calls": 0,
        },
        "control": {
            "kind": "current_runtime_zero_memory_records",
            "receipt_exposure_pairs": 0,
            "historical_pre_feature": False,
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
    parser.add_argument("--samples", type=int, default=_DEFAULT_SAMPLES)
    parser.add_argument(
        "--projection-iterations",
        type=int,
        default=_DEFAULT_PROJECTION_ITERATIONS,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.samples <= 100:
        parser.error("--samples must be between 1 and 100")
    if not 1 <= args.projection_iterations <= 1_000:
        parser.error("--projection-iterations must be between 1 and 1000")

    report = asyncio.run(_run(args.samples, args.projection_iterations))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 1 if args.check and not report["within_ceilings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
