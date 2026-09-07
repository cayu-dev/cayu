#!/usr/bin/env python3
"""Evaluate deterministic automatic-memory re-anchoring without network calls."""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
import tempfile
import time
from pathlib import Path
from statistics import median
from typing import Any

from pydantic import SecretStr

from cayu import (
    KNOWLEDGE_LEXICAL_CHANNEL,
    KNOWLEDGE_SEMANTIC_CHANNEL,
    TRANSCRIPT_LEXICAL_CHANNEL,
    WEIGHTED_RECIPROCAL_RANK_FUSION_VERSION,
    AgentSpec,
    AutomaticRecallContextPolicy,
    AutomaticRecallPolicy,
    AutomaticRecallSourceConfig,
    CayuApp,
    ContextPolicy,
    Environment,
    EnvironmentSpec,
    EventType,
    InMemoryKnowledgeStore,
    InMemorySessionStore,
    KnowledgeAccessScope,
    KnowledgeEntry,
    MemoryDeltaPolicy,
    Message,
    MessageRole,
    ModelStreamEvent,
    RequestFootprintConfig,
    RunRequest,
    ScriptedModelProvider,
    SQLiteKnowledgeStore,
    SQLiteSessionStore,
    TextPart,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
    WeightedReciprocalRankFusionConfig,
)
from cayu.core.messages import copy_message
from cayu.runtime.context import ContextBuildResult, ContextRequest
from cayu.storage.memory import KnowledgeStore

_SCHEMA_VERSION = "cayu.memory_reanchor_evaluation.v1"
_DEFAULT_SAMPLES = 20
_NAMESPACE = "benchmark:memory-reanchor"
_KEY_MATERIAL = "public-hermetic-memory-reanchor-key-v1"
_CASES = (
    "current_relevant",
    "disabled_control",
    "retained_control",
    "irrelevant_context",
    "same_entity_unrelated_aspect",
    "superseded_revision",
)

_CEILINGS = {
    "memory_zero_trigger_p95_ms": 25.0,
    "memory_triggered_p95_ms": 100.0,
    "memory_incremental_p95_ms": 75.0,
    "sqlite_zero_trigger_p50_ms": 30.0,
    "sqlite_zero_trigger_p95_ms": 150.0,
    "sqlite_triggered_p50_ms": 75.0,
    "sqlite_triggered_p95_ms": 300.0,
    "sqlite_incremental_p50_ms": 50.0,
    "sqlite_incremental_p95_ms": 200.0,
    "max_reanchor_bytes": 32_000,
    "max_reanchor_estimated_tokens": 8_000,
}


def _latency_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = max(0, (95 * len(ordered) + 99) // 100 - 1)
    return {
        "p50_ms": round(median(ordered), 6),
        "p95_ms": round(ordered[p95_index], 6),
    }


def _fusion() -> WeightedReciprocalRankFusionConfig:
    return WeightedReciprocalRankFusionConfig(
        configuration_version="memory-reanchor-evaluation-v1",
        channel_weights={
            KNOWLEDGE_LEXICAL_CHANNEL: 1.0,
            KNOWLEDGE_SEMANTIC_CHANNEL: 1.0,
            TRANSCRIPT_LEXICAL_CHANNEL: 1.0,
        },
        max_candidates_per_channel=20,
        fused_head_limit=20,
    )


def _admission() -> AutomaticRecallPolicy:
    return AutomaticRecallPolicy(
        calibration_version="memory-reanchor-evaluation-v1",
        fusion_strategy_version=WEIGHTED_RECIPROCAL_RANK_FUSION_VERSION,
        fusion_configuration_version="memory-reanchor-evaluation-v1",
        minimum_inject_score=0.01,
        minimum_offer_score=0.005,
    )


class _BoundaryContextPolicy(ContextPolicy):
    def __init__(
        self,
        *,
        compact: bool,
        marker: str,
        relevant: bool = True,
        shared_entity: bool = False,
        prior_boundaries: int = 1,
    ) -> None:
        self._compact = compact
        self._marker = marker
        self._relevant = relevant
        self._shared_entity = shared_entity
        self._prior_boundaries = prior_boundaries

    async def build(self, request: ContextRequest) -> list[Message]:
        crossed_boundary = (
            sum(message.role is MessageRole.TOOL for message in request.messages)
            >= self._prior_boundaries
        )
        if not self._compact or not crossed_boundary:
            return [copy_message(message) for message in request.messages]
        query = (
            f"Verified {self._marker} release fact?"
            if self._relevant
            else "Compacted current task: calculate an unrelated geometric area."
        )
        if self._shared_entity:
            query = f"Calculate the geometric area of the {self._marker} logo."
        return [Message.text(MessageRole.USER, query)]


class _TimedAutomaticRecallContextPolicy(AutomaticRecallContextPolicy):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.build_latencies_ms: list[float] = []

    async def build_with_checkpoint(
        self,
        request: ContextRequest,
        *,
        checkpoint: dict[str, Any] | None,
    ) -> ContextBuildResult:
        started = time.perf_counter_ns()
        try:
            return await super().build_with_checkpoint(request, checkpoint=checkpoint)
        finally:
            self.build_latencies_ms.append((time.perf_counter_ns() - started) / 1_000_000)


class _BoundaryTool(Tool):
    spec = ToolSpec(
        name="cross_memory_boundary",
        description="Cross one deterministic model boundary.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
    )

    def __init__(self, *, store: KnowledgeStore, entry_id: str, supersede: bool) -> None:
        super().__init__()
        self._store = store
        self._entry_id = entry_id
        self._supersede = supersede

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        del ctx, args
        if self._supersede:
            current = await self._store.get_entry(self._entry_id)
            if current is None:
                raise RuntimeError("The evaluation entry disappeared before supersession.")
            await self._store.append_entry_revision(
                current.model_copy(
                    update={
                        "revision": current.revision + 1,
                        "text": f"Current {self._entry_id} release fact is Saturday.",
                    }
                ),
                expected_revision=current.revision,
            )
        return ToolResult(content="Boundary crossed.")


def _memory_manifests(request: Any) -> tuple[str, ...]:
    return tuple(
        part.text
        for message in request.messages
        for part in message.content
        if type(part) is TextPart
        and (
            part.text.startswith("<cayu_automatic_memory")
            or part.text.startswith("<cayu_memory_delta")
        )
    )


def _memory_manifest_items(manifest: str) -> tuple[dict[str, Any], ...]:
    payload_start = manifest.find(">\n")
    payload_end = manifest.rfind("\n</cayu_")
    if payload_start < 0 or payload_end <= payload_start:
        raise RuntimeError("The evaluation encountered an invalid memory manifest envelope.")
    payload = json.loads(manifest[payload_start + 2 : payload_end])
    if type(payload) is not dict:
        raise RuntimeError("The evaluation encountered an invalid memory manifest payload.")
    raw_items = payload.get("items")
    if raw_items is None:
        raw_items = []
        for section in ("focus", "offer"):
            group = payload.get(section)
            if group is None:
                continue
            if type(group) is not dict or type(group.get("items")) is not list:
                raise RuntimeError(
                    "The evaluation encountered an invalid automatic-memory section."
                )
            raw_items.extend(group["items"])
    if type(raw_items) is not list:
        raise RuntimeError("The evaluation encountered an invalid memory item collection.")
    items = tuple(raw_items)
    if not all(type(item) is dict for item in items):
        raise RuntimeError("The evaluation encountered an invalid memory manifest item.")
    return items


async def _run_case(
    *,
    backend: str,
    sample: int,
    case: str,
    session_store: Any,
    knowledge_store: KnowledgeStore,
    prior_boundaries: int = 1,
) -> dict[str, Any]:
    marker = f"qzx{backend}{sample:04d}{case.replace('_', '')}"
    entry_id = f"entry-{marker}"
    original_fact = f"Verified {marker} release fact is Friday."
    await knowledge_store.create_entry(
        KnowledgeEntry(id=entry_id, namespace=_NAMESPACE, text=original_fact)
    )
    enabled = case != "disabled_control"
    compact = case != "retained_control"
    relevant = case not in {"irrelevant_context", "same_entity_unrelated_aspect"}
    supersede = case == "superseded_revision"
    policy = _TimedAutomaticRecallContextPolicy(
        _BoundaryContextPolicy(
            compact=compact,
            marker=marker,
            relevant=relevant,
            shared_entity=case == "same_entity_unrelated_aspect",
            prior_boundaries=prior_boundaries,
        ),
        admission_policy=_admission(),
        fusion_config=_fusion(),
        sources=AutomaticRecallSourceConfig(
            knowledge_namespace=_NAMESPACE,
        ),
        delta_policy=MemoryDeltaPolicy(reanchor_on_projection_loss=enabled),
    )
    provider = ScriptedModelProvider(
        [
            *[
                [
                    ModelStreamEvent.tool_call(
                        id=f"call_{marker}_{boundary}",
                        name="cross_memory_boundary",
                        arguments={},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
                for boundary in range(prior_boundaries)
            ],
            [
                ModelStreamEvent.text_delta("Evaluation complete."),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(
        session_store=session_store,
        request_footprint=RequestFootprintConfig(
            fingerprint_key_id="memory-reanchor-evaluation-v1",
            fingerprint_key=SecretStr(_KEY_MATERIAL),
        ),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="memory-reanchor-evaluation"),
            knowledge_store=knowledge_store,
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="evaluation-agent", model="scripted-model"),
        context_policy=policy,
        tools=[
            _BoundaryTool(
                store=knowledge_store,
                entry_id=entry_id,
                supersede=supersede,
            )
        ],
    )
    session_id = f"reanchor-{backend}-{sample:04d}-{case}"
    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="evaluation-agent",
                session_id=session_id,
                messages=[Message.text(MessageRole.USER, f"What does {marker} say?")],
            )
        )
    ]
    if not events or events[-1].type is not EventType.SESSION_COMPLETED:
        raise RuntimeError(f"The {case} evaluation did not complete.")
    if (
        len(provider.requests) != prior_boundaries + 1
        or len(policy.build_latencies_ms) != prior_boundaries + 1
    ):
        raise RuntimeError(f"The {case} evaluation crossed an unexpected number of boundaries.")
    first = _memory_manifests(provider.requests[0])
    second = _memory_manifests(provider.requests[-1])
    second_items = tuple(item for manifest in second for item in _memory_manifest_items(manifest))
    reanchor_items = tuple(
        item for item in second_items if item.get("reason") == "reanchored_current_revision"
    )
    checkpoint = await session_store.load_checkpoint(session_id)
    if checkpoint is None:
        raise RuntimeError(f"The {case} evaluation did not retain its checkpoint.")
    state = checkpoint["automatic_recall"]["delta_state"]
    reanchor_manifests = tuple(
        manifest
        for manifest in second
        if any(
            item.get("reason") == "reanchored_current_revision"
            for item in _memory_manifest_items(manifest)
        )
    )
    reanchor_bytes = sum(len(manifest.encode("utf-8")) for manifest in reanchor_manifests)
    reanchor_estimated_tokens = sum(
        item["projected_estimated_tokens"]
        for item in state["deltas"]
        if item["projection"] is not None
        and item["trigger"]["kind"] == "projection_removed_by_context_policy"
    )
    expected_reanchor = case == "current_relevant"

    def is_exact_current_item(item: dict[str, Any]) -> bool:
        read = item.get("read")
        return bool(
            type(read) is dict
            and read.get("entry_id") == entry_id
            and read.get("revision") == 1
            and item.get("text") == original_fact
        )

    exact_current_reanchor = len(reanchor_items) == 1 and is_exact_current_item(reanchor_items[0])
    exact_target_count = sum(is_exact_current_item(item) for item in reanchor_items)
    useful_reanchor_count = exact_target_count if expected_reanchor else 0
    false_injection_count = len(reanchor_items) - useful_reanchor_count
    stale_text_visible = any("Friday" in str(item.get("text", "")) for item in second_items) and (
        supersede
    )
    item_fingerprints = [
        json.dumps(item, sort_keys=True, separators=(",", ":")) for item in second_items
    ]
    duplicate_projection = (
        len(second) != len(set(second))
        or len(item_fingerprints) != len(set(item_fingerprints))
        or (
            bool(reanchor_items)
            and any(manifest.startswith("<cayu_automatic_memory") for manifest in second)
        )
    )
    if len(first) != 1 or (bool(reanchor_items) != expected_reanchor):
        raise RuntimeError(f"The {case} evaluation produced an unexpected re-anchor decision.")
    expected_second_count = 1 if case in {"current_relevant", "retained_control"} else 0
    if len(second) != expected_second_count:
        raise RuntimeError(f"The {case} evaluation produced an unexpected memory projection.")
    if state["original_projection_suppressed"] != compact:
        raise RuntimeError(f"The {case} evaluation recorded the wrong projection-loss state.")
    if expected_reanchor != exact_current_reanchor:
        raise RuntimeError(f"The {case} evaluation did not preserve the exact current revision.")
    if stale_text_visible or duplicate_projection:
        raise RuntimeError(f"The {case} evaluation produced unsafe memory projection.")
    return {
        "case": case,
        "second_boundary_latency_ms": policy.build_latencies_ms[-1],
        "reanchor_count": len(reanchor_items),
        "useful_reanchor_count": useful_reanchor_count,
        "reanchor_bytes": reanchor_bytes,
        "estimated_reanchor_tokens": reanchor_estimated_tokens,
        "exact_current_reanchor": exact_current_reanchor,
        "false_injection_count": false_injection_count,
        "stale_injection": stale_text_visible,
        "duplicate_projection": duplicate_projection,
        "original_projection_suppressed": state["original_projection_suppressed"],
        "reanchor_dispositions": tuple(
            outcome["disposition"] for outcome in state["reanchor_refresh_outcomes"]
        ),
    }


async def _backend_result(
    backend: str,
    *,
    samples: int,
    directory: Path,
) -> dict[str, Any]:
    scope = KnowledgeAccessScope.for_namespace(_NAMESPACE)
    if backend == "memory":
        session_store: Any = InMemorySessionStore()
        knowledge_store: KnowledgeStore = InMemoryKnowledgeStore(access_scope=scope)
    else:
        session_store = SQLiteSessionStore(directory / "sessions.sqlite")
        knowledge_store = SQLiteKnowledgeStore(
            directory / "knowledge.sqlite",
            access_scope=scope,
        )
    rows = []
    try:
        for sample in range(samples):
            for case in _CASES:
                rows.append(
                    await _run_case(
                        backend=backend,
                        sample=sample,
                        case=case,
                        session_store=session_store,
                        knowledge_store=knowledge_store,
                    )
                )
        history_probe = await _run_case(
            backend=backend,
            sample=samples,
            case="current_relevant",
            session_store=session_store,
            knowledge_store=knowledge_store,
            prior_boundaries=32,
        )
    finally:
        if isinstance(knowledge_store, SQLiteKnowledgeStore):
            await session_store.close()
            await knowledge_store.close()

    by_case = {case: [row for row in rows if row["case"] == case] for case in _CASES}
    useful = sum(row["useful_reanchor_count"] for row in rows)
    all_reanchors = sum(row["reanchor_count"] for row in rows)
    zero_latencies = [row["second_boundary_latency_ms"] for row in by_case["retained_control"]]
    triggered_latencies = [row["second_boundary_latency_ms"] for row in by_case["current_relevant"]]
    disabled_latencies = [row["second_boundary_latency_ms"] for row in by_case["disabled_control"]]
    incremental_latencies = [
        max(0.0, triggered - disabled)
        for triggered, disabled in zip(triggered_latencies, disabled_latencies, strict=True)
    ]
    return {
        "backend": backend,
        "sample_count": samples,
        "case_count": len(rows),
        "useful_reanchor_precision": round(useful / max(all_reanchors, 1), 6),
        "useful_reanchor_recall": round(useful / samples, 6),
        "false_injection_count": sum(row["false_injection_count"] for row in rows),
        "stale_injection_count": sum(row["stale_injection"] for row in rows),
        "exact_revision_mismatch_count": sum(
            row["reanchor_count"] > 0 and not row["exact_current_reanchor"] for row in rows
        ),
        "duplicate_projection_count": sum(row["duplicate_projection"] for row in rows),
        "reanchor_bytes": {
            "maximum": max(row["reanchor_bytes"] for row in rows),
            "total": sum(row["reanchor_bytes"] for row in rows),
        },
        "estimated_reanchor_tokens": {
            "maximum": max(row["estimated_reanchor_tokens"] for row in rows),
            "total": sum(row["estimated_reanchor_tokens"] for row in rows),
        },
        "zero_trigger_latency": _latency_summary(zero_latencies),
        "triggered_latency": _latency_summary(triggered_latencies),
        "incremental_trigger_latency": _latency_summary(incremental_latencies),
        "history_limit_probe": {
            "prior_exposures": 32,
            "latency_ms": history_probe["second_boundary_latency_ms"],
            "reanchor_count": history_probe["reanchor_count"],
            "false_injection_count": history_probe["false_injection_count"],
        },
        "case_outcomes": {
            case: {
                "reanchor_count": sum(row["reanchor_count"] for row in case_rows),
                "dispositions": sorted(
                    {
                        disposition
                        for row in case_rows
                        for disposition in row["reanchor_dispositions"]
                    }
                ),
            }
            for case, case_rows in by_case.items()
        },
    }


def _findings(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for result in results:
        backend = result["backend"]
        exact_checks = {
            "useful_reanchor_precision": result["useful_reanchor_precision"],
            "useful_reanchor_recall": result["useful_reanchor_recall"],
            "false_injection_count": result["false_injection_count"],
            "stale_injection_count": result["stale_injection_count"],
            "exact_revision_mismatch_count": result["exact_revision_mismatch_count"],
            "duplicate_projection_count": result["duplicate_projection_count"],
        }
        for metric, observed in exact_checks.items():
            expected = 1.0 if metric.startswith("useful_reanchor_") else 0
            if observed != expected:
                findings.append(
                    {
                        "backend": backend,
                        "metric": metric,
                        "observed": observed,
                        "expected": expected,
                    }
                )
        latency_checks = {
            f"{backend}_zero_trigger_p95_ms": result["zero_trigger_latency"]["p95_ms"],
            f"{backend}_triggered_p95_ms": result["triggered_latency"]["p95_ms"],
            f"{backend}_incremental_p95_ms": result["incremental_trigger_latency"]["p95_ms"],
        }
        if backend == "sqlite":
            latency_checks.update(
                {
                    "sqlite_zero_trigger_p50_ms": result["zero_trigger_latency"]["p50_ms"],
                    "sqlite_triggered_p50_ms": result["triggered_latency"]["p50_ms"],
                    "sqlite_incremental_p50_ms": result["incremental_trigger_latency"]["p50_ms"],
                }
            )
        for metric, observed in latency_checks.items():
            ceiling = _CEILINGS[metric]
            if observed > ceiling:
                findings.append(
                    {
                        "backend": backend,
                        "metric": metric,
                        "observed": observed,
                        "ceiling": ceiling,
                    }
                )
        observed_bytes = result["reanchor_bytes"]["maximum"]
        if observed_bytes > _CEILINGS["max_reanchor_bytes"]:
            findings.append(
                {
                    "backend": backend,
                    "metric": "max_reanchor_bytes",
                    "observed": observed_bytes,
                    "ceiling": _CEILINGS["max_reanchor_bytes"],
                }
            )
        observed_tokens = result["estimated_reanchor_tokens"]["maximum"]
        if observed_tokens > _CEILINGS["max_reanchor_estimated_tokens"]:
            findings.append(
                {
                    "backend": backend,
                    "metric": "max_reanchor_estimated_tokens",
                    "observed": observed_tokens,
                    "ceiling": _CEILINGS["max_reanchor_estimated_tokens"],
                }
            )
    return findings


async def _run(samples: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="cayu-memory-reanchor-evaluation-") as raw:
        root = Path(raw)
        results = [
            await _backend_result(
                backend,
                samples=samples,
                directory=root / backend,
            )
            for backend in ("memory", "sqlite")
        ]
    findings = _findings(results)
    return {
        "schema_version": _SCHEMA_VERSION,
        "methodology": {
            "kind": "hermetic_real_runtime_projection_loss_matrix",
            "provider": "scripted",
            "provider_calls": samples * len(_CASES) * 2 * 2 + 66,
            "history_limit_probe_prior_exposures": 32,
            "semantic_model_claim": False,
            "token_measurement": (
                "Runtime ObservedDeltaContextEstimator over the exact rendered manifest."
            ),
            "cases": list(_CASES),
            "backends": ["memory", "sqlite"],
            "samples_per_case": samples,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "ceilings": _CEILINGS,
        "results": results,
        "findings": findings,
        "passed": not findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=_DEFAULT_SAMPLES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.samples <= 1_000:
        parser.error("--samples must be between 1 and 1000")

    report = asyncio.run(_run(args.samples))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 1 if args.check and not report["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
