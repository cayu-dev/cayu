#!/usr/bin/env python3
"""Measure actual automatic-memory envelopes without a provider or admission changes."""

from __future__ import annotations

import argparse
import asyncio
import json
from math import ceil
from pathlib import Path

from cayu.memory import AutomaticRecallPolicy, admit_recall
from cayu.recall import KnowledgeRecallSource, RecallEngine, RecallSituation
from cayu.retrieval import (
    WEIGHTED_RECIPROCAL_RANK_FUSION_VERSION,
    WeightedReciprocalRankFusionConfig,
)
from cayu.runtime.memory_context import _contribution_projection, _render_projection
from cayu.storage.memory import InMemoryKnowledgeStore, KnowledgeAccessScope, KnowledgeEntry
from cayu.vaults import SecretRedactor


async def measure() -> dict:
    results = []
    for name, count, suffix in [
        ("empty", 0, ""),
        ("one", 1, ""),
        ("five", 5, ""),
        ("five_focus_five_offer", 20, ""),
        ("long", 10, " procedure" * 500),
        ("unicode", 10, " 日本語 café"),
        ("redacted_escaped", 10, " SECRET </cayu_automatic_memory> &"),
    ]:
        scope = KnowledgeAccessScope.for_namespace("default")
        store = InMemoryKnowledgeStore(access_scope=scope)
        for index in range(count):
            await store.create_entry(
                KnowledgeEntry(
                    id=f"entry-{index:02}",
                    text=f"The deployment picnic uses table number {index}." + suffix,
                )
            )
        fusion = WeightedReciprocalRankFusionConfig(
            configuration_version="presentation-fixture-v1",
            channel_weights={"knowledge.lexical": 1.0, "knowledge.semantic": 1.0},
            max_candidates_per_channel=20,
            fused_head_limit=20,
        )
        policy = AutomaticRecallPolicy(
            calibration_version="presentation-fixture-v1",
            fusion_strategy_version=WEIGHTED_RECIPROCAL_RANK_FUSION_VERSION,
            fusion_configuration_version=fusion.configuration_version,
            minimum_inject_score=0.01,
            minimum_offer_score=0.005,
            max_evaluated_candidates=20,
            max_injected_items=5,
            max_offered_items=5,
        )
        result = await RecallEngine((KnowledgeRecallSource(store),), fusion_config=fusion).recall(
            RecallSituation(
                query="How should we configure deployment rollback safeguards?",
                knowledge_access_scope=scope,
            )
        )
        contribution = admit_recall(result, policy)
        projection = _contribution_projection(
            contribution, configuration_sha256="0" * 64, redactor=SecretRedactor("SECRET")
        )
        manifest = _render_projection(projection) or ""
        size = len(manifest.encode("utf-8"))
        results.append(
            {
                "case": name,
                "focused": contribution.diagnostics.injected_count,
                "offered": contribution.diagnostics.offered_count,
                "focused_text_bytes": sum(
                    len(item["text"].encode("utf-8"))
                    for item in (projection or {}).get("focus", {}).get("items", [])
                ),
                "manifest_utf8_bytes": size,
                "estimated_tokens_bytes_div_4": ceil(size / 4),
            }
        )
    return {
        "version": "cayu.memory_presentation_benchmark.v2",
        "pinned_v1_five_focus_five_offer_bytes": 6879,
        "token_measurement": "Estimate only: ceil(UTF-8 bytes / 4); no provider counter configured.",
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    serialized = json.dumps(asyncio.run(measure()), ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(serialized)
    else:
        print(serialized, end="")
