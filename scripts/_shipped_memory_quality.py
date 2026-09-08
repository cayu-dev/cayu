"""Fixed public qualification cases for the existing recall baseline command.

This is a scripted runtime regression probe, not an ablation scheduler or a
model-quality evaluator. All stores and generated application files are temporary.
"""

from __future__ import annotations

import importlib
import io
import json
import os
import platform
import tempfile
from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from typing import Any

from cayu import (
    AutomaticRecallContextPolicy,
    DefaultContextPolicy,
    EventType,
    InMemoryKnowledgeStore,
    InMemorySessionStore,
    InMemoryTaskStore,
    KnowledgeAccessScope,
    KnowledgeEntry,
    Message,
    ModelStreamEvent,
    ResumeRequest,
    RunRequest,
    ScriptedModelProvider,
    SQLiteKnowledgeStore,
    SQLiteSessionStore,
    TextPart,
)
from cayu.cli import main as cli_main
from cayu.cli.project import project_context
from cayu.memory_evidence import (
    KnowledgeChunkEvidenceLocator,
    KnowledgeEntryEvidenceLocator,
    RecallEvidenceQuery,
)


@dataclass(frozen=True)
class Turn:
    query: str
    focused: tuple[str, ...] = ()
    offered: tuple[str, ...] = ()
    response: str = "Scripted boundary complete."


@dataclass(frozen=True)
class Case:
    id: str
    category: str
    entries: tuple[KnowledgeEntry, ...]
    turns: tuple[Turn, ...]
    correction: str | None = None


def _cases() -> tuple[Case, ...]:
    rollback = KnowledgeEntry(
        id="rollback",
        text="Deployment rollback safeguards: verify health, restore the previous image, and monitor errors.",
    )
    query = "How should we configure deployment rollback safeguards?"
    weather = KnowledgeEntry(id="weather", text="Weather tomorrow sunny.")
    cases: list[Case] = []
    for population in (1, 20, 100):
        distractors = tuple(
            KnowledgeEntry(id=f"picnic-{index}", text=f"The deployment picnic uses table {index}.")
            for index in range(population)
        )
        cases.extend(
            (
                Case(f"weak-overlap-{population}", "silence", distractors, (Turn(query),)),
                Case(
                    f"useful-with-distractors-{population}",
                    "positive",
                    (rollback, *distractors),
                    (Turn(query, ("rollback:1",)),),
                ),
            )
        )
    cases.extend(
        (
            Case("empty-store", "silence", (), (Turn(query),)),
            Case("missing-antecedent", "short-query", (rollback,), (Turn("Why?"),)),
            Case(
                "follow-up-and-topic-switch",
                "continuity",
                (rollback, weather),
                (
                    Turn(query, ("rollback:1",), response="picnic table " * 3000),
                    Turn("Why?", ("rollback:1",)),
                    Turn("Weather tomorrow?", ("weather:1",)),
                ),
            ),
            Case(
                "current-revision",
                "currentness",
                (
                    KnowledgeEntry(
                        id="rollback", text="Deployment rollback safeguards use image OLD."
                    ),
                ),
                (Turn(query, ("rollback:2",)),),
                correction=rollback.text,
            ),
            Case(
                "expired-record",
                "currentness",
                (rollback.model_copy(update={"expires_at": datetime(2000, 1, 1, tzinfo=UTC)}),),
                (Turn(query),),
            ),
            Case(
                "namespace-isolation",
                "authorization",
                (rollback.model_copy(update={"namespace": "private"}),),
                (Turn(query),),
            ),
            Case(
                "title-supported-contract",
                "positive",
                (
                    KnowledgeEntry(
                        id="cache",
                        title="Harbor cache contract",
                        text="Caller-owned batch caches must use tenant-scoped keys and copy returned records.",
                    ),
                ),
                (Turn("Add cached batch lookup to Harbor.", ("cache:1",)),),
            ),
            Case(
                "exact-identifier",
                "positive",
                (
                    KnowledgeEntry(
                        id="atlas", text="Atlas-42 migration uses an expand-contract sequence."
                    ),
                ),
                (Turn("Atlas-42", ("atlas:1",)),),
            ),
            Case(
                "bounded-offers",
                "presentation",
                tuple(
                    KnowledgeEntry(
                        id=f"procedure-{i}",
                        text=f"Deployment rollback safeguards procedure {i}: verify health and restore the previous image.",
                    )
                    for i in range(7)
                ),
                (
                    Turn(
                        query,
                        tuple(f"procedure-{i}:1" for i in range(5)),
                        tuple(f"procedure-{i}:1" for i in range(5, 7)),
                    ),
                ),
            ),
        )
    )
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return tuple(
        replace(
            case,
            entries=tuple(
                entry.model_copy(update={"created_at": timestamp, "updated_at": timestamp})
                for entry in case.entries
            ),
        )
        for case in cases
    )


def _digest(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _policy_material(policy: AutomaticRecallContextPolicy) -> dict[str, Any]:
    if type(policy.base_policy) is not DefaultContextPolicy:
        raise ValueError("The shipped-default probe must qualify a new base policy explicitly.")
    return {
        **policy.configuration_material(),
        "base_policy": {
            "kind": "DefaultContextPolicy",
            "max_attachment_results": policy.base_policy.max_attachment_results,
        },
    }


@contextmanager
def _default_environment():
    # Generated settings must not inherit an operator's candidate configuration.
    # This probe is sequential, like project_context (which changes cwd/imports).
    saved = {key: value for key, value in os.environ.items() if key.startswith("CAYU_")}
    for key in saved:
        del os.environ[key]
    try:
        yield
    finally:
        for key in tuple(os.environ):
            if key.startswith("CAYU_"):
                del os.environ[key]
        os.environ.update(saved)


def _projection(request: Any) -> tuple[dict[str, Any], int]:
    manifests = [
        part.text
        for message in request.messages
        for part in message.content
        if isinstance(part, TextPart) and part.text.startswith("<cayu_automatic_memory")
    ]
    if not manifests:
        return {}, 0
    if len(manifests) != 1:
        raise ValueError("Expected at most one frozen memory projection.")
    manifest = manifests[0]
    start, end = manifest.index(">\n") + 2, manifest.rindex("\n</cayu_automatic_memory>")
    payload = json.loads(manifest[start:end])
    if type(payload) is not dict:
        raise ValueError("Expected an automatic-memory object.")
    return payload, len(manifest.encode("utf-8"))


def _items(payload: dict[str, Any], section: str) -> list[dict[str, Any]]:
    group = payload.get(section)
    return [] if group is None else group["items"]


def _identities(items: list[dict[str, Any]]) -> list[str]:
    return sorted(f"{item['read']['entry_id']}:{item['read']['revision']}" for item in items)


async def _run_case(
    case: Case, backend: str, directory: Path, build_app: Any, build_policy: Any
) -> dict[str, Any]:
    namespace = build_policy().sources.knowledge_namespace
    scope = KnowledgeAccessScope.for_namespace(namespace)
    database = directory / f"{backend}-{case.id}.sqlite"
    knowledge = InMemoryKnowledgeStore() if backend == "memory" else SQLiteKnowledgeStore(database)
    sessions = InMemorySessionStore() if backend == "memory" else SQLiteSessionStore(database)
    rows: list[dict[str, Any]] = []
    configurations: list[dict[str, Any]] = []
    stage = "seed"
    try:
        for entry in case.entries:
            if entry.namespace == "default":
                entry = entry.model_copy(update={"namespace": namespace})
            created = await knowledge.create_entry(
                entry, access_scope=KnowledgeAccessScope.privileged()
            )
            if case.correction is not None:
                await knowledge.append_entry_revision(
                    created.model_copy(update={"revision": 2, "text": case.correction}),
                    expected_revision=1,
                    access_scope=KnowledgeAccessScope.privileged(),
                )
        for turn_index, turn in enumerate(case.turns):
            stage = f"turn-{turn_index}:configuration"
            # New application graph for every turn; SQLite adapters are reopened
            # as well. This proves next-turn reconstruction, not crash recovery.
            if turn_index and backend == "sqlite":
                assert isinstance(sessions, SQLiteSessionStore)
                assert isinstance(knowledge, SQLiteKnowledgeStore)
                await sessions.close()
                await knowledge.close()
                sessions = SQLiteSessionStore(database)
                knowledge = SQLiteKnowledgeStore(database)
            provider = ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.text_delta(turn.response),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ]
                ]
            )
            app = build_app(
                provider=provider,
                session_store=sessions,
                task_store=InMemoryTaskStore(),
                knowledge_store=knowledge,
            )
            agent_name = app.describe().agents[0].name
            # Inspect the actual registered collaborator, not a reconstructed
            # benchmark policy. This is a repository-owned conformance probe.
            policy = app._get_registered_agent(agent_name).context_policy
            expected_policy = build_policy()
            if not isinstance(policy, AutomaticRecallContextPolicy) or not isinstance(
                expected_policy, AutomaticRecallContextPolicy
            ):
                raise ValueError("Generated application did not wire automatic recall.")
            material = _policy_material(policy)
            if (
                material != _policy_material(expected_policy)
                or policy.configuration_fingerprint() != expected_policy.configuration_fingerprint()
            ):
                raise ValueError("Generated and runtime memory configurations differ.")
            configurations.append(material)
            before = (
                await sessions.list_recall_receipts(RecallEvidenceQuery(session_id=case.id))
                if turn_index
                else None
            )
            known = set() if before is None else {item.receipt_id for item in before.items}
            if before is not None and (before.truncated or before.next_cursor is not None):
                raise ValueError("Incomplete prior receipt evidence.")
            stage = f"turn-{turn_index}:runtime"
            started = perf_counter_ns()
            messages = [Message.text("user", turn.query)]
            stream = (
                app.run(RunRequest(agent_name=agent_name, session_id=case.id, messages=messages))
                if turn_index == 0
                else app.resume(ResumeRequest(session_id=case.id, messages=messages))
            )
            events = [event async for event in stream]
            elapsed_ms = (perf_counter_ns() - started) / 1_000_000
            stage = f"turn-{turn_index}:evidence"
            if (
                not events
                or events[-1].type is not EventType.SESSION_COMPLETED
                or len(provider.requests) != 1
            ):
                raise ValueError("Scripted turn did not complete exactly one provider boundary.")
            payload, projected_bytes = _projection(provider.requests[0])
            focused = _identities(_items(payload, "focus"))
            offered = _identities(_items(payload, "offer"))
            receipts = await sessions.list_recall_receipts(RecallEvidenceQuery(session_id=case.id))
            fresh = [item for item in receipts.items if item.receipt_id not in known]
            if receipts.truncated or receipts.next_cursor is not None or len(fresh) != 1:
                raise ValueError("Expected one complete new receipt per turn.")
            receipt = fresh[0]
            exposures = await sessions.list_context_exposures(
                RecallEvidenceQuery(session_id=case.id, interaction_id=receipt.interaction_id)
            )
            if (
                exposures.truncated
                or exposures.next_cursor is not None
                or len(exposures.items) != 1
            ):
                raise ValueError("Expected one complete context exposure per turn.")
            exposure = exposures.items[0]
            item_exposures = await sessions.load_recall_item_exposures(
                case.id, exposure.exposure_id
            )
            receipt_identities = sorted(
                (f"{item.locator.entry_id}:{item.locator.entry_revision}", item.admission.value)
                for item in receipt.items
                if isinstance(
                    item.locator, (KnowledgeEntryEvidenceLocator, KnowledgeChunkEvidenceLocator)
                )
            )
            exposure_identities = sorted(
                (f"{item.locator.entry_id}:{item.locator.entry_revision}", item.admission.value)
                for item in item_exposures
                if isinstance(
                    item.locator, (KnowledgeEntryEvidenceLocator, KnowledgeChunkEvidenceLocator)
                )
            )
            projected_identities = sorted(
                [(identity, "admitted") for identity in focused]
                + [(identity, "offered") for identity in offered]
            )
            exact_item_linkage = (
                len(receipt_identities) == len(receipt.items)
                and len(exposure_identities) == len(item_exposures)
                and receipt_identities == exposure_identities == projected_identities
                and all(item.receipt_id == receipt.receipt_id for item in item_exposures)
                and sorted(
                    (
                        item.identity.sort_key(),
                        item.representation_id,
                        item.content_sha256,
                        item.locator.model_dump_json(),
                    )
                    for item in receipt.items
                )
                == sorted(
                    (
                        item.identity.sort_key(),
                        item.representation_id,
                        item.content_sha256,
                        item.locator.model_dump_json(),
                    )
                    for item in item_exposures
                )
            )
            exact_reads = True
            for item in [*_items(payload, "focus"), *_items(payload, "offer")]:
                read = item["read"]
                record = await knowledge.get_entry(read["entry_id"], access_scope=scope)
                exact_reads = (
                    exact_reads and record is not None and record.revision == read["revision"]
                )
                if item["source"] == "knowledge_chunk":
                    chunks = await knowledge.read_chunks(
                        read["entry_id"],
                        revision=read["revision"],
                        access_scope=scope,
                        chunk_index=read["chunk_index"],
                        around=0,
                        max_chunks=1,
                    )
                    exact_reads = (
                        exact_reads and len(chunks) == 1 and chunks[0].id == item["chunk_id"]
                    )
                elif item["source"] != "knowledge_entry":
                    exact_reads = False
            previews_useful = all(
                isinstance(item.get("preview"), str) and bool(item["preview"].strip())
                for item in _items(payload, "offer")
            )
            # These fixed public records fit in both the full-text and preview
            # budgets. Compare against the fixture, not receipt hashes or a
            # second rendering path: those can agree while delivered text is stale.
            expected_text = {
                entry.id: case.correction if case.correction is not None else entry.text
                for entry in case.entries
            }
            delivered_content_matches_fixture = all(
                item.get(field) == expected_text.get(item["read"]["entry_id"])
                and item["read"]["entry_id"] in expected_text
                and item.get(f"{field}_complete") is True
                for section, field in (("focus", "text"), ("offer", "preview"))
                for item in _items(payload, section)
            )
            rows.append(
                {
                    "turn": turn_index,
                    "expected_focused": sorted(turn.focused),
                    "expected_offered": sorted(turn.offered),
                    "focused": focused,
                    "offered": offered,
                    "exact_current_read_references": exact_reads,
                    "delivered_content_matches_fixture": delivered_content_matches_fixture,
                    "exposure_state": exposure.state.value,
                    "receipt_linked": receipt.receipt_id in exposure.receipt_ids,
                    "exact_item_linkage": exact_item_linkage,
                    "receipt_items": [list(item) for item in receipt_identities],
                    "exposure_items": [list(item) for item in exposure_identities],
                    "offer_previews_present": previews_useful,
                    "sources": [source.model_dump(mode="json") for source in receipt.sources],
                    "query_resolution": None
                    if receipt.query_resolution is None
                    else receipt.query_resolution.model_dump(mode="json"),
                    "inspected_count": receipt.inspected_count,
                    "eligible_count": receipt.eligible_count,
                    "admitted_count": receipt.admitted_count,
                    "offered_count": receipt.offered_count,
                    "silent_count": receipt.silent_count,
                    "omitted_count": receipt.omitted_count,
                    "truncated": receipt.truncated,
                    "projection_bytes": projected_bytes,
                    "projection_estimated_tokens": (projected_bytes + 3) // 4,
                    "runtime_turn_ms": elapsed_ms,
                }
            )
        return {
            "case_id": case.id,
            "category": case.category,
            "backend": backend,
            "knowledge_records": len(case.entries),
            "configuration": configurations[0],
            "configuration_sha256": _digest(configurations[0]),
            "turns": rows,
            "configuration_stable": all(value == configurations[0] for value in configurations),
            "error": None,
        }
    except Exception as exc:
        # Keep an abnormal case in the matrix, without serializing input text or
        # exception payloads into a portable result.
        return {
            "case_id": case.id,
            "category": case.category,
            "backend": backend,
            "knowledge_records": len(case.entries),
            "turns": rows,
            "error": type(exc).__name__,
            "error_stage": stage,
        }
    finally:
        if isinstance(sessions, SQLiteSessionStore):
            await sessions.close()
        if isinstance(knowledge, SQLiteKnowledgeStore):
            await knowledge.close()


def findings(results: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    expected = {(backend, case.id): case for backend in ("memory", "sqlite") for case in _cases()}
    observed: set[tuple[str, str]] = set()
    configurations = []
    for result in results:
        key = result["backend"], result["case_id"]
        label = ":".join(key)
        if key not in expected or key in observed:
            failures.append(f"{label}: unexpected or duplicate case")
            continue
        observed.add(key)
        case = expected[key]
        if result["error"] is not None:
            failures.append(f"{label}: incomplete at {result['error_stage']} ({result['error']})")
            continue
        configurations.append(result["configuration"])
        if not result["configuration_stable"] or len(result["turns"]) != len(case.turns):
            failures.append(f"{label}: configuration or turn-count mismatch")
        for row, turn in zip(result["turns"], case.turns, strict=False):
            prefix = f"{label}:turn-{row['turn']}"
            if row["focused"] != sorted(turn.focused) or row["offered"] != sorted(turn.offered):
                failures.append(
                    f"{prefix}: expected focus={sorted(turn.focused)}, offer={sorted(turn.offered)}; observed focus={row['focused']}, offer={row['offered']}"
                )
            if (
                not row["exact_current_read_references"]
                or not row["delivered_content_matches_fixture"]
                or not row["receipt_linked"]
                or not row["exact_item_linkage"]
                or not row["offer_previews_present"]
                or row["exposure_state"] != "completed"
            ):
                failures.append(
                    f"{prefix}: inconsistent delivered content or exact-read/delivery evidence"
                )
            if row["admitted_count"] != len(row["focused"]) or row["offered_count"] != len(
                row["offered"]
            ):
                failures.append(f"{prefix}: receipt and projection counts disagree")
            if len(row["sources"]) != 1 or any(
                source["source"] != "knowledge"
                or source["failure_code"]
                not in {"semantic_unsupported", "lexical_truncated+semantic_unsupported"}
                or source["state"] != "partial"
                for source in row["sources"]
            ):
                failures.append(f"{prefix}: unexpected source set or dishonest semantic coverage")
    failures.extend(
        f"{backend}:{case_id}: missing case"
        for backend, case_id in sorted(set(expected) - observed)
    )
    if configurations and any(value != configurations[0] for value in configurations):
        failures.append("evaluated configurations differ across backends/cases")
    return failures


def _summary(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for backend in ("memory", "sqlite"):
        cases = [case for case in results if case["backend"] == backend]
        rows = [row for case in cases for row in case["turns"]]
        latencies = sorted(row["runtime_turn_ms"] for row in rows)
        summaries.append(
            {
                "backend": backend,
                "case_count": len(cases),
                "incomplete_case_count": sum(case["error"] is not None for case in cases),
                "observed_turn_count": len(rows),
                "false_focused_items": sum(
                    len(set(row["focused"]) - set(row["expected_focused"])) for row in rows
                ),
                "missing_focused_items": sum(
                    len(set(row["expected_focused"]) - set(row["focused"])) for row in rows
                ),
                "false_offered_items": sum(
                    len(set(row["offered"]) - set(row["expected_offered"])) for row in rows
                ),
                "missing_offered_items": sum(
                    len(set(row["expected_offered"]) - set(row["offered"])) for row in rows
                ),
                "maximum_projection_bytes": max(
                    (row["projection_bytes"] for row in rows), default=0
                ),
                "maximum_projection_estimated_tokens": max(
                    (row["projection_estimated_tokens"] for row in rows), default=0
                ),
                "runtime_turn_p50_ms": median(latencies) if latencies else None,
                "runtime_turn_p95_ms": latencies[(95 * len(latencies) + 99) // 100 - 1]
                if latencies
                else None,
            }
        )
    return summaries


async def run_shipped_default_quality() -> dict[str, Any]:
    cases = _cases()
    with (
        tempfile.TemporaryDirectory(prefix="cayu-shipped-memory-") as temporary,
        _default_environment(),
    ):
        root = Path(temporary).resolve()
        with redirect_stdout(io.StringIO()):
            if cli_main(["new", "memory_quality", "--dir", str(root)]) != 0:
                raise RuntimeError("Unable to generate the standard application.")
        project = root / "memory_quality"
        with project_context(project):
            build_app = importlib.import_module("app").build_app
            build_policy = importlib.import_module("memory.context").build_context_policy
            results = [
                await _run_case(case, backend, root, build_app, build_policy)
                for backend in ("memory", "sqlite")
                for case in cases
            ]
        source_hashes = {
            str(path.relative_to(project)): sha256(path.read_bytes()).hexdigest()
            for path in sorted(project.rglob("*.py"))
        }
    errors = findings(results)
    return {
        "schema_version": "cayu.shipped_memory_quality.v1",
        "policy_kind": "generated_standard_default",
        "corpus_revision": "public-shipped-memory-regression-v1",
        "corpus_sha256": _digest(
            [
                {
                    "id": case.id,
                    "category": case.category,
                    "entries": [entry.model_dump(mode="json") for entry in case.entries],
                    "turns": [asdict(turn) for turn in case.turns],
                    "correction": case.correction,
                }
                for case in cases
            ]
        ),
        "generated_source_sha256": source_hashes,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "methodology": {
            "provider": "scripted",
            "languages": ["en"],
            "split": "fixed_regression_not_calibration_or_held_out",
            "semantic_embeddings": False,
            "token_estimator": "ceil(rendered_utf8_bytes/4)",
            "timing": "entire runtime turn excluding app/store construction; descriptive, not a latency gate",
            "restart": "new app each turn; SQLite connections reopened between turns; no process crash simulated",
        },
        "limitations": [
            "Not evidence of live-model answer improvement or general semantic relevance.",
            "No PostgreSQL evidence; memory/SQLite results cannot establish its parity.",
            "Does not complete multilingual, held-out calibration, semantic failure, compaction, or interrupted-recovery qualification.",
            "Default delta/re-anchoring configuration is not changed; opt-in re-anchoring has a separate runtime matrix.",
        ],
        "results": results,
        "summary": _summary(results),
        "findings": errors,
        "passed": not errors,
    }
