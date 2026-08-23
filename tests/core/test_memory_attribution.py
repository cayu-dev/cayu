from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from cayu import (
    CayuApp,
    ContextExposure,
    ContextExposureEvidenceKind,
    ContextExposurePage,
    ContextExposureState,
    ContextExposureTransition,
    ContextExposureTransitionRequest,
    InMemorySessionStore,
    MemoryAttribution,
    MemoryAttributionBounds,
    MemoryAttributionStatus,
    MemoryAttributionUnavailableReason,
    Message,
    RecallItemExposure,
    RecallReceipt,
    RecallReceiptItem,
    RecallReceiptPage,
    RecallSourceCoverage,
    RequestFootprintConfig,
    RunRequest,
    RuntimeEvidenceReport,
    RuntimeEvidenceRequest,
    SessionIdentity,
    SQLiteSessionStore,
    runtime_evidence,
)
from cayu.memory_evidence import validate_new_context_exposure


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _fingerprint(label: str, domain: str) -> dict[str, str]:
    return {
        "algorithm": "hmac-sha256",
        "domain": domain,
        "key_id": "memory-test",
        "digest": _digest(label),
    }


async def _create_session(
    store: InMemorySessionStore,
    session_id: str = "session",
    *,
    parent_session_id: str | None = None,
) -> None:
    await store.create(
        RunRequest(
            agent_name="agent",
            session_id=session_id,
            parent_session_id=parent_session_id,
            messages=[Message.text("user", "private request")],
        ),
        identity=SessionIdentity(provider_name="provider", model="model"),
    )


def _receipt(label: str, *, session_id: str = "session") -> RecallReceipt:
    item = RecallReceiptItem(
        ordinal=0,
        identity={
            "record_type": "knowledge_entry",
            "record_id": f"private-knowledge-{label}",
            "revision": "1",
        },
        representation_id="entry_text",
        content_sha256=_digest(f"private-content-{label}"),
        locator={
            "kind": "knowledge_entry",
            "entry_id": f"private-knowledge-{label}",
            "entry_revision": 1,
        },
        admission="admitted",
        selection_reason="calibrated_strong_match",
        fused_rank=1,
        match_channels=("knowledge.lexical",),
    )
    return RecallReceipt(
        receipt_id=f"private-receipt-{label}",
        session_id=session_id,
        interaction_id="private-interaction",
        model_step_id=f"mstep_{_digest(f'step-{label}')[:32]}",
        created_at=datetime(2026, 8, 23, 1, 0, tzinfo=UTC),
        situation_fingerprint=_fingerprint(f"situation-{label}", "recall_situation"),
        engine_version="test",
        source_configuration_fingerprint=_fingerprint(
            f"source-{label}", "recall_source_configuration"
        ),
        admission_policy_fingerprint=_fingerprint(f"policy-{label}", "recall_admission_policy"),
        access_scope_fingerprint=_fingerprint(f"scope-{label}", "recall_access_scope"),
        frontier_fingerprint=_fingerprint(f"frontier-{label}", "recall_frontier"),
        sources=(
            RecallSourceCoverage(
                source="private-source-name",
                required=True,
                channels=("knowledge.lexical",),
                state="complete",
                inspected_count=1,
                candidate_limit=1,
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


def _exposure(
    label: str,
    receipt: RecallReceipt,
) -> tuple[ContextExposure, tuple[RecallItemExposure, ...]]:
    transition = ContextExposureTransition(
        transition_id=f"private-transition-{label}",
        revision=0,
        state=ContextExposureState.PLANNED,
        occurred_at=receipt.created_at,
        evidence_kind=ContextExposureEvidenceKind.COMPOSITION_PLANNED,
        evidence_ref=f"private-evidence-ref-{label}",
    )
    exposure = ContextExposure(
        exposure_id=f"private-exposure-{label}",
        session_id=receipt.session_id,
        interaction_id=receipt.interaction_id,
        model_step_id=receipt.model_step_id,
        model_attempt_id=f"matt_{_digest(f'model-{label}')[:32]}",
        provider_attempt_id=f"patt_{_digest(f'provider-{label}')[:32]}",
        provider_name="provider",
        model_name="model",
        composition_fingerprint=_fingerprint(f"composition-{label}", "context_composition"),
        execution_profile_fingerprint=_fingerprint(f"profile-{label}", "execution_profile"),
        context_policy_fingerprint=_fingerprint(f"context-{label}", "context_policy"),
        tool_exposure_fingerprint=_fingerprint(f"tool-{label}", "tool_exposure"),
        request_contract_fingerprint=_fingerprint(f"request-{label}", "provider_request_contract"),
        receipt_ids=(receipt.receipt_id,),
        contributor_ids=("private-contributor",),
        created_at=transition.occurred_at,
        updated_at=transition.occurred_at,
        state=transition.state,
        state_revision=0,
        transitions=(transition,),
    )
    receipt_item = receipt.items[0]
    return exposure, (
        RecallItemExposure(
            exposure_id=exposure.exposure_id,
            receipt_id=receipt.receipt_id,
            ordinal=0,
            receipt_item_ordinal=0,
            identity=receipt_item.identity,
            representation_id=receipt_item.representation_id,
            content_sha256=receipt_item.content_sha256,
            locator=receipt_item.locator,
            admission=receipt_item.admission,
            selection_reason=receipt_item.selection_reason,
        ),
    )


def _app(store: InMemorySessionStore, *, with_alias_key: bool = True) -> CayuApp:
    return CayuApp(
        session_store=store,
        request_footprint=(
            RequestFootprintConfig(
                fingerprint_key_id="memory-test",
                fingerprint_key=SecretStr("a-private-test-key-with-more-than-32-bytes"),
            )
            if with_alias_key
            else RequestFootprintConfig()
        ),
        enable_logging=False,
    )


async def _report(
    app: CayuApp,
    *,
    bounds: MemoryAttributionBounds | None = None,
) -> RuntimeEvidenceReport:
    return await runtime_evidence(
        app,
        RuntimeEvidenceRequest(
            root_session_id="session",
            max_sessions=1,
            max_events=10,
            memory_attribution_bounds=bounds or MemoryAttributionBounds(),
        ),
    )


def _assert_attribution_round_trip(attribution: MemoryAttribution) -> None:
    assert MemoryAttribution.model_validate_json(attribution.model_dump_json()) == attribution


def test_runtime_evidence_projects_complete_memory_attribution_without_private_identity() -> None:
    async def scenario() -> RuntimeEvidenceReport:
        store = InMemorySessionStore()
        await _create_session(store)
        receipt = _receipt("one")
        await store.create_recall_receipt(receipt)
        exposure, items = _exposure("one", receipt)
        exposure = await store.create_context_exposure(exposure, items)
        for revision, state, evidence_kind in (
            (1, ContextExposureState.PREPARED, ContextExposureEvidenceKind.REQUEST_PREPARED),
            (
                2,
                ContextExposureState.DISPATCH_STARTED,
                ContextExposureEvidenceKind.DISPATCH_INTENT_COMMITTED,
            ),
            (
                3,
                ContextExposureState.INDETERMINATE,
                ContextExposureEvidenceKind.AMBIGUOUS_TRANSPORT,
            ),
        ):
            exposure = await store.transition_context_exposure(
                exposure.session_id,
                exposure.exposure_id,
                ContextExposureTransitionRequest(
                    transition_id=f"private-transition-one-{revision}",
                    expected_state=exposure.state,
                    expected_revision=exposure.state_revision,
                    state=state,
                    occurred_at=exposure.created_at + timedelta(seconds=revision),
                    evidence_kind=evidence_kind,
                    evidence_ref=f"private-evidence-ref-one-{revision}",
                ),
            )
        return await _report(_app(store))

    report = asyncio.run(scenario())
    attribution = report.sessions[0].memory_attribution
    assert attribution.status is MemoryAttributionStatus.COMPLETE
    assert attribution.truncated is False
    assert attribution.observed_receipt_count == 1
    assert attribution.observed_exposure_count == 1
    assert attribution.observed_item_count == 2
    assert attribution.receipts[0].receipt_alias == attribution.exposures[0].receipt_aliases[0]
    assert (
        attribution.receipts[0].items[0].item_alias == attribution.exposures[0].items[0].item_alias
    )
    assert attribution.exposures[0].state is ContextExposureState.INDETERMINATE
    assert attribution.exposures[0].provider_exposure_proven is False
    _assert_attribution_round_trip(attribution)

    serialized = report.model_dump_json()
    for private_value in (
        "private-receipt-one",
        "private-exposure-one",
        "private-interaction",
        "private-knowledge-one",
        "private-source-name",
        "private-contributor",
        "private-evidence-ref-one",
        _digest("private-content-one"),
        _digest("situation-one"),
        "a-private-test-key-with-more-than-32-bytes",
    ):
        assert private_value not in serialized
    assert RuntimeEvidenceReport.model_validate_json(serialized) == report

    mixed_key_attribution = attribution.model_dump(mode="python")
    mixed_key_attribution["exposures"][0]["receipt_aliases"][0]["key_id"] = "other-key"
    with pytest.raises(ValidationError, match="one key identity"):
        MemoryAttribution.model_validate(mixed_key_attribution)

    naive_timestamp_attribution = attribution.model_dump(mode="python")
    naive_timestamp_attribution["receipts"][0]["created_at"] = "2026-08-23T01:00:00"
    with pytest.raises(ValidationError, match="timezone-aware"):
        MemoryAttribution.model_validate(naive_timestamp_attribution)

    contradictory_item_attribution = attribution.model_dump(mode="python")
    contradictory_item_attribution["exposures"][0]["items"][0]["item_alias"]["digest"] = "0" * 64
    with pytest.raises(ValidationError, match="item linkage is contradictory"):
        MemoryAttribution.model_validate(contradictory_item_attribution)

    missing_receipt_attribution = attribution.model_dump(mode="python")
    missing_receipt_alias = missing_receipt_attribution["exposures"][0]["receipt_aliases"][0]
    missing_receipt_alias["digest"] = "1" * 64
    missing_receipt_attribution["exposures"][0]["items"][0]["receipt_alias"] = dict(
        missing_receipt_alias
    )
    with pytest.raises(ValidationError, match="references an omitted recall receipt"):
        MemoryAttribution.model_validate(missing_receipt_attribution)

    contradictory_scope_attribution = attribution.model_dump(mode="python")
    contradictory_scope_attribution["receipts"][0]["created_at"] = attribution.exposures[
        0
    ].created_at + timedelta(seconds=1)
    with pytest.raises(ValidationError, match="receipt/exposure linkage is contradictory"):
        MemoryAttribution.model_validate(contradictory_scope_attribution)

    contradictory_counts = attribution.model_dump(mode="python")
    contradictory_counts["receipts"][0]["eligible_count"] = 0
    with pytest.raises(ValidationError, match="eligible count does not match"):
        MemoryAttribution.model_validate(contradictory_counts)

    contradictory_admission = attribution.model_dump(mode="python")
    contradictory_admission["receipts"][0]["items"][0]["admission"] = "offered"
    with pytest.raises(ValidationError, match="selection reason does not match"):
        MemoryAttribution.model_validate(contradictory_admission)

    lost_observed_evidence = attribution.model_dump(mode="python")
    lost_observed_evidence.update(
        {
            "status": "truncated",
            "truncated": True,
            "receipts": (),
            "omitted_receipt_count_at_least": 0,
            "omitted_item_count_at_least": 0,
        }
    )
    with pytest.raises(ValidationError, match="omission counts lose observed evidence"):
        MemoryAttribution.model_validate(lost_observed_evidence)

    complete_with_truncated_items = attribution.model_dump(mode="python")
    complete_with_truncated_items["exposures"][0]["items_truncated"] = True
    with pytest.raises(ValidationError, match="cannot contain truncated item evidence"):
        MemoryAttribution.model_validate(complete_with_truncated_items)

    impossible_source_count = attribution.model_dump(mode="python")
    impossible_source_count["receipts"][0]["complete_source_count"] = 33
    with pytest.raises(ValidationError, match="source count exceeds"):
        MemoryAttribution.model_validate(impossible_source_count)

    impossible_contributor_count = attribution.model_dump(mode="python")
    impossible_contributor_count["exposures"][0]["contributor_count"] = 65
    with pytest.raises(ValidationError, match="less than or equal to 64"):
        MemoryAttribution.model_validate(impossible_contributor_count)

    duplicate_receipt_item = attribution.model_dump(mode="python")
    duplicate_item = dict(duplicate_receipt_item["exposures"][0]["items"][0])
    duplicate_item["ordinal"] = 1
    duplicate_item["item_alias"] = dict(duplicate_item["item_alias"])
    duplicate_item["item_alias"]["digest"] = "2" * 64
    duplicate_receipt_item["exposures"][0]["items"] += (duplicate_item,)
    duplicate_receipt_item["observed_item_count"] += 1
    with pytest.raises(ValidationError, match="may appear only once"):
        MemoryAttribution.model_validate(duplicate_receipt_item)

    missing_receipt_without_omission = attribution.model_dump(mode="python")
    missing_receipt_without_omission.update(
        {
            "status": "truncated",
            "truncated": True,
            "observed_receipt_count": 0,
            "receipts": (),
            "omitted_receipt_count_at_least": 0,
            "omitted_item_count_at_least": 1,
        }
    )
    with pytest.raises(ValidationError, match="loses known omitted recall receipts"):
        MemoryAttribution.model_validate(missing_receipt_without_omission)

    impossible_observed_count = attribution.model_dump(mode="python")
    impossible_observed_count["observed_receipt_count"] = 1_001
    with pytest.raises(ValidationError, match="less than or equal to 1000"):
        MemoryAttribution.model_validate(impossible_observed_count)

    duplicate_cross_receipt_item = attribution.model_dump(mode="python")
    duplicate_receipt = dict(duplicate_cross_receipt_item["receipts"][0])
    duplicate_receipt["receipt_alias"] = dict(duplicate_receipt["receipt_alias"])
    duplicate_receipt["receipt_alias"]["digest"] = "3" * 64
    duplicate_receipt["projection_ordinal"] = 1
    duplicate_cross_receipt_item["receipts"] += (duplicate_receipt,)
    duplicate_cross_receipt_item["observed_receipt_count"] += 1
    duplicate_cross_receipt_item["observed_item_count"] += 1
    with pytest.raises(ValidationError, match="unique across retained receipts"):
        MemoryAttribution.model_validate(duplicate_cross_receipt_item)

    duplicate_attempt = attribution.model_dump(mode="python")
    repeated_exposure = duplicate_attempt["exposures"][0].copy()
    repeated_exposure["exposure_alias"] = repeated_exposure["exposure_alias"].copy()
    repeated_exposure["exposure_alias"]["digest"] = "4" * 64
    repeated_exposure["projection_ordinal"] = 1
    duplicate_attempt["exposures"] += (repeated_exposure,)
    duplicate_attempt["observed_exposure_count"] += 1
    duplicate_attempt["observed_item_count"] += 1
    with pytest.raises(ValidationError, match="attempt identities must be unique"):
        MemoryAttribution.model_validate(duplicate_attempt)

    missing_receipt_conflict = attribution.model_dump(mode="python")
    missing_receipt_conflict.update(
        {
            "status": "truncated",
            "truncated": True,
            "receipts": (),
            "omitted_receipt_count_at_least": 1,
            "omitted_item_count_at_least": 1,
        }
    )
    conflicting_exposure = missing_receipt_conflict["exposures"][0].copy()
    conflicting_exposure["exposure_alias"] = conflicting_exposure["exposure_alias"].copy()
    conflicting_exposure["exposure_alias"]["digest"] = "5" * 64
    conflicting_exposure["interaction_alias"] = conflicting_exposure["interaction_alias"].copy()
    conflicting_exposure["interaction_alias"]["digest"] = "6" * 64
    conflicting_exposure["model_attempt_id"] = f"matt_{'4' * 32}"
    conflicting_exposure["provider_attempt_id"] = f"patt_{'4' * 32}"
    conflicting_exposure["projection_ordinal"] = 1
    missing_receipt_conflict["exposures"] += (conflicting_exposure,)
    missing_receipt_conflict["observed_exposure_count"] += 1
    missing_receipt_conflict["observed_item_count"] += 1
    with pytest.raises(ValidationError, match="across different interactions"):
        MemoryAttribution.model_validate(missing_receipt_conflict)

    missing_item_conflict = attribution.model_dump(mode="python")
    missing_item_conflict.update(
        {
            "status": "truncated",
            "truncated": True,
            "receipts": (),
            "omitted_receipt_count_at_least": 1,
            "omitted_item_count_at_least": 1,
        }
    )
    conflicting_item_exposure = missing_item_conflict["exposures"][0].copy()
    conflicting_item_exposure["exposure_alias"] = conflicting_item_exposure["exposure_alias"].copy()
    conflicting_item_exposure["exposure_alias"]["digest"] = "7" * 64
    conflicting_item_exposure["items"] = tuple(
        item.copy() for item in conflicting_item_exposure["items"]
    )
    conflicting_item_exposure["items"][0]["item_alias"] = conflicting_item_exposure["items"][0][
        "item_alias"
    ].copy()
    conflicting_item_exposure["items"][0]["item_alias"]["digest"] = "8" * 64
    conflicting_item_exposure["model_attempt_id"] = f"matt_{'5' * 32}"
    conflicting_item_exposure["provider_attempt_id"] = f"patt_{'5' * 32}"
    conflicting_item_exposure["projection_ordinal"] = 1
    missing_item_conflict["exposures"] += (conflicting_item_exposure,)
    missing_item_conflict["observed_exposure_count"] += 1
    missing_item_conflict["observed_item_count"] += 1
    with pytest.raises(ValidationError, match="conflicting facts"):
        MemoryAttribution.model_validate(missing_item_conflict)


def test_runtime_evidence_allows_receipt_reuse_by_a_later_model_step() -> None:
    async def scenario() -> RuntimeEvidenceReport:
        store = InMemorySessionStore()
        await _create_session(store)
        receipt = _receipt("cross-step")
        await store.create_recall_receipt(receipt)
        exposure, items = _exposure("cross-step", receipt)
        exposure = exposure.model_copy(
            update={"model_step_id": f"mstep_{_digest('later-step')[:32]}"}
        )
        await store.create_context_exposure(exposure, items)
        return await _report(_app(store))

    attribution = asyncio.run(scenario()).sessions[0].memory_attribution
    assert attribution.status is MemoryAttributionStatus.COMPLETE
    assert attribution.receipts[0].model_step_id != attribution.exposures[0].model_step_id
    assert attribution.receipts[0].interaction_alias == attribution.exposures[0].interaction_alias
    _assert_attribution_round_trip(attribution)


def test_context_exposure_write_contract_caps_item_materialization() -> None:
    receipt = _receipt("one")
    exposure, items = _exposure("one", receipt)

    with pytest.raises(ValueError, match="exceed their count bound"):
        validate_new_context_exposure(exposure, items * 65)


def test_memory_attribution_has_in_memory_and_sqlite_behavioral_parity(tmp_path: Path) -> None:
    async def project(store):
        await _create_session(store)
        receipt = _receipt("one")
        await store.create_recall_receipt(receipt)
        exposure, items = _exposure("one", receipt)
        await store.create_context_exposure(exposure, items)
        return (await _report(_app(store))).sessions[0].memory_attribution

    async def scenario():
        expected = await project(InMemorySessionStore())
        sqlite = SQLiteSessionStore(tmp_path / "sessions.sqlite")
        try:
            actual = await project(sqlite)
        finally:
            await sqlite.close()
        return expected, actual

    expected, actual = asyncio.run(scenario())
    assert actual == expected


def test_runtime_evidence_marks_memory_bounded_away_as_truncated() -> None:
    async def scenario() -> RuntimeEvidenceReport:
        store = InMemorySessionStore()
        await _create_session(store)
        for label in ("one", "two"):
            receipt = _receipt(label)
            await store.create_recall_receipt(receipt)
            exposure, items = _exposure(label, receipt)
            await store.create_context_exposure(exposure, items)
        return await _report(
            _app(store),
            bounds=MemoryAttributionBounds(
                max_receipts=1,
                max_exposures=1,
                max_items=1,
            ),
        )

    attribution = asyncio.run(scenario()).sessions[0].memory_attribution
    assert attribution.status is MemoryAttributionStatus.TRUNCATED
    assert attribution.truncated is True
    assert attribution.omitted_receipt_count_at_least >= 1
    assert attribution.omitted_exposure_count_at_least >= 1
    assert attribution.omitted_item_count_at_least >= 1
    _assert_attribution_round_trip(attribution)


def test_runtime_evidence_enforces_global_source_and_projection_byte_bounds() -> None:
    async def scenario(bounds: MemoryAttributionBounds):
        store = InMemorySessionStore()
        await _create_session(store)
        await store.create_recall_receipt(_receipt("one"))
        return (await _report(_app(store), bounds=bounds)).sessions[0].memory_attribution

    source_bounded = asyncio.run(scenario(MemoryAttributionBounds(max_source_bytes=1)))
    assert source_bounded.status is MemoryAttributionStatus.TRUNCATED
    assert source_bounded.receipts == ()
    assert source_bounded.observed_receipt_count == 0
    assert source_bounded.omitted_receipt_count_at_least == 0

    projection_bounded = asyncio.run(scenario(MemoryAttributionBounds(max_projection_bytes=1)))
    assert projection_bounded.status is MemoryAttributionStatus.TRUNCATED
    assert projection_bounded.receipts == ()
    assert projection_bounded.observed_receipt_count == 1
    assert projection_bounded.omitted_receipt_count_at_least == 1
    assert projection_bounded.omitted_item_count_at_least == 1


def test_runtime_evidence_omits_items_whose_receipt_was_bounded_away() -> None:
    async def scenario() -> RuntimeEvidenceReport:
        store = InMemorySessionStore()
        await _create_session(store)
        await store.create_recall_receipt(_receipt("one"))
        bounded_receipt = _receipt("two")
        await store.create_recall_receipt(bounded_receipt)
        exposure, items = _exposure("two", bounded_receipt)
        await store.create_context_exposure(exposure, items)
        return await _report(
            _app(store),
            bounds=MemoryAttributionBounds(max_receipts=1),
        )

    attribution = asyncio.run(scenario()).sessions[0].memory_attribution
    assert attribution.status is MemoryAttributionStatus.TRUNCATED
    assert attribution.exposures[0].items == ()
    assert attribution.exposures[0].items_truncated is True
    assert attribution.exposures[0].omitted_item_count_at_least == 1
    assert attribution.omitted_item_count_at_least >= 1


def test_runtime_evidence_redacts_existing_memory_without_alias_authority() -> None:
    async def scenario() -> RuntimeEvidenceReport:
        store = InMemorySessionStore()
        await _create_session(store)
        await store.create_recall_receipt(_receipt("one"))
        return await _report(_app(store, with_alias_key=False))

    attribution = asyncio.run(scenario()).sessions[0].memory_attribution
    assert attribution.status is MemoryAttributionStatus.REDACTED
    assert attribution.reason is MemoryAttributionUnavailableReason.ALIAS_KEY_UNAVAILABLE
    assert attribution.receipts == ()
    assert attribution.observed_receipt_count == 1
    _assert_attribution_round_trip(attribution)


def test_runtime_evidence_does_not_claim_redaction_when_a_bound_prevents_lookahead() -> None:
    async def scenario():
        store = InMemorySessionStore()
        await _create_session(store)
        await store.create_recall_receipt(_receipt("one"))
        return (
            (
                await _report(
                    _app(store, with_alias_key=False),
                    bounds=MemoryAttributionBounds(max_source_bytes=1),
                )
            )
            .sessions[0]
            .memory_attribution
        )

    attribution = asyncio.run(scenario())
    assert attribution.status is MemoryAttributionStatus.TRUNCATED
    assert attribution.reason is None
    assert attribution.observed_receipt_count == 0
    assert attribution.omitted_receipt_count_at_least == 0


def test_runtime_evidence_distinguishes_unsupported_and_contradictory_memory() -> None:
    class UnsupportedMemoryStore(InMemorySessionStore):
        supports_recall_evidence = False

    class ContradictoryMemoryStore(InMemorySessionStore):
        async def list_context_exposures(self, query):
            missing_receipt = _receipt("missing")
            exposure, _items = _exposure("orphan", missing_receipt)
            return ContextExposurePage(items=(exposure,), truncated=False)

    class OutOfOrderMemoryStore(InMemorySessionStore):
        async def list_recall_receipts(self, query):
            return RecallReceiptPage(
                items=(_receipt("two"), _receipt("one")),
                truncated=False,
            )

    class DuplicateAttemptMemoryStore(InMemorySessionStore):
        async def list_recall_receipts(self, query):
            return RecallReceiptPage(items=(_receipt("one"),), truncated=False)

        async def list_context_exposures(self, query):
            receipt = _receipt("one")
            first, _ = _exposure("one", receipt)
            second, _ = _exposure("two", receipt)
            second = second.model_copy(update={"model_attempt_id": first.model_attempt_id})
            return ContextExposurePage(items=(first, second), truncated=False)

    async def scenario(store: InMemorySessionStore):
        await _create_session(store)
        return (await _report(_app(store))).sessions[0].memory_attribution

    unsupported = asyncio.run(scenario(UnsupportedMemoryStore()))
    assert unsupported.status is MemoryAttributionStatus.UNAVAILABLE
    assert unsupported.reason is MemoryAttributionUnavailableReason.STORE_UNSUPPORTED
    _assert_attribution_round_trip(unsupported)

    contradictory = asyncio.run(scenario(ContradictoryMemoryStore()))
    assert contradictory.status is MemoryAttributionStatus.CONTRADICTORY
    assert contradictory.reason is MemoryAttributionUnavailableReason.CONTRADICTORY_EVIDENCE
    assert contradictory.exposures == ()
    _assert_attribution_round_trip(contradictory)

    out_of_order = asyncio.run(scenario(OutOfOrderMemoryStore()))
    assert out_of_order.status is MemoryAttributionStatus.CONTRADICTORY
    assert out_of_order.reason is MemoryAttributionUnavailableReason.CONTRADICTORY_EVIDENCE

    duplicate_attempt = asyncio.run(scenario(DuplicateAttemptMemoryStore()))
    assert duplicate_attempt.status is MemoryAttributionStatus.CONTRADICTORY
    assert duplicate_attempt.reason is MemoryAttributionUnavailableReason.CONTRADICTORY_EVIDENCE


def test_runtime_evidence_distinguishes_read_failure_and_shares_bounds_across_sessions() -> None:
    class FailingMemoryStore(InMemorySessionStore):
        async def list_recall_receipts(self, query):
            raise OSError("private database failure")

    async def failed_read():
        store = FailingMemoryStore()
        await _create_session(store)
        return (await _report(_app(store))).sessions[0].memory_attribution

    failed = asyncio.run(failed_read())
    assert failed.status is MemoryAttributionStatus.UNAVAILABLE
    assert failed.reason is MemoryAttributionUnavailableReason.EVIDENCE_READ_FAILED
    _assert_attribution_round_trip(failed)

    async def bounded_tree() -> RuntimeEvidenceReport:
        store = InMemorySessionStore()
        await _create_session(store, "session")
        await _create_session(store, "child", parent_session_id="session")
        await store.create_recall_receipt(_receipt("root", session_id="session"))
        child_receipt = _receipt("child", session_id="child")
        await store.create_recall_receipt(child_receipt)
        child_exposure, child_items = _exposure("child", child_receipt)
        await store.create_context_exposure(child_exposure, child_items)
        return await runtime_evidence(
            _app(store),
            RuntimeEvidenceRequest(
                root_session_id="session",
                max_sessions=2,
                max_events=10,
                memory_attribution_bounds=MemoryAttributionBounds(max_receipts=1),
            ),
        )

    report = asyncio.run(bounded_tree())
    assert report.sessions[0].memory_attribution.status is MemoryAttributionStatus.COMPLETE
    child = report.sessions[1].memory_attribution
    assert child.status is MemoryAttributionStatus.TRUNCATED
    assert child.omitted_receipt_count_at_least == 1
    assert child.exposures[0].receipt_aliases
    _assert_attribution_round_trip(child)


@pytest.mark.parametrize("record_kind", ["receipt", "exposure"])
def test_runtime_evidence_rejects_a_custom_store_page_before_reserializing_excess_rows(
    monkeypatch,
    record_kind: str,
) -> None:
    class OverLimitMemoryStore(InMemorySessionStore):
        async def list_recall_receipts(self, query):
            if record_kind != "receipt":
                return await super().list_recall_receipts(query)
            return RecallReceiptPage(
                items=(_receipt("one"), _receipt("two")),
                truncated=False,
            )

        async def list_context_exposures(self, query):
            if record_kind != "exposure":
                return await super().list_context_exposures(query)
            first, _ = _exposure("one", _receipt("one"))
            second, _ = _exposure("two", _receipt("two"))
            return ContextExposurePage(items=(first, second), truncated=False)

    serialized_rows = 0

    def track_serialized_rows(_value, _label):
        nonlocal serialized_rows
        serialized_rows += 1
        raise AssertionError("An over-limit page must be rejected before projection serialization.")

    monkeypatch.setattr(
        "cayu.runtime._memory_attribution.memory_evidence_document_bytes",
        track_serialized_rows,
    )

    async def scenario():
        store = OverLimitMemoryStore()
        await _create_session(store)
        return (
            (
                await _report(
                    _app(store),
                    bounds=MemoryAttributionBounds(max_receipts=1, max_exposures=1),
                )
            )
            .sessions[0]
            .memory_attribution
        )

    attribution = asyncio.run(scenario())
    assert attribution.status is MemoryAttributionStatus.UNAVAILABLE
    assert attribution.reason is MemoryAttributionUnavailableReason.EVIDENCE_READ_FAILED
    assert serialized_rows == 0


def test_runtime_evidence_rejects_over_limit_item_rows_as_a_bounded_read_failure() -> None:
    class OverLimitItemMemoryStore(InMemorySessionStore):
        async def load_recall_item_exposures(self, session_id, exposure_id):
            items = await super().load_recall_item_exposures(session_id, exposure_id)
            return items * 65

    async def scenario():
        store = OverLimitItemMemoryStore()
        await _create_session(store)
        receipt = _receipt("one")
        await store.create_recall_receipt(receipt)
        exposure, items = _exposure("one", receipt)
        await store.create_context_exposure(exposure, items)
        return (await _report(_app(store))).sessions[0].memory_attribution

    attribution = asyncio.run(scenario())
    assert attribution.status is MemoryAttributionStatus.UNAVAILABLE
    assert attribution.reason is MemoryAttributionUnavailableReason.EVIDENCE_READ_FAILED
