from __future__ import annotations

import asyncio
import hashlib
import hmac
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from cayu._validation import canonical_durable_json_bytes, require_durable_clean_nonblank
from cayu.memory import AutomaticRecallContribution, AutomaticRecallPolicy
from cayu.memory_evidence import (
    ContextExposure,
    ContextExposureEvidenceKind,
    ContextExposureState,
    ContextExposureTransition,
    ContextExposureTransitionRequest,
    KeyedEvidenceFingerprintDomain,
    KnowledgeChunkEvidenceLocator,
    KnowledgeEntryEvidenceLocator,
    OpaqueRecallEvidenceLocator,
    RecallEvidenceLocator,
    RecallItemAdmission,
    RecallItemExposure,
    RecallItemSelectionReason,
    RecallReceipt,
    RecallReceiptItem,
    RecallSourceCoverage,
    RecallSourceCoverageState,
    TranscriptMessageEvidenceLocator,
    keyed_evidence_fingerprint,
    new_context_exposure_id,
    new_context_exposure_transition_id,
    new_provider_attempt_id,
    new_recall_receipt_id,
)
from cayu.providers.base import ModelRequest
from cayu.recall import RecallResult, RecallSituation, RecallSourceStatus
from cayu.runtime._session_control import SessionInterruptedByRequest
from cayu.runtime.execution_profiles import (
    ExecutionProfileComponentClass,
    ExecutionProfileIdentity,
)
from cayu.runtime.execution_units import ModelAttemptIdentity
from cayu.runtime.request_footprints import RequestFootprintConfig
from cayu.runtime.sessions import SessionStore
from cayu.runtime.tool_exposure import ResolvedToolExposure

_MEMORY_EVIDENCE_KEY_DERIVATION_CONTEXT = b"cayu.memory-evidence.request-footprint-key.v1"
_AUTOMATIC_RECALL_CHECKPOINT_BINDING_CONTEXT = b"cayu.automatic-recall-checkpoint-binding.v1"
_AUTOMATIC_RECALL_OPEN_TAG = '<cayu_automatic_memory version="1">'
_AUTOMATIC_RECALL_CLOSE_TAG = "</cayu_automatic_memory>"


@dataclass(frozen=True, slots=True)
class MemoryEvidenceKey:
    """Detached HMAC key used only while constructing private evidence fingerprints."""

    key_id: str
    key: bytes = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "key_id",
            require_durable_clean_nonblank(self.key_id, "memory_evidence_key_id"),
        )
        if type(self.key) is not bytes or len(self.key) != 32:
            raise ValueError("memory evidence keys must contain exactly 32 derived bytes.")


@dataclass(frozen=True, slots=True)
class MemoryEvidenceReference:
    receipt_id: str
    receipt_document_sha256: str
    receipt_manifest_binding_hmac_sha256: str
    manifest_sha256: str | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "receipt_id",
            require_durable_clean_nonblank(self.receipt_id, "receipt_id"),
        )
        if (
            type(self.receipt_document_sha256) is not str
            or (len(self.receipt_document_sha256) != 64)
            or any(
                character not in "0123456789abcdef" for character in self.receipt_document_sha256
            )
        ):
            raise ValueError("receipt_document_sha256 must be a lowercase SHA-256 digest.")
        if (
            type(self.receipt_manifest_binding_hmac_sha256) is not str
            or len(self.receipt_manifest_binding_hmac_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.receipt_manifest_binding_hmac_sha256
            )
        ):
            raise ValueError(
                "receipt_manifest_binding_hmac_sha256 must be a lowercase HMAC-SHA-256 digest."
            )
        if self.manifest_sha256 is not None and (
            type(self.manifest_sha256) is not str
            or len(self.manifest_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.manifest_sha256)
        ):
            raise ValueError("manifest_sha256 must be a lowercase SHA-256 digest.")


_ACTIVE_MEMORY_EVIDENCE_KEY: ContextVar[MemoryEvidenceKey | None] = ContextVar(
    "cayu_active_memory_evidence_key",
    default=None,
)


@contextmanager
def memory_evidence_key_scope(key: MemoryEvidenceKey | None) -> Iterator[None]:
    """Keep derived key material inside the runtime-owned context-build scope."""

    if key is not None and type(key) is not MemoryEvidenceKey:
        raise TypeError("key must be a MemoryEvidenceKey or None.")
    token = _ACTIVE_MEMORY_EVIDENCE_KEY.set(key)
    try:
        yield
    finally:
        _ACTIVE_MEMORY_EVIDENCE_KEY.reset(token)


def active_memory_evidence_key() -> MemoryEvidenceKey | None:
    return _ACTIVE_MEMORY_EVIDENCE_KEY.get()


def memory_evidence_key(config: RequestFootprintConfig) -> MemoryEvidenceKey | None:
    """Derive a purpose-specific key from the configured request-footprint secret."""

    if type(config) is not RequestFootprintConfig:
        raise TypeError("config must be a RequestFootprintConfig.")
    if config.fingerprint_key_id is None or config.fingerprint_key is None:
        return None
    raw = config.fingerprint_key.get_secret_value().encode("utf-8")
    derived = hmac.digest(raw, _MEMORY_EVIDENCE_KEY_DERIVATION_CONTEXT, "sha256")
    return MemoryEvidenceKey(key_id=config.fingerprint_key_id, key=derived)


def build_recall_receipt(
    *,
    session_id: str,
    interaction_id: str,
    model_step_id: str,
    situation: RecallSituation,
    result: RecallResult,
    contribution: AutomaticRecallContribution,
    admission_policy: AutomaticRecallPolicy,
    source_configuration: Mapping[str, Any],
    key: MemoryEvidenceKey,
    created_at: datetime | None = None,
) -> RecallReceipt:
    """Build one receipt from the exact result already used for admission."""

    if result.situation_sha256 != situation.fingerprint():
        raise ValueError("Recall result does not belong to the supplied situation.")
    if contribution.situation_sha256 != result.situation_sha256:
        raise ValueError("Automatic recall contribution belongs to a different result.")
    if contribution.policy_sha256 != admission_policy.fingerprint():
        raise ValueError("Automatic recall contribution belongs to a different policy.")

    selected_items = _receipt_items(contribution, key=key)
    selected_identities = {item.identity.sort_key() for item in selected_items}
    silent_count = 0
    omitted_visible_count = 0
    for index, candidate in enumerate(result.candidates):
        if candidate.record.identity.sort_key() in selected_identities:
            continue
        if (
            index >= admission_policy.max_evaluated_candidates
            or len(candidate.record.text.encode("utf-8"))
            > admission_policy.max_candidate_text_bytes
        ):
            omitted_visible_count += 1
            continue
        score = candidate.fused.score
        if score < admission_policy.minimum_offer_score or (
            score < admission_policy.minimum_inject_score and not admission_policy.mode.emits_offers
        ):
            silent_count += 1
        else:
            omitted_visible_count += 1

    fusion_omitted = result.fusion.omitted_candidate_count + result.omitted_by_result_bytes
    omitted_count = omitted_visible_count + fusion_omitted
    eligible_count = result.fusion.unique_candidate_count
    admitted_count = sum(item.admission is RecallItemAdmission.ADMITTED for item in selected_items)
    offered_count = len(selected_items) - admitted_count
    if eligible_count != admitted_count + offered_count + silent_count + omitted_count:
        raise ValueError("Recall receipt outcomes do not exhaust the fused candidate frontier.")

    result_payload = result.model_dump(mode="json")
    situation_payload = situation.model_dump(mode="json")
    source_configuration_payload = dict(source_configuration)
    access_scope_payload = situation_payload.get("knowledge_access_scope")
    source_coverage = _source_coverage(result)
    return RecallReceipt(
        receipt_id=new_recall_receipt_id(),
        session_id=session_id,
        interaction_id=interaction_id,
        model_step_id=model_step_id,
        created_at=datetime.now(UTC) if created_at is None else created_at,
        situation_fingerprint=_fingerprint(
            result.situation_sha256,
            KeyedEvidenceFingerprintDomain.SITUATION,
            key,
        ),
        engine_version=result.engine_version,
        source_configuration_fingerprint=_fingerprint_payload(
            source_configuration_payload,
            "recall source configuration",
            KeyedEvidenceFingerprintDomain.SOURCE_CONFIGURATION,
            key,
        ),
        admission_policy_fingerprint=_fingerprint(
            contribution.policy_sha256,
            KeyedEvidenceFingerprintDomain.ADMISSION_POLICY,
            key,
        ),
        access_scope_fingerprint=_fingerprint_payload(
            access_scope_payload,
            "recall access scope",
            KeyedEvidenceFingerprintDomain.ACCESS_SCOPE,
            key,
        ),
        frontier_fingerprint=_fingerprint_payload(
            result_payload,
            "recall fused frontier",
            KeyedEvidenceFingerprintDomain.FRONTIER,
            key,
        ),
        sources=source_coverage,
        inspected_count=sum(channel.hit_count for channel in result.fusion.channels),
        eligible_count=eligible_count,
        admitted_count=admitted_count,
        offered_count=offered_count,
        silent_count=silent_count,
        omitted_count=omitted_count,
        truncated=(
            omitted_count > 0
            or any(
                source.state is not RecallSourceCoverageState.COMPLETE for source in source_coverage
            )
        ),
        items=selected_items,
    )


async def persist_recall_receipt(
    *,
    store: SessionStore,
    receipt: RecallReceipt,
) -> RecallReceipt:
    """Persist one immutable receipt and reconcile a lost acknowledgement."""

    try:
        return await store.create_recall_receipt(receipt)
    except Exception as first_failure:
        try:
            return await store.create_recall_receipt(receipt)
        except Exception as replay_failure:
            try:
                current = await store.load_recall_receipt(
                    receipt.session_id,
                    receipt.receipt_id,
                )
            except Exception as readback_failure:
                first_failure.add_note(
                    "Recall-receipt replay and readback also failed: "
                    f"{type(replay_failure).__name__}, {type(readback_failure).__name__}."
                )
                raise first_failure from readback_failure
            if current is not None and current == receipt:
                return current
            first_failure.add_note(
                f"Recall-receipt replay also failed: {type(replay_failure).__name__}."
            )
            raise first_failure from replay_failure


def recall_receipt_document_sha256(receipt: RecallReceipt) -> str:
    """Fingerprint the exact safe receipt document linked by a frozen checkpoint."""

    if type(receipt) is not RecallReceipt:
        raise TypeError("receipt must be a RecallReceipt.")
    return hashlib.sha256(
        canonical_durable_json_bytes(
            receipt.model_dump(mode="json"),
            "automatic recall receipt document",
        )
    ).hexdigest()


def recall_receipt_manifest_binding_hmac_sha256(
    *,
    receipt_document_sha256: str,
    manifest_sha256: str | None,
    key: MemoryEvidenceKey,
) -> str:
    """Authenticate one frozen receipt-to-manifest relationship."""

    if (
        type(receipt_document_sha256) is not str
        or len(receipt_document_sha256) != 64
        or any(character not in "0123456789abcdef" for character in receipt_document_sha256)
    ):
        raise ValueError("receipt_document_sha256 must be a lowercase SHA-256 digest.")
    if manifest_sha256 is not None and (
        type(manifest_sha256) is not str
        or len(manifest_sha256) != 64
        or any(character not in "0123456789abcdef" for character in manifest_sha256)
    ):
        raise ValueError("manifest_sha256 must be a lowercase SHA-256 digest or None.")
    if type(key) is not MemoryEvidenceKey:
        raise TypeError("key must be a MemoryEvidenceKey.")
    material = canonical_durable_json_bytes(
        {
            "schema_version": 1,
            "key_id": key.key_id,
            "receipt_document_sha256": receipt_document_sha256,
            "manifest_sha256": manifest_sha256,
        },
        "automatic recall receipt-manifest binding",
    )
    return hmac.digest(
        key.key,
        _AUTOMATIC_RECALL_CHECKPOINT_BINDING_CONTEXT + b"\0" + material,
        "sha256",
    ).hex()


async def prepare_context_exposure(
    *,
    store: SessionStore,
    session_id: str,
    interaction_id: str,
    model_request: ModelRequest,
    request_fingerprint_sha256: str,
    provider_name: str,
    model_attempt_identity: ModelAttemptIdentity,
    execution_profile: ExecutionProfileIdentity,
    tool_exposure: ResolvedToolExposure,
    reference: MemoryEvidenceReference,
    key: MemoryEvidenceKey,
) -> ContextExposure:
    """Persist the planned and exact prepared states for one provider attempt."""

    if getattr(store, "supports_recall_evidence", False) is not True:
        raise RuntimeError("Automatic recall requires a recall-evidence-capable SessionStore.")
    if len(request_fingerprint_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in request_fingerprint_sha256
    ):
        raise ValueError("request_fingerprint_sha256 must be a lowercase SHA-256 digest.")
    receipt = await store.load_recall_receipt(session_id, reference.receipt_id)
    if receipt is None:
        raise RuntimeError("The automatic recall receipt is not durable.")
    if receipt.interaction_id != interaction_id:
        raise RuntimeError("The automatic recall receipt belongs to another interaction.")
    if receipt.situation_fingerprint.key_id != key.key_id:
        raise RuntimeError("The automatic recall receipt uses another evidence-key identity.")
    if not hmac.compare_digest(
        recall_receipt_document_sha256(receipt),
        reference.receipt_document_sha256,
    ):
        raise RuntimeError("The automatic recall checkpoint does not match its durable receipt.")
    if not hmac.compare_digest(
        recall_receipt_manifest_binding_hmac_sha256(
            receipt_document_sha256=reference.receipt_document_sha256,
            manifest_sha256=reference.manifest_sha256,
            key=key,
        ),
        reference.receipt_manifest_binding_hmac_sha256,
    ):
        raise RuntimeError(
            "The automatic recall checkpoint receipt-to-manifest binding is invalid."
        )
    include_items = _request_includes_exact_frozen_manifest(
        model_request,
        reference.manifest_sha256,
    )

    provider_attempt_id = new_provider_attempt_id()
    exposure_id = new_context_exposure_id()
    composition_sha256 = _payload_sha256(
        [message.model_dump(mode="json") for message in model_request.messages],
        "provider-neutral context composition",
    )
    tool_sha256 = _payload_sha256(
        {
            "tools": model_request.tools,
            "hosted_tools": [tool.model_dump(mode="json") for tool in model_request.hosted_tools],
            "resolved_exposure": tool_exposure.model_dump(mode="json"),
        },
        "provider tool exposure",
    )
    context_component = execution_profile.component(
        ExecutionProfileComponentClass.CONTEXT_SELECTION
    )
    context_sha256 = context_component.fingerprint or _payload_sha256(
        {
            "availability": context_component.availability.value,
            "strength": context_component.strength.value,
        },
        "context policy availability",
    )
    now = max(datetime.now(UTC), receipt.created_at)
    planned = ContextExposureTransition(
        transition_id=new_context_exposure_transition_id(),
        revision=0,
        state=ContextExposureState.PLANNED,
        occurred_at=now,
        evidence_kind=ContextExposureEvidenceKind.COMPOSITION_PLANNED,
        evidence_ref=f"exposure:{exposure_id}:composition",
    )
    exposure = ContextExposure(
        exposure_id=exposure_id,
        session_id=session_id,
        interaction_id=interaction_id,
        model_step_id=model_attempt_identity.model_step_id,
        model_attempt_id=model_attempt_identity.model_attempt_id,
        provider_attempt_id=provider_attempt_id,
        provider_name=provider_name,
        model_name=model_request.model,
        composition_fingerprint=_fingerprint(
            composition_sha256,
            KeyedEvidenceFingerprintDomain.COMPOSITION,
            key,
        ),
        execution_profile_fingerprint=_fingerprint(
            execution_profile.fingerprint,
            KeyedEvidenceFingerprintDomain.EXECUTION_PROFILE,
            key,
        ),
        context_policy_fingerprint=_fingerprint(
            context_sha256,
            KeyedEvidenceFingerprintDomain.CONTEXT_POLICY,
            key,
        ),
        tool_exposure_fingerprint=_fingerprint(
            tool_sha256,
            KeyedEvidenceFingerprintDomain.TOOL_EXPOSURE,
            key,
        ),
        request_contract_fingerprint=_fingerprint(
            request_fingerprint_sha256,
            KeyedEvidenceFingerprintDomain.REQUEST_CONTRACT,
            key,
        ),
        receipt_ids=(receipt.receipt_id,),
        contributor_ids=("automatic_recall",),
        created_at=now,
        updated_at=now,
        state=ContextExposureState.PLANNED,
        state_revision=0,
        transitions=(planned,),
    )
    item_exposures = (
        tuple(
            RecallItemExposure(
                exposure_id=exposure_id,
                receipt_id=receipt.receipt_id,
                ordinal=ordinal,
                receipt_item_ordinal=item.ordinal,
                identity=item.identity,
                representation_id=item.representation_id,
                content_sha256=item.content_sha256,
                locator=item.locator,
                admission=item.admission,
                selection_reason=item.selection_reason,
            )
            for ordinal, item in enumerate(receipt.items)
        )
        if include_items
        else ()
    )
    try:
        exposure = await _create_context_exposure(
            store=store,
            exposure=exposure,
            item_exposures=item_exposures,
        )
        return await transition_context_exposure(
            store=store,
            exposure=exposure,
            state=ContextExposureState.PREPARED,
            evidence_kind=ContextExposureEvidenceKind.REQUEST_PREPARED,
            evidence_ref=f"exposure:{exposure_id}:request",
        )
    except BaseException as preparation_failure:
        cancelled = isinstance(
            preparation_failure,
            (asyncio.CancelledError, GeneratorExit, SessionInterruptedByRequest),
        )
        try:
            durable = await store.load_context_exposure(session_id, exposure_id)
            if durable is not None and not durable.state.terminal:
                await transition_context_exposure(
                    store=store,
                    exposure=durable,
                    state=(
                        ContextExposureState.CANCELLED if cancelled else ContextExposureState.FAILED
                    ),
                    evidence_kind=(
                        ContextExposureEvidenceKind.CONCLUSIVE_CANCELLATION
                        if cancelled
                        else ContextExposureEvidenceKind.CONCLUSIVE_FAILURE
                    ),
                    evidence_ref=f"exposure:{exposure_id}:preparation-failed",
                )
        except BaseException as cleanup_failure:
            preparation_failure.add_note(
                "Context-exposure preparation cleanup also failed: "
                f"{type(cleanup_failure).__name__}."
            )
        raise


async def _create_context_exposure(
    *,
    store: SessionStore,
    exposure: ContextExposure,
    item_exposures: tuple[RecallItemExposure, ...],
) -> ContextExposure:
    try:
        return await store.create_context_exposure(exposure, item_exposures)
    except Exception as first_failure:
        try:
            return await store.create_context_exposure(exposure, item_exposures)
        except Exception as replay_failure:
            try:
                current = await store.load_context_exposure(
                    exposure.session_id,
                    exposure.exposure_id,
                )
                current_items = await store.load_recall_item_exposures(
                    exposure.session_id,
                    exposure.exposure_id,
                )
            except Exception as readback_failure:
                first_failure.add_note(
                    "Context-exposure creation replay and readback also failed: "
                    f"{type(replay_failure).__name__}, "
                    f"{type(readback_failure).__name__}."
                )
                raise first_failure from readback_failure
            if current is not None and current == exposure and current_items == item_exposures:
                return current
            first_failure.add_note(
                f"Context-exposure creation replay also failed: {type(replay_failure).__name__}."
            )
            raise first_failure from replay_failure


async def transition_context_exposure(
    *,
    store: SessionStore,
    exposure: ContextExposure,
    state: ContextExposureState,
    evidence_kind: ContextExposureEvidenceKind,
    evidence_ref: str,
    provider_request_id: str | None = None,
    transition_id: str | None = None,
    occurred_at: datetime | None = None,
) -> ContextExposure:
    transition_time = (
        max(datetime.now(UTC), exposure.updated_at) if occurred_at is None else occurred_at
    )
    request = ContextExposureTransitionRequest(
        transition_id=transition_id or new_context_exposure_transition_id(),
        expected_state=exposure.state,
        expected_revision=exposure.state_revision,
        state=state,
        occurred_at=transition_time,
        evidence_kind=evidence_kind,
        evidence_ref=evidence_ref,
        provider_request_id=provider_request_id,
    )
    try:
        return await store.transition_context_exposure(
            exposure.session_id,
            exposure.exposure_id,
            request,
        )
    except Exception as first_failure:
        try:
            return await store.transition_context_exposure(
                exposure.session_id,
                exposure.exposure_id,
                request,
            )
        except Exception as replay_failure:
            try:
                current = await store.load_context_exposure(
                    exposure.session_id,
                    exposure.exposure_id,
                )
            except Exception as readback_failure:
                first_failure.add_note(
                    "Context-exposure transition replay and readback also failed: "
                    f"{type(replay_failure).__name__}, "
                    f"{type(readback_failure).__name__}."
                )
                raise first_failure from readback_failure
            if current is not None and any(
                transition.transition_id == request.transition_id
                for transition in current.transitions
            ):
                return current
            first_failure.add_note(
                f"Context-exposure transition replay also failed: {type(replay_failure).__name__}."
            )
            raise first_failure from replay_failure


async def recover_context_exposure(
    *,
    store: SessionStore,
    session_id: str,
    stage_id: str,
    stage_intent: Mapping[str, Any],
    state: ContextExposureState,
    evidence_kind: ContextExposureEvidenceKind,
    evidence_ref: str,
    provider_request_id: str | None = None,
) -> ContextExposure | None:
    """Advance exposure evidence from one durable model-stage recovery fact."""

    exposure = await _load_stage_context_exposure(
        store=store,
        session_id=session_id,
        stage_intent=stage_intent,
    )
    if exposure is None:
        return None
    if exposure.state.terminal:
        # A model-completion stage records that publication finished, not that
        # the provider outcome was successful. The exposure's earlier terminal
        # transition is the more precise fact (for example, a typed provider
        # error published as a failed model completion), so recovery must never
        # rewrite or reject it.
        return exposure
    if exposure.state is ContextExposureState.PREPARED:
        exposure = await transition_context_exposure(
            store=store,
            exposure=exposure,
            state=ContextExposureState.DISPATCH_STARTED,
            evidence_kind=ContextExposureEvidenceKind.DISPATCH_INTENT_COMMITTED,
            evidence_ref=f"model-stage:{stage_id}",
        )
    if state in {ContextExposureState.ACKNOWLEDGED, ContextExposureState.COMPLETED} and (
        exposure.state is ContextExposureState.DISPATCH_STARTED
    ):
        exposure = await transition_context_exposure(
            store=store,
            exposure=exposure,
            state=ContextExposureState.ACKNOWLEDGED,
            evidence_kind=ContextExposureEvidenceKind.RECOVERY_ACKNOWLEDGEMENT,
            evidence_ref=evidence_ref,
            provider_request_id=provider_request_id,
        )
    if exposure.state is state:
        return exposure
    return await transition_context_exposure(
        store=store,
        exposure=exposure,
        state=state,
        evidence_kind=evidence_kind,
        evidence_ref=evidence_ref,
        provider_request_id=provider_request_id,
    )


async def close_unrecoverable_context_exposure(
    *,
    store: SessionStore,
    session_id: str,
    stage_id: str,
    stage_intent: Mapping[str, Any],
) -> ContextExposure | None:
    """Close an active synchronous stage that recovery cannot safely replay."""

    exposure = await _load_stage_context_exposure(
        store=store,
        session_id=session_id,
        stage_intent=stage_intent,
    )
    if exposure is None:
        return None
    if exposure.state.terminal:
        return exposure
    if exposure.state is ContextExposureState.PREPARED:
        exposure = await transition_context_exposure(
            store=store,
            exposure=exposure,
            state=ContextExposureState.DISPATCH_STARTED,
            evidence_kind=ContextExposureEvidenceKind.DISPATCH_INTENT_COMMITTED,
            evidence_ref=f"model-stage:{stage_id}:dispatch-receipt-recovered",
        )
    return await transition_context_exposure(
        store=store,
        exposure=exposure,
        state=ContextExposureState.INDETERMINATE,
        evidence_kind=ContextExposureEvidenceKind.RECOVERY_INDETERMINATE,
        evidence_ref=f"model-stage:{stage_id}:manual-recovery-required",
    )


async def close_context_exposure_without_provider_effect(
    *,
    store: SessionStore,
    session_id: str,
    stage_id: str,
    stage_intent: Mapping[str, Any],
    evidence_ref_suffix: str,
) -> ContextExposure | None:
    """Close exposure intent when runtime authority disproves provider effect."""

    exposure = await _load_stage_context_exposure(
        store=store,
        session_id=session_id,
        stage_intent=stage_intent,
    )
    if exposure is None:
        return None
    if exposure.state in {
        ContextExposureState.ACKNOWLEDGED,
        ContextExposureState.COMPLETED,
        ContextExposureState.INDETERMINATE,
    }:
        raise RuntimeError("Context exposure evidence contradicts the absence of provider effect.")
    if exposure.state.terminal:
        return exposure
    return await transition_context_exposure(
        store=store,
        exposure=exposure,
        state=ContextExposureState.FAILED,
        evidence_kind=ContextExposureEvidenceKind.CONCLUSIVE_FAILURE,
        evidence_ref=f"model-stage:{stage_id}:{evidence_ref_suffix}",
    )


async def _load_stage_context_exposure(
    *,
    store: SessionStore,
    session_id: str,
    stage_intent: Mapping[str, Any],
) -> ContextExposure | None:
    identity = stage_intent.get("context_exposure")
    if identity is None:
        return None
    if type(identity) is not dict or set(identity) != {
        "exposure_id",
        "provider_attempt_id",
    }:
        raise RuntimeError("Model-completion context-exposure identity is malformed.")
    exposure_id = identity.get("exposure_id")
    provider_attempt_id = identity.get("provider_attempt_id")
    if type(exposure_id) is not str or type(provider_attempt_id) is not str:
        raise RuntimeError("Model-completion context-exposure identity is malformed.")
    exposure = await store.load_context_exposure(session_id, exposure_id)
    if exposure is None:
        raise RuntimeError("Model-completion stage references missing context exposure evidence.")
    if exposure.provider_attempt_id != provider_attempt_id:
        raise RuntimeError("Model-completion stage changed its provider-attempt identity.")
    validate_context_exposure_stage_scope(exposure, stage_intent)
    return exposure


def memory_evidence_reference_from_checkpoint(
    checkpoint: dict[str, Any] | None,
) -> MemoryEvidenceReference | None:
    if type(checkpoint) is not dict:
        return None
    if "automatic_recall" not in checkpoint:
        return None
    state = checkpoint["automatic_recall"]
    if type(state) is not dict:
        raise ValueError("Automatic-recall memory-evidence checkpoint is malformed.")
    receipt_id = state.get("receipt_id")
    receipt_document_sha256 = state.get("receipt_document_sha256")
    receipt_manifest_binding_hmac_sha256 = state.get("receipt_manifest_binding_hmac_sha256")
    manifest_sha256 = state.get("manifest_sha256")
    if (
        type(receipt_id) is not str
        or type(receipt_document_sha256) is not str
        or type(receipt_manifest_binding_hmac_sha256) is not str
    ):
        raise ValueError("Automatic-recall memory-evidence checkpoint is malformed.")
    if manifest_sha256 is not None and type(manifest_sha256) is not str:
        raise ValueError("Automatic-recall memory-evidence checkpoint is malformed.")
    return MemoryEvidenceReference(
        receipt_id=receipt_id,
        receipt_document_sha256=receipt_document_sha256,
        receipt_manifest_binding_hmac_sha256=receipt_manifest_binding_hmac_sha256,
        manifest_sha256=manifest_sha256,
    )


def context_exposure_identity_payload(exposure: ContextExposure) -> dict[str, str]:
    return {
        "exposure_id": exposure.exposure_id,
        "provider_attempt_id": exposure.provider_attempt_id,
    }


def validate_context_exposure_stage_scope(
    exposure: ContextExposure,
    stage_intent: Mapping[str, Any],
) -> None:
    """Require one loaded exposure to belong to its immutable model-stage intent."""

    if type(exposure) is not ContextExposure:
        raise TypeError("exposure must be a ContextExposure.")
    expected = {
        "model_step_id": exposure.model_step_id,
        "model_attempt_id": exposure.model_attempt_id,
        "provider_name": exposure.provider_name,
        "requested_model": exposure.model_name,
    }
    if any(stage_intent.get(field_name) != value for field_name, value in expected.items()):
        raise RuntimeError("Model-completion stage changed its context exposure scope.")
    interaction_id = stage_intent.get("interaction_id")
    if interaction_id is not None and exposure.interaction_id != interaction_id:
        raise RuntimeError("Model-completion stage changed its context interaction identity.")


def _source_coverage(result: RecallResult) -> tuple[RecallSourceCoverage, ...]:
    channel_by_name = {channel.channel: channel for channel in result.fusion.channels}
    coverage: list[RecallSourceCoverage] = []
    for diagnostic in result.sources:
        channels = tuple(channel_by_name[name] for name in diagnostic.channels)
        channel_truncated = any(channel.truncated for channel in channels)
        if diagnostic.status is RecallSourceStatus.COMPLETE and not channel_truncated:
            state = RecallSourceCoverageState.COMPLETE
            failure_code = None
        elif diagnostic.status is RecallSourceStatus.UNAVAILABLE:
            state = RecallSourceCoverageState.UNAVAILABLE
            failure_code = diagnostic.failure_code or "unavailable"
        else:
            state = RecallSourceCoverageState.PARTIAL
            failure_code = diagnostic.failure_code or "candidate_limit"
        coverage.append(
            RecallSourceCoverage(
                source=diagnostic.source,
                required=diagnostic.required,
                channels=diagnostic.channels,
                state=state,
                failure_code=failure_code,
                inspected_count=sum(channel.hit_count for channel in channels),
                candidate_limit=sum(channel.candidate_limit for channel in channels),
                truncated=state is not RecallSourceCoverageState.COMPLETE,
                continuation_available=any(channel.continuation_available for channel in channels),
            )
        )
    return tuple(coverage)


def _receipt_items(
    contribution: AutomaticRecallContribution,
    *,
    key: MemoryEvidenceKey,
) -> tuple[RecallReceiptItem, ...]:
    material: list[tuple[int, Any, RecallItemAdmission, str]] = []
    if contribution.focus is not None:
        material.extend(
            (
                item.fused_rank,
                item.candidate,
                RecallItemAdmission.ADMITTED,
                item.selection_reason,
            )
            for item in contribution.focus.items
        )
    if contribution.offer is not None:
        material.extend(
            (item.fused_rank, item, RecallItemAdmission.OFFERED, item.reason)
            for item in contribution.offer.items
        )
    items: list[RecallReceiptItem] = []
    for ordinal, (fused_rank, selected, admission, reason) in enumerate(
        sorted(material, key=lambda entry: entry[0])
    ):
        if admission is RecallItemAdmission.ADMITTED:
            record = selected.record
            identity = record.identity
            representation = record.representation
            content_hash = record.content_hash
            locator = record.locator
            matches = selected.fused.matches
        else:
            identity = selected.identity
            representation = selected.representation
            content_hash = selected.content_hash
            locator = selected.locator
            matches = selected.matches
        items.append(
            RecallReceiptItem(
                ordinal=ordinal,
                identity=identity,
                representation_id=representation,
                content_sha256=content_hash,
                locator=_evidence_locator(
                    identity.record_type,
                    locator,
                    key=key,
                ),
                admission=admission,
                selection_reason=RecallItemSelectionReason(reason),
                fused_rank=fused_rank,
                match_channels=tuple(match.channel for match in matches),
            )
        )
    return tuple(items)


def _evidence_locator(
    record_type: str,
    locator: Mapping[str, Any],
    *,
    key: MemoryEvidenceKey,
) -> RecallEvidenceLocator:
    payload = dict(locator)
    if record_type == "knowledge_entry":
        return KnowledgeEntryEvidenceLocator(
            entry_id=payload["entry_id"],
            entry_revision=payload["entry_revision"],
        )
    if record_type == "knowledge_chunk":
        return KnowledgeChunkEvidenceLocator(
            entry_id=payload["entry_id"],
            entry_revision=payload["entry_revision"],
            chunk_id=payload["chunk_id"],
            chunk_index=payload["chunk_index"],
        )
    if record_type == "transcript_message":
        return TranscriptMessageEvidenceLocator(
            session_id=payload["session_id"],
            interaction_id=payload.get("interaction_id"),
            transcript_index=payload["transcript_index"],
            text_part_indexes=tuple(payload["text_part_indexes"]),
        )
    return OpaqueRecallEvidenceLocator(
        fingerprint=_fingerprint_payload(
            payload,
            "recall source locator",
            KeyedEvidenceFingerprintDomain.SOURCE_LOCATOR,
            key,
        )
    )


def _request_manifest_count(request: ModelRequest, manifest_sha256: str) -> int:
    from cayu.core.messages import TextPart

    return sum(
        hashlib.sha256(part.text.encode("utf-8")).hexdigest() == manifest_sha256
        for message in request.messages
        for part in message.content
        if type(part) is TextPart
    )


def _request_includes_exact_frozen_manifest(
    request: ModelRequest,
    manifest_sha256: str | None,
) -> bool:
    reserved_envelope_count = _request_automatic_recall_envelope_count(request)
    if manifest_sha256 is None:
        if reserved_envelope_count:
            raise RuntimeError("The frozen recall manifest changed before provider dispatch.")
        return False
    exact_manifest_count = _request_manifest_count(request, manifest_sha256)
    if exact_manifest_count > 1 or reserved_envelope_count != exact_manifest_count:
        raise RuntimeError("The frozen recall manifest changed before provider dispatch.")
    return exact_manifest_count == 1


def _request_automatic_recall_envelope_count(request: ModelRequest) -> int:
    from cayu.core.messages import TextPart

    return sum(
        _AUTOMATIC_RECALL_OPEN_TAG in part.text or _AUTOMATIC_RECALL_CLOSE_TAG in part.text
        for message in request.messages
        for part in message.content
        if type(part) is TextPart
    )


def _fingerprint_payload(
    payload: Any,
    field_name: str,
    domain: KeyedEvidenceFingerprintDomain,
    key: MemoryEvidenceKey,
):
    return _fingerprint(_payload_sha256(payload, field_name), domain, key)


def _payload_sha256(payload: Any, field_name: str) -> str:
    return hashlib.sha256(canonical_durable_json_bytes(payload, field_name)).hexdigest()


def _fingerprint(
    source_sha256: str,
    domain: KeyedEvidenceFingerprintDomain,
    key: MemoryEvidenceKey,
):
    return keyed_evidence_fingerprint(
        source_sha256,
        domain=domain,
        key_id=key.key_id,
        key=key.key,
    )


__all__ = [
    "MemoryEvidenceKey",
    "MemoryEvidenceReference",
    "active_memory_evidence_key",
    "build_recall_receipt",
    "close_context_exposure_without_provider_effect",
    "close_unrecoverable_context_exposure",
    "context_exposure_identity_payload",
    "memory_evidence_key",
    "memory_evidence_key_scope",
    "memory_evidence_reference_from_checkpoint",
    "persist_recall_receipt",
    "prepare_context_exposure",
    "recall_receipt_document_sha256",
    "recall_receipt_manifest_binding_hmac_sha256",
    "recover_context_exposure",
    "transition_context_exposure",
    "validate_context_exposure_stage_scope",
]
