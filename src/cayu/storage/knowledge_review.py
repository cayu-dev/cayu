from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

from cayu._validation import (
    copy_label_map,
)
from cayu._validation import (
    require_durable_clean_nonblank as require_clean_nonblank,
)
from cayu.storage.memory import (
    DEFAULT_KNOWLEDGE_LIMIT,
    DEFAULT_KNOWLEDGE_MAX_BYTES,
    MAX_KNOWLEDGE_ACTIVATION_CHUNKS,
    MAX_KNOWLEDGE_ACTIVATION_EVIDENCE_RECORDS,
    MAX_KNOWLEDGE_ACTIVATION_REQUEST_BYTES,
    KnowledgeAccessScope,
    KnowledgeActivationAuthority,
    KnowledgeActivationConflict,
    KnowledgeActivationReceipt,
    KnowledgeActivationSource,
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeEvidenceResult,
    KnowledgeGovernanceMode,
    KnowledgeListQuery,
    KnowledgeListResult,
    KnowledgeMaintenanceDecision,
    KnowledgeMaintenanceDecisionKind,
    KnowledgeMaintenanceDecisionReceipt,
    KnowledgeMaintenanceProposal,
    KnowledgePublicationReceipt,
    KnowledgeReviewApproval,
    KnowledgeStatus,
    KnowledgeVisibility,
    _replay_review_approval_from_receipts,
    copy_knowledge_access_scope,
    copy_knowledge_activation_authority,
    copy_knowledge_activation_receipt,
    prepare_knowledge_activation_request,
)

_KNOWLEDGE_REVIEW_STORE_METHODS = (
    "approve_pending_entry",
    "get_entry",
    "read_chunks",
    "read_evidence",
    "transition_entry_status",
    "list_entries",
    "load_entry_publication_receipt",
    "load_activation_receipt",
)
_CURATOR_FORBIDDEN_REVIEW_IDENTITY_FIELDS = (
    "candidate_generator_identity",
    "evaluator_identity",
    "policy_identity",
)


def _review_forbidden_activation_identities(entry: KnowledgeEntry) -> tuple[str, ...]:
    identities = [entry.created_by]
    curator_audit = entry.metadata.get("cayu_curator")
    if type(curator_audit) is dict:
        for field_name in _CURATOR_FORBIDDEN_REVIEW_IDENTITY_FIELDS:
            value = curator_audit.get(field_name)
            if type(value) is str and value.strip() and value not in identities:
                identities.append(value)
    return tuple(identities)


class _KnowledgeReviewStore(Protocol):
    async def get_entry(
        self,
        entry_id: str,
        *,
        revision: int | None = None,
        max_bytes: int | None = None,
        access_scope: KnowledgeAccessScope,
    ) -> KnowledgeEntry | None: ...

    async def transition_entry_status(
        self,
        entry_id: str,
        *,
        expected_revision: int,
        access_scope: KnowledgeAccessScope,
        from_status: KnowledgeStatus,
        to_status: KnowledgeStatus,
        expected_namespace: str | None = None,
        expected_labels: dict[str, str] | None = None,
    ) -> KnowledgeEntry: ...

    async def read_chunks(
        self,
        entry_id: str,
        *,
        revision: int | None = None,
        access_scope: KnowledgeAccessScope,
        max_chunks: int,
        max_bytes: int,
    ) -> list[KnowledgeChunk]: ...

    async def read_evidence(
        self,
        entry_id: str,
        *,
        revision: int | None = None,
        access_scope: KnowledgeAccessScope,
        max_records: int,
        max_bytes: int,
    ) -> KnowledgeEvidenceResult | None: ...

    async def approve_pending_entry(
        self,
        authority: KnowledgeActivationAuthority,
        *,
        access_scope: KnowledgeAccessScope,
        expected_namespace: str | None = None,
        expected_labels: dict[str, str] | None = None,
    ) -> KnowledgeReviewApproval: ...

    async def load_activation_receipt(
        self,
        operation_id: str,
        *,
        access_scope: KnowledgeAccessScope,
    ) -> KnowledgeActivationReceipt | None: ...

    async def load_entry_publication_receipt(
        self,
        operation_id: str,
        *,
        access_scope: KnowledgeAccessScope,
    ) -> KnowledgePublicationReceipt | None: ...

    async def list_entries(
        self,
        query: KnowledgeListQuery,
        *,
        access_scope: KnowledgeAccessScope,
    ) -> KnowledgeListResult: ...

    async def apply_maintenance_decision(
        self,
        proposal: KnowledgeMaintenanceProposal,
        decision: KnowledgeMaintenanceDecision,
        *,
        access_scope: KnowledgeAccessScope,
    ) -> KnowledgeMaintenanceDecisionReceipt: ...


class KnowledgeReviewWorkflow:
    """App-side workflow for reviewing model-authored pending knowledge."""

    def __init__(
        self,
        store: _KnowledgeReviewStore,
        *,
        access_scope: KnowledgeAccessScope | None = None,
        namespace: str | None = None,
        labels: dict[str, str] | None = None,
        default_limit: int = DEFAULT_KNOWLEDGE_LIMIT,
        default_max_bytes: int = DEFAULT_KNOWLEDGE_MAX_BYTES,
    ) -> None:
        _validate_review_store(store)
        self.store = store
        if access_scope is None:
            bound_scope = getattr(store, "bound_access_scope", None)
            if callable(bound_scope):
                access_scope = bound_scope()
        if access_scope is None:
            raise ValueError("KnowledgeReviewWorkflow requires an access scope.")
        self.access_scope = copy_knowledge_access_scope(access_scope)
        self.namespace = (
            require_clean_nonblank(namespace, "namespace") if namespace is not None else None
        )
        self.labels = copy_label_map(labels or {}, "labels")
        self.default_limit = _positive_int(default_limit, "default_limit")
        self.default_max_bytes = _positive_int(default_max_bytes, "default_max_bytes")

    async def list_pending(
        self,
        *,
        namespace: str | None = None,
        labels: dict[str, str] | None = None,
        kinds: Iterable[str] | None = None,
        visibilities: Iterable[KnowledgeVisibility] | None = None,
        aspects: Iterable[str] | None = None,
        source_type: str | None = None,
        source_id: str | None = None,
        include_expired: bool = False,
        limit: int | None = None,
        max_bytes: int | None = None,
    ) -> KnowledgeListResult:
        """List pending entries within the workflow scope."""

        query = KnowledgeListQuery(
            namespace=self._scoped_namespace(namespace),
            labels=self._scoped_labels(labels),
            kinds=_string_list(kinds, "kinds") if kinds is not None else None,
            statuses=[KnowledgeStatus.PENDING],
            visibilities=list(visibilities) if visibilities is not None else None,
            aspects=_string_list(aspects, "aspects"),
            source_type=source_type,
            source_id=source_id,
            include_expired=include_expired,
            limit=self.default_limit if limit is None else limit,
            max_bytes=self.default_max_bytes if max_bytes is None else max_bytes,
        )
        return await self.store.list_entries(query, access_scope=self.access_scope)

    async def get_pending(self, entry_id: str) -> KnowledgeEntry:
        """Load one pending entry after status and scope checks."""

        return await self._require_pending_entry(entry_id)

    async def approve(
        self,
        entry_id: str,
        *,
        operation_id: str,
        reviewer_identity: str,
        reviewer_version: str,
        code: str = "approved",
        annotations: dict[str, object] | None = None,
    ) -> KnowledgeReviewApproval:
        """Approve one exact pending revision with durable reviewer attribution."""

        from cayu.knowledge_governance import reviewed_approval_authority

        clean_entry_id = require_clean_nonblank(entry_id, "entry_id")
        raw_prior = await self.store.load_activation_receipt(
            operation_id,
            access_scope=self.access_scope,
        )
        if raw_prior is not None:
            try:
                prior = copy_knowledge_activation_receipt(raw_prior)
                prior_request = prior.authority.request
                if (
                    prior.entry_id != clean_entry_id
                    or prior_request.mode is not KnowledgeGovernanceMode.REVIEWED
                    or prior_request.source is not KnowledgeActivationSource.REVIEW_APPROVAL
                    or prior_request.forbidden_authority_identities
                    != _review_forbidden_activation_identities(prior_request.candidate_entry)
                ):
                    raise ValueError("Operation is not the requested reviewed approval.")
                self._require_entry_in_scope(prior_request.candidate_entry)
                replay_authority = reviewed_approval_authority(
                    prior_request,
                    reviewer_identity=reviewer_identity,
                    reviewer_version=reviewer_version,
                    code=code,
                    annotations=annotations,
                )
            except (TypeError, ValueError):
                raise KnowledgeActivationConflict("operation_mismatch") from None
            if replay_authority != prior.authority:
                raise KnowledgeActivationConflict("operation_mismatch")
            return await self._approve_with_exact_authority(replay_authority)
        entry = await self._require_pending_entry(clean_entry_id)
        chunks = await self.store.read_chunks(
            entry.id,
            revision=entry.revision,
            access_scope=self.access_scope,
            max_chunks=MAX_KNOWLEDGE_ACTIVATION_CHUNKS,
            max_bytes=MAX_KNOWLEDGE_ACTIVATION_REQUEST_BYTES,
        )
        evidence_result = await self.store.read_evidence(
            entry.id,
            revision=entry.revision,
            access_scope=self.access_scope,
            max_records=MAX_KNOWLEDGE_ACTIVATION_EVIDENCE_RECORDS,
            max_bytes=MAX_KNOWLEDGE_ACTIVATION_REQUEST_BYTES,
        )
        if evidence_result is None:
            raise RuntimeError("Pending knowledge evidence disappeared during review.")
        if evidence_result.truncated:
            raise ValueError("Pending knowledge evidence exceeds the activation request bound.")
        request = prepare_knowledge_activation_request(
            entry,
            chunks,
            evidence=evidence_result.evidence,
            access_scope=self.access_scope,
            operation_id=operation_id,
            governance_mode=KnowledgeGovernanceMode.REVIEWED,
            source=KnowledgeActivationSource.REVIEW_APPROVAL,
            expected_revision=entry.revision,
            forbidden_authority_identities=_review_forbidden_activation_identities(entry),
        )
        authority = reviewed_approval_authority(
            request,
            reviewer_identity=reviewer_identity,
            reviewer_version=reviewer_version,
            code=code,
            annotations=annotations,
        )
        return await self._approve_with_exact_authority(authority)

    async def _approve_with_exact_authority(
        self,
        authority: KnowledgeActivationAuthority,
    ) -> KnowledgeReviewApproval:
        """Keep extension mutation from redefining reviewed attribution."""

        expected_authority = copy_knowledge_activation_authority(authority)
        raw_approval = await self.store.approve_pending_entry(
            copy_knowledge_activation_authority(expected_authority),
            access_scope=copy_knowledge_access_scope(self.access_scope),
            expected_namespace=self.namespace,
            expected_labels=dict(self.labels),
        )
        try:
            if type(raw_approval) is not KnowledgeReviewApproval:
                raise TypeError("Review store returned an invalid approval.")
            approval = KnowledgeReviewApproval.model_validate(
                raw_approval.model_dump(mode="python")
            )
            if approval.receipt.authority != expected_authority:
                raise ValueError("Review store changed activation authority.")
        except (TypeError, ValueError):
            raise KnowledgeActivationConflict("operation_mismatch") from None
        self._require_entry_in_scope(approval.entry)

        try:
            durable_activation = await self.store.load_activation_receipt(
                expected_authority.request.operation_id,
                access_scope=copy_knowledge_access_scope(self.access_scope),
            )
            durable_publication = await self.store.load_entry_publication_receipt(
                expected_authority.request.operation_id,
                access_scope=copy_knowledge_access_scope(self.access_scope),
            )
            if durable_activation is None or durable_publication is None:
                raise ValueError("Review activation receipt is missing.")
            expected_approval = _replay_review_approval_from_receipts(
                durable_publication,
                durable_activation,
                authority=expected_authority,
            )
            if expected_approval is None:
                raise ValueError("Review receipts do not bind the requested approval.")
            durable_receipt = copy_knowledge_activation_receipt(
                expected_approval.receipt,
                replayed=False,
            )
            returned_receipt = copy_knowledge_activation_receipt(
                approval.receipt,
                replayed=False,
            )
        except (TypeError, ValueError):
            raise KnowledgeActivationConflict("operation_mismatch") from None
        if durable_receipt != returned_receipt or expected_approval.entry != approval.entry:
            raise KnowledgeActivationConflict("operation_mismatch")
        return approval

    async def reject(self, entry_id: str) -> KnowledgeEntry:
        """Reject one pending entry while retaining it for audit."""

        entry = await self._require_pending_entry(entry_id)
        return await self.store.transition_entry_status(
            entry.id,
            expected_revision=entry.revision,
            access_scope=self.access_scope,
            from_status=KnowledgeStatus.PENDING,
            to_status=KnowledgeStatus.ARCHIVED,
            expected_namespace=self.namespace,
            expected_labels=self.labels,
        )

    async def decide_maintenance(
        self,
        proposal: KnowledgeMaintenanceProposal,
        decision: KnowledgeMaintenanceDecision,
    ) -> KnowledgeMaintenanceDecisionReceipt:
        """Apply one exact multi-entry review through the store transaction."""

        apply = getattr(self.store, "apply_maintenance_decision", None)
        if not callable(apply):
            raise TypeError("store must implement reviewed knowledge maintenance.")
        if type(proposal) is not KnowledgeMaintenanceProposal:
            raise TypeError("proposal must be a KnowledgeMaintenanceProposal.")
        if type(decision) is not KnowledgeMaintenanceDecision:
            raise TypeError("decision must be a KnowledgeMaintenanceDecision.")
        references = [proposal.replacement]
        if decision.kind is KnowledgeMaintenanceDecisionKind.APPROVE:
            references.extend(proposal.sources)
        for reference in references:
            entry = await self.store.get_entry(
                reference.entry_id,
                revision=reference.revision,
                access_scope=self.access_scope,
            )
            if entry is None:
                raise KeyError("A reviewed maintenance entry revision is unavailable.")
            self._require_entry_in_scope(entry)
        return await apply(
            proposal,
            decision,
            access_scope=self.access_scope,
        )

    async def _require_pending_entry(self, entry_id: str) -> KnowledgeEntry:
        clean_id = require_clean_nonblank(entry_id, "entry_id")
        entry = await self.store.get_entry(clean_id, access_scope=self.access_scope)
        if entry is None:
            raise KeyError(f"Knowledge entry {clean_id!r} does not exist.")
        self._require_entry_in_scope(entry)
        if entry.status is not KnowledgeStatus.PENDING:
            raise ValueError(
                f"Knowledge entry {clean_id!r} is {entry.status.value!r}, not 'pending'."
            )
        return entry

    def _require_entry_in_scope(self, entry: KnowledgeEntry) -> None:
        if self.namespace is not None and entry.namespace != self.namespace:
            raise PermissionError(
                f"Knowledge entry {entry.id!r} is outside review namespace {self.namespace!r}."
            )
        for key, value in self.labels.items():
            if entry.labels.get(key) != value:
                raise PermissionError(
                    f"Knowledge entry {entry.id!r} is outside review label {key}={value!r}."
                )

    def _scoped_namespace(self, namespace: str | None) -> str | None:
        if namespace is None:
            return self.namespace
        clean_namespace = require_clean_nonblank(namespace, "namespace")
        if self.namespace is not None and clean_namespace != self.namespace:
            raise ValueError(
                f"namespace {clean_namespace!r} conflicts with review namespace {self.namespace!r}."
            )
        return clean_namespace

    def _scoped_labels(self, labels: dict[str, str] | None) -> dict[str, str]:
        scoped = dict(self.labels)
        extra = copy_label_map(labels or {}, "labels")
        for key, value in extra.items():
            if key in scoped and scoped[key] != value:
                raise ValueError(
                    f"label {key}={value!r} conflicts with review label {key}={scoped[key]!r}."
                )
            scoped[key] = value
        return scoped


def _validate_review_store(store: Any) -> None:
    for method_name in _KNOWLEDGE_REVIEW_STORE_METHODS:
        if not callable(getattr(store, method_name, None)):
            raise TypeError("store must implement the knowledge review store methods.")


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or type(value) is not int:
        raise ValueError(f"`{name}` must be an integer.")
    if value <= 0:
        raise ValueError(f"`{name}` must be greater than 0.")
    return value


def _string_list(value: Iterable[str] | None, name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str | bytes):
        raise ValueError(f"`{name}` must be an iterable of strings.")
    result: list[str] = []
    for index, item in enumerate(value):
        if type(item) is not str:
            raise ValueError(f"`{name}[{index}]` must be a string.")
        result.append(require_clean_nonblank(item, f"{name}[{index}]"))
    return list(dict.fromkeys(result))
