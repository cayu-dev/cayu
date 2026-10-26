"""Validate an internal addition handoff before any store transaction begins."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from cayu.evals.corpus import EvalCorpusDocument, EvalCorpusInspectionV1
from cayu.evals.read_views import EvalSuiteViewIdentity
from cayu.evals.results import CapturedEvaluationResultV1
from cayu.evals.session_additions import (
    EvalAdditionConflict,
    EvalAdditionRecord,
    PreparedEvalAddition,
)
from cayu.evals.store import (
    EvalCaseCatalogEntry,
    EvalSuiteCatalogEntry,
    _prepare_authored_suite_for_store,
    _prepare_captured_result_for_store,
    _prepare_corpus_catalog_for_store,
)
from cayu.evals.suite_authoring import EvalSuiteDocument


@dataclass(frozen=True)
class AdditionDocuments:
    authored: tuple[EvalSuiteDocument, bytes] | None
    corpus: (
        tuple[
            EvalCorpusDocument,
            bytes,
            EvalCorpusInspectionV1,
            tuple[EvalSuiteCatalogEntry, ...],
            tuple[EvalCaseCatalogEntry, ...],
        ]
        | None
    )
    captured: tuple[CapturedEvaluationResultV1, bytes] | None
    record: EvalAdditionRecord

    @property
    def inherited_membership_scopes(
        self,
    ) -> tuple[tuple[str, str, frozenset[tuple[str, str]]], ...]:
        """Bounded parent/child scopes; portable source fields never authorize links."""
        plan = self.record.receipt.plan
        parent = plan.parent
        if plan.mode != "case" or parent is None or parent == plan.destination:
            return ()
        if self.authored is not None:
            members = frozenset((case.id, case.revision) for case in self.authored[0].cases)
            return ((parent.id, plan.destination.id, members),)
        if self.corpus is not None:
            corpus = self.corpus[0]
            return tuple(
                (
                    EvalSuiteViewIdentity(
                        source="corpus",
                        target_key=parent.target_key,
                        revision=parent.revision,
                        suite_id=suite.id,
                    ).id,
                    EvalSuiteViewIdentity(
                        source="corpus",
                        target_key=corpus.target_key,
                        revision=corpus.revision,
                        suite_id=suite.id,
                    ).id,
                    frozenset(
                        (case.id, case.revision)
                        for case in corpus.cases
                        if case.suite_id == suite.id
                    ),
                )
                for suite in corpus.suites
            )
        return ()


def prepare_addition_documents(
    prepared: PreparedEvalAddition,
    record: EvalAdditionRecord,
    *,
    redact_json: Callable[[Any], Any],
) -> AdditionDocuments:
    if type(prepared) is not PreparedEvalAddition or type(record) is not EvalAdditionRecord:
        raise TypeError("Expected an internal prepared addition and exact record.")
    record = EvalAdditionRecord.model_validate_json(record.model_dump_json())
    plan = record.receipt.plan
    if (
        plan != prepared.plan
        or plan.source_revision != prepared.source.revision
        or record.private_session_id != prepared.private_session_id
        or plan.destination.target_key != prepared.source.target_key
    ):
        raise EvalAdditionConflict("Addition authority does not match its publication.")
    authored = (
        None
        if prepared.authored is None
        else _prepare_authored_suite_for_store(prepared.authored, redact_json=redact_json)
    )
    corpus = (
        None
        if prepared.corpus is None
        else _prepare_corpus_catalog_for_store(prepared.corpus, redact_json=redact_json)
    )
    captured = None
    if prepared.captured is not None:
        if corpus is None:
            raise EvalAdditionConflict("Captured publication requires its execution corpus.")
        captured = _prepare_captured_result_for_store(
            prepared.captured, corpus[0], redact_json=redact_json
        )
    if plan.mode == "score":
        if (
            captured is None
            or authored is not None
            or captured[0].revision != plan.captured_result_revision
        ):
            raise EvalAdditionConflict("Scored addition requires its exact captured result.")
    else:
        document = authored[0] if authored is not None else None if corpus is None else corpus[0]
        if (
            captured is not None
            or document is None
            or document.revision != plan.destination.revision
        ):
            raise EvalAdditionConflict("Case addition requires its exact destination document.")
        if not any(
            item.id == plan.case_id and item.revision == plan.case_revision
            for item in document.cases
        ):
            raise EvalAdditionConflict("Published membership does not contain the planned case.")
    return AdditionDocuments(authored, corpus, captured, record)


def replay_addition(
    existing: EvalAdditionRecord, expected: EvalAdditionRecord
) -> EvalAdditionRecord:
    if (
        existing.receipt.operation_id != expected.receipt.operation_id
        or existing.request_material != expected.request_material
        or existing.receipt.request_revision != expected.receipt.request_revision
        or existing.subject != expected.subject
        or existing.tenant != expected.tenant
        or existing.public_session_id != expected.public_session_id
        or existing.private_session_id != expected.private_session_id
        or existing.receipt.plan != expected.receipt.plan
    ):
        raise EvalAdditionConflict("The operation key is already bound to another addition.")
    return existing
