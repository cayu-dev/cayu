from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from cayu._validation import require_durable_clean_nonblank

KNOWLEDGE_REVISION_RESET_POLICY_VERSION = "cayu.knowledge_revision_reset.v1"


class KnowledgeRevisionTransitionAction(StrEnum):
    INITIALIZE = "initialize"
    REFUSE_RESET_REQUIRED = "refuse_reset_required"


class KnowledgeRevisionTransitionAssessment(BaseModel):
    """Content-free evidence for the clean revision-schema transition decision."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    policy_version: Literal["cayu.knowledge_revision_reset.v1"] = (
        KNOWLEDGE_REVISION_RESET_POLICY_VERSION
    )
    inspected_tables: tuple[str, ...]
    populated_tables: tuple[str, ...]
    action: KnowledgeRevisionTransitionAction

    @field_validator("inspected_tables", "populated_tables", mode="before")
    @classmethod
    def validate_tables(cls, value, info) -> tuple[str, ...]:
        return tuple(_knowledge_table_names(value, field_name=info.field_name))

    @model_validator(mode="after")
    def validate_decision(self) -> KnowledgeRevisionTransitionAssessment:
        if not set(self.populated_tables).issubset(self.inspected_tables):
            raise ValueError("populated_tables must be a subset of inspected_tables.")
        expected = (
            KnowledgeRevisionTransitionAction.REFUSE_RESET_REQUIRED
            if self.populated_tables
            else KnowledgeRevisionTransitionAction.INITIALIZE
        )
        if self.action is not expected:
            raise ValueError("action must match the inspected table population evidence.")
        return self


class KnowledgeRevisionResetRequired(RuntimeError):
    """A populated mutable-knowledge database cannot be upgraded in place."""

    def __init__(self, assessment: KnowledgeRevisionTransitionAssessment) -> None:
        if type(assessment) is not KnowledgeRevisionTransitionAssessment:
            raise TypeError("assessment must be a KnowledgeRevisionTransitionAssessment.")
        self.assessment = KnowledgeRevisionTransitionAssessment(
            inspected_tables=tuple(assessment.inspected_tables),
            populated_tables=tuple(assessment.populated_tables),
            action=assessment.action,
        )
        tables = ", ".join(self.assessment.populated_tables)
        super().__init__(
            "The revision-first knowledge schema requires a database replacement/reset; "
            f"legacy knowledge tables are populated: {tables}. Cayu did not backfill, "
            "delete, or modify that knowledge."
        )


def assess_knowledge_revision_transition(
    row_counts: Mapping[str, int],
    *,
    required_tables: Iterable[str],
) -> KnowledgeRevisionTransitionAssessment:
    """Classify a complete, backend-owned legacy knowledge population inspection.

    Each backend migration supplies its exact legacy table set. Requiring an
    exact match prevents an omitted table from silently turning an incomplete
    inspection into approval.
    """

    if not isinstance(row_counts, Mapping):
        raise TypeError("row_counts must be a mapping.")
    inspected = _knowledge_table_names(row_counts.keys(), field_name="row_counts")
    required = _knowledge_table_names(required_tables, field_name="required_tables")
    if not required:
        raise ValueError("required_tables must not be empty.")
    if set(inspected) != set(required):
        missing = sorted(set(required) - set(inspected))
        unexpected = sorted(set(inspected) - set(required))
        raise ValueError(
            "Knowledge transition inspection must exactly cover required tables "
            f"(missing={missing!r}, unexpected={unexpected!r})."
        )
    populated: list[str] = []
    for table in inspected:
        count = row_counts[table]
        if type(count) is not int or count < 0:
            raise ValueError(f"Row count for {table!r} must be a non-negative integer.")
        if count:
            populated.append(table)
    return KnowledgeRevisionTransitionAssessment(
        inspected_tables=tuple(inspected),
        populated_tables=tuple(populated),
        action=(
            KnowledgeRevisionTransitionAction.REFUSE_RESET_REQUIRED
            if populated
            else KnowledgeRevisionTransitionAction.INITIALIZE
        ),
    )


def require_empty_knowledge_revision_transition(
    row_counts: Mapping[str, int],
    *,
    required_tables: Iterable[str],
) -> KnowledgeRevisionTransitionAssessment:
    """Allow fresh/empty initialization and fail closed for populated legacy data."""

    assessment = assess_knowledge_revision_transition(
        row_counts,
        required_tables=required_tables,
    )
    if assessment.action is KnowledgeRevisionTransitionAction.REFUSE_RESET_REQUIRED:
        raise KnowledgeRevisionResetRequired(assessment)
    return assessment


def _knowledge_table_names(values: Iterable[str], *, field_name: str) -> list[str]:
    try:
        copied = list(values)
    except TypeError:
        raise TypeError(f"{field_name} must be iterable.") from None
    result: list[str] = []
    for value in copied:
        if type(value) is not str:
            raise ValueError(f"{field_name} must contain strings.")
        table = require_durable_clean_nonblank(value, field_name)
        if not table.startswith("cayu_knowledge_"):
            raise ValueError(f"{table!r} is not a Cayu knowledge table.")
        if table in result:
            raise ValueError(f"{field_name} contains duplicate table {table!r}.")
        result.append(table)
    return sorted(result)


__all__ = [
    "KNOWLEDGE_REVISION_RESET_POLICY_VERSION",
    "KnowledgeRevisionResetRequired",
    "KnowledgeRevisionTransitionAction",
    "KnowledgeRevisionTransitionAssessment",
    "assess_knowledge_revision_transition",
    "require_empty_knowledge_revision_transition",
]
