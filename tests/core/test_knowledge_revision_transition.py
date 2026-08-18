from __future__ import annotations

import pytest
from pydantic import ValidationError

from cayu.storage import (
    KNOWLEDGE_REVISION_RESET_POLICY_VERSION,
    KnowledgeRevisionResetRequired,
    KnowledgeRevisionTransitionAction,
    KnowledgeRevisionTransitionAssessment,
    assess_knowledge_revision_transition,
    require_empty_knowledge_revision_transition,
)

_TABLES = (
    "cayu_knowledge_chunks",
    "cayu_knowledge_entries",
    "cayu_knowledge_publication_receipts",
)


def test_empty_legacy_knowledge_is_eligible_for_clean_initialization() -> None:
    assessment = require_empty_knowledge_revision_transition(
        {table: 0 for table in reversed(_TABLES)},
        required_tables=_TABLES,
    )

    assert assessment.policy_version == KNOWLEDGE_REVISION_RESET_POLICY_VERSION
    assert assessment.inspected_tables == tuple(sorted(_TABLES))
    assert assessment.populated_tables == ()
    assert assessment.action is KnowledgeRevisionTransitionAction.INITIALIZE


def test_populated_legacy_knowledge_refuses_without_backfill_or_deletion() -> None:
    counts = {
        "cayu_knowledge_entries": 2,
        "cayu_knowledge_chunks": 5,
        "cayu_knowledge_publication_receipts": 0,
    }

    assessment = assess_knowledge_revision_transition(counts, required_tables=_TABLES)
    assert assessment.action is KnowledgeRevisionTransitionAction.REFUSE_RESET_REQUIRED
    assert assessment.populated_tables == (
        "cayu_knowledge_chunks",
        "cayu_knowledge_entries",
    )

    with pytest.raises(
        KnowledgeRevisionResetRequired,
        match="requires a database replacement/reset",
    ) as caught:
        require_empty_knowledge_revision_transition(counts, required_tables=_TABLES)

    assert caught.value.assessment == assessment
    assert counts == {
        "cayu_knowledge_entries": 2,
        "cayu_knowledge_chunks": 5,
        "cayu_knowledge_publication_receipts": 0,
    }
    assert "did not backfill, delete, or modify" in str(caught.value)


def test_revision_transition_refuses_an_incomplete_or_invalid_inspection() -> None:
    with pytest.raises(ValueError, match="exactly cover required tables"):
        assess_knowledge_revision_transition(
            {"cayu_knowledge_entries": 0},
            required_tables=_TABLES,
        )
    with pytest.raises(ValueError, match="non-negative integer"):
        assess_knowledge_revision_transition(
            {
                "cayu_knowledge_entries": -1,
                "cayu_knowledge_chunks": 0,
                "cayu_knowledge_publication_receipts": 0,
            },
            required_tables=_TABLES,
        )
    with pytest.raises(ValueError, match="not a Cayu knowledge table"):
        assess_knowledge_revision_transition(
            {"cayu_sessions": 0},
            required_tables=("cayu_sessions",),
        )

    with pytest.raises(ValidationError, match="action must match"):
        KnowledgeRevisionTransitionAssessment(
            inspected_tables=("cayu_knowledge_entries",),
            populated_tables=("cayu_knowledge_entries",),
            action=KnowledgeRevisionTransitionAction.INITIALIZE,
        )
