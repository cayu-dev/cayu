"""PostgreSQL catalog constraint matching and index expression comparison.

Callers retain revision gates, validation order and the existing cursor/transaction.
These functions inspect the schema without acquiring connections or applying DDL.
"""

from __future__ import annotations

import re
from collections.abc import Sequence


def _normalize_postgres_index_expression(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.lower().replace('"', "")
    normalized = normalized.replace("::text[]", "").replace("::text", "")
    return re.sub(r"[\s()]", "", normalized)


def _constraint_fragments_match_exactly(
    candidates: Sequence[tuple[str, str]],
    required: Sequence[tuple[str, tuple[str, ...]]],
) -> bool:
    """Match every required constraint to one distinct catalog constraint."""

    if len(candidates) != len(required):
        return False
    compatible_candidates = tuple(
        tuple(
            candidate_index
            for candidate_index, (candidate_kind, definition) in enumerate(candidates)
            if candidate_kind == required_kind
            and all(fragment in definition for fragment in fragments)
        )
        for required_kind, fragments in required
    )
    if any(not compatible for compatible in compatible_candidates):
        return False

    required_by_candidate: dict[int, int] = {}

    def assign(required_index: int, visited_candidates: set[int]) -> bool:
        for candidate_index in compatible_candidates[required_index]:
            if candidate_index in visited_candidates:
                continue
            visited_candidates.add(candidate_index)
            previous_required = required_by_candidate.get(candidate_index)
            if previous_required is None or assign(previous_required, visited_candidates):
                required_by_candidate[candidate_index] = required_index
                return True
        return False

    return all(assign(required_index, set()) for required_index in range(len(required)))
