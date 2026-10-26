"""Exact parent binding and bounded membership for authored-suite publication."""

from __future__ import annotations

from dataclasses import dataclass

from cayu.evals.read_views import EvalSuiteViewIdentity
from cayu.evals.store import EvalAuthoredSuiteConflict
from cayu.evals.suite_authoring import EvalSuiteDocument, eval_suite_document_from_json


@dataclass(frozen=True)
class AuthoredParentScope:
    parent_selector: str
    child_selector: str
    unchanged_members: frozenset[tuple[str, str]]


def authored_parent_scope(
    child: EvalSuiteDocument, parent_revision: str, parent_json: str | None
) -> AuthoredParentScope:
    """Resolve authority only from the exact parent loaded by the transaction owner."""
    if parent_json is None:
        raise EvalAuthoredSuiteConflict("The authored parent is unavailable.")
    parent = eval_suite_document_from_json(parent_json)
    if (
        parent.revision != parent_revision
        or parent.target_key != child.target_key
        or parent.suite.id != child.suite.id
    ):
        raise EvalAuthoredSuiteConflict("The authored parent belongs to another suite.")
    return AuthoredParentScope(
        parent_selector=EvalSuiteViewIdentity(
            source="authored", target_key=parent.target_key, revision=parent.revision
        ).id,
        child_selector=EvalSuiteViewIdentity(
            source="authored", target_key=child.target_key, revision=child.revision
        ).id,
        unchanged_members=frozenset((case.id, case.revision) for case in parent.cases)
        & frozenset((case.id, case.revision) for case in child.cases),
    )


def require_authored_parent_replay(
    child_revision: str, stored_parent: str | None, expected_parent: str | None
) -> None:
    # Saving the already selected immutable document is a read-only no-op. It
    # neither changes its original parent binding nor acquires any associations.
    if expected_parent != child_revision and stored_parent != expected_parent:
        raise EvalAuthoredSuiteConflict("The authored revision has another publication parent.")
