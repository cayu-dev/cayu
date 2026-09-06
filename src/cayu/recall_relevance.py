"""Bounded provider-neutral relevance evidence, independent of retrieval rank."""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict

from cayu.retrieval import RetrievalCandidateIdentity

RELEVANCE_VERSION = "cayu.query_concepts.v1"
RELEVANCE_TEXT_VERSION = f"{RELEVANCE_VERSION}+unicode-{unicodedata.unidata_version}"
# Fixed calibration vocabulary; changing it requires a new version.
_STOP = frozenset(
    [
        "a",
        "an",
        "the",
        "how",
        "should",
        "we",
        "i",
        "you",
        "to",
        "for",
        "of",
        "in",
        "on",
        "is",
        "are",
        "do",
        "does",
        "can",
        "could",
        "would",
        "and",
        "or",
        "about",
        "that",
        "this",
        "it",
        "what",
        "when",
        "please",
        "configure",
        "configuration",
    ]
)
_CONCEPTS = {
    "rollbacks": "rollback",
    "revert": "rollback",
    "reverting": "rollback",
    "reversion": "rollback",
    "undo": "rollback",
    "deployment": "release",
    "deployments": "release",
    "deploy": "release",
    "deploying": "release",
    "releases": "release",
    "safeguards": "safety",
    "safeguard": "safety",
    "safely": "safety",
    "safe": "safety",
    "protection": "safety",
    "procedures": "procedure",
    "steps": "procedure",
    "credentials": "credential",
    "passwords": "credential",
    "password": "credential",
    "authentication": "auth",
    "login": "auth",
    "timeouts": "timeout",
    "deadlines": "timeout",
    "deadline": "timeout",
}


class RecallCandidateDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    identity: RetrievalCandidateIdentity
    eligibility: Literal["eligible", "low_relevance", "insufficient_evidence", "legacy_rank_only"]
    reason: Literal[
        "query_concept_support",
        "weak_query_support",
        "missing_query_evidence",
        "rank_only_calibration",
    ]
    outcome: Literal[
        "focused",
        "offered",
        "low_relevance",
        "insufficient_evidence",
        "below_score",
        "oversized",
        "duplicate",
        "capacity",
        "mode",
        "unevaluated",
    ]


def _concepts(text: str) -> set[str]:
    return {
        _CONCEPTS.get(term, term)
        for term in re.findall(r"[^\W_]+(?:[-./][^\W_]+)*", text.casefold())
        if term not in _STOP
    }


def query_concept_eligibility(query: str | None, text: str) -> tuple[str, str]:
    """At most 8,192 query bytes / 128,000 text bytes; no channel confidence."""
    if query is None:
        return "insufficient_evidence", "missing_query_evidence"
    terms = _concepts(query)
    if not terms:
        return "insufficient_evidence", "missing_query_evidence"
    supported = terms & _concepts(text)
    # One exact identifier is useful; a multi-concept question needs two concepts
    # and a majority of its evidence. Repetition/volume cannot change the result.
    if len(supported) >= min(2, len(terms)) and len(supported) / len(terms) >= 0.6:
        return "eligible", "query_concept_support"
    return "low_relevance", "weak_query_support"
