"""Bounded provider-neutral relevance evidence, independent of retrieval rank."""

from __future__ import annotations

import re
import unicodedata
from collections import deque
from collections.abc import Iterator
from typing import Literal

from pydantic import BaseModel, ConfigDict

from cayu.retrieval import RetrievalCandidateIdentity

RELEVANCE_VERSION = "cayu.query_concepts.v1"
RELEVANCE_TEXT_VERSION = f"{RELEVANCE_VERSION}+unicode-{unicodedata.unidata_version}"
TITLE_RELEVANCE_VERSION = "cayu.query_concepts.v2"
TITLE_RELEVANCE_TEXT_VERSION = f"{TITLE_RELEVANCE_VERSION}+unicode-{unicodedata.unidata_version}"
PHRASE_RELEVANCE_VERSION = "cayu.query_concepts.v3"
PHRASE_RELEVANCE_TEXT_VERSION = f"{PHRASE_RELEVANCE_VERSION}+unicode-{unicodedata.unidata_version}"
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
# Deliberately bounded vocabulary, not general stemming or semantic similarity.
# v1 remains unchanged for persisted policies and replay.
_V2_CONCEPTS = {
    **_CONCEPTS,
    "cached": "cache",
    "caches": "cache",
    "caching": "cache",
    "retries": "retry",
    "retried": "retry",
    "retrying": "retry",
    "buckets": "bucket",
}

# These delivery/schema words cannot supply the additional v3 phrase signal.
# They remain in the original whole-query coverage calculation. A barrier, not
# deletion, prevents joining unrelated topic words across a format instruction.
_PHRASE_BOILERPLATE = frozenset(
    {
        "answer",
        "respond",
        "reply",
        "return",
        "output",
        "format",
        "formatted",
        "json",
        "yaml",
        "xml",
        "markdown",
        "only",
        "use",
        "using",
        "with",
        "without",
        "key",
        "keys",
        "field",
        "fields",
        "status",
        "known",
        "unknown",
        "null",
        "true",
        "false",
        "string",
        "object",
        "value",
        "values",
    }
)
_TERM = re.compile(r"[^\W_]+(?:[-./][^\W_]+)*")
_PHRASE_BREAK = re.compile(
    r"[,.!?;\n\r\v\f\x1c-\x1e\x85\u2028\u2029\u3002\uff0c\uff01\uff1f\uff1b]"
)
# Recognized delivery clauses cannot contribute a phrase, including words after
# a format cue. This is deliberately bounded English syntax, not an instruction
# parser. In particular, 'return the deployment window' is still a factual query.
_DELIVERY_VERBS = frozenset(
    {
        "be",
        "keep",
        "answer",
        "respond",
        "reply",
        "return",
        "write",
        "give",
        "provide",
        "format",
        "use",
        "avoid",
        "ensure",
        "make",
    }
)
_DELIVERY_MARKERS = frozenset(
    {
        "answer",
        "response",
        "reply",
        "output",
        "json",
        "yaml",
        "xml",
        "markdown",
        "concise",
        "brief",
        "brevity",
        "succinct",
        "terse",
        "verbose",
        "verbosity",
        "clear",
        "clarity",
        "simple",
        "short",
        "sentence",
        "sentences",
        "paragraph",
        "paragraphs",
        "bullet",
        "bullets",
        "bulleted",
        "tone",
        "style",
        "format",
        "formatted",
        "formatting",
        "prose",
        "precise",
        "friendly",
        "professional",
        "calm",
        "neutral",
        "polite",
        "formal",
        "informal",
    }
)
_DELIVERY_FILLERS = frozenset(
    {"a", "an", "the", "in", "as", "only", "your", "it", "very", "strictly", "just"}
)
_DELIVERY_PREFIXES = (
    ("do", "not", "guess"),
    ("do", "not", "invent"),
    ("do", "not", "fabricate"),
    ("never", "make", "up"),
)


def _is_delivery_clause(terms: list[str], start: int = 0) -> bool:
    if start < len(terms) and terms[start] == "please":
        start += 1
    if start == len(terms):
        return False
    # Only the leading descriptor establishes a delivery clause. A format word
    # later in 'return the maintenance window in JSON' must not erase the topic.
    if terms[start] in _DELIVERY_VERBS:
        position = start + 1
        while position < len(terms) and terms[position] in _DELIVERY_FILLERS:
            position += 1
        if position < len(terms) and terms[position] in _DELIVERY_MARKERS:
            return True
    return any(tuple(terms[start : start + len(prefix)]) == prefix for prefix in _DELIVERY_PREFIXES)


def _topic_clause(terms: list[str]) -> list[str]:
    if _is_delivery_clause(terms):
        return []
    # Also exclude appended delivery commands without requiring punctuation.
    # Inspect by index rather than repeatedly allocating whole suffixes.
    for index, term in enumerate(terms):
        if term in {"and", "then", "also", "please"} and _is_delivery_clause(
            terms, index if term == "please" else index + 1
        ):
            return terms[:index]
    return terms


def _clauses(text: str) -> Iterator[tuple[list[str], bool]]:
    """Yield terms and whether their preceding boundary resets delivery context.

    Commas break phrase windows but not an ongoing delivery instruction. Token
    gaps keep dots inside exact file/path tokens from becoming boundaries.
    """
    terms: list[str] = []
    previous_end = 0
    reset_delivery = True
    text = text.casefold()
    for match in _TERM.finditer(text):
        boundaries = _PHRASE_BREAK.findall(text[previous_end : match.start()])
        if boundaries:
            if terms:
                yield terms, reset_delivery
            terms = []
            reset_delivery = any(boundary not in {",", "\uff0c"} for boundary in boundaries)
        previous_end = match.end()
        terms.append(match.group())
    if terms:
        yield terms, reset_delivery


def _phrases(text: str, *, query: bool = False) -> Iterator[tuple[str, str, str]]:
    """Linear scan; three distinct concepts in one uninterrupted topic phrase."""
    window: deque[str] = deque(maxlen=3)
    delivery = False
    for clause, reset_delivery in _clauses(text):
        window.clear()
        if query:
            if reset_delivery:
                delivery = False
            if delivery:
                continue
            topic = _topic_clause(clause)
            delivery = len(topic) != len(clause)
            clause = topic
        for term in clause:
            if term in _PHRASE_BOILERPLATE:
                window.clear()
                continue
            if term in _STOP:
                continue
            window.append(_V2_CONCEPTS.get(term, term))
            if len(window) == 3 and len(set(window)) == 3:
                yield window[0], window[1], window[2]


class RecallCandidateDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    identity: RetrievalCandidateIdentity
    eligibility: Literal["eligible", "low_relevance", "insufficient_evidence", "legacy_rank_only"]
    reason: Literal[
        "query_concept_support",
        "query_phrase_support",
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


def _concepts(text: str, *, version: str = RELEVANCE_VERSION) -> set[str]:
    vocabulary = _CONCEPTS if version == RELEVANCE_VERSION else _V2_CONCEPTS
    return {
        vocabulary.get(term, term)
        for term in re.findall(r"[^\W_]+(?:[-./][^\W_]+)*", text.casefold())
        if term not in _STOP
    }


def query_concept_eligibility(
    query: str | None,
    text: str,
    *,
    version: str = RELEVANCE_VERSION,
    title: str | None = None,
) -> tuple[str, str]:
    """At most 8,192 query bytes / 128,000 text bytes; no channel confidence."""
    if version not in {RELEVANCE_VERSION, TITLE_RELEVANCE_VERSION, PHRASE_RELEVANCE_VERSION}:
        raise ValueError("Unsupported query concept version.")
    if query is None:
        return "insufficient_evidence", "missing_query_evidence"
    terms = _concepts(query, version=version)
    if not terms:
        return "insufficient_evidence", "missing_query_evidence"
    evidence = _concepts(text, version=version)
    if version != RELEVANCE_VERSION and title is not None:
        evidence |= _concepts(title, version=version)
    supported = terms & evidence
    # One exact identifier is useful; a multi-concept question needs two concepts
    # and a majority of its evidence. Repetition/volume cannot change the result.
    if len(supported) >= min(2, len(terms)) and len(supported) / len(terms) >= 0.6:
        return "eligible", "query_concept_support"
    if version == PHRASE_RELEVANCE_VERSION and len(supported) >= 3:
        query_phrases = set(_phrases(query, query=True))
        if query_phrases and (
            any(phrase in query_phrases for phrase in _phrases(text))
            or (title is not None and any(phrase in query_phrases for phrase in _phrases(title)))
        ):
            return "eligible", "query_phrase_support"
    return "low_relevance", "weak_query_support"
