"""How alike two knowledge entries are, for suggesting and grouping likely duplicates.

``KnowledgeSimilarity`` is the swappable part: it scores a subject (a draft or an entry)
against candidates. ``LexicalKnowledgeSimilarity`` is the default and makes no model
calls: word overlap (Jaccard on words of four or more letters) is the main signal, and
overlap of code identifiers (backticked names, ``snake_case``, ``CamelCase`` and dotted
names) is a second one. An embedding-based part can replace it by implementing
``score``; nothing else depends on how a score is computed.

Scope comes first and isn't swappable: ``KnowledgeSimilarity.matches`` compares only
candidates in the subject's namespace and visibility that carry the same values for
the part's ``scope_labels`` (``("module", "repository")``, say). Callers hand it entries
they read through their own access scope, so a score never crosses tenants either.

A score suggests and groups. It never keeps, drops or merges knowledge by itself.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from types import MappingProxyType

from cayu._validation import copy_label_map, require_clean_nonblank
from cayu.knowledge.records import (
    DEFAULT_KNOWLEDGE_KIND,
    DEFAULT_KNOWLEDGE_NAMESPACE,
    KnowledgeEntry,
    KnowledgeVisibility,
)

#: The default thresholds, tuned on labelled duplicate proposals (see
#: ``tests/fixtures/knowledge_duplicate_pairs.json``).
DEFAULT_WORD_OVERLAP_THRESHOLD = 0.2
DEFAULT_IDENTIFIER_OVERLAP_THRESHOLD = 0.5
DEFAULT_MIN_SHARED_IDENTIFIERS = 2

_WORD = re.compile(r"[^\W\d_]{4,}")
_BACKTICKED = re.compile(r"`([^`\n]+)`")
_IDENTIFIER = re.compile(
    r"\b(?:"
    r"[A-Za-z_][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+"  # snake_case, _private_name
    r"|[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+"  # CamelCase
    r"|[a-z]+[A-Z][A-Za-z0-9]*"  # camelCase
    # Plain dotted names (file.py, tasks.worker). A qualified name whose first part is
    # an identifier above counts as that part, so `Store.get` and `get` notes still share
    # `Store`; counting it whole too made paraphrases score lower on the labelled pairs.
    r"|[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+"
    r")\b"
)


@dataclass(frozen=True)
class KnowledgeComparable:
    """What a similarity part compares: an entry, or a draft that isn't stored yet."""

    text: str
    title: str | None = None
    kind: str = DEFAULT_KNOWLEDGE_KIND
    namespace: str = DEFAULT_KNOWLEDGE_NAMESPACE
    visibility: KnowledgeVisibility = KnowledgeVisibility.GLOBAL
    labels: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.text) is not str:
            raise TypeError("text must be a string.")
        if self.title is not None and type(self.title) is not str:
            raise TypeError("title must be a string or None.")
        object.__setattr__(self, "kind", require_clean_nonblank(self.kind, "kind"))
        object.__setattr__(self, "namespace", require_clean_nonblank(self.namespace, "namespace"))
        object.__setattr__(self, "visibility", KnowledgeVisibility(self.visibility))
        object.__setattr__(
            self, "labels", MappingProxyType(copy_label_map(dict(self.labels), "labels"))
        )

    @classmethod
    def from_entry(cls, entry: KnowledgeEntry) -> KnowledgeComparable:
        return cls(
            text=entry.text,
            title=entry.title,
            kind=entry.kind,
            namespace=entry.namespace,
            visibility=entry.visibility,
            labels=entry.labels,
        )


@dataclass(frozen=True)
class KnowledgeSimilarityScore:
    """One comparison: ``score`` in [0, 1] ranks, ``similar`` says it passed the part's
    threshold, and ``signals`` holds the part's own measures (for the lexical part,
    ``word_overlap`` and ``identifier_overlap``)."""

    score: float
    similar: bool
    signals: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be between 0 and 1.")
        object.__setattr__(self, "signals", MappingProxyType(dict(self.signals)))


@dataclass(frozen=True)
class KnowledgeSimilarityMatch:
    """A candidate that passed scope and threshold, with its position in the input."""

    index: int
    candidate: KnowledgeComparable
    score: KnowledgeSimilarityScore


class KnowledgeSimilarity(ABC):
    """Scores knowledge entries against each other, within one scope.

    Subclasses implement ``score``. ``same_scope`` and ``matches`` are the scope rule
    every part shares: an implementation never sees a candidate from another
    namespace, visibility or ``scope_labels`` value through ``matches``.
    """

    def __init__(self, *, scope_labels: Iterable[str] = ()) -> None:
        if isinstance(scope_labels, str | bytes):
            raise TypeError("scope_labels must be an iterable of label keys.")
        self.scope_labels: tuple[str, ...] = tuple(
            dict.fromkeys(
                require_clean_nonblank(key, f"scope_labels[{index}]")
                for index, key in enumerate(scope_labels)
            )
        )

    def same_scope(self, a: KnowledgeComparable, b: KnowledgeComparable) -> bool:
        """Same namespace, visibility and value (or absence) of every scope label."""

        return (
            a.namespace == b.namespace
            and a.visibility == b.visibility
            and all(a.labels.get(key) == b.labels.get(key) for key in self.scope_labels)
        )

    @abstractmethod
    async def score(
        self,
        subject: KnowledgeComparable,
        candidates: Sequence[KnowledgeComparable],
    ) -> list[KnowledgeSimilarityScore]:
        """One score per candidate, in order. Called only with in-scope candidates."""

    async def matches(
        self,
        subject: KnowledgeComparable,
        candidates: Sequence[KnowledgeComparable],
        *,
        limit: int | None = None,
    ) -> list[KnowledgeSimilarityMatch]:
        """The in-scope candidates the part calls similar, best first."""

        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError("limit must be a positive integer or None.")
        in_scope = [
            (index, candidate)
            for index, candidate in enumerate(candidates)
            if self.same_scope(subject, candidate)
        ]
        if not in_scope:
            return []
        scores = await self.score(subject, [candidate for _, candidate in in_scope])
        if len(scores) != len(in_scope):
            raise ValueError("A similarity part must return one score per candidate.")
        found = [
            KnowledgeSimilarityMatch(index=index, candidate=candidate, score=score)
            for (index, candidate), score in zip(in_scope, scores, strict=True)
            if score.similar
        ]
        found.sort(key=lambda match: (-match.score.score, match.index))
        return found if limit is None else found[:limit]


class LexicalKnowledgeSimilarity(KnowledgeSimilarity):
    """Word and identifier overlap; no model calls.

    Two entries are similar when their word overlap reaches ``word_threshold``, or
    when they share at least ``min_shared_identifiers`` identifiers and their
    identifier overlap reaches ``identifier_threshold``. The score is the larger of
    the overlaps that count.
    """

    def __init__(
        self,
        *,
        scope_labels: Iterable[str] = (),
        word_threshold: float = DEFAULT_WORD_OVERLAP_THRESHOLD,
        identifier_threshold: float = DEFAULT_IDENTIFIER_OVERLAP_THRESHOLD,
        min_shared_identifiers: int = DEFAULT_MIN_SHARED_IDENTIFIERS,
    ) -> None:
        super().__init__(scope_labels=scope_labels)
        self.word_threshold = _unit_threshold(word_threshold, "word_threshold")
        self.identifier_threshold = _unit_threshold(identifier_threshold, "identifier_threshold")
        if type(min_shared_identifiers) is not int or min_shared_identifiers < 1:
            raise ValueError("min_shared_identifiers must be a positive integer.")
        self.min_shared_identifiers = min_shared_identifiers

    async def score(
        self,
        subject: KnowledgeComparable,
        candidates: Sequence[KnowledgeComparable],
    ) -> list[KnowledgeSimilarityScore]:
        words, identifiers = knowledge_words(subject), knowledge_identifiers(subject)
        return [
            self._score_sets(
                words, identifiers, knowledge_words(candidate), knowledge_identifiers(candidate)
            )
            for candidate in candidates
        ]

    def score_pair(
        self, a: KnowledgeComparable, b: KnowledgeComparable
    ) -> KnowledgeSimilarityScore:
        """The lexical score of two entries, ignoring scope."""

        return self._score_sets(
            knowledge_words(a),
            knowledge_identifiers(a),
            knowledge_words(b),
            knowledge_identifiers(b),
        )

    def _score_sets(
        self,
        a_words: frozenset[str],
        a_identifiers: frozenset[str],
        b_words: frozenset[str],
        b_identifiers: frozenset[str],
    ) -> KnowledgeSimilarityScore:
        words = _jaccard(a_words, b_words)
        shared = len(a_identifiers & b_identifiers)
        identifiers = _jaccard(a_identifiers, b_identifiers)
        counted_identifiers = identifiers if shared >= self.min_shared_identifiers else 0.0
        return KnowledgeSimilarityScore(
            score=max(words, counted_identifiers),
            similar=words >= self.word_threshold
            or counted_identifiers >= self.identifier_threshold,
            signals={"word_overlap": words, "identifier_overlap": identifiers},
        )


def knowledge_words(item: KnowledgeComparable) -> frozenset[str]:
    """The lowercased words of four or more letters in an entry's title and text."""

    material = _material(item)
    return _cached_words(material) if len(material) <= _CACHED_MATERIAL_CHARS else _words(material)


def knowledge_identifiers(item: KnowledgeComparable) -> frozenset[str]:
    """Code identifiers in an entry's title and text, lowercased.

    Backticked names count whole (without call parentheses); outside backticks,
    ``snake_case``, ``CamelCase``, ``camelCase`` and plain dotted names (``file.py``)
    count, and a qualified name such as ``Store.get`` counts as its first part.
    """

    material = _material(item)
    if len(material) <= _CACHED_MATERIAL_CHARS:
        return _cached_identifiers(material)
    return _identifiers(material)


def _words(material: str) -> frozenset[str]:
    return frozenset(word.casefold() for word in _WORD.findall(material))


def _identifiers(material: str) -> frozenset[str]:
    found = {
        name.strip().removesuffix("()").casefold()
        for name in _BACKTICKED.findall(material)
        if name.strip()
    }
    found.update(match.casefold() for match in _IDENTIFIER.findall(_BACKTICKED.sub(" ", material)))
    return frozenset(found)


# Grouping a review queue compares every proposal with every other, so a note-sized text
# is tokenized once rather than once per pair; long documents aren't kept.
_CACHED_MATERIAL_CHARS = 8_192
_cached_words = lru_cache(maxsize=2048)(_words)
_cached_identifiers = lru_cache(maxsize=2048)(_identifiers)


def _material(item: KnowledgeComparable) -> str:
    return item.text if not item.title else f"{item.title}\n{item.text}"


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _unit_threshold(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0 < value <= 1:
        raise ValueError(f"{name} must be a number above 0 and at most 1.")
    return float(value)
