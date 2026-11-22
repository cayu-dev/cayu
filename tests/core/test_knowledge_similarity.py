"""The knowledge similarity part: lexical default, scope rule, and labelled duplicates."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from cayu import CayuApp
from cayu.knowledge.records import KnowledgeEntry, KnowledgeVisibility
from cayu.knowledge.similarity import (
    KnowledgeComparable,
    KnowledgeSimilarity,
    KnowledgeSimilarityScore,
    LexicalKnowledgeSimilarity,
    knowledge_identifiers,
    knowledge_words,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "knowledge_duplicate_pairs.json"


def _fixture_entries() -> tuple[dict[str, KnowledgeComparable], dict]:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    entries = {
        entry["id"]: KnowledgeComparable(
            text=entry["text"], title=entry["title"], labels={"module": entry["module"]}
        )
        for entry in data["entries"]
    }
    return entries, data


def test_default_thresholds_catch_labelled_duplicates_and_rarely_flag_distinct_notes():
    entries, data = _fixture_entries()
    similarity = LexicalKnowledgeSimilarity(scope_labels=("module",))
    caught = [
        pair
        for pair in data["duplicates"]
        if similarity.score_pair(entries[pair[0]], entries[pair[1]]).similar
    ]
    flagged = [
        pair
        for pair in data["distinct"]
        if similarity.score_pair(entries[pair[0]], entries[pair[1]]).similar
    ]
    # Measured when the thresholds were set: 25 of 27 duplicates, 1 of 344 distinct pairs.
    assert len(data["duplicates"]) == 27 and len(data["distinct"]) == 344
    assert len(caught) >= 25
    assert len(flagged) <= 3


def test_fixture_distinct_pairs_share_a_module_and_one_duplicate_does_not():
    entries, data = _fixture_entries()
    for a, b in data["distinct"]:
        assert entries[a].labels["module"] == entries[b].labels["module"]
    # One duplicate was proposed under another module's label: a scope on `module`
    # misses it, a scope on the repository alone would not.
    crossing = [
        (a, b)
        for a, b in data["duplicates"]
        if entries[a].labels["module"] != entries[b].labels["module"]
    ]
    assert crossing == [("n21", "n33")]


def test_words_and_identifiers():
    item = KnowledgeComparable(
        title="Worker leases",
        text=(
            "`run_task_worker()` renews the lease; TaskStoreDispatcher keeps claim_generation "
            "in tasks/dispatch.py, getLease is old and DataStore.get DataStore.set stay. "
            "A cat ran."
        ),
    )
    assert knowledge_words(item) >= {"worker", "leases", "renews", "lease", "keeps"}
    assert "cat" not in knowledge_words(item) and "ran" not in knowledge_words(item)
    assert knowledge_identifiers(item) == {
        "run_task_worker",
        "taskstoredispatcher",
        "claim_generation",
        "dispatch.py",
        "getlease",
        "datastore",
    }


def test_identifier_overlap_counts_only_with_enough_shared_identifiers():
    similarity = LexicalKnowledgeSimilarity()
    one_shared = similarity.score_pair(
        KnowledgeComparable(text="Workers never retry dispatched payloads; see `resume_session`."),
        KnowledgeComparable(text="Branch storage keeps snapshots immutable; see `resume_session`."),
    )
    assert one_shared.signals["identifier_overlap"] == 1.0
    assert one_shared.signals["word_overlap"] < 0.2
    assert not one_shared.similar
    two_shared = similarity.score_pair(
        KnowledgeComparable(text="`resume_session` then `load_checkpoint` first"),
        KnowledgeComparable(text="`load_checkpoint` before `resume_session` always"),
    )
    assert two_shared.similar and two_shared.score == 1.0


def test_matches_compare_only_the_subject_scope_and_rank_best_first():
    text = "Queued delivery retries keep the same delivery identity after an acknowledgement loss."
    subject = KnowledgeComparable(
        text=text, labels={"module": "queue.py", "repository": "example/app"}
    )
    candidates = [
        KnowledgeComparable(text=text, namespace="other", labels=subject.labels),
        KnowledgeComparable(text=text, visibility=KnowledgeVisibility.USER, labels=subject.labels),
        KnowledgeComparable(text=text, labels={"module": "queue.py"}),
        KnowledgeComparable(text=text + " Retries are bounded.", labels=subject.labels),
        KnowledgeComparable(text="Unrelated note about branch storage.", labels=subject.labels),
        KnowledgeComparable(text=text, labels=subject.labels),
    ]
    similarity = LexicalKnowledgeSimilarity(scope_labels=("module", "repository"))
    found = asyncio.run(similarity.matches(subject, candidates))
    assert [match.index for match in found] == [5, 3]
    assert found[0].score.score == 1.0
    assert [m.index for m in asyncio.run(similarity.matches(subject, candidates, limit=1))] == [5]
    # Without scope labels, a candidate that lacks the module label is in scope too.
    unscoped = asyncio.run(LexicalKnowledgeSimilarity().matches(subject, candidates))
    assert [match.index for match in unscoped] == [2, 5, 3]


def test_comparable_from_entry_copies_what_scoring_reads():
    entry = KnowledgeEntry(
        id="entry",
        text="Body",
        title="Title",
        kind="warning",
        namespace="team",
        visibility=KnowledgeVisibility.PROJECT,
        labels={"module": "a.py"},
    )
    item = KnowledgeComparable.from_entry(entry)
    assert (item.text, item.title, item.kind, item.namespace, item.visibility) == (
        "Body",
        "Title",
        "warning",
        "team",
        KnowledgeVisibility.PROJECT,
    )
    assert dict(item.labels) == {"module": "a.py"}
    with pytest.raises(TypeError):
        item.labels["module"] = "b.py"  # type: ignore[index]


class _EverythingAlike(KnowledgeSimilarity):
    def __init__(self) -> None:
        super().__init__(scope_labels=("module",))
        self.seen: list[int] = []

    async def score(
        self, subject: KnowledgeComparable, candidates: Sequence[KnowledgeComparable]
    ) -> list[KnowledgeSimilarityScore]:
        self.seen.append(len(candidates))
        return [KnowledgeSimilarityScore(score=0.5, similar=True) for _ in candidates]


def test_a_replacement_part_is_configured_on_the_app_and_still_scoped():
    assert isinstance(CayuApp().knowledge_similarity, LexicalKnowledgeSimilarity)
    part = _EverythingAlike()
    app = CayuApp(knowledge_similarity=part)
    assert app.knowledge_similarity is part
    subject = KnowledgeComparable(text="a", labels={"module": "a.py"})
    found = asyncio.run(
        part.matches(
            subject,
            [
                KnowledgeComparable(text="b", labels={"module": "b.py"}),
                KnowledgeComparable(text="c", labels={"module": "a.py"}),
            ],
        )
    )
    assert [match.index for match in found] == [1]
    assert part.seen == [1]
    with pytest.raises(TypeError, match="KnowledgeSimilarity"):
        CayuApp(knowledge_similarity=object())  # type: ignore[arg-type]


def test_a_part_must_score_every_candidate():
    class _Short(KnowledgeSimilarity):
        async def score(self, subject, candidates):
            return []

    with pytest.raises(ValueError, match="one score per candidate"):
        asyncio.run(
            _Short().matches(KnowledgeComparable(text="a"), [KnowledgeComparable(text="a")])
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"word_threshold": 0},
        {"word_threshold": 1.5},
        {"identifier_threshold": True},
        {"min_shared_identifiers": 0},
        {"scope_labels": "module"},
        {"scope_labels": [" module"]},
    ],
)
def test_lexical_settings_are_validated(kwargs):
    with pytest.raises((TypeError, ValueError)):
        LexicalKnowledgeSimilarity(**kwargs)


@pytest.mark.parametrize(
    ("metadata", "count"),
    [
        ({}, 1),
        ({"cayu:corroboration": {"source_count": 3, "sources": []}}, 3),
        ({"cayu:corroboration": {"source_count": 0}}, 1),
        ({"cayu:corroboration": {"source_count": True}}, 1),
        ({"cayu:corroboration": "3"}, 1),
    ],
)
def test_source_count_reads_the_corroboration_summary(metadata, count):
    from cayu.knowledge._sources import knowledge_source_count

    assert knowledge_source_count(KnowledgeEntry(id="e", text="t", metadata=metadata)) == count
