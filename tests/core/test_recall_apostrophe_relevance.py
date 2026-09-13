from __future__ import annotations

import pytest

from cayu.recall_relevance import (
    APOSTROPHE_RELEVANCE_TEXT_VERSION,
    APOSTROPHE_RELEVANCE_VERSION,
    _concepts,
    query_concept_eligibility,
)


@pytest.mark.parametrize("apostrophe", ["'", "\u2019"])
@pytest.mark.parametrize("location", ["text", "title"])
def test_possessive_suffix_cannot_supply_a_third_concept(apostrophe, location):
    query = (
        "Implement Harbor customer export CSV escaping and empty records. "
        "Current explicit requirements override older decisions; another "
        f"project{apostrophe}s decisions do not establish this project{apostrophe}s requirements."
    )
    foreign = (
        "Juniper customer exports use comma delimiters, LF endings, and no header. "
        "This agreement belongs to Juniper only; it does not establish another "
        f"project{apostrophe}s requirements."
    )
    kwargs = {"text": foreign} if location == "text" else {"text": "Juniper.", "title": foreign}
    assert query_concept_eligibility(query, **kwargs, version="cayu.query_concepts.v4") == (
        "eligible",
        "query_phrase_support",
    )
    assert query_concept_eligibility(query, **kwargs, version=APOSTROPHE_RELEVANCE_VERSION) == (
        "low_relevance",
        "weak_query_support",
    )
    assert (
        query_concept_eligibility(
            query,
            "Use semicolon delimiters and CRLF endings.",
            title="Harbor customer export CSV format",
            version=APOSTROPHE_RELEVANCE_VERSION,
        )[0]
        == "eligible"
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("project's", {"project"}),
        ("PROJECT\u2019S", {"project"}),
        ("projects'", {"projects"}),
        ("don't", {"don't"}),
        ("we\u2019re", {"we're"}),
        ("S", {"s"}),
        ("project 's'", {"project", "s"}),
        ("project\u2019s.py", {"project's.py"}),
        ("org/project's", {"org/project's"}),
        ("o'reilly", {"o'reilly"}),
        ("it's", set()),
        ("server's", {"server"}),
    ],
)
def test_apostrophe_lexical_units(text, expected):
    assert _concepts(text, version=APOSTROPHE_RELEVANCE_VERSION) == expected


@pytest.mark.parametrize("boundary", ["\n", ". ", "; ", "\u2028"])
def test_normalization_does_not_join_phrase_boundaries(boundary):
    query = "Explain Harbor customer export authentication retries deadlines and backups."
    assert (
        query_concept_eligibility(
            query,
            f"Harbor's customer{boundary}export",
            version=APOSTROPHE_RELEVANCE_VERSION,
        )[0]
        == "low_relevance"
    )


@pytest.mark.parametrize(
    "query,text",
    [
        ("Harbor's customer export", "Harbor customer export"),
        ("Harbor customer export", "Harbor\u2019s customer export"),
        ("sensor S latency", "sensor S latency"),
    ],
)
def test_real_three_concept_phrases_remain_eligible(query, text):
    assert query_concept_eligibility(
        query + ". Explain retries deadlines authentication failures backups and recovery.",
        text,
        version=APOSTROPHE_RELEVANCE_VERSION,
    ) == ("eligible", "query_phrase_support")


def test_policy_identity_round_trips_and_rejects_old_text_version():
    from test_memory_admission import _policy

    policy = _policy(relevance_policy=APOSTROPHE_RELEVANCE_VERSION)
    previous = _policy(relevance_policy="cayu.query_concepts.v4")
    assert policy.relevance_text_version == APOSTROPHE_RELEVANCE_TEXT_VERSION
    assert type(policy).model_validate_json(policy.model_dump_json()) == policy
    assert policy.fingerprint() != previous.fingerprint()
    with pytest.raises(ValueError):
        _policy(
            relevance_policy=APOSTROPHE_RELEVANCE_VERSION,
            relevance_text_version=previous.relevance_text_version,
        )
