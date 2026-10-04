"""Shared keyword matching, field-aware phrase checks and best-match scoring."""

from __future__ import annotations

from collections import Counter

from cayu.knowledge.records import KnowledgeChunk, KnowledgeEntry
from cayu.knowledge.search import (
    KnowledgeQuery,
    _knowledge_query_terms,
    _query_terms_have_positive_terms,
    _SearchTerms,
    _tokenize_search_text,
)


def _score_entry(
    entry: KnowledgeEntry,
    chunks: list[KnowledgeChunk],
    query: KnowledgeQuery,
) -> tuple[float, KnowledgeChunk | None, str, str]:
    terms = _knowledge_query_terms(query)
    if not _query_terms_have_positive_terms(terms):
        return 0.0, None, "empty query", entry.text
    best_score = _score_candidate(entry.text, terms)
    best_chunk: KnowledgeChunk | None = None
    best_reason = "entry text match"
    best_preview_text = entry.text
    if entry.title is not None:
        title_score = _score_candidate(entry.title, terms) * 1.2
        if title_score > best_score:
            best_score = title_score
            best_reason = "title match"
            best_preview_text = entry.title
    for chunk in chunks:
        chunk_search_fields = _entry_chunk_searchable_fields(entry, chunk)
        chunk_score = _score_candidate(
            "\n".join(chunk_search_fields),
            terms,
            phrase_fields=chunk_search_fields,
        )
        if chunk_score > best_score:
            best_score = chunk_score
            best_chunk = chunk
            best_reason = "chunk text match"
            best_preview_text = chunk.text
    return best_score, best_chunk, best_reason, best_preview_text


def _score_candidate(
    text: str,
    terms: _SearchTerms,
    *,
    phrase_fields: list[str] | None = None,
) -> float:
    tokens = _tokenize_search_text(text)
    phrase_token_fields = (
        [tokens]
        if phrase_fields is None
        else [_tokenize_search_text(field) for field in phrase_fields]
    )
    if not _tokens_match_structured_terms(tokens, terms, phrase_token_fields):
        return 0.0
    token_counts = Counter(tokens)
    score = float(sum(token_counts[term] for term in terms["any"]))
    score += float(sum(max(token_counts[term] for term in group) for group in terms["all"]))
    score += float(
        sum(
            2
            for phrase in terms["phrases"]
            if any(_tokens_contain_phrase(field, phrase) for field in phrase_token_fields)
        )
    )
    return score


def _tokens_match_structured_terms(
    tokens: list[str],
    terms: _SearchTerms,
    phrase_token_fields: list[list[str]],
) -> bool:
    token_set = set(tokens)
    if any(term in token_set for term in terms["none"]):
        return False
    if not all(any(term in token_set for term in group) for group in terms["all"]):
        return False
    if terms["any"] and not any(term in token_set for term in terms["any"]):
        return False
    return not terms["phrases"] or any(
        _tokens_contain_phrase(field, phrase)
        for phrase in terms["phrases"]
        for field in phrase_token_fields
    )


def _tokens_contain_phrase(tokens: list[str], phrase: list[str]) -> bool:
    phrase_length = len(phrase)
    return any(
        tokens[index : index + phrase_length] == phrase
        for index in range(len(tokens) - phrase_length + 1)
    )


def _entry_chunk_searchable_fields(entry: KnowledgeEntry, chunk: KnowledgeChunk) -> list[str]:
    parts: list[str] = []
    if entry.title is not None:
        parts.append(entry.title)
    parts.append(entry.text)
    if chunk.text == entry.text:
        return parts
    parts.append(chunk.text)
    return parts


def _entry_matches_none_terms(
    entry: KnowledgeEntry,
    chunks: list[KnowledgeChunk],
    terms: _SearchTerms,
) -> bool:
    if not terms["none"]:
        return False
    texts = [entry.text]
    if entry.title is not None:
        texts.append(entry.title)
    texts.extend(chunk.text for chunk in chunks)
    tokens = {token for text in texts for token in _tokenize_search_text(text)}
    return any(term in tokens for term in terms["none"])
