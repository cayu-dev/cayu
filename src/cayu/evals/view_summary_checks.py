"""Check stored Evals view summaries against a recompute from the documents.

The summary tables are a projection of immutable result documents. These
functions recompute a result's rows from its stored document, with the same
``summarize_result`` the store's writers use, and compare or record them. They
need only an ``EvalStore``; `cayu eval runs check RUN_ID` and
`cayu eval runs rebuild-summaries` are thin commands over them.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, StrictBool, StrictInt, StrictStr

from cayu.evals.view_summaries import EvalCaseOutcomeQuery, EvalResultViewRows, summarize_result

if TYPE_CHECKING:
    from cayu.evals.store import EvalStore


class EvalViewSummaryCheckV1(BaseModel):
    """Whether a result's stored summary rows equal a recompute from its document."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    revision: StrictStr
    stored: StrictBool
    matches: StrictBool
    differences: tuple[StrictStr, ...] = ()


class EvalViewSummaryRebuildV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    written: StrictInt
    remaining: StrictBool


async def recompute_result_view_rows(store: EvalStore, revision: str) -> EvalResultViewRows:
    """A result's summary and case outcomes, derived from its stored document."""
    record = await store.load_result_record(revision)
    result = await store.load_result_by_revision(revision)
    if record is None or result is None:
        raise LookupError(f"Eval result {revision} is not stored.")
    return await asyncio.to_thread(summarize_result, result, document_bytes=record.document_bytes)


async def check_result_view_summary(store: EvalStore, revision: str) -> EvalViewSummaryCheckV1:
    expected = await recompute_result_view_rows(store, revision)
    stored = (await store.load_result_view_summaries((revision,))).get(revision)
    page = await store.list_case_outcomes(EvalCaseOutcomeQuery(revision=revision))
    if stored is None or page is None:
        return EvalViewSummaryCheckV1(revision=revision, stored=False, matches=False)
    differences = []
    if stored != expected.summary:
        differences.extend(
            f"summary.{field}"
            for field in type(stored).model_fields
            if getattr(stored, field) != getattr(expected.summary, field)
        )
    actual = {item.case_id: item for item in page.items}
    wanted = {item.case_id: item for item in expected.outcomes}
    differences.extend(
        f"case {case_id}"
        for case_id in sorted(set(actual) | set(wanted))
        if actual.get(case_id) != wanted.get(case_id)
    )
    return EvalViewSummaryCheckV1(
        revision=revision,
        stored=True,
        matches=not differences,
        differences=tuple(differences),
    )


async def rebuild_result_view_summaries(
    store: EvalStore, *, limit: int = 100
) -> EvalViewSummaryRebuildV1:
    """Record rows for up to ``limit`` results that have none, oldest first."""
    revisions = await store.list_results_without_view_summaries(limit)
    written = 0
    for revision in revisions:
        rows = await recompute_result_view_rows(store, revision)
        written += await store.save_result_view_rows(rows)
    remaining = bool(revisions) and bool(await store.list_results_without_view_summaries(1))
    return EvalViewSummaryRebuildV1(written=written, remaining=remaining)


__all__ = [
    "EvalViewSummaryCheckV1",
    "EvalViewSummaryRebuildV1",
    "check_result_view_summary",
    "rebuild_result_view_summaries",
    "recompute_result_view_rows",
]
