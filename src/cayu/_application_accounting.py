"""Store-backed usage/cost reports with explicit public identity projection."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from typing import Protocol

from cayu._validation import require_clean_nonblank
from cayu.budgets.pricing import CausalBudgetCostSummary, PriceBook, SessionCostSummary
from cayu.budgets.usage import CausalBudgetUsageSummary, SessionUsageSummary
from cayu.runtime._session_queries import query_all_sessions
from cayu.runtime._usage_accounting import UsageAccountingSnapshot
from cayu.sessions.base import SessionStore
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.queries import SessionOrder, SessionQuery


class _CausalBudgetIdProjector(Protocol):
    def __call__(self, value: str, *, session_ids: Iterable[str]) -> str: ...


async def read_session_usage_snapshot(
    session_id: str,
    *,
    session_store: SessionStore,
    resolve_session: Callable[[str], Awaitable[str]],
    project_session: Callable[[str], str],
) -> UsageAccountingSnapshot:
    """Return exposed session usage with its generation and accounted sequence."""

    try:
        session_id = await resolve_session(require_clean_nonblank(session_id, "session_id"))
        session = await session_store.load(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}") from None
        return expose_session_usage_snapshot(
            session_id,
            await session_store.read_usage_accounting(EventQuery(session_id=session_id)),
            project_session=project_session,
        )
    finally:
        # Dependency representations may contain private configuration.
        del session_store, resolve_session, project_session


def expose_session_usage_snapshot(
    session_id: str, snapshot: UsageAccountingSnapshot, *, project_session: Callable[[str], str]
) -> UsageAccountingSnapshot:
    try:
        summary = snapshot.summary.model_copy(
            update={"session_id": project_session(session_id)},
            deep=True,
        )
        return snapshot.model_copy(update={"summary": summary})
    finally:
        # Dependency representations may contain private configuration.
        del project_session


async def read_causal_budget_usage(
    causal_budget_id: str,
    *,
    session_store: SessionStore,
    resolve_budget: Callable[[str], Awaitable[str]],
    project_session: Callable[[str], str],
    project_budget: _CausalBudgetIdProjector,
) -> CausalBudgetUsageSummary:
    try:
        causal_budget_id = await resolve_budget(causal_budget_id)
        sessions = await query_all_sessions(
            session_store,
            SessionQuery(
                causal_budget_id=causal_budget_id,
                order_by=SessionOrder.CREATED_AT_ASC,
            ),
        )
        if not sessions:
            raise KeyError("Causal budget not found") from None
        session_ids = list(dict.fromkeys(session.id for session in sessions))
        snapshot = await session_store.read_usage_accounting(
            EventQuery(
                causal_budget_id=causal_budget_id,
                session_ids=tuple(session_ids),
            ),
            by_session=True,
        )
        per_session = {row.session_id: row for row in snapshot.session_summaries}
        total = snapshot.summary
        summary = CausalBudgetUsageSummary(
            causal_budget_id=causal_budget_id,
            session_ids=session_ids,
            session_count=len(session_ids),
            model_steps=total.model_steps,
            unmeasured_model_attempts=total.unmeasured_model_attempts,
            tool_calls=total.tool_calls,
            provider_names=total.provider_names,
            models=total.models,
            usage=total.usage,
            session_summaries=tuple(
                per_session.get(session_id, SessionUsageSummary(session_id=session_id))
                for session_id in session_ids
            ),
        )
        public_session_ids = [project_session(session_id) for session_id in summary.session_ids]
        public_causal_budget_id = project_budget(
            causal_budget_id,
            session_ids=(session.id for session in sessions),
        )
        return summary.model_copy(
            update={
                "causal_budget_id": public_causal_budget_id,
                "session_ids": public_session_ids,
                "session_summaries": tuple(
                    session_summary.model_copy(
                        update={"session_id": project_session(session_summary.session_id)},
                        deep=True,
                    )
                    for session_summary in summary.session_summaries
                ),
            },
            deep=True,
        )
    finally:
        # Dependency representations may contain private configuration.
        del session_store, resolve_budget, project_session, project_budget


async def read_session_cost(
    session_id: str,
    pricing: PriceBook,
    *,
    session_store: SessionStore,
    resolve_session: Callable[[str], Awaitable[str]],
    project_session: Callable[[str], str],
    currency: str = "USD",
) -> SessionCostSummary:
    try:
        session_id = await resolve_session(require_clean_nonblank(session_id, "session_id"))
        session = await session_store.load(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}") from None
        snapshot = await session_store.read_cost_accounting(
            EventQuery(session_id=session_id),
            pricing,
            currency=currency,
            details=True,
        )
        summary = snapshot.details
        if summary is None:
            raise RuntimeError("Cost accounting store omitted requested details.")
        return summary.model_copy(
            update={"session_id": project_session(session_id)},
            deep=True,
        )
    finally:
        # Dependency representations may contain private configuration.
        del session_store, resolve_session, project_session


async def read_causal_budget_cost(
    causal_budget_id: str,
    pricing: PriceBook,
    *,
    session_store: SessionStore,
    resolve_budget: Callable[[str], Awaitable[str]],
    project_session: Callable[[str], str],
    project_budget: _CausalBudgetIdProjector,
    currency: str = "USD",
) -> CausalBudgetCostSummary:
    try:
        causal_budget_id = await resolve_budget(causal_budget_id)
        sessions = await query_all_sessions(
            session_store,
            SessionQuery(
                causal_budget_id=causal_budget_id,
                order_by=SessionOrder.CREATED_AT_ASC,
            ),
        )
        if not sessions:
            raise KeyError("Causal budget not found") from None
        from cayu.runtime._cost_accounting import causal_cost_summary

        session_ids = list(dict.fromkeys(session.id for session in sessions))
        snapshot = await session_store.read_cost_accounting(
            EventQuery(causal_budget_id=causal_budget_id, session_ids=tuple(session_ids)),
            pricing,
            currency=currency,
            details=True,
            by_session=True,
        )
        summary = causal_cost_summary(snapshot, causal_budget_id, session_ids)
        public_causal_budget_id = project_budget(
            causal_budget_id,
            session_ids=(session.id for session in sessions),
        )
        return summary.model_copy(
            update={
                "causal_budget_id": public_causal_budget_id,
                "session_ids": [project_session(session_id) for session_id in summary.session_ids],
                "session_costs": tuple(
                    session_cost.model_copy(
                        update={"session_id": project_session(session_cost.session_id)},
                        deep=True,
                    )
                    for session_cost in summary.session_costs
                ),
            },
            deep=True,
        )
    finally:
        # Dependency representations may contain private configuration.
        del session_store, resolve_budget, project_session, project_budget
