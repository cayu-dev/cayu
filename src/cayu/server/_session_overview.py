"""Control-plane projection of the shared session overview.

Store joins stay in sessions.overview; application-owned projection and artifact
registration are injected here by router assembly.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, HTTPException, Response
from pydantic import ValidationError

from cayu.artifacts.base import (
    ArtifactListResult,
    ArtifactScope,
    ArtifactStore,
    ArtifactStoreUnavailableError,
)
from cayu.budgets.pricing import PriceBook
from cayu.server.contracts import (
    SessionOverviewCost,
    SessionOverviewCountsResponse,
    SessionOverviewResponse,
    SessionOverviewTokens,
    SessionOverviewUses,
)
from cayu.sessions._browser_control_checkpoint import browser_control_checkpoint_read_scope
from cayu.sessions._run_identity import is_eval_trial, session_run_id, session_run_lineage
from cayu.sessions.checkpoints import BROWSER_CONTROLS_CHECKPOINT_KEY
from cayu.sessions.overview import (
    SessionOverviewFacts,
    SessionOverviewReader,
    SessionOverviewStatus,
    SessionOverviewUnavailable,
)
from cayu.sessions.queries import SessionOrder, SessionQuery
from cayu.sessions.records import Session
from cayu.sessions.store import SessionStore
from cayu.tools._browser_usage import has_retained_browser_usage
from cayu.tools.browser_control import BrowserControlCheckpoint


class SessionOverviewService:
    def __init__(
        self,
        *,
        store: SessionStore,
        pricing: PriceBook | None,
        project_session: Callable[[str], str],
        redact: Callable[[Any], Any],
        artifact_stores: Callable[[], dict[str, ArtifactStore]],
    ) -> None:
        self.store = store
        self.pricing = pricing
        self.project_session = project_session
        self.redact = redact
        self.artifact_stores = artifact_stores

    async def read(
        self, sessions: list[Session]
    ) -> tuple[list[Session], list[SessionOverviewResponse]]:
        expected = {session.id: session.instance_id for session in sessions}
        facts = await SessionOverviewReader(self.store, pricing=self.pricing).read(tuple(expected))
        kept: list[Session] = []
        result: list[SessionOverviewResponse] = []
        for fact in facts:
            session = fact.session
            if session.instance_id != expected.get(session.id):
                continue
            if not fact.pending_complete:
                raise SessionOverviewUnavailable("Pending session evidence is unavailable.")
            try:
                uses = SessionOverviewUses(
                    browser=await self._uses_browser(session),
                    subagents=await self._subagent_count(session),
                    artifacts=await self._artifact_count(session),
                )
            except KeyError:
                current = await self.store.load(session.id)
                if current is None or current.instance_id != session.instance_id:
                    continue
                raise
            current = await self.store.load(session.id)
            if current is None or current.instance_id != session.instance_id:
                continue
            result.append(self._project(fact, uses))
            kept.append(session)
        return kept, result

    async def _uses_browser(self, session: Session) -> bool:
        with browser_control_checkpoint_read_scope(session.id):
            checkpoint = await self.store.load_checkpoint(session.id)
        raw = None if checkpoint is None else checkpoint.get(BROWSER_CONTROLS_CHECKPOINT_KEY)
        if raw is None:
            return await has_retained_browser_usage(self.store, session)
        try:
            controls = BrowserControlCheckpoint.model_validate(raw)
        except ValidationError:
            raise SessionOverviewUnavailable("Browser usage evidence is unavailable.") from None
        if any(
            record.identity.session_id == session.id
            and record.identity.session_instance_id == session.instance_id
            for record in controls.records
        ):
            return True
        return await has_retained_browser_usage(self.store, session)

    async def _subagent_count(self, session: Session) -> int:
        run_id = session_run_id(session)
        seen: set[str] = set()
        cursors: set[str] = set()
        cursor = None
        while True:
            page = await self.store.list_sessions(
                SessionQuery(
                    parent_session_id=session.id,
                    limit=200,
                    cursor=cursor,
                    order_by=SessionOrder.CREATED_AT_ASC,
                )
            )
            for child in page.sessions:
                if (
                    session_run_lineage(child)[1] == run_id
                    and child.invocation.source.value != "fork"
                ):
                    seen.add(session_run_id(child))
            if page.next_cursor is None:
                return len(seen)
            if page.next_cursor in cursors:
                raise SessionOverviewUnavailable("Subagent paging did not advance.")
            cursors.add(page.next_cursor)
            cursor = page.next_cursor

    async def _artifact_count(self, session: Session) -> int:
        # Registration owner deduplicates the same store shared by environments
        # and refuses distinct stores with conflicting IDs. IDs in different
        # stores denote different artifacts, exactly as in GET /artifacts.
        total = 0
        for store in self.artifact_stores().values():
            result = await store.list(scope=ArtifactScope.SESSION, session_id=session.id, limit=1)
            if type(result) is not ArtifactListResult:
                raise SessionOverviewUnavailable("Artifact count evidence is unavailable.")
            # Count evidence is scalar; do not serialize unrelated metadata.
            if (
                type(result.total_count) is not int
                or result.total_count < len(result.artifacts)
                or (not result.truncated and result.total_count != len(result.artifacts))
                or any(
                    artifact.scope != ArtifactScope.SESSION or artifact.session_id != session.id
                    for artifact in result.artifacts
                )
            ):
                raise SessionOverviewUnavailable("Artifact store cannot provide a complete count.")
            total += result.total_count
        return total

    def _text(self, value: object, *, limit: int) -> str | None:
        # Reject wrong types before redaction/serialization can invoke repr/str.
        if type(value) is not str or not value.strip() or len(value) > limit:
            return None
        if any(
            ord(char) < 32 or ord(char) == 127 or 0xD800 <= ord(char) <= 0xDFFF for char in value
        ):
            return None
        redacted = self.redact(value)
        return redacted if type(redacted) is str and len(redacted) <= limit else None

    def _project(
        self, facts: SessionOverviewFacts, uses: SessionOverviewUses
    ) -> SessionOverviewResponse:
        session, usage, cost = facts.session, facts.usage, facts.cost
        metrics = None if usage is None else usage.usage
        tone = session.metadata.get("cayu:summary_tone")
        return SessionOverviewResponse(
            session_id=self.project_session(session.id),
            run_id=session_run_id(session),
            state=facts.status,
            reason=self._text(facts.outcome.reason, limit=1024) or "unknown",
            title=self._text(session.metadata.get("cayu:title"), limit=256),
            summary=self._text(session.metadata.get("cayu:summary"), limit=4096),
            summary_tone=tone if type(tone) is str and tone in {"neutral", "attention"} else None,
            started_at=session.created_at,
            ended_at=facts.ended_at,
            last_activity_at=session.last_activity_at,
            model_steps=0 if usage is None else usage.model_steps,
            tool_calls=0 if usage is None else usage.tool_calls,
            tokens=SessionOverviewTokens(
                input=0 if metrics is None else metrics.input_tokens,
                output=0 if metrics is None else metrics.output_tokens,
                cached=0 if metrics is None else metrics.cache.read_tokens,
            ),
            cost=None
            if self.pricing is None
            else SessionOverviewCost(
                amount=Decimal(0) if cost is None else cost.total_cost,
                currency="USD" if cost is None else cost.currency,
                complete=cost is None
                or (cost.unpriced_model_steps == 0 and cost.unpriced_auxiliary_attempts == 0),
            ),
            uses=uses,
            eval=is_eval_trial(session),
            forked=facts.forked,
        )


def overview_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, NotImplementedError):
        return HTTPException(501, detail="Session overview is unsupported by this deployment.")
    return HTTPException(503, detail="Session overview evidence is unavailable.")


def register_session_overview_counts(
    router: APIRouter,
    *,
    service: SessionOverviewService,
    protected: list[Any],
) -> None:
    @router.get(
        "/sessions/overview/counts",
        response_model=SessionOverviewCountsResponse,
        dependencies=protected,
    )
    async def get_session_overview_counts(
        response: Response,
        since: datetime | None = None,
        until: datetime | None = None,
        include_eval: bool = False,
        include_forks: bool = False,
    ) -> SessionOverviewCountsResponse:
        try:
            counts = await SessionOverviewReader(service.store).counts(
                since=since,
                until=until,
                include_eval=include_eval,
                include_forks=include_forks,
            )
        except ValueError:
            raise HTTPException(
                422, detail="Overview counts require aware since < until."
            ) from None
        except (
            SessionOverviewUnavailable,
            NotImplementedError,
            OSError,
            ArtifactStoreUnavailableError,
        ) as exc:
            raise overview_http_error(exc) from exc
        response.headers["Cache-Control"] = "private, no-store"
        return SessionOverviewCountsResponse(
            since=counts.since,
            until=counts.until,
            running_now=counts.running_now,
            total_count=counts.total_count,
            failed=counts.counts_by_state[SessionOverviewStatus.FAILED],
            counts_by_state=counts.counts_by_state,
        )
