"""Control-plane projection of the shared session overview.

Store joins stay in sessions.overview; application-owned projection and artifact
registration are injected here by router assembly, or by the CLI for a local
store.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime
from decimal import Decimal
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import ValidationError

from cayu._time_buckets import BucketUnit, TimeSeriesSpec
from cayu.artifacts.base import ArtifactStore, ArtifactStoreUnavailableError
from cayu.budgets.pricing import PriceBook, SessionCostTotals
from cayu.budgets.usage import SessionUsageSummary
from cayu.messages import Message, MessageRole, TextPart
from cayu.server._artifact_index_reads import ArtifactIndexReads
from cayu.server._session_overview_query import SessionOverviewSelection
from cayu.server._session_summaries import SessionSummaries, SummaryDocument
from cayu.server.contracts import (
    MAX_SESSION_FIRST_MESSAGE_LENGTH,
    SessionOverviewCost,
    SessionOverviewCountsResponse,
    SessionOverviewResponse,
    SessionOverviewTokens,
    SessionOverviewUses,
)
from cayu.session_metadata import (
    MAX_SESSION_SUMMARY_LENGTH,
    MAX_SESSION_TITLE_LENGTH,
    SUMMARY_METADATA_KEY,
    SUMMARY_TONE_METADATA_KEY,
    SUMMARY_TONES,
    TITLE_METADATA_KEY,
    validate_display_text,
)
from cayu.sessions._browser_control_checkpoint import browser_control_checkpoint_read_scope
from cayu.sessions._label_filter_syntax import parse_label_filters, parse_label_selectors
from cayu.sessions._run_identity import is_eval_trial, session_run_id
from cayu.sessions.checkpoints import BROWSER_CONTROLS_CHECKPOINT_KEY
from cayu.sessions.overview import (
    SessionCostCache,
    SessionOverviewFacts,
    SessionOverviewReader,
    SessionOverviewStatus,
    SessionOverviewUnavailable,
    SessionUsageCache,
    gather_in_order,
)
from cayu.sessions.queries import SessionAggregateFilter
from cayu.sessions.records import Session
from cayu.sessions.store import SessionStore
from cayu.sessions.transcript_queries import TranscriptQuery
from cayu.tools._browser_usage import retained_browser_usage
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
        artifact_index: ArtifactIndexReads | None = None,
    ) -> None:
        self.store = store
        self.pricing = pricing
        self.project_session = project_session
        self.redact = redact
        self.artifact_stores = artifact_stores
        self.artifact_index = artifact_index
        # The service reads one store with one price book for its lifetime, so
        # a page sums usage and prices cost only for sessions whose events
        # changed since an earlier page read them.
        self._cost_cache = SessionCostCache()
        self._usage_cache = SessionUsageCache()
        self.summaries = SessionSummaries(store, uses_browser=uses_browser)

    async def read(
        self,
        sessions: list[Session],
        *,
        selection: SessionOverviewSelection | None = None,
    ) -> tuple[list[Session], list[SessionOverviewResponse]]:
        """Overviews of the listed sessions, from their stored summary rows.

        A store without summary rows, or a read under scoped access bounds, is
        read with the reference reader instead, as ``read_reference``.
        """

        selection = SessionOverviewSelection() if selection is None else selection
        documents = await self.summaries.documents(sessions)
        if documents is None:
            return await self.read_reference(sessions, selection=selection)
        selected: list[tuple[Session, SummaryDocument]] = []
        for session in sessions:
            document = documents.get(session.id)
            if document is None:
                continue
            if (not selection.include_eval and is_eval_trial(session)) or (
                not selection.include_forks and document.forked
            ):
                continue
            if not document.pending_complete:
                raise SessionOverviewUnavailable("Pending session evidence is unavailable.")
            if selection.states and document.state not in selection.states:
                continue
            selected.append((session, document))
        if not selected:
            return [], []
        listed = [session for session, _ in selected]
        # TODO: price from cayu_session_usage, as the summary
        # row's per-session totals, instead of reading cost-bearing events here.
        costs = await SessionOverviewReader(
            self.store, pricing=self.pricing, cost_cache=self._cost_cache
        ).costs(listed)
        # TODO: read artifact_count from the summary row
        # once the artifact index keeps it.
        artifacts = await self._artifact_counts(tuple(session.id for session in listed))
        current = await self._current(listed)
        kept = [(session, document) for session, document in selected if session.id in current]
        return (
            [session for session, _ in kept],
            [
                self._project_values(
                    session,
                    status=document.state,
                    reason=document.reason,
                    error=document.error,
                    ended_at=document.ended_at,
                    usage=document.usage,
                    cost=costs.get(session.id),
                    forked=document.forked,
                    uses=SessionOverviewUses(
                        browser=document.browser,
                        subagents=document.subagents,
                        artifacts=artifacts[session.id],
                    ),
                )
                for session, document in kept
            ],
        )

    async def read_reference(
        self,
        sessions: list[Session],
        *,
        selection: SessionOverviewSelection | None = None,
    ) -> tuple[list[Session], list[SessionOverviewResponse]]:
        """``read`` from the reference reader, without summary rows."""

        selection = SessionOverviewSelection() if selection is None else selection
        facts = await SessionOverviewReader(
            self.store,
            pricing=self.pricing,
            cost_cache=self._cost_cache,
            usage_cache=self._usage_cache,
        ).read_listed(sessions)
        selected: list[SessionOverviewFacts] = []
        for fact in facts:
            session = fact.session
            if (not selection.include_eval and is_eval_trial(session)) or (
                not selection.include_forks and fact.forked
            ):
                continue
            if not fact.pending_complete:
                raise SessionOverviewUnavailable("Pending session evidence is unavailable.")
            if selection.states and fact.status not in selection.states:
                continue
            selected.append(fact)

        # Each read covers every selected session, so a page costs the same
        # number of store reads however many sessions it holds.
        while selected:
            sessions = [fact.session for fact in selected]
            try:
                uses = await self._uses(sessions)
            except KeyError:
                # Only a confirmed missing/replaced incarnation explains the
                # error. Drop those rows and read the rest again; an unrelated
                # store failure must propagate.
                current = await self._current(sessions)
                if len(current) == len(sessions):
                    raise
                selected = [fact for fact in selected if fact.session.id in current]
                continue
            current = await self._current(sessions)
            kept = [fact for fact in selected if fact.session.id in current]
            return (
                [fact.session for fact in kept],
                [self._project(fact, uses[fact.session.id]) for fact in kept],
            )
        return [], []

    async def _current(self, sessions: list[Session]) -> set[str]:
        """The sessions whose selected incarnation is still current, from one read."""

        current = await self.store.load_session_instance_ids(
            tuple(session.id for session in sessions)
        )
        return {
            session.id for session in sessions if current.get(session.id) == session.instance_id
        }

    async def _uses(self, sessions: list[Session]) -> dict[str, SessionOverviewUses]:
        session_ids = tuple(session.id for session in sessions)
        browser = await uses_browser(self.store, sessions)
        subagents = await self.store.count_child_runs(sessions)
        artifacts = await self._artifact_counts(session_ids)
        return {
            session.id: SessionOverviewUses(
                browser=browser[session.id],
                subagents=subagents[session.id],
                artifacts=artifacts[session.id],
            )
            for session in sessions
        }

    async def _artifact_counts(self, session_ids: tuple[str, ...]) -> dict[str, int]:
        # Registration owner deduplicates the same store shared by environments
        # and refuses distinct stores with conflicting IDs. IDs in different
        # stores denote different artifacts, exactly as in GET /artifacts.
        totals = dict.fromkeys(session_ids, 0)
        stores = self.artifact_stores()
        indexed: set[str] = set()
        if self.artifact_index is not None and stores:
            # Stores whose index rows are filled are counted in one indexed read;
            # only the others are still walked.
            indexed = await self.artifact_index.filled(stores)
        if indexed:
            try:
                counts = await self.store.count_indexed_session_artifacts(
                    tuple(indexed), session_ids
                )
            except Exception:
                raise SessionOverviewUnavailable("Artifact counts are unavailable.") from None
            if type(counts) is not dict or counts.keys() != totals.keys():
                raise SessionOverviewUnavailable("Artifact store cannot provide a complete count.")
            for session_id, count in counts.items():
                totals[session_id] += count
        for store_id, store in stores.items():
            if store_id in indexed:
                continue
            counts = await store.count_session_artifacts(session_ids)
            # Count evidence is scalar; do not serialize unrelated metadata.
            if (
                type(counts) is not dict
                or counts.keys() != totals.keys()
                or any(type(count) is not int or count < 0 for count in counts.values())
            ):
                raise SessionOverviewUnavailable("Artifact store cannot provide a complete count.")
            for session_id, count in counts.items():
                totals[session_id] += count
        return totals

    async def with_first_messages(
        self, sessions: Sequence[Session], overviews: Sequence[SessionOverviewResponse]
    ) -> list[SessionOverviewResponse]:
        """Add each session's bounded first user text, read for the whole page at once.

        The caller rechecks every row's incarnation after these reads, as it does
        for the other optional sibling reads, so a replaced session is dropped.
        """

        try:
            records = await self.store.query_first_transcript_records(
                tuple(session.id for session in sessions), MessageRole.USER
            )
        except Exception:
            # One session's failure must degrade only its own row, so read
            # each row on its own. Cancellation is not an Exception.
            return await self._first_messages_one_by_one(sessions, overviews)
        previews = []
        for session, overview in zip(sessions, overviews, strict=True):
            record = records.get(session.id)
            try:
                text = None if record is None else self._preview_text(record.message)
            except Exception:
                previews.append(_without_preview(overview))
                continue
            previews.append(_with_preview(overview, text))
        return previews

    async def _first_messages_one_by_one(
        self, sessions: Sequence[Session], overviews: Sequence[SessionOverviewResponse]
    ) -> list[SessionOverviewResponse]:
        async def preview(session: Session, overview: SessionOverviewResponse):
            try:
                text = await self._first_message(session)
            except Exception:
                # Preview availability must not decide whether a session is
                # listed. Cancellation still propagates through the read group.
                return _without_preview(overview)
            return _with_preview(overview, text)

        return await gather_in_order(
            [
                lambda session=session, overview=overview: preview(session, overview)
                for session, overview in zip(sessions, overviews, strict=True)
            ]
        )

    async def _first_message(self, session: Session) -> str | None:
        # The same store query the transcript route makes for role=user&limit=1;
        # the store applies the caller's data bounds.
        try:
            page = await self.store.query_transcript(
                TranscriptQuery(session_id=session.id, role=MessageRole.USER, limit=1)
            )
        except KeyError:
            current = await self.store.load(session.id)
            if current is None or current.instance_id != session.instance_id:
                return None
            raise
        if not page.records:
            return None
        return self._preview_text(page.records[0].message)

    def _preview_text(self, message: Message) -> str | None:
        # Redact each part as the transcript route does, then the joined text, so a
        # secret split across parts is still caught. Cut only after redacting.
        text = self.redact(
            "".join(
                str(self.redact(part.text)) for part in message.content if type(part) is TextPart
            )
        )
        text = text.strip() if type(text) is str else ""
        if not text:
            return None
        if len(text) > MAX_SESSION_FIRST_MESSAGE_LENGTH:
            text = text[: MAX_SESSION_FIRST_MESSAGE_LENGTH - 1] + "…"
        return text

    def _text(self, value: object, *, limit: int) -> str | None:
        # The writer and reader share the same bounded validation boundary.
        try:
            text = validate_display_text(value, limit=limit)
        except ValueError:
            return None
        redacted = self.redact(text)
        try:
            return validate_display_text(redacted, limit=limit)
        except ValueError:
            return None

    def project_title(self, session: Session) -> str | None:
        """Share the current validated, redacted title with public text search."""
        return self._text(session.metadata.get(TITLE_METADATA_KEY), limit=MAX_SESSION_TITLE_LENGTH)

    def _project(
        self, facts: SessionOverviewFacts, uses: SessionOverviewUses
    ) -> SessionOverviewResponse:
        error = facts.outcome.details.get("error")
        return self._project_values(
            facts.session,
            status=facts.status,
            reason=facts.outcome.reason,
            error=error if type(error) is str else None,
            ended_at=facts.ended_at,
            usage=facts.usage,
            cost=facts.cost,
            forked=facts.forked,
            uses=uses,
        )

    def _project_values(
        self,
        session: Session,
        *,
        status: SessionOverviewStatus,
        reason: str,
        error: str | None,
        ended_at: datetime | None,
        usage: SessionUsageSummary | None,
        cost: SessionCostTotals | None,
        forked: bool,
        uses: SessionOverviewUses,
    ) -> SessionOverviewResponse:
        metrics = None if usage is None else usage.usage
        tone = session.metadata.get(SUMMARY_TONE_METADATA_KEY)
        reason = self._text(reason, limit=1024) or "unknown"
        if status == "failed":
            # Runtime outcome codes remain stable. Only this public display
            # projection explains the error, after redacting the complete value
            # so truncation cannot expose a prefix of a registered secret.
            error = self.redact(error) if type(error) is str else None
            text = " ".join(error.split()) if type(error) is str else ""
            if len(text) > 1024:
                text = text[:1023] + "…"
            try:
                reason = validate_display_text(text, limit=1024)
            except ValueError:
                reason = "This session failed."
        return SessionOverviewResponse(
            session_id=self.project_session(session.id),
            run_id=session_run_id(session),
            state=status,
            reason=reason,
            title=self.project_title(session),
            summary=self._text(
                session.metadata.get(SUMMARY_METADATA_KEY), limit=MAX_SESSION_SUMMARY_LENGTH
            ),
            summary_tone=tone if type(tone) is str and tone in SUMMARY_TONES else None,
            started_at=session.created_at,
            ended_at=ended_at,
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
            forked=forked,
        )


async def uses_browser(store: Any, sessions: list[Session]) -> dict[str, bool]:
    """Whether each session used the browser: its checkpoint's controls or a retained receipt."""

    session_ids = tuple(session.id for session in sessions)
    with browser_control_checkpoint_read_scope(*session_ids):
        checkpoints = await store.load_checkpoint_fields(
            session_ids, (BROWSER_CONTROLS_CHECKPOINT_KEY,)
        )
    uses: dict[str, bool] = {}
    for session in sessions:
        checkpoint = checkpoints.get(session.id)
        raw = None if checkpoint is None else checkpoint.get(BROWSER_CONTROLS_CHECKPOINT_KEY)
        if raw is None:
            continue
        try:
            controls = BrowserControlCheckpoint.model_validate(raw)
        except ValidationError:
            raise SessionOverviewUnavailable("Browser usage evidence is unavailable.") from None
        if any(
            record.identity.session_id == session.id
            and record.identity.session_instance_id == session.instance_id
            for record in controls.records
        ):
            uses[session.id] = True
    unsettled = [session for session in sessions if session.id not in uses]
    uses.update(await retained_browser_usage(store, unsettled))
    return uses


def _with_preview(overview: SessionOverviewResponse, text: str | None) -> SessionOverviewResponse:
    return overview.model_copy(update={"first_message": text, "first_message_unavailable": False})


def _without_preview(overview: SessionOverviewResponse) -> SessionOverviewResponse:
    return overview.model_copy(update={"first_message": None, "first_message_unavailable": True})


def overview_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, NotImplementedError):
        return HTTPException(501, detail="Session overview is unsupported by this deployment.")
    return HTTPException(503, detail="Session overview evidence is unavailable.")


# Resolves public ``parent_session_id`` and ``causal_budget_id`` filters to the
# private ids stored lineage keeps, raising HTTPException for a bad value.
LineageFilterResolver = Callable[..., Awaitable[tuple[str | None, str | None]]]


def register_session_overview_counts(
    router: APIRouter,
    *,
    service: SessionOverviewService,
    protected: list[Any],
    resolve_lineage_filters: LineageFilterResolver,
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
        agent_name: str | None = None,
        environment_name: str | None = None,
        provider_name: str | None = None,
        model: str | None = None,
        parent_session_id: str | None = None,
        causal_budget_id: str | None = None,
        label: Annotated[list[str] | None, Query()] = None,
        label_selector: Annotated[list[str] | None, Query()] = None,
        bucket_unit: BucketUnit | None = None,
        timezone: str = "UTC",
    ) -> SessionOverviewCountsResponse:
        # /api/sessions exposes lineage under public aliases; resolve them as the
        # list and summary routes do before they reach private store queries.
        parent_session_id, causal_budget_id = await resolve_lineage_filters(
            parent_session_id=parent_session_id,
            causal_budget_id=causal_budget_id,
        )
        try:
            # The same current-session filter the Usage rollup takes as session_filter,
            # with labels and selectors in the GET /api/sessions text syntax.
            filters = SessionAggregateFilter(
                agent_name=agent_name,
                environment_name=environment_name,
                provider_name=provider_name,
                model=model,
                parent_session_id=parent_session_id,
                causal_budget_id=causal_budget_id,
                labels=parse_label_filters(label),
                label_selectors=parse_label_selectors(label_selector),
            )
            counts = await SessionOverviewReader(service.store, pricing=service.pricing).counts(
                since=since,
                until=until,
                include_eval=include_eval,
                include_forks=include_forks,
                filters=filters,
                series=None
                if bucket_unit is None
                else TimeSeriesSpec(unit=bucket_unit, timezone=timezone),
            )
        except ValueError:
            raise HTTPException(
                422,
                detail="Overview counts require valid filters, an aware window and a bounded series.",
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
            series=counts.series,
            since=counts.since,
            until=counts.until,
            running_now=counts.running_now,
            total_count=counts.total_count,
            failed=counts.counts_by_state[SessionOverviewStatus.FAILED],
            counts_by_state=counts.counts_by_state,
            tokens=SessionOverviewTokens(
                input=counts.input_tokens, output=counts.output_tokens, cached=counts.cached_tokens
            ),
            tokens_complete=counts.tokens_complete,
            cost=None
            if counts.cost is None
            else SessionOverviewCost(
                amount=counts.cost, currency="USD", complete=counts.cost_complete
            ),
        )
