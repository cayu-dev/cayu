"""``POST /usage/rollup/recompute``: check stored usage totals against their events."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Response

from cayu.budgets.aggregates import (
    UsageRollupInconsistent,
    UsageRollupResultTooLarge,
    UsageRollupStoreResult,
)
from cayu.server.contracts import AGGREGATE_ENDPOINT_RESPONSES, UsageRollupRequest
from cayu.sessions.store import SessionStore
from cayu.sessions.usage import UsageRollupQuery
from cayu.sessions.usage_recompute import UsageRecomputeReport, recompute_usage

if TYPE_CHECKING:
    from fastapi.routing import APIRoute


async def read_usage_rollup(store: SessionStore, query: UsageRollupQuery) -> UsageRollupStoreResult:
    """The store's rollup for ``query``, checked against the query's bounds."""

    result = await store.aggregate_usage(query)
    if type(result) is not UsageRollupStoreResult:
        raise UsageRollupInconsistent("Usage aggregate stores must return UsageRollupStoreResult.")
    return result.validate_for_query(query)


def register_usage_recompute_route(
    router: APIRouter,
    *,
    session_store: SessionStore,
    dependencies: list[Any],
    route_class: type[APIRoute],
    redact: Callable[[Any], Any],
) -> None:
    """Add ``POST /usage/rollup/recompute`` to ``router``.

    ``route_class`` bounds the request body as for ``/usage/rollup``, and
    ``redact`` is the server's secret redaction, applied to the whole report.
    """

    async def recompute_usage_rollup(body: UsageRollupRequest, response: Response):
        response.headers["Cache-Control"] = "private, no-store"
        query = UsageRollupQuery(
            series=body.series,
            start_at=body.start_at,
            end_at=body.end_at,
            sessions=body.session_filter,
            group_limit=body.group_limit,
            session_group_limit=body.session_group_limit,
            include_pricing_inputs=body.pricing is not None,
            pricing_input_limit=body.pricing_input_limit,
            pricing_tier_boundaries=()
            if body.pricing is None
            else body.pricing.input_token_tier_boundaries(),
        )
        try:
            report = await recompute_usage(session_store, query)
        except NotImplementedError as exc:
            raise HTTPException(
                status_code=501,
                detail="The configured session store does not support usage aggregates.",
            ) from exc
        except UsageRollupResultTooLarge as exc:
            raise HTTPException(
                status_code=413,
                detail="Usage rollup result exceeds the server byte limit.",
                headers={"Cache-Control": "private, no-store"},
            ) from exc
        return UsageRecomputeReport.model_validate(redact(report.model_dump(mode="json")))

    router.add_api_route(
        "/usage/rollup/recompute",
        recompute_usage_rollup,
        methods=["POST"],
        response_model=UsageRecomputeReport,
        responses=AGGREGATE_ENDPOINT_RESPONSES,
        dependencies=dependencies,
        description=(
            "Check stored usage totals: read the rollup the request describes from the "
            "stored totals and again from its events, read retained tool totals both ways, "
            "and list every difference (leaf values only, redacted like the rollup). The "
            "reads are separate snapshots, so events written between them show up as "
            "differences."
        ),
        route_class_override=route_class,
    )
