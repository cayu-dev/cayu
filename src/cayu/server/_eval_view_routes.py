"""Suite-centred, bounded eval read API; execution and approval stay elsewhere."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, HTTPException, Query

from cayu.evals.read_views import (
    EvalRunCaseFilter,
    EvalRunViewDetail,
    EvalSuiteViewDetail,
    EvalSuiteViewSummary,
    EvalViewPage,
    EvalViewQuery,
    decode_suite_view,
)
from cayu.evals.store import (
    EVAL_STORE_DEFAULT_PAGE_BYTES,
    EVAL_STORE_MAX_CURSOR_BYTES,
    EVAL_STORE_MAX_PAGE_BYTES,
    EVAL_STORE_MAX_PAGE_SIZE,
    EvalStoreResultTooLarge,
)
from cayu.server._eval_view_reader import EvalViewReader, enforce_view_bytes
from cayu.server.contracts import EVALS_ENDPOINT_RESPONSES

if TYPE_CHECKING:
    from fastapi.params import Depends

    from cayu.evals.store import EvalStore
    from cayu.server.evals_registry import EvalTargetRegistry

ViewLimit = Annotated[int, Query(ge=1, le=EVAL_STORE_MAX_PAGE_SIZE)]
ViewCursor = Annotated[str | None, Query(max_length=EVAL_STORE_MAX_CURSOR_BYTES)]
ViewByteLimit = Annotated[int, Query(ge=1024, le=EVAL_STORE_MAX_PAGE_BYTES)]


@contextmanager
def _view_errors():
    try:
        yield
    except NotImplementedError as exc:
        raise HTTPException(status_code=409, detail="Eval suite views are unavailable.") from exc
    except EvalStoreResultTooLarge as exc:
        raise HTTPException(
            status_code=413, detail="Eval view exceeds the requested byte limit."
        ) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail="Invalid eval view query or stored data."
        ) from exc


def register_eval_view_routes(
    router: APIRouter,
    *,
    store: EvalStore | None,
    registry: EvalTargetRegistry | None,
    protected: Sequence[Depends],
) -> None:
    def reader():
        if store is None or registry is None or not store.suite_views:
            raise HTTPException(status_code=409, detail="Eval suite views are unavailable.")
        return EvalViewReader(store, registry)

    @router.get(
        "/evals/suite-views",
        response_model=EvalViewPage[EvalSuiteViewSummary],
        responses=EVALS_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def list_suite_views(
        target_key: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
        cursor: ViewCursor = None,
        limit: ViewLimit = 50,
        max_result_bytes: ViewByteLimit = EVAL_STORE_DEFAULT_PAGE_BYTES,
    ):
        with _view_errors():
            query = EvalViewQuery(
                target_key=target_key, cursor=cursor, limit=limit, max_result_bytes=max_result_bytes
            )
            return enforce_view_bytes(await reader().catalog(query), max_result_bytes)

    @router.get(
        "/evals/suite-views/{suite_key}",
        response_model=EvalSuiteViewDetail,
        responses=EVALS_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def get_suite_view(
        suite_key: str,
        cases_cursor: ViewCursor = None,
        cases_limit: ViewLimit = 50,
        runs_cursor: ViewCursor = None,
        runs_limit: ViewLimit = 20,
        max_result_bytes: ViewByteLimit = EVAL_STORE_DEFAULT_PAGE_BYTES,
    ):
        with _view_errors():
            result = await reader().suite_detail(
                decode_suite_view(suite_key),
                cases_cursor=cases_cursor,
                cases_limit=cases_limit,
                runs_cursor=runs_cursor,
                runs_limit=runs_limit,
            )
            return enforce_view_bytes(result, max_result_bytes)

    @router.get(
        "/evals/run-views/{run_key}",
        response_model=EvalRunViewDetail,
        responses=EVALS_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def get_run_view(
        run_key: str,
        compared_with: ViewCursor = None,
        case_filter: EvalRunCaseFilter | None = None,
        cases_cursor: ViewCursor = None,
        cases_limit: ViewLimit = 50,
        max_result_bytes: ViewByteLimit = EVAL_STORE_DEFAULT_PAGE_BYTES,
    ):
        with _view_errors():
            result = await reader().run_detail(
                run_key,
                compared_with=compared_with,
                case_filter=case_filter,
                cases_cursor=cases_cursor,
                cases_limit=cases_limit,
            )
            return enforce_view_bytes(result, max_result_bytes)
