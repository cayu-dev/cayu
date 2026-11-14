"""Suite-centred, bounded eval read API; execution and approval stay elsewhere."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse

from cayu.evals.read_views import (
    EvalReportedRunDetail,
    EvalReportedSuiteDetail,
    EvalReportedSuiteView,
    EvalRunCaseFilter,
    EvalRunViewDetail,
    EvalSuiteViewDetail,
    EvalSuiteViewIdentity,
    EvalSuiteViewSummary,
    EvalViewPage,
    EvalViewQuery,
    decode_run_view,
    decode_suite_view,
)
from cayu.evals.store import (
    EVAL_STORE_DEFAULT_PAGE_BYTES,
    EVAL_STORE_MAX_CURSOR_BYTES,
    EVAL_STORE_MAX_PAGE_BYTES,
    EVAL_STORE_MAX_PAGE_SIZE,
    EvalBaselineKey,
    EvalStoreResultTooLarge,
)
from cayu.evals.view_summaries import EvalRunViewPollV1, run_view_version
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
        response_model=EvalViewPage[EvalSuiteViewSummary | EvalReportedSuiteView],
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
        "/evals/suite-views/resolve-corpus",
        response_model=EvalSuiteViewIdentity,
        responses=EVALS_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def resolve_corpus_suite_view(
        corpus: Annotated[str, Query(pattern=r"^sha256:[0-9a-f]{64}$")],
        suite: Annotated[str, Query(min_length=1, max_length=128)],
        target: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    ):
        with _view_errors():
            return await reader().resolve_corpus_suite(corpus, suite, target)

    @router.get(
        "/evals/suite-views/{suite_key}",
        response_model=EvalSuiteViewDetail | EvalReportedSuiteDetail,
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
        response_model=EvalRunViewDetail | EvalReportedRunDetail,
        responses=EVALS_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def get_run_view(
        run_key: str,
        compared_with: ViewCursor = None,
        placement: ViewCursor = None,
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
                max_bytes=max_result_bytes,
            )
            if placement is not None:
                identity = decode_suite_view(placement)
                view_reader = reader()
                await view_reader.suite_summary(identity)
                result = result.model_copy(
                    update={
                        "run": await view_reader.with_history_scope(result.run, identity),
                    }
                )
            return enforce_view_bytes(result, max_result_bytes)

    @router.get(
        "/evals/run-views/{run_key}/status",
        response_model=EvalRunViewPollV1,
        responses={**EVALS_ENDPOINT_RESPONSES, 304: {"description": "The run view is unchanged."}},
        dependencies=protected,
    )
    async def get_run_view_status(run_key: str, request: Request):
        """A live run view's poll: one run row, answered with 304 while it is unchanged."""
        with _view_errors():
            view_reader = reader()
            source, run_id = decode_run_view(run_key)
            if source != "run":
                raise HTTPException(status_code=422, detail="Only native runs have a live status.")
            poll = await view_reader.store.load_run_view_poll(run_id)
            if poll is None:
                raise HTTPException(status_code=404, detail="Eval run not found.")
            view_reader.target(poll.target_key)
            baseline = await view_reader.store.load_baseline(
                EvalBaselineKey(
                    target_key=poll.target_key,
                    corpus_revision=poll.corpus_revision,
                    suite_id=poll.suite_id,
                )
            )
            poll = poll.model_copy(
                update={
                    "version": run_view_version(
                        poll.version,
                        None
                        if baseline is None
                        else [baseline.result_revision, baseline.generation],
                    )
                }
            )
            etag = f'"{poll.version.removeprefix("sha256:")}"'
            headers = {"ETag": etag, "Cache-Control": "private, no-cache"}
            if etag in {
                item.strip() for item in request.headers.get("if-none-match", "").split(",")
            }:
                return Response(status_code=304, headers=headers)
            return JSONResponse(poll.model_dump(mode="json"), headers=headers)

    @router.get("/evals/reports/{run_key}/download", dependencies=protected)
    async def download_report(run_key: str):
        from cayu.evals.reported import report_canonical_bytes

        with _view_errors():
            record = await reader().reports.record(run_key)
            return Response(
                content=report_canonical_bytes(record.report),
                media_type="application/json",
                headers={
                    "Cache-Control": "private, no-store",
                    "Content-Disposition": 'attachment; filename="eval-report.json"',
                },
            )
