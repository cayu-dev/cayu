"""Evaluation corpus import, catalog and download HTTP routes."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response

from cayu.evals.corpus import EvalCorpusDocument, eval_corpus_to_json
from cayu.evals.execution import _validate_corpus_target_compatibility, evaluation_target_identity
from cayu.evals.store import (
    EVAL_STORE_DEFAULT_PAGE_BYTES,
    EVAL_STORE_DEFAULT_PAGE_SIZE,
    EVAL_STORE_MAX_CURSOR_BYTES,
    EVAL_STORE_MAX_IDENTIFIER_CHARS,
    EVAL_STORE_MAX_PAGE_BYTES,
    EVAL_STORE_MAX_PAGE_SIZE,
    EvalCaseCatalogPage,
    EvalCaseCatalogQuery,
    EvalCatalogQuery,
    EvalCorpusCatalogEntry,
    EvalCorpusCatalogPage,
    EvalCorpusConflict,
    EvalStorePublicationRejected,
    EvalStoreResultTooLarge,
    EvalSuiteCatalogPage,
    EvalSuiteCatalogQuery,
)
from cayu.server._http_json import (
    _json_request_openapi,
    _model_json_response,
    _render_utf8,
    _validated_private_json_body,
)
from cayu.server.contracts import EVALS_ENDPOINT_RESPONSES

if TYPE_CHECKING:
    from fastapi.params import Depends

    from cayu.evals.store import EvalStore
    from cayu.server.evals_registry import EvalTargetRegistry


async def load_eval_corpus(
    corpus_revision: str,
    *,
    eval_store: EvalStore,
    active_eval_registry: EvalTargetRegistry,
) -> EvalCorpusDocument:
    """Load one visible corpus revision for catalog reads and run creation."""
    try:
        corpus = await eval_store.load_corpus(corpus_revision)
    except EvalStoreResultTooLarge as exc:
        raise HTTPException(
            status_code=413,
            detail="Eval corpus exceeds the server byte limit.",
        ) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Invalid Evals query.") from exc
    if corpus is None:
        raise HTTPException(status_code=404, detail="Eval corpus not found.")
    if active_eval_registry.get(corpus.target_key) is None:
        raise HTTPException(status_code=404, detail="Eval corpus not found.")
    return corpus


def register_corpus_management_routes(
    bounded_evals_router: APIRouter,
    *,
    eval_store: EvalStore,
    active_eval_registry: EvalTargetRegistry,
    protected: Sequence[Depends],
) -> None:
    """Register corpus management with the shared bounded HTTP boundary."""

    @bounded_evals_router.post(
        "/evals/corpora",
        response_model=EvalCorpusCatalogEntry,
        status_code=201,
        responses=EVALS_ENDPOINT_RESPONSES,
        dependencies=protected,
        openapi_extra=_json_request_openapi("EvalCorpusDocument"),
    )
    async def import_eval_corpus(request: Request):
        corpus = await _validated_private_json_body(
            request,
            EvalCorpusDocument,
            invalid_detail="Invalid Evals request.",
        )
        eval_target = active_eval_registry.get(corpus.target_key)
        if eval_target is None:
            raise HTTPException(
                status_code=400,
                detail="Eval corpus is incompatible with the attached target.",
            )
        try:
            await asyncio.to_thread(
                evaluation_target_identity,
                eval_target,
            )
            await asyncio.to_thread(
                _validate_corpus_target_compatibility,
                corpus,
                eval_target,
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail="Eval corpus is incompatible with the attached target.",
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=409,
                detail="Attached eval target is unavailable.",
            ) from exc
        try:
            return await eval_store.save_corpus(
                corpus,
                redact_json=eval_target.app.redact_json,
            )
        except EvalCorpusConflict as exc:
            raise HTTPException(
                status_code=409,
                detail="Eval corpus revision conflicts with stored content.",
            ) from exc
        except EvalStorePublicationRejected as exc:
            raise HTTPException(
                status_code=422,
                detail="Eval corpus contains unsafe public data.",
            ) from exc
        except EvalStoreResultTooLarge as exc:
            raise HTTPException(
                status_code=413,
                detail="Eval corpus exceeds the server byte limit.",
            ) from exc

    @bounded_evals_router.get(
        "/evals/corpora",
        response_model=EvalCorpusCatalogPage,
        responses=EVALS_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def list_eval_corpora(
        target_key: Annotated[
            str | None,
            Query(max_length=EVAL_STORE_MAX_IDENTIFIER_CHARS),
        ] = None,
        cursor: Annotated[str | None, Query(max_length=EVAL_STORE_MAX_CURSOR_BYTES)] = None,
        limit: Annotated[
            int,
            Query(ge=1, le=EVAL_STORE_MAX_PAGE_SIZE),
        ] = EVAL_STORE_DEFAULT_PAGE_SIZE,
        max_result_bytes: Annotated[
            int,
            Query(ge=1_024, le=EVAL_STORE_MAX_PAGE_BYTES),
        ] = EVAL_STORE_DEFAULT_PAGE_BYTES,
    ):
        selected_key = active_eval_registry.default_target_key if target_key is None else target_key
        eval_target = active_eval_registry.get(selected_key)
        if eval_target is None:
            raise HTTPException(status_code=404, detail="Eval target not found.")
        try:
            return await eval_store.list_corpora(
                EvalCatalogQuery(
                    target_key=eval_target.key,
                    cursor=cursor,
                    limit=limit,
                    max_result_bytes=max_result_bytes,
                )
            )
        except EvalStoreResultTooLarge as exc:
            raise HTTPException(
                status_code=413,
                detail="Eval catalog page exceeds the requested byte limit.",
            ) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="Invalid Evals query.") from exc

    @bounded_evals_router.get(
        "/evals/corpora/{corpus_revision}",
        response_model=EvalCorpusDocument,
        responses=EVALS_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def get_eval_corpus(corpus_revision: str):
        corpus = await load_eval_corpus(
            corpus_revision, eval_store=eval_store, active_eval_registry=active_eval_registry
        )
        return await _model_json_response(corpus, EvalCorpusDocument)

    @bounded_evals_router.get(
        "/evals/corpora/{corpus_revision}/download",
        response_class=Response,
        responses=EVALS_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def download_eval_corpus(corpus_revision: str) -> Response:
        corpus = await load_eval_corpus(
            corpus_revision, eval_store=eval_store, active_eval_registry=active_eval_registry
        )
        corpus_json = await asyncio.to_thread(_render_utf8, eval_corpus_to_json, corpus)
        return Response(
            content=corpus_json,
            media_type="application/json",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{corpus.target_key}-{corpus.revision[7:19]}.eval.json"'
                )
            },
        )

    @bounded_evals_router.get(
        "/evals/corpora/{corpus_revision}/suites",
        response_model=EvalSuiteCatalogPage,
        responses=EVALS_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def list_eval_suites(
        corpus_revision: str,
        cursor: Annotated[str | None, Query(max_length=EVAL_STORE_MAX_CURSOR_BYTES)] = None,
        limit: Annotated[
            int,
            Query(ge=1, le=EVAL_STORE_MAX_PAGE_SIZE),
        ] = EVAL_STORE_DEFAULT_PAGE_SIZE,
        max_result_bytes: Annotated[
            int,
            Query(ge=1_024, le=EVAL_STORE_MAX_PAGE_BYTES),
        ] = EVAL_STORE_DEFAULT_PAGE_BYTES,
    ):
        await load_eval_corpus(
            corpus_revision, eval_store=eval_store, active_eval_registry=active_eval_registry
        )
        try:
            page = await eval_store.list_suites(
                EvalSuiteCatalogQuery(
                    corpus_revision=corpus_revision,
                    cursor=cursor,
                    limit=limit,
                    max_result_bytes=max_result_bytes,
                )
            )
        except EvalStoreResultTooLarge as exc:
            raise HTTPException(
                status_code=413,
                detail="Eval suite page exceeds the requested byte limit.",
            ) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="Invalid Evals query.") from exc
        return await _model_json_response(page, EvalSuiteCatalogPage)

    @bounded_evals_router.get(
        "/evals/corpora/{corpus_revision}/suites/{suite_id}/cases",
        response_model=EvalCaseCatalogPage,
        responses=EVALS_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def list_eval_cases(
        corpus_revision: str,
        suite_id: str,
        cursor: Annotated[str | None, Query(max_length=EVAL_STORE_MAX_CURSOR_BYTES)] = None,
        limit: Annotated[
            int,
            Query(ge=1, le=EVAL_STORE_MAX_PAGE_SIZE),
        ] = EVAL_STORE_DEFAULT_PAGE_SIZE,
        max_result_bytes: Annotated[
            int,
            Query(ge=1_024, le=EVAL_STORE_MAX_PAGE_BYTES),
        ] = EVAL_STORE_DEFAULT_PAGE_BYTES,
    ):
        corpus = await load_eval_corpus(
            corpus_revision, eval_store=eval_store, active_eval_registry=active_eval_registry
        )
        if all(suite.id != suite_id for suite in corpus.suites):
            raise HTTPException(status_code=404, detail="Eval suite not found.")
        try:
            page = await eval_store.list_cases(
                EvalCaseCatalogQuery(
                    corpus_revision=corpus_revision,
                    suite_id=suite_id,
                    cursor=cursor,
                    limit=limit,
                    max_result_bytes=max_result_bytes,
                )
            )
        except EvalStoreResultTooLarge as exc:
            raise HTTPException(
                status_code=413,
                detail="Eval case page exceeds the requested byte limit.",
            ) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="Invalid Evals query.") from exc
        return await _model_json_response(page, EvalCaseCatalogPage)
