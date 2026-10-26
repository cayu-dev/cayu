"""Shared fail-closed stored result boundary for reports and suite-centred reads."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from fastapi import HTTPException

from cayu._validation import compact_json_utf8_size
from cayu.evals.execution import CorpusExecutionResult
from cayu.evals.results import CapturedEvaluationResultV1, EvalResultOrigin
from cayu.evals.store import EvalResultRecord, EvalStoreResultTooLarge, _exact_model

if TYPE_CHECKING:
    from cayu.evals.store import EvalStore
    from cayu.server.evals_registry import EvalTargetRegistry


def _eval_result_record_matches_document(
    record: EvalResultRecord,
    result: CorpusExecutionResult | CapturedEvaluationResultV1,
) -> bool:
    """Fail closed when a custom store returns mismatched result metadata."""

    if type(result) is CorpusExecutionResult:
        origin = EvalResultOrigin.FRESH_EXECUTION
        target = result.target
        corpus_revision = result.run.corpus_revision
        suite_id = result.run.suite_id
        suite_revision = result.run.suite_revision
        status = result.run.status
        score = result.run.score
    elif type(result) is CapturedEvaluationResultV1:
        origin = EvalResultOrigin.CAPTURED_SESSION
        target = result.target
        corpus_revision = result.corpus_revision
        suite_id = result.suite_id
        suite_revision = result.suite_revision
        status = result.score.status
        score = result.score.score
    else:
        return False
    document_bytes = compact_json_utf8_size(result.model_dump(mode="json"))
    return (
        record.revision == result.revision
        and record.origin == origin
        and record.target.target_key == target.target_key
        and record.target.application_release_id == target.application_release_id
        and record.target.app_manifest_schema_version == target.app_manifest_schema_version
        and record.target.app_manifest_fingerprint == target.app_manifest_fingerprint
        and record.corpus_revision == corpus_revision
        and record.suite_id == suite_id
        and record.suite_revision == suite_revision
        and record.status == status
        and record.score == score
        and record.document_bytes == document_bytes
    )


async def load_catalog_eval_result(
    result_revision: str,
    captured_eval_store: EvalStore | None,
    eval_registry: EvalTargetRegistry,
) -> tuple[EvalResultRecord, CorpusExecutionResult | CapturedEvaluationResultV1]:
    if captured_eval_store is None or not captured_eval_store.captured_results:
        raise HTTPException(status_code=409, detail="Eval result catalog is unavailable.")
    try:
        record = await captured_eval_store.load_result_record(result_revision)
        result = await captured_eval_store.load_result_by_revision(result_revision)
    except NotImplementedError as exc:
        raise HTTPException(status_code=409, detail="Eval result catalog is unavailable.") from exc
    except EvalStoreResultTooLarge as exc:
        raise HTTPException(status_code=413, detail="Eval result is too large.") from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Invalid eval result revision.") from exc
    if record is None or result is None:
        raise HTTPException(status_code=404, detail="Eval result not found.")
    try:
        record = await asyncio.to_thread(_exact_model, record, EvalResultRecord, "record")
        if type(result) not in {CorpusExecutionResult, CapturedEvaluationResultV1}:
            raise TypeError("Unsupported eval result document.")
        result = await asyncio.to_thread(_exact_model, result, type(result), "result")
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=409,
            detail="Eval result catalog metadata does not match its stored document.",
        ) from exc
    if record.revision != result_revision:
        raise HTTPException(
            status_code=409,
            detail="Eval result catalog metadata does not match its stored document.",
        )
    if not await asyncio.to_thread(
        _eval_result_record_matches_document,
        record,
        result,
    ):
        raise HTTPException(
            status_code=409,
            detail="Eval result catalog metadata does not match its stored document.",
        )
    if eval_registry.get(record.target.target_key) is None:
        raise HTTPException(status_code=404, detail="Eval result not found.")
    return record, result
