"""Publish completed ordinary/scenario trials without releasing their capacity slot."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from cayu.evals.models import EvalTrialResult
from cayu.evals.result_contract import _EvalTrialPublicData
from cayu.evals.store import (
    EvalRunClaim,
    EvalRunTrialCheckpoint,
    EvalStore,
    EvalStoreTransientContention,
)


async def save_trial_checkpoint_with_retry(
    *,
    store: EvalStore,
    claim: EvalRunClaim,
    case_id: str,
    result: EvalTrialResult,
    public_data: _EvalTrialPublicData,
    redact_json: Callable[[Any], Any],
    poll_seconds: float,
    logger: logging.Logger | None = None,
) -> None:
    selected_logger = logger or logging.getLogger(__name__)
    checkpoint = EvalRunTrialCheckpoint(case_id=case_id, result=result, public_data=public_data)
    retry_seconds = min(max(poll_seconds, 0.05), 1.0)
    started_at = asyncio.get_running_loop().time()
    try:
        while True:
            try:
                await store.save_trial_checkpoint(claim, checkpoint, redact_json=redact_json)
                return
            except EvalStoreTransientContention:
                selected_logger.warning(
                    "Durable eval checkpoint publication exhausted its transient "
                    "contention budget; the completed trial remains attached and "
                    "will be retried."
                )
                await asyncio.sleep(retry_seconds)
    finally:
        selected_logger.debug(
            "Durable eval checkpoint write finished.",
            extra={
                "cayu_eval_store_event": "checkpoint_write",
                "eval_run_id": claim.run_id,
                "duration_seconds": asyncio.get_running_loop().time() - started_at,
            },
        )
