"""The protected model check: one bounded provider request through the served app.

`cayu providers check MODEL --application NAME` reaches it on a Cloud Agent. It runs
the same check as the local command, with the server's providers, credentials and
price book, and answers only the check's report: no prompt, answer text, provider
error text or credentials. Nothing is written to a store.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from fastapi import HTTPException

from cayu.budgets.pricing import PriceBook, default_price_book
from cayu.runtime._model_check import (
    MODEL_CHECK_API_PATH,
    MODEL_CHECK_MAX_OUTPUT_TOKENS,
    MODEL_CHECK_TIMEOUT_SECONDS,
    ModelCheckRequest,
    ModelCheckResult,
    ModelCheckRouteError,
    PriceBookSource,
    check_model,
    resolve_model_check_route,
)
from cayu.server.contracts import ApiErrorResponse

if TYPE_CHECKING:
    from fastapi import APIRouter
    from fastapi.params import Depends

    from cayu.applications import CayuApp

# Each check spends a little of the server's provider quota, so a process runs one
# at a time and waits this long after one finishes before admitting the next.
MODEL_CHECK_MIN_INTERVAL_SECONDS = 5.0


class ModelCheckAdmission:
    """Process-local admission: one check in flight, and a pause after each."""

    def __init__(
        self,
        *,
        min_interval_seconds: float = MODEL_CHECK_MIN_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._min_interval_seconds = min_interval_seconds
        self._clock = clock
        self._running = False
        self._finished_at: float | None = None
        self._lock = threading.Lock()

    def try_start(self) -> float | None:
        """Admit one check, or return how many seconds to wait before asking again."""

        with self._lock:
            if self._running:
                return self._min_interval_seconds
            if self._finished_at is not None:
                wait = self._finished_at + self._min_interval_seconds - self._clock()
                if wait > 0:
                    return wait
            self._running = True
            return None

    def finish(self) -> None:
        with self._lock:
            self._running = False
            self._finished_at = self._clock()


def register_model_check_route(
    router: APIRouter,
    *,
    cayu_app: CayuApp,
    protected: list[Depends],
    price_book: PriceBook | None,
) -> None:
    """Add ``POST {api}/diagnostics/model-check`` for full-access callers.

    ``price_book`` is the server's configured dashboard price book
    (``[tool.cayu.serve].price_book`` under `cayu serve`); without one the check
    prices with Cayu's bundled book, as the local command does.
    """

    pricing = default_price_book() if price_book is None else price_book
    price_book_source: PriceBookSource = "bundled" if price_book is None else "project"
    gate = ModelCheckAdmission()

    @router.post(
        MODEL_CHECK_API_PATH,
        response_model=ModelCheckResult,
        dependencies=protected,
        responses={
            422: {
                "model": ApiErrorResponse,
                "description": (
                    "The request is invalid, or names no single registered provider for "
                    "the model. No provider request was sent."
                ),
            },
            429: {
                "model": ApiErrorResponse,
                "description": (
                    "Another check is running in this process, or one finished less "
                    f"than {MODEL_CHECK_MIN_INTERVAL_SECONDS:g} seconds ago. Retry after "
                    "the `Retry-After` header."
                ),
            },
        },
        description=(
            "Send one bounded request for `model` through the provider an agent using it "
            "resolves to (or `provider`), with this server's credentials, and report the "
            "outcome, latency, route, usage and pricing, as `cayu providers check` does. "
            f"The server fixes the prompt, a {MODEL_CHECK_MAX_OUTPUT_TOKENS}-token output cap, "
            f"no tools and a {MODEL_CHECK_TIMEOUT_SECONDS:g}-second deadline. The report never "
            "carries the prompt, the answer text, the provider's error text or credentials, "
            "and nothing is stored. A provider failure is a report with `ok: false`, not an "
            "HTTP error. One check runs at a time per process."
        ),
    )
    async def run_model_check(body: ModelCheckRequest) -> ModelCheckResult:
        try:
            registered, resolution = resolve_model_check_route(
                cayu_app, model=body.model, provider_name=body.provider
            )
        except ModelCheckRouteError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        wait = gate.try_start()
        if wait is not None:
            raise HTTPException(
                status_code=429,
                detail="A model check is running or has just finished; retry shortly.",
                headers={"Retry-After": str(max(1, math.ceil(wait)))},
            )
        try:
            return await check_model(
                registered,
                model=body.model,
                resolution=resolution,
                pricing=pricing,
                price_book_source=price_book_source,
                redactor=cayu_app._secret_redactor,
                timeout_seconds=MODEL_CHECK_TIMEOUT_SECONDS,
            )
        finally:
            gate.finish()
