"""Runner cancellation fixtures using the ordinary asyncio exception contract."""

import asyncio
from typing import Any

from cayu.runners import attach_cancellation_artifacts


def cancelled_error_with_artifacts(
    message: str = "Runner command was cancelled.",
    *,
    artifacts: list[dict[str, Any]] | None = None,
) -> asyncio.CancelledError:
    error = asyncio.CancelledError(message)
    attach_cancellation_artifacts(error, [] if artifacts is None else artifacts)
    return error
