"""Runtime recording of the generation controls each model attempt sends."""

from __future__ import annotations

import logging
from typing import Any

from cayu.providers.base import ModelProvider, ModelRequest
from cayu.providers.request_controls import REQUEST_CONTROLS_PAYLOAD_KEY, ModelRequestControls

logger = logging.getLogger(__name__)


def model_request_controls_payload(
    provider: ModelProvider,
    request: ModelRequest,
) -> dict[str, Any] | None:
    """The controls ``provider`` reports for ``request``, or None when it reports none.

    The record is diagnostic only. A hook that fails or returns the wrong type
    leaves ``model.started`` without controls rather than failing the attempt;
    the adapter's own request building still decides what is sent.
    """

    try:
        controls = provider.request_controls(request.model_copy(deep=True))
    except Exception as exc:
        logger.debug(
            "Provider %r could not report request controls (%s).",
            getattr(provider, "name", type(provider).__name__),
            type(exc).__name__,
        )
        return None
    if controls is None:
        return None
    if type(controls) is not ModelRequestControls:
        logger.debug(
            "Provider %r returned %s from request_controls(); expected ModelRequestControls.",
            getattr(provider, "name", type(provider).__name__),
            type(controls).__name__,
        )
        return None
    return controls.payload()


def request_controls_field(
    provider: ModelProvider,
    request: ModelRequest,
) -> dict[str, Any]:
    """The ``model.started`` payload entry for ``request``, or nothing when unreported."""

    controls = model_request_controls_payload(provider, request)
    return {} if controls is None else {REQUEST_CONTROLS_PAYLOAD_KEY: controls}
