"""HTTP answers for a run or continuation this server's providers can't serve.

A placeholder provider that stands in for an unselected one raises
``ProviderNotConfigured`` with what to set: that is the server's setup, so the
answer is 503 with its message. A session started under a provider this server
doesn't register can't continue here: 409 with a sentence naming it.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from cayu._application_provider_registry import ProviderNotRegistered
from cayu.providers.base import ProviderNotConfigured


async def provider_setup_http_error(
    cayu_app: Any, error: BaseException, *, session_id: str | None
) -> HTTPException | None:
    """The answer for ``error`` when it is a provider setup failure, else ``None``.

    ``session_id`` is the stored session the request continues (a resume, an
    answer, an approval, a fork's source), or ``None`` for a new run. Only when the
    missing provider is the one that session was started with does the answer say
    so; a provider a request names for itself is just missing here.
    """

    if isinstance(error, ProviderNotConfigured):
        return HTTPException(status_code=503, detail=_redacted(cayu_app, str(error)))
    if not isinstance(error, ProviderNotRegistered):
        return None
    name = _redacted(cayu_app, error.provider_name)
    if session_id is not None and await _stored_provider(cayu_app, session_id) == (
        error.provider_name
    ):
        detail = (
            f"This session was started with provider `{name}`, which this server doesn't "
            f"have. Continue it from a server that registers `{name}`, or close it."
        )
    else:
        detail = f"This server has no provider `{name}`."
    return HTTPException(status_code=409, detail=detail)


async def _stored_provider(cayu_app: Any, session_id: str) -> str | None:
    try:
        session = await cayu_app.session_store.load(session_id)
    except Exception:
        return None
    return None if session is None else session.provider_name


_UNREDACTABLE = "The model provider isn't set up on this server."


def _redacted(cayu_app: Any, text: str) -> str:
    try:
        redacted = cayu_app.redact_json(text)
    except Exception:
        return _UNREDACTABLE
    return redacted if type(redacted) is str else _UNREDACTABLE
