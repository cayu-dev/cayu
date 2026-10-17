"""HTTP entrance for enrolled fork controllers.

Mount this router in a host application that authenticates its callers. The
host's ``authenticate`` callback names the subject; request bodies never select
the controller or carry its authority. ``ForkControllerBearerAuth`` is a stock
``authenticate`` callback that maps bearer tokens to subjects.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from collections.abc import Awaitable, Callable, Mapping
from contextlib import aclosing, suppress
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from cayu._validation import require_durable_clean_nonblank
from cayu.applications import CayuApp
from cayu.approvals.actors import ResolutionActor, ResolutionActorSource
from cayu.fork_controllers import (
    ForkControllerAccess,
    ForkControllerDenied,
    ForkControllerEnrollment,
)
from cayu.runtime.execution_profiles import ExecutionProfileAdoptionRejected
from cayu.server.auth import AuthConfigurationError
from cayu.sessions.authority import SessionRunFenced
from cayu.sessions.forks import ForkRequestConflict
from cayu.sessions.requests import ForkSessionRequest, ResumeRequest

_DENIED = "Fork controller access denied."
FORK_CONTROLLER_TOKENS_VARIABLE = "CAYU_FORK_CONTROLLER_TOKENS"
FORK_CONTROLLER_TOKEN_MIN_CHARS = 32
FORK_CONTROLLER_MAX_TOKENS = 64
# RFC 6750 b64token: the only credential syntax a bearer header may carry.
_BEARER_CREDENTIAL = re.compile(r"[A-Za-z0-9\-._~+/]+=*")


def _token_digest(token: str) -> bytes:
    return hashlib.sha256(token.encode("ascii")).digest()


class ForkControllerBearerAuth:
    """Stock ``authenticate`` callback mapping bearer tokens to controller subjects.

    ``tokens`` maps each token to the subject it authenticates; two tokens may
    name one subject, for rotation. Only SHA-256 digests of the tokens are kept,
    and a presented token is compared with every entry in constant time. A
    missing, malformed, repeated or unknown ``Authorization`` header returns
    ``""``, which the router answers with 401. Tokens never appear in errors,
    logs or ``repr``.
    """

    __slots__ = ("_entries",)

    def __init__(self, tokens: Mapping[str, str]) -> None:
        if not isinstance(tokens, Mapping):
            raise TypeError("tokens must map bearer tokens to subjects.")
        if not tokens:
            raise AuthConfigurationError("At least one fork-controller bearer token is required.")
        if len(tokens) > FORK_CONTROLLER_MAX_TOKENS:
            raise AuthConfigurationError(
                f"At most {FORK_CONTROLLER_MAX_TOKENS} fork-controller bearer tokens are allowed."
            )
        entries: list[tuple[bytes, str]] = []
        for index, (token, subject) in enumerate(tokens.items()):
            if (
                type(token) is not str
                or len(token) < FORK_CONTROLLER_TOKEN_MIN_CHARS
                or _BEARER_CREDENTIAL.fullmatch(token) is None
            ):
                raise AuthConfigurationError(
                    f"Fork-controller bearer token {index} must be a bearer credential of at "
                    f"least {FORK_CONTROLLER_TOKEN_MIN_CHARS} characters."
                )
            if type(subject) is not str:
                raise AuthConfigurationError(
                    f"The subject for fork-controller bearer token {index} must be a string."
                )
            try:
                subject = require_durable_clean_nonblank(subject, "subject")
            except ValueError:
                raise AuthConfigurationError(
                    f"The subject for fork-controller bearer token {index} is not a valid subject."
                ) from None
            entries.append((_token_digest(token), subject))
        self._entries = tuple(entries)

    @classmethod
    def from_environment(
        cls,
        variable: str = FORK_CONTROLLER_TOKENS_VARIABLE,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> ForkControllerBearerAuth:
        """Read a JSON object of ``{"<token>": "<subject>"}`` from one environment variable.

        The variable is read once, when this method is called. ``environ``
        replaces ``os.environ`` for tests and for callers that resolve secrets
        into a mapping first.
        """

        source: Mapping[str, str] = os.environ if environ is None else environ
        raw = source.get(variable)
        if not isinstance(raw, str) or not raw.strip():
            raise AuthConfigurationError(f"{variable} must be set to a JSON object of tokens.")
        return cls(_token_mapping(raw, variable))

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> ForkControllerBearerAuth:
        """Read a JSON object of ``{"<token>": "<subject>"}`` from a file, such as a mounted secret."""

        # Decode errors hold the file's bytes, tokens included, so the
        # configuration error is raised outside the handler: it must not chain them.
        raw: str | None = None
        with suppress(OSError, UnicodeDecodeError):
            raw = Path(path).read_text(encoding="utf-8")
        if raw is None:
            raise AuthConfigurationError("The fork-controller token file could not be read.")
        return cls(_token_mapping(raw, "The fork-controller token file"))

    async def __call__(self, request: Request) -> str:
        headers = request.headers.getlist("authorization")
        if len(headers) != 1:
            return ""
        scheme, separator, credential = headers[0].partition(" ")
        if (
            separator != " "
            or scheme.lower() != "bearer"
            or _BEARER_CREDENTIAL.fullmatch(credential) is None
        ):
            return ""
        presented = _token_digest(credential)
        subject = ""
        # Compare with every entry, without an early exit, so timing does not
        # reveal which entry matched.
        for digest, entry_subject in self._entries:
            if hmac.compare_digest(presented, digest):
                subject = entry_subject
        return subject

    def __repr__(self) -> str:
        return f"ForkControllerBearerAuth(tokens={len(self._entries)})"


def _token_mapping(raw: str, source_name: str) -> dict[str, str]:
    # A JSON error keeps the whole document, tokens included, so it is not chained.
    parsed = True
    try:
        value = json.loads(raw)
    except ValueError:
        parsed = False
    if not parsed:
        raise AuthConfigurationError(f"{source_name} is not valid JSON.")
    if type(value) is not dict:
        raise AuthConfigurationError(f"{source_name} must be a JSON object of tokens to subjects.")
    return value


async def _stream(events):
    # Admission failures surface before any successful HTTP headers are sent.
    # Details are fixed: an absent session and a foreign one look the same.
    try:
        first = await anext(events)
    except StopAsyncIteration:
        return StreamingResponse(iter(()), media_type="text/event-stream")
    except (ForkControllerDenied, KeyError):
        await events.aclose()
        raise HTTPException(status_code=403, detail=_DENIED) from None
    except ExecutionProfileAdoptionRejected:
        await events.aclose()
        raise HTTPException(
            status_code=409, detail="The fork's execution profile was not adopted."
        ) from None
    except ForkRequestConflict:
        await events.aclose()
        raise HTTPException(
            status_code=409, detail="The fork destination exists with a different request."
        ) from None
    except SessionRunFenced:
        # For example, the source moved after the controller's exact snapshot.
        await events.aclose()
        raise HTTPException(
            status_code=409, detail="The session changed during the request; read it again."
        ) from None
    except ValueError:
        await events.aclose()
        raise HTTPException(status_code=422, detail="Invalid fork-controller request.") from None

    async def body():
        async with aclosing(events):
            yield "data: " + first.model_dump_json() + "\n\n"
            async for event in events:
                yield "data: " + event.model_dump_json() + "\n\n"

    return StreamingResponse(body(), media_type="text/event-stream")


def _with_authenticated_actor(body, subject: str):
    """Adoption provenance comes from authentication, never from the request body."""

    intent = body.profile_adoption
    if intent is None:
        return body
    if intent.requested_by.subject != subject:
        # Never silently replace a claimed actor; the audit trail would disagree.
        raise HTTPException(status_code=403, detail=_DENIED)
    return body.model_copy(
        update={
            "profile_adoption": intent.model_copy(
                update={
                    "requested_by": ResolutionActor(
                        subject=subject, source=ResolutionActorSource.HTTP_AUTH
                    )
                }
            )
        }
    )


def create_fork_controller_router(
    app: CayuApp,
    *,
    authenticate: Callable[[Request], Awaitable[str]],
    prefix: str = "/fork-controller",
) -> APIRouter:
    """Mount enrollment, source snapshots, ``POST /fork`` and ``POST /resume`` for controllers."""

    if not callable(authenticate):
        raise TypeError("A trusted authentication callback is required.")
    router = APIRouter(prefix=prefix)

    async def access(request: Request) -> ForkControllerAccess:
        subject = await authenticate(request)
        if type(subject) is not str or not subject.strip() or subject != subject.strip():
            raise HTTPException(status_code=401, detail="Authentication required.")
        try:
            return app.fork_controller_access(subject)
        except ForkControllerDenied:
            raise HTTPException(status_code=403, detail=_DENIED) from None

    access_dependency = Depends(access)

    @router.get("/enrollment", response_model=ForkControllerEnrollment)
    async def enrollment(handle: ForkControllerAccess = access_dependency):
        return handle.enrollment()

    @router.get("/sources/{session_id}/snapshot")
    async def snapshot(session_id: str, handle: ForkControllerAccess = access_dependency):
        try:
            return await handle.snapshot_source(session_id)
        except (ForkControllerDenied, KeyError):
            raise HTTPException(status_code=403, detail=_DENIED) from None

    @router.post("/fork")
    async def fork(body: ForkSessionRequest, handle: ForkControllerAccess = access_dependency):
        return await _stream(handle.fork(_with_authenticated_actor(body, handle.subject)))

    @router.post("/resume")
    async def resume(body: ResumeRequest, handle: ForkControllerAccess = access_dependency):
        return await _stream(handle.resume(_with_authenticated_actor(body, handle.subject)))

    return router


__all__ = [
    "FORK_CONTROLLER_TOKENS_VARIABLE",
    "ForkControllerBearerAuth",
    "create_fork_controller_router",
]
