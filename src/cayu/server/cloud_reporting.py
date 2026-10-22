"""HTTP entrance for Cayu Cloud's reader of run summaries and human touches.

Cayu Cloud reads the ``cayu.cloud-agent`` contract with a bearer token of its
own, never the operator login. The host holds only SHA-256 digests of the
token, so the token never reaches the process. The credential reads this
contract and nothing else: it cannot read conversations, settle a pending
action or call any other route.
"""

from __future__ import annotations

import hmac
import os
from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from cayu.applications import CayuApp
from cayu.cloud_reporting import (
    CLOUD_ACTIVITY_CURSOR_MAX_CHARS,
    CLOUD_ACTIVITY_DEFAULT_LIMIT,
    CLOUD_ACTIVITY_MAX_LIMIT,
    CloudActivityCursorInvalid,
    CloudActivityCursorStale,
    CloudActivityPage,
    CloudContract,
    read_cloud_activity,
    read_cloud_contract,
)
from cayu.server._bearer_digests import TOKEN_SHA256_HEX, presented_bearer_digest
from cayu.server.auth import AuthConfigurationError

CLOUD_READER_TOKEN_SHA256_VARIABLE = "CAYU_CLOUD_READER_TOKEN_SHA256"
CLOUD_READER_MAX_TOKENS = 8
_NO_STORE = {"Cache-Control": "private, no-store"}
#: The error code of a cursor the agent's store cannot continue.
CLOUD_ACTIVITY_CURSOR_STALE = "activity_cursor_stale"


class CloudReaderBearerAuth:
    """Authenticates Cayu Cloud's reader against SHA-256 digests of its bearer tokens.

    Give two digests while rotating the token. A presented token is compared
    with every digest in constant time. Digests never appear in errors, logs
    or ``repr``.
    """

    __slots__ = ("_digests",)

    def __init__(self, digests: Iterable[str]) -> None:
        if isinstance(digests, str | bytes):
            raise TypeError("digests must be an iterable of token digests.")
        values = tuple(digests)
        if not values:
            raise AuthConfigurationError("At least one Cayu Cloud reader token digest is required.")
        if len(values) > CLOUD_READER_MAX_TOKENS:
            raise AuthConfigurationError(
                f"At most {CLOUD_READER_MAX_TOKENS} Cayu Cloud reader token digests are allowed."
            )
        for index, digest in enumerate(values):
            if type(digest) is not str or TOKEN_SHA256_HEX.fullmatch(digest) is None:
                raise AuthConfigurationError(
                    f"Cayu Cloud reader token digest {index} must be a lowercase hex SHA-256 "
                    "digest."
                )
        if len(set(values)) != len(values):
            raise AuthConfigurationError("The Cayu Cloud reader token digests repeat a digest.")
        self._digests = tuple(bytes.fromhex(digest) for digest in values)

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> CloudReaderBearerAuth | None:
        """Read comma-separated digests from ``CAYU_CLOUD_READER_TOKEN_SHA256``.

        Returns ``None`` when the variable is not set. A set variable must name
        at least one valid digest. ``environ`` replaces ``os.environ``.
        """

        source: Mapping[str, str] = os.environ if environ is None else environ
        raw = source.get(CLOUD_READER_TOKEN_SHA256_VARIABLE)
        if raw is None:
            return None
        digests = [item.strip() for item in raw.split(",")] if raw.strip() else []
        try:
            return cls(digests)
        except AuthConfigurationError as exc:
            raise AuthConfigurationError(f"{CLOUD_READER_TOKEN_SHA256_VARIABLE}: {exc}") from None

    async def __call__(self, request: Request) -> bool:
        presented = presented_bearer_digest(request)
        if presented is None:
            return False
        matched = False
        # Compare with every digest, without an early exit, so timing does not
        # reveal which one matched.
        for digest in self._digests:
            matched = hmac.compare_digest(presented, digest) or matched
        return matched

    def __repr__(self) -> str:
        return f"CloudReaderBearerAuth(tokens={len(self._digests)})"


def create_cloud_reporting_router(
    app: CayuApp,
    *,
    authenticate: Callable[[Request], Awaitable[bool]],
    prefix: str = "/cloud",
) -> APIRouter:
    """Serve the contract's discovery and activity routes behind ``authenticate``.

    ``GET {prefix}/contract`` names the contract, the versions served and the
    session store; ``GET {prefix}/v1/activity`` returns one ``CloudActivityPage``,
    or 410 with code ``activity_cursor_stale`` for a cursor the store cannot
    continue. A request ``authenticate`` does not accept gets 401.
    """

    if not callable(authenticate):
        raise TypeError("A trusted authentication callback is required.")

    async def reader(request: Request) -> None:
        if await authenticate(request) is not True:
            raise HTTPException(
                status_code=401,
                detail="Authentication required.",
                headers={**_NO_STORE, "WWW-Authenticate": "Bearer"},
            )

    router = APIRouter(prefix=prefix, dependencies=[Depends(reader)])

    @router.get("/contract", response_model=CloudContract)
    async def contract(response: Response) -> CloudContract:
        response.headers.update(_NO_STORE)
        return await read_cloud_contract(app)

    @router.get("/v1/activity", response_model=CloudActivityPage)
    async def activity(
        response: Response,
        cursor: Annotated[
            str | None, Query(min_length=1, max_length=CLOUD_ACTIVITY_CURSOR_MAX_CHARS)
        ] = None,
        limit: Annotated[
            int, Query(ge=1, le=CLOUD_ACTIVITY_MAX_LIMIT)
        ] = CLOUD_ACTIVITY_DEFAULT_LIMIT,
    ) -> CloudActivityPage:
        try:
            page = await read_cloud_activity(app, cursor=cursor, limit=limit)
        except CloudActivityCursorInvalid:
            raise HTTPException(
                status_code=422, detail="Invalid activity cursor.", headers=_NO_STORE
            ) from None
        except CloudActivityCursorStale:
            raise HTTPException(
                status_code=410,
                detail={
                    "code": CLOUD_ACTIVITY_CURSOR_STALE,
                    "message": "The store cannot continue this cursor; read from the start.",
                },
                headers=_NO_STORE,
            ) from None
        except NotImplementedError:
            raise HTTPException(
                status_code=501,
                detail="The configured session store cannot serve run summaries.",
                headers=_NO_STORE,
            ) from None
        response.headers.update(_NO_STORE)
        return page

    return router


def environment_cloud_reporting_router(
    app: CayuApp,
    *,
    prefix: str = "/cloud",
    environ: Mapping[str, str] | None = None,
) -> APIRouter | None:
    """The router for the reader ``CAYU_CLOUD_READER_TOKEN_SHA256`` names, or ``None``.

    Malformed digests raise ``AuthConfigurationError``, so a misconfigured
    server does not start.
    """

    authenticate = CloudReaderBearerAuth.from_environment(environ)
    if authenticate is None:
        return None
    return create_cloud_reporting_router(app, authenticate=authenticate, prefix=prefix)


__all__ = [
    "CLOUD_ACTIVITY_CURSOR_STALE",
    "CLOUD_READER_MAX_TOKENS",
    "CLOUD_READER_TOKEN_SHA256_VARIABLE",
    "CloudReaderBearerAuth",
    "create_cloud_reporting_router",
    "environment_cloud_reporting_router",
]
