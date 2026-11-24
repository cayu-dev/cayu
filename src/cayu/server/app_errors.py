"""Exceptions an application's own routes raise, answered as API errors.

Cayu's own routes answer a failure with ``{"detail": ...}`` (``ApiErrorResponse``) and a
status that says what went wrong. An application's routes under ``/app`` get the same by
building their router with :data:`AppErrorRoute`, instead of catching exceptions in every
handler or matching them by class name::

    from fastapi import APIRouter

    from cayu.server.app_errors import AppErrorRoute

    router = APIRouter(prefix="/app", route_class=AppErrorRoute)


    @router.get("/invoices/{invoice_id}")
    async def invoice(invoice_id: str) -> dict:
        found = await invoices.get(invoice_id)
        if found is None:
            raise NotFound("Invoice not found.")  # 404 {"detail": "Invoice not found."}
        return found

:data:`DEFAULT_APP_ERRORS` says which exception answers which status. The message goes to
the caller only for the errors raised to be answered (:class:`NotFound`,
:class:`Forbidden`, :class:`Conflict`, :class:`Invalid`); a built-in ``LookupError``,
``PermissionError`` or ``ValueError`` from anywhere else, whose text may name a server path
or a stored value, answers with a fixed text. An application adds its own exceptions with
:meth:`AppErrors.extend`; the rules it adds are checked first, and without a ``detail``
they send the exception's message::

    errors = DEFAULT_APP_ERRORS.extend(ErrorStatus((InvoiceLocked,), 409))
    router = APIRouter(prefix="/app", route_class=errors.route_class())

``HTTPException`` and request validation errors pass through unchanged, and an exception
no rule names propagates as before (a 500). :meth:`AppErrors.http_error` gives the mapping
on its own, for code that answers outside a route.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from starlette.exceptions import HTTPException as StarletteHTTPException

from cayu._resource_access_errors import ResourceAccessDenied
from cayu.approvals import HumanReviewConflict
from cayu.artifacts import ArtifactIdentityConflictError, ArtifactStoreUnavailableError
from cayu.collaboration import ContinuationConflict
from cayu.runtime import SessionSteeringConflict
from cayu.server.service import ProductIdempotencyConflict, ProductResultReceiptConflict
from cayu.sessions import SessionStatusConflict
from cayu.storage import (
    KnowledgeAccessDenied,
    KnowledgeActivationConflict,
    KnowledgeMaintenanceConflict,
    KnowledgePublicationConflict,
    KnowledgeRelationConflict,
    KnowledgeRevisionConflict,
)
from cayu.tasks import (
    TaskCreationConflict,
    TaskGraphConflict,
    TaskGroupConflict,
    TaskScheduleConflict,
)

__all__ = [
    "CAYU_ACCESS_DENIED",
    "CAYU_CONFLICTS",
    "DEFAULT_APP_ERRORS",
    "AppError",
    "AppErrorRoute",
    "AppErrors",
    "Conflict",
    "ErrorStatus",
    "Forbidden",
    "Invalid",
    "NotFound",
]


class AppError(Exception):
    """An error raised to be answered: its message is written for the caller."""


class NotFound(AppError, LookupError):
    """Answered 404 with its message."""


class Forbidden(AppError, PermissionError):
    """Answered 403 with its message."""


class Conflict(AppError):
    """Answered 409 with its message: the request conflicts with the stored state."""


class Invalid(AppError, ValueError):
    """Answered 422 with its message."""


# Cayu's refusals of a resource the person may not use. They answer like a resource that
# doesn't exist, as Cayu's own resource routes do, so a refusal doesn't say it exists.
CAYU_ACCESS_DENIED: tuple[type[Exception], ...] = (ResourceAccessDenied, KnowledgeAccessDenied)

# Cayu's errors for a change that conflicts with the stored state: a stale revision, a
# request id reused with other content, a session or task in a state that refuses it.
# Several subclass ValueError, so they're checked before ValueError's 422.
CAYU_CONFLICTS: tuple[type[Exception], ...] = (
    ArtifactIdentityConflictError,
    ContinuationConflict,
    HumanReviewConflict,
    KnowledgeActivationConflict,
    KnowledgeMaintenanceConflict,
    KnowledgePublicationConflict,
    KnowledgeRelationConflict,
    KnowledgeRevisionConflict,
    ProductIdempotencyConflict,
    ProductResultReceiptConflict,
    SessionStatusConflict,
    SessionSteeringConflict,
    TaskCreationConflict,
    TaskGraphConflict,
    TaskGroupConflict,
    TaskScheduleConflict,
)


def _message(error: BaseException) -> str:
    # KeyError's str() quotes its key; the message is the argument itself.
    if isinstance(error, KeyError) and len(error.args) == 1:
        return str(error.args[0])
    return str(error)


@dataclass(frozen=True, slots=True)
class ErrorStatus:
    """Answer an exception of one of ``errors`` (subclasses too) with ``status``.

    ``detail`` is the answer's ``detail``: a fixed text, a function of the exception, or by
    default the exception's message. Give a fixed text for exceptions that code other than
    your own raises, since their messages may name server paths or stored values.
    """

    errors: tuple[type[BaseException], ...]
    status: int
    detail: str | Callable[[BaseException], Any] | None = None

    def __post_init__(self) -> None:
        if not self.errors or not all(
            isinstance(kind, type) and issubclass(kind, Exception) for kind in self.errors
        ):
            raise TypeError("ErrorStatus.errors must be a non-empty tuple of Exception types.")
        if type(self.status) is not int or not 400 <= self.status <= 599:
            raise ValueError("ErrorStatus.status must be an HTTP error status (400 to 599).")

    def matches(self, error: BaseException) -> bool:
        return isinstance(error, self.errors)

    def http_error(self, error: BaseException) -> HTTPException:
        if isinstance(self.detail, str):
            detail: Any = self.detail
        elif self.detail is not None:
            detail = self.detail(error)
        else:
            detail = _message(error) or "The request failed."
        return HTTPException(status_code=self.status, detail=detail)


class AppErrors:
    """An ordered set of :class:`ErrorStatus` rules; the first that matches answers."""

    __slots__ = ("rules",)

    def __init__(self, *rules: ErrorStatus) -> None:
        if not all(type(rule) is ErrorStatus for rule in rules):
            raise TypeError("AppErrors takes ErrorStatus rules.")
        self.rules: tuple[ErrorStatus, ...] = rules

    def extend(self, *rules: ErrorStatus) -> AppErrors:
        """These rules, checked before the existing ones."""

        return AppErrors(*rules, *self.rules)

    def http_error(self, error: BaseException) -> StarletteHTTPException | None:
        """The ``HTTPException`` that answers ``error``, or None when no rule names it."""

        if isinstance(error, StarletteHTTPException):
            return error
        if isinstance(error, RequestValidationError):
            return None
        for rule in self.rules:
            if rule.matches(error):
                return rule.http_error(error)
        return None

    def route_class(self) -> type[APIRoute]:
        """An ``APIRoute`` class whose handlers answer these errors (``route_class=``)."""

        errors = self

        class _AppErrorRoute(APIRoute):
            def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
                handler = super().get_route_handler()

                async def answer(request: Request) -> Response:
                    try:
                        return await handler(request)
                    except Exception as error:
                        mapped = errors.http_error(error)
                        if mapped is None or mapped is error:
                            raise
                        raise mapped from error

                return answer

        _AppErrorRoute.__name__ = _AppErrorRoute.__qualname__ = "AppErrorRoute"
        return _AppErrorRoute


DEFAULT_APP_ERRORS = AppErrors(
    ErrorStatus((NotFound,), 404),
    ErrorStatus((Forbidden,), 403),
    ErrorStatus((Conflict,), 409),
    ErrorStatus((Invalid,), 422),
    ErrorStatus(CAYU_ACCESS_DENIED, 404, detail="Resource unavailable."),
    ErrorStatus((PermissionError,), 403, detail="Not allowed."),
    ErrorStatus(CAYU_CONFLICTS, 409, detail="The request conflicts with the stored state."),
    ErrorStatus((ArtifactStoreUnavailableError,), 503, detail="File storage is unavailable."),
    ErrorStatus((LookupError, FileNotFoundError), 404, detail="Not found."),
    ErrorStatus((ValueError,), 422, detail="Invalid request."),
)
"""The app's own :class:`AppError` kinds with their messages; then, with fixed texts, Cayu's
refusals 404, other ``PermissionError`` 403, Cayu's conflicts 409, file storage down 503,
``LookupError`` and ``FileNotFoundError`` 404, other ``ValueError`` 422."""

AppErrorRoute: type[APIRoute] = DEFAULT_APP_ERRORS.route_class()
"""The route class for an application router that answers :data:`DEFAULT_APP_ERRORS`."""
