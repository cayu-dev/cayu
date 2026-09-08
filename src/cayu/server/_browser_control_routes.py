"""Authenticated browser-control HTTP boundary (not yet mounted by default).

Operator-session tokens authenticate continuity only; every action still invokes
the application's browser policy. Pixels and native input never use these routes.
The app supplies a dedicated stable signing key, never a workload credential.
"""

import base64
import hashlib
import hmac
import json
import secrets
import time
from collections.abc import Callable
from typing import Annotated, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ValidationError

from cayu._validation import canonical_durable_json_bytes
from cayu.runtime._browser_control_authorization import BrowserControlPermissionDenied
from cayu.runtime._browser_control_coordinator import BrowserControlCoordinator
from cayu.runtime._browser_control_input_tickets import BrowserInputTickets
from cayu.runtime._browser_control_publisher import BrowserControlPublicationPending
from cayu.runtime._browser_control_service import BrowserControlService
from cayu.runtime._browser_control_view_tickets import BrowserViewTickets
from cayu.runtime.browser_control import (
    BrowserControlConflict,
    BrowserControlPrincipal,
    BrowserHandbackIntent,
    BrowserPagesIntent,
    BrowserRenewIntent,
    BrowserSensitiveEntryIntent,
    BrowserTakeoverIntent,
    BrowserTextInputIntent,
    BrowserViewIntent,
)
from cayu.server.auth import AuthContext, AuthDependency, server_auth_dependency

_BODY_LIMIT = 32 * 1024
_TOKEN_DOMAIN = b"cayu.browser.operator-session.v1\x00"
_Intent = TypeVar("_Intent", bound=BaseModel)


async def _parse_control_intent(request: Request, model: type[_Intent]) -> _Intent:
    if request.headers.get("content-type", "").split(";", 1)[0] != "application/json":
        raise HTTPException(415, "Browser control requires a JSON control request.")
    body = bytearray()
    try:
        async for chunk in request.stream():
            if len(body) + len(chunk) > _BODY_LIMIT:
                raise HTTPException(413, "Browser control request exceeds its bound.")
            body.extend(chunk)
        return model.model_validate_json(body)
    finally:
        body.clear()


class BrowserOperatorSessionTokens:
    """Bounded signed continuity tokens, without raw authenticated identities."""

    def __init__(self, key: bytes, *, clock: Callable[[], float] = time.time) -> None:
        if type(key) is not bytes or len(key) != 32:
            raise ValueError("Browser operator sessions require a dedicated 32-byte signing key.")
        self._key = key
        self._clock = clock

    def _principal_digest(self, principal: BrowserControlPrincipal) -> str:
        owned = BrowserControlPrincipal.model_validate(principal)
        return hashlib.sha256(
            canonical_durable_json_bytes(
                owned.model_dump(mode="json"), "authenticated browser principal"
            )
        ).hexdigest()

    def issue(self, principal: BrowserControlPrincipal) -> str:
        payload = canonical_durable_json_bytes(
            {
                "v": 1,
                "id": "bo_" + secrets.token_hex(16),
                "principal": self._principal_digest(principal),
                "expires": int(self._clock()) + 3600,
            },
            "browser operator session",
        )
        signature = hmac.digest(self._key, _TOKEN_DOMAIN + payload, "sha256")
        return base64.urlsafe_b64encode(payload + signature).decode("ascii").rstrip("=")

    def verify(self, token: str, principal: BrowserControlPrincipal) -> str:
        failed = True
        identity = ""
        try:
            if type(token) is not str or not 1 <= len(token) <= 512 or not token.isascii():
                raise ValueError
            raw = base64.b64decode(token + "=" * (-len(token) % 4), altchars=b"-_", validate=True)
            payload, signature = raw[:-32], raw[-32:]
            if not hmac.compare_digest(
                signature, hmac.digest(self._key, _TOKEN_DOMAIN + payload, "sha256")
            ):
                raise ValueError
            data = json.loads(payload)
            if (
                type(data) is not dict
                or set(data) != {"v", "id", "principal", "expires"}
                or type(data["v"]) is not int
                or data["v"] != 1
                or type(data["expires"]) is not int
                or not self._clock() < data["expires"] <= self._clock() + 3600
                or data["principal"] != self._principal_digest(principal)
                or type(data["id"]) is not str
                or len(data["id"]) != 35
                or not data["id"].startswith("bo_")
                or any(char not in "0123456789abcdef" for char in data["id"][3:])
            ):
                raise ValueError
            identity = data["id"]
            failed = False
        except (ValueError, TypeError, UnicodeError):
            pass
        if failed:
            raise BrowserControlPermissionDenied()
        return identity


def create_browser_control_router(
    *,
    coordinator: BrowserControlCoordinator,
    auth: AuthDependency,
    sessions: BrowserOperatorSessionTokens,
    allowed_origin: str,
    view_tickets: BrowserViewTickets | None = None,
    input_tickets: BrowserInputTickets | None = None,
    service: BrowserControlService | None = None,
) -> APIRouter:
    if not callable(auth):
        raise TypeError("Browser control requires an authentication dependency.")
    from urllib.parse import urlsplit

    parsed = urlsplit(allowed_origin)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.path
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
    ):
        raise ValueError("Browser control requires an explicit HTTPS operator origin.")
    router = APIRouter(prefix="/browser-control")
    authenticated = server_auth_dependency(auth)

    def principal_for(request: Request, context: AuthContext) -> BrowserControlPrincipal:
        if request.url.scheme != "https" or request.query_params:
            raise HTTPException(403, "Protected browser control requires HTTPS without query data.")
        origin = request.headers.get("origin")
        if origin is not None and origin != allowed_origin:
            raise HTTPException(403, "Browser control origin is not allowed.")
        # Do not serialize an authentication extension's model before validation.
        try:
            return BrowserControlPrincipal(subject=context.subject, tenant=context.tenant)
        except (ValueError, TypeError):
            pass
        raise HTTPException(403, "Browser operator identity is unavailable.")

    @router.post("/operator-session")
    async def operator_session(
        request: Request, context: Annotated[AuthContext, Depends(authenticated)]
    ):
        principal = principal_for(request, context)
        from starlette.responses import JSONResponse

        return JSONResponse(
            {"operator_session_token": sessions.issue(principal)},
            headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
        )

    @router.get("/sessions/{session_id}")
    async def discover(
        session_id: str, request: Request, context: Annotated[AuthContext, Depends(authenticated)]
    ):
        principal = principal_for(request, context)
        try:
            operator_session_id = sessions.verify(
                request.headers.get("x-cayu-browser-operator", ""), principal
            )
            records = await coordinator.inspect_authorized_browsers(
                session_id=session_id, principal=principal, operator_session_id=operator_session_id
            )
        except (BrowserControlPermissionDenied, BrowserControlConflict, ValueError, TypeError):
            raise HTTPException(403, "Browser control is unavailable.") from None
        from starlette.responses import JSONResponse

        return JSONResponse(
            {
                "browsers": [
                    {
                        "identity": record.identity.model_dump(mode="json"),
                        "revision": record.revision,
                        "control_epoch": record.control_epoch,
                        "state": record.state,
                        "sensitive_entry": record.sensitive_entry,
                        "sensitive_entry_pending": record.sensitive_entry_pending,
                        "fresh_observation_required": record.fresh_observation_required,
                        # This is readback of the authenticated continuity owner's
                        # request, not a permission grant or another operator's
                        # bearer authority. Actions reauthorize the exact revision.
                        "owned_request": (
                            {
                                "request_id": record.request.request_id,
                                "expires_at_ms": record.request.expires_at_ms,
                                "maximum_until_ms": record.request.maximum_until_ms,
                                "lease_until_ms": record.lease_until_ms,
                                "pending_lease_until_ms": record.pending_lease_until_ms,
                                "settled_input_sequence": record.settled_input_sequence,
                                "pending_input_sequence": record.pending_input_sequence,
                                "checkpoint_consent": record.checkpoint_consent,
                            }
                            if record.request is not None
                            and record.request.operator.subject == principal.subject
                            and record.request.operator.tenant == principal.tenant
                            and record.request.operator.operator_session_id == operator_session_id
                            else None
                        ),
                    }
                    for record in records
                ]
            },
            headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
        )

    @router.post("/pages")
    async def pages(request: Request, context: Annotated[AuthContext, Depends(authenticated)]):
        principal = principal_for(request, context)
        if service is None:
            raise HTTPException(503, "Browser page discovery is not configured.")
        try:
            operator_session_id = sessions.verify(
                request.headers.get("x-cayu-browser-operator", ""), principal
            )
            intent = await _parse_control_intent(request, BrowserPagesIntent)
            result = await service.discover_pages(
                principal=principal, operator_session_id=operator_session_id, intent=intent
            )
        except (BrowserControlPermissionDenied, BrowserControlConflict, ValueError, TypeError):
            raise HTTPException(403, "Browser page discovery is unavailable.") from None
        except TimeoutError:
            raise HTTPException(503, "Browser page discovery did not settle.") from None
        from starlette.responses import JSONResponse

        return JSONResponse(
            {
                "active_page_id": result.active_page_id,
                "pages": [page.model_dump(mode="json") for page in result.pages],
                "locations": [item.model_dump(mode="json") for item in result.locations],
            },
            headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
        )

    @router.post("/takeover")
    async def takeover(request: Request, context: Annotated[AuthContext, Depends(authenticated)]):
        principal = principal_for(request, context)
        status = 403
        try:
            operator_session_id = sessions.verify(
                request.headers.get("x-cayu-browser-operator", ""), principal
            )
            intent = await _parse_control_intent(request, BrowserTakeoverIntent)
            result = await coordinator.request_takeover(
                principal=principal, operator_session_id=operator_session_id, intent=intent
            )
            from starlette.responses import JSONResponse

            return JSONResponse(
                {
                    "state": result.state,
                    "revision": result.revision,
                    "control_epoch": result.control_epoch,
                },
                headers={"Cache-Control": "no-store"},
            )
        except BrowserControlPermissionDenied:
            status = 403
        except BrowserControlConflict:
            status = 409
        except (ValidationError, ValueError, TypeError):
            status = 422
        except BrowserControlPublicationPending:
            status = 503
        except HTTPException:
            raise
        except Exception:
            status = 503
        # Fixed diagnostics: no reflected input, token, policy message, or record.
        raise HTTPException(status, "Browser control request was not admitted.")

    @router.post("/view-ticket")
    async def view_ticket(
        request: Request, context: Annotated[AuthContext, Depends(authenticated)]
    ):
        principal = principal_for(request, context)
        status = 503
        try:
            if view_tickets is None:
                raise HTTPException(503, "Browser viewer transport is not configured.")
            operator_session_id = sessions.verify(
                request.headers.get("x-cayu-browser-operator", ""), principal
            )
            intent = await _parse_control_intent(request, BrowserViewIntent)
            token = await view_tickets.issue(
                principal=principal, operator_session_id=operator_session_id, intent=intent
            )
            from starlette.responses import JSONResponse

            return JSONResponse(
                {"ticket": token, "expires_in_seconds": 30},
                headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
            )
        except BrowserControlPermissionDenied:
            status = 403
        except BrowserControlConflict:
            status = 409
        except (ValidationError, ValueError, TypeError):
            status = 422
        except HTTPException:
            raise
        except Exception:
            status = 503
        raise HTTPException(status, "Browser viewer request was not admitted.")

    @router.post("/renew")
    async def renew(request: Request, context: Annotated[AuthContext, Depends(authenticated)]):
        principal = principal_for(request, context)
        status = 503
        try:
            operator_session_id = sessions.verify(
                request.headers.get("x-cayu-browser-operator", ""), principal
            )
            intent = await _parse_control_intent(request, BrowserRenewIntent)
            result = await coordinator.request_renewal(
                principal=principal, operator_session_id=operator_session_id, intent=intent
            )
            from starlette.responses import JSONResponse

            return JSONResponse(
                {
                    "state": result.state,
                    "revision": result.revision,
                    "control_epoch": result.control_epoch,
                    "lease_until_ms": result.lease_until_ms,
                    "pending_lease_until_ms": result.pending_lease_until_ms,
                },
                status_code=202,
                headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
            )
        except BrowserControlPermissionDenied:
            status = 403
        except BrowserControlConflict:
            status = 409
        except (ValidationError, ValueError, TypeError):
            status = 422
        except HTTPException:
            raise
        except Exception:
            status = 503
        raise HTTPException(status, "Browser renewal request was not admitted.")

    @router.post("/handback")
    async def handback(request: Request, context: Annotated[AuthContext, Depends(authenticated)]):
        principal = principal_for(request, context)
        status = 503
        try:
            operator_session_id = sessions.verify(
                request.headers.get("x-cayu-browser-operator", ""), principal
            )
            intent = await _parse_control_intent(request, BrowserHandbackIntent)
            result = await coordinator.request_handback(
                principal=principal,
                operator_session_id=operator_session_id,
                intent=intent,
            )
            from starlette.responses import JSONResponse

            return JSONResponse(
                {
                    "state": result.state,
                    "revision": result.revision,
                    "control_epoch": result.control_epoch,
                },
                headers={"Cache-Control": "no-store"},
            )
        except BrowserControlPermissionDenied:
            status = 403
        except BrowserControlConflict:
            status = 409
        except (ValidationError, ValueError, TypeError):
            status = 422
        except HTTPException:
            raise
        except Exception:
            status = 503
        raise HTTPException(status, "Browser handback request was not admitted.")

    @router.post("/input-ticket")
    async def input_ticket(
        request: Request, context: Annotated[AuthContext, Depends(authenticated)]
    ):
        principal = principal_for(request, context)
        status = 503
        try:
            if input_tickets is None:
                raise HTTPException(503, "Browser input transport is not configured.")
            operator_session_id = sessions.verify(
                request.headers.get("x-cayu-browser-operator", ""), principal
            )
            intent = await _parse_control_intent(request, BrowserTextInputIntent)
            token = await input_tickets.issue(
                principal=principal, operator_session_id=operator_session_id, intent=intent
            )
            from starlette.responses import JSONResponse

            return JSONResponse(
                {"ticket": token, "expires_in_seconds": 10},
                headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
            )
        except BrowserControlPermissionDenied:
            status = 403
        except BrowserControlConflict:
            status = 409
        except (ValidationError, ValueError, TypeError):
            status = 422
        except HTTPException:
            raise
        except Exception:
            status = 503
        raise HTTPException(status, "Browser input request was not admitted.")

    @router.post("/sensitive-entry")
    async def sensitive_entry(
        request: Request, context: Annotated[AuthContext, Depends(authenticated)]
    ):
        principal = principal_for(request, context)
        status = 503
        try:
            operator_session_id = sessions.verify(
                request.headers.get("x-cayu-browser-operator", ""), principal
            )
            intent = await _parse_control_intent(request, BrowserSensitiveEntryIntent)
            result = await coordinator.request_sensitive_entry(
                principal=principal,
                operator_session_id=operator_session_id,
                intent=intent,
            )
            from starlette.responses import JSONResponse

            return JSONResponse(
                {
                    "state": result.state,
                    "revision": result.revision,
                    "control_epoch": result.control_epoch,
                    "sensitive_entry_pending": result.sensitive_entry_pending,
                    "sensitive_entry": result.sensitive_entry,
                },
                status_code=202,
                headers={"Cache-Control": "no-store"},
            )
        except BrowserControlPermissionDenied:
            status = 403
        except BrowserControlConflict:
            status = 409
        except (ValidationError, ValueError, TypeError):
            status = 422
        except HTTPException:
            raise
        except Exception:
            status = 503
        raise HTTPException(status, "Browser sensitive-entry request was not admitted.")

    return router
