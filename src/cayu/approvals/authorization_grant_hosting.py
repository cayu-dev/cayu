"""Tool authorization grants that ``cayu serve`` serves.

``app.host_tool_authorization_grants(service, authorize_read=...)`` names the
application's :class:`~cayu.approvals.ToolAuthorizationGrantService` in its
factory. ``cayu serve`` then mounts the grant routes at
``{api}/tool-authorization-grants``, behind the same authentication and inbox
access rule as its other approval routes, so a person can approve a pending
call and have it remembered, and list, inspect and revoke grants. ``cayu serve``
and ``cayu worker`` also settle the staged grants a stopped process left behind
when they start.

The application still assembles the parts itself: the grant store, the
:class:`~cayu.approvals.ToolAuthorizationGrantPolicy` around its agents' tool
policy, and the service with its grant authority. Hosting adds no authority of
its own. An application that builds its own server passes a
``cayu.server.authorization_grants.ToolAuthorizationGrantServer`` to
``create_server`` as before, and needs none of this.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cayu.applications import CayuApp
    from cayu.approvals.authorization_grant_service import ToolAuthorizationGrantService
    from cayu.approvals.authorization_grants import ToolAuthorizationGrant

logger = logging.getLogger(__name__)

#: How long ``cayu worker`` waits for staged grants to settle before it starts claiming.
STARTUP_RECONCILIATION_SECONDS = 30.0

#: Decides whether one authenticated caller (an ``AuthContext``) may see one grant.
ToolAuthorizationGrantReadAccess = Callable[[Any, "ToolAuthorizationGrant"], Awaitable[bool]]


@dataclass(frozen=True, slots=True)
class ToolAuthorizationGrantHosting:
    """The grant service of one application and who may read its grants.

    Create it with ``app.host_tool_authorization_grants(...)``. ``service``
    decides creation and revocation through its grant authority;
    ``authorize_read`` decides which authenticated callers see a grant.
    """

    service: ToolAuthorizationGrantService
    authorize_read: ToolAuthorizationGrantReadAccess


def tool_authorization_grant_hosting(app: CayuApp) -> ToolAuthorizationGrantHosting | None:
    """The grants ``host_tool_authorization_grants`` set up, or ``None`` before it ran.

    ``app.tool_authorization_grant_hosting`` reads it.
    """

    return app._tool_authorization_grant_hosting


def host_tool_authorization_grants(
    app: CayuApp,
    service: ToolAuthorizationGrantService,
    *,
    authorize_read: ToolAuthorizationGrantReadAccess,
) -> ToolAuthorizationGrantHosting:
    """Have ``cayu serve`` serve this application's tool authorization grants.

    - ``service``: the :class:`~cayu.approvals.ToolAuthorizationGrantService`
      built on this application. Its grant authority decides who may create and
      revoke a grant.
    - ``authorize_read``: an async callable taking the authenticated caller (an
      ``AuthContext``) and a grant, returning ``True`` when that caller may see
      the grant. A grant it hides answers 404 and is left out of listings.

    The routes need an authenticated caller, whose identity becomes the grant's
    creator, so ``cayu serve --dev`` does not serve them.
    """

    from cayu.approvals.authorization_grant_service import ToolAuthorizationGrantService

    if not isinstance(service, ToolAuthorizationGrantService):
        raise TypeError("service must be a ToolAuthorizationGrantService.")
    if not callable(authorize_read):
        raise TypeError("authorize_read must be an async callable taking a caller and a grant.")
    if service.app is not app:
        raise ValueError("The grant service must be built on this application.")
    if app._tool_authorization_grant_hosting is not None:
        raise ValueError("This application already hosts its tool authorization grants.")
    app._tool_authorization_grant_hosting = ToolAuthorizationGrantHosting(
        service=service, authorize_read=authorize_read
    )
    return app._tool_authorization_grant_hosting


async def reconcile_hosted_tool_authorization_grants(
    app: CayuApp, *, timeout_s: float | None = None
) -> int | None:
    """Settle the staged grants of the application's hosted service, once.

    A process that stopped between approving a call and activating its grant
    leaves the grant ``staged``, where it authorizes nothing. ``cayu serve`` and
    ``cayu worker`` call this when they start, so the grant becomes ``active``
    or ``abandoned`` by what the approval recorded. Returns how many grants were
    looked at, or ``None`` when the application hosts none or the pass did not
    finish: a failure or a timeout is logged and left to the next start, never
    raised, because a staged grant only means people are asked again.
    """

    hosting = app._tool_authorization_grant_hosting
    if hosting is None:
        return None
    try:
        settled = await asyncio.wait_for(hosting.service.reconcile_staged(), timeout=timeout_s)
    except Exception as exc:
        # A timeout is this pass's own bound or a store's; either way the rest waits.
        logger.warning(
            "Staged tool authorization grants were not all settled (%s); the next start "
            "settles the rest.",
            type(exc).__name__,
        )
        return None
    return len(settled)


__all__ = [
    "STARTUP_RECONCILIATION_SECONDS",
    "ToolAuthorizationGrantHosting",
    "host_tool_authorization_grants",
    "reconcile_hosted_tool_authorization_grants",
    "tool_authorization_grant_hosting",
]
