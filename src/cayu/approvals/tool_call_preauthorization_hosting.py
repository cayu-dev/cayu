"""Tool-call preauthorizations that ``cayu serve`` serves.

``app.host_tool_call_preauthorizations(store, authorize=...)`` names the
application's :class:`~cayu.approvals.ToolCallPreauthorizationStore` in its
factory. ``cayu serve`` then mounts the routes at
``{api}/tool-call-preauthorizations`` behind its authentication, and ``cayu
preauthorizations`` reaches the same routes from the command line. The person
who issues or revokes a record is always the authenticated caller.

The application still assembles the policy itself
(:class:`~cayu.approvals.PreauthorizedToolCallPolicy` around its agents' tool
policy, on the same store). Hosting adds no authority of its own: ``authorize``
decides every issue, read and revoke.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from cayu.applications import CayuApp
    from cayu.approvals.tool_call_preauthorizations import (
        ToolCallPreauthorization,
        ToolCallPreauthorizationStore,
    )

#: What a caller asks to do with a record.
ToolCallPreauthorizationAction = Literal["issue", "read", "revoke"]

#: Decides whether one authenticated caller (an ``AuthContext``) may take one action on one
#: record. For ``issue`` the record is the one about to be stored, naming the caller.
ToolCallPreauthorizationAccess = Callable[
    [Any, ToolCallPreauthorizationAction, "ToolCallPreauthorization"], Awaitable[bool]
]


@dataclass(frozen=True, slots=True)
class ToolCallPreauthorizationHosting:
    """The preauthorization store of one application and who may use it."""

    store: ToolCallPreauthorizationStore
    authorize: ToolCallPreauthorizationAccess


def tool_call_preauthorization_hosting(app: CayuApp) -> ToolCallPreauthorizationHosting | None:
    """What ``host_tool_call_preauthorizations`` set up, or ``None`` before it ran."""

    return app._tool_call_preauthorization_hosting


def host_tool_call_preauthorizations(
    app: CayuApp,
    store: ToolCallPreauthorizationStore,
    *,
    authorize: ToolCallPreauthorizationAccess,
) -> ToolCallPreauthorizationHosting:
    """Have ``cayu serve`` serve this application's tool-call preauthorizations.

    - ``store``: the :class:`~cayu.approvals.ToolCallPreauthorizationStore` the
      application's ``PreauthorizedToolCallPolicy`` reads.
    - ``authorize``: an async callable taking the authenticated caller (an
      ``AuthContext``), the action (``"issue"``, ``"read"`` or ``"revoke"``) and
      the record, returning ``True`` when the caller may do it. A record it hides
      from ``read`` answers 404 and is left out of listings.

    The routes need an authenticated caller, whom an issued record names, so
    ``cayu serve --dev`` does not serve them.
    """

    from cayu.approvals.tool_call_preauthorizations import ToolCallPreauthorizationStore

    if not isinstance(store, ToolCallPreauthorizationStore):
        raise TypeError("store must be a ToolCallPreauthorizationStore.")
    if not callable(authorize):
        raise TypeError(
            "authorize must be an async callable taking a caller, an action and a record."
        )
    if app._tool_call_preauthorization_hosting is not None:
        raise ValueError("This application already hosts its tool-call preauthorizations.")
    app._tool_call_preauthorization_hosting = ToolCallPreauthorizationHosting(
        store=store, authorize=authorize
    )
    return app._tool_call_preauthorization_hosting


__all__ = [
    "ToolCallPreauthorizationAccess",
    "ToolCallPreauthorizationAction",
    "ToolCallPreauthorizationHosting",
    "host_tool_call_preauthorizations",
    "tool_call_preauthorization_hosting",
]
