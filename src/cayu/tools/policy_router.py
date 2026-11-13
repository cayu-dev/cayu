"""Route each tool to its own policy.

``ToolPolicyRouter(routes, default=...)`` sends a call to the policy its tool is
routed to, and every other tool to ``default``. It decides nothing itself, so
``cayu check`` reads each tool's coverage from that tool's own route: a custom
policy routed to one tool leaves only that tool's coverage ``unknown``.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from cayu._validation import require_clean_nonblank
from cayu.tools.policy import (
    ApprovalFloorToolPolicy,
    GuardedToolPolicy,
    TaintAwareToolPolicy,
    ToolPolicy,
    ToolPolicyRequest,
    ToolPolicyResult,
)
from cayu.tools.setting_policy import SettingToolPolicy


class ToolPolicyRouter(ToolPolicy):
    """Authorize each call with the policy routed to its tool, else ``default``."""

    def __init__(self, routes: Mapping[str, ToolPolicy], *, default: ToolPolicy) -> None:
        if not isinstance(routes, Mapping):
            raise TypeError("routes must map tool names to ToolPolicy instances.")
        if not isinstance(default, ToolPolicy):
            raise TypeError("default must be a ToolPolicy.")
        copied: dict[str, ToolPolicy] = {}
        for name, policy in routes.items():
            tool_name = require_clean_nonblank(name, "tool_name")
            if not isinstance(policy, ToolPolicy):
                raise TypeError(f"routes[{tool_name!r}] must be a ToolPolicy.")
            copied[tool_name] = policy
        self._routes = MappingProxyType(copied)
        self._default = default
        _require_one_taint_policy(self)

    @property
    def routes(self) -> Mapping[str, ToolPolicy]:
        return self._routes

    @property
    def default(self) -> ToolPolicy:
        return self._default

    def policy_for(self, tool_name: str) -> ToolPolicy:
        """The policy that decides calls to ``tool_name``."""

        return self._routes.get(tool_name, self._default)

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        return await self.policy_for(request.tool_name).authorize(request)


def _policies_behind(policy: ToolPolicy, *, structured_commands: bool) -> list[ToolPolicy]:
    """Every distinct policy reached from ``policy`` through Cayu's composition wrappers.

    Follows a router's routes and default, a guarded policy's ``then``, an approval
    floor's ``inner``, an authorization-grant policy's ``then`` and, when
    ``structured_commands`` is set, a structured-command policy's base policy.
    """

    from cayu.approvals.authorization_grant_policy import ToolAuthorizationGrantPolicy
    from cayu.tools.structured_commands import StructuredCommandToolPolicy

    reached: list[ToolPolicy] = []
    pending = [policy]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        reached.append(current)
        if type(current) is ToolPolicyRouter:
            pending.extend((*current.routes.values(), current.default))
        elif type(current) is GuardedToolPolicy or type(current) is ToolAuthorizationGrantPolicy:
            if current.then is not None:
                pending.append(current.then)
        elif type(current) is ApprovalFloorToolPolicy:
            pending.append(current.inner)
        elif (
            structured_commands
            and type(current) is StructuredCommandToolPolicy
            and current._base_policy is not None
        ):
            pending.append(current._base_policy)
    return reached


def taint_policies(policy: ToolPolicy) -> tuple[TaintAwareToolPolicy, ...]:
    """Every distinct ``TaintAwareToolPolicy`` that decides behind ``policy``.

    Follows Cayu's composition wrappers: a router's routes and default, a
    guarded policy's ``then``, an approval floor's ``inner`` and an
    authorization-grant policy's ``then``.
    """

    return tuple(
        item
        for item in _policies_behind(policy, structured_commands=False)
        if isinstance(item, TaintAwareToolPolicy)
    )


def setting_policies(policy: ToolPolicy) -> tuple[SettingToolPolicy, ...]:
    """Every distinct ``SettingToolPolicy`` that decides behind ``policy``, following
    the same wrappers as ``taint_policies`` and a structured-command policy's base."""

    return tuple(
        item
        for item in _policies_behind(policy, structured_commands=True)
        if type(item) is SettingToolPolicy
    )


def _require_one_taint_policy(router: ToolPolicyRouter) -> None:
    """Taint labels come from one taint-aware policy; refuse a router mixing several.

    The runtime collects a session's taint labels from the agent's one
    ``TaintAwareToolPolicy``, so two different ones behind a router would leave
    one of them without the labels it protects against.
    """

    if len(taint_policies(router)) > 1:
        raise ValueError(
            "A ToolPolicyRouter can use at most one TaintAwareToolPolicy across its routes "
            "and default; route the tools it protects to the same instance."
        )


__all__ = ["ToolPolicyRouter"]
