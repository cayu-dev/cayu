"""Host enrollment and the authenticated entrance for outside fork controllers.

A fork controller is an application outside the host, such as a supervisor,
that forks host sessions with its own overlay. The host enrolls the controller
once: which authenticated subjects act for it, which host agents it may fork,
and which tools it serves over a remote MCP connection. The host ships no
controller code. Only ``ForkControllerAccess`` can attach an overlay; message
text and ordinary fork requests never introduce controller tools or authority.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from contextlib import aclosing
from dataclasses import KW_ONLY, dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from cayu._validation import require_durable_clean_nonblank
from cayu.approvals.actors import ResolutionActor, ResolutionActorSource
from cayu.mcp.tools import McpToolset
from cayu.sessions.fork_overlays import (
    ForkOverlay,
    fork_controller_agent_name,
    fork_overlay_record_from_metadata,
    validate_fork_controller_name,
)
from cayu.tools.exposure import (
    TOOL_EXPOSURE_MAX_REGISTERED_TOOLS,
    ToolCapabilityCeiling,
    tool_capability_ceiling_from_session_metadata,
)

if TYPE_CHECKING:
    from cayu.applications import CayuApp
    from cayu.events import Event
    from cayu.execution_profiles import ExecutionProfileAdoptionIntent
    from cayu.runtime._runtime_records import RegisteredAgentState
    from cayu.sessions.forks import ForkSourceSnapshot
    from cayu.sessions.requests import ForkSessionRequest, ResumeRequest

_FORK_CONTROLLER_MAX_SUBJECTS = 64
_FORK_CONTROLLER_MAX_AGENTS = 256
_VERIFIED_FORK_CONTROLLER_TOKEN = object()


class ForkControllerDenied(PermissionError):
    """The caller is not an enrolled controller, or the target is outside its enrollment."""


class ForkControllerMode(StrEnum):
    """What a controller may do beyond its own forks.

    ``observe`` is the only mode: the controller forks, continues and reads its
    own forks and never writes to the session it forked from.
    """

    OBSERVE = "observe"


def _unique_names(
    value: Iterable[str],
    field_name: str,
    max_items: int,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(value, str | bytes):
        raise TypeError(f"{field_name} must be an iterable of strings.")
    names = tuple(value)
    if not names and not allow_empty:
        raise ValueError(f"{field_name} must name at least one entry.")
    if len(names) > max_items:
        raise ValueError(f"{field_name} cannot contain more than {max_items} entries.")
    if any(type(name) is not str for name in names):
        raise TypeError(f"{field_name} must contain strings.")
    names = tuple(
        require_durable_clean_nonblank(name, f"{field_name}[{index}]")
        for index, name in enumerate(names)
    )
    if len(names) != len(set(names)):
        raise ValueError(f"{field_name} must not repeat entries.")
    return names


@dataclass(frozen=True, slots=True)
class ForkController:
    """Host-side enrollment of one outside fork controller.

    ``subjects`` are authenticated identities allowed to act as the
    controller. ``agents`` are the host agents it may fork. ``host_tools`` are
    the only host tools any of its forks may hold; the default is none, and a
    fork's inherited host tools are narrowed to it. It may also name hosted tool
    types such as ``openai_web_search``: an agent registered with a hosted tool
    the allowance doesn't name cannot be forked by this controller. ``mcp_toolsets`` are
    optional controller-served tools; they must use a remote URL transport and
    may not resolve host secrets, so a call carries only model arguments to the
    controller and never host credentials. ``mode`` is the declared operating
    mode; only ``observe`` exists.
    """

    name: str
    subjects: tuple[str, ...]
    agents: tuple[str, ...]
    # Options are keyword-only, so adding one never shifts another's position.
    _: KW_ONLY
    host_tools: tuple[str, ...] = ()
    mcp_toolsets: tuple[McpToolset, ...] = ()
    mode: ForkControllerMode = ForkControllerMode.OBSERVE

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", validate_fork_controller_name(self.name, "name"))
        object.__setattr__(
            self,
            "subjects",
            _unique_names(self.subjects, "subjects", _FORK_CONTROLLER_MAX_SUBJECTS),
        )
        object.__setattr__(
            self,
            "agents",
            _unique_names(self.agents, "agents", _FORK_CONTROLLER_MAX_AGENTS),
        )
        object.__setattr__(
            self,
            "host_tools",
            _unique_names(
                self.host_tools,
                "host_tools",
                TOOL_EXPOSURE_MAX_REGISTERED_TOOLS,
                allow_empty=True,
            ),
        )
        if type(self.mode) is not ForkControllerMode:
            if type(self.mode) is not str:
                raise TypeError("mode must be a ForkControllerMode.")
            try:
                mode = ForkControllerMode(self.mode)
            except ValueError:
                raise ValueError("mode must be 'observe'; no other mode exists yet.") from None
            object.__setattr__(self, "mode", mode)
        if isinstance(self.mcp_toolsets, str | bytes):
            raise TypeError("mcp_toolsets must be an iterable of McpToolset instances.")
        toolsets = tuple(self.mcp_toolsets)
        for index, toolset in enumerate(toolsets):
            if not isinstance(toolset, McpToolset):
                raise TypeError(f"mcp_toolsets[{index}] must be a McpToolset.")
            server = toolset.server
            if server.url is None:
                raise ValueError(
                    f"mcp_toolsets[{index}] must reach the controller over a URL; a local "
                    "stdio server would run on the host."
                )
            if server.secret_env or server.secret_headers:
                raise ValueError(
                    f"mcp_toolsets[{index}] may not resolve host secrets; controller-served "
                    "tools never receive host credentials."
                )
        object.__setattr__(self, "mcp_toolsets", toolsets)

    @property
    def served_tool_names(self) -> tuple[str, ...]:
        """Names of the tools this controller serves, sorted."""

        return tuple(sorted(tool.name for toolset in self.mcp_toolsets for tool in toolset.tools))


class ForkControllerEnrollment(BaseModel):
    """What one authenticated controller subject is allowed to do.

    It describes only the caller's own enrollment: never other controllers,
    other subjects of the same controller, or agents it is not enrolled for.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    controller: str
    subject: str
    mode: ForkControllerMode
    agents: tuple[str, ...]
    host_tools: tuple[str, ...]
    served_tools: tuple[str, ...]


@dataclass(frozen=True, slots=True, repr=False)
class VerifiedForkController:
    """In-process proof that an enrolled subject is acting as one controller."""

    controller: ForkController
    subject: str
    _token: object

    def __post_init__(self) -> None:
        if self._token is not _VERIFIED_FORK_CONTROLLER_TOKEN:
            raise TypeError("VerifiedForkController is created only by fork-controller access.")
        if type(self.controller) is not ForkController:
            raise TypeError("controller must be a ForkController.")
        object.__setattr__(
            self,
            "subject",
            require_durable_clean_nonblank(self.subject, "subject"),
        )
        if self.subject not in self.controller.subjects:
            raise ForkControllerDenied("The subject is not enrolled for this controller.")

    def __repr__(self) -> str:
        return f"VerifiedForkController(controller={self.controller.name!r})"


def require_verified_fork_controller(value: object) -> VerifiedForkController:
    if type(value) is not VerifiedForkController:
        raise ForkControllerDenied("Fork overlays require an authenticated, enrolled controller.")
    value.__post_init__()
    return value


class ForkControllerAccess:
    """Authenticated entrance for one enrolled controller.

    Construct through ``CayuApp.fork_controller_access(subject)`` after the host
    has authenticated ``subject``. Request bodies never select the controller.
    """

    def __init__(self, app: CayuApp, verified: VerifiedForkController) -> None:
        self.__app = app
        self.__verified = require_verified_fork_controller(verified)

    @property
    def controller_name(self) -> str:
        return self.__verified.controller.name

    @property
    def subject(self) -> str:
        return self.__verified.subject

    def enrollment(self) -> ForkControllerEnrollment:
        """Describe this controller's own enrollment for the authenticated subject."""

        controller = self.__verified.controller
        return ForkControllerEnrollment(
            controller=controller.name,
            subject=self.subject,
            mode=controller.mode,
            agents=controller.agents,
            host_tools=controller.host_tools,
            served_tools=controller.served_tool_names,
        )

    async def fork(self, request: ForkSessionRequest) -> AsyncIterator[Event]:
        """Fork one host session with this controller's overlay."""

        from cayu.sessions.requests import ForkSessionRequest

        if type(request) is not ForkSessionRequest:
            raise TypeError("Controller forks require ForkSessionRequest.")
        if request.overlay is None:
            raise ValueError("Controller forks require an overlay.")
        if request.overlay.controller != self.controller_name:
            raise ForkControllerDenied("The overlay names a different controller.")
        if request.session_id is not None:
            # A named destination that exists must be this controller's own fork;
            # otherwise a conflict would reveal that a host session id exists.
            existing = await self.__app.session_store.load(request.session_id)
            existing_record = (
                None if existing is None else fork_overlay_record_from_metadata(existing.metadata)
            )
            if existing is not None and (
                existing_record is None
                or existing_record.overlay.controller != self.controller_name
            ):
                raise ForkControllerDenied("The fork destination is not available.")
        request = request.model_copy(
            update={"profile_adoption": self._verified_adoption(request.profile_adoption)}
        )
        async with aclosing(
            self.__app.fork_session(request, _fork_controller=self.__verified)
        ) as stream:
            async for event in stream:
                yield event

    def _verified_adoption(
        self, intent: ExecutionProfileAdoptionIntent | None
    ) -> ExecutionProfileAdoptionIntent | None:
        """Rebuild adoption provenance from the authenticated subject, never the body."""

        if intent is None:
            return None
        actor = intent.requested_by
        if actor.subject != self.subject:
            raise ForkControllerDenied("profile_adoption.requested_by must be this controller.")
        if actor.source is ResolutionActorSource.SYSTEM:
            raise ForkControllerDenied("A controller cannot claim system provenance.")
        return intent.model_copy(
            update={
                "requested_by": ResolutionActor(
                    subject=self.subject,
                    source=actor.source or ResolutionActorSource.REQUEST,
                )
            }
        )

    async def snapshot_source(self, source_session_id: str) -> ForkSourceSnapshot:
        """Capture an exact source snapshot for an ``initial_invocation`` fork."""

        private_id = await self.__app._resolve_public_session_id(source_session_id)
        source = await self.__app.session_store.load(private_id)
        enrolled = self.__verified.controller.agents
        if source is None or (
            source.agent_name not in enrolled
            and source.agent_name
            not in {fork_controller_agent_name(agent, self.controller_name) for agent in enrolled}
        ):
            raise ForkControllerDenied("The source is not an agent this controller may fork.")
        record = fork_overlay_record_from_metadata(source.metadata)
        if record is not None and record.overlay.controller != self.controller_name:
            # Another controller's fork is private to it, even without its own body.
            raise ForkControllerDenied("The source is not an agent this controller may fork.")
        return await self.__app.snapshot_fork_source(source_session_id)

    async def resume(self, request: ResumeRequest) -> AsyncIterator[Event]:
        """Continue a fork this controller created; its overlay stays exactly as admitted."""

        from cayu.sessions.requests import ResumeRequest

        if type(request) is not ResumeRequest:
            raise TypeError("Controller resumes require ResumeRequest.")
        private_id = await self.__app._resolve_public_session_id(request.session_id)
        session = await self.__app.session_store.load(private_id)
        record = None if session is None else fork_overlay_record_from_metadata(session.metadata)
        if session is None or record is None or record.overlay.controller != self.controller_name:
            raise ForkControllerDenied("The session is not a fork of this controller.")
        # Enrollment is re-checked on every continuation, so removing an agent
        # from the controller revokes its existing forks too.
        try:
            registered_agent = self.__app._get_registered_agent(session.agent_name)
        except KeyError:
            raise ForkControllerDenied(
                "The session is not an agent this controller may fork."
            ) from None
        if registered_agent.policy_agent_name not in self.__verified.controller.agents:
            raise ForkControllerDenied("The session is not an agent this controller may fork.")
        require_requested_fork_controller_host_tools(
            self.__verified.controller, request.tool_capability_ceiling
        )
        # The host-tool allowance is re-checked too: narrowing it revokes host
        # tools that an existing fork's durable ceiling still names.
        require_fork_controller_host_tools(
            self.__verified.controller,
            tool_capability_ceiling_from_session_metadata(session.metadata),
            registered_agent,
        )
        request = request.model_copy(
            update={"profile_adoption": self._verified_adoption(request.profile_adoption)}
        )
        async with aclosing(self.__app.resume(request)) as stream:
            async for event in stream:
                yield event


def fork_controller_access(
    app: CayuApp,
    subject: str,
    controllers: Iterable[ForkController],
) -> ForkControllerAccess:
    """Resolve the enrolled controller for an authenticated subject."""

    subject = require_durable_clean_nonblank(subject, "subject")
    matches = [controller for controller in controllers if subject in controller.subjects]
    if len(matches) != 1:
        raise ForkControllerDenied("The subject is not an enrolled fork controller.")
    return ForkControllerAccess(
        app,
        VerifiedForkController(
            controller=matches[0],
            subject=subject,
            _token=_VERIFIED_FORK_CONTROLLER_TOKEN,
        ),
    )


def fork_overlay_agent_name(
    controller: ForkController,
    overlay: ForkOverlay,
    *,
    host_agent_name: str,
) -> str:
    """Select the child body: the host agent, or its controller body when tools are granted."""

    if host_agent_name not in controller.agents:
        raise ForkControllerDenied("The controller is not enrolled for this agent.")
    if not overlay.tools:
        return host_agent_name
    if not controller.mcp_toolsets:
        raise ValueError("Fork overlay tools require a controller enrolled with mcp_toolsets.")
    return fork_controller_agent_name(host_agent_name, controller.name)


def require_fork_controller_host_tools(
    controller: ForkController,
    ceiling: ToolCapabilityCeiling,
    registered_agent: RegisteredAgentState,
) -> None:
    """Fail closed when a fork's durable ceiling names a host tool outside the allowance.

    A host tool is one the child's agent registers and the controller does not
    serve. A name the agent no longer registers cannot run, so it is not one.
    """

    allowed = {*controller.host_tools, *controller.served_tool_names}
    if any(name in registered_agent.tools and name not in allowed for name in ceiling.tool_names):
        raise ForkControllerDenied("The fork holds a host tool this controller may not use.")
    require_fork_controller_hosted_tools(controller, registered_agent)


def require_fork_controller_hosted_tools(
    controller: ForkController,
    registered_agent: RegisteredAgentState,
) -> None:
    """Fail closed when the fork's agent carries a provider-hosted tool outside the allowance.

    Hosted tools such as OpenAI web search come from the agent's registration,
    not from the tool ceiling, so a fork cannot be narrowed to drop them. A
    controller may fork such an agent only when the host names each hosted
    tool's type in ``host_tools``.
    """

    if any(tool.type not in controller.host_tools for tool in registered_agent.hosted_tools):
        raise ForkControllerDenied(
            "The fork's agent has a hosted tool this controller may not use."
        )


def require_requested_fork_controller_host_tools(
    controller: ForkController,
    *requested: ToolCapabilityCeiling | None,
) -> None:
    """Deny explicit ceilings that name anything but allowed host tools or served tools."""

    allowed = {*controller.host_tools, *controller.served_tool_names}
    if any(
        name not in allowed
        for ceiling in requested
        if ceiling is not None
        for name in ceiling.tool_names
    ):
        raise ForkControllerDenied("The request names a host tool this controller may not use.")


def fork_controller_host_tool_ceiling(
    controller: ForkController,
    ceiling: ToolCapabilityCeiling,
) -> ToolCapabilityCeiling:
    """Narrow a resolved child ceiling to the controller's allowance plus its served tools."""

    allowed = {*controller.host_tools, *controller.served_tool_names}
    return ToolCapabilityCeiling(
        tool_names=tuple(name for name in ceiling.tool_names if name in allowed)
    )


def fork_overlay_tool_capability_ceiling(
    ceiling: ToolCapabilityCeiling,
    registered_agent: RegisteredAgentState,
    overlay: ForkOverlay | None,
) -> ToolCapabilityCeiling:
    """Keep the narrowed host ceiling and add only this overlay's granted controller tools."""

    controller_tools = registered_agent.fork_controller_tool_names
    granted = () if overlay is None else overlay.tool_names
    if not controller_tools and not granted:
        return ceiling
    if any(name not in controller_tools for name in granted):
        raise ValueError("A fork overlay tool is not served by the enrolled controller.")
    selected = {name for name in ceiling.tool_names if name not in controller_tools}
    selected.update(granted)
    return ToolCapabilityCeiling(
        tool_names=tuple(
            capability.name
            for capability in registered_agent.tool_capabilities
            if capability.name in selected
        )
    )


__all__ = [
    "ForkController",
    "ForkControllerAccess",
    "ForkControllerDenied",
    "ForkControllerEnrollment",
    "ForkControllerMode",
    "VerifiedForkController",
    "fork_controller_access",
]
