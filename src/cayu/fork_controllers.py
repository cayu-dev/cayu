"""Host enrollment and the authenticated entrance for outside fork controllers.

A fork controller is an application outside the host, such as a supervisor,
that forks host sessions with its own overlay. The host enrolls the controller
once: which authenticated subjects act for it, which host agents it may fork,
and which tools it serves over a remote MCP connection. The host ships no
controller code. Only ``ForkControllerAccess`` can attach an overlay; message
text and ordinary fork requests never introduce controller tools or authority.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Mapping
from contextlib import aclosing
from dataclasses import KW_ONLY, dataclass
from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from cayu._fork_controller_errors import ForkControllerDenied
from cayu._fork_controller_reads import (
    FORK_CONTROLLER_EVENT_QUERY_FIELDS,
    FORK_CONTROLLER_READ_DEFAULT_BYTES,
    FORK_CONTROLLER_READ_MAX_BYTES,
    ForkControllerReads,
)
from cayu._validation import require_durable_clean_nonblank
from cayu.approvals.actors import ResolutionActor, ResolutionActorSource
from cayu.budgets.base import (
    BudgetLimit,
    _copy_budget_limit,
    _effective_budget_limits,
    _EffectiveBudgetLimit,
)
from cayu.mcp.tools import McpToolset
from cayu.sessions._fork_controller_bounds import fork_lineage_admits
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.fork_overlays import (
    ForkOverlay,
    fork_controller_agent_name,
    fork_overlay_record_from_metadata,
    resolve_fork_lineage_async,
    validate_fork_controller_name,
)
from cayu.sessions.queries import SessionListResult, SessionQuery
from cayu.sessions.supervision_events import (
    SupervisionEvent,
    SupervisionEventPage,
    SupervisionEventQuery,
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
    from cayu.sessions.records import EventRecord, Session
    from cayu.sessions.requests import ForkSessionRequest, ResumeRequest

_FORK_CONTROLLER_MAX_SUBJECTS = 64
_FORK_CONTROLLER_MAX_AGENTS = 256
_VERIFIED_FORK_CONTROLLER_TOKEN = object()


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
    mode; only ``observe`` exists. ``allow_model_runs`` is False by default:
    the controller may create forks but not run them, so a fork with an
    ``initial_invocation`` and every resume are refused before any work starts.
    Turning it on requires ``fork_budget_limit``, the host's spend cap for each
    session in the controller's fork lineage; see ``fork_lineage_budget_limits``.
    """

    name: str
    subjects: tuple[str, ...]
    agents: tuple[str, ...]
    # Options are keyword-only, so adding one never shifts another's position.
    _: KW_ONLY
    host_tools: tuple[str, ...] = ()
    mcp_toolsets: tuple[McpToolset, ...] = ()
    mode: ForkControllerMode = ForkControllerMode.OBSERVE
    allow_model_runs: bool = False
    fork_budget_limit: BudgetLimit | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", validate_fork_controller_name(self.name, "name"))
        if type(self.allow_model_runs) is not bool:
            raise TypeError("allow_model_runs must be a bool.")
        if self.fork_budget_limit is not None:
            object.__setattr__(
                self, "fork_budget_limit", _validated_fork_budget_limit(self.fork_budget_limit)
            )
        if self.allow_model_runs and self.fork_budget_limit is None:
            raise ValueError(
                "allow_model_runs=True requires fork_budget_limit, the host's spend cap "
                "for each fork, such as BudgetLimit.usd(5, reservation=BudgetReservation("
                "max_input_tokens=200_000, max_output_tokens=16_000))."
            )
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


def _validated_fork_budget_limit(value: object) -> BudgetLimit:
    """Copy a per-fork cap, refusing anything that would not bound a fork's spend."""

    if type(value) is not BudgetLimit:
        raise TypeError("fork_budget_limit must be a BudgetLimit.")
    limit = _copy_budget_limit(value)
    if limit.scope != "session":
        raise ValueError(
            "fork_budget_limit must have scope='session'; it caps each fork session, "
            "as BudgetLimit.usd(amount) does by default."
        )
    if limit.action != "interrupt":
        raise ValueError("fork_budget_limit must use action='interrupt' to stop a fork.")
    if limit.allow_unpriced:
        raise ValueError(
            "fork_budget_limit cannot allow unpriced models; their spend would not count."
        )
    if limit.window.kind != "all_time":
        raise ValueError("fork_budget_limit must count a fork's whole life (an all_time window).")
    if limit.reservation is None:
        # Without a reservation, a dispatch that ends without usage evidence (a
        # stream the caller abandons, a lost response) would cost nothing, so a
        # controller could spend past the cap by abandoning its own requests.
        raise ValueError(
            "fork_budget_limit needs a reservation: the most tokens one model step may "
            "use, charged until the step's real usage is known."
        )
    return limit


class ForkBudgetLimitView(BaseModel):
    """The per-fork cap as a controller sees it: the amount, never the price book."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_estimated_cost: Decimal
    currency: str


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
    allow_model_runs: bool
    fork_budget_limit: ForkBudgetLimitView | None = None


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
        self.__reads = ForkControllerReads(app, self.__verified.controller)

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
            allow_model_runs=controller.allow_model_runs,
            fork_budget_limit=(
                None
                if controller.fork_budget_limit is None
                else ForkBudgetLimitView(
                    max_estimated_cost=controller.fork_budget_limit.max_estimated_cost,
                    currency=controller.fork_budget_limit.currency,
                )
            ),
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
        if request.initial_invocation is not None:
            require_fork_controller_model_runs(self.__verified.controller)
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
        lineage = await resolve_fork_lineage_async(source, self.__app.session_store.load)
        if not fork_lineage_admits(lineage, self.controller_name):
            # Another controller's fork, and every session created under it, is
            # private to that controller even without its own body.
            raise ForkControllerDenied("The source is not an agent this controller may fork.")
        return await self.__app.snapshot_fork_source(source_session_id)

    def event_correlation_ids(self, record: EventRecord) -> dict[str, str]:
        """Keyed correlation values for one event this controller read.

        SDK reads return the store's private records; these are the values the
        HTTP routes add to the same record's public payload. They join events of
        one session, stay the same across reads and restarts while the keyring
        keeps its active key, and carry no authority: no resolver accepts them.
        Without a configured alias keyring the result is empty.
        """

        return self.__reads.event_correlation_ids(record)

    def supervision_correlation_ids(
        self, event: SupervisionEvent
    ) -> dict[str, str | list[str] | None]:
        """Keyed correlation values for one supervision touch this controller read.

        ``tool_call_correlation_ids`` equal the ``tool_call_correlation_id`` of
        the touched calls' tool events, in ``tool_call_ids`` order; the model
        and round values equal those on the session's model and tool events.
        Without a configured alias keyring the result is empty.
        """

        return self.__reads.supervision_correlation_ids(event)

    async def sessions(self, query: SessionQuery | None = None) -> SessionListResult:
        """List the sessions this controller may read; others are absent, never errors."""

        return await self.__reads.sessions(query)

    async def session(self, session_id: str) -> Session:
        """Load one readable session; absent, foreign and other-controller ids are denied alike."""

        return await self.__reads.session(session_id)

    async def events(
        self,
        session_id: str,
        *,
        after_sequence: int | None = None,
        before_sequence: int | None = None,
        order_by: EventOrder = EventOrder.SEQUENCE_ASC,
        limit: int = 100,
        max_bytes: int = FORK_CONTROLLER_READ_DEFAULT_BYTES,
    ) -> list[EventRecord]:
        """Page one readable session's durable events in sequence order."""

        session = await self.session(session_id)
        return await self.query_events(
            EventQuery(
                session_id=session.id,
                after_sequence=after_sequence,
                before_sequence=before_sequence,
                order_by=order_by,
                limit=limit,
            ),
            max_bytes=max_bytes,
        )

    async def query_events(
        self,
        query: EventQuery,
        *,
        exclude_forked_sessions: bool = False,
        max_bytes: int = FORK_CONTROLLER_READ_DEFAULT_BYTES,
    ) -> list[EventRecord]:
        """Query events across readable sessions; ``since`` may be omitted.

        Sessions this controller may not read are filtered out by the store, so
        foreign ``session_ids`` simply match nothing. A query that is not for
        exactly one session reads in ascending sequence order only and starts
        after ``after_sequence``, 0 when omitted, so continuing from the last
        returned sequence never skips an event committed late with a lower one.
        ``exclude_forked_sessions`` drops every session in a forked lineage: a
        fork, including this controller's own, and every session created under
        one. Raises ``EventQueryResultTooLarge`` when the matching records
        exceed ``max_bytes``.
        """

        return await self.__reads.query_events(
            query, exclude_forked_sessions=exclude_forked_sessions, max_bytes=max_bytes
        )

    async def _query_event_page(
        self, query: EventQuery, *, exclude_forked_sessions: bool, max_bytes: int
    ) -> tuple[list[EventRecord], bool]:
        return await self.__reads.query_event_page(
            query, exclude_forked_sessions=exclude_forked_sessions, max_bytes=max_bytes
        )

    async def supervision_events(self, query: SupervisionEventQuery) -> SupervisionEventPage:
        """Page settled supervision touches on readable sessions; never includes content."""

        page, _anchors = await self._supervision_page_with_anchors(query)
        return page

    async def _supervision_page_with_anchors(
        self, query: SupervisionEventQuery
    ) -> tuple[SupervisionEventPage, dict[int, EventRecord]]:
        return await self.__reads.supervision_page_with_anchors(query)

    async def resume(self, request: ResumeRequest) -> AsyncIterator[Event]:
        """Continue a fork this controller created; its overlay stays exactly as admitted."""

        from cayu.sessions.requests import ResumeRequest

        if type(request) is not ResumeRequest:
            raise TypeError("Controller resumes require ResumeRequest.")
        # Every resume runs the model, so the gate refuses it before any lookup.
        require_fork_controller_model_runs(self.__verified.controller)
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


def require_fork_controller_model_runs(controller: ForkController) -> None:
    """Fail closed unless the host enrolled the controller with ``allow_model_runs``.

    Runs before any child is created or dispatched, so a refused request never
    reaches a model, a hosted tool or a controller-served MCP tool.
    """

    if not controller.allow_model_runs:
        raise ForkControllerDenied("The host has not allowed this controller to run its forks.")


FORK_CONTROLLER_BUDGET_NAMESPACE_PREFIX = "fork_controller:"


def fork_controller_budget_limits(
    controller: ForkController, session_id: str, instance_id: str
) -> tuple[_EffectiveBudgetLimit, ...]:
    """Return the controller's cap for one session, with that session's own ledger identity.

    The budget ledger counts reservations by limit identity, so the identity
    names the controller, the currency and the session incarnation (its id and
    instance id): each fork's
    reservations count only against its own cap, and a host's own limit with
    the same definition never shares a ledger entry or budget events with it.
    The identity does not depend on the amount, price book or reservation, so
    a host that changes the cap keeps charging the fork's earlier reservations,
    and a new session that reuses a deleted session's id starts a fresh entry.
    """

    limit = controller.fork_budget_limit
    if limit is None:
        return ()
    session_id = require_durable_clean_nonblank(session_id, "session_id")
    instance_id = require_durable_clean_nonblank(instance_id, "instance_id")
    return _effective_budget_limits(
        (limit,),
        identity_namespace=(
            f"{FORK_CONTROLLER_BUDGET_NAMESPACE_PREFIX}{controller.name}"
            f":currency:{limit.currency}:session:{session_id}:instance:{instance_id}"
        ),
    )


def fork_lineage_budget_limits(
    owner: str | None,
    session_id: str,
    instance_id: str,
    controllers: Mapping[str, ForkController],
) -> tuple[_EffectiveBudgetLimit, ...]:
    """Budget limits the host adds to every model step of a session with this lineage owner.

    ``owner`` is the controller that owns the session's fork lineage (its fork,
    or anything created under the fork), or ``None`` outside any controller's
    lineage, which gets no limits. A controller's lineage runs only while that
    controller is enrolled with ``allow_model_runs``, and then under its
    per-fork cap. The runtime applies this before each model step, whatever
    started the run, so a host resume, a dispatch, a recovery or a subagent
    under the fork is gated and capped exactly like the controller's own resume.
    """

    if owner is None:
        return ()
    controller = controllers.get(owner)
    if controller is None:
        raise ForkControllerDenied("The host has not allowed this controller to run its forks.")
    require_fork_controller_model_runs(controller)
    return fork_controller_budget_limits(controller, session_id, instance_id)


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
    "FORK_CONTROLLER_EVENT_QUERY_FIELDS",
    "FORK_CONTROLLER_READ_DEFAULT_BYTES",
    "FORK_CONTROLLER_READ_MAX_BYTES",
    "ForkBudgetLimitView",
    "ForkController",
    "ForkControllerAccess",
    "ForkControllerDenied",
    "ForkControllerEnrollment",
    "ForkControllerMode",
    "VerifiedForkController",
    "fork_controller_access",
]
