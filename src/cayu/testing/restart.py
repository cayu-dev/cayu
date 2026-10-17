"""Test an application across process restarts against its durable stores."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Iterable
from contextlib import aclosing
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cayu.approvals import (
    ResolutionActor,
    ResolutionActorSource,
    ToolApprovalDecision,
    ToolApprovalRequest,
    UserInputResponse,
)
from cayu.evals.testing import ScriptedModelProvider, _normalize_batches
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.pending_action_contracts import PendingActionQuery, PendingActionRecord
from cayu.sessions.records import PendingActionKind, SessionStatus
from cayu.sessions.requests import ResumeRequest, RunRequest
from cayu.storage.application import ApplicationStores, open_application_stores
from cayu.storage.targets import parse_database_url
from cayu.tasks.memory import InMemoryTaskStore

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from cayu.applications import CayuApp
    from cayu.providers import ModelRequest, ModelStreamEvent

_SQLITE_FILENAME = "cayu.db"


@dataclass(frozen=True)
class RestartStep:
    """Where a session stopped after one harness call.

    ``status`` and ``pending_actions`` are read back from the durable store after
    the event stream ends, so they show what the next process will see.
    ``final_text`` is the text of the last completed model turn in this step, and
    ``error`` is the failure message when the step ended in ``session.failed``.
    """

    session_id: str
    status: SessionStatus
    final_text: str
    events: tuple[Event, ...]
    pending_actions: tuple[PendingActionRecord, ...]
    error: str | None = None

    @property
    def pending_approvals(self) -> tuple[PendingActionRecord, ...]:
        """Pending tool approvals, in the order the store lists them."""

        return tuple(
            action
            for action in self.pending_actions
            if action.kind is PendingActionKind.TOOL_APPROVAL
        )

    @property
    def pending_user_inputs(self) -> tuple[PendingActionRecord, ...]:
        """Pending ``ask_user`` questions, in the order the store lists them."""

        return tuple(
            action for action in self.pending_actions if action.kind is PendingActionKind.USER_INPUT
        )

    @property
    def completed(self) -> bool:
        """True when the session completed in this step."""

        return self.status is SessionStatus.COMPLETED


class RestartHarness:
    """Run one application across simulated process restarts.

    Each generation is what one OS process would build: fresh stores opened on
    the same database, a fresh scripted model provider, and a fresh app from
    ``factory``. :meth:`restart` closes the current generation (the app first,
    then the stores the harness opened) and builds the next one. Nothing else
    carries over in memory, so the second generation sees only what the first
    one made durable. The scripted model turns carry over: the next generation's
    provider starts at the first turn the previous ones did not use, so a test
    reads as one conversation across processes.

    ``factory`` is called with keyword arguments ``provider``, ``session_store``
    and ``task_store``, which is the signature of a generated project's
    ``build_app``. Adapt any other factory with a small function that passes
    them through. It must register ``provider`` as the model provider and
    build its app on the given stores.

    Pass ``store_dir`` for a SQLite database in that directory, or
    ``database_url`` for an absolute ``sqlite:///`` file or a migrated
    PostgreSQL database. In-memory stores are refused: they lose everything on
    restart, so a passing test would prove nothing.

    The harness does not change runtime checks. Execution-profile admission
    runs as in production, so a factory that changes a tool's declared
    identity between generations fails the way a redeployed process would,
    and a component without a stable declared identity fails the same way.

    Use it as an async context manager, or call :meth:`aclose` yourself. With
    ``call`` and ``final`` building scripted batches (``cayu guide restart-tests``
    has a complete, runnable version)::

        async with RestartHarness(
            build_app,
            store_dir=tmp_path,
            model_script=[call("issue_refund", {...}), final("Refunded.")],
        ) as harness:
            step = await harness.run(RunRequest(agent_name="refunds", messages=[...]))
            assert step.pending_approvals
            await harness.restart()
            step = await harness.approve(step.pending_approvals[0], actor="ops@example.test")
            assert step.completed
            await harness.assert_tool_ran_once(step.session_id, "issue_refund")

    The harness drives the app the way a host process would, through ``run``,
    ``resume``, ``resolve_tool_approval`` and ``resolve_user_input``. Errors from
    those calls propagate unchanged.
    """

    def __init__(
        self,
        factory: Callable[..., CayuApp],
        *,
        model_script: Iterable[ModelStreamEvent] | Iterable[Iterable[ModelStreamEvent]] = (),
        store_dir: str | os.PathLike[str] | None = None,
        database_url: str | None = None,
        provider_name: str = "scripted",
    ) -> None:
        if not callable(factory):
            raise TypeError("factory must be callable.")
        if (store_dir is None) == (database_url is None):
            raise ValueError("Pass exactly one of store_dir or database_url.")
        if database_url is not None:
            if not isinstance(database_url, str):
                raise TypeError("database_url must be a string.")
            if ":memory:" in database_url:
                raise ValueError(_IN_MEMORY_MESSAGE)
            parse_database_url(database_url, source="database_url")
            # open_application_stores ignores sqlite_path when given a URL.
            self._sqlite_path = Path(tempfile.gettempdir()).resolve() / _SQLITE_FILENAME
        else:
            assert store_dir is not None
            directory = Path(store_dir).resolve()
            directory.mkdir(parents=True, exist_ok=True)
            self._sqlite_path = directory / _SQLITE_FILENAME
        # Validated once; each generation's provider gets the batches not used yet.
        self._batches = _normalize_batches(model_script)
        self._factory = factory
        self._database_url = database_url
        self._provider_name = provider_name
        self._consumed_turns = 0
        self._previous_requests: list[ModelRequest] = []
        self._generation = 0
        self._app: CayuApp | None = None
        self._stores: ApplicationStores | None = None
        self._provider: ScriptedModelProvider | None = None
        self._closed = False

    @property
    def generation(self) -> int:
        """How many applications the harness has built, counting the current one."""

        return self._generation

    @property
    def app(self) -> CayuApp:
        """The current generation's application.

        It exists after :meth:`start`, :meth:`restart` or the first call that runs
        the app.
        """

        if self._closed:
            raise RuntimeError("This RestartHarness is closed.")
        if self._app is None:
            raise RuntimeError("No application yet; call start() or run() first.")
        return self._app

    @property
    def model_requests(self) -> tuple[ModelRequest, ...]:
        """Every request the scripted model received, across all generations."""

        current = () if self._provider is None else tuple(self._provider.requests)
        return (*self._previous_requests, *current)

    @property
    def remaining_model_turns(self) -> int:
        """Scripted model turns not used yet."""

        used = len(self._provider.requests) if self._provider is not None else 0
        # A request past the end of the script is recorded before it fails.
        return max(0, len(self._batches) - self._consumed_turns - used)

    async def start(self) -> CayuApp:
        """Build the first generation if it does not exist yet, and return the app."""

        if self._closed:
            raise RuntimeError("This RestartHarness is closed.")
        if self._app is None:
            # Finish closing stores a failed restart kept before opening new ones.
            await self._close_generation()
            await self._build()
        return self.app

    async def restart(self) -> CayuApp:
        """Close the current generation and build the next one on the same stores."""

        if self._closed:
            raise RuntimeError("This RestartHarness is closed.")
        await self._close_generation()
        await self._build()
        return self.app

    async def run(self, request: RunRequest | ResumeRequest) -> RestartStep:
        """Run or resume a session until it completes, fails or pauses."""

        app = await self.start()
        if isinstance(request, ResumeRequest):
            return await self._drain(app.resume(request), request.session_id)
        if not isinstance(request, RunRequest):
            raise TypeError("request must be a RunRequest or ResumeRequest.")
        return await self._drain(app.run(request), request.session_id)

    async def resume(
        self,
        session_id: str,
        *,
        messages: Iterable[Message] = (),
    ) -> RestartStep:
        """Resume ``session_id`` with optional new messages."""

        return await self.run(ResumeRequest(session_id=session_id, messages=list(messages)))

    async def approve(
        self,
        action: PendingActionRecord | str,
        *,
        actor: str,
        reason: str | None = None,
    ) -> RestartStep:
        """Approve a pending tool call and run the session on.

        ``action`` is a pending approval, or a session ID with exactly one.
        """

        return await self._resolve_approval(action, ToolApprovalDecision.APPROVE, actor, reason)

    async def deny(
        self,
        action: PendingActionRecord | str,
        *,
        actor: str,
        reason: str | None = None,
    ) -> RestartStep:
        """Deny a pending tool call and run the session on."""

        return await self._resolve_approval(action, ToolApprovalDecision.DENY, actor, reason)

    async def answer_user_input(
        self,
        action: PendingActionRecord | str,
        answer: str,
        *,
        actor: str | None = None,
    ) -> RestartStep:
        """Answer a pending ``ask_user`` question and run the session on.

        ``action`` is a pending question, or a session ID with exactly one.
        """

        pending = await self._single_action(action, PendingActionKind.USER_INPUT)
        assert pending.input_id is not None
        response = UserInputResponse(
            session_id=pending.session.id,
            input_id=pending.input_id,
            answer=answer,
            resolved_by=_actor(actor),
        )
        app = await self.start()
        return await self._drain(app.resolve_user_input(response), pending.session.id)

    async def pending_actions(
        self, session_id: str | None = None
    ) -> tuple[PendingActionRecord, ...]:
        """List durable pending actions, for one session or for all of them."""

        actions: list[PendingActionRecord] = []
        cursor: str | None = None
        store = (await self.start())._runtime_session_store
        while True:
            page = await store.query_pending_actions(
                PendingActionQuery(session_id=session_id, cursor=cursor, limit=200)
            )
            actions.extend(page.actions)
            if not page.has_more or page.next_cursor is None:
                return tuple(actions)
            cursor = page.next_cursor

    async def events(self, session_id: str) -> tuple[Event, ...]:
        """Load the session's durable event history."""

        app = await self.start()
        return tuple(await app.session_store.load_events(session_id))

    async def assert_tool_ran_once(self, session_id: str, tool_name: str) -> Event:
        """Assert that Cayu dispatched ``tool_name`` exactly once and it completed.

        This counts ``tool.call.started`` events in the durable history across
        every generation, so a call repeated after a restart fails the check.
        It returns the call's ``tool.call.completed`` event. It proves what Cayu
        dispatched; check your own service's records for the effect itself.
        """

        history = await self.events(session_id)
        started = [
            event
            for event in history
            if event.type == EventType.TOOL_CALL_STARTED and event.tool_name == tool_name
        ]
        if len(started) != 1:
            raise AssertionError(
                f"Expected {tool_name!r} to start exactly once in session {session_id!r}; "
                f"it started {len(started)} times."
            )
        call_id = started[0].payload.get("tool_call_id")
        terminals = [
            event
            for event in history
            if event.tool_name == tool_name
            and event.payload.get("tool_call_id") == call_id
            and event.type
            in {
                EventType.TOOL_CALL_COMPLETED,
                EventType.TOOL_CALL_FAILED,
                EventType.TOOL_CALL_BLOCKED,
            }
        ]
        if len(terminals) != 1 or terminals[0].type != EventType.TOOL_CALL_COMPLETED:
            outcome = ", ".join(str(event.type) for event in terminals) or "no terminal event"
            raise AssertionError(
                f"Expected {tool_name!r} call {call_id!r} to complete once; found {outcome}."
            )
        return terminals[0]

    async def aclose(self) -> None:
        """Close the current generation. Later calls do nothing."""

        if self._closed:
            return
        await self._close_generation()
        self._closed = True

    async def __aenter__(self) -> RestartHarness:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def _build(self) -> None:
        stores = open_application_stores(self._database_url, sqlite_path=self._sqlite_path)
        provider = ScriptedModelProvider(
            list(self._batches[self._consumed_turns :]),
            name=self._provider_name,
        )
        try:
            app = self._factory(
                provider=provider,
                session_store=stores.session_store,
                task_store=stores.task_store,
            )
            _require_durable_app(app)
        except BaseException:
            # Own the stores while closing them, so a failed or cancelled close
            # is retried, or joined while it finishes, by the next call.
            self._stores = stores
            await stores.close()
            self._stores = None
            raise
        self._stores = stores
        self._provider = provider
        self._app = app
        self._generation += 1

    async def _close_generation(self) -> None:
        app, stores, provider = self._app, self._stores, self._provider
        if app is not None:
            await app.aclose()
            outcome = app.shutdown_outcome
            if outcome is not None and not outcome.settled:
                raise RuntimeError(
                    f"The application did not shut down cleanly ({outcome.status}); "
                    "its stores are still open. Finish or close every event stream "
                    "before restarting."
                )
        self._app = None
        if provider is not None:
            self._previous_requests.extend(provider.requests)
            self._consumed_turns = min(
                len(self._batches), self._consumed_turns + len(provider.requests)
            )
            self._provider = None
        if stores is not None:
            # Keep ownership until the close succeeds: a failed or cancelled close
            # is retried, or joined while it finishes, by the next call.
            await stores.close()
            self._stores = None

    async def _drain(
        self, stream: AsyncGenerator[Event, None], session_id: str | None
    ) -> RestartStep:
        events: list[Event] = []
        turn_text: list[str] = []
        final_text = ""
        error: str | None = None
        # Closing the stream ends its run even if this loop stops early, so a
        # later restart never waits on an abandoned stream.
        async with aclosing(stream) as owned:
            async for event in owned:
                events.append(event)
                session_id = event.session_id
                if event.type == EventType.MODEL_STARTED:
                    turn_text = []
                elif event.type == EventType.MODEL_TEXT_DELTA:
                    delta = event.payload.get("delta")
                    if isinstance(delta, str):
                        turn_text.append(delta)
                elif event.type == EventType.MODEL_COMPLETED:
                    final_text = "".join(turn_text)
                elif event.type == EventType.SESSION_FAILED:
                    failure = event.payload.get("error")
                    error = failure if isinstance(failure, str) else None
        if session_id is None:
            raise RuntimeError("The run ended without emitting an event for its session.")
        session = await self.app.session_store.load(session_id)
        if session is None:
            raise RuntimeError(f"Session {session_id!r} is not in the durable store.")
        return RestartStep(
            session_id=session_id,
            status=session.status,
            final_text=final_text,
            events=tuple(events),
            pending_actions=await self.pending_actions(session_id),
            error=error,
        )

    async def _resolve_approval(
        self,
        action: PendingActionRecord | str,
        decision: ToolApprovalDecision,
        actor: str,
        reason: str | None,
    ) -> RestartStep:
        pending = await self._single_action(action, PendingActionKind.TOOL_APPROVAL)
        assert pending.approval_id is not None
        assert pending.round_id is not None
        assert pending.tool_call_id is not None
        request = ToolApprovalRequest(
            session_id=pending.session.id,
            approval_id=pending.approval_id,
            tool_round_id=pending.round_id,
            tool_call_id=pending.tool_call_id,
            decision=decision,
            reason=reason,
            resolved_by=_actor(actor),
        )
        app = await self.start()
        return await self._drain(app.resolve_tool_approval(request), pending.session.id)

    async def _single_action(
        self, action: PendingActionRecord | str, kind: PendingActionKind
    ) -> PendingActionRecord:
        if isinstance(action, PendingActionRecord):
            if action.kind is not kind:
                raise ValueError(f"Expected a {kind.value} action, got {action.kind.value}.")
            return action
        if not isinstance(action, str):
            raise TypeError("action must be a PendingActionRecord or a session ID.")
        matching = [item for item in await self.pending_actions(action) if item.kind is kind]
        if len(matching) != 1:
            raise ValueError(
                f"Session {action!r} has {len(matching)} pending {kind.value} actions; "
                "pass the one to resolve."
            )
        return matching[0]


_IN_MEMORY_MESSAGE = (
    "RestartHarness needs durable stores: in-memory stores lose every session when "
    "the app restarts. Use store_dir for SQLite or database_url for PostgreSQL."
)


def _require_durable_app(app: Any) -> None:
    from cayu.applications import CayuApp

    if not isinstance(app, CayuApp):
        raise TypeError("factory must return a CayuApp.")
    if isinstance(app.session_store, InMemorySessionStore) or isinstance(
        app.task_store, InMemoryTaskStore
    ):
        raise ValueError(
            _IN_MEMORY_MESSAGE + " The factory built its app on an in-memory store; "
            "pass through the session_store and task_store the harness gives it."
        )


def _actor(subject: str | None) -> ResolutionActor | None:
    if subject is None:
        return None
    return ResolutionActor(subject=subject, source=ResolutionActorSource.REQUEST)


__all__ = ["RestartHarness", "RestartStep"]
