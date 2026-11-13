"""A demonstration provider: canned, clearly marked replies with no model behind them.

``DemoModelProvider`` lets an application show its conversations, questions and
approvals without a model provider or an API key. It never calls a model and is
priced at zero. Every reply it makes up says it is a demonstration reply.

A provider may be given a script: its turns, in order, are what the agent "says"
in a session's first model steps (reply text, or a tool call such as an
``ask_user`` question or a call its tool policy sends for approval). A request's
turn is the number of assistant messages in it, or, when the session's context
policy has dropped earlier turns, one past the turn this provider last played in
that session, which it keeps per session in memory. The same request again (a
retried step) gets the same turn, and a restarted process counts again. A fork
has its own position, starting from the assistant messages its first request
carries, so a fork made under a context policy that already dropped them starts
the script over. After the script, or without one, each step gets a canned reply
that echoes what it answers.

The provider's identity is its name and Cayu's version, never its script, so a
session started by a script-bearing provider in one process (a seeding command)
continues under a plain ``DemoModelProvider()`` in another (``cayu serve --dev``
registers one under ``DEMO_PROVIDER_NAME``), and a question it asked can be
answered there.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import AsyncIterator, Iterable
from typing import Any

from cayu._validation import require_nonblank
from cayu.budgets.billing import BillingIdentity
from cayu.messages import Message, TextPart, ToolResultPart
from cayu.providers.base import (
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelStreamEvent,
    ModelStreamEventType,
    UsageDialect,
)

DEMO_PROVIDER_NAME = "demo"
# The price book zero-prices this model under the billing identity below, so a cost
# budget admits a demo step before the provider resolves that identity.
DEMO_MODEL = "demo"
# The runtime-owned zero-priced identity shared with native tool dispatch.
_BILLING_PROVIDER_NAME = "cayu-tool-dispatch"
_BILLING_RESOURCE_ID = "tool-dispatch"
_ZERO_USAGE = {"usage": {"input_tokens": 0, "output_tokens": 0}}
_MARK = "Demonstration reply, no model was called."
_ECHO_CHARS = 200
# Sessions whose script position this provider remembers, most recent last.
_MAX_TRACKED_SESSIONS = 4096

DemoTurn = str | ModelStreamEvent


class DemoModelProvider(ModelProvider):
    """Canned, clearly marked replies, after an optional script of first turns.

    ``script`` holds reply texts and ``ModelStreamEvent.tool_call(...)`` events.
    A scripted tool call the agent isn't offered fails the model step with a
    message naming it, rather than being refused later as an unknown tool.
    """

    name = DEMO_PROVIDER_NAME
    billing_provider_name = _BILLING_PROVIDER_NAME
    usage_dialect = UsageDialect.GENERIC

    def __init__(self, script: Iterable[DemoTurn] = ()) -> None:
        turns: list[DemoTurn] = []
        for turn in script:
            if isinstance(turn, str):
                require_nonblank(turn, "script reply")
            elif not (
                isinstance(turn, ModelStreamEvent) and turn.type is ModelStreamEventType.TOOL_CALL
            ):
                raise TypeError(
                    "A demonstration turn is reply text or ModelStreamEvent.tool_call(...)."
                )
            turns.append(turn)
        self._script = tuple(turns)
        self._turns_played = 0
        # Per session (its session key): the last request's step key and its turn.
        self._played: OrderedDict[str, tuple[str, int]] = OrderedDict()

    @property
    def script(self) -> tuple[DemoTurn, ...]:
        return self._script

    @property
    def turns_played(self) -> int:
        """How many of the script's turns this instance has played, in this process."""

        return self._turns_played

    async def billing_identity_for_request(self, request: ModelRequest) -> BillingIdentity:
        return BillingIdentity(
            provider_name=_BILLING_PROVIDER_NAME, resource_id=_BILLING_RESOURCE_ID
        )

    def _turn(self, request: ModelRequest) -> int:
        """The request's script turn.

        A context policy may have dropped earlier turns from ``request.messages``, so
        the count alone can go back. The session key, which a fork doesn't share,
        finds where this provider got to in the session; the step key, which a retry
        keeps and the next step doesn't, tells a repeated request from a new step
        even when their messages are identical.
        """

        turn = sum(1 for message in request.messages if message.role == "assistant")
        key, step = request.session_key, request.step_key
        if key is None or step is None:
            return turn
        played = self._played.get(key)
        if played is not None:
            played_step, played_turn = played
            turn = played_turn if step == played_step else max(turn, played_turn + 1)
        return turn

    def _record(self, request: ModelRequest, turn: int) -> None:
        key, step = request.session_key, request.step_key
        if key is None or step is None:
            return
        self._played[key] = (step, turn)
        self._played.move_to_end(key)
        while len(self._played) > _MAX_TRACKED_SESSIONS:
            self._played.popitem(last=False)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        turn = self._turn(request)
        scripted = self._script[turn] if turn < len(self._script) else None
        if scripted is not None:
            self._turns_played = max(self._turns_played, turn + 1)
        if isinstance(scripted, ModelStreamEvent):
            tool_name = scripted.payload["name"]
            if tool_name not in {tool.get("name") for tool in request.tools}:
                raise ModelProviderError(
                    f"The demonstration script calls {tool_name}, which this agent isn't offered.",
                    provider=self.name,
                    error_type="demo_tool_not_offered",
                    retryable=False,
                )
            self._record(request, turn)
            yield ModelStreamEvent.tool_call(
                name=tool_name,
                arguments=scripted.payload["arguments"],
                id=scripted.payload.get("id") or f"demo-call-{turn}",
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls", **_ZERO_USAGE})
            return
        self._record(request, turn)
        yield ModelStreamEvent.text_delta(scripted or _canned_reply(request.messages))
        yield ModelStreamEvent.completed({"finish_reason": "stop", **_ZERO_USAGE})


def _canned_reply(messages: list[Message]) -> str:
    for message in reversed(messages):
        results = [part for part in message.content if isinstance(part, ToolResultPart)]
        if results:
            if any(part.is_error for part in results):
                return f"{_MARK} The tool call didn't succeed."
            answered = _excerpt(" ".join(part.content for part in results if part.content))
            return f"{_MARK} Noted: {answered}" if answered else f"{_MARK} Noted."
        if message.role != "user":
            continue
        text = _excerpt(" ".join(p.text for p in message.content if isinstance(p, TextPart)))
        if text:
            return f"{_MARK} You wrote: “{text}”. Set up a model provider for real answers."
        break
    return f"{_MARK} Set up a model provider for real answers."


def _excerpt(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _ECHO_CHARS else text[: _ECHO_CHARS - 1].rstrip() + "…"


def _execution_profile_material(provider: DemoModelProvider) -> dict[str, Any]:
    """The script is demonstration content, not behavior; Cayu's version is the identity."""

    del provider
    return {}


__all__ = ["DEMO_MODEL", "DEMO_PROVIDER_NAME", "DemoModelProvider"]
