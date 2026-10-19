"""Hold each provider request of a capped fork inside its cap's reservation.

A fork controller's per-fork cap reserves one step's worst case, the
reservation's tokens, before each dispatch. That bounds the fork's spend only if
the request itself cannot use more: its input must fit the reserved input
tokens, and the provider must stop output at the reserved output tokens. Input
is bounded by the request's serialized size in bytes, which is at least its token
count for text; content whose tokens bytes cannot bound (files, peer content,
tool-result artifacts) is refused. Output is clamped through the adapter's own
output-token option, and an adapter that declares none is refused.
"""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from typing import Any

from cayu._fork_controller_errors import ForkControllerDenied
from cayu._validation import compact_json_utf8_size
from cayu.artifacts.attachments import RESOLVED_FILE_ATTACHMENTS_OPTION
from cayu.budgets.base import (
    BudgetLimit,
    _is_fork_controller_session_cap,
    fork_controller_total_owner,
)
from cayu.messages import FilePart, PeerContentPart, ToolResultPart
from cayu.providers.base import ModelProvider, ModelRequest


class ForkCapRequestRefused(ForkControllerDenied):
    """A capped fork's provider request could exceed its cap's reservation."""


def fork_cap_reservation(limits: Iterable[BudgetLimit]) -> tuple[int, int] | None:
    """The tightest reserved (input, output) tokens among a session's fork caps.

    A controller total reserves each step against its own reservation too, so
    it joins the minimum: a request held only to a looser per-fork reservation
    could use more than the total reserved for it.
    """

    caps = [
        limit.reservation
        for limit in limits
        if (
            _is_fork_controller_session_cap(limit) or fork_controller_total_owner(limit) is not None
        )
        and limit.reservation is not None
    ]
    if not caps:
        return None
    return (
        min(reservation.max_input_tokens for reservation in caps),
        min(reservation.max_output_tokens for reservation in caps),
    )


def request_input_token_upper_bound(request: ModelRequest) -> int | None:
    """An upper bound on the request's input tokens, or ``None`` when bytes cannot bound it."""

    if request.options.get(RESOLVED_FILE_ATTACHMENTS_OPTION):
        return None
    for message in request.messages:
        for part in message.content:
            if isinstance(part, FilePart | PeerContentPart):
                return None
            if isinstance(part, ToolResultPart) and part.artifacts:
                return None
    # Every text token covers at least one byte, and the JSON framing adds more
    # bytes per message than a provider adds tokens for roles and separators.
    return compact_json_utf8_size(
        {
            "messages": [message.model_dump(mode="json") for message in request.messages],
            "tools": request.tools,
        }
    )


def _option_path(provider: ModelProvider) -> tuple[str, ...]:
    path = provider.output_token_option_path()
    if path is None:
        raise ForkCapRequestRefused(
            f"Provider {provider.name!r} cannot cap a request's output tokens, so it cannot "
            "run a fork under a per-fork cap."
        )
    return path


def _require_input_within(request: ModelRequest, max_input_tokens: int) -> None:
    bound = request_input_token_upper_bound(request)
    if bound is None:
        raise ForkCapRequestRefused(
            "A request with files or other non-text input cannot run under a per-fork cap."
        )
    if bound > max_input_tokens:
        raise ForkCapRequestRefused(
            f"The request may use up to {bound} input tokens, more than the per-fork cap "
            f"reserves ({max_input_tokens}); raise the reservation's max_input_tokens."
        )


def bound_fork_cap_request(
    request: ModelRequest, provider: ModelProvider, limits: Iterable[BudgetLimit]
) -> ModelRequest:
    """Clamp output to the reservation and refuse input that may not fit it."""

    reserved = fork_cap_reservation(limits)
    if reserved is None:
        return request
    max_input_tokens, max_output_tokens = reserved
    path = _option_path(provider)
    _require_input_within(request, max_input_tokens)
    options: dict[str, Any] = deepcopy(request.options)
    target: dict[str, Any] = options
    for key in path[:-1]:
        nested = target.get(key)
        if not isinstance(nested, dict):
            nested = {}
            target[key] = nested
        target = nested
    current = target.get(path[-1])
    if not (type(current) is int and 0 < current <= max_output_tokens):
        target[path[-1]] = max_output_tokens
    return request.model_copy(update={"options": options})


def require_fork_cap_request(
    request: ModelRequest, provider: ModelProvider, limits: Iterable[BudgetLimit]
) -> None:
    """Refuse a request someone else built that may exceed the reservation.

    For dispatches the runtime cannot rewrite, such as a compactor's: its output
    cap must already be set within the reservation.
    """

    reserved = fork_cap_reservation(limits)
    if reserved is None:
        return
    max_input_tokens, max_output_tokens = reserved
    path = _option_path(provider)
    _require_input_within(request, max_input_tokens)
    value: Any = request.options
    for key in path:
        value = value.get(key) if isinstance(value, dict) else None
    if not (type(value) is int and 0 < value <= max_output_tokens):
        raise ForkCapRequestRefused(
            "This request does not cap its output tokens within the per-fork cap's "
            f"reservation ({max_output_tokens})."
        )
