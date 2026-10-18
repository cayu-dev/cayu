"""Live external-run authority shared by the gateway and native tool bridge."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass, field
from weakref import WeakKeyDictionary

from cayu.budgets.base import BudgetLimit
from cayu.external_agents.base import ExternalAgentContext, ExternalAgentError


@dataclass
class ExternalRunAuthority:
    context: ExternalAgentContext
    instance_id: str
    run_epoch: int
    budget_limits: tuple[BudgetLimit, ...]
    quiesced: asyncio.Event = field(default_factory=asyncio.Event)


_runs: WeakKeyDictionary = WeakKeyDictionary()
_stop_requests: WeakKeyDictionary = WeakKeyDictionary()


def cancel_run_once(session, targets: Iterable[asyncio.Task], *, message: str) -> None:
    """Deliver one run-owned stop to each target; caller cancellation stays separate."""
    identity = (session.id, session.instance_id, session.run_epoch)
    for target in targets:
        if _stop_requests.get(target) == identity:
            continue
        _stop_requests[target] = identity
        target.cancel(message)


def register_run(app, session, context, budget_limits) -> ExternalRunAuthority:
    authority = ExternalRunAuthority(context, session.instance_id, session.run_epoch, budget_limits)
    _runs.setdefault(app, {})[session.id] = authority
    return authority


def release_run(app, session_id) -> None:
    _runs.get(app, {}).pop(session_id, None)


def run_authority(app, session_id) -> ExternalRunAuthority:
    authority = _runs.get(app, {}).get(session_id)
    if authority is None:
        raise ExternalAgentError("External caller is no longer active.")
    return authority


def confirm_quiescence(app, session) -> None:
    authority = _runs.get(app, {}).get(session.id)
    if authority is not None and (authority.instance_id, authority.run_epoch) == (
        session.instance_id,
        session.run_epoch,
    ):
        authority.quiesced.set()
