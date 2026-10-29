"""Ask Cayu Cloud whether this worker's Release is draining.

On Cayu Cloud a Release whose worker declares ``[worker] drain_seconds`` isn't stopped when
a later Release replaces it: it keeps running, so it can finish the tasks it holds, while the
new worker takes every new task. ECS can't send a task SIGTERM without starting its kill timer,
so Cloud doesn't signal the old worker; the worker asks instead. Cloud binds
``CAYU_WORKER_DRAIN_URL`` for that, and the worker asks with its Agent's attention credential
(``CAYU_ATTENTION_HANDOFF_TOKEN``) and its own Release (``CAYU_RELEASE_ID``)::

    GET $CAYU_WORKER_DRAIN_URL?release_id=$CAYU_RELEASE_ID
    -> {"draining": true, "deadline_at": "...", "replaced_by": "...", ...}

Once the answer is ``draining``, `cayu worker` sets the same stop event a SIGTERM sets: its task
loops stop claiming, the handlers it holds finish, and it exits. Outside Cayu Cloud (any of the
three variables unset) nothing is asked. Every failure is quiet and retried later: a drain
noticed late only delays the stop, and the platform still stops the worker at the deadline.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from math import isfinite

import httpx

from cayu.human_attention_handoff import TOKEN_ENVIRONMENT
from cayu.releases import RELEASE_ID_ENV

DRAIN_URL_ENVIRONMENT = "CAYU_WORKER_DRAIN_URL"
#: How often the worker asks.
POLL_SECONDS = 30.0
#: The longest a Retry-After may put off the next question.
_MAX_RETRY_AFTER_SECONDS = 300.0
_REQUEST_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class DrainRequest:
    """Cloud's answer that this worker's Release is draining."""

    release_id: str
    replaced_by: str | None
    deadline_at: str | None


async def watch_cloud_drain(
    on_drain: Callable[[DrainRequest], None],
    *,
    environ: Mapping[str, str] = os.environ,
    transport: httpx.AsyncBaseTransport | None = None,
    interval_seconds: float = POLL_SECONDS,
    sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
) -> None:
    """Ask every ``interval_seconds`` and call ``on_drain`` once the Release is draining.

    Returns after calling it, at once outside Cayu Cloud, or when Cloud doesn't serve the
    question (HTTP 404, a control plane that predates it). Run it as a task beside the worker
    and cancel it when the worker stops.
    """

    url = environ.get(DRAIN_URL_ENVIRONMENT, "").strip()
    token = environ.get(TOKEN_ENVIRONMENT, "").strip()
    release_id = environ.get(RELEASE_ID_ENV, "").strip()
    if not (url and token and release_id):
        return
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(
        timeout=_REQUEST_TIMEOUT_SECONDS, transport=transport, follow_redirects=False
    ) as client:
        while True:
            try:
                response = await client.get(url, params={"release_id": release_id}, headers=headers)
            except httpx.HTTPError:
                await sleep(interval_seconds)
                continue
            if response.status_code == 404:
                return
            delay = interval_seconds
            if response.status_code == 200:
                request = _drain_request(response, release_id)
                if request is not None:
                    on_drain(request)
                    return
            else:
                delay = _retry_after(response, interval_seconds)
            await sleep(delay)


def _drain_request(response: httpx.Response, release_id: str) -> DrainRequest | None:
    try:
        body = response.json()
    except ValueError:
        return None
    if not isinstance(body, dict) or body.get("draining") is not True:
        return None
    if body.get("release_id") not in (None, release_id):
        # An answer about another Release says nothing about this one.
        return None

    def text(key: str) -> str | None:
        value = body.get(key)
        return value if isinstance(value, str) else None

    return DrainRequest(
        release_id=release_id, replaced_by=text("replaced_by"), deadline_at=text("deadline_at")
    )


def _retry_after(response: httpx.Response, default: float) -> float:
    try:
        seconds = float(response.headers.get("retry-after", ""))
    except ValueError:
        return default
    if not isfinite(seconds):
        return default
    return min(max(seconds, 1.0), _MAX_RETRY_AFTER_SECONDS)


def drain_notice(request: DrainRequest) -> str:
    """What the worker says when it starts draining."""

    replaced = f" by Release {request.replaced_by}" if request.replaced_by else ""
    deadline = (
        f" Cayu Cloud stops it at {request.deadline_at} if they haven't finished by then."
        if request.deadline_at
        else ""
    )
    return (
        f"Release {request.release_id} was replaced{replaced}, so this worker is draining: it "
        f"claims no new tasks and exits once the ones it holds finish.{deadline}"
    )
