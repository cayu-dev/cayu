"""`--wait` for the commands Cayu Cloud refuses while an Agent's service switches.

While Cloud publishes a release to an Agent's service (its database migrations and the service
update) or changes the service another way, it holds that Agent's service lock. A service
restart, sleep, wake, pause, resume or teardown, and a schedule run on demand, waits about ten
seconds for the lock and is then refused with HTTP 409 and Cloud's plain-text reason "The
Agent's service is being deployed or changed right now; nothing changed. Try again once that
finishes." Today's Clouds send no code with it, so the CLI knows the refusal by that reason,
or by the code `service_busy` once Cloud sends one.

Cloud refuses before it changes anything, so sending the same request again is safe. With
`--wait`, these commands send it again every `--poll-seconds` until Cloud takes it or
`--wait-seconds` pass; without it the refusal says that `--wait` would.
"""

from __future__ import annotations

import shlex
import sys
import time
from collections.abc import Callable, Sequence
from typing import Any

from cayu.cli._cloud_api import CloudApiError
from cayu.cli._cloud_project import is_application_slug

# The start of Cloud's refusal while it holds the Agent's service lock.
_SWITCHING = "The Agent's service is being deployed or changed right now; nothing changed."
# The code a Cloud that structures the refusal sends with it.
SERVICE_BUSY = "service_busy"


def service_switching(exc: CloudApiError) -> bool:
    """Whether Cloud refused the request, unchanged, only because the service is switching."""

    if exc.status_code != 409:
        return False
    return exc.code == SERVICE_BUSY or (
        exc.detail is not None and exc.detail.startswith(_SWITCHING)
    )


class ServiceStillSwitchingError(CloudApiError):
    """Cloud kept refusing a request while the Agent's service switched, until the wait ended."""

    def __init__(
        self,
        cause: CloudApiError,
        *,
        application_id: str,
        waited_seconds: float,
        recovery_arguments: Sequence[str] = (),
    ) -> None:
        super().__init__(
            "service_busy",
            f"The Agent's service was still being deployed or changed after "
            f"{waited_seconds:.0f}s, so Cayu Cloud kept refusing the request; nothing changed. "
            "Run the command again once that finishes.",
            status_code=cause.status_code,
            detail=cause.detail,
        )
        self.details: dict[str, object] = {"waited_seconds": round(waited_seconds)}
        if is_application_slug(application_id):
            self.details["application"] = application_id
            self.details["commands"] = {
                "status": shlex.join(
                    [
                        "cayu",
                        "cloud",
                        *recovery_arguments,
                        "service",
                        "status",
                        "--application",
                        application_id,
                    ]
                )
            }


def send_when_not_switching(
    request: Callable[[], dict[str, Any]],
    *,
    wait: bool,
    application_id: str,
    poll_seconds: float,
    wait_seconds: float,
    recovery_arguments: Sequence[str] = (),
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Cloud's answer to ``request``, sent again while ``wait`` and Cloud refuses it for a switch.

    ``request`` sends the same request every time, with the same idempotency key when it
    has one. Any other refusal, and every other failure, is raised at once. The wait counts
    from the first refusal, so a request Cloud takes at once reads no clock, and the request
    is never sent again once ``wait_seconds`` have passed since then.
    """

    refused_at: float | None = None
    while True:
        try:
            return request()
        except CloudApiError as exc:
            if not service_switching(exc):
                raise
            if not wait:
                raise CloudApiError(
                    exc.category,
                    f"{exc} Pass --wait to wait for it and send the request again.",
                    status_code=exc.status_code,
                    detail=exc.detail,
                ) from exc
            refusal = exc
        now = monotonic()
        if refused_at is None:
            refused_at = now
            print(
                "cayu cloud: The Agent's service is being deployed or changed right now; "
                "the request is sent again once that finishes.",
                file=sys.stderr,
            )
        remaining = refused_at + wait_seconds - now
        if remaining > 0:
            sleep(min(poll_seconds, remaining))
            now = monotonic()
        if now - refused_at >= wait_seconds:
            raise ServiceStillSwitchingError(
                refusal,
                application_id=application_id,
                waited_seconds=now - refused_at,
                recovery_arguments=recovery_arguments,
            ) from refusal
